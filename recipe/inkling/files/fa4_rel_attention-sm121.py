# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
    zip_inputs,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def bucket_max_seqlen_q(max_seqlen_q: int) -> int:
    """Round the FA4 scheduling bound up to a power of two."""
    return 1 << max(0, max_seqlen_q - 1).bit_length()


@cache
def _use_sheared_bias() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major in (10, 11)


# [sm121-relattn-hotfix] SM12 (GB10/DGX Spark) rel-attention fallback gate.
#
# Active on device capability major >= 12 (GB10 is sm_121), or forced with
# INKLING_REL_ATTN_BACKEND=triton; forced off with =fa4. v0.28.0 has no
# working FA4 rel-attention backend on SM 12.0 (cute: paged KV unsupported;
# tml_fa4: tile_n==128 assert, then a page_table-is-None assert in its SM120
# kernel class). The fallback below routes decode to the portable Triton
# split-KV kernel from the ROCm lane and prefill/extend to a torch-native
# SDPA with the inkling relative bias applied explicitly.


@cache
def _use_rel_attn_fallback() -> bool:
    import os as _os

    override = _os.environ.get("INKLING_REL_ATTN_BACKEND", "").strip().lower()
    if override in ("triton", "sdpa", "fallback", "1", "true", "on"):
        return True
    if override in ("fa4", "off", "0", "false", "none"):
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major >= 12


@cache
def _triton_decode_enabled() -> bool:
    import os as _os

    return _os.environ.get(
        "INKLING_REL_ATTN_TRITON_DECODE", "1"
    ).strip().lower() not in ("", "0", "false", "no", "off")


@cache
def _get_triton_decode_fn():
    """Import the ROCm-lane Triton split-KV decode kernel, or None."""
    if not _triton_decode_enabled():
        return None
    try:
        from vllm.models.inkling.amd.ops.rel_attention_decode import (
            inkling_rel_attention_split_kv_decode,
        )
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "inkling sm12 fallback: Triton decode kernel unavailable (%r); "
            "decode will use the torch SDPA path",
            exc,
        )
        return None
    return inkling_rel_attention_split_kv_decode


@cache
def _get_score_mod(rel_extent: int) -> Callable:
    """Return the score modification that adds Inkling relative bias."""
    import cutlass.cute as cute
    from cutlass.cute import Float32

    from vllm.vllm_flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        rel_logits = aux_tensors[0]

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        rel_dist_0 = rel_dist[0]
        rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
        rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

        rel_bias = rel_logits[global_q_idx[0], h_idx[0], rel_idx]
        rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)
        return scores + rel_bias

    return score_mod_rel_bias


def inkling_fa4_num_splits(
    *,
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Return the split-KV cap for Inkling relative attention."""
    capability = current_platform.get_device_capability()
    if capability is not None and capability.major == 9:
        return 1
    if is_local:
        return 1

    q_rows = max_query_len * (num_heads // num_kv_heads)
    q_tiles = (q_rows + 255) // 256
    base_ctas = batch_size * num_kv_heads * q_tiles
    # Shearing makes split/combine overhead more visible. Multi-tile causal
    # prefill saturates around 64 CTAs. Batch-1 decode at very long context is
    # memory-bound and uses a TP-specific cap measured through 1M KV tokens.
    target_ctas = (
        256 if q_tiles == 1 and batch_size == 1 else (128 if q_tiles == 1 else 64)
    )
    max_splits = 128
    if q_tiles == 1 and batch_size == 1:
        if num_kv_heads == 8:
            max_splits = 16
        elif num_kv_heads == 4 or max_kv_len <= 8192:
            max_splits = 32
        elif max_kv_len <= 65536:
            max_splits = 64
        else:
            max_splits = 128
    return max(
        1,
        min(target_ctas // base_ctas, max_splits, (max_kv_len + 127) // 128),
    )


def _num_warps_bucket(num_reqs: int) -> int:
    num_warps = min((num_reqs + 30) // 31, 32)
    return 1 << (num_warps - 1).bit_length()


class InklingFA4RelAttentionKernel(
    VllmJitKernel["InklingFA4RelAttentionKernel.CompileKey"]
):
    @dataclass(frozen=True)
    class CompileKey:
        is_local: bool
        num_heads: int
        num_kv_heads: int
        head_dim: int
        rel_extent: int
        dtype: torch.dtype
        kv_dtype: torch.dtype
        block_size: int
        window_size: tuple[int, int]
        max_seqlen_q: int
        num_splits: int
        num_warps_bucket: int | None
        large_num_reqs: bool

    @staticmethod
    def kernel(
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int],
        rel_extent: int,
        rel_logits: torch.Tensor,
        num_splits: int = 32,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Paged varlen FA4 over the bound K/V cache with the Inkling relative bias.

        ``q`` is ``(num_tokens, num_heads, head_dim)``; ``key_cache`` / ``value_cache``
        are the paged caches ``(num_blocks, block_size, num_kv_heads, head_dim)``;
        ``block_table`` is the per-request page table and ``cache_seqlens`` the
        per-request KV lengths (``seqused_k``). ``rel_logits`` is
        ``(num_tokens, num_heads, rel_extent)``.

        Hopper uses standard FA4's score-mod gather. Blackwell uses tml-fa4's
        sheared relative-bias layout.
        """
        if _use_rel_attn_fallback():
            return _sm12_rel_attention_fallback(
                q,
                key_cache,
                value_cache,
                block_table=block_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                rel_extent=rel_extent,
                rel_logits=rel_logits,
                out=out,
            )

        # cute uses (None, None) to mean "no window".
        cute_window = (None, None) if window_size == (-1, -1) else window_size

        rel_logits = rel_logits.contiguous()
        flash_attn_varlen_func: Callable[..., Any]
        if _use_sheared_bias():
            from vllm.third_party.tml_fa4 import (
                flash_attn_varlen_func as tml_flash_attn_varlen_func,
            )

            flash_attn_varlen_func = tml_flash_attn_varlen_func
            bias_kwargs: dict[str, Any] = {"rel_bias": rel_logits}
        else:
            from vllm.vllm_flash_attn.cute import (
                flash_attn_varlen_func as cute_flash_attn_varlen_func,
            )

            flash_attn_varlen_func = cute_flash_attn_varlen_func
            bias_kwargs = {
                "score_mod": _get_score_mod(rel_extent),
                "aux_tensors": [rel_logits],
            }

        ret = flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=cache_seqlens,
            max_seqlen_q=max_seqlen_q,
            page_table=block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=cute_window,
            num_splits=num_splits,
            return_lse=False,
            out=out,
            **bias_kwargs,
        )
        if isinstance(ret, tuple):
            return ret[0]
        return ret

    def dispatch(  # type: ignore[override]
        self,
        *,
        is_local: bool,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rel_extent: int,
        dtype: torch.dtype,
        kv_dtype: torch.dtype,
        block_size: int,
        window_size: tuple[int, int],
        max_kv_len: int,
        query_len: int,
        num_reqs: int,
    ) -> CompileKey:
        max_seqlen_q = bucket_max_seqlen_q(query_len)
        num_splits = inkling_fa4_num_splits(
            is_local=is_local,
            batch_size=num_reqs,
            max_query_len=max_seqlen_q,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_kv_len=max_kv_len,
        )
        return self.CompileKey(
            is_local=is_local,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rel_extent=rel_extent,
            dtype=dtype,
            kv_dtype=kv_dtype,
            block_size=block_size,
            window_size=window_size,
            max_seqlen_q=max_seqlen_q,
            num_splits=num_splits,
            num_warps_bucket=(_num_warps_bucket(num_reqs) if num_splits > 1 else None),
            large_num_reqs=num_reqs > 1024,
        )

    def _is_valid_warmup_dispatch(
        self,
        *,
        query_len: int,
        num_reqs: int,
        max_num_batched_tokens: int,
    ) -> bool:
        return query_len + num_reqs <= max_num_batched_tokens + 1

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        if _use_rel_attn_fallback():
            # The SM12 fallback kernels are exercised by the ordinary vLLM
            # warmup forward; never JIT-compile FA4 on this arch (that
            # compile is what killed every v0.28.0 boot on GB10).
            return []
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        if max_num_reqs <= 0 or max_num_batched_tokens <= 0:
            return []

        hf_config = vllm_config.model_config.hf_config
        get_text_config = getattr(hf_config, "get_text_config", None)
        config = get_text_config() if callable(get_text_config) else hf_config

        tp_size = get_tensor_model_parallel_world_size()
        dtype = vllm_config.model_config.dtype
        kv_dtype = kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype,
            vllm_config.model_config,
        )
        block_size = vllm_config.cache_config.block_size
        local_extent = config.sliding_window_size

        global_num_kv_heads = config.num_key_value_heads
        local_num_kv_heads = config.swa_num_key_value_heads
        assert config.num_attention_heads % tp_size == 0
        assert config.swa_num_attention_heads % tp_size == 0
        if global_num_kv_heads >= tp_size:
            assert global_num_kv_heads % tp_size == 0
        else:
            assert tp_size % global_num_kv_heads == 0
        if local_num_kv_heads >= tp_size:
            assert local_num_kv_heads % tp_size == 0
        else:
            assert tp_size % local_num_kv_heads == 0

        return self._trace_dispatch(self.dispatch)(
            zip_inputs(
                dict(
                    is_local=False,
                    num_heads=config.num_attention_heads // tp_size,
                    num_kv_heads=max(1, global_num_kv_heads // tp_size),
                    head_dim=config.head_dim,
                    rel_extent=config.rel_extent,
                    dtype=dtype,
                    kv_dtype=kv_dtype,
                    block_size=block_size,
                    window_size=(-1, -1),
                    max_kv_len=vllm_config.model_config.max_model_len,
                ),
                dict(
                    is_local=True,
                    num_heads=config.swa_num_attention_heads // tp_size,
                    num_kv_heads=max(1, local_num_kv_heads // tp_size),
                    head_dim=config.swa_head_dim,
                    rel_extent=local_extent,
                    dtype=dtype,
                    kv_dtype=kv_dtype,
                    block_size=block_size,
                    window_size=(local_extent - 1, 0),
                    max_kv_len=local_extent,
                ),
            ),
            query_len=WarmupIntRange(
                1,
                max_num_batched_tokens + 1,
                advance=lambda value: bucket_max_seqlen_q(value) + 1,
            ),
            num_reqs=WarmupIntRange(1, max_num_reqs + 1),
            max_num_batched_tokens=max_num_batched_tokens,
            _when=self._is_valid_warmup_dispatch,
        )

    def compile(self, compile_key: CompileKey) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        if compile_key.num_splits == 1 or compile_key.num_warps_bucket is None:
            num_reqs = 1025 if compile_key.large_num_reqs else 1
        else:
            min_num_warps = (
                1
                if compile_key.num_warps_bucket == 1
                else compile_key.num_warps_bucket // 2 + 1
            )
            min_num_reqs = (min_num_warps - 1) * 31 + 1
            num_reqs = max(
                min_num_reqs,
                1025 if compile_key.large_num_reqs else 1,
            )

        with FakeTensorMode():
            device = torch.accelerator.current_accelerator()
            total_q = compile_key.max_seqlen_q + num_reqs - 1
            q = torch.empty(
                total_q,
                compile_key.num_heads,
                compile_key.head_dim,
                dtype=compile_key.dtype,
                device=device,
            )
            kv = torch.empty(
                1,
                2,
                compile_key.block_size,
                compile_key.num_kv_heads,
                compile_key.head_dim,
                dtype=compile_key.kv_dtype,
                device=device,
            )
            key_cache, value_cache = kv.unbind(1)
            self.kernel(
                q,
                key_cache,
                value_cache,
                block_table=torch.empty(
                    num_reqs,
                    1,
                    dtype=torch.int32,
                    device=device,
                ),
                cache_seqlens=torch.empty(
                    num_reqs,
                    dtype=torch.int32,
                    device=device,
                ),
                cu_seqlens_q=torch.empty(
                    num_reqs + 1,
                    dtype=torch.int32,
                    device=device,
                ),
                max_seqlen_q=compile_key.max_seqlen_q,
                softmax_scale=compile_key.head_dim**-1,
                causal=True,
                window_size=compile_key.window_size,
                rel_extent=compile_key.rel_extent,
                rel_logits=torch.empty(
                    total_q,
                    compile_key.num_heads,
                    compile_key.rel_extent,
                    dtype=compile_key.dtype,
                    device=device,
                ),
                num_splits=compile_key.num_splits,
                out=torch.empty_like(q),
            )

    def __call__(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int],
        rel_extent: int,
        rel_logits: torch.Tensor,
        num_splits: int = 32,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel(
            q,
            key_cache,
            value_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            rel_extent=rel_extent,
            rel_logits=rel_logits,
            num_splits=num_splits,
            out=out,
        )


INKLING_FA4_REL_ATTENTION_KERNEL = InklingFA4RelAttentionKernel()


# [sm121-relattn-hotfix] SM12 (GB10/DGX Spark) relative-attention fallback.
#
# decode  -> ROCm-lane Triton split-KV kernel (portable; paged-KV args),
#            with a per-process escape to the torch path on compile/run
#            failure (the exact error is logged).
# prefill -> correctness-first torch-native SDPA with the inkling relative
#            bias applied explicitly: for query absolute position i and KV
#            position j, bias = rel_logits[t, h, i-j] when 0 <= i-j <
#            rel_extent else 0 (matches the cute score-mod and the Triton
#            decode kernel); causal; optional left sliding window
#            (window_size[0], -1 = disabled). KV is gathered from the paged
#            cache via block_table. Slow is fine; wrong is not.

import logging as _logging

_sm12_logger = _logging.getLogger(__name__)

# Set when the Triton decode kernel raises on this platform: decode then uses
# the torch SDPA path for the rest of the process.
_TRITON_DECODE_BROKEN = False


def _sm12_rel_attention_fallback(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    global _TRITON_DECODE_BROKEN
    if not causal:
        raise ValueError("inkling sm12 fallback: only causal=True is supported")
    if out is None:
        raise ValueError("inkling sm12 fallback: out must be provided")
    window_left = int(window_size[0])
    num_reqs = cache_seqlens.shape[0]
    if (
        max_seqlen_q == 1
        and q.shape[0] == num_reqs
        and not _TRITON_DECODE_BROKEN
    ):
        decode_fn = _get_triton_decode_fn()
        if decode_fn is not None:
            try:
                max_kv_len = max(1, int(cache_seqlens.max().item()))
                return decode_fn(
                    q,
                    key_cache,
                    value_cache,
                    block_table=block_table,
                    cache_seqlens=cache_seqlens,
                    softmax_scale=softmax_scale,
                    window_left=window_left,
                    rel_extent=rel_extent,
                    rel_logits=rel_logits,
                    max_kv_len=max_kv_len,
                    out=out,
                )
            except Exception as exc:
                _TRITON_DECODE_BROKEN = True
                _sm12_logger.warning(
                    "inkling sm12 fallback: Triton decode failed (%r); "
                    "decode falls back to the torch SDPA path for the rest "
                    "of this process",
                    exc,
                    exc_info=True,
                )
    return _sdpa_rel_attention_varlen(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        softmax_scale=softmax_scale,
        window_left=window_left,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        out=out,
    )


@torch.no_grad()
def _sdpa_rel_attention_varlen(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    window_left: int,
    rel_extent: int,
    rel_logits: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Correctness-first torch SDPA for inkling rel-attention (varlen, paged).

    ``q`` is ``(num_tokens, num_heads, head_dim)`` flattened across requests;
    per request b the query rows are a causal suffix of the KV cache:
    query row t has absolute position ``cache_seqlens[b] - qlen_b + t`` and
    attends KV positions ``j <= i`` (plus the optional left window). Scores
    and softmax are fp32; the probs @ V matmul runs in the cache dtype.
    """
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = key_cache.shape[2]
    page_size = key_cache.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"inkling sm12 fallback: num_heads={num_heads} not divisible by "
            f"num_kv_heads={num_kv_heads}"
        )
    group = num_heads // num_kv_heads
    device = q.device
    cu = cu_seqlens_q.tolist()
    lens = cache_seqlens.tolist()
    query_chunk = 64
    for b in range(len(lens)):
        kvlen = int(lens[b])
        t0, t1 = int(cu[b]), int(cu[b + 1])
        qlen = t1 - t0
        if qlen <= 0:
            continue
        if not 0 < qlen <= kvlen:
            raise ValueError(
                f"inkling sm12 fallback: request {b} has qlen={qlen} "
                f"kvlen={kvlen}"
            )
        npages = (kvlen + page_size - 1) // page_size
        pages = block_table[b, :npages].to(torch.long)
        k = key_cache.index_select(0, pages).view(-1, num_kv_heads, head_dim)[
            :kvlen
        ]
        v = value_cache.index_select(0, pages).view(-1, num_kv_heads, head_dim)[
            :kvlen
        ]
        # GQA expansion: q head h reads kv head h // group (matches the
        # Triton decode kernel's head tiling).
        k = k.permute(1, 0, 2).repeat_interleave(group, dim=0)
        v = v.permute(1, 0, 2).repeat_interleave(group, dim=0)
        kf = k.to(torch.float32)
        j_pos = torch.arange(kvlen, device=device)
        prefix = kvlen - qlen
        rel_b = rel_logits[t0:t1].to(torch.float32).permute(1, 0, 2)
        for s in range(0, qlen, query_chunk):
            e = min(s + query_chunk, qlen)
            i_pos = prefix + torch.arange(s, e, device=device)
            rel = i_pos[:, None] - j_pos[None, :]  # (qc, kvlen) = i - j
            rel_idx = rel.clamp(0, rel_extent - 1)
            gather_idx = rel_idx.unsqueeze(0).expand(num_heads, -1, -1)
            bias = torch.gather(rel_b[:, s:e], 2, gather_idx)
            in_range = (rel >= 0) & (rel < rel_extent)
            bias = torch.where(
                in_range.unsqueeze(0), bias, torch.zeros((), device=device)
            )
            visible = rel >= 0  # causal: j <= i
            if window_left >= 0:
                visible = visible & (rel <= window_left)
            neg_mask = torch.where(
                visible,
                torch.zeros((), device=device),
                torch.full((), float("-inf"), device=device),
            ).unsqueeze(0)
            qb = q[t0 + s : t0 + e].permute(1, 0, 2).to(torch.float32)
            scores = (
                torch.matmul(qb, kf.transpose(1, 2)) * softmax_scale
                + bias
                + neg_mask
            )
            probs = torch.softmax(scores, dim=-1)
            o = torch.matmul(probs.to(v.dtype), v)  # (H, qc, D)
            out[t0 + s : t0 + e] = o.permute(1, 0, 2).to(out.dtype)
    return out
