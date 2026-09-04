#!/usr/bin/env python3
"""Hotfix: SM12 (GB10 / DGX Spark) relative-attention fallback for the Inkling
architecture (vllm/models/inkling/nvidia/ops/fa4_rel_attention.py, vLLM
v0.28.0).

Why this exists — vLLM v0.28.0 has NO working FA4 rel-attention backend on
SM 12.0 (all confirmed live on 2x DGX Spark, 2026-09-02/03):

  * cute score-mod path (Hopper): asserts
    "Paged KV not supported on SM 12.0 in this PR".
  * tml_fa4 sheared path (Blackwell): asserts tile_n == 128 in the rel_bias
    metadata — inkling's SplitKV heuristic shrinks tile_n to 64 — and, deeper,
    its SM120 kernel class asserts ``page_table is None`` (non-paged KV only)
    while vLLM is always paged. Dead end; the fix is upstream.

What the patch does — routes inkling rel-attention AWAY from FA4 on sm_12x,
keeping the stock module's public contract (``INKLING_FA4_REL_ATTENTION_KERNEL``,
``bucket_max_seqlen_q``, ``inkling_fa4_num_splits``) so ``attention.py`` is
untouched:

  * pure decode (one query token per request): the portable Triton split-KV
    decode kernel from the ROCm lane,
    ``vllm.models.inkling.amd.ops.rel_attention_decode
    .inkling_rel_attention_split_kv_decode`` (LightSeek TokenSpeed port, same
    paged-KV args). If it fails to compile/run on GB10, the failure is logged
    with the exact error and decode falls back to the torch path for the rest
    of the process.
  * prefill / extend / mixed / MTP batches: a correctness-first torch-native
    SDPA with the inkling relative bias applied explicitly (bias =
    rel_logits[t, h, i-j] for 0 <= i-j < rel_extent, else 0; causal; optional
    left sliding window). KV is gathered from the paged cache via block_table.
    Slow is fine; wrong is not.
  * FA4 JIT warmup (``get_warmup_keys``) returns [] when the fallback is
    active — the engine must never try to compile FA4 on SM12 (that compile
    is what killed every v0.28.0 boot on GB10 ~28s after weight load).

Gating (env):

  INKLING_REL_ATTN_BACKEND   unset -> auto: active iff device capability
                             major >= 12. "triton"/"sdpa"/"on" -> force ON.
                             "fa4"/"off" -> force OFF (stock behavior).
  INKLING_REL_ATTN_TRITON_DECODE=0 -> disable the Triton decode kernel;
                             decode also uses the torch SDPA path.

Failure semantics (fail-closed):

  * Anchors not found (file drifted from v0.28.0 stock): exit 1, before any
    boot — a boot asked for the SM12 fallback must not silently serve the
    stock FA4 dispatch.
  * Runtime: the fallback raises on non-causal calls or a missing ``out``
    (conditions the stock kernel never produces) rather than guessing.

Patches the target file in-place (called from the serve script before
``exec vllm serve``, or offline against a staged copy that is then
bind-mounted over the container path). Idempotent: re-applying is a no-op
once the marker is present. ``--status`` reports state.

    # offline staging (the recipe flow):
    docker run --rm --entrypoint cat vllm/vllm-openai:v0.28.0 \
      /usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/ops/fa4_rel_attention.py \
      > files/fa4_rel_attention-sm121.py
    WEIGHTLESS_RELATTN_FA4_PY=files/fa4_rel_attention-sm121.py \
      python3 hotfix-inkling-sm121-relattn.py
"""
import os
from pathlib import Path
import sys

# Overridable for dry-runs against copies outside the container.
P = Path(os.environ.get(
    "WEIGHTLESS_RELATTN_FA4_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/ops/fa4_rel_attention.py",
))
MARK = "# [sm121-relattn-hotfix-v2]"

# ---------------------------------------------------------------------------
# Injected source 1: gate helpers, inserted before ``_get_score_mod``.
# ---------------------------------------------------------------------------
HELPER_SRC = MARK + ''' SM12 (GB10/DGX Spark) rel-attention fallback gate.
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


'''

# ---------------------------------------------------------------------------
# Injected source 2: dispatch inside InklingFA4RelAttentionKernel.kernel(),
# inserted before the cute_window line.
# ---------------------------------------------------------------------------
KERNEL_ANCHOR = '''        # cute uses (None, None) to mean "no window".
        cute_window = (None, None) if window_size == (-1, -1) else window_size
'''
KERNEL_DISPATCH = '''        if _use_rel_attn_fallback():
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

'''

# ---------------------------------------------------------------------------
# Injected source 3: warmup guard inside get_warmup_keys().
# ---------------------------------------------------------------------------
WARMUP_ANCHOR = '''    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
'''
WARMUP_GUARD = '''    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        if _use_rel_attn_fallback():
            # The SM12 fallback kernels are exercised by the ordinary vLLM
            # warmup forward; never JIT-compile FA4 on this arch (that
            # compile is what killed every v0.28.0 boot on GB10).
            return []
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
'''

# ---------------------------------------------------------------------------
# Injected source 4: the fallback implementation, appended at EOF.
# ---------------------------------------------------------------------------
FALLBACK_SRC = '''

''' + MARK + ''' SM12 (GB10/DGX Spark) relative-attention fallback.
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
    capturing = torch.cuda.is_current_stream_capturing()
    if (
        max_seqlen_q == 1
        and q.shape[0] == num_reqs
        and not _TRITON_DECODE_BROKEN
    ):
        decode_fn = _get_triton_decode_fn()
        if decode_fn is not None:
            try:
                # CUDA graph capture forbids device-to-host scalar copies.
                # The block-table capacity is static and safely upper-bounds
                # every runtime sequence captured by this graph.
                max_kv_len = (
                    max(1, block_table.shape[1] * key_cache.shape[1])
                    if capturing
                    else max(1, int(cache_seqlens.max().item()))
                )
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
                if capturing:
                    # The torch fallback below uses Python list/scalar copies
                    # and is intentionally not graph-capturable. Preserve the
                    # real Triton error instead of obscuring it downstream.
                    raise
                _TRITON_DECODE_BROKEN = True
                _sm12_logger.warning(
                    "inkling sm12 fallback: Triton decode failed (%r); "
                    "decode falls back to the torch SDPA path for the rest "
                    "of this process",
                    exc,
                    exc_info=True,
                )
    if capturing:
        raise RuntimeError(
            "inkling sm12 fallback: CUDA graph capture requires Triton decode"
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
'''

EOF_ANCHOR = "INKLING_FA4_REL_ATTENTION_KERNEL = InklingFA4RelAttentionKernel()"


def main() -> int:
    if not P.exists():
        print(f"missing {P}", file=sys.stderr)
        return 1
    src = P.read_text()

    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        print(f"{P}: {'PATCHED' if MARK in src else 'stock'}")
        return 0

    if MARK in src:
        print(f"{P}: already patched (marker present), no-op")
        return 0

    # Fail-closed anchor checks: the file must be v0.28.0 stock.
    anchors = {
        "_get_score_mod def": "@cache\ndef _get_score_mod(rel_extent: int) -> Callable:",
        "kernel() cute_window": KERNEL_ANCHOR,
        "get_warmup_keys def": WARMUP_ANCHOR,
        "module singleton": EOF_ANCHOR,
    }
    missing = [name for name, a in anchors.items() if a not in src]
    if missing:
        print(
            f"{P}: anchor(s) not found {missing} — file is not v0.28.0 "
            "stock; refusing to patch (fail-closed).",
            file=sys.stderr,
        )
        return 1

    # 1. gate helpers before _get_score_mod
    src = src.replace(
        "@cache\ndef _get_score_mod(rel_extent: int) -> Callable:",
        HELPER_SRC + "@cache\ndef _get_score_mod(rel_extent: int) -> Callable:",
        1,
    )
    # 2. dispatch at the top of kernel()
    src = src.replace(KERNEL_ANCHOR, KERNEL_DISPATCH + KERNEL_ANCHOR, 1)
    # 3. warmup guard
    src = src.replace(WARMUP_ANCHOR, WARMUP_GUARD, 1)
    # 4. fallback implementation at EOF
    src = src + FALLBACK_SRC

    P.write_text(src)
    print(f"{P}: patched (SM12 rel-attention fallback installed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
