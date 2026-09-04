# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling model implementation for NVIDIA GPUs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.models.inkling.common.mm_preprocess import (
    InklingDummyInputsBuilder,
    InklingMultiModalProcessor,
    InklingProcessingInfo,
    inkling_audio_enabled,
    inkling_vision_enabled,
)
from vllm.models.inkling.common.towers import InklingAudio, InklingVision
from vllm.multimodal import MULTIMODAL_REGISTRY
import os

from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Projective activation steering + DSPARK capture (Inkling).
# [steering-hotfix]
#
# h <- h - alpha * (h . d_hat) d_hat on the materialized post-layer
# residual stream [T, hidden] at chosen layers (no hyper-connection
# widening on this arch). Same intervention as the other lanes; see
# weightless/spec/GLP.md.
#
# Steering is inert unless WEIGHTLESS_STEER_PATH is set; capture is
# inert unless DSPARK_PROBE_DUMP_DIR is set.
# ---------------------------------------------------------------------------
# [steering-hotfix] projective activation steering (Inkling)

# Recorded for provenance in the startup log. Only "post_layer" is implemented
# here, which is the shipped setting and the one every measurement was taken at.
_WEIGHTLESS_STEER_HOOK = (os.environ.get("WEIGHTLESS_STEER_HOOK") or "post_layer").strip()
# layer id -> (k, hidden) orthonormal rows. Populated for inspection and for
# offline tooling; NOT read by the forward path.
_GLP_HOOK_DIRS: dict[int, torch.Tensor] = {}


def _read_gguf_cvec(path):
    """Minimal GGUF v3 reader for control vectors (F32 tensors only).

    The serving image does not ship the gguf package, so the container is
    parsed directly. Only what the control-vector format needs: metadata
    scalars/strings and direction.<N> tensor payloads.
    """
    import struct

    import numpy as np

    _SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    _FM = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"GGUF":
        raise ValueError(f"{path}: bad magic, not a GGUF file")
    ver, n_tensors, n_kv = struct.unpack_from("<IQQ", data, 4)
    if ver != 3:
        raise ValueError(f"{path}: GGUF version {ver}, expected 3")
    off = 24

    def rd_string(o):
        (n,) = struct.unpack_from("<Q", data, o)
        o += 8
        return data[o:o + n].decode("utf-8"), o + n

    meta = {}
    for _ in range(n_kv):
        key, off = rd_string(off)
        (vtype,) = struct.unpack_from("<I", data, off)
        off += 4
        if vtype == 8:  # string
            val, off = rd_string(off)
        elif vtype == 9:  # array
            etype, cnt = struct.unpack_from("<IQ", data, off)
            off += 12
            if etype == 8:
                arr = []
                for _ in range(cnt):
                    s, off = rd_string(off)
                    arr.append(s)
                val = arr
            else:
                val = struct.unpack_from(f"<{cnt}{_FM[etype][1]}", data, off)
                off += _SZ[etype] * cnt
        else:
            (val,) = struct.unpack_from(_FM[vtype], data, off)
            off += _SZ[vtype]
        meta[key] = val

    infos = []
    for _ in range(n_tensors):
        name, off = rd_string(off)
        (nd,) = struct.unpack_from("<I", data, off)
        off += 4
        dims = struct.unpack_from(f"<{nd}Q", data, off)
        off += 8 * nd
        (tt,) = struct.unpack_from("<I", data, off)
        off += 4
        (toff,) = struct.unpack_from("<Q", data, off)
        off += 8
        infos.append((name, dims, tt, toff))

    align = meta.get("general.alignment", 32)
    base = (off + align - 1) // align * align
    tensors = {}
    for name, dims, tt, toff in infos:
        if tt != 0:
            raise ValueError(f"{path}: {name} is not F32 (ggml type {tt})")
        n = 1
        for d in dims:
            n *= d
        tensors[name] = np.frombuffer(
            data, dtype="<f4", count=n, offset=base + toff
        ).copy()
    return meta, tensors


def _load_gguf_control_vector(path: str) -> dict:
    """Load a projective control vector from GGUF into {layer_id: tensor}.

    Container and tensor convention follow llama.cpp: tensors named
    "direction.<N>", fp32, 1-D, N >= 1, and **N is the layer index**.

    That last point is easy to get wrong, and we did.
    common_control_vector_load_one() stores direction.N at data offset
    (N-1)*n_embd, which reads like "N-1 is the layer". But
    llama_adapter_cvec::apply() then fills tensors[il] from offset
    (il-1)*n_embd, so the two -1s cancel: tensors[il] holds direction.il,
    applied at graph layer il. Confirmed by measurement in
    tests/test-cvec-layer-map.cpp in github.com/msuiche/llama.cpp -- a
    direction placed in one slot moves logits only when the layer range is
    restricted to the matching index.

    An off-by-one here degrades quietly instead of failing: adjacent-layer
    refusal directions are highly correlated, so a stack shifted by one
    layer still produces plausible output. Layer 0 cannot be expressed.

    `glp.mode` is enforced, not advisory. llama.cpp ADDS a control
    vector; we PROJECT one out. The same file under the wrong operation
    produces no error, just wrong output -- an additive apply pushes every
    token along the refusal axis instead of removing the component. So an
    unrecognised mode is a hard failure rather than a fallback.
    """
    import numpy as np

    meta, tensors = _read_gguf_cvec(path)

    mode = meta.get("glp.mode")
    if mode is None:
        raise ValueError(
            f"{path}: no glp.mode. Refusing to guess: an additive "
            f"control vector and a projective one are different operations."
        )
    if mode != "project":
        raise ValueError(
            f"{path}: glp.mode={mode!r}, but this runtime only "
            f"implements projective ablation (h -= alpha*(h.d)d). "
            f"Refusing to apply."
        )

    hook = meta.get("glp.hook_point")
    if hook is not None and hook != "residual_stream_post_layer":
        raise ValueError(
            f"{path}: glp.hook_point={hook!r} does not match this hook "
            f"(residual_stream_post_layer). The file's alpha was calibrated "
            f"for that hook; applying it here degrades silently rather "
            f"than erroring; refusing to apply."
        )

    logger.info(
        "weightless GLP vector: mode=%s spec_version=%s base_model=%s rev=%s",
        mode,
        meta.get("glp.spec_version", "?"),
        meta.get("general.base_model.0.name") or meta.get("glp.base_model"),
        str(meta.get("general.base_model.0.version")
            or meta.get("glp.base_revision") or "?")[:12],
    )

    out = {}
    for name, arr in tensors.items():
        dot = name.find(".")
        if dot < 0 or name[:dot] != "direction":
            continue
        try:
            idx = int(name[dot + 1:])
        except ValueError:
            raise ValueError(
                f"{path}: malformed tensor name {name!r} -- 'direction.' "
                f"must be followed by an integer layer id"
            ) from None
        if idx < 1:
            raise ValueError(
                f"{path}: {name} is invalid; direction.0 is rejected "
                f"upstream and layer 0 cannot be expressed in this "
                f"container"
            )
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        out[idx] = torch.from_numpy(arr.copy())  # N is the layer, no offset
    if not out:
        raise ValueError(f"{path}: no direction.<N> tensors found")
    widths = {v.numel() for v in out.values()}
    if len(widths) != 1:
        raise ValueError(
            f"{path}: inconsistent n_embd across directions: {widths}"
        )

    # Cross-check the tensor names against the informational layer list.
    # These are written from the same source, so a disagreement means the
    # file was produced by a broken exporter -- which is exactly what
    # happened here once: an exporter wrote direction.11..39 for directions
    # derived at layers 10..38 while this field still read 10..38. The
    # names are what execute, so the mismatch shifted the whole stack one
    # layer, and a one-layer shift degrades rather than fails (adjacent-
    # layer cosine 0.83-0.91). The file carried the evidence and nothing
    # looked at it. Now something does.
    declared = meta.get("glp.layer_ids_zero_based")
    if declared:
        try:
            want = sorted(int(x) for x in declared.split(",") if x.strip())
        except ValueError:
            raise ValueError(
                f"{path}: glp.layer_ids_zero_based is not a comma list of "
                f"integers: {declared!r}"
            ) from None
        if want and want != sorted(out):
            raise ValueError(
                f"{path}: glp.layer_ids_zero_based declares layers "
                f"{want[0]}..{want[-1]} ({len(want)} entries) but the "
                f"direction tensors resolve to {sorted(out)[0]}.."
                f"{sorted(out)[-1]} ({len(out)} entries). The tensor names "
                f"are what get applied, so this file would steer the wrong "
                f"layers. Re-export it."
            )
    logger.info(
        "weightless GLP vector: %d directions, n_embd=%d, layers %s",
        len(out), next(iter(widths)), sorted(out),
    )
    return out

from ..configs import InklingMMConfig, InklingModelConfig
from .attention import InklingAttention, compute_log_scaling_tau
from .layernorm import InklingRMSNorm
from .logits_processor import InklingLogitsProcessor
from .mlp import InklingDenseMLP
from .moe import InklingMoE
from .ops.lamport import get_lamport_rs_conv, initialize_lamport_rs_conv
from .ops.norm import add_rmsnorm, embed_rmsnorm
from .sconv_swa_attn import _ATTN, _MLP, InklingConvState, InklingSconvMetadata
from .short_conv import InklingShortConv

InklingDelta: TypeAlias = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _layer_id(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _sconv_add_norm(
    delta: InklingDelta,
    hidden: torch.Tensor,
    sconv: InklingShortConv,
    norm: InklingRMSNorm | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``h = hidden + sconv(TP-sum(delta)); y = rmsnorm(h)``.

    The Lamport path performs reduce-scatter + shard sconv + all-gather +
    residual add + norm. The NCCL path handles unsupported configurations."""
    attn_metadata = get_forward_context().attn_metadata
    m = (
        attn_metadata.get(sconv.owner.prefix)
        if isinstance(attn_metadata, dict)
        else None
    )
    cache = sconv.owner.kv_cache
    off_s, ws = sconv.owner.stream_ranges[sconv.stream_idx]
    norm_w = norm.weight if norm is not None else None
    eps = norm.variance_epsilon if norm is not None else 0.0
    if isinstance(delta, tuple):
        delta, shared_delta = delta
    else:
        shared_delta = None

    mm = get_lamport_rs_conv(hidden.shape[-1], sconv.kernel_size)
    if mm is not None and mm.usable(delta.shape[0]) and m is not None:
        assert cache.numel() > 0
        assert isinstance(m, InklingSconvMetadata)
        return mm.rs_sconv_ag_add_norm(
            delta,
            hidden,
            sconv.weight.squeeze(1),
            norm_w,
            eps,
            cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            off_s,
            ws,
            sconv.owner.block_size,
            shared_tensor=shared_delta,
        )

    # Fallback: NCCL RS -> shard sconv -> AG -> fused add(+rmsnorm).
    if shared_delta is not None:
        delta.add_(shared_delta)
    shard = tensor_model_parallel_reduce_scatter(delta, dim=-1)
    shard = sconv(shard.contiguous(), positions)
    full = tensor_model_parallel_all_gather(shard, dim=-1)
    if norm is None:
        return None, hidden + full
    return add_rmsnorm(hidden, full, norm_w, eps)


class InklingDecoderLayer(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        layer_id: int,
        is_local: bool,
        quant_config: QuantizationConfig | None,
        prefix: str,
        force_dense_mlp: bool = False,
    ) -> None:
        super().__init__()
        # Per-layer owner of the conv state as a paged SWA cache. The 4 sconv
        # streams (K/V/attn/mlp) are packed head-major into one block and share
        # it. Built first so the attention layer can wire its K/V sconv to it.
        self.conv_state = InklingConvState(
            num_kv_heads=(
                config.swa_num_key_value_heads
                if is_local
                else config.num_key_value_heads
            ),
            head_dim=config.swa_head_dim if is_local else config.head_dim,
            hidden_size=config.hidden_size,
            kernel_size=config.sconv_kernel_size,
            prefix=f"{prefix}.conv_state",
        )
        self.attn_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = InklingAttention(
            config,
            num_heads=(
                config.swa_num_attention_heads
                if is_local
                else config.num_attention_heads
            ),
            num_kv_heads=(
                config.swa_num_key_value_heads
                if is_local
                else config.num_key_value_heads
            ),
            head_dim=config.swa_head_dim if is_local else config.head_dim,
            rel_extent=config.rel_extent,
            local_extent=config.sliding_window_size,
            is_local=is_local,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
            conv_owner=self.conv_state,
        )
        self.mlp_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if force_dense_mlp or layer_id < config.dense_mlp_idx:
            self.mlp: nn.Module = InklingDenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                use_global_scale=config.use_global_scale,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = InklingMoE(
                config,
                prefix=f"{prefix}.mlp",
                quant_config=quant_config,
            )

        # Short convolution on the attention-output and MLP-output residual
        # streams, hidden-sharded: the sublayer outputs are reduce-scattered
        # to [T, H/tp], the sconv runs on the shard, and an all-gather
        # restores the full residual — all fused with the residual add + next
        # rmsnorm via the Lamport P2P kernels for decode-sized batches.
        tp_size = get_tensor_model_parallel_world_size()
        sconv_dim = config.hidden_size // tp_size
        self.attn_sconv = InklingShortConv(
            sconv_dim, config.sconv_kernel_size, owner=self.conv_state, stream_idx=_ATTN
        )
        self.mlp_sconv = InklingShortConv(
            sconv_dim, config.sconv_kernel_size, owner=self.conv_state, stream_idx=_MLP
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        pending: tuple[InklingDelta, InklingShortConv] | None = None,
        defer_mlp_add: bool = False,
        attn_in: torch.Tensor | None = None,
        log_scaling: torch.Tensor | None = None,
    ) -> (
        torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor | None, InklingShortConv]]
    ):
        # The previous sublayer's (pre-reduce, pre-sconv) delta is folded in
        # fused with its RS/sconv/AG and this layer's pre-attention rmsnorm.
        # A None delta means the partials sit in the NVLS symm buffer.
        if pending is None:
            if attn_in is None:
                # First layer; on the text path attn_norm comes fused with
                # the embedding gather (chain_weight in embed_rmsnorm).
                attn_in = self.attn_norm(hidden_states)
        else:
            attn_in, hidden_states = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.attn_norm, positions
            )
        attn_output = self.attn(positions, attn_in, log_scaling)
        mlp_in, hidden_states = _sconv_add_norm(
            attn_output, hidden_states, self.attn_sconv, self.mlp_norm, positions
        )
        mlp_output = (
            self.mlp.forward_partials(mlp_in)
            if isinstance(self.mlp, InklingMoE)
            else self.mlp(mlp_in)
        )
        if defer_mlp_add:
            # Caller folds mlp_output (pre-reduce, pre-sconv) into the next
            # fused sconv+add+rmsnorm.
            return hidden_states, (mlp_output, self.mlp_sconv)
        return _sconv_add_norm(
            mlp_output, hidden_states, self.mlp_sconv, None, positions
        )[1]


class InklingReplicatedEmbedding(nn.Module):
    """Full-vocab embedding table replicated on every TP rank.

    Trades the full table per rank (~2.3 GiB at V=201k / H=6144 bf16, vs a
    1/tp shard) for no masked lookup or per-lookup TP all-reduce, and keeps the
    full table on-rank for the fused gather+norm kernel. Bit-exact vs
    vocab-parallel: the all-reduce there only ever summed one real row against
    exact zeros. The LM head stays vocab-sharded.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=torch.get_default_dtype()),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return embed_rmsnorm(input_ids, self.weight, None, 0.0)


class InklingModel(nn.Module):
    def __init__(
        self,
        *,
        config: InklingModelConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = InklingReplicatedEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = (
            InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )
        local_ids = set(config.local_layer_ids)

        def get_layer(prefix: str) -> InklingDecoderLayer:
            idx = _layer_id(prefix + ".") or int(prefix.split(".")[-1])
            return InklingDecoderLayer(
                config, idx, idx in local_ids, quant_config, prefix
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

        # ---- projective activation steering [steering-hotfix] ---------------
        self._steer_alpha_val = float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
        self._steer_dirs: dict[int, torch.Tensor] = {}
        # Buffers are allocated on CPU (device=None): this module imports no
        # current_platform, and registered buffers ride the model's post-init
        # move to the accelerator. InklingModel.__init__ receives no
        # vllm_config, so the dtype is torch.get_default_dtype() -- vLLM's
        # loader sets the default dtype to the model dtype for the duration
        # of model construction.
        _dev = None
        _dtype = torch.get_default_dtype()
        # Allocated unconditionally, zeros when steering is off. A
        # None-when-disabled branch changes the traced graph, and that
        # difference is not part of vLLM's compile cache key: a compiled
        # artifact from one layer set was reused by another and died with
        # KeyError. A dense stack indexed by layer id keeps the graph
        # identical for every layer set, so only values change. The stream
        # here is hidden_size wide (no hyper-connection widening).
        self.register_buffer(
            "_steer_stack",
            torch.zeros(
                config.num_hidden_layers, 1, config.hidden_size,
                device=_dev, dtype=_dtype,
            ),
            persistent=False,
        )
        # alpha is a TENSOR, not a Python float. torch.compile bakes Python
        # scalars into the traced graph as constants and vLLM caches compiled
        # graphs, so a float alpha is frozen at first compile and every later
        # boot silently reuses it. A tensor is read at runtime, so alpha
        # takes effect without recompiling.
        self.register_buffer(
            "_steer_alpha",
            torch.zeros((), device=_dev, dtype=_dtype),
            persistent=False,
        )
        # Buffers are non-persistent so they never enter the state dict, which
        # would make load_weights report them as unexpected keys.
        self._load_steering(config, _dev, _dtype)

        # ---- dspark activation probe (capture lane) -------------------------
        # Inert unless DSPARK_PROBE_DUMP_DIR is set. Dumps the materialized
        # post-layer residual stream per layer during prefill; see the dump
        # site at the end of forward.
        self._probe_dump_dir = os.environ.get("DSPARK_PROBE_DUMP_DIR", "").strip()
        if self._probe_dump_dir:
            _probe_spec = os.environ.get("DSPARK_PROBE_LAYERS", "").strip()
            if _probe_spec:
                _probe_ids = {
                    int(t) for t in _probe_spec.replace(" ", "").split(",") if t
                }
            else:
                _probe_ids = set(range(config.num_hidden_layers))
            self._probe_layer_set = frozenset(
                i for i in _probe_ids if 0 <= i < config.num_hidden_layers
            )
            logger.info(
                "dspark inkling probe active: %d layers -> %s "
                "(EAGER ONLY: no torch.compile / CUDA graphs)",
                len(self._probe_layer_set),
                self._probe_dump_dir,
            )
        else:
            self._probe_layer_set = frozenset()
        self._probe_min_tokens = int(
            os.environ.get("DSPARK_PROBE_MIN_TOKENS", "4") or 4)
        self._probe_max_tokens = int(
            os.environ.get("DSPARK_PROBE_MAX_TOKENS", "1024") or 1024)
        self._probe_seq = 0

    def _load_steering(self, config, device, dtype) -> None:
        """Fill _steer_stack from WEIGHTLESS_STEER_PATH. No-op when unset.

        Loaded on every rank and indexed by GLOBAL layer id, so this is
        correct under pipeline parallelism: each rank's forward loop only
        visits its own layers and looks them up by the same global index.
        """
        path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
        if not path:
            return
        try:
            if path.endswith(".gguf"):
                raw = _load_gguf_control_vector(path)
            else:
                raw = torch.load(path, map_location="cpu")
            want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
            selected = (
                {int(t) for t in want.replace(" ", "").split(",") if t} if want else None
            )
            fallback = raw.get("global") if isinstance(raw, dict) else None

            for layer_id in range(config.num_hidden_layers):
                if selected is not None and layer_id not in selected:
                    continue
                vec = None
                if isinstance(raw, dict):
                    vec = raw.get(layer_id, raw.get(str(layer_id), fallback))
                if vec is None:
                    continue
                vec = vec.detach().to(torch.float32)
                if vec.ndim == 1:
                    vec = vec.unsqueeze(0)
                vec = vec.reshape(-1, vec.shape[-1])
                # Width guard (from the capture-validated reference lane): a
                # direction that does not match this arch's stream width must
                # fail here, not as an opaque broadcast error at serve time.
                # Inkling: the stream is plain hidden_size (no HC widening).
                if vec.shape[-1] != config.hidden_size:
                    raise RuntimeError(
                        f"steering vector layer {layer_id} width "
                        f"{vec.shape[-1]} != {config.hidden_size} "
                        f"(hidden_size)"
                    )
                # Rank-k: orthonormalise the basis, otherwise overlapping
                # components get subtracted more than once.
                q, _ = torch.linalg.qr(vec.T)
                vec = q.T[: vec.shape[0]]
                # Both of these are per layer and must stay inside this loop.
                # A previous revision (on the DSV4 lane) dedented them out,
                # which kept only the final iteration and silently steered
                # one layer instead of the full set.
                self._steer_dirs[layer_id] = vec.to(device=device, dtype=dtype)
                _GLP_HOOK_DIRS[layer_id] = vec.to(
                    device=device, dtype=torch.bfloat16
                )

            if isinstance(raw, dict):
                out_of_range = sorted(
                    int(k) for k in raw
                    if str(k).lstrip("-").isdigit()
                    and not 0 <= int(k) < config.num_hidden_layers
                )
                if out_of_range:
                    raise RuntimeError(
                        f"{path}: direction layers {out_of_range} out of range "
                        f"for this model ({config.num_hidden_layers} layers)"
                    )
            if not self._steer_dirs:
                raise RuntimeError(
                    f"WEIGHTLESS_STEER_PATH={path} matched no layers; "
                    f"refusing to run unsteered"
                )

            k = max(v.shape[0] for v in self._steer_dirs.values())
            stack = torch.zeros(
                config.num_hidden_layers, k, config.hidden_size,
                device=device, dtype=dtype,
            )
            for layer_id, vec in self._steer_dirs.items():
                stack[layer_id, : vec.shape[0]] = vec
            self._steer_stack = stack
            self._steer_alpha = torch.tensor(
                self._steer_alpha_val, device=device, dtype=dtype
            )
            logger.info(
                "weightless GLP steering active: hook=%s alpha=%.3f rank=%s "
                "layers=%d %s",
                _WEIGHTLESS_STEER_HOOK,
                self._steer_alpha_val,
                {k_: int(v.shape[0]) for k_, v in sorted(self._steer_dirs.items())},
                len(self._steer_dirs),
                sorted(self._steer_dirs),
            )
        except Exception as exc:
            # Fail closed: a boot asked for steering must not serve unsteered.
            logger.error("Inkling steering load failed (%s); failing closed", exc)
            raise

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Row gather + embed_norm in one launch.
        norm = self.embed_norm
        return embed_rmsnorm(
            input_ids,
            self.embed_tokens.weight,
            norm.weight if norm is not None else None,
            norm.variance_epsilon if norm is not None else 0.0,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        attn_in0: torch.Tensor | None = None
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                # embed_norm was already applied when producing inputs_embeds.
                hidden_states = inputs_embeds
            else:
                # Gather + embed_norm + the first layer's attn_norm, one launch.
                norm = self.embed_norm
                hidden_states, attn_in0 = embed_rmsnorm(
                    input_ids,
                    self.embed_tokens.weight,
                    norm.weight if norm is not None else None,
                    self.config.rms_norm_eps,
                    chain_weight=self.layers[self.start_layer].attn_norm.weight,
                )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        log_scaling = None
        if self.config.log_scaling_n_floor is not None:
            log_scaling = compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )

        _dspark_probe_store: dict[int, torch.Tensor] = {}
        pending: tuple[InklingDelta, InklingShortConv] | None = None
        for _wlayer_off, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            layer_idx = self.start_layer + _wlayer_off
            hidden_states, pending = layer(
                positions,
                hidden_states,
                pending=pending,
                defer_mlp_add=True,
                attn_in=attn_in0,
                log_scaling=log_scaling,
            )
            attn_in0 = None
            # ---- projective activation steering [steering-hotfix] -------
            # The MLP residual add is DEFERRED: pending carries this
            # layer's pre-reduce, pre-sconv MLP delta, fused into the
            # next layer's sconv+add+rmsnorm. The true post-layer
            # residual stream is only materialized by flushing pending --
            # the exact call the PP-boundary branch below uses. Flush
            # unconditionally (the traced graph must not depend on which
            # layers are steered), capture the pre-steer stream, steer
            # the materialized [T, hidden] stream, and continue with
            # pending=None: the next layer takes its pending=None path
            # (attn_norm of the materialized stream) -- same math, one
            # kernel less fused, the same tradeoff the qwen38fn hotfix
            # documents. Zero rows in _steer_stack are a numeric no-op.
            if pending is not None:
                hidden_states = _sconv_add_norm(
                    pending[0], hidden_states, pending[1], None, positions
                )[1]
                pending = None
            if self._probe_dump_dir and layer_idx in self._probe_layer_set:
                # Pre-steer stream; steering below rebinds hidden_states
                # to a fresh tensor, so the stored tensor stays pre-steer.
                _dspark_probe_store[layer_idx] = hidden_states
            steer_dirs = self._steer_stack[layer_idx]
            steer_coef = torch.einsum("...h,kh->...k", hidden_states, steer_dirs)
            hidden_states = hidden_states - self._steer_alpha * torch.einsum(
                "...k,kh->...h", steer_coef, steer_dirs
            )

        if not get_pp_group().is_last_rank:
            if pending is not None:
                hidden_states = _sconv_add_norm(
                    pending[0], hidden_states, pending[1], None, positions
                )[1]
            return IntermediateTensors({"hidden_states": hidden_states})
        if pending is not None:
            # Final RS/sconv/AG + residual add fused with the final rmsnorm.
            norm_out = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.norm, positions
            )[0]
            assert norm_out is not None
            return norm_out
        hidden_states = self.norm(hidden_states)

        # ---- dspark probe: dump prefill activations [steering-hotfix] --
        # Eager/debug lane: the GPU->CPU copies and torch.save here are
        # not trace/graph-safe (run capture with enforce_eager, as the
        # glm53 lane did); the stream-capture guard skips dumping under
        # CUDA graph capture instead of crashing. Non-last PP ranks
        # returned above, so only the last rank's layers are dumped.
        if (
            _dspark_probe_store
            and self._probe_dump_dir
            and get_tensor_model_parallel_rank() == 0
        ):
            _probe_t = hidden_states.shape[0]
            _capturing = (
                torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
            )
            if not _capturing and (
                self._probe_min_tokens <= _probe_t <= self._probe_max_tokens
            ):
                try:
                    _layers = sorted(_dspark_probe_store)
                    _mat = torch.stack(
                        [_dspark_probe_store[l] for l in _layers])
                    os.makedirs(self._probe_dump_dir, exist_ok=True)
                    torch.save(
                        {
                            "act_mean": _mat.float().mean(dim=1).cpu(),
                            "act_last": _mat[:, -1, :].float().cpu(),
                            "layers": _layers,
                            "n_tokens": int(_probe_t),
                        },
                        os.path.join(
                            self._probe_dump_dir,
                            "probe_%06d.pt" % self._probe_seq,
                        ),
                    )
                    self._probe_seq += 1
                except Exception as exc:  # never take the model down
                    logger.warning(
                        "dspark inkling probe dump failed: %s", exc)
        return hidden_states


class _TmlForCausalLMBase(nn.Module, SupportsPP, SupportsLoRA):
    """Shared text-backbone causal-LM scaffolding for both entry classes."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".w13_dn": ".gate_up_proj",
            ".w2_md": ".down_proj",
        },
        orig_to_new_stacked={
            ".attn.wq_du.": (".attn.qkvr.", 0),
            ".attn.wk_dv.": (".attn.qkvr.", 1),
            ".attn.wv_dv.": (".attn.qkvr.", 2),
            ".attn.wr_du.": (".attn.qkvr.", 3),
        },
        orig_to_new_prefix={
            "model.llm.layers.": "model.layers.",
            "model.llm.embed_norm": "model.embed_norm",
            "model.llm.embed": "model.embed_tokens",
            "model.llm.norm": "model.norm",
            "model.llm.unembed": "lm_head",
            "language_model.layers.": "model.layers.",
            "language_model.lm_head.": "lm_head.",
        },
        orig_to_new_suffix={
            # ModelOpt NVFP4 scales
            ".w13_weight.scale": ".w13_weight_scale",
            ".w13_weight.scale2": ".w13_weight_scale_2",
            ".w2_weight.scale": ".w2_weight_scale",
            ".w2_weight.scale2": ".w2_weight_scale_2",
            # Compressed tensors NVFP4 parameters
            ".w13_weight.input_global_scale": ".w13_input_global_scale",
            ".w13_weight.weight_global_scale": ".w13_weight_global_scale",
            ".w13_weight.weight_packed": ".w13_weight_packed",
            ".w13_weight.weight_scale": ".w13_weight_scale",
            ".w2_weight.input_global_scale": ".w2_input_global_scale",
            ".w2_weight.weight_global_scale": ".w2_weight_global_scale",
            ".w2_weight.weight_packed": ".w2_weight_packed",
            ".w2_weight.weight_scale": ".w2_weight_scale",
        },
    )
    packed_modules_mapping = {
        "qkvr": ["wq_du", "wk_dv", "wv_dv", "wr_du"],
        "w13": ["w1", "w3"],
    }
    embedding_modules = {
        "lm_head": "output_embeddings",
    }

    def _build(
        self,
        vllm_config: VllmConfig,
        text_config: InklingModelConfig,
        prefix: str,
    ) -> None:
        quant_config = vllm_config.quant_config
        self.config = text_config
        # Read by the MRV2 runner to publish per-request short-conv metadata.
        # Short convolution is intrinsic to Inkling, so this is always set.
        self.uses_sconv = True
        self.model = InklingModel(
            config=text_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        initialize_lamport_rs_conv(
            text_config.hidden_size,
            text_config.sconv_kernel_size,
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size,
            text_config.hidden_size,
            org_num_embeddings=text_config.padded_vocab_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = InklingLogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
            logits_mup_width_multiplier=text_config.logits_mup_width_multiplier,
        )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_inkling_weights(self, weights, self.config)


class InklingForCausalLM(_TmlForCausalLMBase):
    """Text-only entry point (``inkling_model`` checkpoints)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self._build(vllm_config, vllm_config.model_config.hf_config, prefix)


@MULTIMODAL_REGISTRY.register_processor(
    InklingMultiModalProcessor,
    info=InklingProcessingInfo,
    dummy_inputs=InklingDummyInputsBuilder,
)
class InklingForConditionalGeneration(_TmlForCausalLMBase, SupportsMultiModal):
    """Top-level (multimodal) entry point.

    Builds the vision + audio towers on top of the shared text backbone. Inkling has
    NO cross-modal fusion (the vision tower emits one token per patch, the audio
    tower one token per frame), so generation reuses the inherited backbone
    ``forward`` / ``compute_logits`` (the latter already applies muP) and this
    class only adds multimodal embedding + merge.
    """

    hf_to_vllm_mapper = _TmlForCausalLMBase.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "model.audio.": "audio.",
            "model.visual.": "visual.vision_encoder.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|content_image|>"
        if modality.startswith("audio"):
            return "<|content_audio_input|>"
        raise ValueError("Only image or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: InklingMMConfig = vllm_config.model_config.hf_config

        self.visual = (
            InklingVision(config.vision_config, prefix=maybe_prefix(prefix, "visual"))
            if inkling_vision_enabled(config)
            else None
        )
        self.audio = (
            InklingAudio(config.audio_config, prefix=maybe_prefix(prefix, "audio"))
            if inkling_audio_enabled(config)
            else None
        )

        self._build(vllm_config, config.text_config, prefix)

    # -- multimodal embedding -------------------------------------------

    def _process_image_input(
        self, pixel_values: Any, num_patches: Any
    ) -> tuple[torch.Tensor, ...]:
        assert self.visual is not None
        # pixel_values is a list (per item) of [P_i, 2, P, P, 3] tensors,
        # or a single concatenated tensor. Normalize to a flat batch, run the
        # tower once, then split back per item.
        if isinstance(pixel_values, (list, tuple)):
            if not pixel_values:
                return ()
            sizes = [int(p.shape[0]) for p in pixel_values]
            patches = torch.cat(list(pixel_values), dim=0)
        else:
            patches = pixel_values
            sizes = self._sizes_from(num_patches, patches.shape[0])

        patches = patches.to(device=self.visual.device, dtype=self.visual.dtype)
        embeds = self.visual(patches)  # [total_patches, D]
        return tuple(embeds.split(sizes))

    def _process_audio_input(
        self, input_audio_features: Any, num_audio_tokens: Any
    ) -> tuple[torch.Tensor, ...]:
        assert self.audio is not None
        if isinstance(input_audio_features, (list, tuple)):
            if not input_audio_features:
                return ()
            sizes = [int(d.shape[0]) for d in input_audio_features]
            dmel = torch.cat(list(input_audio_features), dim=0)
        else:
            dmel = input_audio_features
            sizes = self._sizes_from(num_audio_tokens, dmel.shape[0])

        dmel = dmel.to(device=self.audio.device)
        embeds = self.audio(dmel)  # [total_frames, D]
        return tuple(embeds.split(sizes))

    @staticmethod
    def _sizes_from(counts: Any, total: int) -> list[int]:
        if counts is None:
            return [total]
        if isinstance(counts, torch.Tensor):
            return [int(c) for c in counts.flatten().tolist()]
        if isinstance(counts, (list, tuple)):
            flat: list[int] = []
            for c in counts:
                flat.append(int(c.item()) if isinstance(c, torch.Tensor) else int(c))
            return flat
        return [int(counts)]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # Iterate modalities in a stable order so the returned per-item tensors
        # line up with their appearance order; the positional merge in
        # embed_input_ids handles actual placement.
        pixel_values = kwargs.get("pixel_values")
        num_patches = kwargs.get("num_patches")
        input_audio_features = kwargs.get("input_audio_features")
        num_audio_tokens = kwargs.get("num_audio_tokens")

        embeddings: tuple[torch.Tensor, ...] = ()
        if pixel_values is not None and self.visual is not None:
            embeddings += self._process_image_input(pixel_values, num_patches)
        if input_audio_features is not None and self.audio is not None:
            embeddings += self._process_audio_input(
                input_audio_features, num_audio_tokens
            )
        return embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Override the base's 1-arg embed_input_ids: the runner calls this 3-arg
        # signature for multimodal models. Text embeddings come from the shared
        # backbone (which applies embed_norm); MM embeddings are scattered in.
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        # Placeholder ids use unused vocabulary slots and these positions are
        # overwritten by MM embeds below.
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def get_language_model(self) -> nn.Module:
        # This class IS the causal LM (the towers are side branches), so the
        # language model is self — callers expect a module exposing ``.model``
        # and ``.lm_head``.
        return self


# ===========================================================================
# Weight loading
# ===========================================================================


_MOE_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<mlp>.*\.mlp)\.(?P<rest>(?:shared_)?experts\..+)$"
)


def _load_inkling_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    config: InklingModelConfig,
) -> set[str]:
    moe_modules = {
        name: mod for name, mod in module.named_modules() if isinstance(mod, InklingMoE)
    }
    loaded: set[str] = set()
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    local_ids = set(config.local_layer_ids)

    def _iter_loadable_weights() -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in module.hf_to_vllm_mapper.apply(weights):
            shard_id = getattr(weight, "shard_id", None)
            # Replicate K/V conv-free GQA heads when tp_size > num_kv_heads.
            if (
                shard_id in (1, 2)
                and name.endswith(".attn.qkvr.weight")
                and weight.shape[0] > 0
            ):
                lid = _layer_id(name)
                if lid is not None:
                    is_local = lid in local_ids
                    n_kv = (
                        config.swa_num_key_value_heads
                        if is_local
                        else config.num_key_value_heads
                    )
                    head_dim = config.swa_head_dim if is_local else config.head_dim
                    if tp_size > n_kv and weight.shape[0] == n_kv * head_dim:
                        kv_idx = (tp_rank * n_kv) // tp_size
                        weight = weight.narrow(0, kv_idx * head_dim, head_dim)
                        weight.shard_id = shard_id

            # MoE expert tensors (fused stacked, routed + shared sink): translate
            # the checkpoint layout to per-expert FusedMoE loads.
            moe_match = _MOE_EXPERT_WEIGHT_RE.match(name)
            if moe_match is not None and moe_match.group("mlp") in moe_modules:
                moe = moe_modules[moe_match.group("mlp")]
                for rel in moe.load_expert_weight(moe_match.group("rest"), weight):
                    loaded.add(f"{moe_match.group('mlp')}.{rel}")
                continue

            yield name, weight

    # The release checkpoint also carries auxiliary prediction-head weights;
    # they are not part of the causal LM served by this implementation.
    loader = AutoWeightsLoader(module, skip_prefixes=["model.mtp."])
    loaded |= loader.load_weights(_iter_loadable_weights())

    # Post-load MoE fixups (default input scales, zeroed EP-padding experts).
    for moe_name, moe in moe_modules.items():
        for rel in moe.finalize_load():
            loaded.add(f"{moe_name}.{rel}")
    return loaded


EntryClass = [InklingForCausalLM, InklingForConditionalGeneration]
