#!/usr/bin/env python3
"""Hotfix: projective activation steering + DSPARK activation capture for the
Inkling architecture (vllm/models/inkling/nvidia/model.py, vLLM v0.28.0).

Same intervention as the other lanes' steering patches (spec in
weightless/spec/GLP.md), modeled on hotfix-qwen38fn-steering-projective.py:

    h <- h - alpha * (h . d_hat) d_hat

on the post-layer residual stream, layers/alpha/vector gated by env:

    WEIGHTLESS_STEER_PATH    .gguf (spec-conformant cvec) or .pt {layer: tensor}
    WEIGHTLESS_STEER_ALPHA   float, default 1.0
    WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids

PLUS a capture lane (the DSPARK_PROBE idiom from the glm53 capture patch,
refusal-research/experiments/20260829-glm53-flagship/patch_glm53.py):

    DSPARK_PROBE_DUMP_DIR    when set, prefill activations are dumped here
    DSPARK_PROBE_LAYERS      optional comma list restricting captured layers
                             (default: all layers)
    DSPARK_PROBE_MIN_TOKENS  prefill gate, default 4
    DSPARK_PROBE_MAX_TOKENS  prefill gate, default 1024

Dump format per prefill forward (last PP rank, TP rank 0 only):
    {"act_mean": [n_layers, hidden] fp32, "act_last": [n_layers, hidden] fp32,
     "layers": sorted layer ids, "n_tokens": int}

Arch notes (what differs from the Qwen3.8-Flash-Next lane):

- NO hyper-connection widening: the stream between layers is plain
  [T, hidden] = [T, 4096]. The steer stack is hidden_size wide, not
  hc_count*hidden_size.
- The MLP residual add is DEFERRED: each decoder layer called with
  defer_mlp_add=True returns (hidden_states, pending) where pending carries
  this layer's pre-reduce, pre-sconv MLP delta, fused into the NEXT layer's
  sconv+add+rmsnorm (_sconv_add_norm). The true post-layer residual stream
  at iteration i is only materialized by flushing pending -- the same call
  the file's own PP-boundary branch uses:

      hidden_states = _sconv_add_norm(
          pending[0], hidden_states, pending[1], None, positions
      )[1]

  The apply flushes pending, captures (pre-steer) and steers the
  materialized stream, and continues with pending=None: the next layer
  takes its pending=None path (attn_norm of the materialized stream) --
  mathematically the same input, one kernel less fused. This is the same
  tradeoff the qwen38fn hotfix documents for its consumed pending combine.
- The apply is UNCONDITIONAL per layer with a dense zero-padded stack (zero
  rows are a numeric no-op): the traced graph stays identical for every
  layer set (the change-the-graph-between-layer-sets trap bit the Qwen3.8
  lane). The flush runs unconditionally too, so the graph does not depend on
  whether any given layer is steered.
- Layer ids: the loop iterates a SLICE of the layer list with a start_layer
  offset (pipeline parallelism). InklingDecoderLayer.__init__ takes layer_id
  but does not store it, so the apply enumerates the slice and indexes the
  steer stack by layer_idx = start_layer + offset (the GLOBAL layer id,
  correct under PP: each rank's loop only visits its own layers).
- InklingModel is a direct nn.Module whose own __init__ runs (both serving
  entry classes delegate to it via _build), so buffers registered there
  exist on the class whose forward reads them -- no __init__-override trap.
- Buffers are allocated on CPU and ride the model's post-init move to the
  accelerator (this module imports no current_platform). InklingModel.
  __init__ receives no vllm_config, so the buffer dtype is
  torch.get_default_dtype(), which vLLM's loader sets to the model dtype
  for the duration of model construction.
- Capture lane: GPU->CPU copies and torch.save are not trace/graph-safe.
  Run capture with enforce_eager (as the glm53 lane did); a stream-capture
  guard skips dumping under CUDA graph capture rather than crashing, and
  dump failures only warn -- the probe never takes the model down.
- The GGUF reader and every spec check are verbatim from the other lanes'
  hotfixes: the container contract (glp.mode=project, hook_point, layer-id
  cross-check, direction.0 rejection) is lane-independent.

Failure semantics (fail-closed where it matters):

- Anchors not found: exit 1 if WEIGHTLESS_STEER_PATH is set (a boot that was
  asked for steering must not silently serve unsteered), exit 0 otherwise.
- WEIGHTLESS_STEER_PATH set but the vector file is missing/invalid/non-project:
  exit 1, before the model load.
- Runtime load failures with steering armed (direction width != hidden_size,
  no layers matched, direction layers out of range for the model) re-raise:
  the engine boot dies rather than serving unsteered.

Patches
/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/model.py
in-place inside the container (called from the serve script before
``exec vllm serve``). Idempotent: re-applying is a no-op once the marker is
present. ``--status`` reports state; ``--check`` validates the vector named
by WEIGHTLESS_STEER_PATH without touching the file.
"""
import os
from pathlib import Path
import sys

# Overridable for dry-runs against copies outside the container.
P = Path(os.environ.get(
    "WEIGHTLESS_STEERING_MODEL_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/model.py",
))
MARK = "# [steering-hotfix] projective activation steering (Inkling)"

# ---------------------------------------------------------------------------
# Injected source: minimal GGUF reader + spec-conformant cvec loader.
# VERBATIM from hotfix-qwen38fn-steering-projective.py -- the container
# contract is shared across lanes, so the code that enforces it must be too.
# ---------------------------------------------------------------------------
GGUF_SRC = r'''
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
            f"(residual_stream_post_layer). The same vector at the wrong "
            f"hook point measured ~9x weaker; refusing to apply."
        )

    # A transferred vector (captured at one site, calibrated for another) is
    # legal but must not be silent: derived_at != hook_point means the
    # direction was estimated on a different distribution than the one it is
    # about to edit. Warning, never a refusal (GLP.md); alpha_default belongs
    # to the apply site.
    derived_at = meta.get("glp.derived_at")
    if derived_at and hook and derived_at != hook:
        logger.warning(
            "%s: transferred vector -- glp.derived_at=%r, apply hook %r",
            path, derived_at, hook,
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
'''

# Module-level block injected after the imports: logger instantiation (the
# stock file has none) plus the steering marker and the GGUF reader.
MODULE_BLOCK = (
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Projective activation steering + DSPARK capture (Inkling).\n"
    "# [steering-hotfix]\n"
    "#\n"
    "# h <- h - alpha * (h . d_hat) d_hat on the materialized post-layer\n"
    "# residual stream [T, hidden] at chosen layers (no hyper-connection\n"
    "# widening on this arch). Same intervention as the other lanes; see\n"
    "# weightless/spec/GLP.md.\n"
    "#\n"
    "# Steering is inert unless WEIGHTLESS_STEER_PATH is set; capture is\n"
    "# inert unless DSPARK_PROBE_DUMP_DIR is set.\n"
    "# ---------------------------------------------------------------------------\n"
    + MARK
    + "\n"
    "\n"
    "# Recorded for provenance in the startup log. Only \"post_layer\" is implemented\n"
    "# here, which is the shipped setting and the one every measurement was taken at.\n"
    "_WEIGHTLESS_STEER_HOOK = (os.environ.get(\"WEIGHTLESS_STEER_HOOK\") or \"post_layer\").strip()\n"
    "# layer id -> (k, hidden) orthonormal rows. Populated for inspection and for\n"
    "# offline tooling; NOT read by the forward path.\n"
    "_GLP_HOOK_DIRS: dict[int, torch.Tensor] = {}\n"
    "\n"
    + GGUF_SRC
)

# _load_steering method, injected into InklingModel. Verbatim from the
# Qwen3.8 lane except the stream width: here a layer's stream is hidden_size
# (4096), not hc_count*hidden_size -- Inkling has no hyper-connection
# widening.
LOAD_METHOD_BLOCK = '''\

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
'''

# Buffer registration + probe state, injected into InklingModel.__init__
# (the one and only __init__ on the serving path -- both entry classes
# delegate to InklingModel via _build; no __init__-override trap here).
INIT_BLOCK = '''\

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
'''

# Per-layer flush + capture + apply, injected at the InklingModel.forward
# decoder loop. The loop header gains an enumerate so the stack is indexed
# by the GLOBAL layer id (start_layer offset under PP); the probe store is
# initialized just before the loop.
ANCHOR_FORWARD = (
    "        pending: tuple[InklingDelta, InklingShortConv] | None = None\n"
    "        for layer in self.layers[self.start_layer : self.end_layer]:\n"
    "            hidden_states, pending = layer(\n"
    "                positions,\n"
    "                hidden_states,\n"
    "                pending=pending,\n"
    "                defer_mlp_add=True,\n"
    "                attn_in=attn_in0,\n"
    "                log_scaling=log_scaling,\n"
    "            )\n"
    "            attn_in0 = None\n"
)
REPLACEMENT_FORWARD = (
    "        _dspark_probe_store: dict[int, torch.Tensor] = {}\n"
    "        pending: tuple[InklingDelta, InklingShortConv] | None = None\n"
    "        for _wlayer_off, layer in enumerate(\n"
    "            self.layers[self.start_layer : self.end_layer]\n"
    "        ):\n"
    "            layer_idx = self.start_layer + _wlayer_off\n"
    "            hidden_states, pending = layer(\n"
    "                positions,\n"
    "                hidden_states,\n"
    "                pending=pending,\n"
    "                defer_mlp_add=True,\n"
    "                attn_in=attn_in0,\n"
    "                log_scaling=log_scaling,\n"
    "            )\n"
    "            attn_in0 = None\n"
    "            # ---- projective activation steering [steering-hotfix] -------\n"
    "            # The MLP residual add is DEFERRED: pending carries this\n"
    "            # layer's pre-reduce, pre-sconv MLP delta, fused into the\n"
    "            # next layer's sconv+add+rmsnorm. The true post-layer\n"
    "            # residual stream is only materialized by flushing pending --\n"
    "            # the exact call the PP-boundary branch below uses. Flush\n"
    "            # unconditionally (the traced graph must not depend on which\n"
    "            # layers are steered), capture the pre-steer stream, steer\n"
    "            # the materialized [T, hidden] stream, and continue with\n"
    "            # pending=None: the next layer takes its pending=None path\n"
    "            # (attn_norm of the materialized stream) -- same math, one\n"
    "            # kernel less fused, the same tradeoff the qwen38fn hotfix\n"
    "            # documents. Zero rows in _steer_stack are a numeric no-op.\n"
    "            if pending is not None:\n"
    "                hidden_states = _sconv_add_norm(\n"
    "                    pending[0], hidden_states, pending[1], None, positions\n"
    "                )[1]\n"
    "                pending = None\n"
    "            if self._probe_dump_dir and layer_idx in self._probe_layer_set:\n"
    "                # Pre-steer stream; steering below rebinds hidden_states\n"
    "                # to a fresh tensor, so the stored tensor stays pre-steer.\n"
    "                _dspark_probe_store[layer_idx] = hidden_states\n"
    "            steer_dirs = self._steer_stack[layer_idx]\n"
    "            steer_coef = torch.einsum(\"...h,kh->...k\", hidden_states, steer_dirs)\n"
    "            hidden_states = hidden_states - self._steer_alpha * torch.einsum(\n"
    "                \"...k,kh->...h\", steer_coef, steer_dirs\n"
    "            )\n"
)

# Dump site: the steering flush above always consumes pending, so the last
# rank reaches `return self.norm(hidden_states)`; rewrite it to dump the
# probe store (glm53 idiom) before returning.
ANCHOR_DUMP = (
    "        return self.norm(hidden_states)\n"
)
REPLACEMENT_DUMP = (
    "        hidden_states = self.norm(hidden_states)\n"
    "\n"
    "        # ---- dspark probe: dump prefill activations [steering-hotfix] --\n"
    "        # Eager/debug lane: the GPU->CPU copies and torch.save here are\n"
    "        # not trace/graph-safe (run capture with enforce_eager, as the\n"
    "        # glm53 lane did); the stream-capture guard skips dumping under\n"
    "        # CUDA graph capture instead of crashing. Non-last PP ranks\n"
    "        # returned above, so only the last rank's layers are dumped.\n"
    "        if (\n"
    "            _dspark_probe_store\n"
    "            and self._probe_dump_dir\n"
    "            and get_tensor_model_parallel_rank() == 0\n"
    "        ):\n"
    "            _probe_t = hidden_states.shape[0]\n"
    "            _capturing = (\n"
    "                torch.cuda.is_available()\n"
    "                and torch.cuda.is_current_stream_capturing()\n"
    "            )\n"
    "            if not _capturing and (\n"
    "                self._probe_min_tokens <= _probe_t <= self._probe_max_tokens\n"
    "            ):\n"
    "                try:\n"
    "                    _layers = sorted(_dspark_probe_store)\n"
    "                    _mat = torch.stack(\n"
    "                        [_dspark_probe_store[l] for l in _layers])\n"
    "                    os.makedirs(self._probe_dump_dir, exist_ok=True)\n"
    "                    torch.save(\n"
    "                        {\n"
    "                            \"act_mean\": _mat.float().mean(dim=1).cpu(),\n"
    "                            \"act_last\": _mat[:, -1, :].float().cpu(),\n"
    "                            \"layers\": _layers,\n"
    "                            \"n_tokens\": int(_probe_t),\n"
    "                        },\n"
    "                        os.path.join(\n"
    "                            self._probe_dump_dir,\n"
    "                            \"probe_%06d.pt\" % self._probe_seq,\n"
    "                        ),\n"
    "                    )\n"
    "                    self._probe_seq += 1\n"
    "                except Exception as exc:  # never take the model down\n"
    "                    logger.warning(\n"
    "                        \"dspark inkling probe dump failed: %s\", exc)\n"
    "        return hidden_states\n"
)

# --- anchors (validated against vLLM v0.28.0's
# vllm/models/inkling/nvidia/model.py) ---------------------------------------
ANCHOR_IMPORT = (
    "from vllm.sequence import IntermediateTensors\n"
)
REPLACEMENT_IMPORT = (
    "import os\n"
    "\n"
    "from vllm.logger import init_logger\n"
    "from vllm.sequence import IntermediateTensors\n"
    "\n"
    "logger = init_logger(__name__)\n"
    + MODULE_BLOCK
)

ANCHOR_INIT = (
    "        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(\n"
    "            [\"hidden_states\"], config.hidden_size\n"
    "        )\n"
)
REPLACEMENT_INIT = ANCHOR_INIT + INIT_BLOCK + LOAD_METHOD_BLOCK

PATCHES = (
    ("imports + module steering block", ANCHOR_IMPORT, REPLACEMENT_IMPORT),
    ("__init__ buffers + probe state + _load_steering", ANCHOR_INIT, REPLACEMENT_INIT),
    ("forward flush + capture + apply", ANCHOR_FORWARD, REPLACEMENT_FORWARD),
    ("probe dump after final norm", ANCHOR_DUMP, REPLACEMENT_DUMP),
)

FILE_PATCHES = (
    ("inkling/nvidia/model.py", P, PATCHES),
)


def steer_requested() -> bool:
    return bool(os.environ.get("WEIGHTLESS_STEER_PATH", "").strip())


def check_vector() -> int:
    """Validate the vector named by WEIGHTLESS_STEER_PATH. 0 ok, 1 bad."""
    path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
    if not path:
        print("[steering-hotfix] --check: WEIGHTLESS_STEER_PATH unset; nothing to check")
        return 0
    if not os.path.isfile(path):
        print(f"[steering-hotfix] --check: {path} not found", file=sys.stderr)
        return 1
    try:
        if path.endswith(".gguf"):
            import torch

            class _StderrLogger:  # stand-in for vllm's logger in the loader
                @staticmethod
                def info(msg, *args):
                    print("[steering-hotfix] --check: " + (msg % args))

                warning = info
                error = info

            # Execute the exact code that gets injected into model.py, with
            # its two free names bound, and run the full spec validation
            # (mode, hook point, layer cross-check), not a partial re-do.
            ns: dict = {"torch": torch, "logger": _StderrLogger}
            exec(GGUF_SRC, ns)  # noqa: S102 - same code that is injected
            out = ns["_load_gguf_control_vector"](path)
            layers = sorted(out)
            print(
                f"[steering-hotfix] --check: {os.path.basename(path)} OK: "
                f"mode=project layers {layers[0]}..{layers[-1]} ({len(layers)})"
            )
        else:
            import torch

            raw = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(raw, dict) or not raw:
                raise ValueError("expected non-empty {layer_id: tensor} dict")
            layers = sorted(int(k) for k in raw if str(k).isdigit())
            print(
                f"[steering-hotfix] --check: {os.path.basename(path)} OK: "
                f"pt layers {layers[0]}..{layers[-1]} ({len(layers)})"
            )
        want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
        if want:
            [int(t) for t in want.replace(" ", "").split(",") if t]
        float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
    except Exception as exc:
        print(f"[steering-hotfix] --check: {path}: {exc}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        for label, path, _ in FILE_PATCHES:
            src = path.read_text() if path.is_file() else ""
            print(
                f"steering (projective cvec) {label}:",
                "APPLIED" if MARK in src else "NOT APPLIED",
            )
        print(
            "WEIGHTLESS_STEER_PATH",
            "set" if steer_requested() else "unset",
        )
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return check_vector()

    rc = 0
    for label, path, patches in FILE_PATCHES:
        src = path.read_text()
        if MARK in src:
            print(f"[steering-hotfix] already applied to {path}")
            continue

        missing = [name for name, old, _ in patches if old not in src]
        if missing:
            msg = (
                f"[steering-hotfix] anchors not found in {label}: "
                f"{missing}; refusing to patch"
            )
            if steer_requested():
                print(msg + " (WEIGHTLESS_STEER_PATH is set; failing closed)",
                      file=sys.stderr)
                return 1
            print(msg + " (steering off; leaving the file stock)")
            rc = 0
            continue

        for name, old, new in patches:
            assert src.count(old) == 1, f"anchor {name!r} not unique in {label}"
            src = src.replace(old, new, 1)
        path.write_text(src)
        print(f"[steering-hotfix] applied to {path} ({len(patches)} anchors)")
    return check_vector() if steer_requested() and rc == 0 else rc


if __name__ == "__main__":
    raise SystemExit(main())
