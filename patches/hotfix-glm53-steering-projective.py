#!/usr/bin/env python3
"""Hotfix: projective activation steering for GLM-5.3-Flash (NVFP4) on the
sm121-patched day-0 image (``radixark/vllm-glm53-flash:sm121-v8`` — the
stock ``vllm/vllm-openai:glm53-flash-arm64-cu130`` dies on GB10 without the
patch stack; see recipe/glm53/README.md).

Same intervention as the other lanes (spec in weightless/spec/GLP.md):

    h <- h - alpha * (h . d_hat) d_hat

on the post-layer residual stream, layers/alpha/vector gated by env:

    WEIGHTLESS_STEER_PATH    .gguf (spec-conformant cvec) or .pt {layer: tensor}
    WEIGHTLESS_STEER_ALPHA   float, default 2.0 for this lane; the shipping
                       GLP-44 vector is calibrated AT alpha 2.0. alpha >= 2.5
                       GARBLES this model (measured) — the cliff is abrupt.
    WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids

ARCHITECTURE — where the apply lives, and why (this arch is mHC, not the
qwen3_8_flash_next delayed-combine scheme):

- Glm5NextDecoderLayer carries the residual stream WIDENED to
  [T, n, hidden] (n = config.mhc_num_residual_streams = 4, so 16384 floats =
  4x4096 — exactly the GLP-44 derivation space) split across the returned
  tuple (hidden_states = layer output [T, hidden], residual = widened
  stream, post/comb = the deferred hc_post mixes). Each layer's hc_post is
  DEFERRED and fused into the next layer's pre (MHCFusedPostPreOp).
- The full post-layer stream an HF forward hook sees is the materialization
      layer.hc_post(hidden_states, residual, post, comb)
  MHCPostOp is parameter-free given the precomputed mixes
  (out_j = post_mix_j * x + sum_i comb_mix_ij * residual_i), and
  MHCFusedPostPreOp is documented as exactly MHCPostOp followed by
  MHCPreOp — so materializing and letting the next layer take its
  standalone hc_pre path is the documented-equivalent decomposition, one
  kernel less fused. This is the apply site.
- The projection flattens the stream HC-outer to [T, n*hidden] (stream k
  occupies columns [k*hidden:(k+1)*hidden] — the same layout convention the
  Qwen3.8-Flash-Next capture validated at cos 0.9931), projects, reshapes
  back, and continues materialized (residual/post/comb cleared).
- The LAST mHC layer is the one special case: stock code runs hc_post +
  hc_contract inside the decoder and returns the contracted [T, hidden]
  stream, which would escape steering (GLP-44 covers layer 44). The hotfix
  removes that in-layer special case; the forward loop materializes, steers,
  and contracts there instead.

Discipline carried over from the other lanes:

- The apply is UNCONDITIONAL per layer with a dense zero-padded stack (zero
  rows are a numeric no-op): the traced graph stays identical for every
  layer set, which vLLM's compile cache does not key on.
- alpha is a tensor buffer, not a Python float (compile-baking trap).
- Buffers register on Glm5NextModel — the class whose forward contains the
  apply. No Qwen3_5Model-style skip-parent trap here: Glm5NextForCausalLM
  (and the multimodal wrapper) delegate to Glm5NextModel, whose own
  __init__ runs. This module already imports current_platform and a logger.
- Four-node lane: the start script runs this hotfix in ALL FOUR containers
  (each rank patches its own in-image copy) before ``vllm serve``.
- The GGUF reader and every spec check are verbatim from the other lanes'
  hotfixes: the container contract (glp.mode=project, hook_point, layer-id
  cross-check, direction.0 rejection) is lane-independent.

Residual risks, documented honestly:

- The anchors were validated against the day-0 PR source
  (vllm-project/vllm#53906 @ 142062f1, vendored in
  weightless/patches/reference/glm5next.py), NOT against the image's
  installed file (the image predates that revision by a day and was not
  pullable where this was written). The anchor check is fail-closed: drift
  aborts the boot when steering is requested instead of serving unsteered.
  Verify against the image on first deploy (see recipe README).
- The stream flattening order (stream-major / HC-outer) matches the
  derivation convention by analogy with the validated qwen38fn capture; it
  has NOT been re-measured for GLM-5.3. A wrong order would steer garbage —
  confirm on first hardware boot (a refusal32 A/B arm, or the boot log's
  layer count plus a known-refusal prompt).
- Non-mHC layers (MTP draft, the 70B fallback path) return post=None and
  are never steered; GLM-5.3-Flash's 45 base layers are all mHC.
- PP is gated off upstream for this model (no
  make_empty_intermediate_tensors); this lane is TP=4, PP=1.

Failure semantics (fail-closed where it matters):

- Anchors not found: exit 1 if WEIGHTLESS_STEER_PATH is set (a boot that was
  asked for steering must not silently serve unsteered), exit 0 otherwise.
- WEIGHTLESS_STEER_PATH set but the vector file is missing/invalid/non-project:
  exit 1, before the model load.
- Runtime load failures with steering armed (direction width != the model's
  stream width, no layers matched, direction layers out of range for the
  model) re-raise: the engine boot dies rather than serving unsteered.

Patches
/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py
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
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py",
))
MARK = "# [steering-hotfix] projective activation steering (GLM-5.3-Flash)"

# ---------------------------------------------------------------------------
# Injected source: minimal GGUF reader + spec-conformant cvec loader.
# VERBATIM from hotfix-dsv4-steering-projective.py -- the container contract
# is shared across lanes, so the code that enforces it must be too.
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
                f"{path}: malformed tensor name {name!r} — 'direction.' "
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

# Module-level block injected after the logger instantiation (the stock file
# already has one) plus the steering marker and the GGUF reader.
MODULE_BLOCK = (
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Projective activation steering (GLM-5.3-Flash). [steering-hotfix]\n"
    "#\n"
    "# h <- h - alpha * (h . d_hat) d_hat on the materialized post-layer mHC\n"
    "# stream [T, n, hidden] (flattened HC-outer to n*hidden for the\n"
    "# projection) at chosen layers. Same intervention as the other lanes;\n"
    "# see weightless/spec/GLP.md.\n"
    "#\n"
    "# Everything here is inert unless WEIGHTLESS_STEER_PATH is set.\n"
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

# _load_steering method, injected into Glm5NextModel. Verbatim from the other
# lanes except the stream width: here a layer's stream is
# mhc_num_residual_streams*hidden (16384), not hidden_size.
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
                if vec.shape[-1] != config.hidden_size * config.mhc_num_residual_streams:
                    raise RuntimeError(
                        f"steering vector layer {layer_id} width "
                        f"{vec.shape[-1]} != "
                        f"{config.hidden_size * config.mhc_num_residual_streams} "
                        f"(mhc_num_residual_streams*hidden_size)"
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
                config.num_hidden_layers, k,
                config.hidden_size * config.mhc_num_residual_streams,
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
            logger.error("GLM-5.3-Flash steering load failed (%s); failing closed", exc)
            raise
'''

# Buffer registration, injected into Glm5NextModel.__init__ (the class whose
# forward contains the apply — Glm5NextForCausalLM and the multimodal wrapper
# delegate to it; no skip-parent trap).
INIT_BLOCK = '''\

        # ---- projective activation steering ---------------------------------
        # GLP-44 is calibrated AT alpha 2.0 (alpha >= 2.5 garbles this model),
        # so this lane's code default is 2.0 — not the other lanes' 1.0.
        self._steer_alpha_val = float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "2.0") or 2.0)
        self._steer_dirs: dict[int, torch.Tensor] = {}
        _dev = current_platform.device_type
        _dtype = vllm_config.model_config.dtype
        # Allocated unconditionally, zeros when steering is off. A
        # None-when-disabled branch changes the traced graph, and that
        # difference is not part of vLLM's compile cache key: a compiled
        # artifact from one layer set was reused by another and died with
        # KeyError. A dense stack indexed by layer id keeps the graph
        # identical for every layer set, so only values change. The stream
        # here is mhc_num_residual_streams*hidden wide (mHC multi-stream).
        self.register_buffer(
            "_steer_stack",
            torch.zeros(
                config.num_hidden_layers, 1,
                config.hidden_size * config.mhc_num_residual_streams,
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
'''

# Per-layer apply in Glm5NextModel.forward, right after each layer call.
FORWARD_BLOCK = '''\
            # ---- projective activation steering [steering-hotfix] -----------
            # Unconditional per-layer projection on the MATERIALIZED
            # post-layer stream. mHC defers each layer's hc_post (fused into
            # the next layer's pre); the full post-layer stream an HF hook
            # sees is the parameter-free layer.hc_post(hidden_states,
            # residual, post, comb) — MHCFusedPostPreOp is documented as
            # exactly MHCPostOp + MHCPreOp, so materializing and letting the
            # next layer take its standalone hc_pre is the same math, one
            # kernel less fused. The stream is widened [T, n, hidden]; the
            # projection flattens it HC-outer to n*hidden (the GLP-44
            # derivation space) and reshapes back. Zero rows in _steer_stack
            # are a numeric no-op, so the traced graph is identical for every
            # layer set.
            if post is not None:
                steer_stream = layer.hc_post(hidden_states, residual, post, comb)
                steer_flat = steer_stream.flatten(-2)
                steer_dirs = self._steer_stack[layer.layer_idx]
                steer_coef = torch.einsum("...h,kh->...k", steer_flat, steer_dirs)
                hidden_states = (
                    steer_flat
                    - self._steer_alpha
                    * torch.einsum("...k,kh->...h", steer_coef, steer_dirs)
                ).reshape(steer_stream.shape)
                if layer.layer_idx == self.config.num_hidden_layers - 1:
                    # The hotfix removed the last layer's in-decoder
                    # hc_post+hc_contract so layer N-1 is steered too;
                    # contract here instead, after steering.
                    hidden_states = hc_contract(hidden_states, layer.n)
                residual = None
                post = None
                comb = None
'''

# Decoder last-layer deferral: stock materializes + contracts the last mHC
# layer inside the decoder (which would escape steering); defer like every
# other layer so the loop apply sees it.
DEFER_BLOCK = '''\
        # [steering-hotfix] the last mHC layer defers its hc_post like every
        # other layer: Glm5NextModel.forward materializes + steers the
        # post-layer stream and contracts there, so the GLP hook covers the
        # final layer too.
        return x, residual, post, comb
'''

# --- anchors (validated against the day-0 PR source, vllm-project/vllm#53906
# @ 142062f1 = patches/reference/glm5next.py) --------------------------------
ANCHOR_IMPORT = (
    "import torch\n"
    "from torch import nn\n"
)
REPLACEMENT_IMPORT = (
    "import os\n"
    "import torch\n"
    "from torch import nn\n"
)

ANCHOR_MODULE = (
    "logger = init_logger(__name__)\n"
)
REPLACEMENT_MODULE = ANCHOR_MODULE + MODULE_BLOCK

ANCHOR_INIT = (
    "        world_size = get_tensor_model_parallel_world_size()\n"
    "        assert config.num_attention_heads % world_size == 0, (\n"
    "            \"num_attention_heads must be divisible by world_size\"\n"
    "        )\n"
)
REPLACEMENT_INIT = ANCHOR_INIT + INIT_BLOCK + LOAD_METHOD_BLOCK

ANCHOR_FORWARD = (
    "        for layer in self._active_layers:\n"
    "            hidden_states, residual, post, comb = layer(\n"
    "                positions, hidden_states, residual, post, comb\n"
    "            )\n"
)
REPLACEMENT_FORWARD = ANCHOR_FORWARD + FORWARD_BLOCK

ANCHOR_DEFER = (
    "        if self.layer_idx == self.num_hidden_layers - 1:\n"
    "            x = self.hc_post(x, residual, post, comb)\n"
    "            x = hc_contract(x, self.n)\n"
    "            return x, None, None, None\n"
    "\n"
    "        return x, residual, post, comb\n"
)
REPLACEMENT_DEFER = DEFER_BLOCK

PATCHES = (
    ("import os", ANCHOR_IMPORT, REPLACEMENT_IMPORT),
    ("module steering block", ANCHOR_MODULE, REPLACEMENT_MODULE),
    ("__init__ buffers + _load_steering", ANCHOR_INIT, REPLACEMENT_INIT),
    ("forward apply", ANCHOR_FORWARD, REPLACEMENT_FORWARD),
    ("last-layer deferral", ANCHOR_DEFER, REPLACEMENT_DEFER),
)

FILE_PATCHES = (
    ("glm5next/nvidia/model.py", P, PATCHES),
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
