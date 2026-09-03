#!/usr/bin/env python3
"""Hotfix: projective activation steering for Qwen3.5/3.8 on vLLM 0.27.

Same intervention as the DSV4 steering patch (spec in
weightless/spec/GLP.md), ported to the Qwen3-Next-family model
stack that Qwen3.8-27B loads as in the eugr/drowzeys GB10 images
(``eugr/spark-vllm-b12x``, vLLM 0.27):

    h <- h - alpha * (h . d_hat) d_hat

on the post-layer residual stream, layers/alpha/vector gated by env:

    WEIGHTLESS_STEER_PATH    .gguf (spec-conformant cvec) or .pt {layer: tensor}
    WEIGHTLESS_STEER_ALPHA   float, default 1.0; the shipping Qwen vector is
                       calibrated AT alpha 1.0 (unlike the DSV4 lane's 4.0)
    WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids

TWO FILES are patched, and the reason bit us in production:

- ``qwen3_next.py`` gets the module block (GGUF reader + loader), the
  ``_load_steering`` method, and the per-layer apply in
  ``Qwen3NextModel.forward`` (which Qwen3_5Model inherits).
- ``qwen3_5.py`` gets the buffer registration in ``Qwen3_5Model.__init__``.
  Qwen3_5Model OVERRIDES __init__ and deliberately skips its parent's
  (``super(Qwen3NextModel, self).__init__()``), so buffers registered only
  in Qwen3NextModel.__init__ never exist on the class that actually serves
  — while the inherited forward still reads them. First boot crashed at
  torch.compile with "'Qwen3_5Model' object has no attribute '_steer_stack'".
  ``_load_steering`` itself stays on Qwen3NextModel: methods inherit, and
  its globals resolve in qwen3_next.py's module namespace.

Other architecture notes:

- The (hidden_states, residual) pair here is vLLM's decomposed convention:
  the true residual stream after a layer is ``hidden_states + residual`` (the
  add is fused into the next layernorm). The derivation measured the FULL
  post-layer stream (HF layer outputs), so the apply steers the sum and
  writes it back into hidden_states, leaving residual untouched.
- The GGUF reader and every spec check are verbatim from the DSV4 hotfix:
  the container contract (glp.mode=project, hook_point, layer-id
  cross-check, direction.0 rejection) is lane-independent.

Failure semantics (fail-closed where it matters):

- Anchors not found in ANY file: exit 1 if WEIGHTLESS_STEER_PATH is set (a
  boot that was asked for steering must not silently serve unsteered), exit 0
  otherwise — and either way NOTHING is written: the two-file patch is
  all-or-nothing (a half-patched tree breaks even unsteered boots).
- WEIGHTLESS_STEER_PATH set but the vector file is missing/invalid/non-project:
  exit 1, before the model load.
- Runtime load failures with steering armed (direction width != the model's
  stream width, no layers matched, direction layers out of range for the
  model) re-raise: the engine boot dies rather than serving unsteered.

Patches
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py
and .../qwen3_5.py in-place inside the container (called from the serve
script before ``exec vllm serve``). Idempotent per file: re-applying is a
no-op once the marker is present. ``--status`` reports state; ``--check``
validates the vector named by WEIGHTLESS_STEER_PATH without touching the files.
"""
import os
from pathlib import Path
import sys

# Overridable for dry-runs against copies outside the container.
P = Path(os.environ.get(
    "WEIGHTLESS_STEERING_MODEL_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py",
))
P35 = Path(os.environ.get(
    "WEIGHTLESS_STEERING_MODEL_PY_35",
    str(P.parent / "qwen3_5.py"),
))
MARK = "# [steering-hotfix] projective activation steering (Qwen3.5/3.8)"

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

# Module-level block injected into qwen3_next.py after the logger
# instantiation.
MODULE_BLOCK = (
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Projective activation steering (Qwen3.5/3.8). [steering-hotfix]\n"
    "#\n"
    "# h <- h - alpha * (h . d_hat) d_hat on the residual stream at chosen layers.\n"
    "# Same intervention as the DSV4 lane; see weightless/spec/GLP.md.\n"
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

# _load_steering method, injected into Qwen3NextModel (qwen3_next.py).
# Qwen3_5Model inherits it; its globals resolve in this module's namespace.
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
                if vec.shape[-1] != config.hidden_size:
                    raise RuntimeError(
                        f"steering vector layer {layer_id} width "
                        f"{vec.shape[-1]} != {config.hidden_size} (hidden_size)"
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
            logger.error("Qwen steering load failed (%s); failing closed", exc)
            raise
'''

# Buffer registration, injected into BOTH Qwen3NextModel.__init__
# (qwen3_next.py) and Qwen3_5Model.__init__ (qwen3_5.py). The qwen3_5 copy
# carries the marker: Qwen3_5Model.__init__ deliberately skips its parent's
# __init__, and a boot where only the parent got the buffers crashed at
# torch.compile ('Qwen3_5Model' object has no attribute '_steer_stack').
INIT_BLOCK = '''\

        # ---- projective activation steering ---------------------------------
        self._steer_alpha_val = float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
        self._steer_dirs: dict[int, torch.Tensor] = {}
        _dev = current_platform.device_type
        _dtype = vllm_config.model_config.dtype
        # Allocated unconditionally, zeros when steering is off. A
        # None-when-disabled branch changes the traced graph, and that
        # difference is not part of vLLM's compile cache key: a compiled
        # artifact from one layer set was reused by another and died with
        # KeyError. A dense stack indexed by layer id keeps the graph
        # identical for every layer set, so only values change.
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
'''

# Per-layer apply in Qwen3NextModel.forward (inherited by Qwen3_5Model).
FORWARD_BLOCK = '''\
            # Unconditional per-layer projection: h <- h - alpha (h.d) d. Rows
            # are zero for layers we do not steer, so this is a numeric no-op
            # there while the traced graph stays identical for every layer set.
            # In this decomposed (hidden_states, residual) convention the true
            # post-layer residual stream is hidden_states + residual (the add
            # is otherwise fused into the next layernorm). The derivation
            # measured the FULL stream, so steer the sum and write it back
            # into hidden_states, leaving residual untouched.
            steer_dirs = self._steer_stack[layer_idx]
            steer_stream = hidden_states + residual
            steer_coef = torch.einsum("...h,kh->...k", steer_stream, steer_dirs)
            hidden_states = steer_stream - self._steer_alpha * torch.einsum(
                "...k,kh->...h", steer_coef, steer_dirs
            ) - residual
'''

# --- qwen3_next.py anchors -------------------------------------------------
NEXT_ANCHOR_IMPORT = (
    "import torch\n"
    "from torch import nn"
)
NEXT_REPLACEMENT_IMPORT = (
    "import os\n"
    "import torch\n"
    "from torch import nn"
)

NEXT_ANCHOR_MODULE = (
    "logger = init_logger(__name__)\n"
)
NEXT_REPLACEMENT_MODULE = NEXT_ANCHOR_MODULE + MODULE_BLOCK

NEXT_ANCHOR_INIT = (
    "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    "\n"
    "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:"
)
NEXT_REPLACEMENT_INIT = (
    "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    + INIT_BLOCK
    + LOAD_METHOD_BLOCK
    + "\n"
    "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:"
)

NEXT_ANCHOR_FORWARD = (
    "            hidden_states, residual = layer(\n"
    "                positions=positions,\n"
    "                hidden_states=hidden_states,\n"
    "                residual=residual,\n"
    "            )\n"
    "            if (layer_idx + 1) in self.aux_hidden_state_layers"
)
NEXT_REPLACEMENT_FORWARD = (
    "            hidden_states, residual = layer(\n"
    "                positions=positions,\n"
    "                hidden_states=hidden_states,\n"
    "                residual=residual,\n"
    "            )\n"
    + FORWARD_BLOCK
    + "            if (layer_idx + 1) in self.aux_hidden_state_layers"
)

# v0.28.0 drift: the aux-hidden-state tail was refactored into a helper call.
NEXT_ANCHOR_FORWARD_V28 = (
    "            hidden_states, residual = layer(\n"
    "                positions=positions,\n"
    "                hidden_states=hidden_states,\n"
    "                residual=residual,\n"
    "            )\n"
    "            self._maybe_add_hidden_state(\n"
    "                aux_hidden_states, layer_idx + 1, hidden_states, residual\n"
    "            )"
)
NEXT_REPLACEMENT_FORWARD_V28 = (
    "            hidden_states, residual = layer(\n"
    "                positions=positions,\n"
    "                hidden_states=hidden_states,\n"
    "                residual=residual,\n"
    "            )\n"
    + FORWARD_BLOCK
    + "            self._maybe_add_hidden_state(\n"
    "                aux_hidden_states, layer_idx + 1, hidden_states, residual\n"
    "            )"
)


def _forward_anchor(src: str):
    """Pick the forward anchor variant matching the image's actual source."""
    if NEXT_ANCHOR_FORWARD not in src and NEXT_ANCHOR_FORWARD_V28 in src:
        return NEXT_ANCHOR_FORWARD_V28, NEXT_REPLACEMENT_FORWARD_V28
    return NEXT_ANCHOR_FORWARD, NEXT_REPLACEMENT_FORWARD


NEXT_PATCHES_BASE = (
    ("import os", NEXT_ANCHOR_IMPORT, NEXT_REPLACEMENT_IMPORT),
    ("module steering block", NEXT_ANCHOR_MODULE, NEXT_REPLACEMENT_MODULE),
    ("__init__ buffers + _load_steering", NEXT_ANCHOR_INIT, NEXT_REPLACEMENT_INIT),
)

NEXT_PATCHES = (
    ("import os", NEXT_ANCHOR_IMPORT, NEXT_REPLACEMENT_IMPORT),
    ("module steering block", NEXT_ANCHOR_MODULE, NEXT_REPLACEMENT_MODULE),
    ("__init__ buffers + _load_steering", NEXT_ANCHOR_INIT, NEXT_REPLACEMENT_INIT),
    ("forward apply", NEXT_ANCHOR_FORWARD, NEXT_REPLACEMENT_FORWARD),
)

# --- qwen3_5.py anchors ------------------------------------------------------
# Qwen3_5Model.__init__ overrides and SKIPS Qwen3NextModel.__init__, so the
# buffers must be registered here too. _load_steering is inherited.
P35_ANCHOR_IMPORT = (
    "import torch\n"
    "from torch import nn"
)
P35_REPLACEMENT_IMPORT = (
    "import os\n"
    "import torch\n"
    "from torch import nn"
)

P35_ANCHOR_PLATFORM = (
    "from vllm.logger import init_logger\n"
)
P35_REPLACEMENT_PLATFORM = (
    "from vllm.logger import init_logger\n"
    "from vllm.platforms import current_platform\n"
)

P35_ANCHOR_INIT = (
    "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    "\n"
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:"
)
P35_REPLACEMENT_INIT = (
    "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    + INIT_BLOCK
    + "        " + MARK + "\n"
    + "\n"
    "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:"
)

P35_PATCHES = (
    ("import os", P35_ANCHOR_IMPORT, P35_REPLACEMENT_IMPORT),
    ("import current_platform", P35_ANCHOR_PLATFORM, P35_REPLACEMENT_PLATFORM),
    ("__init__ buffers (Qwen3_5Model)", P35_ANCHOR_INIT, P35_REPLACEMENT_INIT),
)

FILE_PATCHES = (
    ("qwen3_next.py", P, NEXT_PATCHES),
    ("qwen3_5.py", P35, P35_PATCHES),
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

            # Execute the exact code that gets injected into qwen3_next.py,
            # with its two free names bound, and run the full spec validation
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

    # Preflight anchors for ALL files before writing any: the two-file patch
    # is all-or-nothing. A half-patched tree breaks even unsteered boots
    # (Qwen3_5Model.__init__ would call a _load_steering that was never
    # injected into qwen3_next.py), so drift anywhere means: steering
    # requested -> exit 1 with nothing written; steering off -> leave every
    # file stock.
    plans = []
    drifted = {}
    for label, path, patches in FILE_PATCHES:
        src = path.read_text()
        if MARK in src:
            print(f"[steering-hotfix] already applied to {path}")
            continue
        if label == "qwen3_next.py":
            # v0.28.0 refactored the aux-hidden-state tail into a helper;
            # pick the forward anchor that matches this image's source.
            _old, _new = _forward_anchor(src)
            patches = NEXT_PATCHES_BASE + (("forward apply", _old, _new),)

        missing = [name for name, old, _ in patches if old not in src]
        if missing:
            drifted[label] = missing
            continue
        plans.append((label, path, patches, src))

    if drifted:
        msg = (f"[steering-hotfix] anchors not found: {drifted}; "
               f"refusing to patch (nothing written)")
        if steer_requested():
            print(msg + " (WEIGHTLESS_STEER_PATH is set; failing closed)",
                  file=sys.stderr)
            return 1
        print(msg + " (steering off; leaving ALL files stock)")
        return 0

    for label, path, patches, src in plans:
        for name, old, new in patches:
            assert src.count(old) == 1, f"anchor {name!r} not unique in {label}"
            src = src.replace(old, new, 1)
        path.write_text(src)
        print(f"[steering-hotfix] applied to {path} ({len(patches)} anchors)")
    return check_vector() if steer_requested() else 0


if __name__ == "__main__":
    raise SystemExit(main())
