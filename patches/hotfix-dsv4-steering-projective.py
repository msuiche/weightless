#!/usr/bin/env python3
"""Hotfix: projective activation steering (DSpark) for DeepSeek-V4-Flash.

Port of the v0.27 steering patch
(weightless/patches/0001-dspark-projective-steering.patch, spec in
weightless/spec/GLP.md) to the Anemll 0.1.1 image's vLLM
0.25.2.dev0 tree.

    h <- h - alpha * (h . d_hat) d_hat

on the FFN writer output, pre-fold (``ffn_out``, the pending write the
decoder layer returns before the mHC fold lands in the next layer's fused
post/pre call). layers/alpha/vector gated by env:

    WEIGHTLESS_STEER_PATH    .gguf (spec-conformant cvec) or .pt {layer: tensor}
    WEIGHTLESS_STEER_ALPHA   float, default 1.0 in code; deployment pins 4.0
                               (keysdir vector) or 6.0 (GLP-29 cyber, 2026-09-04)
    WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids

2026-09-04 hook-site correction: this hotfix was documented as steering the
post-layer residual stream. It does not, and never did on this tree — the
model-loop anchor's ``hidden_states`` IS the layer's pending FFN write
(shape-probe verified in-image against the 0.25.2 mHC kernel signatures; the
same return convention exists in the v0.27 tree). Every published vLLM-lane
DSV4 number was measured at this FFN-writer anchor, and the measured site
ordering on DSV4-Flash-0731 is FFN >> residual >> attention, so the anchor
stays; only the label was wrong. The declared hook is now
``ffn_out_pre_residual`` and a ``residual_stream_post_layer`` file is the
mismatch case — it fails closed. Vectors captured at the residual and applied
here are transferred vectors: legal, but ``glp.derived_at`` must then differ
from ``glp.hook_point`` and the loader warns loudly.

Differences from the v0.27 patch:

- The Anemll image does NOT ship the ``gguf`` package, so the GGUF reader
  is a small embedded parser (GGUF v3, F32 1-D tensors only) instead of
  ``gguf.GGUFReader``. All spec checks are unchanged: glp.mode is
  enforced (missing or non-"project" is fatal), glp.hook_point must be
  ffn_out_pre_residual, direction.0 is rejected, and
  glp.layer_ids_zero_based is cross-checked against the tensor names.
- Everything else is verbatim: dense zero-padded _steer_stack indexed by
  layer id, unconditional per-layer apply (never a Python ``if``), alpha as
  a registered tensor buffer. The comments in the injected block explain
  why; those were all measured failure modes on the v0.27 line.

Failure semantics (fail-closed where it matters):

- Anchors not found: exit 1 if WEIGHTLESS_STEER_PATH is set (a boot that was
  asked for steering must not silently serve unsteered), exit 0 otherwise
  (stock behaviour, other hotfixes still guard the stack).
- WEIGHTLESS_STEER_PATH set but the vector file is missing/invalid/non-project:
  exit 1, before the ~8-minute model load.
- Runtime load failures with steering armed (direction width != the model's
  stream width, no layers matched, direction layers out of range for the
  model) re-raise: the engine boot dies rather than serving unsteered.

Patches
/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py
in-place inside the container (called from the compose entrypoint before
``exec vllm serve``). Idempotent: re-applying is a no-op once the marker
is present. ``--status`` reports state; ``--check`` validates the vector
named by WEIGHTLESS_STEER_PATH without touching model.py.
"""
import os
from pathlib import Path
import sys

# Overridable for dry-runs against a copy of model.py outside the container.
P = Path(os.environ.get(
    "WEIGHTLESS_STEERING_MODEL_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py",
))
MARK = "# [steering-hotfix] projective activation steering (DSpark)"

# ---------------------------------------------------------------------------
# Injected source: minimal GGUF reader + spec-conformant cvec loader.
# Kept as one constant so the hotfix's own --check mode executes the exact
# code that gets injected into model.py.
# ---------------------------------------------------------------------------
GGUF_SRC = r'''
def _read_gguf_cvec(path):
    """Minimal GGUF v3 reader for control vectors (F32 tensors only).

    The Anemll 0.1.1 image does not ship the gguf package, so the container
    is parsed directly. Only what the control-vector format needs: metadata
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
    refusal directions have cosine 0.83-0.91, so a stack shifted by one
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
    if hook is not None and hook != "ffn_out_pre_residual":
        raise ValueError(
            f"{path}: glp.hook_point={hook!r} does not match this hook "
            f"(ffn_out_pre_residual — the pending FFN write, pre-fold; see "
            f"the 2026-09-04 site correction in the hotfix docstring). "
            f"Refusing to apply at the wrong site."
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

# Module-level block injected after the ubatching import.
MODULE_BLOCK = (
    "\n"
    "from vllm.logger import init_logger\n"
    "\n"
    "logger = init_logger(__name__)\n"
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Projective activation steering (DSpark). [steering-hotfix]\n"
    "#\n"
    "# h <- h - alpha * (h . d_hat) d_hat on the residual stream at chosen layers.\n"
    "# Activation space rather than weight space on purpose: on this checkpoint the\n"
    "# fp8 requantisation floor is ~39% of a lambda=3.5 weight edit, so small\n"
    "# coefficients are not representable in the weights at all.\n"
    "#\n"
    "# Everything here is inert unless WEIGHTLESS_STEER_PATH is set.\n"
    "# ---------------------------------------------------------------------------\n"
    + MARK
    + "\n"
    "\n"
    "# Recorded for provenance in the startup log. The only site implemented at\n"
    "# this anchor is \"ffn_out_pre_residual\": on this tree the model loop's\n"
    "# hidden_states IS the layer's pending FFN write (the mHC fold is deferred to\n"
    "# the next layer's fused post/pre call). Every published vLLM-lane DSV4\n"
    "# number was measured here. Measured 2026-09-04 (hook-site experiment, same\n"
    "# direction/layers): FFN writer >> post-layer residual >> attention writer;\n"
    "# FFN window alpha 4-6, residual garbles at 4.0, attention ~4-5/32 at 4.0.\n"
    "# WEIGHTLESS_STEER_HOOK set to anything else fails closed in _load_steering.\n"
    "_WEIGHTLESS_STEER_HOOK = (os.environ.get(\"WEIGHTLESS_STEER_HOOK\") or \"ffn_out_pre_residual\").strip()\n"
    "_GLP_HOOK_ALPHA = float(os.environ.get(\"WEIGHTLESS_STEER_ALPHA\", \"1.0\") or 1.0)\n"
    "# layer id -> (k, hidden) orthonormal rows. Populated for inspection and for\n"
    "# offline tooling; NOT read by the forward path. The pre-fold writer-isolation\n"
    "# hooks that used it are deliberately not ported: they branch on\n"
    "# _WEIGHTLESS_STEER_HOOK inside the traced region, and that choice is not part of\n"
    "# vLLM's compile cache key, so a cached artifact would silently ignore a changed\n"
    "# hook site. Reintroduce them behind a separate image if that experiment is\n"
    "# re-run, and clear the compile cache when doing so.\n"
    "_GLP_HOOK_DIRS: dict[int, torch.Tensor] = {}\n"
    "\n"
    + GGUF_SRC
)

# __init__ tail block + _load_steering method, injected between the
# _mtp_hidden_buffer else-branch and embed_input_ids.
INIT_BLOCK = '''\

        # ---- projective activation steering ---------------------------------
        self._steer_alpha_val = float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
        self._steer_dirs: dict[int, torch.Tensor] = {}
        _dev = current_platform.device_type
        _dtype = vllm_config.model_config.dtype
        # Allocated unconditionally, zeros when steering is off. A
        # None-when-disabled branch changes the traced graph, and that
        # difference is not part of vLLM's compile cache key: a compiled
        # artifact from a 29-layer run was reused by a 16-layer run and died
        # with KeyError: 23. Clearing torch_compile_cache was not enough,
        # because an AOT artifact kept it. A dense stack indexed by layer id
        # keeps the graph identical for every layer set, so only values change.
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
        # graphs under VLLM_CACHE_ROOT, so a float alpha is frozen at first
        # compile and every later boot silently reuses it. Measured: alpha of
        # 1.0, 2.0 and 4.0 produced byte-identical refusal rates. A tensor is
        # read at runtime, so alpha takes effect without recompiling.
        self.register_buffer(
            "_steer_alpha",
            torch.zeros((), device=_dev, dtype=_dtype),
            persistent=False,
        )
        # Buffers are non-persistent so they never enter the state dict, which
        # would make load_weights report them as unexpected keys.
        self._load_steering(config, _dev, _dtype)

    def _load_steering(self, config, device, dtype) -> None:
        """Fill _steer_stack from WEIGHTLESS_STEER_PATH. No-op when unset.

        Loaded on every rank and indexed by GLOBAL layer id, so this is correct
        under pipeline parallelism: each rank's forward loop only visits its own
        layers and looks them up by the same global index.
        """
        path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
        if not path:
            return
        if _WEIGHTLESS_STEER_HOOK != "ffn_out_pre_residual":
            raise RuntimeError(
                f"WEIGHTLESS_STEER_HOOK={_WEIGHTLESS_STEER_HOOK!r} is not "
                f"implemented at this anchor; the only site here is "
                f"ffn_out_pre_residual (the pending FFN write, pre-fold). "
                f"Refusing to serve with a silently wrong hook site."
            )
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
                # components get subtracted more than once. Refusal measured
                # only 0.147 rank-1 energy here, so k>1 is expected to matter.
                q, _ = torch.linalg.qr(vec.T)
                vec = q.T[: vec.shape[0]]
                # Both of these are per layer and must stay inside this loop.
                # A previous revision dedented them out, which kept only the
                # final iteration and silently steered one layer instead of 29.
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
            logger.error("DSpark steering load failed (%s); failing closed", exc)
            raise
'''

# Per-layer apply in the forward loop.
FORWARD_BLOCK = '''\
            # Unconditional per-layer projection: h <- h - alpha (h.d) d. Rows
            # are zero for layers we do not steer, so this is a numeric no-op
            # there while the traced graph stays identical for every layer set.
            # hidden_states at this anchor is the layer's PENDING FFN WRITE,
            # shape (tokens, hidden_size) — the mHC fold of that write into the
            # hyper-connection streams is deferred to the next layer's fused
            # post/pre call (verified 2026-09-04 against the 0.25.2 kernel
            # signatures; an earlier version of this comment claimed
            # (tokens, hc_mult, hidden_size) — that was wrong).
            steer_dirs = self._steer_stack[idx]
            steer_coef = torch.einsum("...h,kh->...k", hidden_states, steer_dirs)
            hidden_states = hidden_states - self._steer_alpha * torch.einsum(
                "...k,kh->...h", steer_coef, steer_dirs
            )
'''

ANCHOR_IMPORT = (
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "import typing"
)
REPLACEMENT_IMPORT = (
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
    "import os\n"
    "import typing"
)

ANCHOR_MODULE = (
    "from vllm.v1.worker.ubatching import dbo_current_ubatch_id\n"
)
REPLACEMENT_MODULE = ANCHOR_MODULE + MODULE_BLOCK

ANCHOR_INIT = (
    "        else:\n"
    "            self._mtp_hidden_buffer = None\n"
    "\n"
    "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:"
)
REPLACEMENT_INIT = (
    "        else:\n"
    "            self._mtp_hidden_buffer = None\n"
    + INIT_BLOCK
    + "\n"
    "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:"
)

ANCHOR_FORWARD = (
    "                residual,\n"
    "            )\n"
    "            if idx + 1 in self.aux_hidden_state_layers:"
)
REPLACEMENT_FORWARD = (
    "                residual,\n"
    "            )\n"
    + FORWARD_BLOCK
    + "            if idx + 1 in self.aux_hidden_state_layers:"
)

PATCHES = (
    ("import os", ANCHOR_IMPORT, REPLACEMENT_IMPORT),
    ("module steering block", ANCHOR_MODULE, REPLACEMENT_MODULE),
    ("__init__/_load_steering", ANCHOR_INIT, REPLACEMENT_INIT),
    ("forward apply", ANCHOR_FORWARD, REPLACEMENT_FORWARD),
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
        status_src = P.read_text() if P.is_file() else ""
        print(
            "steering (projective cvec)          :",
            "APPLIED" if MARK in status_src else "NOT APPLIED",
            "| WEIGHTLESS_STEER_PATH",
            "set" if steer_requested() else "unset",
        )
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        return check_vector()

    src = P.read_text()
    if MARK in src:
        print(f"[steering-hotfix] already applied to {P}")
        return check_vector() if steer_requested() else 0

    missing = [name for name, old, _ in PATCHES if old not in src]
    if missing:
        msg = f"[steering-hotfix] anchors not found: {missing}; refusing to patch"
        if steer_requested():
            print(msg + " (WEIGHTLESS_STEER_PATH is set; failing closed)", file=sys.stderr)
            return 1
        print(msg + " (steering off; leaving model.py stock)")
        return 0

    for name, old, new in PATCHES:
        assert src.count(old) == 1, f"anchor {name!r} not unique"
        src = src.replace(old, new, 1)
    P.write_text(src)
    print(f"[steering-hotfix] applied to {P} ({len(PATCHES)} anchors)")
    return check_vector() if steer_requested() else 0


if __name__ == "__main__":
    raise SystemExit(main())
