#!/usr/bin/env python3
"""Hotfix: projective activation steering (GLP) for Nemotron-H models in vLLM.

Applies

    h <- h - alpha * (h . d_hat) d_hat

at the post-layer residual stream of every steered decoder layer of
vLLM's nemotron_h model (NemotronHModel.forward). nemotron_h uses vLLM's
fused add+norm convention: each decoder layer returns (mixer_output,
residual) where the mixer output is NOT yet folded in (the fold happens in
the NEXT layer's norm, or norm_f at the end). So the post-layer residual
stream at the model loop is

    h = hidden_states + residual

and steering writes back ``hidden_states <- h' - residual`` so the next
fold reproduces h'. derived_at == hook_point == residual_stream_post_layer.
Validated on Modal (vLLM v0.27.1, H100, greedy, 2026-09-05): alpha=1.0 with
the GLP-51 L1-51 vector gives refusal32 0->32/32, cyber32 1->31/32,
benign32 32/32 unchanged, propaganda32 28->32/32, termination intact.

Port of the Modal reference
(refusal-research/experiments/20260905-nemotron35-glp/staging/
patch_nemotron35.py, which reads .pt) to the weightless GGUF reader
discipline of hotfix-dsv4-steering-projective.py: fail-closed glp.*
metadata gates, embedded minimal GGUF v3 parser (the container may not
ship the gguf package), dense zero-padded _steer_stack indexed by GLOBAL
layer id (correct under PP; zero rows are a numeric no-op so the traced
graph is identical for every layer set), alpha as a registered tensor
buffer (a Python float would be baked into the torch.compile cache).

Vector format (GGUF, glp.* namespace):

    glp.mode                  "project" (missing or anything else is FATAL —
                              an additive control vector under a projective
                              apply silently produces wrong output)
    glp.hook_point            must be "residual_stream_post_layer"
                              (mismatch is FATAL: this patch steers exactly
                              one site)
    glp.derived_at            if present and != hook_point, warn loudly
                              (transferred vector) but apply
    glp.layer_ids_zero_based  informational; cross-checked against the
                              direction.<N> tensor names (N IS the layer
                              index — see the layer-map note in the loader)
    direction.<N>             F32 1-D tensors, one per steered layer;
                              layer 0 is intentionally absent from the
                              published GLP-51 vector

Env vars:

    WEIGHTLESS_STEER_PATH    .gguf control vector (or .pt {layer: tensor})
    WEIGHTLESS_STEER_ALPHA   float, default 1.0 (the vector's alpha_default)
    WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids
    WEIGHTLESS_STEER_HOOK    if set, must be residual_stream_post_layer

MTP caveat: the checkpoint carries 1 MTP nextn layer which is NOT steered.
The recipe must serve WITHOUT speculative decoding (the launcher enforces
SPECULATIVE_MODE=none when a GLP vector is configured).

Failure semantics (fail-closed where it matters):

- Anchors not found: exit 1 if WEIGHTLESS_STEER_PATH is set (a boot that
  was asked for steering must not silently serve unsteered), exit 0
  otherwise (stock behaviour).
- WEIGHTLESS_STEER_PATH set but the vector file is missing/invalid/
  non-project/wrong hook: exit 1, before the multi-minute model load.
- Runtime load failures with steering armed (direction width != hidden,
  no layers matched, direction layers out of range) re-raise: the engine
  boot dies rather than serving unsteered.

Patches
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/nemotron_h.py
in-place inside the container (the recipe runs it from the container
entrypoint before ``exec vllm serve``; the ``&&`` makes boot fail-closed).
Anchors verified unique against the v0.27.1 reference (extracted at
refusal-research/experiments/20260905-nemotron35-glp/staging/srcdl/
vllm_model_executor_models_nemotron_h.py) and against the
vllm/vllm-openai:v0.28.0 image deployed on the DGX Spark (vendored at
weightless/patches/reference/nemotron_h_v0280.py). Idempotent: re-applying
is a no-op once the marker is present. ``--status`` reports state;
``--check`` validates the vector named by WEIGHTLESS_STEER_PATH without
touching nemotron_h.py.
"""
import os
from pathlib import Path
import sys

# Overridable for dry-runs against a copy of nemotron_h.py outside the
# container.
P = Path(os.environ.get(
    "WEIGHTLESS_STEERING_MODEL_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/"
    "nemotron_h.py",
))
MARK = "# [steering-hotfix] projective activation steering (nemotron_h residual)"

# ---------------------------------------------------------------------------
# Injected source: minimal GGUF reader + spec-conformant cvec loader.
# Kept as one constant so the hotfix's own --check mode executes the exact
# code that gets injected into nemotron_h.py.
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
    Layer 0 cannot be expressed in this container and is intentionally
    absent from the published GLP-51 vector.

    `glp.mode` is enforced, not advisory. llama.cpp ADDS a control
    vector; we PROJECT one out. The same file under the wrong operation
    produces no error, just wrong output — an additive apply pushes every
    token along the refusal axis instead of removing the component. So an
    unrecognised mode is a hard failure rather than a fallback.

    `glp.hook_point` is enforced the same way: this patch steers exactly
    one site (the post-layer residual stream, h = hidden_states + residual
    under nemotron_h's fused add+norm convention). A vector calibrated for
    a different site fails closed instead of editing the wrong stream.
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
    if hook != "residual_stream_post_layer":
        raise ValueError(
            f"{path}: glp.hook_point={hook!r} does not match this hook "
            f"(residual_stream_post_layer — h = hidden_states + residual "
            f"at the nemotron_h model loop, mixer output not yet folded). "
            f"Refusing to apply at the wrong site."
        )

    # A transferred vector (captured at one site, calibrated for another) is
    # legal but must not be silent: derived_at != hook_point means the
    # direction was estimated on a different distribution than the one it is
    # about to edit. Warning, never a refusal (GLP.md); alpha_default belongs
    # to the apply site.
    derived_at = meta.get("glp.derived_at")
    if derived_at and derived_at != hook:
        logger.warning(
            "%s: transferred vector -- glp.derived_at=%r, apply hook %r",
            path, derived_at, hook,
        )

    logger.info(
        "weightless GLP vector: mode=%s spec_version=%s base_model=%s rev=%s "
        "alpha_default=%s",
        mode,
        meta.get("glp.spec_version", "?"),
        meta.get("general.base_model.0.name") or meta.get("glp.base_model"),
        str(meta.get("general.base_model.0.version")
            or meta.get("glp.base_revision") or "?")[:12],
        meta.get("glp.alpha_default", "?"),
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
    # file was produced by a broken exporter. The names are what execute;
    # a one-layer shift degrades rather than fails (adjacent-layer refusal
    # directions are highly correlated), so this check is load-bearing.
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

# Module-level block injected after the NemotronHConfig import.
MODULE_BLOCK = (
    "\n"
    "from vllm.logger import init_logger as _weightless_init_logger\n"
    "\n"
    "logger = _weightless_init_logger(__name__)\n"
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Projective activation steering (GLP) on the post-layer residual stream.\n"
    "#\n"
    "# h <- h - alpha * (h . d_hat) d_hat, where h = hidden_states + residual\n"
    "# at the NemotronHModel layer loop (nemotron_h's fused add+norm returns\n"
    "# (mixer_output, residual) with the fold deferred to the next norm).\n"
    "# Everything here is inert unless WEIGHTLESS_STEER_PATH is set.\n"
    "# ---------------------------------------------------------------------------\n"
    + MARK
    + "\n"
    "\n"
    "# The only site implemented at this anchor. WEIGHTLESS_STEER_HOOK set to\n"
    "# anything else fails closed in _load_steering.\n"
    "_WEIGHTLESS_STEER_HOOK = (os.environ.get(\"WEIGHTLESS_STEER_HOOK\") or \"residual_stream_post_layer\").strip()\n"
    "\n"
    + GGUF_SRC
)

# __init__ tail block + _load_steering method, injected between the
# norm_f construction and embed_input_ids in NemotronHModel.
INIT_BLOCK = '''\

        # ---- projective activation steering (GLP) --------------------------
        # Dense zero-padded stack indexed by GLOBAL layer id (zero rows are
        # a numeric no-op) and alpha as a tensor buffer: torch.compile bakes
        # Python scalars and None-when-disabled branches into the cached
        # graph, and that cache does not key on these env vars. Both were
        # measured failure modes on the DSV4 lane.
        self._steer_alpha_val = float(
            os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
        _steer_dtype = vllm_config.model_config.dtype
        self.register_buffer(
            "_steer_stack",
            torch.zeros(config.num_hidden_layers, 1, config.hidden_size,
                        dtype=_steer_dtype),
            persistent=False,
        )
        self.register_buffer(
            "_steer_alpha",
            torch.zeros((), dtype=_steer_dtype),
            persistent=False,
        )
        # Buffers are non-persistent so they never enter the state dict,
        # which would make load_weights report them as unexpected keys.
        self._load_steering(config)

    def _load_steering(self, config) -> None:
        """Fill _steer_stack from WEIGHTLESS_STEER_PATH. No-op when unset.

        Loaded on every rank and indexed by GLOBAL layer id, so this is
        correct under pipeline parallelism: each rank's forward loop only
        visits its own layers and looks them up by the same global index.
        """
        path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
        if not path:
            return
        if _WEIGHTLESS_STEER_HOOK != "residual_stream_post_layer":
            raise RuntimeError(
                f"WEIGHTLESS_STEER_HOOK={_WEIGHTLESS_STEER_HOOK!r} is not "
                f"implemented at this anchor; the only site here is "
                f"residual_stream_post_layer (h = hidden_states + residual "
                f"at the model loop). Refusing to serve with a silently "
                f"wrong hook site."
            )
        try:
            if path.endswith(".gguf"):
                raw = _load_gguf_control_vector(path)
            else:
                try:
                    raw = torch.load(path, map_location="cpu",
                                     weights_only=True)
                except Exception:
                    raw = torch.load(path, map_location="cpu",
                                     weights_only=False)
                if isinstance(raw, dict) and isinstance(
                        raw.get("per_layer"), dict):
                    raw = raw["per_layer"]
            want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
            selected = (
                {int(t) for t in want.replace(" ", "").split(",") if t}
                if want else None
            )

            dirs = {}
            for layer_id, vec in raw.items():
                layer_id = int(layer_id)
                if selected is not None and layer_id not in selected:
                    continue
                vec = vec.detach().to(torch.float32).reshape(-1)
                # Width guard: a direction that does not match this arch's
                # stream width must fail here, not as an opaque broadcast
                # error at serve time.
                if vec.numel() != config.hidden_size:
                    raise RuntimeError(
                        f"steering vector layer {layer_id} width "
                        f"{vec.numel()} != {config.hidden_size} "
                        f"(hidden_size; plain single stream)"
                    )
                # The published vector ships unit directions; normalise
                # anyway so a non-unit export cannot silently scale alpha.
                dirs[layer_id] = vec / (vec.norm() + 1e-9)

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
            if not dirs:
                raise RuntimeError(
                    f"WEIGHTLESS_STEER_PATH={path} matched no layers; "
                    f"refusing to run unsteered"
                )

            for layer_id, vec in dirs.items():
                self._steer_stack[layer_id, 0] = vec.to(_steer_dtype)
            self._steer_alpha.fill_(self._steer_alpha_val)
            logger.info(
                "weightless GLP steering active: hook=%s alpha=%.3f "
                "layers=%d..%d (%d) width=%d",
                _WEIGHTLESS_STEER_HOOK,
                self._steer_alpha_val,
                min(dirs), max(dirs), len(dirs),
                config.hidden_size,
            )
        except Exception as exc:
            # Fail closed: a boot asked for steering must not serve
            # unsteered.
            logger.error("GLP steering load failed (%s); failing closed",
                         exc)
            raise
'''

# Per-layer apply in the forward loop, after layer() returns.
FORWARD_BLOCK = '''\
            # [steering-hotfix] unconditional per-layer projection:
            # h <- h - alpha (h.d) d at the post-layer residual stream.
            # nemotron_h uses vLLM's fused add+norm convention — layer()
            # returns (mixer_output, residual) with the fold deferred to
            # the next norm — so the post-layer stream is
            # h = hidden_states + residual, and steering writes back
            # hidden_states <- h' - residual so the next fold reproduces
            # h'. Stack rows are zero for layers we do not steer, making
            # this a numeric no-op there while the traced graph stays
            # identical for every layer set.
            _steer_h = hidden_states + residual
            _steer_dirs = self._steer_stack[self.start_layer + idx]
            _steer_coef = torch.einsum("...h,kh->...k", _steer_h, _steer_dirs)
            hidden_states = (_steer_h - self._steer_alpha * torch.einsum(
                "...k,kh->...h", _steer_coef, _steer_dirs)) - residual
'''

ANCHOR_IMPORT = (
    "from collections.abc import Iterable, Mapping\n"
    "from itertools import islice\n"
)
REPLACEMENT_IMPORT = (
    "import os\n"
    + ANCHOR_IMPORT
)

ANCHOR_MODULE = (
    "from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig\n"
)
REPLACEMENT_MODULE = ANCHOR_MODULE + MODULE_BLOCK

ANCHOR_INIT = (
    "        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)\n"
)
REPLACEMENT_INIT = ANCHOR_INIT + INIT_BLOCK

ANCHOR_FORWARD = (
    "        for idx, layer in enumerate(\n"
    "            islice(self.layers, self.start_layer, self.end_layer)\n"
    "        ):\n"
    "            hidden_states, residual = layer(\n"
    "                positions=positions,\n"
    "                hidden_states=hidden_states,\n"
    "                residual=residual,\n"
    "            )\n"
)
REPLACEMENT_FORWARD = ANCHOR_FORWARD + FORWARD_BLOCK

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

            # Execute the exact code that gets injected into nemotron_h.py,
            # with its two free names bound, and run the full spec
            # validation (mode, hook point, layer cross-check), not a
            # partial re-do.
            ns: dict = {"torch": torch, "logger": _StderrLogger}
            exec(GGUF_SRC, ns)  # noqa: S102 - same code that is injected
            out = ns["_load_gguf_control_vector"](path)
            layers = sorted(out)
            print(
                f"[steering-hotfix] --check: {os.path.basename(path)} OK: "
                f"mode=project hook=residual_stream_post_layer "
                f"layers {layers[0]}..{layers[-1]} ({len(layers)})"
            )
        else:
            import torch

            raw = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(raw, dict) and isinstance(raw.get("per_layer"), dict):
                raw = raw["per_layer"]
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
            "steering (projective GLP, residual_stream_post_layer):",
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
        print(msg + " (steering off; leaving nemotron_h.py stock)")
        return 0

    for name, old, new in PATCHES:
        assert src.count(old) == 1, f"anchor {name!r} not unique"
        src = src.replace(old, new, 1)
    P.write_text(src)
    print(f"[steering-hotfix] applied to {P} ({len(PATCHES)} anchors)")
    return check_vector() if steer_requested() else 0


if __name__ == "__main__":
    raise SystemExit(main())
