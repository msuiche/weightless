#!/usr/bin/env python3
"""Hotfix: projective activation steering for Qwen3.5/3.8 on vLLM 0.27.

Same intervention as the DSV4 steering patch (spec in
dspark-deploy/spec/CONTROL-VECTOR.md), ported to the Qwen3-Next-family model
stack that Qwen3.8-27B loads as in the eugr/drowzeys GB10 images
(``eugr/spark-vllm-b12x``, vLLM 0.27):

    h <- h - alpha * (h . d_hat) d_hat

on the post-layer residual stream, layers/alpha/vector gated by env:

    QWEN_STEER_PATH    .gguf (spec-conformant cvec) or .pt {layer: tensor}
    QWEN_STEER_ALPHA   float, default 1.0; the shipping Qwen vector is
                       calibrated AT alpha 1.0 (unlike the DSV4 lane's 4.0)
    QWEN_STEER_LAYERS  optional comma list restricting layer ids

Architecture differences from the DSV4 hotfix, and why:

- Qwen3_5Model inherits its forward from Qwen3NextModel, so this patches
  ``qwen3_next.py`` (one file, one class) rather than ``qwen3_5.py``. The
  machinery is inert unless QWEN_STEER_PATH is set, so Qwen3-Next models
  sharing the file are unaffected.
- The (hidden_states, residual) pair here is vLLM's decomposed convention:
  the true residual stream after a layer is ``hidden_states + residual`` (the
  add is fused into the next layernorm). The derivation measured the FULL
  post-layer stream (HF layer outputs), so the apply steers the sum and
  writes it back into hidden_states, leaving residual untouched. Steering
  hidden_states alone would remove the component from only part of the
  stream.
- alpha defaults to 1.0: the shipping Qwen vector
  (Qwen3.8-27B-refusal-cvec-per_layer-L10-58-a1.gguf) is a re-export at the
  measured alpha, per the evals. Do not import the DSV4 lane's 4.0.

The GGUF reader and every spec check are verbatim from the DSV4 hotfix: the
container contract (dspark.mode=project, hook_point, layer-id cross-check,
direction.0 rejection) is lane-independent. eugr/drowzeys images do not ship
the ``gguf`` package either, so the embedded parser is kept.

Failure semantics (fail-closed where it matters):

- Anchors not found: exit 1 if QWEN_STEER_PATH is set (a boot that was asked
  for steering must not silently serve unsteered), exit 0 otherwise.
- QWEN_STEER_PATH set but the vector file is missing/invalid/non-project:
  exit 1, before the model load.

Patches
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py
in-place inside the container (called from the serve script before
``exec vllm serve``). Idempotent: re-applying is a no-op once the marker is
present. ``--status`` reports state; ``--check`` validates the vector named
by QWEN_STEER_PATH without touching qwen3_next.py.
"""
import os
from pathlib import Path
import sys

# Overridable for dry-runs against a copy of qwen3_next.py outside the
# container.
P = Path(os.environ.get(
    "QWEN_STEERING_MODEL_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py",
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

    `dspark.mode` is enforced, not advisory. llama.cpp ADDS a control
    vector; we PROJECT one out. The same file under the wrong operation
    produces no error, just wrong output -- an additive apply pushes every
    token along the refusal axis instead of removing the component. So an
    unrecognised mode is a hard failure rather than a fallback.
    """
    import numpy as np

    meta, tensors = _read_gguf_cvec(path)

    mode = meta.get("dspark.mode")
    if mode is None:
        raise ValueError(
            f"{path}: no dspark.mode. Refusing to guess: an additive "
            f"control vector and a projective one are different operations."
        )
    if mode != "project":
        raise ValueError(
            f"{path}: dspark.mode={mode!r}, but this runtime only "
            f"implements projective ablation (h -= alpha*(h.d)d). "
            f"Refusing to apply."
        )

    hook = meta.get("dspark.hook_point")
    if hook is not None and hook != "residual_stream_post_layer":
        raise ValueError(
            f"{path}: dspark.hook_point={hook!r} does not match this hook "
            f"(residual_stream_post_layer). The same vector at the wrong "
            f"hook point measured ~9x weaker; refusing to apply."
        )

    logger.info(
        "DSpark GGUF control vector: mode=%s spec_version=%s base_model=%s rev=%s",
        mode,
        meta.get("dspark.spec_version", "?"),
        meta.get("general.base_model.0.name") or meta.get("dspark.base_model"),
        str(meta.get("general.base_model.0.version")
            or meta.get("dspark.base_revision") or "?")[:12],
    )

    out = {}
    for name, arr in tensors.items():
        dot = name.find(".")
        if dot < 0 or name[:dot] != "direction":
            continue
        try:
            idx = int(name[dot + 1:])
        except ValueError:
            continue
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
    declared = meta.get("dspark.layer_ids_zero_based")
    if declared:
        try:
            want = sorted(int(x) for x in declared.split(",") if x.strip())
        except ValueError:
            want = None
        if want and want != sorted(out):
            raise ValueError(
                f"{path}: dspark.layer_ids_zero_based declares layers "
                f"{want[0]}..{want[-1]} ({len(want)} entries) but the "
                f"direction tensors resolve to {sorted(out)[0]}.."
                f"{sorted(out)[-1]} ({len(out)} entries). The tensor names "
                f"are what get applied, so this file would steer the wrong "
                f"layers. Re-export it."
            )
    logger.info(
        "DSpark GGUF control vector: %d directions, n_embd=%d, layers %s",
        len(out), next(iter(widths)), sorted(out),
    )
    return out
'''

# Module-level block injected after the logger instantiation.
MODULE_BLOCK = (
    "\n"
    "# ---------------------------------------------------------------------------\n"
    "# Projective activation steering (Qwen3.5/3.8). [steering-hotfix]\n"
    "#\n"
    "# h <- h - alpha * (h . d_hat) d_hat on the residual stream at chosen layers.\n"
    "# Same intervention as the DSV4 lane; see dspark-deploy/spec/CONTROL-VECTOR.md.\n"
    "#\n"
    "# Everything here is inert unless QWEN_STEER_PATH is set.\n"
    "# ---------------------------------------------------------------------------\n"
    + MARK
    + "\n"
    "\n"
    "# Recorded for provenance in the startup log. Only \"post_layer\" is implemented\n"
    "# here, which is the shipped setting and the one every measurement was taken at.\n"
    "_QWEN_STEER_HOOK = (os.environ.get(\"QWEN_STEER_HOOK\") or \"post_layer\").strip()\n"
    "# layer id -> (k, hidden) orthonormal rows. Populated for inspection and for\n"
    "# offline tooling; NOT read by the forward path.\n"
    "_QWEN_HOOK_DIRS: dict[int, torch.Tensor] = {}\n"
    "\n"
    + GGUF_SRC
)

# __init__ tail block + _load_steering method, injected between the
# aux_hidden_state_layers assignment and embed_input_ids.
INIT_BLOCK = '''\

        # ---- projective activation steering ---------------------------------
        self._steer_alpha_val = float(os.environ.get("QWEN_STEER_ALPHA", "1.0") or 1.0)
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

    def _load_steering(self, config, device, dtype) -> None:
        """Fill _steer_stack from QWEN_STEER_PATH. No-op when unset.

        Loaded on every rank and indexed by GLOBAL layer id, so this is
        correct under pipeline parallelism: each rank's forward loop only
        visits its own layers and looks them up by the same global index.
        """
        path = os.environ.get("QWEN_STEER_PATH", "").strip()
        if not path:
            return
        try:
            if path.endswith(".gguf"):
                raw = _load_gguf_control_vector(path)
            else:
                raw = torch.load(path, map_location="cpu")
            want = os.environ.get("QWEN_STEER_LAYERS", "").strip()
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
                # Rank-k: orthonormalise the basis, otherwise overlapping
                # components get subtracted more than once.
                q, _ = torch.linalg.qr(vec.T)
                vec = q.T[: vec.shape[0]]
                # Both of these are per layer and must stay inside this loop.
                # A previous revision (on the DSV4 lane) dedented them out,
                # which kept only the final iteration and silently steered
                # one layer instead of the full set.
                self._steer_dirs[layer_id] = vec.to(device=device, dtype=dtype)
                _QWEN_HOOK_DIRS[layer_id] = vec.to(
                    device=device, dtype=torch.bfloat16
                )

            if not self._steer_dirs:
                logger.warning(
                    "QWEN_STEER_PATH=%s matched no layers; serving unsteered", path
                )
                return

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
                "Qwen refusal steering active: hook=%s alpha=%.3f rank=%s "
                "layers=%d %s",
                _QWEN_STEER_HOOK,
                self._steer_alpha_val,
                {k_: int(v.shape[0]) for k_, v in sorted(self._steer_dirs.items())},
                len(self._steer_dirs),
                sorted(self._steer_dirs),
            )
        except Exception as exc:
            logger.error("Qwen steering load failed (%s); serving unsteered", exc)
            self._steer_dirs = {}
'''

# Per-layer apply in the forward loop.
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

ANCHOR_IMPORT = (
    "import torch\n"
    "from torch import nn"
)
REPLACEMENT_IMPORT = (
    "import os\n"
    "import torch\n"
    "from torch import nn"
)

ANCHOR_MODULE = (
    "logger = init_logger(__name__)\n"
)
REPLACEMENT_MODULE = ANCHOR_MODULE + MODULE_BLOCK

ANCHOR_INIT = (
    "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    "\n"
    "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:"
)
REPLACEMENT_INIT = (
    "        self.aux_hidden_state_layers: tuple[int, ...] = ()\n"
    + INIT_BLOCK
    + "\n"
    "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:"
)

ANCHOR_FORWARD = (
    "            hidden_states, residual = layer(\n"
    "                positions=positions,\n"
    "                hidden_states=hidden_states,\n"
    "                residual=residual,\n"
    "            )\n"
    "            if (layer_idx + 1) in self.aux_hidden_state_layers"
)
REPLACEMENT_FORWARD = (
    "            hidden_states, residual = layer(\n"
    "                positions=positions,\n"
    "                hidden_states=hidden_states,\n"
    "                residual=residual,\n"
    "            )\n"
    + FORWARD_BLOCK
    + "            if (layer_idx + 1) in self.aux_hidden_state_layers"
)

PATCHES = (
    ("import os", ANCHOR_IMPORT, REPLACEMENT_IMPORT),
    ("module steering block", ANCHOR_MODULE, REPLACEMENT_MODULE),
    ("__init__/_load_steering", ANCHOR_INIT, REPLACEMENT_INIT),
    ("forward apply", ANCHOR_FORWARD, REPLACEMENT_FORWARD),
)


def steer_requested() -> bool:
    return bool(os.environ.get("QWEN_STEER_PATH", "").strip())


def check_vector() -> int:
    """Validate the vector named by QWEN_STEER_PATH. 0 ok, 1 bad."""
    path = os.environ.get("QWEN_STEER_PATH", "").strip()
    if not path:
        print("[steering-hotfix] --check: QWEN_STEER_PATH unset; nothing to check")
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
        want = os.environ.get("QWEN_STEER_LAYERS", "").strip()
        if want:
            [int(t) for t in want.replace(" ", "").split(",") if t]
        float(os.environ.get("QWEN_STEER_ALPHA", "1.0") or 1.0)
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
            "| QWEN_STEER_PATH",
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
            print(msg + " (QWEN_STEER_PATH is set; failing closed)", file=sys.stderr)
            return 1
        print(msg + " (steering off; leaving qwen3_next.py stock)")
        return 0

    for name, old, new in PATCHES:
        assert src.count(old) == 1, f"anchor {name!r} not unique"
        src = src.replace(old, new, 1)
    P.write_text(src)
    print(f"[steering-hotfix] applied to {P} ({len(PATCHES)} anchors)")
    return check_vector() if steer_requested() else 0


if __name__ == "__main__":
    raise SystemExit(main())
