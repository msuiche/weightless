#!/usr/bin/env python3
"""Hotfix: projective activation steering for Qwen3.8-Flash-Next (NVFP4)
on the day-0 image ``vllm/vllm-openai:qwen38-flash-next``.

Same intervention as the DSV4/Qwen steering patches (spec in
weightless/spec/GLP.md):

    h <- h - alpha * (h . d_hat) d_hat

on the post-layer residual stream, layers/alpha/vector gated by env:

    WEIGHTLESS_STEER_PATH    .gguf (spec-conformant cvec) or .pt {layer: tensor}
    WEIGHTLESS_STEER_ALPHA   float, default 1.0; the shipping GLP-47 vector is
                       calibrated AT alpha 1.0 (do NOT import the DSV4 lane's 4.0)
    WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids

This arch is NOT the decomposed (hidden_states, residual) convention of the
Qwen3-Next stack — it is qwen3_8_flash_next's delayed-combine
hyper-connection scheme (anchors validated in
refusal-research/experiments/20260826-flash-next-vllm-capture against this
exact image; that experiment's patch also carries a capture probe, which a
serving lane does not want — this hotfix is the steering half only):

- Between layers the stream is [T, hc_count*hidden] = [T, 10240], HC-outer /
  H-inner. GLP-47's directions live in exactly this space (the vLLM capture
  lane reproduced the HF-derived direction at cos 0.9931).
- Decoder layers return (hidden_states, block_output, injection) with the
  layer's MLP output still PENDING. The post-layer stream is the
  parameter-free mlp_hyper_connection.combine() of the three, so the apply
  materializes per layer, steers the materialized stream, and continues with
  it (pending combine consumed): the next layer takes its mix() path —
  mathematically the same input, one kernel less fused. The final mixer's
  combine_and_mix is guarded for the same reason (mix() when the pending
  combine was already consumed).
- The apply is UNCONDITIONAL per layer with a dense zero-padded stack (zero
  rows are a numeric no-op): the traced graph stays identical for every
  layer set, which vLLM's compile cache does not key on — the
  change-the-graph-between-layer-sets trap bit the Qwen3.8 lane.

ONE file is patched, and unlike the Qwen3.8 lane there is no
__init__-override trap: Qwen3_8FlashNextModel is a direct nn.Module whose
own __init__ runs (the serving class Qwen3_8FlashNextForCausalLM delegates
to it), so buffers registered there exist on the class whose forward reads
them.

Other notes:

- Buffers are allocated on CPU and ride the model's post-init move to the
  accelerator (this module imports no current_platform; the capture lane
  validated the CPU-alloc pattern on this image).
- ``enable_prefix_caching`` MUST stay off with this arch: it forces
  mamba_cache_mode="align", which splits every prefill at a block boundary
  and corrupts capture/steering (experiment run 3). The recipe sets
  --no-enable-prefix-caching.
- Two-node lane: the start script runs this hotfix in BOTH containers (each
  rank patches its own in-image copy) before ``vllm serve``.
- The GGUF reader and every spec check are verbatim from the other lanes'
  hotfixes: the container contract (glp.mode=project, hook_point, layer-id
  cross-check, direction.0 rejection) is lane-independent.

Failure semantics (fail-closed where it matters):

- Anchors not found: exit 1 if WEIGHTLESS_STEER_PATH is set (a boot that was
  asked for steering must not silently serve unsteered), exit 0 otherwise.
- WEIGHTLESS_STEER_PATH set but the vector file is missing/invalid/non-project:
  exit 1, before the model load.

Patches
/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/model.py
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
    "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/model.py",
))
MARK = "# [steering-hotfix] projective activation steering (Qwen3.8-Flash-Next)"

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
    declared = meta.get("glp.layer_ids_zero_based")
    if declared:
        try:
            want = sorted(int(x) for x in declared.split(",") if x.strip())
        except ValueError:
            want = None
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
    "# Projective activation steering (Qwen3.8-Flash-Next). [steering-hotfix]\n"
    "#\n"
    "# h <- h - alpha * (h . d_hat) d_hat on the materialized post-layer\n"
    "# multi-stream [T, hc_count*hidden] at chosen layers. Same intervention\n"
    "# as the DSV4/Qwen lanes; see weightless/spec/GLP.md.\n"
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

# _load_steering method, injected into Qwen3_8FlashNextModel. Verbatim from
# the Qwen3.8 lane except the stream width: here a layer's stream is
# hc_count*hidden (10240), not hidden_size.
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

            if not self._steer_dirs:
                logger.warning(
                    "WEIGHTLESS_STEER_PATH=%s matched no layers; serving unsteered", path
                )
                return

            k = max(v.shape[0] for v in self._steer_dirs.values())
            stack = torch.zeros(
                config.num_hidden_layers, k, config.hidden_size * config.hc_count,
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
            logger.error("Qwen3.8-Flash-Next steering load failed (%s); serving unsteered", exc)
            self._steer_dirs = {}
'''

# Buffer registration, injected into Qwen3_8FlashNextModel.__init__ (the one
# and only __init__ on the serving path — Qwen3_8FlashNextForCausalLM
# delegates, it does not skip the parent's init; no Qwen3_5Model-style trap
# here).
INIT_BLOCK = '''\

        # ---- projective activation steering ---------------------------------
        self._steer_alpha_val = float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
        self._steer_dirs: dict[int, torch.Tensor] = {}
        # Buffers are allocated on CPU (device=None): this module imports no
        # current_platform, and registered buffers ride the model's post-init
        # move to the accelerator — the pattern the capture lane validated on
        # this image.
        _dev = None
        _dtype = vllm_config.model_config.dtype
        # Allocated unconditionally, zeros when steering is off. A
        # None-when-disabled branch changes the traced graph, and that
        # difference is not part of vLLM's compile cache key: a compiled
        # artifact from one layer set was reused by another and died with
        # KeyError. A dense stack indexed by layer id keeps the graph
        # identical for every layer set, so only values change. The stream
        # here is hc_count*hidden wide (hyper-connection multi-stream).
        self.register_buffer(
            "_steer_stack",
            torch.zeros(
                config.num_hidden_layers, 1, config.hidden_size * config.hc_count,
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

# Per-layer apply at the end of the Qwen3_8FlashNextModel.forward loop.
FORWARD_BLOCK = '''\
            # ---- projective activation steering [steering-hotfix] -----------
            # Unconditional per-layer projection on the MATERIALIZED
            # post-layer multi-stream [T, hc_count*hidden]: the delayed-combine
            # scheme returns this layer's MLP output still pending in
            # (block_output, injection), and the parameter-free
            # mlp_hyper_connection.combine() of the three is the post-layer
            # residual stream the GLP directions were derived on (vLLM capture
            # reproduced the HF direction at cos 0.9931). Zero rows in
            # _steer_stack are a numeric no-op, so the traced graph is
            # identical for every layer set. Steered layers continue
            # materialized (pending combine consumed): the next layer takes
            # its mix() path -- same math, one kernel less fused. The final
            # mixer below is guarded for the same reason.
            if block_output is None:
                # deepstack already materialized the state (multimodal only)
                steer_stream = hidden_states
            else:
                steer_stream = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
            steer_dirs = self._steer_stack[layer_idx]
            steer_coef = torch.einsum("...h,kh->...k", steer_stream, steer_dirs)
            hidden_states = steer_stream - self._steer_alpha * torch.einsum(
                "...k,kh->...h", steer_coef, steer_dirs
            )
'''

# Final-mixer guard: when the (unconditional) apply consumed the last layer's
# pending combine, mix() is the same input path for an already-materialized
# state (multi_hidden is the materialized multi-stream itself).
MIXER_BLOCK = '''\
        if block_output is not None:
            multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        else:
            # Steering consumed the last layer's pending combine; for an
            # already-materialized state, mix() is the same input path
            # (multi_hidden is the materialized multi-stream itself).
            multi_hidden, sample_hidden_states, _ = final_mixer.mix(
                hidden_states
            )
'''

# --- anchors (validated against the day-0 image's file; see
# refusal-research/experiments/20260826-flash-next-vllm-capture) -------------
ANCHOR_IMPORT = (
    "from vllm.distributed import get_pp_group\n"
)
REPLACEMENT_IMPORT = (
    "import os\n"
    "\n"
    "from vllm.distributed import get_pp_group\n"
    "from vllm.logger import init_logger\n"
    "\n"
    "logger = init_logger(__name__)\n"
    + MODULE_BLOCK
)

ANCHOR_INIT = (
    "        else:\n"
    "            self._mtp_hidden_buffer = None\n"
)
REPLACEMENT_INIT = ANCHOR_INIT + INIT_BLOCK + LOAD_METHOD_BLOCK

ANCHOR_FORWARD = (
    "                hidden_states = hidden_states + deepstack_embed\n"
)
REPLACEMENT_FORWARD = ANCHOR_FORWARD + FORWARD_BLOCK

ANCHOR_MIXER = (
    "        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(\n"
    "            hidden_states, block_output, injection\n"
    "        )\n"
)
REPLACEMENT_MIXER = MIXER_BLOCK

PATCHES = (
    ("imports + module steering block", ANCHOR_IMPORT, REPLACEMENT_IMPORT),
    ("__init__ buffers + _load_steering", ANCHOR_INIT, REPLACEMENT_INIT),
    ("forward apply", ANCHOR_FORWARD, REPLACEMENT_FORWARD),
    ("final mixer guard", ANCHOR_MIXER, REPLACEMENT_MIXER),
)

FILE_PATCHES = (
    ("qwen3_8_flash_next/nvidia/model.py", P, PATCHES),
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
