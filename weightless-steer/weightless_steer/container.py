"""GLP GGUF container reader — the spec gates as a real module, one copy.

This is the hotfix fleet's ``GGUF_SRC`` constant promoted to an importable
module (docs/vllm-plugin-design.md: "container.py — read_gguf_cvec(),
load_control_vector() — GGUF_SRC as a real module, one copy, tested"). The
parse is deliberately dependency-light: serving images may not ship the
``gguf`` package, so the container is read directly with struct + numpy.

Container and tensor convention follow llama.cpp: tensors named
``direction.<N>``, fp32, 1-D, N >= 1, and **N is the layer index**. Layer 0
cannot be expressed in this container. A rank-k (subspace) file stacks the
rest of the basis as ``direction.<N>.<j>`` (j = 1..k-1); such a layer comes
back as a (k, n_embd) tensor, a single direction as 1-D. Rank > 1 requires
``glp.spec_version = 2`` and a verified orthonormal basis — the
per-direction alphas in ``h -= sum_j alpha_j (h . d_j) d_j`` only commute
on one (spec/GLP.md, *Rank-k*).

``glp.mode`` is enforced, not advisory. llama.cpp ADDS a control vector; we
PROJECT one out. The same file under the wrong operation produces no error,
just wrong output — an additive apply pushes every token along the refusal
axis instead of removing the component. So an unrecognised mode is a hard
failure rather than a fallback.

``glp.hook_point`` is enforced the same way: each caller steers exactly one
site and passes it in. A vector calibrated for a different site fails closed
instead of editing the wrong stream.
"""
from __future__ import annotations

import logging
import re
import struct

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FM = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
       7: "<B", 10: "<Q", 11: "<q", 12: "<d"}

_DIRECTION_RE = re.compile(r"direction\.(\d+)(?:\.(\d+))?")


def _int_meta(meta: dict, key: str, default: int) -> int:
    """An integer metadata value, coerced. Fixture files carry strings
    ("1.0"); real GGUF writers emit proper uint32. Both are fine."""
    v = meta.get(key)
    if v is None or v == "":
        return default
    return int(float(str(v)))


def read_gguf_cvec(path: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Minimal GGUF v3 reader for control vectors (F32 tensors only).

    Returns (metadata dict, {tensor_name: F32 numpy array}). Only what the
    control-vector format needs: metadata scalars/strings and direction.<N>
    tensor payloads.
    """
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


def load_control_vector(
    path: str,
    hook: str = "residual_stream_post_layer",
) -> tuple[dict, dict[int, torch.Tensor]]:
    """Load a projective GLP control vector into (metadata, {layer: tensor}).

    Enforces the reader-conformance gates of spec/GLP.md that this runtime
    can check: mode is present and "project", hook_point matches `hook`
    exactly, direction.0 is rejected, tensor names resolve to integer layer
    ids, widths are uniform, glp.spec_version is one this reader implements,
    and glp.layer_ids_zero_based — when present — agrees with the tensor
    names. A rank-k (subspace) file additionally must carry directions
    0..k-1 at every layer, a matching glp.rank, and a verified orthonormal
    basis; its layers come back as (k, n_embd) tensors. A transferred
    vector (derived_at != hook_point) is legal but logged loudly;
    alpha_default belongs to the apply site.
    """
    meta, tensors = read_gguf_cvec(path)

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

    file_hook = meta.get("glp.hook_point")
    if file_hook != hook:
        raise ValueError(
            f"{path}: glp.hook_point={file_hook!r} does not match this hook "
            f"({hook}). Refusing to apply at the wrong site."
        )

    # The version gate: rank > 1 and the alpha-scaling keys change what a
    # reader computes, so they are only valid under spec_version 2. A
    # version-1 reader would ignore the keys it doesn't know and apply the
    # file with the wrong alphas — silently. Refuse instead.
    version = _int_meta(meta, "glp.spec_version", 1)
    if version not in (1, 2):
        raise ValueError(
            f"{path}: glp.spec_version={meta.get('glp.spec_version')!r} is "
            f"not implemented by this reader (1 and 2 are). Refusing."
        )
    rank_declared = _int_meta(meta, "glp.rank", 1)
    if version == 1 and (rank_declared != 1
                         or "glp.dir_scales" in meta
                         or "glp.layer_scales" in meta):
        raise ValueError(
            f"{path}: glp.rank={rank_declared} / alpha-scaling keys require "
            f"glp.spec_version 2 (this file declares 1). Refusing."
        )

    # A transferred vector (captured at one site, calibrated for another) is
    # legal but must not be silent: derived_at != hook_point means the
    # direction was estimated on a different distribution than the one it is
    # about to edit. Warning, never a refusal (GLP.md); alpha_default belongs
    # to the apply site.
    derived_at = meta.get("glp.derived_at")
    if derived_at and derived_at != file_hook:
        logger.warning(
            "%s: transferred vector -- glp.derived_at=%r, apply hook %r",
            path, derived_at, file_hook,
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

    grouped = {}
    for name, arr in tensors.items():
        m = _DIRECTION_RE.fullmatch(name)
        if m is None:
            if name.startswith("direction."):
                raise ValueError(
                    f"{path}: malformed tensor name {name!r} — 'direction.' "
                    f"must be followed by an integer layer id"
                )
            continue
        idx, j = int(m.group(1)), int(m.group(2) or 0)
        if idx < 1:
            raise ValueError(
                f"{path}: {name} is invalid; direction.0 is rejected "
                f"upstream and layer 0 cannot be expressed in this "
                f"container"
            )
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        grouped.setdefault(idx, {})[j] = torch.from_numpy(arr.copy())
    if not grouped:
        raise ValueError(f"{path}: no direction.<N> tensors found")

    # A layer's directions must run 0..k-1: a gap is a partial subspace, and
    # applying that steers a different subspace than the file describes.
    out = {}
    for idx, js in grouped.items():
        if sorted(js) != list(range(len(js))):
            raise ValueError(
                f"{path}: layer {idx}: direction indices {sorted(js)} are "
                f"not 0..k-1 — refusing to apply a partial subspace"
            )
        stack = torch.stack([js[j] for j in sorted(js)])
        out[idx] = stack[0] if len(js) == 1 else stack  # 1-D for rank 1
    widths = {v.shape[-1] for v in out.values()}
    if len(widths) != 1:
        raise ValueError(
            f"{path}: inconsistent n_embd across directions: {widths}"
        )

    ks = {(v.shape[0] if v.dim() > 1 else 1) for v in out.values()}
    if len(ks) != 1:
        raise ValueError(
            f"{path}: direction count differs across layers: {sorted(ks)}"
        )
    k = ks.pop()
    if k != rank_declared:
        raise ValueError(
            f"{path}: glp.rank declares {rank_declared} but the tensors "
            f"carry {k} direction(s) per layer. Refusing."
        )
    if k > 1:
        # Per-direction alphas only commute on an orthonormal basis; the
        # producer orthogonalises at write time, and this is where the claim
        # is checked. |G - I| catches non-unit rows as well as cross terms.
        if str(meta.get("glp.orthonormal")).lower() not in ("true", "1"):
            raise ValueError(
                f"{path}: rank-{k} vector without glp.orthonormal=true — "
                f"per-direction alphas only commute on an orthonormal "
                f"basis. Refusing."
            )
        for idx, v in sorted(out.items()):
            off = float((v @ v.T - torch.eye(k)).abs().max())
            if off > 1e-3:
                raise ValueError(
                    f"{path}: layer {idx}: basis is not orthonormal "
                    f"(max |G-I| {off:.4f}) despite glp.orthonormal. "
                    f"Refusing."
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
        "weightless GLP vector: %d steered layers, n_embd=%d, rank=%d, "
        "layers %s",
        len(out), next(iter(widths)), k, sorted(out),
    )
    return meta, out
