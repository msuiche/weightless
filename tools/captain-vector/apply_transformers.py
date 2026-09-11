#!/usr/bin/env python3
"""apply_transformers -- project GLP control vectors onto any HF decoder model.

Experiment-time use: hook the residual stream of a transformers model and
apply  h <- h - sum_j alpha_j*(h.d_j)d_j  at exactly the layers a GLP GGUF
lists -- a single direction per layer (rank 1) or an orthonormal subspace
basis (rank k, directions direction.N plus direction.N.j). The production
serving path stays the patches/ hotfixes; this module exists for
measuring what a vector does on a checkpoint you have loaded in transformers
anyway.

    from apply_transformers import glp_steered
    with glp_steered(model, "v.gguf"):
        model.generate(**enc)

The GGUF is read with captain_vector's STDLIB reader -- no gguf package, and
the bytes are parsed exactly once, here, so this file and the validator can
never disagree about the format. torch is required (import guard fails loud);
transformers is NOT imported by this module -- anything exposing the standard
decoder-layer attribute paths works.
"""
from __future__ import annotations

import os
import sys

try:
    import torch
except ImportError as e:
    raise ImportError(
        "apply_transformers needs torch; the stdlib commands "
        "(validate/inspect/export) do not -- run captain_vector.py directly") from e

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import captain_vector as _cv  # the stdlib GGUF reader lives there


def _load(gguf_path):
    """Parse once: (metadata dict, {layer: (k, n_embd) F32 tensor}).

    direction.N is direction 0 of layer N; direction.N.j (j >= 1) stacks the
    rest of a rank-k subspace basis on top. The indices must run 0..k-1 per
    layer -- a gap is a partial subspace, and applying that steers a
    different subspace than the file describes.
    """
    meta, tensors = _cv._read_gguf(gguf_path)
    grouped = {}
    for name, dims, dtype, raw in tensors:
        pj = _cv._parse_direction_name(name)
        if pj is None:
            continue
        vals = _cv._tensor_floats(dims, dtype, raw)
        if vals is None or dtype != 0 or len(dims) != 1:
            raise ValueError(f"{name}: GLP directions are F32 1-D; got dtype "
                             f"{dtype}, shape {dims} -- refusing to reinterpret")
        grouped.setdefault(pj[0], {})[pj[1]] = torch.tensor(
            vals, dtype=torch.float32)
    if not grouped:
        raise ValueError("not a GLP control vector: no direction.N tensors")
    dirs = {}
    for L, js in grouped.items():
        if sorted(js) != list(range(len(js))):
            raise ValueError(f"layer {L}: direction indices {sorted(js)} are "
                             "not 0..k-1 -- refusing to apply a partial "
                             "subspace")
        dirs[L] = torch.stack([js[j] for j in sorted(js)])
    return meta, dirs


def load_directions(gguf_path):
    """Read a rank-1 GLP GGUF's directions as {layer: torch.Tensor} (F32, 1-D).

    Layer ids are zero-based, per glp.layer_ids_zero_based. Unit norm is the
    writer's contract and is not re-checked here -- run the `validate` command
    for that; anything not F32 1-D is refused rather than reinterpreted.
    Rank-k (subspace) files are refused: returning only direction 0 would
    silently drop the rest of the basis. Use load_glp for those.
    """
    meta, dirs = _load(gguf_path)
    if any(d.shape[0] != 1 for d in dirs.values()):
        raise ValueError("rank-k (subspace) vector: load_directions returns "
                         "single directions only -- use load_glp")
    return {L: d[0] for L, d in dirs.items()}


def load_glp(gguf_path):
    """Read a GLP GGUF fully: (metadata, {layer: (k, n_embd) tensor}, alphas).

    alphas is {layer: [a_0..a_{k-1}]}, the effective per-direction strength
    at each layer (glp.alpha_default x glp.dir_scales x glp.layer_scales).
    Rank-1 files come back with k=1 -- the pre-subspace format is the k=1
    special case, not a separate path.
    """
    meta, dirs = _load(gguf_path)
    alphas, _ = _cv.glp_alphas(meta, sorted(dirs))
    return meta, dirs, alphas


def _find_layers(model):
    """Locate the decoder layer list across HF architectures.

    Same candidate paths as captain_vector's Adapter, but any non-empty list
    qualifies (a 4-layer test model is legitimate). On failure, say what each
    path actually held -- "no layers" with no evidence has cost debugging time
    before.
    """
    found = []
    for path in _cv.Adapter.LAYER_PATHS:
        o = model
        try:
            for p in path:
                o = getattr(o, p)
        except AttributeError:
            continue
        n = len(o) if hasattr(o, "__len__") else None
        found.append(f"{'.'.join(path)} ({type(o).__name__}, len={n})")
        if hasattr(o, "__len__") and hasattr(o, "__getitem__") and n and n >= 1:
            return o
    detail = "; found: " + ", ".join(found) if found else \
        "; none of them exist on this object"
    raise RuntimeError(
        "could not locate the decoder layer list -- this model's architecture "
        "is not one apply_transformers knows; tried "
        + ", ".join(".".join(p) for p in _cv.Adapter.LAYER_PATHS) + detail)


def _hidden_size(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise RuntimeError("model has no .config; cannot check the hidden size")
    return int(getattr(getattr(cfg, "text_config", cfg), "hidden_size"))


def _make_hook(D, alphas):
    """h <- h - sum_j alpha_j*(h.d_j)d_j on the layer output.

    D is the layer's (k, n_embd) basis, alphas the k effective strengths.
    With an orthonormal basis (glp.orthonormal, required for rank > 1) the
    per-direction terms commute, so the plain sum IS the projection and no
    apply order exists to get wrong; at k=1 this is the original one-line
    edit h <- h - alpha*(h.d)d.

    Why the layer output IS the residual stream here: a forward hook on a
    decoder *layer* module fires after that layer's forward returns, and
    for standard HF decoder layers the returned hidden_states already has
    attention + MLP folded into the residual -- exactly the tensor the GLP
    spec calls residual_stream_post_layer. The hook's return value replaces
    the module's output, so returning the projected tensor IS the edit; no
    surgery on the model file. Archs whose layer output is not the plain
    stream (widened hyper-connection streams, exotic fused norms) would
    need their own hook-site adapter -- same lesson as the vLLM lanes.
    Output convention varies by arch (bare tensor vs tuple with
    hidden_states first), hence unwrap/rewrap.
    """
    def hook(mod, args, out):
        t, was_tuple = _cv.Adapter.unwrap(out)
        Dv = D.to(device=t.device, dtype=t.dtype)
        av = alphas.to(device=t.device, dtype=t.dtype)
        t = t - ((t @ Dv.T) * av) @ Dv
        return _cv.Adapter.rewrap(t, out, was_tuple)
    return hook


class GLPSteering:
    """A live set of steering hooks. Context manager; detach() removes them.

    A leaked hook silently contaminates every later run in the process, so the
    handle tracks whether it is attached and detach() is idempotent.
    """

    def __init__(self, model, source, alpha, dirs, handles, alphas=None):
        self.model, self.source, self.alpha = model, source, alpha
        self.layers = sorted(dirs)
        first = dirs[self.layers[0]]
        self.rank = first.shape[0] if first.dim() > 1 else 1
        self.alphas = alphas or {}
        self._handles = handles

    @property
    def attached(self):
        return bool(self._handles)

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.detach()
        return False


def _attach(model, source, meta, dirs, alpha):
    """Validate (meta, dirs) against the model and register the hooks.

    Split from attach_glp_steering so the safetensors lane (glp.py at the
    repo root) shares exactly one implementation of the checks and the
    hook registration -- the two loaders can never drift apart. dirs values
    may be a 1-D direction or a (k, n_embd) stacked basis; 1-D is treated
    as k=1. Metadata values may be strings (safetensors __metadata__ is a
    string map), so numbers are coerced. Refuses (ValueError) anything whose
    mode is not project -- an additive consumer applying a projective
    vector, or vice versa, is silently wrong, and silent is the failure
    mode this file exists to avoid. Also refuses a hook_point other than
    residual_stream_post_layer, a direction width that is not the model's
    hidden size, layer ids past the model's depth, and rank > 1 without a
    verified orthonormal basis (per-direction alphas only commute on one).
    """
    mode = meta.get("glp.mode")
    if mode != "project":
        raise ValueError(
            f"glp.mode is {mode!r}, not 'project'"
            + (" (no glp.* metadata: legacy additive llama.cpp vector -- "
               "adding it would apply the wrong intervention)" if mode is None
               else " -- refusing to apply"))
    hook = meta.get("glp.hook_point")
    if hook is not None and hook != "residual_stream_post_layer":
        raise ValueError(f"glp.hook_point is {hook!r}; this helper applies at "
                         "residual_stream_post_layer only -- refusing")

    dirs = {L: (d.unsqueeze(0) if d.dim() == 1 else d.float())
            for L, d in dirs.items()}
    ks = {d.shape[0] for d in dirs.values()}
    if len(ks) != 1:
        raise ValueError(f"direction count differs across layers: "
                         f"{sorted(ks)} -- refusing")
    k = ks.pop()
    declared = int(meta.get("glp.rank") or 1)
    if declared != k:
        raise ValueError(f"glp.rank declares {declared} but the tensors "
                         f"carry {k} direction(s) per layer -- refusing")
    if k > 1:
        if meta.get("glp.orthonormal") not in (True, 1, "true", "True", "1"):
            raise ValueError("rank-k vector without glp.orthonormal=true: "
                             "per-direction alphas only commute on an "
                             "orthonormal basis -- refusing")
        for L, d in sorted(dirs.items()):
            off = d @ d.T - torch.eye(k)
            if float(off.abs().max()) > 1e-3:
                raise ValueError(
                    f"layer {L}: basis is not orthonormal (max |G-I| "
                    f"{float(off.abs().max()):.4f}) despite glp.orthonormal "
                    "-- refusing")
    # alpha overrides the base (glp.alpha_default); per-direction and
    # per-layer scales from the metadata still apply on top of it.
    alphas, _ = _cv.glp_alphas(meta, sorted(dirs), alpha)
    alpha = float(alpha if alpha is not None
                  else meta.get("glp.alpha_default", 1.0))

    hidden = _hidden_size(model)
    for i, d in sorted(dirs.items()):
        if d.shape[-1] != hidden:
            raise ValueError(f"direction.{i} has width {d.shape[-1]}, model "
                             f"hidden size is {hidden} -- this vector was not "
                             "derived for this checkpoint; refusing")
    layers = _find_layers(model)
    deep = max(dirs)
    if deep >= len(layers):
        raise ValueError(f"direction.{deep} but the model has {len(layers)} "
                         "decoder layers -- refusing")

    # register_forward_hook fires after each listed layer's forward, on every
    # call -- prefill and every decode step of generate(), KV cache or not, so
    # every token that passes through a steered layer gets projected. This is
    # the easy lane precisely because transformers runs eager: no CUDA graphs
    # or torch.compile capture, so none of the vLLM hotfix discipline (dense
    # zero-padded stacks, tensor alpha buffers, unconditional apply inside the
    # traced region) is needed here.
    handles = [
        layers[i].register_forward_hook(
            _make_hook(dirs[i], torch.tensor(alphas[i], dtype=torch.float32)))
        for i in sorted(dirs)]
    return GLPSteering(model, source, alpha, dirs, handles, alphas=alphas)


def attach_glp_steering(model, gguf_path, alpha=None):
    """Register projection hooks on the decoder layers a GLP GGUF lists.

    alpha defaults to glp.alpha_default. See _attach for what is refused
    and why.
    """
    meta, dirs = _load(gguf_path)
    return _attach(model, gguf_path, meta, dirs, alpha)


def glp_steered(model, gguf_path, alpha=None):
    """Context-manager form: `with glp_steered(model, path): ...`.

    Hooks come off on exit, exception or not. Identical to attach_glp_steering;
    the handle also works without `with` (call .detach() yourself).
    """
    return attach_glp_steering(model, gguf_path, alpha)
