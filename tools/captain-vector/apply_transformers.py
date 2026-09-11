#!/usr/bin/env python3
"""apply_transformers -- project GLP control vectors onto any HF decoder model.

Experiment-time use: hook the residual stream of a transformers model and
apply  h <- h - alpha*(h.d)d  at exactly the layers a GLP GGUF lists. The
production serving path stays the patches/ hotfixes; this module exists for
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
import re
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
    """Parse once: (metadata dict, {layer: F32 tensor})."""
    meta, tensors = _cv._read_gguf(gguf_path)
    dirs = {}
    for name, dims, dtype, raw in tensors:
        m = re.fullmatch(r"direction\.(\d+)", name)
        if not m:
            continue
        vals = _cv._tensor_floats(dims, dtype, raw)
        if vals is None or dtype != 0 or len(dims) != 1:
            raise ValueError(f"{name}: GLP directions are F32 1-D; got dtype "
                             f"{dtype}, shape {dims} -- refusing to reinterpret")
        dirs[int(m.group(1))] = torch.tensor(vals, dtype=torch.float32)
    if not dirs:
        raise ValueError("not a GLP control vector: no direction.N tensors")
    return meta, dirs


def load_directions(gguf_path):
    """Read a GLP GGUF's direction tensors as {layer: torch.Tensor} (F32, 1-D).

    Layer ids are zero-based, per glp.layer_ids_zero_based. Unit norm is the
    writer's contract and is not re-checked here -- run the `validate` command
    for that; anything not F32 1-D is refused rather than reinterpreted.
    """
    return _load(gguf_path)[1]


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


def _make_hook(d, alpha):
    """h <- h - alpha*(h.d)d on the layer output, tuple or bare tensor.

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
        dv = d.to(device=t.device, dtype=t.dtype)
        t = t - alpha * (t @ dv).unsqueeze(-1) * dv
        return _cv.Adapter.rewrap(t, out, was_tuple)
    return hook


class GLPSteering:
    """A live set of steering hooks. Context manager; detach() removes them.

    A leaked hook silently contaminates every later run in the process, so the
    handle tracks whether it is attached and detach() is idempotent.
    """

    def __init__(self, model, gguf_path, alpha, dirs, handles):
        self.model, self.gguf_path, self.alpha = model, gguf_path, alpha
        self.layers = sorted(dirs)
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


def attach_glp_steering(model, gguf_path, alpha=None):
    """Register projection hooks on the decoder layers a GLP GGUF lists.

    alpha defaults to glp.alpha_default. Refuses (ValueError) anything whose
    mode is not project -- an additive consumer applying a projective vector,
    or vice versa, is silently wrong, and silent is the failure mode this file
    exists to avoid. Also refuses a hook_point other than
    residual_stream_post_layer, a direction width that is not the model's
    hidden size, and layer ids past the model's depth.
    """
    meta, dirs = _load(gguf_path)
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
    if alpha is None:
        alpha = float(meta.get("glp.alpha_default", 1.0))

    hidden = _hidden_size(model)
    for i, d in sorted(dirs.items()):
        if d.numel() != hidden:
            raise ValueError(f"direction.{i} has width {d.numel()}, model "
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
    handles = [layers[i].register_forward_hook(_make_hook(dirs[i], alpha))
               for i in sorted(dirs)]
    return GLPSteering(model, gguf_path, alpha, dirs, handles)


def glp_steered(model, gguf_path, alpha=None):
    """Context-manager form: `with glp_steered(model, path): ...`.

    Hooks come off on exit, exception or not. Identical to attach_glp_steering;
    the handle also works without `with` (call .detach() yourself).
    """
    return attach_glp_steering(model, gguf_path, alpha)
