#!/usr/bin/env python3
"""glp -- first-class GLP steering for Hugging Face transformers models.

GLP (Guided Linear Projection) steers a model at inference by projecting
the residual stream at every listed layer:  h <- h - alpha*(h.d)d  with a
per-layer unit direction d. This module applies a GLP vector -- a GGUF v3
container with direction.<N> tensors (N >= 1, applied at layer N) and
glp.* metadata, spec: spec/GLP.md -- to any loaded transformers model:

    import glp

    with glp.apply_glp(model, "v.glp"):                 # local file
        model.generate(**enc)

    st = glp.apply_glp(model, "user/glp-vector")        # HF hub repo
    model.generate(**enc)
    st.detach()                                         # idempotent

alpha precedence: the alpha= argument, then WEIGHTLESS_STEER_ALPHA, then
the file's glp.alpha_default, then 1.0. The handle is a context manager
and detach() removes the hooks -- a leaked hook silently contaminates
every later run in the process, so prefer the `with` form.

A hub repo id may carry an explicit filename as "repo_id:path/file.glp";
without one the repo must contain exactly one *.glp / *.gguf file.

The spec gates are enforced at load, not at apply, by the one canonical
container reader (weightless-steer's container.py): glp.mode must be
present and "project" -- absent means add per spec, and an additive file
in a projective consumer fails silently, so it is fatal here;
glp.hook_point must be exactly residual_stream_post_layer, this module's
only apply site; direction.0 is rejected; and glp.layer_ids_zero_based
must agree with the tensor names. The hook registration and model-side
checks are apply_transformers._attach, so this module, the vLLM plugin,
and the experiment-time GGUF lane can never disagree about what the file
says or what the projection does.
"""
from __future__ import annotations

import importlib.util
import os

try:
    import numpy  # noqa: F401 -- container.py parses tensor payloads with it
    import torch  # noqa: F401 -- the hooks run on a live model
except ImportError as e:
    raise ImportError(
        "glp needs torch and numpy; it hooks a live transformers model -- "
        "there is no stdlib-only use of this module") from e

_HERE = os.path.dirname(os.path.abspath(__file__))

# The only site this module applies at; passed to the container reader so a
# vector calibrated anywhere else fails closed instead of editing the wrong
# stream.
HOOK_POINT = "residual_stream_post_layer"


def _load_module(name, *relpath):
    """Import a repo module by file path, without sys.path edits."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, *relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_at = _load_module("apply_transformers",
                   "tools", "captain-vector", "apply_transformers.py")
# container.py imports numpy and torch only -- no vLLM dependency comes
# along for the ride.
_container = _load_module("weightless_steer_container",
                          "weightless-steer", "weightless_steer", "container.py")


def _resolve(source):
    """Map a local path or HF hub repo id to a local GLP GGUF file.

    A hub id may carry an explicit filename as "repo_id:path/file.glp".
    Without one, the repo must contain exactly one *.glp / *.gguf file --
    picking among several silently would apply the wrong vector, and
    silent is the failure mode this project exists to avoid.
    """
    if os.path.exists(source):
        return source
    repo, _, fname = source.partition(":")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as e:
        raise ImportError(
            f"{source!r} is not a local path and huggingface_hub is not "
            "installed -- cannot resolve it as a hub repo id") from e
    if fname:
        return hf_hub_download(repo, fname)
    cands = [f for f in list_repo_files(repo)
             if f.endswith((".glp", ".gguf"))]
    if len(cands) != 1:
        raise ValueError(
            f"{repo}: expected exactly one *.glp / *.gguf file, found "
            f"{len(cands)} ({', '.join(cands) or 'none'}) -- pass "
            "'repo_id:filename' to pick one explicitly")
    return hf_hub_download(repo, cands[0])


def load_glp(source):
    """Read a GLP vector as (metadata dict, {layer: F32 tensor}).

    source is a local path or a HF hub repo id (optionally
    "repo_id:filename"). All spec/GLP.md reader gates are enforced here
    (mode, hook_point, direction.0, uniform width, layer-ids cross-check,
    spec_version, rank/orthonormal); direction.N applies at layer N, no
    offset, so the returned dict is keyed by the layer each direction
    steers. A layer with a single direction comes back 1-D; a rank-k
    subspace basis comes back (k, n_embd).
    """
    return _container.load_control_vector(_resolve(source), hook=HOOK_POINT)


def apply_glp(model, source, alpha=None):
    """Steer model with a GLP vector; returns a GLPSteering handle.

    Registers a forward hook on each decoder layer the vector lists; the
    hook applies h <- h - alpha*(h.d)d to the layer's residual-stream
    output, re-wrapping tuple outputs so HF's return convention is
    preserved. alpha precedence: this argument, then
    WEIGHTLESS_STEER_ALPHA, then glp.alpha_default, then 1.0. Use as a
    context manager (`with apply_glp(...) as st:`) or call st.detach();
    detach is idempotent. The spec gates refuse (ValueError) additive or
    modeless files, a hook_point other than residual_stream_post_layer,
    direction.0, directions whose width is not the model's hidden size,
    and layer ids past the model's depth.
    """
    if alpha is None:
        env = os.environ.get("WEIGHTLESS_STEER_ALPHA")
        if env is not None:
            alpha = float(env)
    meta, dirs = load_glp(source)
    return _at._attach(model, source, meta, dirs, alpha)


def glp_steered(model, source, alpha=None):
    """Context-manager form: `with glp_steered(model, src): ...`.

    Identical to apply_glp; mirrors apply_transformers.glp_steered for
    the experiment-time lane.
    """
    return apply_glp(model, source, alpha)
