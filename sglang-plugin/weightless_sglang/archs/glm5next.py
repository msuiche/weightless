"""GLM-5.3-Flash (``Glm5NextForConditionalGeneration``): the widened mHC
stream after each layer's feed-forward post-combine (hc_post).

mHC keeps the residual as hc_mult streams. A layer reads its attention input
from them (hc_pre), writes the attention output back in (hc_post), reads the
FFN input (hc_pre), and writes the FFN output back in with hc_post: the
post-combine. That last hc_post returns the layer's post-layer stream,
[T, hc_mult x hidden] flat and stream-major, the stream the GLP file was
derived on; the next layer reads it with no fusion across the layer
boundary. On the last layer hc_contract follows it inside the layer, so the
last layer is steered on the wide stream too, before the contract. The edit
has no residual (r=None). One class for the DSA and the KDA layers;
``is_linear_attn`` tells them apart. TP > 1 is refused. Same stream as
vllm-plugin archs/glm5next.py.

Two SGLang layouts, told apart by what each layer holds:

- ``layer.layer_communicator`` is an ``MHCLayerCommunicator`` (SGLang before
  upstream #41554 / #41750, 2026-09-28/29; the serving build b62df575): the
  FFN post-combine is ``layer_communicator.mhc.mlp_combine``, which
  postprocess_layer calls once per layer before the last layer's contract.
  The site wraps that method on the MHCState instance.
- ``layer.attn_boundary`` / ``layer.ffn_boundary`` (stage boundaries, #41554
  and later): mHC lives in ``layer_boundary/residual/mhc.py``. The FFN
  boundary's declared update is the layer's ``_FfnUpdate``, which the FFN exit
  calls once per layer: ``MHCState.apply_post`` (the post-combine), then
  ``clear_coefficients`` and, on the last layer, ``hc_contract``. The
  attention's ``_AttnUpdate`` calls the same ``apply_post`` and must stay
  unsteered. So the site wraps the FFN update instance and steers the output
  of the ``apply_post`` call made inside it, exactly once per call (any other
  count fails closed); ``apply_post`` called from anywhere else passes
  through untouched.

Anything else fails closed at install.
"""
from __future__ import annotations

from ..core import SiteError, stage_boundaries
from .base import HOOK_POINT, ArchRow, register_special


def mhc_width(cfg):
    """Width of a widened mHC stream: hidden_size x hc_mult (the stream the
    GLP file was derived on). A config without mHC has no such stream."""
    if not getattr(cfg, "mhc", False):
        raise RuntimeError(
            "weightless: this model config has mhc off, so there is no widened mHC stream; "
            "the row steers that stream only. Refusing."
        )
    return int(cfg.hidden_size) * int(cfg.hc_mult)


def _attn_kind(layer):
    return "linear" if getattr(layer, "is_linear_attn", False) else "full"


_GLM5_NEXT = dict(
    layers=frozenset({"Glm5NextDecoderLayer"}),
    hooks=frozenset({HOOK_POINT}),
    width=mhc_width,
    arity=3, hidden_index=0, residual_index=None,
    install="special:glm_mhc_combine", exec_id="layer", tp="refuse",
    hint=frozenset({"glm5_next"}),
    kind=_attn_kind,
    boundary=True,
)

ROWS = {
    "Glm5NextForConditionalGeneration": ArchRow(backbone=("model",), **_GLM5_NEXT),
}


# The site text of each layout (the log line and the manifest).
SITE_COMMUNICATOR = "mhc.mlp_combine"
SITE_BOUNDARY = "ffn_boundary.update.apply_post"


@register_special("glm_mhc_combine", site=SITE_COMMUNICATOR, boundary_site=SITE_BOUNDARY)
def glm_mhc_combine(layer, site, *, layer_id, backbone, row):
    """GLM-5.3-Flash (Glm5NextDecoderLayer): steer the mHC stream that the
    layer's feed-forward post-combine returns, on either SGLang layout (see
    the module text). The wrappers are instance attributes that stay for
    every forward and are recorded into every captured graph."""
    where = f"weightless: GLM layer {layer_id}"
    if getattr(layer, "is_nextn", False):
        raise RuntimeError(f"{where} is a NEXTN (MTP) layer; the draft stays stock. Refusing.")
    own_id = getattr(layer, "layer_id", layer_id)
    if int(own_id) != int(layer_id):
        raise RuntimeError(f"{where}: the module at index {layer_id} says layer_id={own_id}; "
                           f"the file's absolute layer ids would land on the wrong layer. "
                           f"Failing closed.")
    width = int(row.width(backbone.config))
    hidden = int(getattr(layer, "hidden_size", 0) or backbone.config.hidden_size)
    if stage_boundaries(layer) is not None:
        return _install_boundary(layer, site, where=where, width=width, hidden=hidden,
                                 layer_id=layer_id)
    if getattr(layer, "layer_communicator", None) is not None:
        return _install_communicator(layer, site, where=where, width=width, hidden=hidden,
                                     layer_id=layer_id)
    raise RuntimeError(
        f"{where} ({type(layer).__name__}) has neither layer_communicator (an "
        f"MHCLayerCommunicator) nor attn_boundary/ffn_boundary (SGLang's stage boundaries); "
        f"this SGLang layout is not one the GLM row knows. Failing closed.")


def _check_hc(where, hc, hidden, width):
    if hc * hidden != width:
        raise RuntimeError(
            f"{where}: mhc.hc_mult {hc} x hidden {hidden} = {hc * hidden}, but the row's "
            f"stream width is {width}. Failing closed."
        )


def _install_communicator(layer, site, *, where, width, hidden, layer_id):
    """The older layout: wrap ``layer_communicator.mhc.mlp_combine``.

    ``mlp_combine`` is hc_post: it mixes the layer's FFN output into the
    hc_mult residual streams and returns them flat, stream-major, as
    [T, hc_mult x hidden]. The communicator applies hc_contract AFTER it on the
    last layer. ``MHCState`` is a plain dataclass made once per communicator.
    """
    comm = getattr(layer, "layer_communicator", None)
    mhc = getattr(comm, "mhc", None)
    if comm is None or type(comm).__name__ != "MHCLayerCommunicator":
        raise RuntimeError(
            f"{where} has communicator {type(comm).__name__}, not MHCLayerCommunicator; "
            f"its post-layer stream is not the mHC stream this row steers. Failing closed."
        )
    if mhc is None or not callable(getattr(mhc, "mlp_combine", None)):
        raise RuntimeError(f"{where}: layer_communicator.mhc.mlp_combine is missing; "
                           f"the SGLang mHC layout changed. Failing closed.")
    if not hasattr(mhc, "__dict__"):
        raise RuntimeError(f"{where}: {type(mhc).__name__} takes no instance attributes, so "
                           f"mlp_combine cannot be wrapped. Failing closed.")
    if "mlp_combine" in vars(mhc):
        raise RuntimeError(f"{where}: mhc.mlp_combine is already wrapped on this instance; "
                           f"refusing to steer twice. Failing closed.")
    _check_hc(where, int(getattr(mhc, "hc_mult", 0) or 0), hidden, width)

    orig = mhc.mlp_combine

    def mlp_combine(*args, **kwargs):
        return site(orig(*args, **kwargs), None)

    mlp_combine._weightless_layer_id = int(layer_id)
    mhc.mlp_combine = mlp_combine

    def undo():
        if vars(mhc).get("mlp_combine") is mlp_combine:
            del mhc.mlp_combine

    return undo


def _install_boundary(layer, site, *, where, width, hidden, layer_id):
    """The stage-boundary layout: steer the ``apply_post`` output inside the
    FFN boundary's declared update (``_FfnUpdate.update``), before its
    clear_coefficients and the last layer's hc_contract.

    The FFN update object is the one the FFN boundary's plan calls (the plan
    holds the same object as the declaration), so the wrapper on its instance
    reaches every call. ``apply_post`` is wrapped on the layer's MHCState
    instance (a dataclass, one per layer) and steers only while this layer's
    FFN update runs; the attention update's call passes through.
    """
    attn, ffn = stage_boundaries(layer)
    upd = ffn.declaration.update
    state = getattr(upd, "state", None)
    if type(upd).__name__ != "_FfnUpdate" or getattr(upd, "applied_at_exit", None) is not True:
        raise RuntimeError(
            f"{where}: the FFN boundary's update is {type(upd).__name__} "
            f"(applied_at_exit={getattr(upd, 'applied_at_exit', None)}), not mHC's _FfnUpdate, "
            f"which writes the FFN output into the streams at the layer's exit; the post-layer "
            f"stream is not the one this row steers. Failing closed.")
    if (type(state).__name__ != "MHCState" or not callable(getattr(state, "apply_post", None))
            or not hasattr(state, "is_last_layer") or not callable(getattr(upd, "update", None))):
        raise RuntimeError(f"{where}: the FFN update's state is {type(state).__name__}, not an "
                           f"MHCState with apply_post; the SGLang mHC layout changed. Failing closed.")
    if getattr(attn.declaration.update, "state", None) is not state:
        raise RuntimeError(f"{where}: the attention and FFN updates do not share one MHCState; "
                           f"the SGLang mHC layout changed. Failing closed.")
    if not hasattr(state, "__dict__") or not hasattr(upd, "__dict__"):
        raise RuntimeError(f"{where}: the mHC state or update takes no instance attributes, so "
                           f"they cannot be wrapped. Failing closed.")
    if "apply_post" in vars(state) or "update" in vars(upd):
        raise RuntimeError(f"{where}: the FFN update or apply_post is already wrapped on this "
                           f"instance; refusing to steer twice. Failing closed.")
    _check_hc(where, int(getattr(state, "hc_mult", 0) or 0), hidden, width)

    orig_apply = state.apply_post  # the bound methods, called through the wrappers
    orig_update = upd.update
    inside = [0]  # 0: not in this layer's FFN update; n > 0: n - 1 apply_post calls so far

    def apply_post(hidden_states, residual):
        out = orig_apply(hidden_states, residual)
        if inside[0]:
            inside[0] += 1
            out = site(out, None)
        return out

    def update(hidden_states, residual):
        inside[0] = 1
        try:
            out = orig_update(hidden_states, residual)
            calls = inside[0] - 1
        finally:
            inside[0] = 0
        if calls != 1:
            raise SiteError(
                f"{where}: the FFN update called apply_post {calls} times in one call, not once; "
                f"the post-combine this row steers is not where it was. Failing closed.")
        return out

    apply_post._weightless_layer_id = int(layer_id)
    update._weightless_layer_id = int(layer_id)
    state.apply_post = apply_post
    upd.update = update

    def undo():
        if vars(upd).get("update") is update:
            del upd.update
        if vars(state).get("apply_post") is apply_post:
            del state.apply_post

    return undo
