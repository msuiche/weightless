"""GLM-5.3-Flash (``Glm5NextForConditionalGeneration``): the widened mHC
stream that each layer's ``layer_communicator.mhc.mlp_combine`` returns.

The decoder layer returns ``(hidden, residual, topk_indices)`` with
residual None: the mHC stream [T, hc_mult x hidden] is carried whole in
hidden. Each layer's communicator ends with ``mhc.mlp_combine`` (hc_post),
which materialises that stream; on the last layer hc_contract follows it
inside the layer. So the site is mlp_combine's output on every steered
layer, the last one included (``glm_mhc_combine`` below). One class for the
DSA and the KDA layers; ``is_linear_attn`` tells them apart. TP > 1 is
refused. Same stream as vllm-plugin archs/glm5next.py.
"""
from __future__ import annotations

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
)

ROWS = {
    "Glm5NextForConditionalGeneration": ArchRow(backbone=("model",), **_GLM5_NEXT),
}


@register_special("glm_mhc_combine", site="mhc.mlp_combine")
def glm_mhc_combine(layer, site, *, layer_id, backbone, row):
    """GLM-5.3-Flash (Glm5NextDecoderLayer): steer the mHC stream that
    ``layer.layer_communicator.mhc.mlp_combine`` returns.

    ``mlp_combine`` is hc_post: it mixes the layer's FFN output into the
    hc_mult residual streams and returns them flat, stream-major, as
    [T, hc_mult x hidden]. That is the post-layer stream the GLP file was
    derived on, and the next layer reads it (its hc_pre) with no fusion
    across the layer boundary. On the last layer the communicator applies
    hc_contract AFTER mlp_combine, so the last layer is steered on the wide
    stream too, before the contract.

    ``MHCState`` is a plain dataclass made once per communicator, so the
    wrapper is an instance attribute that stays for every forward and is
    recorded into every captured graph. The edit has no residual (r=None).
    """
    comm = getattr(layer, "layer_communicator", None)
    mhc = getattr(comm, "mhc", None)
    where = f"weightless: GLM layer {layer_id}"
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
    if getattr(layer, "is_nextn", False):
        raise RuntimeError(f"{where} is a NEXTN (MTP) layer; the draft stays stock. Refusing.")
    own_id = getattr(layer, "layer_id", layer_id)
    if int(own_id) != int(layer_id):
        raise RuntimeError(f"{where}: the module at index {layer_id} says layer_id={own_id}; "
                           f"the file's absolute layer ids would land on the wrong layer. "
                           f"Failing closed.")
    width = int(row.width(backbone.config))
    hc = int(getattr(mhc, "hc_mult", 0) or 0)
    hidden = int(getattr(layer, "hidden_size", 0) or backbone.config.hidden_size)
    if hc * hidden != width:
        raise RuntimeError(
            f"{where}: mhc.hc_mult {hc} x hidden {hidden} = {hc * hidden}, but the row's "
            f"stream width is {width}. Failing closed."
        )

    orig = mhc.mlp_combine

    def mlp_combine(*args, **kwargs):
        return site(orig(*args, **kwargs), None)

    mlp_combine._weightless_layer_id = int(layer_id)
    mhc.mlp_combine = mlp_combine

    def undo():
        if vars(mhc).get("mlp_combine") is mlp_combine:
            del mhc.mlp_combine

    return undo
