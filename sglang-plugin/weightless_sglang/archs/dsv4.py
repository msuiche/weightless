"""DeepSeek-V4-Flash (``DeepseekV4ForCausalLM`` with model_type deepseek_v4):
steering at the FFN write, before the mHC fold (glp.hook_point =
ffn_out_pre_residual).

This is the site the vLLM plugin steers (vllm-plugin archs/dsv4.py) and the
site the published GLP-29 file was measured at. In SGLang's fused mode
(``layer.use_fused_mhc_post_pre``, on by default where the TileLang or
aiter mHC kernels are there) each ``DeepseekV4DecoderLayer`` returns

    (ffn_out [T, hidden], residual [T, hc_mult, hidden], post, comb)

and folds nothing itself: the next layer runs hc_post(ffn_out, residual,
post, comb) at its start, and the model loop runs it for the last layer
after the loop. So ``output[0]`` is the pending FFN write, and the edit is
``x' = x - alpha (x.d) d`` on it with no residual (r=None); the other three
slots pass through untouched. Every consumer of the layer's output (the
next layer, the last-layer hc_post, the DSpark aux capture and the
pipeline hand-over) reads the loop variable after the hook, so the last
layer is steered too.

The site is linear in x, so a TP partial sum would still be edited exactly
(the edit of a sum is the sum of the edits). SGLang reduces the MoE output
inside the layer anyway (``_run_moe_ffn_dp_sync``); the shared site
refuses the partial-sum marker and a deferred MoE handoff all the same.

What this row refuses, before anything is hooked (``dsv4_preflight``):
- any model_type other than exactly deepseek_v4 (on the hf config and the
  backbone config, at least one of which must carry it): the row was built
  from that model's layer loop and tuple, and a model type that reuses the
  class could change either, so it fails closed instead of guessing.
- DeepSeek-V4.1: SGLang serves it through this class (it rewrites the
  architecture name) with model_type deepseek_v41 and
  ``hc_pre_from_prev_sublayer``. That loop calls
  ``forward_hc_pre_from_prev`` directly (no hook fires) and precomputes the
  next layer's input from the unsteered stream.
- a residual-site file (glp.hook_point = residual_stream_post_layer): the
  GLP-42 residual vector, and the plain Vision-Exp GLP-29 file, which holds
  the same tensors as its ``-ffn`` file under a residual label. In fused mode
  no post-layer stream exists between layers, and the last layer's fold
  runs in the model loop (a last-layer trap). The vLLM plugin refuses these
  files too.
- the unfused mode (``use_fused_mhc_post_pre`` false, for example
  SGLANG_OPT_FUSE_MHC_POST_PRE=0 or no TileLang/aiter mHC kernels): the layer
  then folds its FFN write itself and returns the stream, so the site does
  not exist at the layer boundary.
"""
from __future__ import annotations

from .base import ArchRow, hidden_size, register_preflight, register_special

HOOK = "ffn_out_pre_residual"
MODEL_TYPE = "deepseek_v4"
RESIDUAL_HOOK = "residual_stream_post_layer"
LAYER_CLASS = "DeepseekV4DecoderLayer"

# In SGLang's fused mHC mode the layer returns (ffn_out, residual, post,
# comb) and the next layer (or the model loop, after the last layer) folds
# ffn_out into the streams. The site is ffn_out, edited with no residual. The
# site is linear, so TP is allowed. The tuple fields below (arity,
# hidden_index, residual_index) are declarative only: the special:dsv4_ffn
# handler owns the tuple, checks the 4-tuple on every call and edits slot 0
# itself; the row test pins the values so the row and the handler stay in
# step.
_DSV4 = dict(
    layers=frozenset({"DeepseekV4DecoderLayer"}),
    hooks=frozenset({"ffn_out_pre_residual"}),
    width=hidden_size,
    arity=4, hidden_index=0, residual_index=None,
    install="special:dsv4_ffn", exec_id="layer", tp="ok",
    hint=frozenset({"deepseek_v4"}),
)

ROWS = {
    "DeepseekV4ForCausalLM": ArchRow(backbone=("model",), **_DSV4),
}

_UNFUSED = (
    "SGLang runs DeepSeek-V4 in its unfused mHC mode here (use_fused_mhc_post_pre is "
    "false: SGLANG_OPT_FUSE_MHC_POST_PRE is off, or the TileLang/aiter mHC kernels are "
    "not available). In that mode the layer folds its FFN write into the streams itself "
    "and returns the stream, so the ffn_out_pre_residual site does not exist at the layer "
    "boundary. Only the fused mode is served. Refusing."
)


def _types(runner, backbone):
    out = []
    for cfg in (getattr(getattr(runner, "model_config", None), "hf_config", None),
                getattr(backbone, "config", None)):
        if cfg is not None:
            out.append(cfg)
    return out


def dsv41_reason(runner, backbone):
    """Why this model is DeepSeek-V4.1 served through the V4 class, or None."""
    for cfg in _types(runner, backbone):
        mt = str(getattr(cfg, "model_type", "") or "")
        if mt.startswith("deepseek_v41"):
            return f"model_type={mt!r}"
        if getattr(cfg, "hc_pre_from_prev_sublayer", False):
            return "hc_pre_from_prev_sublayer=True"
    if getattr(backbone, "hc_pre_from_prev_sublayer", False):
        return "the backbone has hc_pre_from_prev_sublayer=True"
    return None


def model_type_reason(runner, backbone):
    """Why this model is not model_type deepseek_v4 exactly, or None."""
    seen = []
    for cfg in _types(runner, backbone):
        mt = getattr(cfg, "model_type", None)
        if mt is not None and str(mt) != "":
            seen.append(str(mt))
    if not seen:
        return "no model_type on the hf config or the backbone config"
    other = sorted({m for m in seen if m != MODEL_TYPE})
    if other:
        return "model_type " + ", ".join(repr(m) for m in other)
    return None


def residual_file_reason(meta, path):
    """The named refusal of a residual-site file on this row."""
    base = str(meta.get("glp.base_model") or "")
    if "vision-exp" in base.lower():
        what = (f"{path} is the plain DeepSeek-V4-Flash-Vision-Exp GLP-29 file: the same "
                f"tensors as its -ffn file (glp.content_sha256 "
                f"{str(meta.get('glp.content_sha256') or '?')[:16]}...) under a "
                f"residual_stream_post_layer label. Serve the -ffn file, whose "
                f"glp.hook_point is ffn_out_pre_residual.")
    else:
        what = (f"{path} is a DeepSeek-V4 residual-site file (glp.hook_point="
                f"residual_stream_post_layer, for example the GLP-42 residual vector).")
    return (
        f"weightless: {what} The DeepseekV4ForCausalLM row steers the FFN write before "
        f"the mHC fold (ffn_out_pre_residual) only: in SGLang's fused mode no post-layer "
        f"stream exists between layers, and the last layer's fold runs in the model loop "
        f"after the last layer (a last-layer trap). The vLLM plugin refuses these files "
        f"too. Refusing."
    )


@register_preflight("dsv4_ffn")
def dsv4_preflight(*, runner, backbone, meta, path, row, row_name):
    """Row checks that run before anything is hooked."""
    why = dsv41_reason(runner, backbone)
    if why is not None:
        raise RuntimeError(
            f"weightless: this {row_name} is DeepSeek-V4.1 ({why}). SGLang serves V4.1 "
            f"through the V4 class, but its layer loop calls forward_hc_pre_from_prev "
            f"directly (no forward hook runs there) and precomputes the next layer's input "
            f"from the unsteered stream. DeepSeek-V4.1 is not supported by this plugin (nor "
            f"by the vLLM plugin). Refusing."
        )
    why = model_type_reason(runner, backbone)
    if why is not None:
        raise RuntimeError(
            f"weightless: the {row_name} row serves model_type {MODEL_TYPE!r} only, and this "
            f"model has {why}. A model type that reuses the class may change the layer loop "
            f"or the layer's return tuple, so the row does not guess. Refusing."
        )
    hint = str(meta.get("controlvector.model_hint") or "")
    if hint.startswith("deepseek_v41"):
        raise RuntimeError(
            f"weightless: {path} is a DeepSeek-V4.1 file (model_hint={hint!r}); "
            f"DeepSeek-V4.1 is not supported by this plugin. Refusing."
        )
    hook = meta.get("glp.hook_point")
    if hook == RESIDUAL_HOOK and (not hint or hint in row.hint):
        # a DeepSeek-V4 residual file gets its named reason; another model's
        # file falls through to the hook-point and model_hint refusals
        raise RuntimeError(residual_file_reason(meta, path))
    if not getattr(backbone, "use_fused_mhc_post_pre", False):
        raise RuntimeError(f"weightless: {row_name} backbone: {_UNFUSED}")
    start, end = int(backbone.start_layer), int(backbone.end_layer)
    unfused = [i for i in range(start, end)
               if type(backbone.layers[i]).__name__ == LAYER_CLASS
               and not getattr(backbone.layers[i], "use_fused_mhc_post_pre", False)]
    if unfused:
        raise RuntimeError(f"weightless: {row_name} layers {unfused[:8]}: {_UNFUSED}")


class Dsv4SiteError(RuntimeError):
    pass


@register_special("dsv4_ffn", site="ffn_out (layer output[0], pre-fold)")
def dsv4_ffn(layer, site, *, layer_id, backbone, row):
    """Steer ``output[0]`` of one ``DeepseekV4DecoderLayer`` in fused mode.

    A forward hook on the layer INSTANCE (the loop calls ``layer(...)``).
    It checks the fused-mode 4-tuple on every call: an output whose deferred
    slots are None is the unfused return (the stream, already folded), and
    fails closed rather than being edited at the wrong site.
    """
    where = f"weightless: DeepSeek-V4 layer {layer_id}"
    if getattr(layer, "hc_pre_from_prev_sublayer", False):
        raise RuntimeError(f"{where} has hc_pre_from_prev_sublayer (DeepSeek-V4.1); "
                           f"not supported. Refusing.")
    if not getattr(layer, "use_fused_mhc_post_pre", False):
        raise RuntimeError(f"{where}: {_UNFUSED}")
    own_id = getattr(layer, "layer_id", layer_id)
    if int(own_id) != int(layer_id):
        raise RuntimeError(f"{where}: the module at index {layer_id} says layer_id={own_id}; "
                           f"the file's absolute layer ids would land on the wrong layer. "
                           f"Failing closed.")
    if getattr(layer, "_weightless_dsv4_hook", None) is not None:
        raise RuntimeError(f"{where} is already steered; refusing to steer twice. "
                           f"Failing closed.")
    width = int(row.width(backbone.config))
    hidden = int(getattr(layer, "hidden_size", width))
    if hidden != width:
        raise RuntimeError(f"{where}: layer hidden_size {hidden} != row width {width}. "
                           f"Failing closed.")

    def hook(module, args, output):
        if not isinstance(output, tuple) or len(output) != 4:
            n = f" of {len(output)}" if isinstance(output, tuple) else ""
            raise Dsv4SiteError(
                f"{where} returned {type(output).__name__}{n}, expected the fused-mode "
                f"4-tuple (ffn_out, residual, post, comb); refusing")
        x, residual, post, comb = output
        if residual is None or post is None or comb is None:
            raise Dsv4SiteError(
                f"{where} returned the unfused form (stream, None, None, None): the FFN "
                f"write is already folded, so the ffn_out_pre_residual site is gone. "
                f"Failing closed.")
        if getattr(x, "dim", lambda: 0)() != 2:
            raise Dsv4SiteError(
                f"{where}: output[0] has shape {tuple(getattr(x, 'shape', ()))}, expected "
                f"the FFN write [T, {width}]; refusing")
        return (site(x, None), residual, post, comb)

    handle = layer.register_forward_hook(hook)
    layer._weightless_dsv4_hook = handle

    def undo():
        handle.remove()
        if getattr(layer, "_weightless_dsv4_hook", None) is handle:
            del layer._weightless_dsv4_hook

    return undo
