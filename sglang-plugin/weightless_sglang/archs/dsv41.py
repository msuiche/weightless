"""DeepSeek-V4.1 (``DeepseekV4ForCausalLM`` with model_type deepseek_v41):
the post-layer hyper-connection streams, after the layer's FFN post and
before the next layer reads Engram (glp.hook_point =
residual_stream_post_layer).

SGLang serves V4.1 through the V4 class: configs/deepseek_v41.py rewrites
the architecture name DeepseekV41ForCausalLM to DeepseekV4ForCausalLM and
sets model_type deepseek_v41 and ``hc_pre_from_prev_sublayer``. So this row
is a variant of that served class, picked by the model type
(``selects``, wired in archs/__init__.py ``VARIANTS``); the DeepSeek-V4 row
keeps model_type deepseek_v4.

Where the stream is (SGLang models/deepseek_v4*.py, CUDA):

- The residual is ``R [T, hc_mult, hidden]``, hc_mult = 4 copies of the
  5120-wide stream (deepseek_v4_mhc.py, the module text). Each decoder
  layer (``DeepseekV4CudaDecoderLayer``) is built from two stage boundaries.
  Its FFN boundary declares an ``HcUpdate`` (deepseek_v4_boundary.py) whose
  ``post(output, state, *, next=None)`` folds the layer's FFN write into
  the copies, ``R'[i] = post[i] * y + sum_j comb[j][i] * R[j]``, and returns
  an ``HcState``: ``R'`` (the post-layer streams), the ``pre`` weights the
  next combine uses, and optionally the next sublayer's input and mix
  statistics, precomputed from ``R'`` by a fused kernel.
- The layer leaves that post pending. Every reader runs it through the same
  method on the same update object: the next layer's attention prepare
  (``update_and_read`` / ``_complete_read``), ``materialize`` at an Engram,
  tail or capture seam, and the ``FinalRead`` after the rank's last layer
  (the head, or the pipeline hand-over).
- Engram: the model loop materializes the pending post (that call) and only
  then calls ``engram(state.residual, ...)`` and ``with_residual`` on its
  result. So an edit made inside the post is pre-Engram by construction.

The site wraps ``post`` on each steered layer's FFN update INSTANCE. The
wrapper runs the original post, edits ``R'`` and returns
``state.with_residual(R'')``: the same HcState with the next input and
statistics dropped, because they were computed from the unedited rows. The
next combine and mix statistics then recompute from ``R''``, the path an
Engram layer always takes. ``pre`` is kept: it weights the copies, and the
model computes it from the previous sublayer's input by design (the "lag"
in deepseek_v4_mhc.py), steered or not.

The edit, on the four copies (``WEIGHTLESS_STEER_HC_COPIES``):

- ``mean`` (the default): the GLP file was derived on the mean of the four
  copies, m = (R0 + R1 + R2 + R3) / 4, and says to apply
  h <- h - alpha (h.d) d there. With c = m.d, every copy gets the same
  subtraction, R''[k] = R'[k] - alpha c d. The mean of the copies is then
  exactly the projected mean, and the differences between the copies are
  untouched: the smallest change to the four copies that applies the file's
  edit to the stream it was measured on.
- ``each``: every copy is projected on its own, R''[k] = R'[k] -
  alpha (R'[k].d) d. Its mean is the projected mean too (the edit is
  linear), and it also scales the d part of the differences between the
  copies, so any weighted combine of the copies has its d part scaled by
  (1 - alpha). This is what the spark-ds41f vLLM series does.

Both give the same mean. Which one the file's published numbers were
measured with is not documented; it stays open until a GPU A/B against
those numbers. A spec_version 3 file settles it for itself: the spec
defines its edit as ``mean``.

GLP spec_version 3 files (glp.hook_point = hc_fold_pre_engram, the GLP-27
vector) name this same site: the fold that consumes layer N's FFN write,
before Engram, with direction.N, and the edit of the mean broadcast to every
copy, which is ``mean`` above (docs/spec/GLP.md, "Multi-copy streams and the
fold site"). The spec adds the companion writer: the attention output
(o_proj / wo_b, the write the attention boundary's ``HcUpdate.post`` folds
in) is projected with the same stack and the same alpha, and a reader that
cannot arm it must refuse the file. For such a file this row also wraps the
attention update's ``post`` and edits its ``output`` argument row by row
(``dsv41_attn_writer``). The layer at position k uses stack row k + 1, the
mapping of the vLLM plugin's archs/dsv41.py and its tests ("layer i uses row
i+1"): for GLP-27 (rows 11..37) the folds of layers 11..37 and the writers
of layers 10..36. A spec 3 file runs ``mean`` only, and is refused with TP
for the attention (its write reaches the post as a TP-partial sum).

What this row refuses, before anything is hooked (``dsv41_preflight``):
- a model_type other than exactly deepseek_v41, or a model without
  ``hc_pre_from_prev_sublayer`` (the post this row wraps exists only there);
- a file for another site or model: ``ffn_out_pre_residual`` (the
  DeepSeek-V4-0731 GLP-29 site; the V4.1 file says a reader that applies
  only that hook must refuse it) and DeepSeek-V4 files;
- hc_mult that is not a power of two (the mean is computed exactly);
- SGLang's decoder tail graph (``decoder_replay_tail``): it replays the late
  layers from a graph inside an otherwise eager forward, where the fired
  check cannot see them. Turn it off with
  SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=0 (the tail layers then run in
  the model loop) or --no-enable-decoder-swa-bounded-replay;
- an unknown WEIGHTLESS_STEER_HC_COPIES.
Per layer (``dsv41_hc_post``): a layer that is not built from SGLang's stage
boundaries, an FFN update that is not the layer's own HcUpdate, a post that
returns anything but an HcState holding a [T, hc_mult, hidden] tensor.

TP: the post returns the full streams on every TP rank (the sum is
complete before or inside the post), so the edit is the same on every
rank and TP is allowed. Pipeline: each rank wraps its own layers; the
rank's last layer is folded by its FinalRead before the hand-over, so it is
steered there.
"""
from __future__ import annotations

import os

import torch

from ..core import TEMP_BUDGET_BYTES, dot_fixed
from .base import (HOOK_POINT, ArchRow, hidden_size, register_companion, register_preflight,
                   register_special)

MODEL_TYPE = "deepseek_v41"
ROW_NAME = "DeepseekV4ForCausalLM:deepseek_v41"  # the served class, and the variant
SERVED_CLASS = "DeepseekV4ForCausalLM"
LAYER_CLASS = "DeepseekV4CudaDecoderLayer"
FFN_HOOK = "ffn_out_pre_residual"
FOLD_HOOK = "hc_fold_pre_engram"  # GLP spec_version 3: the same fold, plus the companion writer
FOLD_SPEC_VERSION = 3
WRITER_ROW_OFFSET = 1  # the layer at position k arms stack row k + 1 (vLLM archs/dsv41.py)
WRITER_SITE = "attn_boundary.update.post output (the attention write, o_proj/wo_b)"
COPIES_ENV = "WEIGHTLESS_STEER_HC_COPIES"
COPIES_MODES = ("mean", "each")

SITE = "ffn_boundary.update.post (hc streams, pre-Engram)"

# The tuple fields are declarative only: the special handler owns the call
# and its HcState. The site's stream is [T, hc_mult, hidden] (per_stream);
# the edit has no residual.
_DSV41 = dict(
    layers=frozenset({LAYER_CLASS}),
    hooks=frozenset({HOOK_POINT, FOLD_HOOK}),
    width=hidden_size,
    arity=1, hidden_index=0, residual_index=None, per_stream=True,
    install="special:dsv41_hc_post", exec_id="layer", tp="ok",
    hint=frozenset({MODEL_TYPE}),
    boundary=True,
)

ROW = ArchRow(backbone=("model",), **_DSV41)
ROWS = {ROW_NAME: ROW}


def _configs(model, runner):
    out = []
    for cfg in (getattr(getattr(runner, "model_config", None), "hf_config", None),
                getattr(model, "config", None),
                getattr(getattr(model, "model", None), "config", None)):
        if cfg is not None:
            out.append(cfg)
    return out


def selects(model, runner=None):
    """True when this DeepseekV4ForCausalLM is DeepSeek-V4.1: a model_type
    starting with deepseek_v41, or hc_pre_from_prev_sublayer set. The row's
    preflight then demands model_type deepseek_v41 exactly."""
    for cfg in _configs(model, runner):
        if str(getattr(cfg, "model_type", "") or "").startswith(MODEL_TYPE):
            return True
        if getattr(cfg, "hc_pre_from_prev_sublayer", False) is True:
            return True
    bb = getattr(model, "model", None)
    return getattr(bb, "hc_pre_from_prev_sublayer", False) is True


# ---------------------------------------------------------------- the edit


def copies_mode():
    """WEIGHTLESS_STEER_HC_COPIES: "mean" (default) or "each"."""
    mode = os.environ.get(COPIES_ENV, "").strip().lower() or "mean"
    if mode not in COPIES_MODES:
        raise RuntimeError(f"weightless: {COPIES_ENV}={mode!r} is not one of "
                           f"{', '.join(COPIES_MODES)}")
    return mode


def copies_sum(xf):
    """Sum over the copies axis (-2) of an fp32 [..., S, W] tensor, added in a
    fixed order: ((x0 + x1) + x2) + ... The fused kernel adds in the same
    order, so both give the same bits."""
    acc = xf[..., 0, :]
    for k in range(1, xf.shape[-2]):
        acc = acc + xf[..., k, :]
    return acc


def hc_mean_delta(x, d, alpha):
    """One pass of the mean-coefficient edit on x [..., S, W]:

        m   = (sum_k f32(x_k)) * (1 / S)      (exact: S is a power of two)
        c   = dot_fixed(m, d)                 (order-independent, as core.py)
        x'k = round(f32(x_k) - (alpha * c) d) for every copy k

    The same steps and roundings as core.steer_delta, with the coefficient
    of the mean in place of each row's own. float64 inputs (the tests'
    exact reference) use a plain float64 dot product.
    """
    S = x.shape[-2]
    ct = torch.promote_types(x.dtype, torch.float32)
    xf = x.to(ct)
    d = d.to(ct)
    m = copies_sum(xf) * (1.0 / S)
    c = dot_fixed(m, d) if ct == torch.float32 else m @ d
    del m
    t = (alpha.to(ct) * c).unsqueeze(-1)
    td = (t * d).unsqueeze(-2)  # [..., 1, W]: one subtraction for every copy
    return (xf - td).to(x.dtype)


# Per token (bf16 copies, fp32 math), at most: f32(x) 4S, the result in fp32
# 4S and rounded 2S, plus the sum, the mean, p, |p|, int64 q and t*d (W each,
# 4+4+4+4+8+4): W * (10 S + 28) bytes, under core's TEMP_BUDGET_BYTES.
def hc_chunk_tokens(S, W):
    """Tokens per slice of the torch path: the largest power of two whose
    temporaries fit in core.TEMP_BUDGET_BYTES (512 at S = 4, W = 5120)."""
    n = max(1, TEMP_BUDGET_BYTES // (int(W) * (10 * int(S) + 28)))
    return 1 << (n.bit_length() - 1)


def hc_mean_rows(x, d, alpha):
    """``hc_mean_delta`` over x [T, S, W], at most ``hc_chunk_tokens`` tokens at
    a time (every token is computed on its own, so the slices give the same
    bits as one pass; the slice count depends only on the shape)."""
    T, S, W = x.shape
    step = hc_chunk_tokens(S, W)
    if T <= step:
        return hc_mean_delta(x, d, alpha)
    out = torch.empty_like(x)
    for s in range(0, T, step):
        e = min(T, s + step)
        out[s:e].copy_(hc_mean_delta(x[s:e], d, alpha))
    return out


class HcEdit:
    """The edit the site runs on the [T, S, W] streams, for one copies mode.

    Called by core.make_site as ``edit(x, r, d, alpha, kernel)`` (r is always
    None here). ``diag_view`` hands the eager diagnostics the stream the file
    was derived on in "mean" mode (the copies' mean, before and after), and
    the copies as rows in "each" mode. ``self_check`` is the start-up check of
    the fused kernel for this edit (install.pick_kernel)."""

    def __init__(self, mode):
        self.mode = mode
        self.label = ("hc copies: mean coefficient" if mode == "mean"
                      else "hc copies: each projected")

    def __call__(self, x, r, d, alpha, kernel):
        if r is not None:
            raise RuntimeError("weightless: the DeepSeek-V4.1 site has no residual; refusing")
        from ..core import steer_rows
        if self.mode == "each":
            if kernel == "triton" and x.is_cuda:
                from ..fused import steer_delta_fused
                return steer_delta_fused(x, None, d, alpha)
            return steer_rows(x, None, d, alpha)
        if kernel == "triton" and x.is_cuda:
            from ..fused import steer_hc_mean_fused
            return steer_hc_mean_fused(x, d, alpha)
        return hc_mean_rows(x, d, alpha)

    def diag_view(self, x, x_new):
        if self.mode == "each":
            return x, x_new
        S = x.shape[-2]
        return (copies_sum(x.float()) * (1.0 / S), copies_sum(x_new.float()) * (1.0 / S))

    def self_check(self, device, dtype, width):
        from .. import fused
        if self.mode == "each":
            return fused.self_check(device, dtype=dtype, width=width, with_residual=False)
        return fused.self_check_hc_mean(device, dtype=dtype, width=width)

    @property
    def form(self):
        return f"{self.label}"


def make_edit():
    """The edit for WEIGHTLESS_STEER_HC_COPIES (install time)."""
    return HcEdit(copies_mode())


# ---------------------------------------------------------------- refusals


_TAIL = (
    "SGLang's decoder tail graph is on (the model has a decoder_replay_tail: DeepSeek-V4.1 "
    "decoder SWA bounded replay with SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS > 0). It replays "
    "the late layers from a separately captured graph inside an eager forward, where the "
    "fired check cannot see the steered sites, so every such forward would fail. Set "
    "SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=0 (the tail layers then run in the model loop, "
    "still bounded) or pass --no-enable-decoder-swa-bounded-replay. Refusing."
)


def _types(runner, backbone):
    out = []
    for cfg in (getattr(getattr(runner, "model_config", None), "hf_config", None),
                getattr(backbone, "config", None)):
        mt = getattr(cfg, "model_type", None) if cfg is not None else None
        if mt is not None and str(mt) != "":
            out.append(str(mt))
    return out


@register_preflight("dsv41_hc_post")
def dsv41_preflight(*, runner, backbone, meta, path, row, row_name):
    """Row checks that run before anything is hooked."""
    seen = _types(runner, backbone)
    if not seen:
        raise RuntimeError(f"weightless: {row_name}: no model_type on the hf config or the "
                           f"backbone config. Refusing.")
    other = sorted({m for m in seen if m != MODEL_TYPE})
    if other:
        raise RuntimeError(
            f"weightless: the {row_name} row serves model_type {MODEL_TYPE!r} only, and this "
            f"model has model_type {', '.join(repr(m) for m in other)}. A model type that reuses "
            f"the class may change the layer loop or the mHC post, so the row does not guess. "
            f"Refusing.")
    cfg = backbone.config
    if not (getattr(backbone, "hc_pre_from_prev_sublayer", False) is True
            and getattr(cfg, "hc_pre_from_prev_sublayer", False) is True):
        raise RuntimeError(
            f"weightless: {row_name}: the backbone or its config does not have "
            f"hc_pre_from_prev_sublayer=True, so the V4.1 mHC post this row wraps is not the one "
            f"SGLang runs. Refusing.")
    hc = int(getattr(cfg, "hc_mult", 0) or 0)
    if hc < 1 or hc & (hc - 1):
        raise RuntimeError(f"weightless: {row_name}: hc_mult={hc} is not a power of two; the "
                           f"mean over the copies would not be exact. Refusing.")
    hook = meta.get("glp.hook_point")
    hint = str(meta.get("controlvector.model_hint") or "")
    if hook == FFN_HOOK:
        raise RuntimeError(
            f"weightless: {path} has glp.hook_point={FFN_HOOK!r}, which the DeepSeek-V4.1 row "
            f"does not serve: that is the FFN write alone (the DeepSeek-V4-0731 GLP-29 site), and "
            f"this row steers the post-layer hyper-connection streams "
            f"(residual_stream_post_layer), the site the V4.1 files were derived and are applied "
            f"at. Refusing.")
    if hook == FOLD_HOOK:
        from weightless_steer.container import _int_meta  # the shared loader's coercion
        try:
            version = _int_meta(meta, "glp.spec_version", 1)
        except ValueError:
            version = -1
        if version < FOLD_SPEC_VERSION:
            raise RuntimeError(
                f"weightless: {path} has glp.hook_point={FOLD_HOOK!r}, an hc fold site, with "
                f"glp.spec_version={meta.get('glp.spec_version')!r}; the site requires spec_version "
                f"{FOLD_SPEC_VERSION} (docs/spec/GLP.md). Refusing.")
        mode = os.environ.get(COPIES_ENV, "").strip().lower() or "mean"
        if mode != "mean":
            raise RuntimeError(
                f"weightless: {path} is a spec_version 3 fold-site file, whose edit is defined on "
                f"the mean over the copies broadcast to every copy ({COPIES_ENV}=mean); "
                f"{COPIES_ENV}={mode!r} would apply another edit. Refusing.")
    if hint and hint != MODEL_TYPE and hint.startswith("deepseek_v4"):
        raise RuntimeError(
            f"weightless: {path} is a DeepSeek-V4 file (model_hint={hint!r}); the "
            f"{row_name} row serves DeepSeek-V4.1 (model_hint {MODEL_TYPE!r}) files only. "
            f"Refusing.")
    if getattr(backbone, "decoder_replay_tail", None) is not None:
        raise RuntimeError(f"weightless: {row_name}: {_TAIL}")
    copies_mode()  # a bad WEIGHTLESS_STEER_HC_COPIES fails here, before anything is hooked


# ---------------------------------------------------------------- the site


class Dsv41SiteError(RuntimeError):
    pass


@register_special("dsv41_hc_post", site=SITE, boundary_site=SITE, edit=make_edit)
def dsv41_hc_post(layer, site, *, layer_id, backbone, row):
    """Steer the HcState that one DeepseekV4CudaDecoderLayer's FFN post
    returns: a wrapper on ``layer.ffn_boundary.declaration.update.post`` (the
    instance). See the module text."""
    from ..core import stage_boundaries

    where = f"weightless: DeepSeek-V4.1 layer {layer_id}"
    own_id = getattr(layer, "layer_id", layer_id)
    if int(own_id) != int(layer_id):
        raise RuntimeError(f"{where}: the module at index {layer_id} says layer_id={own_id}; "
                           f"the file's absolute layer ids would land on the wrong layer. "
                           f"Failing closed.")
    if getattr(layer, "boundary_pre_from_prev", None) is not True:
        raise RuntimeError(f"{where}: boundary_pre_from_prev is not True (a NEXTN layer, or not "
                           f"the V4.1 mHC scheme); its FFN post does not return the HcState this "
                           f"row steers. Failing closed.")
    pair = stage_boundaries(layer)
    if pair is None:
        raise RuntimeError(f"{where} ({type(layer).__name__}) is not built from SGLang's stage "
                           f"boundaries; the FFN update this row wraps is not there. Failing "
                           f"closed.")
    attn, ffn = pair
    upd = ffn.declaration.update
    aupd = attn.declaration.update
    if (type(upd).__name__ != "HcUpdate" or getattr(upd, "attention", None) is not False
            or getattr(upd, "layer", None) is not layer):
        raise RuntimeError(
            f"{where}: the FFN boundary's update is {type(upd).__name__} "
            f"(attention={getattr(upd, 'attention', None)!r}), not this layer's mHC HcUpdate for "
            f"the FFN; the SGLang DeepSeek-V4.1 layout changed. Failing closed.")
    if type(aupd).__name__ != "HcUpdate" or getattr(aupd, "attention", None) is not True:
        raise RuntimeError(f"{where}: the attention boundary's update is "
                           f"{type(aupd).__name__}, not the attention HcUpdate. Failing closed.")
    if not hasattr(upd, "__dict__") or not callable(getattr(upd, "post", None)):
        raise RuntimeError(f"{where}: the FFN HcUpdate has no post method, or takes no instance "
                           f"attributes. Failing closed.")
    if "post" in vars(upd):
        raise RuntimeError(f"{where}: the FFN update's post is already wrapped on this instance; "
                           f"refusing to steer twice. Failing closed.")
    width = int(row.width(backbone.config))
    hidden = int(getattr(layer, "hidden_size", width))
    hc = int(getattr(layer, "hc_mult", 0) or 0)
    if hidden != width or hc != int(getattr(backbone.config, "hc_mult", -1)):
        raise RuntimeError(f"{where}: layer hidden_size {hidden} / hc_mult {hc} do not match the "
                           f"row width {width} / config hc_mult. Failing closed.")

    orig = upd.post  # the bound method

    def post(*args, **kwargs):
        state = orig(*args, **kwargs)
        if type(state).__name__ != "HcState" or not callable(getattr(state, "with_residual", None)):
            raise Dsv41SiteError(
                f"{where}: the FFN post returned {type(state).__name__}, not an HcState; the "
                f"post-layer streams are not where this row reads them. Failing closed.")
        streams = getattr(state, "streams", None)
        if not isinstance(streams, torch.Tensor):
            raise Dsv41SiteError(
                f"{where}: the FFN post left its streams as {type(streams).__name__} (an "
                f"outstanding post, the HIP fused boundary); refusing to steer them.")
        if streams.dim() != 3 or streams.shape[1] != hc or streams.shape[2] != width:
            raise Dsv41SiteError(
                f"{where}: the post-layer streams have shape {tuple(streams.shape)}, expected "
                f"[T, {hc}, {width}]. Failing closed.")
        return state.with_residual(site(streams, None))

    post._weightless_layer_id = int(layer_id)
    upd.post = post

    def undo():
        if vars(upd).get("post") is post:
            del upd.post

    return undo


# ------------------------------------------------- the companion writer (spec 3)


@register_companion("dsv41_hc_post", hooks=(FOLD_HOOK,), row_offset=WRITER_ROW_OFFSET,
                    site=WRITER_SITE)
def dsv41_attn_writer(layer, site, *, layer_id, row_id, backbone, row):
    """Arm the companion writer of a spec 3 fold-site file on one layer: a
    wrapper on ``layer.attn_boundary.declaration.update.post`` (the instance)
    that edits the attention write, ``output`` [T, hidden], with stack row
    ``row_id`` before the original post folds it into the copies."""
    from ..core import stage_boundaries

    where = f"weightless: DeepSeek-V4.1 layer {layer_id} (companion writer, row {row_id})"
    own_id = getattr(layer, "layer_id", layer_id)
    if int(own_id) != int(layer_id):
        raise RuntimeError(f"{where}: the module at index {layer_id} says layer_id={own_id}. "
                           f"Failing closed.")
    if getattr(layer, "boundary_pre_from_prev", None) is not True:
        raise RuntimeError(f"{where}: boundary_pre_from_prev is not True; the attention post is "
                           f"not the V4.1 one. Failing closed.")
    pair = stage_boundaries(layer)
    if pair is None:
        raise RuntimeError(f"{where} ({type(layer).__name__}) is not built from SGLang's stage "
                           f"boundaries. Failing closed.")
    aupd = pair[0].declaration.update
    if (type(aupd).__name__ != "HcUpdate" or getattr(aupd, "attention", None) is not True
            or getattr(aupd, "layer", layer) is not layer):
        raise RuntimeError(f"{where}: the attention boundary's update is {type(aupd).__name__} "
                           f"(attention={getattr(aupd, 'attention', None)!r}), not this layer's "
                           f"attention HcUpdate. Failing closed.")
    if not hasattr(aupd, "__dict__") or not callable(getattr(aupd, "post", None)):
        raise RuntimeError(f"{where}: the attention HcUpdate has no post method, or takes no "
                           f"instance attributes. Failing closed.")
    if "post" in vars(aupd):
        raise RuntimeError(f"{where}: the attention update's post is already wrapped on this "
                           f"instance; refusing to steer twice. Failing closed.")
    attn_tp = int(getattr(getattr(layer, "self_attn", None), "attn_tp_size", 1) or 1)
    if attn_tp > 1:
        raise RuntimeError(
            f"{where}: attention TP is {attn_tp}; the attention write reaches the post as a "
            f"TP-partial sum, which the companion writer is not validated on. Serve the "
            f"spec_version 3 file with attention TP 1 (pipeline parallelism is fine). Refusing.")
    width = int(row.width(backbone.config))
    orig = aupd.post  # the bound method

    def post(output, state, *args, **kwargs):
        if not isinstance(output, torch.Tensor):
            raise Dsv41SiteError(
                f"{where}: the attention write is {type(output).__name__}, not a tensor (a "
                f"deferred all-reduce or a fused boundary); refusing to fold it unsteered.")
        if output.dim() != 2 or output.shape[-1] != width:
            raise Dsv41SiteError(
                f"{where}: the attention write has shape {tuple(output.shape)}, expected "
                f"[T, {width}]. Failing closed.")
        return orig(site(output, None), state, *args, **kwargs)

    post._weightless_layer_id = int(layer_id)
    post._weightless_row_id = int(row_id)
    aupd.post = post

    def undo():
        if vars(aupd).get("post") is post:
            del aupd.post

    return undo
