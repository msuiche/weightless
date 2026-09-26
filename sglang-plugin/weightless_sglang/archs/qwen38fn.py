"""Qwen3.8-Flash-Next (``Qwen4ExpForConditionalGeneration``): the combined
hyper-connection stream.

Each ``Qwen4ExpLinearDecoderLayer`` / ``Qwen4ExpAttentionDecoderLayer``
ends with ``mlp_hyper_connection.combine`` and returns
``(hidden_states, None)``: the hyper-connection streams materialised, flat
and stream-major, ``[T, hc_count x hidden_size]`` (``unflatten(-1, (hc,
hidden))``, the layout the GLP-47 file was derived on: 4 x 2560 = 10240).
Unlike the vLLM build (vllm-plugin archs/qwen38fn.py), SGLang has no pending
combine at the layer boundary, so the post-layer stream is the tensor the
layer returns and the row is a plain hook row with no residual.

These classes subclass the Qwen3_5 decoder layers, so the row matches exact
class names only: an ``isinstance`` test would take a plain Qwen3_5 layer
(a ``hidden_size``-wide stream) for one of these.
"""
from __future__ import annotations

from .base import HOOK_POINT, ArchRow


def hc_stream_width(cfg):
    """Width of the widened stream: ``hidden_size x hc_count``. Refuses a
    config with fewer than two hyper-connection streams."""
    hc = int(getattr(cfg, "hc_count", 0) or 0)
    if hc < 2:
        raise RuntimeError(
            f"weightless: this qwen4_exp config has hc_count={hc}: there is no widened "
            f"hyper-connection stream, and the row steers that stream. Refusing."
        )
    return int(cfg.hidden_size) * hc


# The layers return (combined stream [T, hc_count x hidden], None); exact
# class names, because they subclass the Qwen3_5 layers (see above).
_QWEN4_EXP = dict(
    layers=frozenset({"Qwen4ExpLinearDecoderLayer", "Qwen4ExpAttentionDecoderLayer"}),
    hooks=frozenset({HOOK_POINT}),
    width=hc_stream_width,
    arity=2, hidden_index=0, residual_index=None,
    install="hook", exec_id="layer", tp="ok",
    hint=frozenset({"qwen4_exp"}),
)

ROWS = {
    "Qwen4ExpForConditionalGeneration": ArchRow(backbone=("model",), **_QWEN4_EXP),
}
