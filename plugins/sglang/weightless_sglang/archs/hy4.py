"""Hy4-preview (``HYV4ForCausalLM``): each iHC stream, one direction per stream.

hy_v4 runs iHC (identity hyper-connections): every ``HYV4DecoderLayer``
returns ``(hidden_states, topk_indices)`` with ``hidden_states`` the whole
multi-stream state ``[T, hc_mult, hidden_size]``, already merged by the
layer's own ``hc_mlp_layer.post``. There is no residual slot and no pending
combine, so the post-layer stream is the tensor the layer returns, and the
next layer reads it as is (``prepare_input`` passes a 3-D tensor through).
The GLP file holds one ``hidden_size``-wide direction per layer, applied to
each of the ``hc_mult`` streams on its own (vllm-plugin archs/hy4.py: the
core's einsum contracts the last axis only). The row is therefore a plain
hook row with ``per_stream`` set, and its width is ``hidden_size``, never
``hc_mult x hidden_size``.

SGLang's own config (configs/hy_v4.py) refuses a model without iHC, and
``HYV4Model`` refuses pipeline parallelism; the width below refuses a config
without iHC as well, so a changed config fails the boot here.
"""
from __future__ import annotations

from .base import HOOK_POINT, ArchRow


def ihc_stream_width(cfg):
    """Width of one iHC stream: ``hidden_size``. Refuses a config that has
    no iHC multi-stream state (``enable_ihc`` off or ``hc_mult`` < 1)."""
    hc = int(getattr(cfg, "hc_mult", 0) or 0)
    if not getattr(cfg, "enable_ihc", False) or hc < 1:
        raise RuntimeError(
            f"weightless: this hy_v4 config has enable_ihc="
            f"{getattr(cfg, 'enable_ihc', None)!r}, hc_mult={hc}: there is no iHC "
            f"multi-stream state, and the Hy4 row steers each iHC stream. Refusing."
        )
    return int(cfg.hidden_size)


# HYV4DecoderLayer returns ([T, hc_mult, hidden], topk_indices): the iHC
# streams already merged, no residual; slot 1 is topk_indices, passed
# through. SGLang refuses PP for this model.
_HY4 = dict(
    layers=frozenset({"HYV4DecoderLayer"}),
    hooks=frozenset({HOOK_POINT}),
    width=ihc_stream_width,
    arity=2, hidden_index=0, residual_index=None, per_stream=True,
    install="hook", exec_id="layer", tp="ok",
    hint=frozenset({"hy_v4"}),
)

ROWS = {
    "HYV4ForCausalLM": ArchRow(backbone=("model",), **_HY4),
}
