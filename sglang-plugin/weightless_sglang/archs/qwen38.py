"""Qwen3.8-27B (qwen3_5): the post-layer residual stream ``hidden + residual``.

Both served classes hold the backbone (``models/qwen3_5.py``
``Qwen3_5ForCausalLM``) at ``.model``: the multimodal
``Qwen3_5ForConditionalGeneration``, and the text-only ``Qwen3_5ForCausalLM``
of ``models/qwen3_5_text.py``, a wrapper around it (so its backbone path
tries ``model`` first and the model itself second). The loop calls
``layer(...)`` and the add of the layer's output to the residual is left to
the next layer's fused add+RMSNorm. Same site as vllm-plugin archs/qwen38.py.

Two SGLang layouts, told apart by what each layer holds:

- the older one (SGLang before upstream #41553, 2026-09-28; the serving
  build b62df575): each layer returns ``(hidden, residual)``; the forward
  hook edits ``hidden`` with ``r = residual`` (``core.make_post_layer_hook``).
- stage boundaries (``attn_boundary`` / ``ffn_boundary``; #41553 and later):
  each layer returns ``hidden`` alone and ``residual`` stays in
  ``forward_batch.residual_stream``; the forward hook reads both there and
  edits ``hidden`` in place (``core.make_stream_hook``).

Same stream, same edit and the same bits either way; the structure tests pin
both layouts per SGLang tree.
"""
from __future__ import annotations

from .base import HOOK_POINT, ArchRow, hidden_size

_QWEN3_5 = dict(
    layers=frozenset({"Qwen3_5LinearDecoderLayer", "Qwen3_5AttentionDecoderLayer"}),
    hooks=frozenset({HOOK_POINT}),
    width=hidden_size,
    arity=2, hidden_index=0, residual_index=1,
    install="hook", exec_id="layer", tp="ok",
    hint=frozenset({"qwen3_5"}),
    boundary=True,
)

ROWS = {
    "Qwen3_5ForConditionalGeneration": ArchRow(backbone=("model",), **_QWEN3_5),
    "Qwen3_5ForCausalLM": ArchRow(backbone=("model", ""), **_QWEN3_5),
}
