"""Qwen3.8-27B (qwen3_5): the post-layer residual stream ``hidden + residual``.

Both served classes hold the backbone (``models/qwen3_5.py``
``Qwen3_5ForCausalLM``) at ``.model``: the multimodal
``Qwen3_5ForConditionalGeneration``, and the text-only ``Qwen3_5ForCausalLM``
of ``models/qwen3_5_text.py``, a wrapper around it (so its backbone path
tries ``model`` first and the model itself second). The loop calls
``layer(...)`` and each layer returns ``(hidden, residual)``, with the add
left to the next layer's fused add+RMSNorm; the structure tests pin this in
both SGLang trees tested. Same site as vllm-plugin archs/qwen38.py.
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
)

ROWS = {
    "Qwen3_5ForConditionalGeneration": ArchRow(backbone=("model",), **_QWEN3_5),
    "Qwen3_5ForCausalLM": ArchRow(backbone=("model", ""), **_QWEN3_5),
}
