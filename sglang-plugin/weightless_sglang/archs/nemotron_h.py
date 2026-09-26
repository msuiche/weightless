"""Nemotron-3.5-Lightning (``NemotronHForCausalLM``,
``NemotronHPuzzleForCausalLM``): the post-layer residual stream.

Four decoder-layer classes, each a single mixer (Mamba2, attention, dense
MLP or MoE), all returning ``(hidden, residual)`` with the add left to the
next layer's fused add+norm, so the stream is ``hidden + residual`` (the
Qwen convention; vllm-plugin archs/nemotron_h.py). The model loop calls
``layer.forward(...)``, where a forward hook never runs, so the row wraps
each steered layer's instance forward (``core.wrap_forward``). An attention
or Mamba layer that feeds an MLP/MoE layer hands a TP-partial plain tensor
to it (upstream: skip_reduce), with no marker: TP > 1 is refused.
"""
from __future__ import annotations

from .base import HOOK_POINT, ArchRow, hidden_size

_NEMOTRON_H = dict(
    layers=frozenset({"NemotronHMambaDecoderLayer", "NemotronHAttentionDecoderLayer",
                      "NemotronHMLPDecoderLayer", "NemotronHMoEDecoderLayer"}),
    hooks=frozenset({HOOK_POINT}),
    width=hidden_size,
    arity=2, hidden_index=0, residual_index=1,
    install="forward_wrap", exec_id="layer", tp="refuse",
    hint=frozenset({"nemotron_h"}),
)

ROWS = {
    "NemotronHForCausalLM": ArchRow(backbone=("model",), **_NEMOTRON_H),
    "NemotronHPuzzleForCausalLM": ArchRow(backbone=("model",), **_NEMOTRON_H),
}
