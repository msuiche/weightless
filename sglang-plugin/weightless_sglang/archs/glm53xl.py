"""GLM-5.3 (``GlmMoeDsaForCausalLM``): the post-layer residual stream.

``DeepseekV2DecoderLayer`` returns ``(hidden, residual, topk_indices)`` with
the fold left to the next layer's add+norm: the Qwen pair edit, with
``topk_indices`` passed through (vllm-plugin archs/glm53xl.py). At TP > 1 the
site reduces an ``UnreducedOutput`` and refuses the partial-sum attribute
and a deferred MoE handoff (``core.make_site``).
"""
from __future__ import annotations

from .base import HOOK_POINT, ArchRow, hidden_size

_GLM53 = dict(
    layers=frozenset({"DeepseekV2DecoderLayer"}),
    hooks=frozenset({HOOK_POINT}),
    width=hidden_size,
    arity=3, hidden_index=0, residual_index=1,
    install="hook", exec_id="layer", tp="ok",
    hint=frozenset({"glm_moe_dsa"}),
)

ROWS = {
    "GlmMoeDsaForCausalLM": ArchRow(backbone=("model",), **_GLM53),
}
