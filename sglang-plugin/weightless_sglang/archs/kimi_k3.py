"""Kimi-K3 (``KimiK3ForConditionalGeneration``, ``KimiK3LinearForCausalLM``):
the post-layer stream.

``KimiK3DecoderLayer`` returns ``(hidden, residual, sp_sharded)``. With
attention residuals on (K3: attn_res_block_size 12) the MLP folds the
pending prefix sum into its output and the layer returns ``(stream, None,
bool)``: hidden is the whole post-layer stream (the vLLM adapter's
prefix_sum + hidden_states, vllm-plugin archs/kimi_k3.py). Without them it
returns ``(mlp_out, residual, False)`` and the stream is ``hidden +
residual``. The loop calls ``self.layers[i](...)``. TP > 1 is refused: the
layer has several output forms at TP > 1 (the dense MLP, the MoE tails, the
fused attention all-reduce, the SP-MoE token shards) and no structure test
pins that each one is reduced before the layer returns. GLP-92's hint is
the text config's model_type, kimi_linear.

SGLang's ``KimiLinearForCausalLM`` is a different model (Kimi Linear), so
that name is refused in ``archs/__init__.py``.
"""
from __future__ import annotations

from .base import HOOK_POINT, ArchRow, hidden_size


def _kimi_kind(layer):
    return "linear" if type(getattr(layer, "self_attn", None)).__name__ == "KimiK3DeltaAttention" \
        else "full"


_KIMI_K3 = dict(
    layers=frozenset({"KimiK3DecoderLayer"}),
    hooks=frozenset({HOOK_POINT}),
    width=hidden_size,
    arity=3, hidden_index=0, residual_index=1, residual_may_be_none=True,
    install="hook", exec_id="layer", tp="refuse",
    hint=frozenset({"kimi_linear"}),
    kind=_kimi_kind,
)

ROWS = {
    "KimiK3ForConditionalGeneration": ArchRow(backbone=("language_model.model",), **_KIMI_K3),
    "KimiK3LinearForCausalLM": ArchRow(backbone=("model",), **_KIMI_K3),
}
