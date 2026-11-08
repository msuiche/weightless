"""The architecture rows, one module per architecture, and the served
model classes the plugin refuses by name.

``ARCH`` maps each row name to its row. A row name is a served model class
name (exact), or ``Class:model_type`` for a variant: a model that SGLang
serves through another model's class. ``VARIANTS`` lists, per served class,
the variants and the test that picks each one (install.resolve_row tries
them first, in order); a class with no matching variant gets its own row.
Anything that is in neither ``ARCH`` nor ``REFUSED`` fails the boot with the
generic "no SGLang steering adapter" error. A class in ``REFUSED`` fails with
its own reason: the vLLM plugin has an adapter for it, but SGLang either has
no such model, runs it in a way this plugin cannot steer without an SGLang
change, or uses the name for a different model (KimiLinearForCausalLM).

DeepSeek-V4.1 is served as ``DeepseekV4ForCausalLM`` (SGLang's
configs/deepseek_v41.py rewrites the architecture name) with model_type
deepseek_v41: the ``DeepseekV4ForCausalLM:deepseek_v41`` variant
(dsv41.py). A DeepSeek-V4 residual-site file is refused by the DeepSeek-V4
row itself (dsv4.py, ``dsv4_preflight``).
"""
from __future__ import annotations

from . import dsv4, dsv41, glm5next, glm53xl, hy4, kimi_k3, nanbeige, nemotron_h, qwen38, qwen38fn
from .base import BOUNDARY_SITE, COMPANION, EDIT, PREFLIGHT, SPECIAL  # noqa: F401  (install.py reads them here)

# Row name (a served model class name, exact, or Class:model_type for a
# variant) -> row. Anything else fails closed.
ARCH = {
    **qwen38.ROWS,
    **glm5next.ROWS,
    **dsv4.ROWS,
    **glm53xl.ROWS,
    **hy4.ROWS,
    **qwen38fn.ROWS,
    **nemotron_h.ROWS,
    **kimi_k3.ROWS,
    **nanbeige.ROWS,
    **dsv41.ROWS,
}

# Served model class name -> ((picks(model, runner) -> bool, row name), ...):
# the variants served through that class, tried in order before the class's
# own row.
VARIANTS = {
    dsv41.SERVED_CLASS: ((dsv41.selects, dsv41.ROW_NAME),),
}

# Served model classes refused by name -> the reason, quoted in the boot error.
_INKLING = (
    "Inkling's post-layer stream never exists between its decoder layers: each layer's "
    "mlp_sconv (a stateful short convolution with a cache) is deferred to the next "
    "layer's fused head, and the last one to the model tail, fused with the MoE "
    "all-reduce. A hook on the layer output would see the stream before that "
    "convolution, and recomputing it would advance the convolution state twice. "
    "Steering Inkling needs a site inside SGLang's Inkling model, which is an engine change."
)

REFUSED = {
    "OuroForCausalLM": (
        "Ouro (a looped model, total_ut_steps) has no SGLang model in any tree this plugin "
        "was tested on; serving it would need a new SGLang model file, which is an engine "
        "change. The vLLM plugin's Ouro adapter has no SGLang counterpart."
    ),
    "InklingForConditionalGeneration": _INKLING,
    "InklingForCausalLM": _INKLING,
    "DeepseekV4ForConditionalGeneration": (
        "SGLang has no DeepseekV4ForConditionalGeneration class. The published "
        "DeepSeek-V4-Flash-Vision-Exp config declares DeepseekV4ForCausalLM with "
        "model_type deepseek_v4, which the DeepSeek-V4 row serves, text only, with the "
        "-ffn file."
    ),
    "KimiLinearForCausalLM": (
        "SGLang's KimiLinearForCausalLM is the Kimi Linear model (models/kimi_linear.py), "
        "not Kimi-K3. The vLLM plugin serves Kimi-K3 under that name; SGLang serves Kimi-K3 "
        "as KimiK3ForConditionalGeneration or KimiK3LinearForCausalLM, and those rows are "
        "the ones steered here. No published GLP file targets Kimi Linear, and its layer "
        "outputs were not checked."
    ),
    "DeepseekV41ForCausalLM": (
        "SGLang has no DeepseekV41ForCausalLM class: configs/deepseek_v41.py rewrites that "
        "architecture name to DeepseekV4ForCausalLM, and the DeepSeek-V4.1 row steers the "
        "model there (DeepseekV4ForCausalLM:deepseek_v41). A class under this name was never "
        "checked, so it is not steered."
    ),
}
