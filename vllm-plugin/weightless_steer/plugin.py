"""vLLM general plugin entry point: shadow steered archs via the registry.

Loaded by vLLM in every process that resolves a model class (process0,
engine core, workers) through the `vllm.general_plugins` entry point. When
WEIGHTLESS_STEER_PATH is set, each supported arch name is overwritten in
the ModelRegistry with its Steered* subclass — legal and documented
(register_model overwrites, debug-logged). When the env var is unset the
registry is left stock, so an installed-but-unconfigured plugin has zero
behavioural effect.

Fail-closed note: vLLM's plugin loader catches and merely logs exceptions
raised by entry points (vllm/plugins/__init__.py load_plugins_by_group),
so register() must NOT try to validate the vector here — a raise would be
swallowed and the engine would boot unsteered. Validation lives in
SteeringCore.from_env, called from the Steered* model __init__, where a
failure kills the model load and the engine boot with it.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# arch name -> lazy "module:class" shadow target. Lazy string form keeps the
# adapter (and its vllm.model_executor imports) out of plugin-load time, as
# the registry intends for out-of-tree models.
SHADOWED_ARCHS = {
    "NemotronHForCausalLM":
        "weightless_steer.archs.nemotron_h:SteeredNemotronHForCausalLM",
    "Glm5NextForCausalLM":
        "weightless_steer.archs.glm5next:SteeredGlm5NextForCausalLM",
    # Qwen3.8-27B declares the multimodal wrapper arch, and that wrapper
    # builds its Qwen3_5ForCausalLM DIRECTLY (not via the registry), so both
    # names must be shadowed — see archs/qwen38.py.
    "Qwen3_5ForCausalLM":
        "weightless_steer.archs.qwen38:SteeredQwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration":
        "weightless_steer.archs.qwen38:SteeredQwen3_5ForConditionalGeneration",
    "HYV4ForCausalLM":
        "weightless_steer.archs.hy4:SteeredHy4ForCausalLM",
    "DeepseekV4ForCausalLM":
        "weightless_steer.archs.dsv4:SteeredDeepseekV4ForCausalLM",
    # qwen38fn: the RadixArk NVFP4 checkpoint declares Qwen4ExpForCausalLM's
    # multimodal sibling (the day-0 image's registry name for the arch); the
    # wrapper instantiates its language model directly, so the CausalLM
    # shadow alone would not cover it. The image registers both Qwen4Exp*
    # and Qwen3_8FlashNext* names (preflight-verified).
    "Qwen4ExpForConditionalGeneration":
        "weightless_steer.archs.qwen38fn:SteeredQwen3_8FlashNextForConditionalGeneration",
    "Qwen4ExpForCausalLM":
        "weightless_steer.archs.qwen38fn:SteeredQwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration":
        "weightless_steer.archs.qwen38fn:SteeredQwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextForCausalLM":
        "weightless_steer.archs.qwen38fn:SteeredQwen3_8FlashNextForCausalLM",
    "GlmMoeDsaForCausalLM":
        "weightless_steer.archs.glm53xl:SteeredGlmMoeDsaForCausalLM",
    "KimiLinearForCausalLM":
        "weightless_steer.archs.kimi_k3:SteeredKimiLinearForCausalLM",
    "OuroForCausalLM":
        "weightless_steer.archs.ouro:SteeredOuroForCausalLM",
    # Both inkling entry classes: the multimodal wrapper builds InklingModel
    # directly (not through the registry), so one shadow does not cover the
    # other — and Inkling-Small-NVFP4 resolves to the conditional one.
    "InklingForCausalLM":
        "weightless_steer.archs.inkling:SteeredInklingForCausalLM",
    "InklingForConditionalGeneration":
        "weightless_steer.archs.inkling:SteeredInklingForConditionalGeneration",
}


def register() -> None:
    """Entry point for the `vllm.general_plugins` group."""
    path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
    if not path:
        return  # steering not requested: leave every arch stock

    from vllm.model_executor.models import ModelRegistry

    for arch, target in SHADOWED_ARCHS.items():
        ModelRegistry.register_model(arch, target)
        logger.info("weightless-steer: shadowed %s -> %s", arch, target)
