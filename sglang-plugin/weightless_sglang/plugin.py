"""SGLang general-plugin entry point (group ``sglang.srt.plugins``).

SGLang calls ``load_plugins()`` at the top of every scheduler (TP/PP rank)
process, before the ModelRunner exists (managers/scheduler.py,
run_scheduler_process). ``register()`` puts one AFTER hook on
``ModelRunner.load_model``; the hook loads the GLP file and installs the
steering once the weights are in, which is before the memory pools are
sized and before any CUDA graph is captured, so the edit is recorded into
every captured graph.

Fail-closed note: SGLang's plugin loader catches and only logs exceptions
raised by ``register()`` (plugins/__init__.py, load_plugins), so nothing is
validated here. Every check lives in the load hook, where an exception
propagates out of ``load_model`` and kills the boot. An installed plugin
with ``WEIGHTLESS_STEER_PATH`` unset registers nothing at all.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

TARGET = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"


def _after_load_model(result, runner, *args, **kwargs):
    from .install import install_steering

    install_steering(runner, source="env")
    return None  # keep load_model's own return value


def register() -> None:
    """Entry point for the ``sglang.srt.plugins`` group."""
    if not os.environ.get("WEIGHTLESS_STEER_PATH", "").strip():
        return  # steering not requested: leave SGLang stock
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(TARGET, _after_load_model, HookType.AFTER)
    logger.info("weightless-steer: AFTER hook registered on %s", TARGET)
