"""Offline tests for the plugin entry point: when and what it shadows.

The vllm ModelRegistry is stubbed in sys.modules — register() must import
it lazily so the plugin module stays importable without vllm installed.
"""
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weightless_steer import plugin  # noqa: E402


class _FakeRegistry:
    def __init__(self):
        self.registered = {}

    def register_model(self, arch, target):
        self.registered[arch] = target


class RegisterTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("WEIGHTLESS_STEER_PATH", None)
        self.addCleanup(lambda: os.environ.pop("WEIGHTLESS_STEER_PATH", None))
        self.registry = _FakeRegistry()
        fake_models = types.ModuleType("vllm.model_executor.models")
        fake_models.ModelRegistry = self.registry
        self._saved = sys.modules.get("vllm.model_executor.models")
        sys.modules["vllm.model_executor.models"] = fake_models
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            sys.modules.pop("vllm.model_executor.models", None)
        else:
            sys.modules["vllm.model_executor.models"] = self._saved

    def test_unset_path_registers_nothing(self):
        plugin.register()
        self.assertEqual(self.registry.registered, {})

    def test_set_path_shadows_supported_archs(self):
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_STEER_PATH": "/tmp/v.gguf"}):
            plugin.register()
        self.assertEqual(
            self.registry.registered,
            {"NemotronHForCausalLM":
             "weightless_steer.archs.nemotron_h:SteeredNemotronHForCausalLM",
             "Glm5NextForCausalLM":
             "weightless_steer.archs.glm5next:SteeredGlm5NextForCausalLM",
             "OuroForCausalLM":
             "weightless_steer.archs.ouro:SteeredOuroForCausalLM"},
        )

    def test_registration_is_lazy_module_class_string(self):
        # The registry's documented lazy form is "<module>:<class>" — the
        # adapter (and its vllm imports) must not load at plugin time.
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_STEER_PATH": "/tmp/v.gguf"}):
            plugin.register()
        for target in self.registry.registered.values():
            module, _, cls = target.partition(":")
            self.assertTrue(module.startswith("weightless_steer.archs."))
            self.assertTrue(cls.startswith("Steered"))
        self.assertNotIn("weightless_steer.archs.nemotron_h", sys.modules)
        self.assertNotIn("weightless_steer.archs.glm5next", sys.modules)
        self.assertNotIn("weightless_steer.archs.ouro", sys.modules)


if __name__ == "__main__":
    unittest.main()
