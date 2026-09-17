"""Real-vLLM guard: every registered adapter must import for real.

The offline suites stub the upstream modules, so a moved or renamed
upstream path — glm5next's nvidia.model -> common.model rehoming is the
live example — keeps the pin tests green (they pin against a vendored
snapshot) and only breaks at serve time. With a real vLLM installed
(image build, serving container, preflight), this imports every adapter
in SHADOWED_ARCHS so the failure happens before the GPU boots. Without a
real vLLM it skips, leaving the offline run unchanged.
"""
import importlib
import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))          # vllm-plugin/

try:
    import vllm  # noqa: F401
except ImportError:
    vllm = None


@unittest.skipUnless(vllm is not None,
                     "no real vLLM installed — offline suites stub it")
class TestRealVllmAdapters(unittest.TestCase):
    def test_registered_adapters_import(self):
        import torch

        from weightless_steer.plugin import SHADOWED_ARCHS

        self.assertTrue(SHADOWED_ARCHS)
        for arch_name, target in SHADOWED_ARCHS.items():
            module_path, _, class_name = target.partition(":")
            with self.subTest(arch=arch_name, target=target):
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
                self.assertTrue(issubclass(cls, torch.nn.Module))


if __name__ == "__main__":
    unittest.main()
