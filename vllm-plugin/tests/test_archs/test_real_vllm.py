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
sys.path.insert(0, str(_HERE.parents[1]))          # vllm-plugin/tests/

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


@unittest.skipUnless(vllm is not None,
                     "no real vLLM installed — offline suites stub it")
class TestRealVllmGlm5Next(unittest.TestCase):
    """glm5next lane, scoped: the installed vLLM must carry the arch, and
    the adapter's copied forwards must match the INSTALLED glm5next source.

    Scoped per-lane because not every shadowed arch exists in every vLLM
    vintage (stock 0.31 has glm5next but not ouro/qwen3_8_flash_next), so
    the all-arch guard above cannot pass on a stock image. This one gates
    the stock glm5next lane's container builds: import drift or upstream
    forward drift fails the build, not the first boot.
    """

    def test_installed_glm5next_matches_adapter_anchors(self):
        try:
            import vllm.models.glm5next  # noqa: F401
        except ImportError:
            self.skipTest("installed vLLM has no glm5next arch")
        from test_archs.test_glm5next import StructureTests

        adapter = importlib.import_module("weightless_steer.archs.glm5next")
        # The module the adapter actually imported its parents from (stock
        # common.model on vLLM >= 0.30, fork nvidia.model on the day-0
        # image) — drift in EITHER the import path or the copied bodies
        # fails here.
        parent_module = sys.modules[adapter.Glm5NextModel.__module__]
        installed_src = Path(parent_module.__file__).read_text()
        for name in ("LOOP_ANCHOR", "LAYER_PRE_ANCHOR", "LAYER_LAST_ANCHOR"):
            anchor = getattr(StructureTests, name)
            self.assertIn(anchor, installed_src,
                          f"{name} missing from {parent_module.__file__} — "
                          f"upstream drifted; re-copy the adapter forwards "
                          f"and re-pin patches/reference/")


if __name__ == "__main__":
    unittest.main()

