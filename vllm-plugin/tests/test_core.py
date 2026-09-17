"""Offline tests for SteeringCore: env gating, buffers, and the apply math.

Small CPU tensors only — no GPU, no vllm. The math check is the contract:
h <- h - alpha * (h . d_hat) d_hat per steered layer, with zero stack rows
as an exact no-op everywhere else.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weightless_steer.core import SteeringCore  # noqa: E402
from glpfiles import good_meta, good_tensors, write_gguf  # noqa: E402

HOOK = "residual_stream_post_layer"
STEER_ENV = ["WEIGHTLESS_STEER_PATH", "WEIGHTLESS_STEER_ALPHA",
             "WEIGHTLESS_STEER_LAYERS", "WEIGHTLESS_STEER_HOOK"]


def steer_env(**kw):
    """Patch exactly the WEIGHTLESS_STEER_* vars, restoring after."""
    clear = {k: v for k, v in kw.items()}
    patcher = mock.patch.dict(os.environ, clear)
    return patcher


class FromEnvTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "vec.gguf")
        # No steering env leaks in from the outside world.
        for k in STEER_ENV:
            os.environ.pop(k, None)
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in STEER_ENV])

    def write(self, meta=None, tensors=None):
        write_gguf(self.path, good_meta() if meta is None else meta,
                   good_tensors() if tensors is None else tensors)

    def from_env(self, **env):
        with steer_env(**env):
            return SteeringCore.from_env(hook=HOOK, num_layers=4,
                                         hidden_size=8)

    def test_unset_path_returns_none(self):
        self.assertIsNone(self.from_env())

    def test_good_file_loads_normalised(self):
        self.write()
        core = self.from_env(WEIGHTLESS_STEER_PATH=self.path)
        self.assertEqual(sorted(core.dirs), [1, 2, 3])
        for d in core.dirs.values():
            self.assertAlmostEqual(float(d.norm()), 1.0, places=5)
        # alpha defaults to the file's glp.alpha_default.
        self.assertAlmostEqual(core.alpha, 2.5)

    def test_env_alpha_overrides_file(self):
        self.write()
        core = self.from_env(WEIGHTLESS_STEER_PATH=self.path,
                             WEIGHTLESS_STEER_ALPHA="4.0")
        self.assertAlmostEqual(core.alpha, 4.0)

    def test_layer_filter(self):
        self.write()
        core = self.from_env(WEIGHTLESS_STEER_PATH=self.path,
                             WEIGHTLESS_STEER_LAYERS="1,3")
        self.assertEqual(sorted(core.dirs), [1, 3])

    def test_hook_env_mismatch_fails_closed(self):
        self.write()
        with self.assertRaisesRegex(RuntimeError, "WEIGHTLESS_STEER_HOOK"):
            self.from_env(WEIGHTLESS_STEER_PATH=self.path,
                          WEIGHTLESS_STEER_HOOK="ffn_out_pre_residual")

    def test_width_mismatch_fails_closed(self):
        self.write(tensors=good_tensors(width=16))
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.from_env(WEIGHTLESS_STEER_PATH=self.path)

    def test_out_of_range_layer_fails_closed(self):
        self.write(meta=good_meta(layers=(1, 9)),
                   tensors=good_tensors(layers=(1, 9)))
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            self.from_env(WEIGHTLESS_STEER_PATH=self.path)

    def test_filter_to_empty_fails_closed(self):
        self.write()
        with self.assertRaisesRegex(RuntimeError, "matched no layers"):
            self.from_env(WEIGHTLESS_STEER_PATH=self.path,
                          WEIGHTLESS_STEER_LAYERS="0")

    def test_bad_file_fails_closed(self):
        self.write(meta={**good_meta(), "glp.mode": "add"})
        with self.assertRaisesRegex(ValueError, "glp.mode"):
            self.from_env(WEIGHTLESS_STEER_PATH=self.path)

    def test_rank2_vector_fails_closed(self):
        # This lane implements rank 1; a subspace vector must not be served
        # as direction 0 alone.
        e0 = np.eye(8, dtype=np.float32)[0].copy()
        e1 = np.eye(8, dtype=np.float32)[1].copy()
        tensors = {f"direction.{i}{suf}": (v, 0)
                   for i in (1, 2)
                   for suf, v in (("", e0), (".1", e1))}
        self.write(meta={**good_meta(layers=(1, 2)),
                         "glp.spec_version": "2", "glp.rank": "2",
                         "glp.orthonormal": "true"},
                   tensors=tensors)
        with self.assertRaisesRegex(RuntimeError, "rank-k"):
            self.from_env(WEIGHTLESS_STEER_PATH=self.path)


class BufferTests(unittest.TestCase):
    def test_buffers_dense_shaped_and_non_persistent(self):
        module = torch.nn.Linear(4, 4)  # any nn.Module owner
        dirs = {1: torch.nn.functional.normalize(
            torch.arange(8, dtype=torch.float32), dim=0)}
        core = SteeringCore(dirs, 2.0, HOOK, num_layers=4, hidden_size=8)
        core.register_buffers(module, torch.float32)

        self.assertEqual(tuple(module._steer_stack.shape), (4, 1, 8))
        self.assertEqual(tuple(module._steer_alpha.shape), ())
        self.assertAlmostEqual(float(module._steer_alpha), 2.0)
        # Non-persistent: load_weights (state_dict) never sees them.
        self.assertNotIn("_steer_stack", module.state_dict())
        self.assertNotIn("_steer_alpha", module.state_dict())
        # Dense stack: filled row is the unit direction, the rest are zeros.
        self.assertTrue(torch.allclose(module._steer_stack[1, 0], dirs[1]))
        for i in (0, 2, 3):
            self.assertTrue(torch.equal(
                module._steer_stack[i], torch.zeros(1, 8)))


class ApplyMathTests(unittest.TestCase):
    def make_core(self, dirs, alpha, num_layers=4, hidden=8):
        module = torch.nn.Module()
        core = SteeringCore(dirs, alpha, HOOK, num_layers=num_layers,
                            hidden_size=hidden)
        core.register_buffers(module, torch.float32)
        return core

    def test_apply_equals_manual_projection(self):
        torch.manual_seed(0)
        d = torch.nn.functional.normalize(torch.randn(8), dim=0)
        core = self.make_core({2: d}, alpha=2.5)
        h = torch.randn(7, 8)

        got = core.apply(2, h)
        want = h - 2.5 * (h @ d).unsqueeze(-1) * d
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

        # alpha=1 removes the d-component exactly.
        core1 = self.make_core({2: d}, alpha=1.0)
        out = core1.apply(2, h)
        self.assertTrue(torch.allclose(out @ d, torch.zeros(7), atol=1e-5))

    def test_apply_handles_batched_token_dims(self):
        torch.manual_seed(1)
        d = torch.nn.functional.normalize(torch.randn(8), dim=0)
        core = self.make_core({1: d}, alpha=1.5)
        h = torch.randn(2, 3, 8)  # [batch, tokens, hidden]

        got = core.apply(1, h)
        want = h - 1.5 * (h @ d).unsqueeze(-1) * d
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_zero_row_is_exact_noop(self):
        torch.manual_seed(2)
        d = torch.nn.functional.normalize(torch.randn(8), dim=0)
        core = self.make_core({2: d}, alpha=4.0)
        h = torch.randn(5, 8)
        for unsteered in (0, 1, 3):
            self.assertTrue(torch.equal(core.apply(unsteered, h), h))

    def test_disabled_core_is_noop_everywhere(self):
        module = torch.nn.Module()
        core = SteeringCore.disabled(hook=HOOK, num_layers=4, hidden_size=8)
        core.register_buffers(module, torch.float32)
        h = torch.randn(5, 8)
        for i in range(4):
            self.assertTrue(torch.equal(core.apply(i, h), h))


if __name__ == "__main__":
    unittest.main()
