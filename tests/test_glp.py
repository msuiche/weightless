"""Tests for glp.py -- the GLP (GGUF) steering API for transformers models.

Offline tests write tiny GGUF v3 fixtures with weightless-steer's shared
test writer (vllm-plugin/tests/glpfiles.py -- the same bytes the
container reader's own gate tests use) and steer a fake HF-style decoder
stack. The network smoke test downloads a tiny random model and verifies
the projection end to end; it skips when the hub is unreachable.
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(ROOT))
    import glp

    _spec = importlib.util.spec_from_file_location(
        "glpfiles", ROOT / "vllm-plugin" / "tests" / "glpfiles.py")
    glpfiles = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(glpfiles)


def unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def meta(layers, alpha="1.0", **overrides):
    m = glpfiles.good_meta(layers=layers, alpha=alpha)
    m.update(overrides)
    return m


def tensors(dirs):
    """dirs: {layer: array-like} -> glpfiles tensor map (F32, ggml type 0)."""
    return {f"direction.{i}": (np.asarray(d, dtype=np.float32), 0)
            for i, d in dirs.items()}


@unittest.skipIf(torch is None, "torch/numpy not installed")
class LoadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "v.glp")
        self.dirs = {1: unit([1.0] * 8), 2: unit([1.0, -1.0] * 4)}

    def write(self, m, t=None):
        glpfiles.write_gguf(self.path, m,
                            tensors(self.dirs) if t is None else t)
        return self.path

    def test_project_mode_loads(self):
        m, dirs = glp.load_glp(self.write(meta(sorted(self.dirs))))
        self.assertEqual(m["glp.mode"], "project")
        self.assertEqual(sorted(dirs), [1, 2])
        self.assertTrue(
            torch.allclose(dirs[1], torch.tensor(self.dirs[1])))

    def test_add_mode_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "glp.mode='add'"):
            glp.load_glp(self.write(meta([1, 2], **{"glp.mode": "add"})))

    def test_missing_mode_is_fatal(self):
        m = meta([1, 2])
        del m["glp.mode"]
        with self.assertRaisesRegex(ValueError, "no glp.mode"):
            glp.load_glp(self.write(m))

    def test_hook_point_mismatch_is_fatal(self):
        m = meta([1, 2], **{"glp.hook_point": "ffn_out_pre_residual"})
        with self.assertRaisesRegex(ValueError, "glp.hook_point"):
            glp.load_glp(self.write(m))

    def test_missing_hook_point_is_fatal(self):
        m = meta([1, 2])
        del m["glp.hook_point"]
        with self.assertRaisesRegex(ValueError, "glp.hook_point"):
            glp.load_glp(self.write(m))

    def test_direction_zero_is_fatal(self):
        t = tensors(self.dirs)
        t["direction.0"] = (np.zeros(8, dtype=np.float32), 0)
        with self.assertRaisesRegex(ValueError, "direction.0"):
            glp.load_glp(self.write(meta([1, 2]), t))

    def test_layer_ids_mismatch_is_fatal(self):
        m = meta([1, 2], **{"glp.layer_ids_zero_based": "1,2,3"})
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            glp.load_glp(self.write(m))

    def test_rank2_loads_stacked(self):
        t = {"direction.1": (np.eye(8, dtype=np.float32)[0].copy(), 0),
             "direction.1.1": (np.eye(8, dtype=np.float32)[1].copy(), 0)}
        m = meta([1], **{"glp.spec_version": "2", "glp.rank": "2",
                         "glp.orthonormal": "true",
                         "glp.dir_scales": "1.0,0.5"})
        _, dirs = glp.load_glp(self.write(m, t))
        self.assertEqual(tuple(dirs[1].shape), (2, 8))

    def test_rank2_non_orthonormal_is_fatal(self):
        e0 = np.eye(8, dtype=np.float32)[0].copy()
        t = {"direction.1": (e0, 0), "direction.1.1": (e0, 0)}
        m = meta([1], **{"glp.spec_version": "2", "glp.rank": "2",
                         "glp.orthonormal": "true"})
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            glp.load_glp(self.write(m, t))

    def test_rank2_under_spec_version_1_is_fatal(self):
        t = {"direction.1": (np.eye(8, dtype=np.float32)[0].copy(), 0),
             "direction.1.1": (np.eye(8, dtype=np.float32)[1].copy(), 0)}
        m = meta([1], **{"glp.spec_version": "1", "glp.rank": "2",
                         "glp.orthonormal": "true"})
        with self.assertRaisesRegex(ValueError, "spec_version 2"):
            glp.load_glp(self.write(m, t))

    def test_hub_repo_single_file(self):
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            self.skipTest("huggingface_hub not installed")
        with mock.patch("huggingface_hub.list_repo_files",
                        return_value=["README.md", "v.glp"]), \
             mock.patch("huggingface_hub.hf_hub_download",
                        return_value="/cache/v.glp") as dl:
            self.assertEqual(glp._resolve("user/vector"), "/cache/v.glp")
            dl.assert_called_once_with("user/vector", "v.glp")

    def test_hub_repo_ambiguous_refused(self):
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            self.skipTest("huggingface_hub not installed")
        with mock.patch("huggingface_hub.list_repo_files",
                        return_value=["a.glp", "b.gguf"]):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                glp._resolve("user/vector")


class _Config:
    hidden_size = 8


if torch is not None:

    class _Layer(nn.Module):
        """HF-style decoder layer: tuple or bare-tensor output, residual add."""

        def __init__(self, tuple_out):
            super().__init__()
            self.tuple_out = tuple_out

        def forward(self, h):
            h = h + 1.0
            return (h, None) if self.tuple_out else h


    class _FakeModel(nn.Module):
        """Minimal HF decoder stand-in: .model.layers + .config.hidden_size."""

        def __init__(self, n_layers=4, tuple_out=True):
            super().__init__()
            self.config = _Config()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(
                [_Layer(tuple_out) for _ in range(n_layers)])

        def forward(self, h):
            for layer in self.model.layers:
                out = layer(h)
                h = out[0] if isinstance(out, tuple) else out
            return h


@unittest.skipIf(torch is None, "torch/numpy not installed")
class ApplyTests(unittest.TestCase):
    # direction.N steers layer N (no offset); the fake model has 4 layers,
    # so layers 1 and 3 get unit directions along dims 0 and 2.
    DIRS = {1: unit([1.0] + [0.0] * 7), 3: unit([0.0, 0.0, 1.0] + [0.0] * 5)}

    def _fixture(self, m=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = os.path.join(tmp.name, "v.glp")
        glpfiles.write_gguf(
            p, meta(sorted(self.DIRS)) if m is None else m,
            tensors(self.DIRS))
        return p

    def _captures(self, model, dirs):
        """Hooks registered AFTER apply_glp's, so they see steered output."""
        seen = {}
        for i in dirs:
            def cap(mod, args, out, i=i):
                seen[i] = out
            model.model.layers[i].register_forward_hook(cap)
        return seen

    def test_projection_zeroes_component_and_rewraps_tuple(self):
        model = _FakeModel(tuple_out=True)
        st = glp.apply_glp(model, self._fixture())
        seen = self._captures(model, self.DIRS)
        model(torch.full((2, 8), 3.0))
        for i, d in self.DIRS.items():
            out = seen[i]
            self.assertIsInstance(out, tuple)  # HF convention preserved
            dv = torch.tensor(d)
            self.assertLess((out[0] @ dv).abs().max().item(), 1e-5)
        self.assertTrue(st.attached)
        st.detach()
        st.detach()  # idempotent
        self.assertFalse(st.attached)

    def test_bare_tensor_output(self):
        model = _FakeModel(tuple_out=False)
        with glp.apply_glp(model, self._fixture()) as st:
            seen = self._captures(model, self.DIRS)
            model(torch.full((2, 8), 3.0))
            self.assertTrue(st.attached)
        self.assertFalse(st.attached)
        for i, d in self.DIRS.items():
            self.assertFalse(isinstance(seen[i], tuple))
            self.assertLess((seen[i] @ torch.tensor(d)).abs().max().item(),
                            1e-5)

    def test_alpha_default_from_metadata(self):
        model = _FakeModel(tuple_out=False)
        st = glp.apply_glp(model, self._fixture(meta([1, 3], alpha="0.5")))
        self.assertAlmostEqual(st.alpha, 0.5)
        seen = self._captures(model, self.DIRS)
        model(torch.full((1, 8), 3.0))
        # layer 1: h along d is 3.0 + 1 (layer 0) + 1 (layer 1) = 5.0;
        # alpha 0.5 leaves half
        got = (seen[1] @ torch.tensor(self.DIRS[1]))[0].item()
        self.assertAlmostEqual(got, 5.0 * 0.5, places=5)

    def test_alpha_argument_overrides_metadata(self):
        model = _FakeModel(tuple_out=False)
        st = glp.apply_glp(model, self._fixture(meta([1, 3], alpha="9.0")),
                           alpha=0.0)
        self.assertAlmostEqual(st.alpha, 0.0)
        seen = self._captures(model, self.DIRS)
        model(torch.full((1, 8), 3.0))
        got = (seen[1] @ torch.tensor(self.DIRS[1]))[0].item()
        self.assertAlmostEqual(got, 5.0, places=5)  # alpha 0: untouched

    def test_alpha_env_overrides_metadata_argument_wins(self):
        model = _FakeModel(tuple_out=False)
        with mock.patch.dict(os.environ, {"WEIGHTLESS_STEER_ALPHA": "0.25"}):
            st = glp.apply_glp(model, self._fixture(meta([1, 3], alpha="9.0")))
            self.assertAlmostEqual(st.alpha, 0.25)
            st.detach()
            st = glp.apply_glp(model, self._fixture(), alpha=0.75)
            self.assertAlmostEqual(st.alpha, 0.75)
            st.detach()

    def test_detach_restores_unsteered_output(self):
        model = _FakeModel(tuple_out=False)
        h0 = torch.full((1, 8), 3.0)
        want = model(h0)
        st = glp.apply_glp(model, self._fixture())
        model(h0)
        st.detach()
        self.assertTrue(torch.equal(model(h0), want))

    def _rank2_fixture(self, tmp, **overrides):
        p = os.path.join(tmp.name, "v.glp")
        m = meta([1], **{"glp.spec_version": "2", "glp.rank": "2",
                         "glp.orthonormal": "true",
                         "glp.dir_scales": "1.0,0.5", **overrides})
        glpfiles.write_gguf(
            p, m,
            {"direction.1": (np.eye(8, dtype=np.float32)[0].copy(), 0),
             "direction.1.1": (np.eye(8, dtype=np.float32)[1].copy(), 0)})
        return p

    def test_rank2_per_direction_alpha(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = _FakeModel(tuple_out=False)
        st = glp.apply_glp(model, self._rank2_fixture(tmp))
        self.assertEqual(st.rank, 2)
        seen = self._captures(model, {1: None})
        model(torch.full((1, 8), 3.0))
        st.detach()
        # layer 1 output is 5.0 along every dim (3.0 + 1 + 1); d0 at alpha
        # 1.0 is removed, d1 at 1.0 x dir_scale 0.5 is halved
        self.assertAlmostEqual(seen[1][0, 0].item(), 0.0, places=5)
        self.assertAlmostEqual(seen[1][0, 1].item(), 5.0 * 0.5, places=5)
        # orthogonal content passes through untouched
        self.assertAlmostEqual(seen[1][0, 2].item(), 5.0, places=5)

    def test_rank2_layer_scale(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = _FakeModel(tuple_out=False)
        st = glp.apply_glp(
            model, self._rank2_fixture(tmp, **{"glp.layer_scales": "1:2.0"}))
        seen = self._captures(model, {1: None})
        model(torch.full((1, 8), 3.0))
        st.detach()
        # d1 alpha = 1.0 x 0.5 x layer_scale 2.0 = 1.0: removed too
        self.assertAlmostEqual(seen[1][0, 1].item(), 0.0, places=5)

    def test_refuses_wrong_width(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = os.path.join(tmp.name, "v.glp")
        glpfiles.write_gguf(p, meta([1]), tensors({1: unit([1.0] * 16)}))
        with self.assertRaisesRegex(ValueError, "width 16"):
            glp.apply_glp(_FakeModel(), p)

    def test_refuses_layer_past_depth(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = os.path.join(tmp.name, "v.glp")
        glpfiles.write_gguf(p, meta([5]),
                            tensors({5: unit([1.0] + [0.0] * 7)}))
        with self.assertRaisesRegex(ValueError, "4 decoder layers"):
            glp.apply_glp(_FakeModel(n_layers=4), p)


@unittest.skipIf(torch is None, "torch/numpy not installed")
class SmokeTests(unittest.TestCase):
    """End-to-end on a real tiny model from the hub; skips when offline."""

    MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    def test_tiny_model_steering(self):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            self.skipTest("transformers not installed")
        try:
            model = AutoModelForCausalLM.from_pretrained(self.MODEL)
        except Exception as e:
            self.skipTest(f"hub unreachable: {e}")
        model.eval()
        cfg = model.config
        hidden, n_layers = cfg.hidden_size, cfg.num_hidden_layers
        # direction.N steers layer N and N >= 1, so layer 0 is unsteerable;
        # cover every layer the container can express.
        layers = list(range(1, n_layers))

        gen = torch.Generator().manual_seed(0)
        dirs = {}
        for i in layers:
            d = torch.randn(hidden, generator=gen)
            dirs[i] = (d / d.norm()).numpy().astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "v.glp")
            glpfiles.write_gguf(p, meta(layers), tensors(dirs))
            st = glp.apply_glp(model, p)

            seen = {}
            for i in dirs:
                def cap(mod, args, out, i=i):
                    t = out[0] if isinstance(out, tuple) else out
                    seen[i] = t.detach()
                model.model.layers[i].register_forward_hook(cap)

            ids = torch.tensor([[1, 2, 3, 4]])
            with torch.no_grad():
                model(ids)
            worst = 0.0
            for i, dl in dirs.items():
                dv = torch.tensor(dl)
                worst = max(worst, (seen[i] @ dv).abs().max().item())
            st.detach()
        print(f"\n  [smoke] {self.MODEL}: layers {layers} steered, "
              f"max |h.d| after projection = {worst:.3e}")
        self.assertEqual(sorted(seen), layers)
        self.assertLess(worst, 1e-4)


if __name__ == "__main__":
    unittest.main()
