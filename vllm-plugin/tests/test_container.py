"""Offline tests for the GLP container reader: the spec gates.

Synthetic GGUF files, no GPU, no vllm. Port of the GgufGateTests in
tests/test_nemotron35_hotfix.py to the plugin's container module — same
gates, now against the one shared copy of the loader.
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weightless_steer import container  # noqa: E402
from glpfiles import good_meta, good_tensors, write_gguf  # noqa: E402


class GgufGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "vec.gguf")

    def load(self, meta, tensors=None, **kw):
        write_gguf(self.path, meta,
                   good_tensors() if tensors is None else tensors)
        return container.load_control_vector(self.path, **kw)

    def test_valid_vector_loads(self):
        meta, out = self.load(good_meta())
        self.assertEqual(sorted(out), [1, 2, 3])
        self.assertEqual(out[1].numel(), 8)
        self.assertEqual(out[1].dtype, torch.float32)
        self.assertEqual(meta["glp.mode"], "project")

    def test_layer_ids_crosscheck_is_informational_when_consistent(self):
        meta = good_meta()
        del meta["glp.layer_ids_zero_based"]
        self.assertEqual(sorted(self.load(meta)[1]), [1, 2, 3])

    def test_missing_mode_is_fatal(self):
        meta = good_meta()
        del meta["glp.mode"]
        with self.assertRaisesRegex(ValueError, "glp.mode"):
            self.load(meta)

    def test_additive_mode_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "glp.mode='add'"):
            self.load({**good_meta(), "glp.mode": "add"})

    def test_unknown_mode_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "glp.mode"):
            self.load({**good_meta(), "glp.mode": "multiply"})

    def test_wrong_hook_point_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "glp.hook_point"):
            self.load({**good_meta(),
                       "glp.hook_point": "ffn_out_pre_residual"})

    def test_missing_hook_point_is_fatal(self):
        meta = good_meta()
        del meta["glp.hook_point"]
        with self.assertRaisesRegex(ValueError, "glp.hook_point"):
            self.load(meta)

    def test_hook_argument_is_enforced(self):
        # A reader applying at another site must refuse this vector.
        with self.assertRaisesRegex(ValueError, "glp.hook_point"):
            self.load(good_meta(), hook="ffn_out_pre_residual")

    def test_transferred_vector_warns_but_loads(self):
        meta = {**good_meta(), "glp.derived_at": "ffn_out_pre_residual"}
        with self.assertLogs("weightless_steer.container", level="WARNING"):
            self.assertEqual(sorted(self.load(meta)[1]), [1, 2, 3])

    def test_direction_zero_is_rejected(self):
        tensors = good_tensors()
        tensors["direction.0"] = (np.zeros(8, dtype=np.float32), 0)
        with self.assertRaisesRegex(ValueError, "direction.0"):
            self.load(good_meta(), tensors)

    def test_layer_ids_mismatch_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.load({**good_meta(), "glp.layer_ids_zero_based": "0,1,2"})

    def test_non_f32_tensor_is_fatal(self):
        tensors = good_tensors()
        tensors["direction.2"] = (tensors["direction.2"][0], 1)
        with self.assertRaisesRegex(ValueError, "not F32"):
            self.load(good_meta(), tensors)

    def test_malformed_tensor_name_is_fatal(self):
        tensors = good_tensors()
        tensors["direction.x"] = (np.zeros(8, dtype=np.float32), 0)
        with self.assertRaisesRegex(ValueError, "malformed tensor name"):
            self.load(good_meta(), tensors)

    def test_not_a_gguf_is_fatal(self):
        with open(self.path, "wb") as f:
            f.write(b"NOPE" + b"\0" * 64)
        with self.assertRaisesRegex(ValueError, "not a GGUF"):
            container.load_control_vector(self.path)

    def test_no_direction_tensors_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "no direction"):
            self.load(good_meta(), {})


def _e(i, width=8):
    v = np.zeros(width, dtype=np.float32)
    v[i] = 1.0
    return v


def rank2_meta(layers=(1, 2), **overrides):
    m = good_meta(layers=layers)
    m.update({"glp.spec_version": "2", "glp.rank": "2",
              "glp.orthonormal": "true", "glp.dir_scales": "1.0,0.5"})
    m.update(overrides)
    return m


def rank2_tensors(layers=(1, 2), j1=None):
    t = {}
    for i in layers:
        t[f"direction.{i}"] = (_e(0), 0)
        t[f"direction.{i}.1"] = ((_e(1) if j1 is None else j1), 0)
    return t


class Rank2GateTests(unittest.TestCase):
    """The rank-k (subspace) gates: version, contiguity, rank, orthonormal."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "vec.gguf")

    def load(self, meta, tensors):
        write_gguf(self.path, meta, tensors)
        return container.load_control_vector(self.path)

    def test_rank2_loads_stacked(self):
        meta, out = self.load(rank2_meta(), rank2_tensors())
        self.assertEqual(tuple(out[1].shape), (2, 8))
        self.assertEqual(out[1].dtype, torch.float32)
        self.assertTrue(torch.allclose(out[1][0], torch.from_numpy(_e(0))))
        self.assertTrue(torch.allclose(out[1][1], torch.from_numpy(_e(1))))

    def test_rank1_still_loads_1d(self):
        _, out = self.load(good_meta(), good_tensors())
        self.assertEqual(out[1].dim(), 1)

    def test_unimplemented_spec_version_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "spec_version"):
            self.load(rank2_meta(**{"glp.spec_version": "3"}),
                      rank2_tensors())

    def test_rank2_under_version_1_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "spec_version 2"):
            self.load(rank2_meta(**{"glp.spec_version": "1"}),
                      rank2_tensors())

    def test_scale_keys_under_version_1_are_fatal(self):
        meta = {**good_meta(), "glp.dir_scales": "1.0"}
        with self.assertRaisesRegex(ValueError, "spec_version 2"):
            self.load(meta, good_tensors())

    def test_missing_direction_is_fatal(self):
        tensors = rank2_tensors()
        del tensors["direction.2.1"]
        with self.assertRaisesRegex(ValueError, "direction count"):
            self.load(rank2_meta(), tensors)

    def test_non_contiguous_indices_are_fatal(self):
        tensors = rank2_tensors()
        tensors["direction.1.2"] = tensors.pop("direction.1.1")
        with self.assertRaisesRegex(ValueError, "0..k-1"):
            self.load(rank2_meta(), tensors)

    def test_rank_mismatch_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "glp.rank declares"):
            self.load(rank2_meta(**{"glp.rank": "3"}), rank2_tensors())

    def test_orthonormal_claim_is_required(self):
        meta = rank2_meta()
        del meta["glp.orthonormal"]
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            self.load(meta, rank2_tensors())

    def test_non_orthonormal_basis_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "not orthonormal"):
            self.load(rank2_meta(), rank2_tensors(j1=_e(0)))


if __name__ == "__main__":
    unittest.main()
