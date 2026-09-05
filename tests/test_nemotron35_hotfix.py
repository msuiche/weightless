"""Offline tests for the nemotron_h GLP steering hotfix.

Covers, without Docker or a GPU:
  - the hotfix applies cleanly to the vendored v0.28.0 nemotron_h.py
    reference (patches/reference/nemotron_h_v0280.py, extracted from the
    vllm/vllm-openai:v0.28.0 image on the DGX Spark), is idempotent, and
    the patched source still compiles;
  - anchor drift fails closed when steering is requested and is a no-op
    otherwise;
  - the GGUF reader enforces the glp.* metadata gates (mode=project,
    hook_point=residual_stream_post_layer, direction.0 rejected,
    layer_ids_zero_based cross-check) using synthetic GGUF files.
"""
import importlib.util
import os
from pathlib import Path
import py_compile
import struct
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HOTFIX_PATH = ROOT / "patches" / "hotfix-nemotron35-steering-projective.py"
REFERENCE = ROOT / "patches" / "reference" / "nemotron_h_v0280.py"

spec = importlib.util.spec_from_file_location("nemotron35_hotfix", HOTFIX_PATH)
hotfix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hotfix)


def write_gguf(path: Path, meta: dict, tensors: dict) -> None:
    """Write a minimal GGUF v3 file (string metadata, 1-D tensors).

    tensors: {name: (numpy float32 array, ggml_type)} — ggml_type 0 is F32.
    """
    out = bytearray()

    def w_str(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    out += b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        out += w_str(k) + struct.pack("<I", 8) + w_str(v)  # string metadata
    blobs = []
    off = 0
    for name, (arr, gtype) in tensors.items():
        data = np.asarray(arr, dtype="<f4").tobytes()
        out += w_str(name)
        out += struct.pack("<I", 1) + struct.pack("<Q", arr.size)
        out += struct.pack("<I", gtype) + struct.pack("<Q", off)
        blobs.append(data)
        off += (len(data) + 31) // 32 * 32
    out += b"\0" * ((-len(out)) % 32)  # data base alignment
    for data in blobs:
        out += data
        out += b"\0" * ((-len(data)) % 32)
    path.write_bytes(bytes(out))


class _FakeTensor:
    def __init__(self, arr):
        self.arr = arr

    def numel(self):
        return self.arr.size


def load_vector(path: Path):
    """Run the hotfix's injected GGUF loader with a torch stub."""
    ns = {
        "torch": types.SimpleNamespace(from_numpy=lambda a: _FakeTensor(a)),
        "logger": types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        ),
    }
    exec(hotfix.GGUF_SRC, ns)  # noqa: S102 - the code under test
    return ns["_load_gguf_control_vector"](str(path))


GOOD_META = {
    "glp.mode": "project",
    "glp.hook_point": "residual_stream_post_layer",
    "glp.derived_at": "residual_stream_post_layer",
    "glp.spec_version": "1.0",
    "glp.alpha_default": "1.0",
    "glp.layer_ids_zero_based": "1,2,3",
}
GOOD_TENSORS = {
    f"direction.{i}": (np.full(8, float(i), dtype=np.float32), 0)
    for i in (1, 2, 3)
}


class HotfixApplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = Path(self.tmp.name) / "nemotron_h.py"
        self.target.write_text(REFERENCE.read_text())

    def run_hotfix(self, **env):
        return subprocess.run(
            [sys.executable, str(HOTFIX_PATH)],
            env={**os.environ, "WEIGHTLESS_STEERING_MODEL_PY": str(self.target),
                 **env},
            capture_output=True, text=True,
        )

    def test_applies_to_v0280_reference_and_compiles(self):
        result = self.run_hotfix()
        self.assertEqual(result.returncode, 0, result.stderr)
        src = self.target.read_text()
        self.assertIn(hotfix.MARK, src)
        # Residual-site math: steer h = hidden_states + residual, write back
        # hidden_states <- h' - residual.
        self.assertIn("_steer_h = hidden_states + residual", src)
        self.assertIn(")) - residual", src)
        py_compile.compile(str(self.target), doraise=True)

    def test_idempotent_reapply(self):
        self.assertEqual(self.run_hotfix().returncode, 0)
        once = self.target.read_text()
        result = self.run_hotfix()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already applied", result.stdout)
        self.assertEqual(self.target.read_text(), once)

    def test_anchor_drift_fails_closed_when_steering_requested(self):
        # Simulate image drift: the forward-loop anchor is gone.
        self.target.write_text(self.target.read_text().replace(
            hotfix.ANCHOR_FORWARD, "", 1))
        glp = Path(self.tmp.name) / "v.gguf"
        glp.write_bytes(b"GGUF")
        result = self.run_hotfix(WEIGHTLESS_STEER_PATH=str(glp))
        self.assertEqual(result.returncode, 1)
        self.assertIn("failing closed", result.stderr)
        self.assertNotIn(hotfix.MARK, self.target.read_text())

    def test_anchor_drift_without_steering_is_a_noop(self):
        original = self.target.read_text().replace(hotfix.ANCHOR_FORWARD, "", 1)
        self.target.write_text(original)
        result = self.run_hotfix()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("leaving nemotron_h.py stock", result.stdout)
        self.assertEqual(self.target.read_text(), original)


class GgufGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "vec.gguf"

    def load(self, meta, tensors=GOOD_TENSORS):
        write_gguf(self.path, meta, tensors)
        return load_vector(self.path)

    def test_valid_vector_loads(self):
        out = self.load(GOOD_META)
        self.assertEqual(sorted(out), [1, 2, 3])
        self.assertEqual(out[1].numel(), 8)

    def test_layer_ids_crosscheck_is_informational_when_consistent(self):
        meta = dict(GOOD_META)
        del meta["glp.layer_ids_zero_based"]
        self.assertEqual(sorted(self.load(meta)), [1, 2, 3])

    def test_missing_mode_is_fatal(self):
        meta = dict(GOOD_META)
        del meta["glp.mode"]
        with self.assertRaisesRegex(ValueError, "glp.mode"):
            self.load(meta)

    def test_additive_mode_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "glp.mode='add'"):
            self.load({**GOOD_META, "glp.mode": "add"})

    def test_wrong_hook_point_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "glp.hook_point"):
            self.load({**GOOD_META,
                       "glp.hook_point": "ffn_out_pre_residual"})

    def test_missing_hook_point_is_fatal(self):
        meta = dict(GOOD_META)
        del meta["glp.hook_point"]
        with self.assertRaisesRegex(ValueError, "glp.hook_point"):
            self.load(meta)

    def test_direction_zero_is_rejected(self):
        tensors = dict(GOOD_TENSORS)
        tensors["direction.0"] = (np.zeros(8, dtype=np.float32), 0)
        with self.assertRaisesRegex(ValueError, "direction.0"):
            self.load(GOOD_META, tensors)

    def test_layer_ids_mismatch_is_fatal(self):
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.load({**GOOD_META, "glp.layer_ids_zero_based": "0,1,2"})

    def test_non_f32_tensor_is_fatal(self):
        tensors = dict(GOOD_TENSORS)
        tensors["direction.2"] = (GOOD_TENSORS["direction.2"][0], 1)
        with self.assertRaisesRegex(ValueError, "not F32"):
            self.load(GOOD_META, tensors)

    def test_malformed_tensor_name_is_fatal(self):
        tensors = dict(GOOD_TENSORS)
        tensors["direction.x"] = (np.zeros(8, dtype=np.float32), 0)
        with self.assertRaisesRegex(ValueError, "malformed tensor name"):
            self.load(GOOD_META, tensors)


if __name__ == "__main__":
    unittest.main()
