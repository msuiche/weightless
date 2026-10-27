"""captain-vector recover: SVD direction recovery from a weight edit.

Synthetic rank-1 pair: W' = W - alpha*d(d^T W) with a known d. recover
must reproduce d to cos ~1 and ship a spec-valid GGUF. Requires torch +
safetensors (the same deps as bake); skipped when unavailable, since the
weightless runtime itself is stdlib-only.
"""
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "tools" / "captain-vector" / "captain_vector.py"


def _load():
    spec = importlib.util.spec_from_file_location("cv_test_recover", CV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecoverSyntheticTest(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
            from safetensors.torch import save_file  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch/safetensors unavailable")

    def test_recovers_known_rank1_direction(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base"
            edited = Path(td) / "edited"
            base.mkdir()
            edited.mkdir()
            torch.manual_seed(0)
            d = torch.randn(64)
            d = d / d.norm()
            W = torch.randn(64, 32)
            Wp = W - 1.0 * d.unsqueeze(1) * (d @ W).unsqueeze(0)
            idx = {
                "metadata": {"total_size": 64 * 32 * 4},
                "weight_map": {
                    "layers.5.attn.wo_b.weight": "model-00001-of-00001.safetensors",
                    "other.weight": "model-00001-of-00001.safetensors",
                },
            }
            for d_, tensors in ((base, W), (edited, Wp)):
                save_file(
                    {"layers.5.attn.wo_b.weight": tensors,
                     "other.weight": torch.randn(8, 8)},
                    str(d_ / "model-00001-of-00001.safetensors"))
                json.dump(idx, open(d_ / "model.safetensors.index.json", "w"))
            out = str(Path(td) / "recovered.gguf")
            pin = "deadbeef1234567890abcdef1234567890abcdef"
            r = subprocess.run(
                [sys.executable, str(CV), "recover", str(base), str(edited),
                 "--modules", "attn.wo_b", "--out", out,
                 "--base-repo", "test/synthetic", "--revision", pin],
                capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

            # read direction.6 back and compare against the known d
            import gguf
            reader = gguf.GGUFReader(out)
            rec = None
            for tensor in reader.tensors:
                if tensor.name == "direction.6":
                    rec = torch.from_numpy(tensor.data.copy()).float()
            self.assertIsNotNone(rec, "direction.6 missing from output")
            cos = float(torch.dot(rec / rec.norm(), d))
            self.assertGreater(cos, 0.9999, f"recovered cos {cos}")

            # report json ships per-tensor rank info
            report = json.load(open(out.replace(".gguf", "-report.json")))
            hit = report["layers.5.attn.wo_b.weight"]
            self.assertGreater(hit["top1_energy"], 0.999)


    def test_refuses_diffuse_edit_unless_forced(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "base"
            edited = Path(td) / "edited"
            base.mkdir()
            edited.mkdir()
            torch.manual_seed(0)
            W = torch.randn(64, 32)
            # diffuse delta: iid noise, top1_energy ~0.09, no direction
            Wp = W + torch.randn(64, 32)
            idx = {
                "metadata": {"total_size": 64 * 32 * 4},
                "weight_map": {
                    "layers.5.attn.wo_b.weight": "model-00001-of-00001.safetensors",
                },
            }
            for d_, tensors in ((base, W), (edited, Wp)):
                save_file(
                    {"layers.5.attn.wo_b.weight": tensors,
                     "other.weight": torch.randn(8, 8)},
                    str(d_ / "model-00001-of-00001.safetensors"))
                idx["weight_map"]["other.weight"] = \
                    "model-00001-of-00001.safetensors"
                json.dump(idx, open(d_ / "model.safetensors.index.json", "w"))
            out = str(Path(td) / "recovered.gguf")
            pin = "deadbeef1234567890abcdef1234567890abcdef"
            cmd = [sys.executable, str(CV), "recover", str(base), str(edited),
                   "--modules", "attn.wo_b", "--out", out,
                   "--base-repo", "test/synthetic", "--revision", pin]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, "diffuse edit must refuse")
            self.assertIn("diffuse", r.stderr)
            self.assertFalse(os.path.exists(out),
                             "refused run must not write the GLP")
            self.assertTrue(os.path.exists(out.replace(".gguf", "-report.json")),
                            "refused run must still write the rank report")

            r = subprocess.run(cmd + ["--force"], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertTrue(os.path.exists(out), "--force must write the GLP")


if __name__ == "__main__":
    unittest.main()