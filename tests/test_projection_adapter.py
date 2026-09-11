"""Runs the projection-adapter reference demo (tools/captain-vector/examples/projection_adapter.py).

The demo is a standalone script (companion to docs/peft-contribution-draft.md),
kept runnable on its own; this wrapper folds it into the unittest suite. It
needs torch + peft, so it skips cleanly where those are absent -- the same
pattern as the captain-vector self-test wrapper.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "captain-vector" / "examples" / "projection_adapter.py"

try:
    import peft  # noqa: F401
    import torch  # noqa: F401
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


class ProjectionAdapterTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_DEPS, "torch/peft not installed")
    def test_reference_demo_passes(self):
        r = subprocess.run([sys.executable, str(SCRIPT)],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        self.assertEqual(r.returncode, 0,
                         f"projection-adapter demo failed:\n{r.stdout}\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
