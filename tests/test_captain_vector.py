"""Runs the captain-vector self-test script (tools/captain-vector/test_captain_vector.py).

The real test is a standalone script, not a unittest module, so it stays
runnable on its own and from refusal-research's shim; this wrapper folds it
into the unittest suite. The script is torch-optional: the stdlib-only
sections (GGUF validate/inspect/export) always run, and the torch-dependent
sections skip gracefully when torch is absent.
"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "captain-vector" / "test_captain_vector.py"


class CaptainVectorTests(unittest.TestCase):
    def test_self_test_script_passes(self):
        r = subprocess.run([sys.executable, str(SCRIPT)],
                           capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        self.assertEqual(r.returncode, 0,
                         f"captain-vector self-test failed:\n{r.stdout}\n{r.stderr}")


if __name__ == "__main__":
    unittest.main()
