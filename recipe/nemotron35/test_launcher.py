"""Exercise the launcher without starting Docker or using a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / "config"
        self.config.write_text((HERE / ".env.nemotron35.example").read_text()
            + f'\nHF_CACHE="{self.root}/cache with spaces"\n')
        docker = self.root / "docker"
        docker.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
            "with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
            "if sys.argv[1:3] == ['container','inspect']: sys.exit(int(os.environ.get('INSPECT_EXIT','1')))\n")
        docker.chmod(0o755)
        self.calls = self.root / "calls"
        self.env = dict(os.environ, NEMOTRON_ENV_FILE=str(self.config),
            PATH=str(self.root) + os.pathsep + os.environ["PATH"], CALLS=str(self.calls))

    def run_launcher(self, *args):
        return subprocess.run(["bash", str(HERE / "serve-nemotron35.sh"), *args],
                              env=self.env, capture_output=True, text=True)

    def test_dry_run_has_no_docker_or_cache_side_effect(self):
        result = self.run_launcher("--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.root / "cache with spaces").exists())
        self.assertIn("qwen3_coder", result.stdout)

    def test_existing_container_is_not_replaced(self):
        self.env["INSPECT_EXIT"] = "0"
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.calls.read_text().splitlines()), 1)

    def test_stock_target_launch_preserves_arguments(self):
        with self.config.open("a") as f:
            f.write('\nSPECULATIVE_MODE=none\nSERVED_MODEL_NAME="name with spaces"\n')
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.calls.read_text().splitlines()[-1])
        self.assertEqual(args[0], "run")
        self.assertEqual(args[args.index("--served-model-name") + 1], "name with spaces")
        self.assertNotIn("--speculative_config.model", args)
        self.assertEqual(args[args.index("--mount") + 1],
            f"type=bind,src={self.root}/cache with spaces,dst=/root/.cache/huggingface")

    def test_invalid_mode_fails_before_docker(self):
        with self.config.open("a") as f:
            f.write('\nSPECULATIVE_MODE=typo\n')
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
