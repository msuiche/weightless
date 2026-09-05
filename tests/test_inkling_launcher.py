"""Exercise the real launcher with isolated Docker/SSH/hardware stand-ins."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "recipe/inkling/start-inkling-sm121.sh"
MOCK = r'''#!/usr/bin/env python3
import hashlib, json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "ssh":
    env = dict(os.environ, MOCK_WORKER="1")
    sys.exit(subprocess.run(["bash", "-c", args[-1]], env=env).returncode)
elif name == "sha256sum":
    value = hashlib.sha256(pathlib.Path(args[0]).read_bytes()).hexdigest()
    if os.environ.get("MOCK_WORKER") and os.environ.get("MISMATCH") == args[0]:
        value = "0" * 64
    print(value + "  " + args[0])
elif name == "awk":
    if "/proc/meminfo" in args:
        print(121)
    else:
        sys.exit(subprocess.run(["/usr/bin/awk", *args]).returncode)
elif name == "docker":
    with open(os.environ["DOCKER_LOG"], "a") as f:
        f.write(json.dumps({"worker": bool(os.environ.get("MOCK_WORKER")), "args": args}) + "\n")
    if args[0] == "run":
        print("mock-container-id")
elif name == "sudo":
    sys.stdin.read()
elif name in ("sync", "nvidia-smi", "curl"):
    pass
else:
    sys.exit("Unexpected mock command: " + name)
'''


class InklingLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("ssh", "sha256sum", "awk", "docker", "sudo", "sync", "nvidia-smi", "curl"):
            p = self.bin / name
            p.write_text(MOCK)
            p.chmod(0o755)
        self.vector = self.root / "vector.gguf"
        self.vector.write_bytes(b"test vector")
        self.model = self.root / "model.py"
        self.model.write_text("# gb10-load-reclaim-hotfix\n# [steering-hotfix] projective activation steering (Inkling)\n")
        self.fa4 = self.root / "fa4.py"
        self.fa4.write_text("# sm121-relattn-hotfix\n")
        self.log = self.root / "docker.jsonl"
        self.env_file = self.root / ".env.inkling"
        self.settings = dict(MASTER_ADDR="head", WORKER_HOST="worker", HF_CACHE=str(self.root),
                             INKLING_IMAGE="test-image", MODEL="test-model", NCCL_IB_HCA="test-hca",
                             NCCL_SOCKET_IFNAME="test-iface", MODEL_PATCHED_PY=str(self.model),
                             FA4_PATCHED_PY=str(self.fa4), WEIGHTLESS_STEER_PATH=str(self.vector),
                             WEIGHTLESS_STEER_ALPHA="0.25")

    def run_launcher(self, **overrides):
        values = dict(self.settings, **overrides)
        self.env_file.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                   ENV_FILE=str(self.env_file), DOCKER_LOG=str(self.log))
        env.pop("MOCK_WORKER", None)
        return subprocess.run(["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, timeout=20)

    def test_both_ranks_receive_vector_alpha_and_combined_patch(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        runs = [json.loads(line) for line in self.log.read_text().splitlines() if json.loads(line)["args"][0] == "run"]
        self.assertEqual([r["worker"] for r in runs], [True, False])
        for run in runs:
            args = run["args"]
            self.assertIn("WEIGHTLESS_STEER_PATH=/opt/weightless/steering.gguf", args)
            self.assertIn("WEIGHTLESS_STEER_ALPHA=0.25", args)
            self.assertIn(str(self.vector) + ":/opt/weightless/steering.gguf:ro", args)
            self.assertIn(str(self.model) + ":/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/model.py:ro", args)
            self.assertIn("--tool-call-parser inkling", args[-1])

    def test_unpatched_model_fails_before_any_docker_action(self):
        self.model.write_text("# gb10-load-reclaim-hotfix\n")
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lacks the Inkling steering patch", result.stderr)
        self.assertFalse(self.log.exists())

    def test_missing_vector_fails_before_any_docker_action(self):
        result = self.run_launcher(WEIGHTLESS_STEER_PATH=str(self.root / "missing.gguf"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_mismatched_worker_vector_fails_before_any_docker_action(self):
        with patch.dict(os.environ, {"MISMATCH": str(self.vector)}):
            result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Steering vector differs on worker", result.stderr)
        self.assertFalse(self.log.exists())

    def test_uncalibrated_alpha_fails_before_any_docker_action(self):
        result = self.run_launcher(WEIGHTLESS_STEER_ALPHA="0.5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("0 < alpha <= 0.25", result.stderr)
        self.assertFalse(self.log.exists())

    def test_empty_path_disables_steering_environment(self):
        result = self.run_launcher(WEIGHTLESS_STEER_PATH="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("WEIGHTLESS_STEER_PATH=", self.log.read_text())

    def test_empty_alpha_uses_sm121_default(self):
        result = self.run_launcher(WEIGHTLESS_STEER_ALPHA="")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WEIGHTLESS_STEER_ALPHA=0.1", self.log.read_text())


if __name__ == "__main__":
    unittest.main()
