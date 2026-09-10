#!/usr/bin/env python3
"""Structural checks on the Muse-Glimmer-30B lane wiring.

Muse-Glimmer is the first stock-only lane: stock vLLM 0.28.0 serves the
muse_glimmer arch, the ATEM tool parser, the reasoning parser, and the
DFlash block-diffusion drafter natively, so there is no hotfix to AST-check.
What can regress silently instead is the lane wiring itself — a renamed env
key, a drifted revision pin, a spec-config typo, a setup.py LANES entry that
no longer matches the recipe on disk. This test checks all of that without a
GPU or Docker:

  1. the recipe files exist and the env example carries every key the serve
     script requires (the script's `: "${VAR:?}"` guards are the contract);
  2. the setup.py lane entry agrees with the recipe (image, repos, revision
     pins, port, container name, recipe files) and is registered in
     DEPLOY_MAP / CONTAINER_GREP;
  3. the serve script passes bash -n and its --dry-run renders the
     card-tested flags: muse_glimmer tool + reasoning parsers, and a valid
     --speculative-config JSON with method dflash, the pinned drafter repo,
     and the 16-token block;
  4. SPECULATIVE_MODE=none drops the spec config, and an invalid mode fails
     before docker is ever invoked;
  5. the omp provider template (tests/smoke/models.yml) lists the served
     model on the lane's port.

Run: python3 tests/structure/test-museglimmer-structure.py
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
RECIPE = REPO / "recipe/museglimmer"
EXAMPLE = RECIPE / ".env.museglimmer.example"
SERVE = RECIPE / "serve-museglimmer.sh"

# Ground truth checked on Hugging Face 2026-09-09.
MODEL_REPO = "nvidia/Muse-Glimmer-30B-NVFP4"
MODEL_REVISION = "f45fad5689e9a4d937f7e872fbec20c4e8a74154"
DRAFTER_REPO = "meta-models/Muse-Glimmer-30B-assistant"
DRAFTER_REVISION = "e8192f3a8f617f74be2ce220360c89ef4789f39f"
IMAGE = "vllm/vllm-openai:v0.28.0"
PORT = "8084"
CONTAINER = "museglimmer"
SERVED = "muse-glimmer-nvfp4"

REQUIRED_KEYS = ["MUSE_IMAGE", "MODEL_ID", "MODEL_REVISION", "HF_CACHE",
                 "CONTAINER_NAME", "VLLM_PORT", "SERVED_MODEL_NAME",
                 "MAX_MODEL_LEN", "GPU_MEMORY_UTILIZATION", "MAX_NUM_SEQS"]


def load_setup():
    import importlib.util
    spec = importlib.util.spec_from_file_location("wizard", REPO / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example_env():
    env = {}
    for line in EXAMPLE.read_text().splitlines():
        m = re.match(r"^([A-Z0-9_]+)=(.*)$", line)
        if m:
            env[m.group(1)] = m.group(2)
    return env


def dry_run(extra=""):
    """Render the launcher with a stub docker; --dry-run never invokes it."""
    with tempfile.TemporaryDirectory() as td:
        config = pathlib.Path(td) / "config"
        config.write_text(EXAMPLE.read_text().replace("<user>", "tester")
                          + f'\nHF_CACHE="{td}/cache with spaces"\n' + extra)
        r = subprocess.run(["bash", str(SERVE), "--dry-run"],
                           env=dict(os.environ, MUSE_ENV_FILE=str(config)),
                           capture_output=True, text=True)
        return r


def main() -> int:
    failures = 0

    def check(ok, label, detail=""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail and not ok else ""))
        failures += 0 if ok else 1

    # 1. recipe files + env contract
    check(EXAMPLE.is_file(), "env example ships")
    check(SERVE.is_file(), "serve script ships")
    check((RECIPE / "README.md").is_file(), "recipe README ships")
    env = example_env()
    for key in REQUIRED_KEYS:
        check(key in env and env[key], f"env example sets {key}")
    check(env.get("MODEL_ID") == MODEL_REPO, "env pins the NVFP4 repo")
    check(env.get("MODEL_REVISION") == MODEL_REVISION,
          "env pins the reviewed model revision")
    check(env.get("DRAFTER_MODEL_ID") == DRAFTER_REPO, "env pins the drafter repo")
    check(env.get("DRAFTER_REVISION") == DRAFTER_REVISION,
          "env pins the reviewed drafter revision")
    check(env.get("VLLM_PORT") == PORT, "env port is the lane port")
    check(env.get("CONTAINER_NAME") == CONTAINER, "env container name is the lane key")
    check(env.get("SERVED_MODEL_NAME") == SERVED, "env served model name")
    check("<user>" in EXAMPLE.read_text(), "env keeps the <user> placeholder")

    # 2. setup.py lane wiring
    setup = load_setup()
    lane = next((l for l in setup.LANES if l.get("example", "").startswith(
        "recipe/museglimmer/")), None)
    check(lane is not None, "setup.py LANES carries the museglimmer lane")
    if lane is not None:
        idx = setup.LANES.index(lane)
        check(lane["model_repo"] == MODEL_REPO, "lane model_repo matches the env")
        check(lane["docker_image"] == env.get("MUSE_IMAGE") == IMAGE,
              "lane docker_image == env MUSE_IMAGE")
        check(lane.get("image_key") == "MUSE_IMAGE", "lane image_key is MUSE_IMAGE")
        check(lane.get("nodes") == 1 and lane.get("remote_dir") == CONTAINER,
              "lane is single-node with its own remote_dir")
        check(str(lane.get("port")) == PORT, "lane port matches the env")
        check(lane.get("steering_supported") is False,
              "lane declines steering (no GLP vector exists)")
        check(sorted(lane.get("recipe_files", [])) ==
              sorted([".env.museglimmer", "serve-museglimmer.sh"]),
              "lane recipe_files match the recipe dir")
        deploy = setup.DEPLOY_MAP.get(idx)
        check(deploy is not None, "DEPLOY_MAP covers the lane")
        for local, _remote in deploy or []:
            # The generated .env target only exists after the wizard writes
            # it; everything else must ship in the repo.
            check((REPO / local).is_file() or local == lane["target"],
                  f"DEPLOY_MAP local exists: {local}")
        check(setup.CONTAINER_GREP.get(idx) == CONTAINER,
              "CONTAINER_GREP tracks the container name")

    # 3. serve script: parses + dry-run renders the card-tested flags
    r = subprocess.run(["bash", "-n", str(SERVE)], capture_output=True, text=True)
    check(r.returncode == 0, "serve script parses (bash -n)", r.stderr)
    r = dry_run()
    check(r.returncode == 0, "dry-run renders", r.stderr)
    words = shlex.split(r.stdout)
    check(words[:2] == ["docker", "run"], "dry-run emits a docker run command")
    check("muse_glimmer" == words[words.index("--tool-call-parser") + 1]
          if "--tool-call-parser" in words else False,
          "ATEM tool-call parser wired")
    check("muse_glimmer" == words[words.index("--reasoning-parser") + 1]
          if "--reasoning-parser" in words else False,
          "reasoning parser wired")
    check("--enable-auto-tool-choice" in words, "auto tool choice enabled")
    spec = None
    if "--speculative-config" in words:
        try:
            spec = json.loads(words[words.index("--speculative-config") + 1])
        except json.JSONDecodeError as exc:
            check(False, "speculative-config is valid JSON", str(exc))
    else:
        check(False, "dflash mode emits --speculative-config")
    if spec is not None:
        check(spec.get("method") == "dflash", "spec method is dflash")
        check(spec.get("model") == DRAFTER_REPO, "spec model is the drafter repo")
        check(spec.get("revision") == DRAFTER_REVISION, "spec pins the drafter revision")
        check(spec.get("num_speculative_tokens") == 16,
              "spec k matches the drafter's 16-token block")

    # 4. spec-mode gating
    r = dry_run("\nSPECULATIVE_MODE=none\n")
    check(r.returncode == 0 and "--speculative-config" not in r.stdout,
          "SPECULATIVE_MODE=none drops the drafter", r.stderr)
    r = dry_run("\nSPECULATIVE_MODE=typo\n")
    check(r.returncode != 0 and "dflash or none" in r.stderr,
          "invalid SPECULATIVE_MODE fails before docker")

    # 5. omp provider template lists the served model on the lane port
    yml = (REPO / "tests/smoke/models.yml").read_text()
    check(f"- id: {SERVED}" in yml, "models.yml lists the served model")
    check(f":{PORT}/v1" in yml, "models.yml routes the lane port")

    print()
    print("museglimmer lane structure: "
          + ("all checks passed" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
