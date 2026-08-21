#!/usr/bin/env python3
"""Structural checks on the Qwen3.5/3.8 steering hotfix.

Same reason the DSV4 structure test exists: the per-layer assignment once got
dedented out of its loop on that lane, and the server steered one layer while
reporting all of them. Coverage is the variable that dominates this
intervention; nothing in the logs distinguishes the two cases.

This test applies patches/hotfix-qwen38-steering-projective.py to a SCRATCH
COPY of a reference qwen3_next.py (never the original) and AST-checks the
result:

  1. all four anchors match and the patched file still parses;
  2. the per-layer assignments are INSIDE the per-layer loop;
  3. the forward apply steers hidden_states + residual, not hidden_states
     alone (vLLM's decomposed convention — see the hotfix docstring);
  4. re-applying is a no-op, and anchors-missing fails closed when
     QWEN_STEER_PATH is set.

Run: python3 scripts/test-qwen-steering-structure.py [path/to/qwen3_next.py]
Default path is the vllm checkout beside this repo in the spark workspace.
No GPU, no torch, no vLLM import -- this runs the hotfix and parses source.
"""

from __future__ import annotations

import ast
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
HOTFIX = REPO / "patches/hotfix-qwen38-steering-projective.py"
DEFAULT_MODEL = (
    REPO.parent / "vllm/vllm/model_executor/models/qwen3_next.py"
)

PER_LAYER_TARGETS = ("self._steer_dirs[layer_id]", "_QWEN_HOOK_DIRS[layer_id]")


def run_hotfix(model_py: pathlib.Path, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["QWEN_STEERING_MODEL_PY"] = str(model_py)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(HOTFIX)], env=env, capture_output=True, text=True
    )


def find_load_loop(tree: ast.AST) -> ast.For | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "layer_id"
        ):
            return node
    return None


def main(reference: pathlib.Path) -> int:
    if not reference.is_file():
        print(f"  [SKIP] reference qwen3_next.py not found at {reference}")
        print("         pass one explicitly: test-qwen-steering-structure.py <path>")
        return 2

    failures = 0

    def check(ok: bool, label: str, detail: str = ""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
        failures += 0 if ok else 1

    with tempfile.TemporaryDirectory() as td:
        scratch = pathlib.Path(td) / "qwen3_next.py"
        shutil.copy(reference, scratch)

        # 1. applies + parses
        r = run_hotfix(scratch)
        check(r.returncode == 0 and "applied" in r.stdout, "hotfix applies (4 anchors)", r.stderr.strip())
        src = scratch.read_text()
        try:
            tree = ast.parse(src)
            check(True, "patched file parses")
        except SyntaxError as exc:
            check(False, "patched file parses", str(exc))
            return failures

        # 2. per-layer assignments inside the loop
        loop = find_load_loop(tree)
        check(loop is not None, "per-layer loop found in _load_steering")
        if loop is not None:
            body = ast.get_source_segment(src, loop) or ""
            for target in PER_LAYER_TARGETS:
                check(target in body, f"{target} inside the per-layer loop")

        # 3. forward apply steers the full residual stream
        check(
            "steer_stream = hidden_states + residual" in src,
            "forward apply steers hidden_states + residual (decomposed convention)",
        )
        check(
            "self._steer_stack[layer_idx]" in src,
            "forward apply indexes the steer stack by global layer id",
        )

        # 4a. idempotent
        r2 = run_hotfix(scratch)
        check(r2.returncode == 0 and "already applied" in r2.stdout, "re-apply is a no-op")

        # 4b. fail-closed: steering requested but anchors missing
        bogus = pathlib.Path(td) / "bogus.py"
        bogus.write_text("# not qwen3_next.py\n")
        r3 = run_hotfix(bogus, {"QWEN_STEER_PATH": "/nonexistent.gguf"})
        check(r3.returncode == 1, "anchors missing + QWEN_STEER_PATH set fails closed")
        r4 = run_hotfix(bogus)
        check(r4.returncode == 0, "anchors missing + steering off stays stock")

    print()
    print("qwen steering structure: " + ("all checks passed" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    sys.exit(main(target))
