#!/usr/bin/env python3
"""Structural checks on the Qwen3.5/3.8 steering hotfix.

Two production lessons live here:

1. The per-layer assignment once got dedented out of its loop on the DSV4
   lane; the server steered one layer while reporting all of them.
2. The first version of this hotfix patched only qwen3_next.py. But
   Qwen3_5Model.__init__ OVERRIDES Qwen3NextModel.__init__ and skips it
   (super(Qwen3NextModel, self).__init__()), so the steering buffers never
   existed on the class that actually serves, and the first boot died at
   torch.compile: "'Qwen3_5Model' object has no attribute '_steer_stack'".
   Anchors matching is NOT semantics being right.

This test applies patches/hotfix-qwen38-steering-projective.py to SCRATCH
COPIES of reference qwen3_next.py + qwen3_5.py (never the originals) and
AST-checks the result:

  1. all anchors match in both files and the patched files still parse;
  2. the per-layer assignments are INSIDE the per-layer loop;
  3. the forward apply steers hidden_states + residual, not hidden_states
     alone (vLLM's decomposed convention);
  4. Qwen3_5Model.__init__ registers the steering buffers (the lesson-2
     regression guard);
  5. re-applying is a no-op, and anchors-missing fails closed when
     WEIGHTLESS_STEER_PATH is set.

Run: python3 scripts/test-qwen-steering-structure.py [models_dir]
Default is the vllm checkout beside this repo in the spark workspace.
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
DEFAULT_MODELS = REPO.parent / "vllm/vllm/model_executor/models"

PER_LAYER_TARGETS = ("self._steer_dirs[layer_id]", "_GLP_HOOK_DIRS[layer_id]")


def run_hotfix(next_py: pathlib.Path, p35_py: pathlib.Path,
               env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["WEIGHTLESS_STEERING_MODEL_PY"] = str(next_py)
    env["WEIGHTLESS_STEERING_MODEL_PY_35"] = str(p35_py)
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


def find_class_init(tree: ast.AST, class_name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
    return None


def main(models_dir: pathlib.Path) -> int:
    ref_next = models_dir / "qwen3_next.py"
    ref_35 = models_dir / "qwen3_5.py"
    for ref in (ref_next, ref_35):
        if not ref.is_file():
            print(f"  [SKIP] reference file not found: {ref}")
            print("         pass the models dir explicitly: test-qwen-steering-structure.py <dir>")
            return 2

    failures = 0

    def check(ok: bool, label: str, detail: str = ""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
        failures += 0 if ok else 1

    with tempfile.TemporaryDirectory() as td:
        scratch_next = pathlib.Path(td) / "qwen3_next.py"
        scratch_35 = pathlib.Path(td) / "qwen3_5.py"
        shutil.copy(ref_next, scratch_next)
        shutil.copy(ref_35, scratch_35)

        # 1. applies to both files + both parse
        r = run_hotfix(scratch_next, scratch_35)
        check(
            r.returncode == 0 and r.stdout.count("applied to") == 2,
            "hotfix applies to both files",
            r.stdout + r.stderr,
        )
        trees = {}
        for label, scratch in (("qwen3_next.py", scratch_next), ("qwen3_5.py", scratch_35)):
            src = scratch.read_text()
            try:
                trees[label] = (ast.parse(src), src)
                check(True, f"patched {label} parses")
            except SyntaxError as exc:
                check(False, f"patched {label} parses", str(exc))
        if len(trees) != 2:
            return failures
        tree_next, src_next = trees["qwen3_next.py"]
        tree_35, src_35 = trees["qwen3_5.py"]

        # 2. per-layer assignments inside the loop
        loop = find_load_loop(tree_next)
        check(loop is not None, "per-layer loop found in _load_steering")
        if loop is not None:
            body = ast.get_source_segment(src_next, loop) or ""
            for target in PER_LAYER_TARGETS:
                check(target in body, f"{target} inside the per-layer loop")

        # 3. forward apply steers the full residual stream
        check(
            "steer_stream = hidden_states + residual" in src_next,
            "forward apply steers hidden_states + residual (decomposed convention)",
        )
        check(
            "self._steer_stack[layer_idx]" in src_next,
            "forward apply indexes the steer stack by global layer id",
        )

        # 4. the lesson-2 regression guard: Qwen3_5Model.__init__ registers
        #    the buffers, because it skips Qwen3NextModel.__init__ entirely.
        init35 = find_class_init(tree_35, "Qwen3_5Model")
        check(init35 is not None, "Qwen3_5Model.__init__ found in qwen3_5.py")
        if init35 is not None:
            seg = ast.get_source_segment(src_35, init35) or ""
            check(
                '"_steer_stack"' in seg and "_load_steering" in seg,
                "Qwen3_5Model.__init__ registers the steering buffers",
            )
        init_next = find_class_init(tree_next, "Qwen3NextModel")
        seg_next = ast.get_source_segment(src_next, init_next) if init_next else ""
        check(
            bool(seg_next) and '"_steer_stack"' in seg_next,
            "Qwen3NextModel.__init__ registers the steering buffers",
        )

        # 5a. idempotent on both files
        r2 = run_hotfix(scratch_next, scratch_35)
        check(
            r2.returncode == 0 and r2.stdout.count("already applied") == 2,
            "re-apply is a no-op on both files",
            r2.stdout + r2.stderr,
        )

        # 5b. fail-closed: steering requested but anchors missing
        bogus = pathlib.Path(td) / "bogus.py"
        bogus.write_text("# not a model file\n")
        r3 = run_hotfix(bogus, bogus, {"WEIGHTLESS_STEER_PATH": "/nonexistent.gguf"})
        check(r3.returncode == 1, "anchors missing + WEIGHTLESS_STEER_PATH set fails closed")
        r4 = run_hotfix(bogus, bogus)
        check(r4.returncode == 0, "anchors missing + steering off stays stock")

    print()
    print("qwen steering structure: " + ("all checks passed" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODELS
    sys.exit(main(target))
