#!/usr/bin/env python3
"""Structural checks on the Qwen3.8-Flash-Next steering hotfix.

Same production lessons as the Qwen3.8 lane's test, adapted to the
delayed-combine hyper-connection arch:

1. The per-layer assignment once got dedented out of its loop on the DSV4
   lane; the server steered one layer while reporting all of them.
2. The Qwen3.8 lane's first boot died because the buffers were registered
   on a class whose __init__ the serving subclass skips. Anchors matching
   is NOT semantics being right — so this test checks that the class whose
   forward contains the apply registers the buffers in its OWN __init__.
   (qwen3_8_flash_next has no Qwen3_5Model-style __init__ override:
   Qwen3_8FlashNextModel is a direct nn.Module and the serving
   Qwen3_8FlashNextForCausalLM delegates to it — the test asserts the
   general property anyway, so an upstream refactor that introduces the
   trap fails here.)
3. The apply must steer the MATERIALIZED post-layer stream
   (mlp_hyper_connection.combine of hidden_states/block_output/injection),
   and the final mixer must be guarded for the consumed pending combine.

This test applies patches/hotfix-qwen38fn-steering-projective.py to a
SCRATCH COPY of the reference model file (never the original) and
AST-checks the result:

  1. all anchors match and the patched file still parses;
  2. the per-layer assignments are INSIDE the per-layer loop;
  3. the forward apply steers the materialized post-layer stream and
     indexes the steer stack by global layer id;
  4. the class whose forward applies steering registers the buffers in its
     own __init__ (the lesson-2 regression guard);
  5. the final mixer takes the mix() path when the pending combine was
     consumed (last-layer-steered guard);
  6. re-applying is a no-op, and anchors-missing fails closed when
     WEIGHTLESS_STEER_PATH is set.

Run: python3 scripts/test-qwen38fn-steering-structure.py [reference.py]
Default is the vendored copy of the day-0 image's
vllm/models/qwen3_8_flash_next/nvidia/model.py in patches/reference/ (the
local ../vllm checkout predates the arch — support is image-only).
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
HOTFIX = REPO / "patches/hotfix-qwen38fn-steering-projective.py"
DEFAULT_REFERENCE = REPO / "patches/reference/qwen3_8_flash_next.py"

PER_LAYER_TARGETS = ("self._steer_dirs[layer_id]", "_GLP_HOOK_DIRS[layer_id]")


def run_hotfix(model_py: pathlib.Path,
               env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["WEIGHTLESS_STEERING_MODEL_PY"] = str(model_py)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(HOTFIX)], env=env, capture_output=True, text=True
    )


def find_load_loop(tree: ast.AST, src: str) -> ast.For | None:
    """The per-layer loop in _load_steering. The stock file has an unrelated
    `for layer_id in qsa_layer_ids` loop, so match the one whose body holds
    the steering assignments, not just the first `for layer_id`."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "layer_id"
            and "_steer_dirs" in (ast.get_source_segment(src, node) or "")
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


def find_apply_class(tree: ast.AST, src: str) -> ast.ClassDef | None:
    """The class whose forward contains the steering apply."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            seg = ast.get_source_segment(src, node) or ""
            if "self._steer_stack[layer_idx]" in seg:
                return node
    return None


def main(reference: pathlib.Path) -> int:
    if not reference.is_file():
        print(f"  [SKIP] reference file not found: {reference}")
        print("         pass the reference explicitly: test-qwen38fn-steering-structure.py <model.py>")
        return 2

    failures = 0

    def check(ok: bool, label: str, detail: str = ""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
        failures += 0 if ok else 1

    with tempfile.TemporaryDirectory() as td:
        scratch = pathlib.Path(td) / "model.py"
        shutil.copy(reference, scratch)

        # 1. applies + parses
        r = run_hotfix(scratch)
        check(
            r.returncode == 0 and r.stdout.count("applied to") == 1,
            "hotfix applies to the reference file",
            r.stdout + r.stderr,
        )
        src = scratch.read_text()
        try:
            tree = ast.parse(src)
            check(True, "patched model.py parses")
        except SyntaxError as exc:
            check(False, "patched model.py parses", str(exc))
            return failures

        # 2. per-layer assignments inside the loop
        loop = find_load_loop(tree, src)
        check(loop is not None, "per-layer loop found in _load_steering")
        if loop is not None:
            body = ast.get_source_segment(src, loop) or ""
            for target in PER_LAYER_TARGETS:
                check(target in body, f"{target} inside the per-layer loop")

        # 3. forward apply steers the materialized post-layer stream
        check(
            "layer.mlp_hyper_connection.combine(" in src
            and "steer_stream" in src,
            "forward apply materializes the post-layer stream (delayed combine)",
        )
        check(
            "self._steer_stack[layer_idx]" in src,
            "forward apply indexes the steer stack by global layer id",
        )
        check(
            "steer_dirs = self._steer_stack[layer_idx]" in src
            and "steer_stream - self._steer_alpha" in src,
            "forward apply projects the materialized stream",
        )

        # 4. the lesson-2 regression guard, generalized: the class whose
        #    forward applies steering must register the buffers in its own
        #    __init__ (a serving subclass that skips this __init__ would boot
        #    into '... has no attribute _steer_stack').
        cls = find_apply_class(tree, src)
        check(cls is not None, "steering apply found in a model class")
        if cls is not None:
            init = find_class_init(tree, cls.name)
            check(
                init is not None,
                f"{cls.name}.__init__ exists (no skip-parent trap)",
            )
            if init is not None:
                seg = ast.get_source_segment(src, init) or ""
                check(
                    '"_steer_stack"' in seg and "_load_steering" in seg,
                    f"{cls.name}.__init__ registers the steering buffers",
                )

        # 5. final mixer guarded for a consumed pending combine (the apply
        #    materializes the last steered layer; stock combine_and_mix would
        #    then re-combine None).
        check(
            "final_mixer.mix(" in src,
            "final mixer guarded: mix() when the pending combine was consumed",
        )

        # 6a. idempotent
        r2 = run_hotfix(scratch)
        check(
            r2.returncode == 0 and r2.stdout.count("already applied") == 1,
            "re-apply is a no-op",
            r2.stdout + r2.stderr,
        )

        # 6b. fail-closed: steering requested but anchors missing
        bogus = pathlib.Path(td) / "bogus.py"
        bogus.write_text("# not a model file\n")
        r3 = run_hotfix(bogus, {"WEIGHTLESS_STEER_PATH": "/nonexistent.gguf"})
        check(r3.returncode == 1, "anchors missing + WEIGHTLESS_STEER_PATH set fails closed")
        r4 = run_hotfix(bogus)
        check(r4.returncode == 0, "anchors missing + steering off stays stock")

    print()
    print("qwen38fn steering structure: " + ("all checks passed" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFERENCE
    sys.exit(main(target))
