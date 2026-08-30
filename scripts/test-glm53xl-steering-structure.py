#!/usr/bin/env python3
"""Structural checks on the GLM-5.3 743B (glm_moe_dsa) steering hotfix.

Same production lessons as the other lanes' tests, adapted to the
deepseek_v2 decomposed convention (this arch has NO hyperconnection
widening — the post-layer stream is hidden_states + residual):

1. The per-layer assignment once got dedented out of its loop on the DSV4
   lane; the server steered one layer while reporting all of them.
2. The Qwen3.8 lane's first boot died because the buffers were registered
   on a class whose __init__ the serving subclass skips. Anchors matching
   is NOT semantics being right — so this test checks that the class whose
   forward contains the apply registers the buffers in its OWN __init__,
   and that GlmMoeDsaForCausalLM (the served class) does NOT override
   __init__ (it is a plain `pass` subclass of DeepseekV2ForCausalLM).
3. The apply must steer hidden_states + residual (the decomposed
   convention) — steering hidden_states alone removes the component from
   only part of the stream.

This test applies patches/hotfix-glm53xl-steering-projective.py to a SCRATCH
COPY of the reference model file (never the original) and AST-checks the
result:

  1. all anchors match and the patched file still parses;
  2. the per-layer assignments are INSIDE the per-layer loop;
  3. the forward apply steers hidden_states + residual and indexes the
     steer stack by global layer id;
  4. DeepseekV2Model.__init__ registers the buffers and
     GlmMoeDsaForCausalLM has no __init__ of its own;
  5. re-applying is a no-op, and anchors-missing fails closed when
     WEIGHTLESS_STEER_PATH is set.

Run: python3 scripts/test-glm53xl-steering-structure.py [reference.py]
Default is the vendored copy of the GB10 kernel-overlay deepseek_v2.py in
patches/reference/ (the file tonyd2wild's stack actually serves).
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
HOTFIX = REPO / "patches/hotfix-glm53xl-steering-projective.py"
DEFAULT_REFERENCE = REPO / "patches/reference/deepseek_v2_glm53xl.py"

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
    """The per-layer loop in _load_steering: the `for layer_id` loop whose
    body holds the steering assignments (not any stock loop of that name)."""
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
            if "self._steer_stack[idx]" in seg:
                return node
    return None


def main(reference: pathlib.Path) -> int:
    if not reference.is_file():
        print(f"  [SKIP] reference file not found: {reference}")
        print("         pass the reference explicitly: test-glm53xl-steering-structure.py <model.py>")
        return 2

    failures = 0

    def check(ok: bool, label: str, detail: str = ""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
        failures += 0 if ok else 1

    with tempfile.TemporaryDirectory() as td:
        scratch = pathlib.Path(td) / "deepseek_v2.py"
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
            check(True, "patched deepseek_v2.py parses")
        except SyntaxError as exc:
            check(False, "patched deepseek_v2.py parses", str(exc))
            return failures

        # 2. per-layer assignments inside the loop
        loop = find_load_loop(tree, src)
        check(loop is not None, "per-layer loop found in _load_steering")
        if loop is not None:
            body = ast.get_source_segment(src, loop) or ""
            for target in PER_LAYER_TARGETS:
                check(target in body, f"{target} inside the per-layer loop")

        # 3. forward apply steers the full residual stream (decomposed
        #    convention: hidden_states + residual)
        check(
            "steer_stream = hidden_states + residual" in src,
            "forward apply steers hidden_states + residual (decomposed convention)",
        )
        check(
            "self._steer_stack[idx]" in src,
            "forward apply indexes the steer stack by global layer id",
        )
        check(
            "steer_stream - self._steer_alpha" in src
            and "- residual" in src.split("steer_stream - self._steer_alpha")[1][:200],
            "forward apply writes the steered sum back into hidden_states",
        )

        # 4. the lesson-2 regression guard, generalized: the class whose
        #    forward applies steering must register the buffers in its own
        #    __init__, and the served subclass must not skip it.
        cls = find_apply_class(tree, src)
        check(cls is not None, "steering apply found in a model class")
        if cls is not None:
            check(
                cls.name == "DeepseekV2Model",
                "steering apply is on DeepseekV2Model (the deepseek_v2 path)",
                f"found on {cls.name}",
            )
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
        glm_init = find_class_init(tree, "GlmMoeDsaForCausalLM")
        check(
            glm_init is None,
            "GlmMoeDsaForCausalLM has no __init__ override (plain subclass)",
        )

        # 5a. idempotent
        r2 = run_hotfix(scratch)
        check(
            r2.returncode == 0 and r2.stdout.count("already applied") == 1,
            "re-apply is a no-op",
            r2.stdout + r2.stderr,
        )

        # 5b. fail-closed: steering requested but anchors missing
        bogus = pathlib.Path(td) / "bogus.py"
        bogus.write_text("# not a model file\n")
        r3 = run_hotfix(bogus, {"WEIGHTLESS_STEER_PATH": "/nonexistent.gguf"})
        check(r3.returncode == 1, "anchors missing + WEIGHTLESS_STEER_PATH set fails closed")
        r4 = run_hotfix(bogus)
        check(r4.returncode == 0, "anchors missing + steering off stays stock")

    print()
    print("glm53xl steering structure: " + ("all checks passed" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFERENCE
    sys.exit(main(target))
