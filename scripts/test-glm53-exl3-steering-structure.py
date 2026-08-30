#!/usr/bin/env python3
"""Structural checks on the GLM-5.3-Flash EXL3/B12X steering hotfix.

Same production lessons as the other lanes' tests, plus the fork-specific
one this variant exists for: the B12X fork added a DFlash aux-hidden-state
branch, so the stock single decoder loop is TWO loops (non-aux and aux),
nested one level deeper. Steering must land in BOTH — and in the aux loop it
must come AFTER the aux capture, so DFlash features are pre-steering while
the stream continues steered.

This test applies patches/hotfix-glm53-exl3-steering-projective.py to a
SCRATCH COPY of the reference model file (never the original) and
AST-checks the result:

  1. all anchors match and the patched file still parses;
  2. the per-layer assignments are INSIDE the per-layer loop;
  3. BOTH loops got the steering block: each materializes via hc_post,
     flattens the widened stream HC-outer, indexes the steer stack by
     global layer id, and contracts the last layer in-loop;
  4. in the aux loop the steering block comes AFTER the aux capture
     (pre-steering features for the drafter);
  5. the class whose forward applies steering registers the buffers in its
     own __init__ (the lesson-2 regression guard);
  6. the decoder's last-layer in-line contract is deferred;
  7. re-applying is a no-op, and anchors-missing fails closed when
     WEIGHTLESS_STEER_PATH is set.

Run: python3 scripts/test-glm53-exl3-steering-structure.py [reference.py]
Default is the vendored copy of the EXL3/B12X image's glm5next model.py in
patches/reference/. No GPU, no torch, no vLLM import -- this runs the
hotfix and parses source.
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
HOTFIX = REPO / "patches/hotfix-glm53-exl3-steering-projective.py"
DEFAULT_REFERENCE = REPO / "patches/reference/glm5next_b12x_exl3.py"

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
            if "self._steer_stack[layer.layer_idx]" in seg:
                return node
    return None


def main(reference: pathlib.Path) -> int:
    if not reference.is_file():
        print(f"  [SKIP] reference file not found: {reference}")
        print("         pass the reference explicitly: test-glm53-exl3-steering-structure.py <model.py>")
        return 2

    failures = 0

    def check(ok: bool, label: str, detail: str = ""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail and not ok else ""))
        failures += 0 if ok else 1

    with tempfile.TemporaryDirectory() as td:
        scratch = pathlib.Path(td) / "model.py"
        shutil.copy(reference, scratch)

        # 1. applies (6 anchors) + parses
        r = run_hotfix(scratch)
        check(
            r.returncode == 0 and r.stdout.count("applied to") == 1,
            "hotfix applies to the reference file",
            r.stdout + r.stderr,
        )
        check(
            "(6 anchors)" in r.stdout,
            "all 6 anchors applied (incl. both decoder loops)",
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

        # 3. BOTH loops got the steering block
        n_apply = src.count("projective activation steering [steering-hotfix] ---")
        check(
            n_apply == 2,
            "steering apply present in BOTH decoder loops",
            f"found {n_apply}",
        )
        check(
            src.count("steer_stream = layer.hc_post(hidden_states, residual, post, comb)") == 2,
            "both loops materialize the post-layer stream (deferred hc_post)",
        )
        check(
            src.count("self._steer_stack[layer.layer_idx]") == 2,
            "both loops index the steer stack by global layer id",
        )
        check(
            src.count("steer_flat = steer_stream.flatten(-2)") == 2,
            "both loops flatten the widened stream HC-outer (n*hidden)",
        )
        check(
            src.count("hc_contract(hidden_states, layer.n)") == 2,
            "both loops contract the last layer in-loop, after steering",
        )

        # 4. aux loop: steering AFTER the aux capture (pre-steer features).
        lines = src.splitlines()
        aux_capture_line = next(
            i for i, l in enumerate(lines)
            if "self._materialize_aux_hidden_state(" in l
        )
        steer_lines = [
            i for i, l in enumerate(lines)
            if "projective activation steering [steering-hotfix] ---" in l
        ]
        check(
            len(steer_lines) == 2
            and any(i > aux_capture_line for i in steer_lines)
            and any(i < aux_capture_line for i in steer_lines),
            "aux-loop steering comes AFTER the aux capture (pre-steer features)",
        )

        # 5. the lesson-2 regression guard, generalized.
        cls = find_apply_class(tree, src)
        check(cls is not None, "steering apply found in a model class")
        if cls is not None:
            check(
                cls.name == "Glm5NextModel",
                "steering apply is on Glm5NextModel (the serving wrappers delegate to it)",
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

        # 6. last-layer coverage: the in-decoder hc_post+hc_contract special
        #    case must be gone (it would escape steering).
        check(
            "if self.layer_idx == self.num_hidden_layers - 1:" not in src,
            "decoder last-layer in-line contract deferred (layer N-1 is steered)",
        )

        # 7a. idempotent
        r2 = run_hotfix(scratch)
        check(
            r2.returncode == 0 and r2.stdout.count("already applied") == 1,
            "re-apply is a no-op",
            r2.stdout + r2.stderr,
        )

        # 7b. fail-closed: steering requested but anchors missing
        bogus = pathlib.Path(td) / "bogus.py"
        bogus.write_text("# not a model file\n")
        r3 = run_hotfix(bogus, {"WEIGHTLESS_STEER_PATH": "/nonexistent.gguf"})
        check(r3.returncode == 1, "anchors missing + WEIGHTLESS_STEER_PATH set fails closed")
        r4 = run_hotfix(bogus)
        check(r4.returncode == 0, "anchors missing + steering off stays stock")

    print()
    print("glm53-exl3 steering structure: " + ("all checks passed" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REFERENCE
    sys.exit(main(target))
