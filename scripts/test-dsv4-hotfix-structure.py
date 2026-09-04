#!/usr/bin/env python3
"""Structural checks on the DSV4 steering boot hotfix (Anemll 0.25.2 lane).

Sibling of test-qwen-steering-structure.py. The production lesson lives in
test-steering-structure.py's docstring (a dedented per-layer assignment once
steered 1 layer while reporting 29); that older test targets the retired
v027 overlay's model.py, which is no longer in the repo. This one guards the
artifact that actually ships: patches/hotfix-dsv4-steering-projective.py.

Always-on checks (no reference sources needed):

  1. the hotfix parses, exposes the 4-anchor PATCHES table, and its embedded
     code blocks (GGUF_SRC / INIT_BLOCK / FORWARD_BLOCK) are extractable;
  2. the embedded GGUF loader parses and keeps the spec enforcement
     (dspark.mode=project, ffn_out_pre_residual, direction.0
     rejection) — an additive reader must refuse our files;
  3. the per-layer assignments in the injected INIT_BLOCK are INSIDE the
     per-layer loop (the dedent regression guard);
  4. fail-closed surface: anchors-missing + WEIGHTLESS_STEER_PATH set exits 1,
     and re-apply is a no-op.

With a reference model.py from the 0.25.2 image (arg or
WEIGHTLESS_STEERING_MODEL_PY), it also applies the hotfix to a SCRATCH COPY
(never the original) and AST-checks the result. Without one it exits 2
(SKIP) after the always-on checks.

    docker run --rm --entrypoint cat ghcr.io/anemll/dspark-vllm-gx10:0.1.1 \
      /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py \
      > /tmp/model.py
    python3 scripts/test-dsv4-hotfix-structure.py /tmp/model.py

No GPU, no torch -- the --check vector path needs torch and is not run here.
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
HOTFIX = REPO / "patches/hotfix-dsv4-steering-projective.py"

PER_LAYER_TARGETS = ("self._steer_dirs[layer_id]", "_GLP_HOOK_DIRS[layer_id]")
SPEC_TOKENS = ("glp.mode", "ffn_out_pre_residual", "direction.0")


def extract_strings(tree: ast.AST) -> dict:
    """Evaluate module-level string assignments (constants, +-concat, Name
    refs) in source order. Enough to reconstruct the hotfix's blocks without
    executing the file."""
    out = {}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            l, r = ev(node.left), ev(node.right)
            return l + r if l is not None and r is not None else None
        if isinstance(node, ast.Name):
            return out.get(node.id)
        return None

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            v = ev(stmt.value)
            if v is not None:
                out[stmt.targets[0].id] = v
    return out


def check_indent_contained(block: str) -> dict:
    """For the dedent guard: locate the per-layer loop, assert each target
    assignment appears after it at greater indent, before dedent back out."""
    lines = block.splitlines()
    loop_i = loop_indent = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("for layer_id in range(config.num_hidden_layers)"):
            loop_i, loop_indent = i, len(ln) - len(ln.lstrip())
            break
    result = {"loop": loop_i is not None}
    for target in PER_LAYER_TARGETS:
        ok = False
        if loop_i is not None:
            for ln in lines[loop_i + 1:]:
                indent = len(ln) - len(ln.lstrip())
                if ln.strip() and indent <= loop_indent:
                    break  # left the loop body
                if target in ln and indent > loop_indent:
                    ok = True
                    break
        result[target] = ok
    return result


def main(ref_model: pathlib.Path | None) -> int:
    failures = 0

    def check(ok: bool, label: str, detail: str = ""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
              + (f" — {detail}" if detail and not ok else ""))
        failures += 0 if ok else 1

    src = HOTFIX.read_text()
    try:
        tree = ast.parse(src)
        check(True, "hotfix parses")
    except SyntaxError as exc:
        check(False, "hotfix parses", str(exc))
        return 1

    # 1. PATCHES table: exactly the 4 documented anchors
    n_patches = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PATCHES" for t in node.targets
        ) and isinstance(node.value, ast.Tuple):
            n_patches = len(node.value.elts)
    check(n_patches == 4, "PATCHES has the 4 anchors", f"found {n_patches}")

    blocks = extract_strings(tree)
    for name in ("MARK", "GGUF_SRC", "MODULE_BLOCK", "INIT_BLOCK", "FORWARD_BLOCK"):
        check(name in blocks, f"embedded block {name} extractable")

    # 2. the GGUF loader keeps the spec enforcement
    gguf = blocks.get("GGUF_SRC", "")
    try:
        ast.parse(gguf)
        check(True, "embedded GGUF loader parses")
    except SyntaxError as exc:
        check(False, "embedded GGUF loader parses", str(exc))
    check("_load_gguf_control_vector" in gguf, "loader entry point present")
    for tok in SPEC_TOKENS:
        check(tok in gguf, f"spec enforcement token {tok!r} in the loader")

    # 3. the dedent regression guard on the injected __init__ fragment
    guards = check_indent_contained(blocks.get("INIT_BLOCK", ""))
    check(guards.pop("loop"), "per-layer loop found in INIT_BLOCK")
    for target, ok in guards.items():
        check(ok, f"{target} inside the per-layer loop")

    # 4. fail-closed / idempotency surface
    check("failing closed" in src and "steer_requested" in src,
          "anchors-missing + WEIGHTLESS_STEER_PATH fails closed")
    check("already applied" in src, "re-apply is a no-op marker")

    # 5. optional: apply to a scratch copy of the real 0.25.2 model.py
    if ref_model is None:
        print("  [SKIP] no reference model.py — scratch-apply checks skipped")
        print("         (docker cat the image's model.py, pass its path)")
    elif not ref_model.is_file():
        print(f"  [SKIP] reference file not found: {ref_model}")
    else:
        with tempfile.TemporaryDirectory() as td:
            scratch = pathlib.Path(td) / "model.py"
            shutil.copy(ref_model, scratch)
            env = dict(os.environ, WEIGHTLESS_STEERING_MODEL_PY=str(scratch))
            env.pop("WEIGHTLESS_STEER_PATH", None)
            r = subprocess.run([sys.executable, str(HOTFIX)], env=env,
                               capture_output=True, text=True)
            check(r.returncode == 0 and "applied to" in r.stdout,
                  "hotfix applies to the reference model.py", r.stdout + r.stderr)
            if r.returncode == 0:
                patched = scratch.read_text()
                try:
                    ast.parse(patched)
                    check(True, "patched model.py parses")
                except SyntaxError as exc:
                    check(False, "patched model.py parses", str(exc))
                r2 = subprocess.run([sys.executable, str(HOTFIX)], env=env,
                                    capture_output=True, text=True)
                check(r2.returncode == 0 and "already applied" in r2.stdout,
                      "re-apply is a no-op", r2.stdout + r2.stderr)
            bogus = pathlib.Path(td) / "bogus.py"
            bogus.write_text("# not a model file\n")
            env2 = dict(env, WEIGHTLESS_STEERING_MODEL_PY=str(bogus),
                        WEIGHTLESS_STEER_PATH="/nonexistent.gguf")
            r3 = subprocess.run([sys.executable, str(HOTFIX)], env=env2,
                                capture_output=True, text=True)
            check(r3.returncode == 1,
                  "anchors missing + WEIGHTLESS_STEER_PATH set fails closed")

    print()
    print("dsv4 hotfix structure: "
          + ("all checks passed" if not failures else f"{failures} failure(s)"))
    if failures:
        return 1
    return 0 if ref_model and ref_model.is_file() else 2  # 2 = partial (skip tier)


if __name__ == "__main__":
    ref = None
    if len(sys.argv) > 1:
        ref = pathlib.Path(sys.argv[1])
    elif os.environ.get("DSV4_REFERENCE_MODEL_PY"):
        ref = pathlib.Path(os.environ["DSV4_REFERENCE_MODEL_PY"])
    sys.exit(main(ref))
