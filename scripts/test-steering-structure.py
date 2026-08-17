#!/usr/bin/env python3
"""Structural checks on the steering code in the DeepSeek V4 model overlay.

These exist because of a real regression. Commit 38d0202 added the pre-fold hook
registry and dedented the existing per-layer assignment out of the loop that
builds it:

    for _lid in range(config.num_hidden_layers):
        ...
        _v = _q.T[: _v.shape[0]]
    _DSPARK_HOOK_DIRS[_lid] = ...      # <-- outside the loop
    self._steer_dirs[_lid] = ...       # <-- outside the loop

The loop still ran, still orthonormalised every layer's direction, and then threw
all but the last away. The result is a server that steers exactly one layer while
the operator believes it steers twenty-nine, and coverage is the variable that
dominates this intervention: 6 layers leaves 18.0% refusal, 16 leaves 3.8%, 29
leaves 0.0%. Nothing in the logs distinguishes the two, because the direction
loads fine and the model answers fine.

Run: python3 scripts/test-steering-structure.py [path/to/model.py]
No GPU, no torch, no vLLM import -- this parses the source.
"""

from __future__ import annotations

import ast
import pathlib
import sys

DEFAULT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "recipe/overlay/vllm/models/deepseek_v4/nvidia/model.py"
)

# Assignments that are per-layer and must therefore live inside the per-layer loop.
PER_LAYER_TARGETS = ("_steer_dirs[_lid]", "_DSPARK_HOOK_DIRS[_lid]")

# Module-level entry points the overlay guard also asserts on.
REQUIRED_SYMBOLS = ("_load_gguf_control_vector", "_DSPARK_STEER_HOOK")


def find_layer_loop(tree: ast.AST) -> ast.For | None:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_lid"
        ):
            return node
    return None


def main(path: pathlib.Path) -> int:
    src = path.read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        print(f"  [FAIL] {path} does not parse: {exc}")
        return 1

    failures = 0

    for sym in REQUIRED_SYMBOLS:
        present = any(
            (isinstance(n, ast.FunctionDef) and n.name == sym)
            or (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == sym for t in n.targets)
            )
            for n in tree.body
        )
        if present:
            print(f"  [PASS] module-level {sym} present")
        else:
            print(f"  [FAIL] module-level {sym} missing")
            failures += 1

    loop = find_layer_loop(tree)
    if loop is None:
        print("  [FAIL] no `for _lid in ...` per-layer loop found")
        return failures + 1

    body = ast.get_source_segment(src, loop) or ""
    for target in PER_LAYER_TARGETS:
        if target in body:
            print(f"  [PASS] {target} is inside the per-layer loop")
        else:
            # Distinguish "moved out of the loop" from "removed entirely".
            where = "elsewhere in the file" if target in src else "nowhere"
            print(
                f"  [FAIL] {target} is {where}, not inside the per-layer loop "
                f"(lines {loop.lineno}-{loop.end_lineno}). Only the final "
                f"iteration would be kept, so the server would steer one layer."
            )
            failures += 1

    print()
    print("steering structure: " + ("all checks passed" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    sys.exit(main(target))
