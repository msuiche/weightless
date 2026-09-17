#!/usr/bin/env python3
"""Structural check on the per-request xargs patcher for vLLM v0.27.0.

patches/vllm/v0_27_0/patcher.py rewrites four files inside an installed
vLLM tree by matching exact source anchors. Anchors are the fragile part:
when the upstream tree moves, a rewrite either stops matching (loud, and
the patcher fails closed) or matches something it should not (quiet, and
the server boots with half a control plane). Neither shows up in a unit
test of the patcher's own helpers, because those never see real vLLM
source.

So this runs the real patcher against a skeleton built from the real tree
and checks four things: every anchor matched, every rewritten file still
parses, the engine-neutral core landed beside the package, and a second
run is a no-op. It never touches the source tree -- the four target files
are copied into a temp skeleton first.

Run: python3 tests/structure/test-weightless-xargs-patcher-structure.py [path/to/vllm]
Default target: ../vllm, the source checkout the live lanes are built from.
"""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import shutil
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
PATCHER = REPO / "patches" / "vllm" / "v0_27_0" / "patcher.py"

# The files the patcher rewrites, relative to the vllm package root.
TARGETS = (
    "v1/request.py",
    "entrypoints/openai/engine/protocol.py",
    "v1/worker/gpu/model_runner.py",
    "v1/worker/gpu_model_runner.py",
)
# The model file whose path the patcher walks up to find the package root.
MODEL_REL = "model_executor/models/nemotron_h.py"


def load_patcher():
    spec = importlib.util.spec_from_file_location("weightless_patcher",
                                                  PATCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_skeleton(source_root: pathlib.Path, tmp: pathlib.Path):
    """Copy just the files the patcher touches into a throwaway tree."""
    root = tmp / "site" / "vllm"
    for rel in TARGETS + (MODEL_REL,):
        src = source_root / rel
        if not src.is_file():
            return None, rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return root, None


def main(argv):
    source_root = pathlib.Path(
        argv[1] if len(argv) > 1 else REPO.parent / "vllm" / "vllm"
    ).resolve()
    if source_root.name != "vllm":
        source_root = source_root / "vllm"
    if not source_root.is_dir():
        print(f"SKIP: no vLLM source tree at {source_root}")
        return 0
    if not PATCHER.is_file():
        print(f"SKIP: patcher not present at {PATCHER}")
        return 0

    failures = 0

    def check(label, ok, detail=""):
        nonlocal failures
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
              + (f" -- {detail}" if detail and not ok else ""))
        if not ok:
            failures += 1

    patcher = load_patcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        root, missing = build_skeleton(source_root, tmp)
        if root is None:
            print(f"SKIP: {source_root} has no {missing} "
                  f"(not the layout this patcher targets)")
            return 0

        before = {rel: (root / rel).read_text() for rel in TARGETS}
        ok, msgs = patcher._apply_runtime_xargs_patches(root / MODEL_REL)
        check("patcher applies against the real tree", ok,
              "; ".join(msgs))
        if not ok:
            for line in msgs:
                print(f"         {line}")
            print(f"\nweightless xargs patcher: {failures} failure(s)")
            return 1

        for rel in TARGETS:
            text = (root / rel).read_text()
            check(f"{rel} was actually rewritten", text != before[rel])
            try:
                ast.parse(text)
                check(f"{rel} still parses", True)
            except SyntaxError as exc:
                check(f"{rel} still parses", False,
                      f"line {exc.lineno}: {exc.msg}")

        core = root.parent / "weightless_runtime"
        check("engine-neutral core installed beside the package",
              core.is_dir()
              and {"__init__.py", "controls.py", "telemetry.py"}
              <= {p.name for p in core.iterdir()})
        # The patcher COPIES the repo's core in, so a lane it patches must
        # pick up the server alpha policy window automatically. If this
        # ever fails, the patched lane accepts unbounded client alphas
        # while the plugin lane does not.
        check("installed core carries the alpha policy window",
              core.is_dir()
              and "_bounded_alpha" in (core / "controls.py").read_text())

        after = {rel: (root / rel).read_text() for rel in TARGETS}
        ok2, _ = patcher._apply_runtime_xargs_patches(root / MODEL_REL)
        check("second run reports success", ok2)
        check("second run is a no-op (idempotent)",
              all((root / rel).read_text() == after[rel] for rel in TARGETS))

    print()
    print("weightless xargs patcher: "
          + ("all checks passed" if not failures
             else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
