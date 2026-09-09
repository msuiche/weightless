#!/usr/bin/env python3
"""Structural checks on the Kimi K3 steering hotfix (2x H200:8 PP2xTP8 lane).

Same production lessons as the other lanes' tests, adapted to the kimi_k3
day-0 fork (post-layer stream is prefix_sum + hidden_states; the attn_res
side stream rides in `residual`):

1. The per-layer assignment once got dedented out of its loop on the DSV4
   lane; the server steered one layer while reporting all of them.
2. The Qwen3.8 lane's first boot died because the buffers were registered
   on a class whose __init__ the serving subclass skips. Anchors matching
   is NOT semantics being right — so this test checks that the class whose
   forward contains the apply (KimiLinearModel) registers the buffers in
   its OWN __init__.
3. The apply must steer prefix_sum + hidden_states (the post-layer
   accumulated stream), not hidden_states alone, and must write the delta
   back into hidden_states (hidden_states = steered - prefix_sum) so the
   loop carries it.
4. The MLA DCP fix (boot #14/#15: cp_world_size must be positive under
   PP=2) must resolve the sentinel at construction AND in forward_impl.

This test applies patches/hotfix-kimi-k3-steering-projective.py to SCRATCH
COPIES of the vendored references (never the originals) and AST-checks the
result:

  1. all anchors match and the patched files still parse;
  2. the per-layer probe write and steering apply are INSIDE the per-layer
     loop of KimiLinearModel.forward;
  3. the apply steers prefix_sum + hidden_states, indexes the steer stack
     by layer_idx, and writes back into hidden_states;
  4. KimiLinearModel.__init__ registers _steer_stack / _steer_alpha;
  5. the runner patch arms the gate before the model call and flushes
     after the forward context;
  6. the MLA patch carries both DCP resolutions;
  7. re-applying is a no-op, and a missing anchor fails closed.

Run: python3 scripts/test-k3-steering-structure.py
No GPU, no torch, no vLLM import — this runs the hotfix and parses source.
"""

from __future__ import annotations

import ast
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
HOTFIX = REPO / "patches/hotfix-kimi-k3-steering-projective.py"
REFERENCES = {
    "DSPARK_K3_MODEL_PY": REPO / "patches/reference/kimi_k3_nvidia_model.py",
    "DSPARK_K3_RUNNER_PY": REPO / "patches/reference/kimi_k3_gpu_model_runner.py",
    "DSPARK_K3_MLA_PY": REPO / "patches/reference/kimi_k3_mla_attention.py",
}

FAILURES = []


def check(ok: bool, label: str):
    print(("ok   " if ok else "FAIL ") + label)
    if not ok:
        FAILURES.append(label)


def run_hotfix(scratch: dict[str, pathlib.Path]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for key, path in scratch.items():
        env[key] = str(path)
    return subprocess.run([sys.executable, str(HOTFIX)], env=env,
                          capture_output=True, text=True)


def find_class(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def find_method(cls, name):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def text_of(src: str, node) -> str:
    return ast.get_source_segment(src, node) or ""


def main():
    for key, ref in REFERENCES.items():
        if not ref.is_file():
            print(f"missing reference: {ref}")
            return 1

    with tempfile.TemporaryDirectory() as td:
        scratch = {}
        for key, ref in REFERENCES.items():
            dst = pathlib.Path(td) / ref.name
            shutil.copy(ref, dst)
            scratch[key] = dst

        # 1. applies + parses -------------------------------------------------
        r = run_hotfix(scratch)
        check(r.returncode == 0,
              f"hotfix applies to all three references "
              f"({(r.stderr or r.stdout).strip().splitlines()[-1] if r.returncode else 'anchors matched'})")
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            return 1

        model_src = scratch["DSPARK_K3_MODEL_PY"].read_text()
        runner_src = scratch["DSPARK_K3_RUNNER_PY"].read_text()
        mla_src = scratch["DSPARK_K3_MLA_PY"].read_text()
        tree = ast.parse(model_src)

        # 2./3. apply inside the layer loop, correct stream, write-back ------
        cls = find_class(tree, "KimiLinearModel")
        check(cls is not None, "KimiLinearModel found in patched model.py")
        fwd = find_method(cls, "forward") if cls else None
        check(fwd is not None, "KimiLinearModel.forward found")
        loop = None
        if fwd:
            for node in ast.walk(fwd):
                if (isinstance(node, ast.For)
                        and "start_layer" in text_of(model_src, node)):
                    loop = node
                    break
        check(loop is not None,
              "per-layer loop (start_layer..end_layer) found in forward")
        if loop:
            body = text_of(model_src, loop)
            check("self._steer_stack[layer_idx]" in body,
                  "steer stack indexed by layer_idx INSIDE the layer loop")
            check("prefix_sum + hidden_states" in body,
                  "apply steers the post-layer stream "
                  "(prefix_sum + hidden_states)")
            check("hidden_states = (" in body and "_ds_h - prefix_sum" in body,
                  "steered stream written back into hidden_states")
            check("_probe_views[self._probe_pos[layer_idx]]" in body,
                  "probe write indexed by layer position INSIDE the loop")

        # 4. buffers on the class whose forward steers ------------------------
        init = find_method(cls, "__init__") if cls else None
        init_src = text_of(model_src, init) if init else ""
        check('"_steer_stack"' in init_src and '"_steer_alpha"' in init_src,
              "KimiLinearModel.__init__ registers _steer_stack + _steer_alpha")
        check("WEIGHTLESS_STEER_PATH" in init_src,
              "steering load is env-gated in __init__")
        check('raise RuntimeError' in init_src,
              "steering load fails closed (bad path/width/no layers)")

        # 5. runner: gate before the model call, flush after ------------------
        gate = runner_src.find("arm the capture gate")
        call = runner_src.find("model_output = self._model_forward(")
        flush = runner_src.find("flush the capture slot")
        post = runner_src.find("gpu_model_runner: postprocess")
        check(0 < gate < call, "gate armed immediately before _model_forward")
        check(0 < flush < post,
              "flush after the forward, before postprocess")

        # 6. MLA DCP fix, both resolutions ------------------------------------
        check(mla_src.count("dspark k3 probe") >= 2,
              "MLA patch: sentinel resolved at construction AND forward_impl")
        check("decode_context_parallel_size" in mla_src,
              "MLA patch resolves DCP from the config, not the group")

        # 7. idempotent + fail-closed -----------------------------------------
        r2 = run_hotfix(scratch)
        check(r2.returncode == 0 and "already patched" in r2.stdout,
              "re-applying is a no-op")
        broken = pathlib.Path(td) / "broken_model.py"
        broken.write_text(
            REFERENCES["DSPARK_K3_MODEL_PY"].read_text().replace(
                "import math\nfrom collections.abc import Iterable",
                "import math\nfrom collections import Iterable"))
        r3 = run_hotfix({**scratch, "DSPARK_K3_MODEL_PY": broken})
        check(r3.returncode != 0
              and "PATCH FAILED" in (r3.stdout + r3.stderr),
              "missing anchor fails closed")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES")
        return 1
    print("all k3 steering structure checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
