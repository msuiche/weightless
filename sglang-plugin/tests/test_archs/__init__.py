"""Helpers shared by the per-architecture tests.

- ``_trees(*files)``: the SGLang package folders to read, the installed
  package first, then every folder listed in WEIGHTLESS_TEST_SGLANG_TREES
  (os.pathsep separated), keeping those that hold every given file (paths
  relative to the package, for example "srt/models/qwen3_5.py"). The
  structure tests parse those files with ``ast`` and never import them.
- Small ``ast`` helpers over those files.
- ``unit_dirs`` (synthetic unit directions for a GLP file), ``proj_ok``
  (the edit's two properties) and the pieces the tests that run SGLang's
  own classes share (a one-rank group, a runner around a model, a spy).
"""
import ast
import importlib.util
import os
import types

import numpy as np
import torch

def _trees(*files):
    out = []
    spec = importlib.util.find_spec("sglang")
    if spec is not None and spec.submodule_search_locations:
        out.append(list(spec.submodule_search_locations)[0])
    for d in os.environ.get("WEIGHTLESS_TEST_SGLANG_TREES", "").split(os.pathsep):
        if d.strip():
            out.append(d.strip())
    seen, res = set(), []
    for d in out:
        r = os.path.realpath(d)
        if r not in seen and all(os.path.isfile(os.path.join(r, *f.split("/"))) for f in files):
            seen.add(r)
            res.append(r)
    return res


def _parse(tree, rel):
    with open(os.path.join(tree, *rel.split("/"))) as f:
        return ast.parse(f.read())


def _cls(mod, name):
    for n in mod.body:
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    raise AssertionError(f"class {name} not found")


def _fn(cls, name):
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(f"{cls.name}.{name} not found")


def _has_fn(cls, name):
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in cls.body)


def _src(node):
    return ast.unparse(node)


def _calls(node, attr):
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Attribute) and f.attr == attr) or (isinstance(f, ast.Name) and f.id == attr):
                out.append(n)
    return sorted(out, key=lambda c: (c.lineno, c.col_offset))


def _calls_where(node, pred):
    out = [n for n in ast.walk(node) if isinstance(n, ast.Call) and pred(n)]
    return sorted(out, key=lambda c: (c.lineno, c.col_offset))


def _entry(mod):
    for n in mod.body:
        if isinstance(n, ast.Assign) and _src(n.targets[0]) == "EntryClass":
            return _src(n.value)
    raise AssertionError("no EntryClass")


def _entry_names(mod):
    return [_src(e) for e in ast.parse(_entry(mod), mode="eval").body.elts]


def _returns(fn):
    return [_src(r.value) for r in ast.walk(fn) if isinstance(r, ast.Return)]


def _returns_in_order(fn):
    """The returned expressions of ``fn``, in source order."""
    rets = sorted((r for r in ast.walk(fn) if isinstance(r, ast.Return)),
                  key=lambda r: (r.lineno, r.col_offset))
    return [_src(r.value) for r in rets]


def _loops(fn):
    return [n for n in ast.walk(fn) if isinstance(n, ast.For)]


def _layer_calls(node):
    """Assignments whose value is a call of ``layer(...)``."""
    return [n for n in ast.walk(node) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
            and _src(n.value.func) == "layer"]


def _assign(mod, name):
    for n in mod.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return n.value
    raise AssertionError(f"{name} not found")


def _flat(text):
    """Source text with its whitespace collapsed (ast.unparse's indentation
    depends on the nesting it starts from)."""
    return " ".join(text.split())


def unit_dirs(layers, width, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in layers:
        v = rng.standard_normal(width)
        out[f"direction.{i}"] = ((v / np.linalg.norm(v)).astype(np.float32), 0)
    return out


def proj_ok(test, pre, post, d, alpha, rel=1e-6):
    """post . d == (1 - alpha)(pre . d), and post == pre across d (last
    axis), to ``rel`` of the stream's norm: the stack holds fp32 unit
    vectors, so |d|^2 - 1 is about 1e-8 in fp64."""
    tol = rel * (1.0 + float(pre.norm(dim=-1).max()))
    test.assertLess(float((post @ d - (1 - alpha) * (pre @ d)).abs().max()), tol)
    e = torch.randn(d.shape[0], dtype=d.dtype, generator=torch.Generator().manual_seed(1))
    e = e - (e @ d) * d
    test.assertLess(float((post @ e - pre @ e).abs().max()), tol)


class OneRank:
    """A one-rank process group: all_reduce returns its input."""

    world_size = 1

    def all_reduce(self, x):
        return x


def runner_of(model, hf_model_type, *, tp_size=1, pp_rank=0, pp_size=1, dtype=torch.float32):
    return types.SimpleNamespace(
        model=model, is_draft_worker=False, tp_rank=0, tp_size=tp_size, pp_rank=pp_rank,
        pp_size=pp_size, server_args=types.SimpleNamespace(enable_two_batch_overlap=False),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf_model_type),
                                           dtype=dtype))


def spy_out(layer, store, key):
    return layer.register_forward_hook(lambda m, a, o: store.__setitem__(key, o))
