"""The Nanbeige4.2-3B row (``NanbeigeForCausalLM``), CPU only, and the
``looped`` execution ids it uses.

A looped model runs its 22 physical layers twice per forward (44 execution
steps). Its GLP file has one direction per execution step (``direction.N``
steers step N-1), and one hook per physical layer dispatches on the loop
index the model loop passes. Each layer returns ``(hidden, residual)``.

- Fakes (the first section), in every call style, against the edit
  written out by hand; the step order, the layer filter, a file on one loop
  pass, ``forward_wrap`` on a looped row, TP, alpha 0 and every refusal of
  the loop geometry and the file structure.
- With WEIGHTLESS_TEST_GLP_DIR set, the published GLP-44 file on a fake of
  the checkpoint's shape (22 layers x 2 loops x 3072); with
  WEIGHTLESS_TEST_CONFIG_DIR, the loop geometry from the config.json.
- ``RealNanbeige``: SGLang's real ``NanbeigeModel.forward`` loop and
  ``NanbeigeDecoderLayer.forward``, built with ``__new__``; SGLang's own
  unrolled-depth capture (``layers_to_capture``, logical ids) is compared
  with the steered stream.
- ``StructureNanbeige``: a parse of ``models/nanbeige.py`` and the linear
  layer it relies on, in every SGLang tree found.
"""
import ast
import dataclasses
import importlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import glpfiles  # noqa: E402  (sets up the import paths)
from glpfiles import good_meta, write_gguf  # noqa: E402

from test_archs import (_trees, _parse, _cls, _fn, _src, _calls_where, _returns_in_order,
                         _assign, unit_dirs)  # noqa: E402

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.archs import ARCH  # noqa: E402
from weightless_sglang.archs import nanbeige as nanbeige_row  # noqa: E402
from weightless_sglang.archs.base import ArchRow  # noqa: E402
from weightless_sglang.archs.nanbeige import LoopSpec, config_loop, read_loop_idx  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402
from weightless_steer.container import load_control_vector, read_gguf_cvec  # noqa: E402


# ---------------------------------------------------------------- fakes


def _rms(x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class _PairLayer(nn.Module):
    """residual <- hidden + residual (the fold); write <- f(norm(residual))."""

    def __init__(self, layer_id, width, loops=1):
        super().__init__()
        self.layer_id = layer_id
        g = torch.Generator().manual_seed(1000 + layer_id)
        self.w = nn.Parameter(torch.randn(loops, width, generator=g) * 0.3, requires_grad=False)

    def write(self, residual, loop=0):
        return torch.tanh(_rms(residual) * self.w[loop].to(residual.dtype))

    def fold(self, hidden_states, residual):
        return hidden_states if residual is None else hidden_states + residual


def _wrapper(name, attr="model"):
    def init(self, inner):
        nn.Module.__init__(self)
        setattr(self, attr, inner)

    def forward(self, *a, **k):
        return getattr(self, attr).forward(*a, **k)

    return type(name, (nn.Module,), {"__init__": init, "forward": forward})


def _runner(model, hf_type, tp_size=1, pp_rank=0, pp_size=1, dtype=torch.float32):
    return types.SimpleNamespace(
        model=model, is_draft_worker=False, tp_rank=0, tp_size=tp_size, pp_rank=pp_rank,
        pp_size=pp_size,
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf_type),
                                           dtype=dtype))


class NanbeigeDecoderLayer(_PairLayer):
    def forward(self, positions, hidden_states, forward_batch, loop_idx, residual):
        residual = self.fold(hidden_states, residual)
        return self.write(residual, loop_idx), residual


class NanbeigeModel(nn.Module):
    def __init__(self, num_layers=22, num_loops=2, width=8, skip_loop_final_norm=False,
                 by_keyword=False):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=width, num_hidden_layers=num_layers,
                                            num_loops=num_loops,
                                            skip_loop_final_norm=skip_loop_final_norm,
                                            model_type="nanbeige")
        self.layers = nn.ModuleList(NanbeigeDecoderLayer(i, width, loops=num_loops)
                                    for i in range(num_layers))
        self.start_layer, self.end_layer = 0, num_layers
        self.by_keyword = by_keyword
        self.calls = []  # (loop_idx, layer id) in call order

    def forward(self, hidden_states=None):
        residual = None
        self.calls = []
        for loop_idx in range(self.config.num_loops):
            for i in range(self.start_layer, self.end_layer):
                self.calls.append((loop_idx, i))
                layer = self.layers[i]
                if self.by_keyword:
                    hidden_states, residual = layer(positions=None, hidden_states=hidden_states,
                                                    forward_batch=None, loop_idx=loop_idx,
                                                    residual=residual)
                else:
                    hidden_states, residual = layer(None, hidden_states, None, loop_idx, residual)
            if loop_idx != self.config.num_loops - 1:
                hidden_states = hidden_states + residual
                residual = None
                if not self.config.skip_loop_final_norm:
                    hidden_states = _rms(hidden_states)
        return hidden_states, residual


NanbeigeForCausalLM = _wrapper("NanbeigeForCausalLM")


def nanbeige_runner(num_layers=22, num_loops=2, width=8, **kw):
    bb = NanbeigeModel(num_layers, num_loops, width,
                       **{k: kw.pop(k) for k in ("skip_loop_final_norm", "by_keyword") if k in kw})
    return _runner(NanbeigeForCausalLM(bb), "nanbeige", **kw)


def nanbeige_reference(run, x, steps, alpha):
    """``steps``: {execution step: direction}. Step = loop x L + layer."""
    bb = run.model.model
    L, loops = bb.config.num_hidden_layers, bb.config.num_loops
    h, r = x, None
    for loop in range(loops):
        for i in range(L):
            lay = bb.layers[i]
            r = lay.fold(h, r)
            h = lay.write(r, loop)
            e = loop * L + i
            if e in steps:
                s = (h + r).double()
                d = steps[e].double()
                h = (s - alpha * (s @ d).unsqueeze(-1) * d - r.double()).to(h.dtype)
        if loop != loops - 1:
            h, r = h + r, None
            if not bb.config.skip_loop_final_norm:
                h = _rms(h)
    return h, r


# ---------------------------------------------------------------- the row on the fakes


HOOK = "residual_stream_post_layer"
NB = "NanbeigeForCausalLM"

# The published file (huggingface.co/msuiche/<repo name>).
NAN_FILE = glpfiles.find_glp("glp.nanbeige42-GLP-44-L1-44-a2.gguf")
CONFIG_DIRS = [d.strip() for d in os.environ.get("WEIGHTLESS_TEST_CONFIG_DIR", "").split(os.pathsep)
               if d.strip()]


def config(repo):
    """The config.json of ``<org>__<repo>`` under WEIGHTLESS_TEST_CONFIG_DIR, or None."""
    for d in CONFIG_DIRS:
        p = os.path.join(d, repo, "config.json")
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    return None


def dirs_of(path, shift=0):
    """{id - shift: unit direction} and the default alpha of a GLP file."""
    meta, raw = load_control_vector(path, hook=HOOK)
    return ({int(k) - shift: (v.double() / v.double().norm()).float() for k, v in raw.items()},
            float(meta.get("glp.alpha_default", 1.0)))


def stream(out):
    h, r = out
    return h if r is None else h + r


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        saved = dict(ARCH)
        self.addCleanup(lambda: (ARCH.clear(), ARCH.update(saved)))
        self.n = 0

    def write(self, ids, width, *, alpha="1.5", hint=None, structure=None, method=None,
              seed=0, extra=None):
        m = good_meta(layers=ids, alpha=alpha)
        if hint:
            m["controlvector.model_hint"] = hint
        if structure:
            m["glp.structure"] = structure
        if method:
            m["glp.method"] = method
        m.update(extra or {})
        self.n += 1
        path = os.path.join(self.tmp.name, f"v{self.n}.gguf")
        write_gguf(path, m, unit_dirs(ids, width, seed))
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        return path

    def install(self, run, **env):
        for k, v in env.items():
            os.environ[k] = str(v)
        err = io.StringIO()
        with redirect_stderr(err):
            rec = install_steering(run, source="env")
        self.log = err.getvalue()
        return rec

    def patch_row(self, name, **kw):
        ARCH[name] = dataclasses.replace(ARCH[name], **kw)

    def close(self, out, ref, atol=2e-5):
        for a, b in zip(out, ref):
            if a is None or b is None:
                self.assertIs(a, b)
            else:
                torch.testing.assert_close(a, b, atol=atol, rtol=1e-5)


def fwd(run, x):
    return run.model.forward(hidden_states=x)


class NanbeigeRows(Base):
    W, L, LOOPS = 16, 22, 2

    def looped_file(self, steps=None, **kw):
        """direction.N steers execution step N-1 (ids 1..44 by default)."""
        steps = range(self.L * self.LOOPS) if steps is None else steps
        kw.setdefault("structure", "per-execution-step")
        return self.write([s + 1 for s in steps], self.W, hint="nanbeige", **kw)

    def test_call_styles_match_the_reference(self):
        for kw in (dict(), dict(by_keyword=True), dict(skip_loop_final_norm=True)):
            with self.subTest(**kw):
                path = self.looped_file()
                run = nanbeige_runner(self.L, self.LOOPS, self.W, **kw)
                rec = self.install(run)
                self.assertEqual(rec["row"], NB)
                self.assertEqual(rec["exec_id"], "looped")
                self.assertEqual(rec["local_layer_ids"], list(range(44)))
                self.assertEqual(rec["physical_layers"], list(range(22)))
                self.assertEqual(rec["loop"], {
                    "per_loop": 22, "num_loops": 2, "exec_steps": 44, "loop_arg": 3,
                    "loop_kwarg": "loop_idx", "file_entry": "direction.N = execution step N-1"})
                self.assertIn("exec=looped(2x22, direction.N=step N-1)", self.log)
                x = torch.randn(5, self.W, generator=torch.Generator().manual_seed(3))
                out = fwd(run, x)
                steps, _ = dirs_of(path, shift=1)
                self.close(out, nanbeige_reference(run, x, steps, 1.5))
                # direction.N on step N (no shift) is a different model
                wrong = {k + 1: v for k, v in steps.items() if k + 1 < 44}
                self.assertGreater(float((stream(out) - stream(
                    nanbeige_reference(run, x, wrong, 1.5))).abs().max()), 1e-2)

    def test_steps_fire_in_execution_order(self):
        self.looped_file()
        run = nanbeige_runner(self.L, self.LOOPS, self.W)
        self.install(run)
        order = []
        orig = wcore.FiredCheck.bump
        with mock.patch.object(wcore.FiredCheck, "bump",
                               lambda s, i, x: (order.append(i), orig(s, i, x))[1]):
            fwd(run, torch.randn(3, self.W))
        self.assertEqual(order, list(range(44)))
        self.assertEqual(run.model.model.calls, [(l, i) for l in range(2) for i in range(22)])

    def test_layer_filter_names_execution_steps(self):
        path = self.looped_file()
        run = nanbeige_runner(self.L, self.LOOPS, self.W)
        rec = self.install(run, WEIGHTLESS_STEER_LAYERS="0,21,22,43")
        self.assertEqual(rec["local_layer_ids"], [0, 21, 22, 43])
        self.assertEqual(rec["physical_layers"], [0, 21])
        x = torch.randn(4, self.W)
        steps, _ = dirs_of(path, shift=1)
        keep = {k: v for k, v in steps.items() if k in (0, 21, 22, 43)}
        self.close(fwd(run, x), nanbeige_reference(run, x, keep, 1.5))

    def test_a_file_on_one_loop_pass_leaves_the_other_stock(self):
        path = self.looped_file(steps=range(22, 44))
        run = nanbeige_runner(self.L, self.LOOPS, self.W)
        self.assertEqual(self.install(run)["local_layer_ids"], list(range(22, 44)))
        x = torch.randn(4, self.W)
        steps, _ = dirs_of(path, shift=1)
        self.close(fwd(run, x), nanbeige_reference(run, x, steps, 1.5))

    def test_forward_wrap_looped(self):
        self.patch_row(NB, install="forward_wrap")
        path = self.looped_file()
        run = nanbeige_runner(self.L, self.LOOPS, self.W, by_keyword=True)
        self.install(run)
        self.assertEqual(sum("forward" in vars(l) for l in run.model.model.layers), 22)
        x = torch.randn(4, self.W)
        steps, _ = dirs_of(path, shift=1)
        self.close(fwd(run, x), nanbeige_reference(run, x, steps, 1.5))

    def test_alpha_zero_is_bitwise_stock(self):
        self.looped_file()
        run = nanbeige_runner(self.L, self.LOOPS, self.W)
        self.install(run, WEIGHTLESS_STEER_ALPHA="0")
        x = torch.randn(4, self.W)
        out, stock = fwd(run, x), fwd(nanbeige_runner(self.L, self.LOOPS, self.W), x)
        self.assertTrue(torch.equal(out[0], stock[0]) and torch.equal(out[1], stock[1]))

    def test_tp_is_allowed(self):
        self.looped_file()
        rec = self.install(nanbeige_runner(self.L, self.LOOPS, self.W, tp_size=2))
        self.assertEqual(rec["tp_size"], 2)

    def test_wrong_loop_slot_fails_closed(self):
        for arg in (2, 4):
            with self.subTest(arg=arg):
                self.patch_row(NB, loop=config_loop(per_loop="num_hidden_layers",
                                                    num_loops="num_loops", arg=arg,
                                                    kwarg="loop_idx"))
                self.looped_file()
                run = nanbeige_runner(self.L, self.LOOPS, self.W)
                self.install(run)
                with self.assertRaisesRegex(wcore.SiteError, "loop index"):
                    fwd(run, torch.randn(3, self.W))
        self.patch_row(NB, loop=config_loop(per_loop="num_hidden_layers", num_loops="num_loops",
                                            arg=3, kwarg="loop"))
        self.looped_file()
        run = nanbeige_runner(self.L, self.LOOPS, self.W, by_keyword=True)
        self.install(run)
        with self.assertRaisesRegex(wcore.SiteError, "without its loop index"):
            fwd(run, torch.randn(3, self.W))

    def test_a_tensor_loop_index_fails_closed(self):
        self.looped_file()
        run = nanbeige_runner(self.L, self.LOOPS, self.W)
        self.install(run)
        layer = run.model.model.layers[0]
        with self.assertRaisesRegex(wcore.SiteError, "got loop index tensor"):
            layer(None, torch.randn(2, self.W), None, torch.tensor(0), None)

    def test_fewer_loops_at_serve_time_fail_closed(self):
        self.looped_file()
        run = nanbeige_runner(self.L, self.LOOPS, self.W)
        self.install(run)
        run.model.model.config.num_loops = 1
        with self.assertRaisesRegex(RuntimeError, r"22 steered decoder-layer output\(s\) did not run"):
            fwd(run, torch.randn(3, self.W))

    def test_structure_and_geometry_refusals(self):
        cases = [
            ("per-layer file", dict(structure="per-layer"), None, "not 'per-execution-step'"),
            ("no structure", dict(structure=None), None, "not 'per-execution-step'"),
            ("per-layer method", dict(method="dom_per_layer"), None, "contradicts itself"),
            ("45 steps", dict(steps=range(45)), None, r"exec steps \[44\] out of range"),
            ("width", dict(), dict(width=lambda c: 2 * c.hidden_size), "width 16 != 32"),
            ("per_loop", dict(), dict(loop=config_loop(per_loop="hidden_size", num_loops="num_loops",
                                                       arg=3, kwarg="loop_idx")), "backbone holds 22"),
            ("no loop", dict(), dict(loop=None), "declares no loop geometry"),
            ("not looped", dict(), dict(exec_id="layer"), "runs each layer once"),
            ("hint", dict(hint="nemotron_h"), None, "model_hint='nemotron_h'"),
        ]
        for name, fkw, rkw, msg in cases:
            with self.subTest(name):
                ARCH[NB] = ArchRow(backbone=("model",), **nanbeige_row._NANBEIGE)
                if rkw:
                    self.patch_row(NB, **rkw)
                fkw = dict(fkw)
                if fkw.get("structure", "x") is None:
                    fkw.pop("structure")
                    self.write([s + 1 for s in range(44)], self.W, hint="nanbeige")
                elif "hint" in fkw:
                    self.write([s + 1 for s in range(44)], self.W, hint=fkw["hint"],
                               structure="per-execution-step")
                else:
                    self.looped_file(**fkw)
                with self.assertRaisesRegex(RuntimeError, msg):
                    self.install(nanbeige_runner(self.L, self.LOOPS, self.W))

    def test_read_loop_idx(self):
        spec = LoopSpec(per_loop=22, num_loops=2, arg=3, kwarg="loop_idx")
        self.assertEqual(read_loop_idx(spec, (0, 1, 2, 1), {}, 5), 1)
        self.assertEqual(read_loop_idx(spec, (0, 1, 2), {"loop_idx": 0}, 5), 0)
        for bad in (2, -1, True, 1.0, None):
            with self.subTest(bad=bad), self.assertRaises(wcore.SiteError):
                read_loop_idx(spec, (0, 1, 2, bad), {}, 5)
        with self.assertRaises(wcore.SiteError):
            read_loop_idx(spec, (0, 1, 2), {}, 5)
        self.assertEqual((spec.steps, spec.physical(30), spec.loop_of(30), spec.exec_id(1, 8)),
                         (44, 8, 1, 30))


@unittest.skipUnless(glpfiles.have(NAN_FILE), "set WEIGHTLESS_TEST_GLP_DIR to the folder with "
                     "the Nanbeige4.2-3B GLP-44 file")
class PublishedFiles(Base):
    def run_file(self, run, path, ref, dirs, alpha, width, tokens=3):
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        rec = self.install(run)
        x = torch.randn(tokens, width, generator=torch.Generator().manual_seed(7))
        out = fwd(run, x)
        self.close(out, ref(run, x, dirs, alpha), atol=5e-5)
        return rec, out

    def test_metadata(self):
        want = {NAN_FILE: ("nanbeige", 44, 3072, "2.0", "per-execution-step", (NB,))}
        for path, (hint, n, width, alpha, structure, rows) in want.items():
            with self.subTest(file=os.path.basename(path)):
                meta, raw = read_gguf_cvec(path)
                self.assertEqual(meta["controlvector.model_hint"], hint)
                for row in rows:  # served under a wrapper config, the row names the hint
                    self.assertIn(hint, ARCH[row].hint)
                self.assertEqual(meta["glp.hook_point"], HOOK)
                self.assertEqual(float(meta["glp.alpha_default"]), float(alpha))
                self.assertEqual(str(meta.get("glp.structure")), structure)
                ids = sorted(int(k.rsplit(".", 1)[-1]) for k in raw)
                self.assertEqual(ids, list(range(1, n + 1)))
                self.assertEqual({int(np.asarray(v).size) for v in raw.values()}, {width})

    def test_nanbeige_glp44(self):
        L, loops, width = 22, 2, 3072
        cfg = config("Nanbeige__Nanbeige4.2-3B")
        if cfg is not None:
            self.assertEqual((cfg["num_hidden_layers"], cfg["num_loops"], cfg["hidden_size"],
                              cfg["skip_loop_final_norm"]), (L, loops, width, False))
        meta, _ = read_gguf_cvec(NAN_FILE)
        # The file's glp.layer_ids_zero_based says 1..44, but its description and
        # the model card says direction.N = execution step N-1 (44 steps, 0..43).
        # The key is not read for a per-execution-step file.
        ids = [int(t) for t in str(meta["glp.layer_ids_zero_based"]).split(",")]
        self.assertEqual(ids, list(range(1, 45)))
        steps, alpha = dirs_of(NAN_FILE, shift=1)
        self.assertEqual(alpha, 2.0)
        for kw in (dict(), dict(by_keyword=True), dict(skip_loop_final_norm=True)):
            with self.subTest(**kw):
                run = nanbeige_runner(L, loops, width, **kw)
                rec, out = self.run_file(run, NAN_FILE, nanbeige_reference, steps, alpha, width)
                self.assertEqual(rec["local_layer_ids"], list(range(44)))
                self.assertEqual(rec["physical_layers"], list(range(22)))
                self.assertEqual((rec["width"], rec["alpha"], rec["model_hint"]),
                                 (width, 2.0, "nanbeige"))
        x = torch.randn(3, width, generator=torch.Generator().manual_seed(7))
        unshifted = {k + 1: v for k, v in steps.items() if k + 1 < 44}
        wrong = nanbeige_reference(run, x, unshifted, alpha)
        self.assertGreater(float((stream(fwd(run, x)) - stream(wrong)).abs().max()), 1e-2)


# ---------------------------------------------------------------- SGLang's real classes (R)


def _try(name):
    try:
        return importlib.import_module(name), None
    except Exception as e:  # no SGLang, or a build without this model
        return None, f"{type(e).__name__}: {e}"


NBM, NB_WHY = _try("sglang.srt.models.nanbeige")


def rms(x, eps=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def real_dirs_of(path, shift=0):
    """{id - shift: unit direction as the plugin stores it (fp32)} and the
    file's default alpha."""
    meta, raw = load_control_vector(path, hook=HOOK)
    return ({int(k) - shift: (v.double() / v.double().norm()).float().double()
             for k, v in raw.items()}, float(meta.get("glp.alpha_default", 1.0)))


def project(s, d, alpha):
    d = d.to(s.dtype)
    return s - alpha * (s @ d).unsqueeze(-1) * d


def named(name, attr):
    """A served model class ``name`` whose ``attr`` path leads to the backbone."""

    def init(self, inner):
        nn.Module.__init__(self)
        setattr(self, attr, inner)

    def forward(self, *a, **k):
        return getattr(self, attr)(*a, **k)

    return type(name, (nn.Module,), {"__init__": init, "forward": forward})


def runner(model, hf_type, dtype, tp_size=1):
    return types.SimpleNamespace(
        model=model, is_draft_worker=False, tp_rank=0, tp_size=tp_size, pp_rank=0, pp_size=1,
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf_type),
                                           dtype=dtype))


class RealBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        saved = dict(ARCH)
        self.addCleanup(lambda: (ARCH.clear(), ARCH.update(saved)))
        self.n = 0

    def write(self, ids, width, alpha="1.5", hint=None, structure=None):
        m = good_meta(layers=ids, alpha=alpha)
        if hint:
            m["controlvector.model_hint"] = hint
        if structure:
            m["glp.structure"] = structure
        self.n += 1
        path = os.path.join(self.tmp.name, f"v{self.n}.gguf")
        write_gguf(path, m, unit_dirs(ids, width))
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        return path

    def install(self, run):
        with redirect_stderr(io.StringIO()):
            return install_steering(run, source="env")


class _NNorm(nn.Module):
    def __init__(self, rec=None, key=None):
        super().__init__()
        self.rec, self.key = rec, key

    def forward(self, x, residual=None):
        if residual is None:
            if self.rec is not None:
                self.rec.append((self.key, x.clone()))
            return rms(x)
        s = x + residual
        if self.rec is not None:
            self.rec.append((self.key, s.clone()))
        return rms(s), s


class _NAttn(nn.Module):
    def __init__(self, h, i, loops):
        super().__init__()
        g = torch.Generator().manual_seed(100 + i)
        self.w = nn.Parameter(torch.randn(loops, h, generator=g, dtype=torch.float64) * 0.5,
                              requires_grad=False)

    def forward(self, positions, hidden_states, forward_batch, loop_idx):
        return torch.tanh(hidden_states * self.w[loop_idx])


class _NMLP(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.scale = 0.3 + 0.01 * i

    def forward(self, x, use_reduce_scatter=False):
        return x * self.scale


def build_nanbeige(L, loops, h, skip_norm=False, rec=None):
    m = NBM.NanbeigeModel.__new__(NBM.NanbeigeModel)
    nn.Module.__init__(m)
    m.config = types.SimpleNamespace(num_hidden_layers=L, num_loops=loops,
                                     skip_loop_final_norm=skip_norm, hidden_size=h,
                                     model_type="nanbeige")
    m.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    layers = []
    for i in range(L):
        layer = NBM.NanbeigeDecoderLayer.__new__(NBM.NanbeigeDecoderLayer)
        nn.Module.__init__(layer)
        layer.self_attn = _NAttn(h, i, loops)
        layer.mlp = _NMLP(i)
        layer.input_layernorm = _NNorm(rec, i)
        layer.post_attention_layernorm = _NNorm()
        layers.append(layer)
    m.layers = nn.ModuleList(layers)
    m.start_layer, m.end_layer = 0, L
    m.norm = _NNorm(rec, "norm")
    m.layers_to_capture = []
    m.register_buffer("_anchor", torch.zeros(1, dtype=torch.float64), persistent=False)
    return m


def real_nanbeige_reference(m, x, steps, alpha):
    """The stream after each execution step (steered where a direction is
    given), and the model output."""
    L, loops = m.config.num_hidden_layers, m.config.num_loops
    s, after = x.clone(), []
    for loop in range(loops):
        for i in range(L):
            lay = m.layers[i]
            s = s + torch.tanh(rms(s) * lay.self_attn.w[loop])
            s = s + rms(s) * lay.mlp.scale
            e = loop * L + i
            if e in steps:
                s = project(s, steps[e], alpha)
            after.append(s)
        if loop != loops - 1 and not m.config.skip_loop_final_norm:
            s = rms(s)
    return after, rms(s)


@unittest.skipUnless(NBM is not None, f"SGLang's Nanbeige model is not importable here ({NB_WHY})")
class RealNanbeige(RealBase):
    L, LOOPS = 22, 2

    def run_model(self, h, skip_norm=False, capture=(), tokens=3):
        rec = []
        m = build_nanbeige(self.L, self.LOOPS, h, skip_norm, rec)
        m.layers_to_capture = list(capture)
        run = runner(named("NanbeigeForCausalLM", "model")(m), "nanbeige", torch.float64)
        return m, run, rec

    def forward(self, run, h, tokens=3):
        x = torch.randn(tokens, h, dtype=torch.float64, generator=torch.Generator().manual_seed(9))
        return x, run.model(None, torch.arange(tokens), None, input_embeds=x.clone())

    def check(self, h, path, steps, alpha, skip_norm=False):
        capture = [1, 21, 22, 23, 43]
        m, run, rec = self.run_model(h, skip_norm, capture)
        info = self.install(run)
        self.assertEqual(info["local_layer_ids"], sorted(steps))
        order = []
        orig = wcore.FiredCheck.bump
        with mock.patch.object(wcore.FiredCheck, "bump",
                               lambda s, i, x: (order.append(i), orig(s, i, x))[1]):
            x, (out, aux) = self.forward(run, h)
        self.assertEqual(order, sorted(steps))
        after, ref = real_nanbeige_reference(m, x, steps, alpha)
        torch.testing.assert_close(out, ref, atol=1e-7, rtol=1e-7)
        # SGLang's own capture before logical id k is the stream step k-1 left
        for k, a in zip(capture, aux):
            want = after[k - 1]
            if k == self.L and not skip_norm:  # the pass boundary norms the stream
                want = rms(want)
            torch.testing.assert_close(a, want, atol=1e-7, rtol=1e-7, msg=f"capture {k}")
        # every step's input norm is handed the stream the previous step left
        into = [s for key, s in rec if key != "norm"]
        self.assertEqual(len(into), 44)
        for e in range(1, 44):
            want = after[e - 1] if e != self.L or skip_norm else rms(after[e - 1])
            torch.testing.assert_close(into[e], want, atol=1e-7, rtol=1e-7, msg=f"step {e}")
        stock_m, stock_run, _ = self.run_model(h, skip_norm)
        _, stock = self.forward(stock_run, h)
        self.assertGreater(float((out - stock).abs().max()), 1e-3)
        return after

    def test_real_loop_real_layers(self):
        for skip_norm in (False, True):
            with self.subTest(skip_loop_final_norm=skip_norm):
                path = self.write(range(1, 45), 32, hint="nanbeige", structure="per-execution-step")
                steps, alpha = real_dirs_of(path, shift=1)
                self.check(32, path, steps, alpha, skip_norm)

    def test_last_step_alpha_one(self):
        path = self.write(range(1, 45), 32, alpha="1.0", structure="per-execution-step")
        steps, alpha = real_dirs_of(path, shift=1)
        after = self.check(32, path, steps, alpha)
        self.assertLess(float((after[43] @ steps[43]).abs().max()), 1e-6)

    def test_one_pass_only(self):
        path = self.write(range(23, 45), 32, structure="per-execution-step")
        steps, alpha = real_dirs_of(path, shift=1)
        self.assertEqual(sorted(steps), list(range(22, 44)))
        self.check(32, path, steps, alpha)

    @unittest.skipUnless(glpfiles.have(NAN_FILE), "set WEIGHTLESS_TEST_GLP_DIR to the folder with "
                         "the Nanbeige4.2-3B GLP-44 file")
    def test_published_glp44(self):
        os.environ["WEIGHTLESS_STEER_PATH"] = NAN_FILE
        steps, alpha = real_dirs_of(NAN_FILE, shift=1)
        self.assertEqual(alpha, 2.0)
        self.check(3072, NAN_FILE, steps, alpha)


# ---------------------------------------------------------------- structure


MODEL_FILE = "srt/models/nanbeige.py"


@unittest.skipUnless(_trees(MODEL_FILE), "no SGLang package with models/nanbeige.py found")
class StructureNanbeige(unittest.TestCase):
    """Pins, per SGLang tree, the code paths the Nanbeige row depends on."""

    def each(self, rel):
        return [(t, _parse(t, rel)) for t in _trees(MODEL_FILE)]

    def test_nanbeige_classes_backbone_and_pp(self):
        row = ARCH["NanbeigeForCausalLM"]
        self.assertEqual((row.backbone, set(row.layers)), (("model",), {"NanbeigeDecoderLayer"}))
        for t, m in self.each("srt/models/nanbeige.py"):
            with self.subTest(tree=t):
                self.assertIn("NanbeigeForCausalLM", _src(_assign(m, "EntryClass")))
                self.assertIn("self.model = NanbeigeModel(",
                              _src(_fn(_cls(m, "NanbeigeForCausalLM"), "__init__")))
                init = _src(_fn(_cls(m, "NanbeigeModel"), "__init__"))
                self.assertIn("assert pp_size == 1", init)
                self.assertIn("make_layers(config.num_hidden_layers", init)
                self.assertIn("decoder_layer_type = decoder_layer_type or NanbeigeDecoderLayer", init)

    def test_nanbeige_loop_and_loop_index_slot(self):
        row = ARCH["NanbeigeForCausalLM"]
        per_loop, num_loops, arg, kwarg = row.loop.fields
        self.assertEqual((row.exec_id, per_loop, num_loops), ("looped", "num_hidden_layers",
                                                              "num_loops"))
        for t, m in self.each("srt/models/nanbeige.py"):
            with self.subTest(tree=t):
                fwd = _fn(_cls(m, "NanbeigeModel"), "forward")
                loops = [n for n in ast.walk(fwd) if isinstance(n, ast.For)]
                self.assertEqual([_src(l.iter) for l in loops],
                                 ["range(self.config.num_loops)", "range(self.start_layer, self.end_layer)"])
                self.assertEqual(_src(loops[0].target), "loop_idx")
                s = _src(fwd)
                self.assertIn("num_physical_layers = self.config.num_hidden_layers", s)
                self.assertIn("logical_id = loop_idx * num_physical_layers + i", s)
                calls = _calls_where(fwd, lambda c: _src(c.func) in ("layer", "layer.forward"))
                self.assertEqual([_src(c) for c in calls],
                                 ["layer(positions, hidden_states, forward_batch, loop_idx, residual)"])
                self.assertEqual(_src(calls[0].args[arg]), kwarg)
                self.assertIn("hidden_states, residual = layer(", _src(loops[1]))
                lf = _fn(_cls(m, "NanbeigeDecoderLayer"), "forward")
                names = [a.arg for a in lf.args.args][1:]
                self.assertEqual(names.index(kwarg), arg)
                self.assertEqual(_returns_in_order(lf), ["(hidden_states, residual)"])
                self.assertNotIn("tbo", _src(m))

    def test_nanbeige_layer_output_is_reduced_inside_the_layer(self):
        # decides tp="ok": o_proj and down_proj keep RowParallelLinear's
        # default all-reduce, nothing asks them to skip it, and no layer
        # communicator (which could defer it) is used
        self.assertEqual(ARCH["NanbeigeForCausalLM"].tp, "ok")
        for t, m in self.each("srt/models/nanbeige.py"):
            with self.subTest(tree=t):
                s = _src(m)
                for word in ("LayerCommunicator", "get_forward", "reduce_results",
                             "_sglang_needs_allreduce_fusion", "UnreducedOutput"):
                    self.assertNotIn(word, s)
                rpl = _calls_where(m, lambda c: _src(c.func) == "RowParallelLinear")
                self.assertEqual(len(rpl), 2)
                attn = _cls(m, "NanbeigeAttention")
                self.assertIn("output, _ = self.o_proj(attn_output)", _src(_fn(attn, "forward")))
                mlp = _cls(m, "NanbeigeMLP")
                mf = _fn(mlp, "forward")
                self.assertEqual(_src(mf.args.defaults[-1]), "False")
                self.assertIn("x, _ = self.down_proj(x, skip_all_reduce=use_reduce_scatter)", _src(mf))
                lf = _src(_fn(_cls(m, "NanbeigeDecoderLayer"), "forward"))
                self.assertIn("hidden_states = self.mlp(hidden_states)", lf)
                self.assertEqual(len(_calls_where(_fn(_cls(m, "NanbeigeDecoderLayer"), "forward"),
                                            lambda c: _src(c.func) == "self.mlp")), 1)
        for t, m in self.each("srt/layers/linear.py"):
            with self.subTest(tree=t):
                c = _cls(m, "RowParallelLinear")
                init = _fn(c, "__init__")
                args = [a.arg for a in init.args.args]
                defaults = dict(zip(args[len(args) - len(init.args.defaults):], init.args.defaults))
                self.assertEqual(_src(defaults["reduce_results"]), "True")
                fwd = _fn(c, "forward")
                args = [a.arg for a in fwd.args.args]
                defaults = dict(zip(args[len(args) - len(fwd.args.defaults):], fwd.args.defaults))
                self.assertEqual(_src(defaults["skip_all_reduce"]), "False")


if __name__ == "__main__":
    unittest.main(verbosity=2)
