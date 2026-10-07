"""The Nemotron-3.5-Lightning rows (``NemotronHForCausalLM``,
``NemotronHPuzzleForCausalLM``), CPU only, and the ``forward_wrap`` install
kind they use.

Every layer returns ``(hidden, residual)`` with the add left to the next
layer, so the stream is ``hidden + residual``. The model loop calls
``layer.forward(...)``, where a forward hook never runs, so the row wraps
each steered layer's instance ``forward``.

- Fakes (the first section) keep the class names, the backbone path, the
  loop call style and the output tuple; the installed forward is compared
  with the edit written out by hand. Plus the pipeline ranks, TP, alpha 0
  and every boot refusal the rows rely on, and ``core.wrap_forward`` itself.
- With WEIGHTLESS_TEST_GLP_DIR set, the published GLP-51 file on a fake with
  the checkpoint's layer pattern (52 x 2688); with WEIGHTLESS_TEST_CONFIG_DIR,
  the pattern and width from the checkpoint's config.json.
- ``RealNemotron``: SGLang's real ``NemotronHModel.forward`` over the real
  four decoder-layer classes (with the real layer communicator where the
  tree has it), built with ``__new__``; only the mixers and the norms are
  stubs. Every steered site fires, the next layer receives the steered
  stream, the last layer is steered, and the draft aux capture sees it.
- ``StructureNemotronH``: a parse of ``models/nemotron_h.py`` in every
  SGLang tree found (the installed package and
  WEIGHTLESS_TEST_SGLANG_TREES, os.pathsep separated).
"""
import ast
import contextlib
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

from test_archs import (_trees, _parse, _cls, _fn, _has_fn, _src, _calls_where,
                         _returns_in_order, _assign, unit_dirs)  # noqa: E402

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.archs import ARCH  # noqa: E402
from weightless_sglang.core import wrap_forward  # noqa: E402
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


class _NemotronLayer(_PairLayer):
    def forward(self, *, hidden_states, residual, forward_batch=None):
        residual = self.fold(hidden_states, residual)
        return self.write(residual), residual


NemotronHMambaDecoderLayer = type("NemotronHMambaDecoderLayer", (_NemotronLayer,), {})
NemotronHAttentionDecoderLayer = type("NemotronHAttentionDecoderLayer", (_NemotronLayer,), {})
NemotronHMLPDecoderLayer = type("NemotronHMLPDecoderLayer", (_NemotronLayer,), {})
NemotronHMoEDecoderLayer = type("NemotronHMoEDecoderLayer", (_NemotronLayer,), {})
NEMOTRON_KINDS = {"M": NemotronHMambaDecoderLayer, "*": NemotronHAttentionDecoderLayer,
                  "-": NemotronHMLPDecoderLayer, "E": NemotronHMoEDecoderLayer}
# The published checkpoint's pattern (nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4,
# config.json layers_block_type, as SGLang's NemotronHConfig turns it into
# hybrid_override_pattern): 52 layers, 23 Mamba, 23 MoE, 6 attention.
NEMOTRON35_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"


class NemotronHModel(nn.Module):
    def __init__(self, pattern, width, start=0, end=None, call="forward"):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=width, hybrid_override_pattern=pattern,
                                            model_type="nemotron_h")
        self.layers = nn.ModuleList(NEMOTRON_KINDS[k](i, width) if start <= i < (end or len(pattern))
                                    else nn.Identity() for i, k in enumerate(pattern))
        self.start_layer, self.end_layer = start, len(pattern) if end is None else end
        self.call = call  # "forward" as in SGLang; "call" only to show what a hook would need

    def forward(self, hidden_states=None, residual=None):
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if self.call == "forward":
                hidden_states, residual = layer.forward(hidden_states=hidden_states,
                                                        residual=residual, forward_batch=None)
            else:
                hidden_states, residual = layer(hidden_states=hidden_states, residual=residual,
                                                forward_batch=None)
        return hidden_states, residual


NemotronHForCausalLM = _wrapper("NemotronHForCausalLM")
NemotronHPuzzleForCausalLM = type("NemotronHPuzzleForCausalLM", (NemotronHForCausalLM,), {})


def nemotron_runner(pattern="M*E-MEM*E-M", width=8, cls=None, **kw):
    bb = NemotronHModel(pattern, width, **{k: kw.pop(k) for k in ("start", "end", "call") if k in kw})
    return _runner((cls or NemotronHForCausalLM)(bb), "nemotron_h", **kw)


def nemotron_reference(run, x, dirs, alpha):
    """The Nemotron fake's forward with the edit written out by hand:
    h = hidden + residual; h' = h - alpha (h.d) d; hidden' = h' - residual."""
    bb = run.model.model
    h, r = x, None
    for i in range(bb.start_layer, bb.end_layer):
        L = bb.layers[i]
        r = L.fold(h, r)
        h = L.write(r)
        if i in dirs:
            s = (h + r).double()
            d = dirs[i].double()
            h = (s - alpha * (s @ d).unsqueeze(-1) * d - r.double()).to(h.dtype)
    return h, r


# ---------------------------------------------------------------- the row on the fakes


HOOK = "residual_stream_post_layer"
NEM, NEMP = "NemotronHForCausalLM", "NemotronHPuzzleForCausalLM"

# The published file (huggingface.co/msuiche/<repo name>).
NEM_FILE = glpfiles.find_glp("Nemotron-3.5-Lightning-30B-A3B-abliterated-GLP-51-L1-51-a1.0.gguf")
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


PAT = "M*E-MEM*E-M"  # every layer kind; 11 layers


class NemotronRows(Base):
    W = 16

    def test_both_classes_match_the_reference(self):
        for cls in (NemotronHForCausalLM, NemotronHPuzzleForCausalLM):
            with self.subTest(cls=cls.__name__):
                path = self.write(range(1, len(PAT)), self.W, hint="nemotron_h")
                run = nemotron_runner(PAT, self.W, cls=cls)
                rec = self.install(run)
                self.assertEqual(rec["row"], cls.__name__)
                self.assertEqual(rec["install"], "forward_wrap")
                self.assertEqual(rec["site"], "decoder-layer forward")
                self.assertEqual(rec["local_layer_ids"], list(range(1, len(PAT))))
                self.assertEqual(rec["exec_id"], "layer")
                self.assertNotIn("loop", rec)
                x = torch.randn(5, self.W, generator=torch.Generator().manual_seed(1))
                out = fwd(run, x)
                dirs, _ = dirs_of(path)
                self.close(out, nemotron_reference(run, x, dirs, 1.5))
                stock = fwd(nemotron_runner(PAT, self.W, cls=cls), x)
                self.assertGreater(float((stream(out) - stream(stock)).abs().max()), 1e-2)

    def test_forward_is_wrapped_on_the_instance_not_hooked(self):
        self.write(range(1, len(PAT)), self.W)
        run = nemotron_runner(PAT, self.W)
        self.install(run)
        bb = run.model.model
        self.assertEqual([i for i, l in enumerate(bb.layers) if "forward" in vars(l)],
                         list(range(1, len(PAT))))
        self.assertFalse(any(l._forward_hooks for l in bb.layers))
        for layer in bb.layers:  # the class forward is untouched
            self.assertNotIn("_weightless_layer_id", vars(type(layer).forward))

    def test_a_forward_hook_would_never_run(self):
        # the SGLang loop calls layer.forward(...): with install=hook the fired
        # check fails the first forward instead of serving unsteered
        self.patch_row(NEM, install="hook")
        self.write(range(1, len(PAT)), self.W)
        run = nemotron_runner(PAT, self.W)
        self.install(run)
        with self.assertRaisesRegex(RuntimeError, r"10 steered decoder-layer output\(s\) did not run"):
            fwd(run, torch.randn(3, self.W))

    def test_layer_call_style_reaches_the_wrapper_too(self):
        path = self.write(range(1, len(PAT)), self.W)
        run = nemotron_runner(PAT, self.W, call="call")
        self.install(run)
        x = torch.randn(4, self.W)
        dirs, _ = dirs_of(path)
        self.close(fwd(run, x), nemotron_reference(run, x, dirs, 1.5))

    def test_class_forward_call_fails_closed(self):
        # a loop that calls the class forward goes past the instance wrapper
        class ClassCall(NemotronHModel):
            def forward(self, hidden_states=None, residual=None):
                for i in range(self.start_layer, self.end_layer):
                    layer = self.layers[i]
                    hidden_states, residual = type(layer).forward(
                        layer, hidden_states=hidden_states, residual=residual, forward_batch=None)
                return hidden_states, residual

        self.write(range(1, len(PAT)), self.W)
        run = _runner(NemotronHForCausalLM(ClassCall(PAT, self.W)), "nemotron_h")
        self.install(run)
        with self.assertRaisesRegex(RuntimeError, "did not run in this forward"):
            fwd(run, torch.randn(3, self.W))

    def test_pipeline_ranks_equal_one_rank(self):
        path = self.write(range(1, len(PAT)), self.W)
        x = torch.randn(4, self.W)
        one = nemotron_runner(PAT, self.W)
        self.install(one)
        ref = fwd(one, x)
        r0 = nemotron_runner(PAT, self.W, end=6, pp_size=2)
        r1 = nemotron_runner(PAT, self.W, start=6, pp_rank=1, pp_size=2)
        self.assertEqual(self.install(r0)["local_layer_ids"], list(range(1, 6)))
        self.assertEqual(self.install(r1)["local_layer_ids"], list(range(6, 11)))
        h, r = fwd(r0, x)
        out = r1.model.forward(hidden_states=h, residual=r)
        self.close(out, ref, atol=0)
        dirs, _ = dirs_of(path)
        self.close(out, nemotron_reference(one, x, dirs, 1.5))

    def test_alpha_zero_is_bitwise_stock(self):
        self.write(range(1, len(PAT)), self.W)
        run = nemotron_runner(PAT, self.W)
        self.install(run, WEIGHTLESS_STEER_ALPHA="0")
        x = torch.randn(4, self.W)
        out, stock = fwd(run, x), fwd(nemotron_runner(PAT, self.W), x)
        for a, b in zip(out, stock):
            self.assertTrue(torch.equal(a, b))

    def test_tp_is_refused(self):
        self.write(range(1, len(PAT)), self.W)
        with self.assertRaisesRegex(RuntimeError, "not validated at TP=2"):
            self.install(nemotron_runner(PAT, self.W, tp_size=2))

    def test_unknown_layer_class_and_width_are_refused(self):
        self.write(range(1, len(PAT)), self.W)
        run = nemotron_runner(PAT, self.W)
        run.model.model.layers[3] = type("NemotronHNewDecoderLayer", (_NemotronLayer,), {})(3, self.W)
        with self.assertRaisesRegex(RuntimeError, r"layer 3 is NemotronHNewDecoderLayer"):
            self.install(run)
        self.write(range(1, len(PAT)), 2 * self.W)
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.install(nemotron_runner(PAT, self.W))

    def test_a_second_install_does_not_wrap_twice(self):
        self.write(range(1, len(PAT)), self.W)
        run = nemotron_runner(PAT, self.W)
        rec = self.install(run)
        self.assertIs(self.install(run), rec)  # same file: installed once
        run._weightless_steer_installed = None
        with self.assertRaisesRegex(RuntimeError, "already wrapped"):
            self.install(run)


# ---------------------------------------------------------------- forward_wrap

class _Toy(nn.Module):
    def forward(self, x, scale=1.0):
        return x * scale, x


class ForwardWrap(unittest.TestCase):
    def test_both_call_styles_run_the_codec(self):
        m = _Toy()
        seen = []

        def codec(module, args, output):
            seen.append((module, args))
            return output[0] + 1, output[1]

        undo = wrap_forward(m, codec, layer_id=4)
        x = torch.ones(2)
        self.assertTrue(torch.equal(m.forward(x)[0], x + 1))
        self.assertTrue(torch.equal(m(x, scale=2.0)[0], 2 * x + 1))
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0][0], m)
        self.assertEqual(m.forward._weightless_layer_id, 4)
        undo()
        self.assertNotIn("forward", vars(m))
        self.assertTrue(torch.equal(m(x)[0], x))

    def test_with_kwargs_codec_sees_the_call(self):
        m = _Toy()
        got = {}

        def codec(module, args, kwargs, output):
            got.update(args=args, kwargs=kwargs)
            return output

        h = wrap_forward(m, codec, layer_id=0, with_kwargs=True)
        m(torch.ones(1), scale=3.0)
        self.assertEqual(got["kwargs"], {"scale": 3.0})
        self.assertEqual(len(got["args"]), 1)
        h.remove()
        self.assertNotIn("forward", vars(m))

    def test_a_second_wrap_is_refused_and_a_foreign_forward_is_restored(self):
        m = _Toy()
        foreign = m.forward
        m.forward = lambda x, scale=1.0: foreign(x, scale)  # someone else's instance forward
        mine = m.forward
        undo = wrap_forward(m, lambda mod, a, o: o, layer_id=1)
        with self.assertRaisesRegex(RuntimeError, "already wrapped"):
            wrap_forward(m, lambda mod, a, o: o, layer_id=1)
        undo()
        self.assertIs(vars(m)["forward"], mine)

    def test_a_layer_without_instance_attributes_is_refused(self):
        class Slotted:
            __slots__ = ()

            def forward(self, x):
                return x

        with self.assertRaisesRegex(RuntimeError, "takes no instance attributes"):
            wrap_forward(Slotted(), lambda mod, a, o: o, layer_id=0)

    def test_the_codec_is_the_hook_codec(self):
        # a forward_wrap row and a hook row give the same bits on the same layer
        W = 16
        d = torch.randn(W, generator=torch.Generator().manual_seed(0))
        outs = []
        for mode in ("hook", "forward_wrap"):
            with self.subTest(mode=mode):
                owner = nn.Module()
                owner.register_buffer("_steer_stack", torch.zeros(3, 1, W))
                owner.register_buffer("_steer_alpha", torch.tensor(1.25))
                owner._steer_stack[1, 0] = d / d.norm()
                layer = NemotronHMambaDecoderLayer(1, W)
                h = wcore.make_post_layer_hook(owner, 1, arity=2, hidden_index=0, residual_index=1,
                                            width=W)
                if mode == "hook":
                    layer.register_forward_hook(h)
                    out = layer(hidden_states=torch.ones(3, W), residual=torch.ones(3, W))
                else:
                    wrap_forward(layer, h, layer_id=1)
                    out = layer.forward(hidden_states=torch.ones(3, W), residual=torch.ones(3, W))
                outs.append(out)
        self.assertTrue(torch.equal(outs[0][0], outs[1][0]))
        self.assertTrue(torch.equal(outs[0][1], outs[1][1]))
        stock = NemotronHMambaDecoderLayer(1, W)(hidden_states=torch.ones(3, W),
                                                   residual=torch.ones(3, W))
        self.assertFalse(torch.equal(outs[0][0], stock[0]))


@unittest.skipUnless(glpfiles.have(NEM_FILE), "set WEIGHTLESS_TEST_GLP_DIR to the folder with "
                     "the Nemotron-3.5-Lightning GLP-51 file")
class PublishedFiles(Base):
    def run_file(self, run, path, ref, dirs, alpha, width, tokens=3):
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        rec = self.install(run)
        x = torch.randn(tokens, width, generator=torch.Generator().manual_seed(7))
        out = fwd(run, x)
        self.close(out, ref(run, x, dirs, alpha), atol=5e-5)
        return rec, out

    def test_metadata(self):
        want = {NEM_FILE: ("nemotron_h", 51, 2688, "1.0", "per-layer", (NEM, NEMP))}
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

    def test_nemotron_glp51(self):
        pattern = NEMOTRON35_PATTERN
        cfg = config("nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
        if cfg is not None:
            kinds = {"mamba": "M", "moe": "E", "attention": "*", "mlp": "-"}
            self.assertEqual("".join(kinds[t] for t in cfg["layers_block_type"]), pattern)
            self.assertEqual(cfg["hidden_size"], 2688)
        dirs, alpha = dirs_of(NEM_FILE)
        for cls in (NemotronHForCausalLM, NemotronHPuzzleForCausalLM):
            with self.subTest(cls=cls.__name__):
                run = nemotron_runner(pattern, 2688, cls=cls)
                rec, _ = self.run_file(run, NEM_FILE, nemotron_reference, dirs, alpha, 2688)
                self.assertEqual(rec["local_layer_ids"], list(range(1, 52)))
                self.assertEqual((rec["width"], rec["alpha"], rec["model_hint"]),
                                 (2688, 1.0, "nemotron_h"))
                self.assertEqual(rec["install"], "forward_wrap")


# ---------------------------------------------------------------- SGLang's real classes (R)


def _try(name):
    try:
        return importlib.import_module(name), None
    except Exception as e:  # no SGLang, or a build without this model
        return None, f"{type(e).__name__}: {e}"


NH, NH_WHY = _try("sglang.srt.models.nemotron_h")


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


NEM_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"  # the published checkpoint's layer pattern


class _Recorder:
    """Records the stream each norm is handed (hidden + residual)."""

    def __init__(self):
        self.seen = {}


class _NemNorm(nn.Module):
    def __init__(self, rec, key):
        super().__init__()
        self.rec, self.key = rec, key

    def forward(self, x, residual=None, post_residual_addition=None):
        if residual is None:
            self.rec.seen[self.key] = x.clone()
            return rms(x)
        residual.add_(x)
        self.rec.seen[self.key] = residual.clone()
        return rms(residual), residual


class _NemMixer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states, **kwargs):
        return hidden_states * self.scale


def nem_scale(i, kind):
    return (0.5 if kind in "M*" else 0.25) * (1 + 0.01 * i)


def build_nemotron(pattern, width, rec):
    """SGLang's NemotronHModel with its real decoder layers; the norms and
    mixers are stubs. The upstream tree builds the real layer communicator
    (as test_nemotron_h_aux_capture.py does); older trees' layers take the
    norm directly and only ask the communicator whether to fuse."""
    config = types.SimpleNamespace(hybrid_override_pattern=pattern, hidden_size=width,
                                   model_type="nemotron_h")
    m = NH.NemotronHModel.__new__(NH.NemotronHModel)
    nn.Module.__init__(m)
    m.config = config
    m.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
    m.start_layer, m.end_layer = 0, len(pattern)
    m.norm_f = _NemNorm(rec, "final")
    m.layers_to_capture = set()
    layers = []
    for i, kind in enumerate(pattern):
        cls = NH.ALL_DECODER_LAYER_TYPES[kind]
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.norm = _NemNorm(rec, i)
        layer.mixer = _NemMixer(nem_scale(i, kind))
        layer.layer_id = layer.layer_idx = i
        if hasattr(layer, "_init_layer_communicator"):
            if kind in "M*":
                layer._init_layer_communicator(config, i)
            else:
                layer._init_layer_communicator(config, i, is_sparse=False)
        else:
            layer.layer_communicator = types.SimpleNamespace(
                should_fuse_mlp_allreduce_with_next_layer=lambda fb: False)
            if hasattr(layer, "_set_prev_layer_is_attn"):
                layer._set_prev_layer_is_attn(config, i)
        if kind == "M":
            layer._forward_mamba = lambda h, batch, mixer=layer.mixer: mixer(h)
        layers.append(layer)
    m.layers = nn.ModuleList(layers)
    m.register_buffer("_anchor", torch.zeros(1, dtype=torch.float64), persistent=False)
    return m


@contextlib.contextmanager
def nemotron_context():
    """TP=1 and no DP attention, as the upstream test sets them."""
    with contextlib.ExitStack() as st:
        if hasattr(NH, "is_dp_attention_enabled"):
            st.enter_context(mock.patch.object(NH, "is_dp_attention_enabled", lambda: False))
        from sglang.srt import runtime_context as rc
        if hasattr(rc, "get_context") and hasattr(rc, "get_flags"):
            from sglang.srt.layers import communicator as comm
            group = types.SimpleNamespace(all_reduce=lambda x: x)
            st.enter_context(rc.get_context().override_server_args(tp_size=1,
                                                                   enable_dp_attention=False))
            st.enter_context(rc.get_flags().dp.override(enabled=False))
            st.enter_context(rc.get_parallel().override(
                attn_tp_group=group, launch_world_rank=0, tp_rank=0, tp_size=1, attn_tp_rank=0,
                attn_tp_size=1, attn_dp_rank=0, attn_dp_size=1, attn_cp_rank=0, attn_cp_size=1,
                moe_tp_rank=0, moe_tp_size=1, moe_ep_rank=0, moe_ep_size=1, moe_dp_rank=0,
                moe_dp_size=1))
            for name in ("get_moe_cp_size", "apply_flashinfer_allreduce_fusion",
                         "apply_aiter_all_reduce_fusion"):
                if hasattr(comm, name):
                    st.enter_context(mock.patch.object(
                        comm, name, return_value=1 if name == "get_moe_cp_size" else False))
        yield


def real_nemotron_reference(pattern, x, dirs, alpha):
    """The stream after each layer (steered where a direction is given)."""
    s, after = x.clone(), []
    for k, kind in enumerate(pattern):
        s = s + rms(s) * nem_scale(k, kind)
        if k in dirs:
            s = project(s, dirs[k], alpha)
        after.append(s)
    return after


@unittest.skipUnless(NH is not None, f"SGLang's Nemotron-H model is not importable here ({NH_WHY})")
class RealNemotron(RealBase):
    def run_model(self, pattern, width, path=None, install="forward_wrap", tp_size=1,
                  dirs=None, alpha=None, tokens=3, capture=()):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        rec = _Recorder()
        with nemotron_context():
            bb = build_nemotron(pattern, width, rec)
            bb.layers_to_capture = set(capture)
            run = runner(named("NemotronHForCausalLM", "model")(bb), "nemotron_h", torch.float64,
                         tp_size=tp_size)
            if install != "forward_wrap":
                ARCH["NemotronHForCausalLM"] = dataclasses.replace(
                    ARCH["NemotronHForCausalLM"], install=install)
            info = self.install(run) if path else None
            batch = types.SimpleNamespace(input_ids=torch.zeros(tokens, dtype=torch.long),
                                          forward_mode=ForwardMode.DECODE,
                                          global_num_token_non_padded_cpu=tokens)
            x = torch.randn(tokens, width, dtype=torch.float64,
                            generator=torch.Generator().manual_seed(5))
            out = run.model(batch.input_ids, torch.arange(tokens), batch, inputs_embeds=x.clone())
        return info, x, out, rec, bb

    def check(self, pattern, width, path, dirs, alpha):
        info, x, out, rec, bb = self.run_model(pattern, width, path)
        n = len(pattern)
        self.assertEqual(info["install"], "forward_wrap")
        self.assertEqual(info["local_layer_ids"], sorted(dirs))
        self.assertEqual(sum("forward" in vars(l) for l in bb.layers), len(dirs))
        after = real_nemotron_reference(pattern, x, dirs, alpha)
        # each layer is handed the stream the previous layer left, steered
        for k in range(1, n):
            torch.testing.assert_close(rec.seen[k], after[k - 1], atol=1e-7, rtol=1e-7,
                                       msg=f"stream into layer {k}")
        # the last layer is steered: the final norm sees its edited stream
        torch.testing.assert_close(rec.seen["final"], after[-1], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(out, rms(after[-1]), atol=1e-7, rtol=1e-7)
        stock = self.run_model(pattern, width)[2]
        self.assertGreater(float((out - stock).abs().max()), 1e-3)
        return after

    def test_real_loop_real_layers(self):
        pattern, width = "M*E-MEM*E-M", 32
        path = self.write(range(1, len(pattern)), width, hint="nemotron_h")
        dirs, alpha = real_dirs_of(path)
        self.check(pattern, width, path, dirs, alpha)

    def test_checkpoint_pattern_alpha_one_last_layer(self):
        width = 64
        path = self.write(range(1, 52), width, alpha="1.0")
        dirs, alpha = real_dirs_of(path)
        after = self.check(NEM_PATTERN, width, path, dirs, alpha)
        # alpha 1 removes the last direction from the stream the norm reads
        self.assertLess(float((after[-1] @ dirs[51]).abs().max()), 1e-6)

    @unittest.skipUnless(glpfiles.have(NEM_FILE), "set WEIGHTLESS_TEST_GLP_DIR to the folder with "
                         "the Nemotron-3.5-Lightning GLP-51 file")
    def test_published_glp51(self):
        os.environ["WEIGHTLESS_STEER_PATH"] = NEM_FILE
        dirs, alpha = real_dirs_of(NEM_FILE)
        self.check(NEM_PATTERN, 2688, NEM_FILE, dirs, alpha)

    def test_aux_capture_sees_the_steered_stream(self):
        """SGLang's draft aux capture (layers_to_capture) reads hidden +
        residual at the top of the next iteration, after the wrapped forward
        returned: it sees the steered stream, the last boundary included.
        Boundary 2 follows an attention layer that feeds a MoE layer (the
        upstream capture reduces a copy there)."""
        pattern, width = "M*E-MEM*E-M", 32
        path = self.write(range(1, len(pattern)), width, hint="nemotron_h")
        dirs, alpha = real_dirs_of(path)
        capture = (2, 5, len(pattern))
        _, x, out, _, _ = self.run_model(pattern, width, path, capture=capture)
        hidden, aux = out
        after = real_nemotron_reference(pattern, x, dirs, alpha)
        self.assertEqual(len(aux), len(capture))
        for b, got in zip(capture, aux):
            torch.testing.assert_close(got, after[b - 1], atol=1e-7, rtol=1e-7,
                                       msg=f"aux capture at boundary {b}")
        torch.testing.assert_close(hidden, rms(after[-1]), atol=1e-7, rtol=1e-7)

    def test_a_forward_hook_is_never_called_by_this_loop(self):
        self.write(range(1, 11), 16)
        with self.assertRaisesRegex(RuntimeError, r"10 steered decoder-layer output\(s\) did not run"):
            self.run_model("M*E-MEM*E-M", 16, path=True, install="hook")

    def test_tp_is_refused(self):
        self.write(range(1, 11), 16)
        with self.assertRaisesRegex(RuntimeError, "not validated at TP=2"):
            self.run_model("M*E-MEM*E-M", 16, path=True, tp_size=2)


# ---------------------------------------------------------------- structure


MODEL_FILE = "srt/models/nemotron_h.py"


@unittest.skipUnless(_trees(MODEL_FILE), "no SGLang package with models/nemotron_h.py found")
class StructureNemotronH(unittest.TestCase):
    """Pins, per SGLang tree, the code paths the Nemotron-H rows depend on."""

    def each(self, rel):
        return [(t, _parse(t, rel)) for t in _trees(MODEL_FILE)]

    def test_nemotron_classes_and_backbone(self):
        row = ARCH["NemotronHForCausalLM"]
        for t, m in self.each("srt/models/nemotron_h.py"):
            with self.subTest(tree=t):
                self.assertEqual(_src(_assign(m, "EntryClass")),
                                 "[NemotronHForCausalLM, NemotronHPuzzleForCausalLM]")
                puzzle = _cls(m, "NemotronHPuzzleForCausalLM")
                self.assertEqual([_src(b) for b in puzzle.bases], ["NemotronHForCausalLM"])
                self.assertEqual([type(n).__name__ for n in puzzle.body], ["Pass"])
                causal = _cls(m, "NemotronHForCausalLM")
                self.assertIn("self.model = self._init_model(", _src(_fn(causal, "__init__")))
                self.assertEqual([r.split("(")[0] for r in _returns_in_order(_fn(causal, "_init_model"))],
                                 ["NemotronHModel"])
                types_ = _assign(m, "ALL_DECODER_LAYER_TYPES")
                self.assertEqual({_src(v) for v in types_.values}, set(row.layers))
                self.assertEqual(ARCH["NemotronHPuzzleForCausalLM"], row)
                init = _src(_fn(_cls(m, "NemotronHModel"), "__init__"))
                self.assertIn("layer_class = ALL_DECODER_LAYER_TYPES[config.hybrid_override_pattern[idx]]",
                              init)
                self.assertIn("self.layers, self.start_layer, self.end_layer = make_layers(", init)

    def test_nemotron_loop_calls_layer_forward(self):
        self.assertEqual(ARCH["NemotronHForCausalLM"].install, "forward_wrap")
        for t, m in self.each("srt/models/nemotron_h.py"):
            with self.subTest(tree=t):
                fwd = _fn(_cls(m, "NemotronHModel"), "forward")
                loops = [n for n in ast.walk(fwd) if isinstance(n, ast.For)]
                self.assertEqual(len(loops), 1)
                self.assertEqual(_src(loops[0].iter), "range(self.start_layer, self.end_layer)")
                s = _src(loops[0])
                self.assertIn("layer = self.layers[i]", s)
                calls = _calls_where(fwd, lambda c: _src(c.func) in ("layer", "layer.forward"))
                self.assertEqual([_src(c) for c in calls], [
                    "layer.forward(hidden_states=hidden_states, residual=residual, "
                    "forward_batch=forward_batch)"])
                self.assertIn("hidden_states, residual = layer.forward(", s)
                self.assertNotIn("tbo", _src(m))

    def test_nemotron_layers_return_pairs(self):
        classes = ("NemotronHMLPLikeDecoderLayer", "NemotronHAttnLikeDecoderLayer",
                   "NemotronHMambaDecoderLayer", "NemotronHAttentionDecoderLayer",
                   "NemotronHMLPDecoderLayer", "NemotronHMoEDecoderLayer")
        for t, m in self.each("srt/models/nemotron_h.py"):
            with self.subTest(tree=t):
                n_fwd = 0
                for name in classes:
                    c = _cls(m, name)
                    if not _has_fn(c, "forward"):
                        continue
                    n_fwd += 1
                    fn = _fn(c, "forward")
                    self.assertEqual([a.arg for a in fn.args.kwonlyargs],
                                     ["hidden_states", "residual", "forward_batch"])
                    for r in _returns_in_order(fn):
                        self.assertTrue(r.startswith("(") and r.endswith(", residual)"), (name, r))
                self.assertGreaterEqual(n_fwd, 2)
                for leaf, base in (("NemotronHMLPDecoderLayer", "NemotronHMLPLikeDecoderLayer"),
                                   ("NemotronHMoEDecoderLayer", "NemotronHMLPLikeDecoderLayer"),
                                   ("NemotronHMambaDecoderLayer", "NemotronHAttnLikeDecoderLayer"),
                                   ("NemotronHAttentionDecoderLayer", "NemotronHAttnLikeDecoderLayer")):
                    self.assertEqual([_src(b) for b in _cls(m, leaf).bases], [base])

    def test_nemotron_partial_sums_decide_tp_refuse(self):
        # a layer before an MLP/MoE layer hands over a TP partial (upstream: a
        # plain tensor with skip_reduce; older trees: a tensor marked for the
        # fused all-reduce) -> the row must refuse TP > 1
        self.assertEqual(ARCH["NemotronHForCausalLM"].tp, "refuse")
        for t, m in self.each("srt/models/nemotron_h.py"):
            with self.subTest(tree=t):
                s = _src(m)
                self.assertTrue("skip_reduce = self.feeds_mlp_layer or fuse_mlp_allreduce" in s
                                or "_sglang_needs_allreduce_fusion = True" in s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
