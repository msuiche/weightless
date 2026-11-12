"""The Qwen3.8-27B rows (``Qwen3_5ForConditionalGeneration``,
``Qwen3_5ForCausalLM``), CPU only.

- A fake SGLang runner with the qwen3_5 layout (the layers return
  ``(hidden, residual)`` and the next layer folds them), used here and by
  test_core.py and test_plugin.py.
- The two published files (GLP-49 and GLP-63) through install_steering: they
  land in the stack exactly as SteeringCore reads them. Needs
  WEIGHTLESS_TEST_GLP_DIR.
- A parse of SGLang's qwen3_5.py and qwen3_5_text.py (the installed package
  and every package folder in WEIGHTLESS_TEST_SGLANG_TREES, os.pathsep
  separated) that pins what the rows rely on: ``StructureQwen35`` on the
  older layout (each layer returns (hidden, residual)), ``StructureBoundary``
  on the stage-boundary layout (upstream #41553: each layer returns hidden
  and the residual stays in forward_batch.residual_stream).
- ``RealModel``: SGLang's real Qwen3_5ForConditionalGeneration built by its
  own __init__ from WEIGHTLESS_TEST_QWEN_CONFIG (width shrunk,
  sglang_tiny.py), on this SGLang's layout: the edit at each site against
  steer_delta by hand, alpha 0, the fired count, three pipeline ranks.
- ``StreamHookRefusals``: the residual-stream site's refusals, on SGLang's
  real ResidualStream (stage-boundary layout only).
"""
import ast
import os
import sys
import types
import unittest
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import glpfiles  # noqa: E402  (sets up the import paths)

from test_archs import (_trees, _parse, _cls, _fn, _src, _calls, _entry_names, _returns,
                         _loops, _layer_calls, unit_dirs, stage_built)  # noqa: E402
from test_archs import sglang_tiny as tiny  # noqa: E402

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.archs import ARCH  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402
from weightless_steer.container import read_gguf_cvec  # noqa: E402
from weightless_steer.core import SteeringCore  # noqa: E402


# ---------------------------------------------------------------- the fake runner


class _Layer(nn.Module):
    """Returns (hidden, residual) like SGLang's qwen3_5 decoder layers:
    residual <- hidden + residual (the fold), hidden <- a layer write."""

    def __init__(self, layer_id, width):
        super().__init__()
        self.layer_id = layer_id
        self.w = nn.Parameter(torch.full((width,), 0.01 * (layer_id + 1)), requires_grad=False)

    def forward(self, positions=None, hidden_states=None, residual=None, forward_batch=None, **kw):
        residual = hidden_states if residual is None else hidden_states + residual
        return residual * self.w, residual


Qwen3_5LinearDecoderLayer = type("Qwen3_5LinearDecoderLayer", (_Layer,), {})
Qwen3_5AttentionDecoderLayer = type("Qwen3_5AttentionDecoderLayer", (_Layer,), {})
OtherLayer = type("SomeOtherDecoderLayer", (_Layer,), {})


class Backbone(nn.Module):
    def __init__(self, num_layers=64, width=8, start=0, end=None, model_type="qwen3_5_text",
                 layer_cls=None):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=width, model_type=model_type)
        self.layers = nn.ModuleList(
            (layer_cls or (Qwen3_5AttentionDecoderLayer if i % 4 == 3 else Qwen3_5LinearDecoderLayer))(i, width)
            for i in range(num_layers))
        self.start_layer = start
        self.end_layer = num_layers if end is None else end

    def forward(self, x):
        h, r = x, None
        for i in range(self.start_layer, self.end_layer):
            h, r = self.layers[i](hidden_states=h, residual=r)
        return h + r


Qwen3_5ForConditionalGeneration = type(
    "Qwen3_5ForConditionalGeneration", (nn.Module,),
    {"__init__": lambda self, bb: (nn.Module.__init__(self), setattr(self, "model", bb))[0]})
Unsupported = type(
    "LlamaForCausalLM", (nn.Module,),
    {"__init__": lambda self, bb: (nn.Module.__init__(self), setattr(self, "model", bb))[0]})


def runner(num_layers=64, width=8, draft=False, hf_model_type="qwen3_5", cls=None, **kw):
    bb = Backbone(num_layers, width, **kw)
    model = (cls or Qwen3_5ForConditionalGeneration)(bb)
    return types.SimpleNamespace(
        model=model, is_draft_worker=draft, tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf_model_type)))


# ---------------------------------------------------------------- the published files

CASES = ((glpfiles.GLP49, list(range(10, 59))), (glpfiles.GLP63, list(range(1, 64))))


@unittest.skipUnless(glpfiles.have(glpfiles.GLP49, glpfiles.GLP63),
                     "set WEIGHTLESS_TEST_GLP_DIR to the folder with the two GLP files")
class RealFiles(unittest.TestCase):
    def test_both_files(self):
        for path, layers in CASES:
            with self.subTest(path=os.path.basename(path)), \
                    mock.patch.dict(os.environ, {"WEIGHTLESS_STEER_PATH": path}, clear=False):
                for k in ("WEIGHTLESS_STEER_ALPHA", "WEIGHTLESS_STEER_LAYERS", "WEIGHTLESS_STEER_DIAG"):
                    os.environ.pop(k, None)
                meta, _ = read_gguf_cvec(path)
                self.assertEqual(meta["controlvector.model_hint"], "qwen3_5")
                self.assertEqual(meta["glp.hook_point"], "residual_stream_post_layer")
                run = runner(width=5120)
                rec = install_steering(run, source="env")
                core = SteeringCore.from_env(hook="residual_stream_post_layer",
                                             num_layers=64, hidden_size=5120)
                bb = run.model.model
                self.assertEqual(rec["hooked_layers"], layers)
                self.assertEqual(sorted(core.dirs), layers)
                self.assertEqual(rec["alpha"], 1.0)
                for i in range(64):
                    row = bb._steer_stack[i, 0]
                    if i in core.dirs:
                        self.assertTrue(torch.equal(row, core.dirs[i]))
                        self.assertAlmostEqual(float(row.double().norm()), 1.0, places=6)
                    else:
                        self.assertEqual(float(row.abs().sum()), 0.0)
                full = [i for i in layers if i % 4 == 3]
                self.assertEqual(rec["full_attn_layers"], full)
                print(f"  {os.path.basename(path)}: {len(layers)} layers "
                      f"{layers[0]}..{layers[-1]}, {len(full)} full-attn, sha256 {rec['file_sha256'][:16]}")


# ---------------------------------------------------------------- structure


QWEN_LAYERS = {"Qwen3_5LinearDecoderLayer", "Qwen3_5AttentionDecoderLayer"}
QWEN35_ALL = _trees("srt/models/qwen3_5_text.py")
QWEN35_TREES = [t for t in QWEN35_ALL if not stage_built(t, "srt/models/qwen3_5.py", QWEN_LAYERS)]
QWEN35_STAGE_TREES = [t for t in QWEN35_ALL if stage_built(t, "srt/models/qwen3_5.py", QWEN_LAYERS)]


@unittest.skipUnless(QWEN35_TREES, "no SGLang package with models/qwen3_5_text.py on the older "
                                   "layout found (StructureBoundary reads the stage-boundary trees)")
class StructureQwen35(unittest.TestCase):
    """Qwen3.8-27B, the two rows ``Qwen3_5ForConditionalGeneration`` and
    ``Qwen3_5ForCausalLM``: both served classes hold the same backbone
    (``qwen3_5.Qwen3_5ForCausalLM``) at ``.model``; its loop calls
    ``layer(...)`` through ``__call__`` (where the forward hook runs) and
    unpacks (hidden, residual); the layers return that pair; the pair of the
    last layer is folded by the final norm after the loop, so the last
    steered layer's edit reaches it."""

    def trees(self):
        return [(t, _parse(t, "srt/models/qwen3_5.py")) for t in QWEN35_TREES]

    def test_served_classes_and_backbones(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                self.assertIn("Qwen3_5ForConditionalGeneration", _entry_names(q))
                w = _cls(q, "Qwen3_5ForConditionalGeneration")
                self.assertEqual([_src(b) for b in w.bases], ["Qwen3VLForConditionalGeneration"])
                init = _src(_fn(w, "__init__"))
                self.assertIn("language_model_cls=Qwen3_5ForCausalLM", init)
                self.assertIn("super().__init__(config, quant_config, prefix, language_model_cls)", init)
                vl = _parse(t, "srt/models/qwen3_vl.py")
                self.assertIn("self.model = language_model_cls(",
                              _src(_fn(_cls(vl, "Qwen3VLForConditionalGeneration"), "__init__")))
                self.assertEqual(ARCH["Qwen3_5ForConditionalGeneration"].backbone, ("model",))
                # the text-only entry is a wrapper that holds the same body at .model
                tx = _parse(t, "srt/models/qwen3_5_text.py")
                self.assertIn("Qwen3_5ForCausalLM", _entry_names(tx))
                ct = _cls(tx, "Qwen3_5ForCausalLM")
                self.assertIn("body_cls = qwen3_5.Qwen3_5ForCausalLM", _src(ct))
                self.assertIn("self.model = self.body_cls(", _src(_fn(ct, "__init__")))
                self.assertEqual(ARCH["Qwen3_5ForCausalLM"].backbone[0], "model")

    def test_backbone_and_layer_classes(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                body = _cls(q, "Qwen3_5ForCausalLM")
                self.assertIn("decoder_layer_types = ALL_DECODER_LAYER_TYPES", _src(body))
                init = _src(_fn(body, "__init__"))
                self.assertIn("layer_class = self.decoder_layer_types[layer_type]", init)
                self.assertIn("layer_id=idx", init)
                self.assertIn("self.layers, self._start_layer, self._end_layer = make_layers(", init)
                for prop in ("start_layer", "end_layer"):
                    _fn(body, prop)
                types_map = [n for n in q.body if isinstance(n, ast.Assign)
                             and _src(n.targets[0]) == "ALL_DECODER_LAYER_TYPES"]
                self.assertEqual(len(types_map), 1)
                self.assertEqual({_src(v) for v in types_map[0].value.values},
                                 set(ARCH["Qwen3_5ForConditionalGeneration"].layers))

    def test_model_loop_calls_each_layer_and_unpacks_two(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(q, "Qwen3_5ForCausalLM"), "forward")
                loops = _loops(fwd)
                self.assertEqual(len(loops), 1)
                self.assertEqual(_src(loops[0].iter), "range(self.start_layer, self.end_layer)")
                self.assertIn("layer = self.layers[layer_idx]", _src(loops[0]))
                calls = _layer_calls(loops[0])
                self.assertEqual(len(calls), 1)
                self.assertEqual(_src(calls[0].targets[0]), "(hidden_states, residual)")
                self.assertTrue({"hidden_states", "residual"} <= {k.arg for k in calls[0].value.keywords})
                self.assertEqual(len(_calls(fwd, "layer")), 1)
                self.assertEqual(len(_calls(fwd, "forward")), 0)
                s = _src(fwd)
                self.assertNotIn("tbo", s)
                # PP hands on the pair
                self.assertIn("'hidden_states': hidden_states, 'residual': residual", s)
                self.assertIn("residual = pp_proxy_tensors['residual']", s)
                # where the last layer contracts: after the loop, the final norm
                # folds the last layer's pair (after finish_layer_stack, which only
                # completes a deferred reduction, where the tree has it)
                self.assertIn("self.norm(hidden_states, residual)", s)
                after = "\n".join(_src(n) for n in fwd.body if n.lineno > loops[0].end_lineno)
                self.assertIn("self.norm(hidden_states, residual)", after)
                if "finish_layer_stack" in s:
                    self.assertIn("hidden_states, residual = last_layer.layer_communicator."
                                  "finish_layer_stack(hidden_states, residual, forward_batch)", after)
                    comm = _parse(t, "srt/layers/communicator.py")
                    self.assertEqual(_returns(_fn(_cls(comm, "LayerCommunicator"), "finish_layer_stack")),
                                     ["(reduce_output(hidden_states), residual)"])

    def test_layers_return_the_pair(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                for name in ("Qwen3_5LinearDecoderLayer", "Qwen3_5AttentionDecoderLayer"):
                    fwd = _fn(_cls(q, name), "forward")
                    s = _src(fwd)
                    if "ffn_exit" in s:  # newer trees: FfnExit.finish returns the pair
                        self.assertEqual(_returns(fwd), ["ffn_exit.finish(hidden_states, residual)"])
                        comm = _parse(t, "srt/layers/communicator.py")
                        for r in _returns(_fn(_cls(comm, "FfnExit"), "finish")):
                            self.assertTrue(r.startswith("(") and r.endswith(", residual)")
                                            or r.startswith("self.communicator.postprocess_layer("), r)
                    else:  # older trees: the pair, a TP-partial sum carrying the marker
                        self.assertEqual(_returns(fwd), ["(hidden_states, residual)"])
                        self.assertIn("hidden_states = _finish_mlp_output(", s)
                        fin = _src([n for n in q.body if isinstance(n, ast.FunctionDef)
                                    and n.name == "_finish_mlp_output"][0])
                        self.assertIn(f"hidden_states.{wcore.PARTIAL_SUM_MARKER} = True", fin)


@unittest.skipUnless(QWEN35_STAGE_TREES, "no SGLang package with models/qwen3_5.py on the "
                                         "stage-boundary layout found (WEIGHTLESS_TEST_SGLANG_TREES)")
class StructureBoundary(unittest.TestCase):
    """The rows on the stage-boundary layout: the backbone and the served
    classes as before; each layer returns its FFN output alone through the
    FFN exit, whose plain-add update is left for the next layer's add; the
    loop calls ``layer(...)`` (the hook runs) with ``forward_batch`` by
    keyword (the hook reads the residual stream from it)."""

    def trees(self):
        return [(t, _parse(t, "srt/models/qwen3_5.py")) for t in QWEN35_STAGE_TREES]

    def test_served_classes_backbone_and_layer_classes(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                self.assertIn("Qwen3_5ForConditionalGeneration", _entry_names(q))
                w = _cls(q, "Qwen3_5ForConditionalGeneration")
                self.assertIn("language_model_cls=Qwen3_5ForCausalLM", _src(_fn(w, "__init__")))
                vl = _parse(t, "srt/models/qwen3_vl.py")
                self.assertIn("self.model = language_model_cls(",
                              _src(_fn(_cls(vl, "Qwen3VLForConditionalGeneration"), "__init__")))
                tx = _parse(t, "srt/models/qwen3_5_text.py")
                self.assertIn("self.model = self.body_cls(", _src(_fn(_cls(tx, "Qwen3_5ForCausalLM"),
                                                                     "__init__")))
                body = _cls(q, "Qwen3_5ForCausalLM")
                self.assertIn("decoder_layer_types = ALL_DECODER_LAYER_TYPES", _src(body))
                for prop in ("start_layer", "end_layer"):
                    _fn(body, prop)
                types_map = [n for n in q.body if isinstance(n, ast.Assign)
                             and _src(n.targets[0]) == "ALL_DECODER_LAYER_TYPES"]
                self.assertEqual({_src(v) for v in types_map[0].value.values},
                                 set(ARCH["Qwen3_5ForConditionalGeneration"].layers))

    def test_layers_leave_a_plain_add_through_the_ffn_exit(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                for name in sorted(QWEN_LAYERS):
                    cls = _cls(q, name)
                    fwd = _fn(cls, "forward")
                    self.assertEqual(_returns(fwd), ["ffn_exit.finish(hidden_states)"])
                    self.assertIn("with self.ffn_boundary.exit(forward_batch) as ffn_exit:", _src(fwd))
                    decl = _calls(_fn(cls, "__init__"), "declare_ffn")
                    self.assertEqual(len(decl), 1)
                    # no update= keyword: the default, the plain add
                    self.assertNotIn("update", {k.arg for k in decl[0].keywords})
                fac = _parse(t, "srt/layers/layer_boundary/factories.py")
                dfn = [n for n in fac.body if isinstance(n, ast.FunctionDef) and n.name == "declare_ffn"][0]
                self.assertIn("update=PLAIN_ADD", _src(dfn).replace(" ", ""))
                ex = _parse(t, "srt/layers/layer_boundary/exit.py")
                ro = _src(_fn(_cls(ex, "ExitPolicy"), "_record_output"))
                self.assertIn("return stream.record(hidden_states, update, declared_sum=declared_sum)", ro)

    def test_model_loop_calls_each_layer_with_its_forward_batch(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(q, "Qwen3_5ForCausalLM"), "forward")
                loops = _loops(fwd)
                self.assertEqual(len(loops), 1)
                self.assertEqual(_src(loops[0].iter), "range(self.start_layer, self.end_layer)")
                calls = _layer_calls(loops[0])
                self.assertEqual(len(calls), 1)
                self.assertEqual(_src(calls[0].targets[0]), "hidden_states")
                kws = {k.arg: _src(k.value) for k in calls[0].value.keywords}
                self.assertEqual(kws.get("forward_batch"), "forward_batch")
                self.assertEqual(kws.get("hidden_states"), "hidden_states")
                self.assertEqual(len(_calls(fwd, "forward")), 0)
                s = _src(fwd)
                self.assertNotIn("tbo", s)
                self.assertIn("residual_batch.start(forward_batch)", s)
                self.assertIn("return residual_batch.to_pp(hidden_states, forward_batch)", s)
                after = "\n".join(_src(n) for n in fwd.body if n.lineno > loops[0].end_lineno)
                self.assertIn("residual_batch.final_norm(", after)

    def test_the_stream_accessors_the_hook_reads(self):
        for t, _ in self.trees():
            with self.subTest(tree=t):
                b = _parse(t, "srt/layers/layer_boundary/residual/batch.py")
                fns = {n.name: n for n in b.body if isinstance(n, ast.FunctionDef)}
                self.assertEqual(_returns(fns["complete_output"]),
                                 ["stream_of(forward_batch).complete(hidden_states)"])
                self.assertEqual(_returns(fns["written_residual"]), ["stream_of(forward_batch).residual"])
                st = _cls(_parse(t, "srt/layers/layer_boundary/residual/stream.py"), "ResidualStream")
                self.assertEqual(_returns(_fn(st, "complete")),
                                 ["self.pending.complete() if self.pending is not None else hidden"])
                # the next layer takes the stream's own tensor back: the edit is written in place
                self.assertIn("valid = hidden is pending.value", _src(_fn(st, "check")))


# ---------------------------------------------------------------- the real model, either layout


QWEN_CFG = tiny.config_path("qwen")
HID = 64
QSTEERED = list(range(1, 64))  # like GLP-63


@unittest.skipIf(tiny.WHY, f"this SGLang cannot build the real models here ({tiny.WHY})")
@unittest.skipUnless(QWEN_CFG, "set WEIGHTLESS_TEST_QWEN_CONFIG to the Qwen3.8-27B config.json")
class RealModel(unittest.TestCase):
    """SGLang's real Qwen3_5ForConditionalGeneration built by its own
    __init__ (sglang_tiny.py) on this SGLang's layout."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        self.write("1.0")

    def write(self, alpha):
        m = glpfiles.good_meta(layers=QSTEERED, alpha=alpha)
        m["controlvector.model_hint"] = "qwen3_5"
        path = os.path.join(self.tmp.name, f"q-{alpha}.gguf")
        glpfiles.write_gguf(path, m, unit_dirs(QSTEERED, HID, seed=5))
        os.environ["WEIGHTLESS_STEER_PATH"] = path

    def build(self, dtype=torch.float32, rank=0, size=1, steer=True, mdir=None, before=None):
        import io
        from contextlib import redirect_stderr
        ctx = tiny.Context(pp_rank=rank, pp_size=size)
        model, bb, ctx = tiny.build("qwen", QWEN_CFG, hidden=HID, dtype=dtype, ctx=ctx)
        run = tiny.runner_of(model, "qwen3_5", dtype,
                             ranks=None if tiny.NEW_LAYOUT else (0, 1, rank, size))
        if before:
            before(bb)
        rec = None
        if steer:
            if mdir:
                os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = mdir
            with ctx, redirect_stderr(io.StringIO()) as err:
                rec = install_steering(run, source="env")
            self.log = err.getvalue()
            os.environ.pop("WEIGHTLESS_STEER_MANIFEST_DIR", None)
        return bb, ctx, run, rec

    def x(self, T=5, dtype=torch.float32):
        return torch.randn(T, HID, generator=torch.Generator().manual_seed(T)).to(dtype)

    def test_install(self):
        bb, ctx, run, rec = self.build()
        self.assertEqual((rec["row"], rec["width"], rec["num_layers"]),
                         ("Qwen3_5ForConditionalGeneration", HID, 64))
        self.assertEqual(rec["local_layer_ids"], QSTEERED)
        self.assertEqual(rec["full_attn_layers"], [i for i in QSTEERED if i % 4 == 3])
        hooked = [i for i in range(64) if bb.layers[i]._forward_hooks]
        self.assertEqual(hooked, QSTEERED)
        if tiny.NEW_LAYOUT:
            self.assertEqual((rec["site"], rec["layout"], rec["rank_source"]),
                             ("decoder-layer output (residual stream)", "stage-boundary", "get_parallel"))
            # the residual-stream hook reads forward_batch from the call's keywords
            self.assertTrue(all(bb.layers[i]._forward_hooks_with_kwargs for i in QSTEERED))
        else:
            self.assertEqual((rec["site"], rec["layout"], rec["rank_source"]),
                             ("decoder-layer output", "layer-output", "runner"))

    def test_each_site_writes_exactly_the_edit(self):
        """A spy registered before the steering sees each steered layer's
        (output, residual) before the edit, one registered after sees the
        output after it: the output after is steer_delta(output, residual)
        computed by hand, bit for bit, and the residual is untouched."""
        pre, post = {}, {}
        stores = []

        def spies(bb, store):
            for i in range(64):
                layer = bb.layers[i]
                if tiny.NEW_LAYOUT:
                    def h(m, a, kw, o, _i=i):
                        st = kw["forward_batch"].residual_stream
                        store[_i] = (o.detach().clone(), st.residual.detach().clone())
                    layer.register_forward_hook(h, with_kwargs=True)
                else:
                    layer.register_forward_hook(
                        lambda m, a, o, _i=i: store.__setitem__(_i, (o[0].detach().clone(),
                                                                     o[1].detach().clone())))

        bb, ctx, run, _ = self.build(torch.bfloat16, before=lambda b: spies(b, pre))
        spies(bb, post)
        tiny.run_backbone("qwen", bb, ctx, x=self.x(7, torch.bfloat16))
        for i in QSTEERED:
            x, r = pre[i]
            want = wcore.steer_delta(x, r, bb._steer_stack[i, 0], bb._steer_alpha)
            self.assertTrue(torch.equal(post[i][0], want), i)
            self.assertTrue(torch.equal(post[i][1], r), i)
            self.assertFalse(torch.equal(post[i][0], x), i)
        self.assertTrue(torch.equal(post[0][0], pre[0][0]))  # layer 0 is not steered

    def test_alpha_zero_is_bitwise_stock_and_alpha_one_is_not(self):
        x = self.x(dtype=torch.bfloat16)
        bb, ctx, _, _ = self.build(torch.bfloat16, steer=False)
        stock = tiny.run_backbone("qwen", bb, ctx, x=x)
        self.write("0")
        bb, ctx, _, rec = self.build(torch.bfloat16)
        self.assertEqual(rec["alpha"], 0.0)
        self.assertTrue(torch.equal(tiny.run_backbone("qwen", bb, ctx, x=x), stock))
        self.write("1.0")
        bb, ctx, _, _ = self.build(torch.bfloat16)
        self.assertFalse(torch.equal(tiny.run_backbone("qwen", bb, ctx, x=x), stock))

    def test_every_site_fires_once_per_forward(self):
        bb, ctx, run, _ = self.build()
        for T in (5, 1, 0):
            with self.subTest(tokens=T):
                y = tiny.run_backbone("qwen", bb, ctx, x=self.x(T))
                self.assertEqual(tuple(y.shape), (T, HID))
                fired = run._weightless_steer_check.fired
                self.assertEqual(sorted(fired), QSTEERED)
                self.assertEqual(set(fired.values()), {1})

    def test_a_site_that_runs_twice_fails_closed(self):
        """A second steering hook on one layer edits the edited stream again
        (at alpha 1 a projection twice is the projection once, so only the
        count shows it): the forward fails closed."""
        bb, ctx, run, _ = self.build()
        layer = bb.layers[31]
        (hook,) = layer._forward_hooks.values()
        layer.register_forward_hook(hook, with_kwargs=bool(layer._forward_hooks_with_kwargs))
        with self.assertRaisesRegex(RuntimeError, r"ran more than once in this forward "
                                                  r"\(layers \[31\], \[2\] times\)"):
            tiny.run_backbone("qwen", bb, ctx, x=self.x())

    def test_three_pipeline_ranks(self):
        import json
        x = self.x(7)
        bb, ctx, _, _ = self.build()
        ref = tiny.run_backbone("qwen", bb, ctx, x=x)
        ranks, layers = [], []
        for r in range(3):
            mdir = os.path.join(self.tmp.name, f"m{r}")
            bb_r, ctx_r, run_r, rec = self.build(rank=r, size=3, mdir=mdir)
            local = [i for i in QSTEERED if bb_r.start_layer <= i < bb_r.end_layer]
            self.assertEqual(rec["local_layer_ids"], local)
            self.assertEqual((rec["pp_rank"], rec["pp_size"]), (r, 3))
            self.assertEqual(rec["rank_source"], "get_parallel" if tiny.NEW_LAYOUT else "runner")
            man = json.load(open(os.path.join(mdir, f"weightless-manifest-{os.getpid()}.json")))
            self.assertEqual((man["pp_rank"], man["hooked_layers"]), (r, local))
            layers += local
            ranks.append((bb_r, ctx_r))
        self.assertEqual(layers, QSTEERED)
        out = None
        for r, (bb_r, ctx_r) in enumerate(ranks):
            out = (tiny.run_backbone("qwen", bb_r, ctx_r, x=x) if r == 0 else
                   tiny.run_backbone("qwen", bb_r, ctx_r, proxy=out))
        self.assertTrue(torch.equal(out, ref))


@unittest.skipUnless(tiny.NEW_LAYOUT and not tiny.WHY, "the stage-boundary layout only")
class StreamHookRefusals(unittest.TestCase):
    """The residual-stream site refuses what it cannot edit, on SGLang's own
    ResidualStream."""

    def hook(self):
        owner = nn.Module()
        stack = torch.zeros(4, 1, 8)
        stack[2, 0, 0] = 1.0
        owner.register_buffer("_steer_stack", stack, persistent=False)
        owner.register_buffer("_steer_alpha", torch.tensor(1.0), persistent=False)
        return wcore.make_stream_hook(owner, 2, width=8, row="Qwen3_5ForConditionalGeneration")

    def test_edit_in_place_on_a_plain_add(self):
        from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
        fb = types.SimpleNamespace(residual_stream=ResidualStream(torch.ones(3, 8)))
        out = fb.residual_stream.record(torch.full((3, 8), 2.0), PLAIN_ADD)
        got = self.hook()(None, (), {"forward_batch": fb}, out)
        self.assertIs(got, out)  # the stream still owns the tensor it returns
        self.assertTrue(torch.equal(got[:, 0], torch.full((3,), -1.0)))  # h.d = 3, x' = 2 - 3
        self.assertTrue(torch.equal(got[:, 1:], torch.full((3, 7), 2.0)))
        fb.residual_stream.check(got)  # what the next layer's prepare runs

    def test_refusals(self):
        from sglang.srt.layers.layer_boundary.layout import SumGroup
        from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
        x = torch.zeros(3, 8)
        written = types.SimpleNamespace(residual_stream=ResidualStream(torch.ones(3, 8)))
        declared = types.SimpleNamespace(residual_stream=ResidualStream(torch.ones(3, 8)))
        h_decl = declared.residual_stream.record(x.clone(), PLAIN_ADD, declared_sum=SumGroup.TP)
        other = types.SimpleNamespace(residual_stream=ResidualStream(torch.ones(3, 8)))
        h_other = other.residual_stream.record(x.clone(), types.SimpleNamespace(
            is_plain_add=False, applied_at_exit=False))
        for name, fb, out, pattern in (
                ("no forward batch", None, x, "holds no pending output"),
                ("written stream", written, x, "holds no pending output"),
                ("declared sum", declared, h_decl, "declared partial sum"),
                ("not a plain add", other, h_other, "not the plain add")):
            with self.subTest(name):
                kw = {"forward_batch": fb} if fb is not None else {}
                with self.assertRaisesRegex(wcore.SiteError, pattern):
                    self.hook()(None, (), kw, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
