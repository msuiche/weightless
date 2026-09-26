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
  separated) that pins what the rows rely on.
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
                         _loops, _layer_calls)  # noqa: E402

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
        model=model, is_draft_worker=draft, tp_rank=0, tp_size=1,
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


QWEN35_TREES = _trees("srt/models/qwen3_5_text.py")


@unittest.skipUnless(QWEN35_TREES, "no SGLang package with models/qwen3_5_text.py found")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
