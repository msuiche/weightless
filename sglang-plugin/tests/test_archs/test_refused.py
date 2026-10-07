"""The served model classes the plugin refuses by name (``archs.REFUSED``),
CPU only.

- A generic fake runner: for any served class name, the backbone at the
  row's path with layers carrying exact class names, ``start_layer`` /
  ``end_layer`` and a ``config``. Install only; the layers never run. The
  published-file matrix of test_plugin.py uses it too.
- Every refused class, served by that fake with a valid file set, fails
  ``install_steering`` with its own reason and hooks nothing.
- A parse of the SGLang trees (the installed package and every package
  folder in WEIGHTLESS_TEST_SGLANG_TREES, os.pathsep separated) that pins
  why: no Ouro model, Inkling's deferred short convolution, no
  ``DeepseekV4ForConditionalGeneration`` class.
"""
import ast
import io
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

from test_archs import _trees, _parse, _cls, _fn, _src, _flat  # noqa: E402

from weightless_sglang.archs import ARCH, REFUSED  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402


# ---------------------------------------------------------------- a fake for any served class


class _Layer(nn.Module):
    def __init__(self, layer_id, hidden):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden
        self.w = nn.Parameter(torch.zeros(1), requires_grad=False)  # a device for the buffers


_CLASSES = {}


def layer_class(name):
    """One Python class per exact SGLang class name."""
    if name not in _CLASSES:
        _CLASSES[name] = type(name, (_Layer,), {})
    return _CLASSES[name]


class _Backbone(nn.Module):
    def __init__(self, cfg, layer_names, start=0, end=None):
        super().__init__()
        self.config = cfg
        self.layers = nn.ModuleList(layer_class(n)(i, cfg.hidden_size)
                                    for i, n in enumerate(layer_names))
        self.start_layer = start
        self.end_layer = len(layer_names) if end is None else end


def _module_class(name):
    return type(name, (nn.Module,), {})


def runner(served, backbone_path, layer_names, cfg, hf_model_type, tp_size=1):
    """A runner whose ``model`` is an instance of a class named ``served``
    and whose backbone sits at ``backbone_path``."""
    bb = _Backbone(types.SimpleNamespace(**cfg), layer_names)
    model = _module_class(served)()
    nn.Module.__init__(model)
    parts = [p for p in backbone_path.split(".") if p]
    if not parts:  # the model is the backbone: give the served class the backbone's body
        model = type(served, (_Backbone,), {})(bb.config, layer_names)
    else:
        owner = model
        for p in parts[:-1]:
            sub = _module_class(p.capitalize())()
            nn.Module.__init__(sub)
            setattr(owner, p, sub)
            owner = sub
        setattr(owner, parts[-1], bb)
    return types.SimpleNamespace(
        model=model, is_draft_worker=False, tp_rank=0, tp_size=tp_size, pp_rank=0, pp_size=1,
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf_model_type),
                                           dtype=torch.bfloat16))


def backbone_of(run, backbone_path):
    bb = run.model
    for p in [p for p in backbone_path.split(".") if p]:
        bb = getattr(bb, p)
    return bb


class RefusedClasses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)

    def write(self, layers, hint, width=8):
        m = good_meta(layers=layers, alpha="1.0")
        m["controlvector.model_hint"] = hint
        rng = np.random.default_rng(0)
        path = os.path.join(self.tmp.name, "v.gguf")
        write_gguf(path, m, {f"direction.{i}": (rng.standard_normal(width).astype(np.float32), 0)
                             for i in layers})
        os.environ["WEIGHTLESS_STEER_PATH"] = path

    def install(self, run):
        with redirect_stderr(io.StringIO()):
            return install_steering(run, source="env")

    def test_every_named_class_refusal_boots_and_fails_with_its_reason(self):
        """Each class in archs.REFUSED, served by a fake runner with a valid
        file set, fails install_steering with its own named reason and hooks
        nothing. The set is pinned, so a name dropped from (or added to)
        REFUSED fails here."""
        expect = {
            "OuroForCausalLM": ("ouro", r"Ouro .*no SGLang model"),
            "InklingForConditionalGeneration": ("inkling", r"Inkling's post-layer stream.*mlp_sconv"),
            "InklingForCausalLM": ("inkling", r"Inkling's post-layer stream.*mlp_sconv"),
            "DeepseekV4ForConditionalGeneration": (
                "deepseek_v4", r"SGLang has no DeepseekV4ForConditionalGeneration class.*"
                               r"DeepseekV4ForCausalLM.*-ffn file"),
            "KimiLinearForCausalLM": ("kimi_linear", r"SGLang's KimiLinearForCausalLM is the Kimi "
                                                     r"Linear model .*not Kimi-K3"),
            "DeepseekV41ForCausalLM": ("deepseek_v41", r"DeepSeek-V4.1 is not supported: .*"
                                                       r"forward_hc_pre_from_prev"),
        }
        self.assertEqual(set(REFUSED), set(expect))
        self.assertEqual(set(REFUSED) & set(ARCH), set())  # a refused class has no row
        self.write(layers=(1, 2), hint="")
        for name, (model_type, reason) in expect.items():
            with self.subTest(name):
                run = runner(name, "model", ["DecoderLayer"] * 4,
                                dict(hidden_size=8, num_hidden_layers=4, model_type=model_type),
                                model_type)
                with self.assertRaisesRegex(
                        RuntimeError, rf"^weightless: served model class {name} is not supported: "
                                      rf"{reason}.* Refusing to serve unsteered\.$"):
                    self.install(run)
                bb = backbone_of(run, "model")
                self.assertFalse(hasattr(bb, "_steer_stack"))
                self.assertFalse(hasattr(run, "_weightless_steer_installed"))
                self.assertEqual([i for i, layer in enumerate(bb.layers)
                                  if layer._forward_hooks or layer._forward_pre_hooks], [])


# ---------------------------------------------------------------- the SGLang trees


MODEL_FILE = "srt/models/deepseek_v4.py"


@unittest.skipUnless(_trees(MODEL_FILE), "no SGLang package with models/deepseek_v4.py found")
class Structure(unittest.TestCase):
    """Pins, per SGLang tree, why the refused classes cannot be steered."""

    def assertIn(self, member, container, msg=None):  # noqa: N802
        if isinstance(member, str) and isinstance(container, str):
            member, container = _flat(member), _flat(container)
        super().assertIn(member, container, msg)

    def trees(self):
        return [(t, _parse(t, MODEL_FILE)) for t in _trees(MODEL_FILE)]

    def test_refused_architectures(self):
        """Ouro: no model in the tree. Inkling: the loop hands each layer's
        mlp_sconv to the next layer (the post-layer stream never exists).
        No DeepseekV4ForConditionalGeneration class."""
        for t, _ in self.trees():
            with self.subTest(tree=t):
                models = os.path.join(t, "srt", "models")
                found = []
                for f in sorted(os.listdir(models)):
                    if f.endswith(".py"):
                        with open(os.path.join(models, f)) as fh:
                            txt = fh.read()
                        if "class OuroForCausalLM" in txt or "total_ut_steps" in txt:
                            found.append(f)
                        self.assertNotIn("class DeepseekV4ForConditionalGeneration", txt, f)
                self.assertEqual(found, [])
                ink = _parse(t, "srt/models/inkling.py")
                llm = _src(_fn(_cls(ink, "InklingCausalLLM"), "forward"))
                self.assertIn("prev_mlp_sconv = layer.mlp_sconv", llm)
                self.assertIn("hidden_states, residual = layer(hidden_states, positions, "
                              "forward_batch, residual, prev_mlp_sconv,", llm)
                layer_fwd = _fn(_cls(ink, "InklingDecoderLayer"), "forward")
                self.assertIn("prev_mlp_sconv", [a.arg for a in layer_fwd.args.args])
                entry = [n for n in ink.body if isinstance(n, ast.Assign)
                         and _src(n.targets[0]) == "EntryClass"]
                self.assertIn("InklingForConditionalGeneration", _src(entry[0].value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
