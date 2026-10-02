"""The GLM-5.3 row (``GlmMoeDsaForCausalLM``), CPU only.

``DeepseekV2DecoderLayer`` returns ``(hidden, residual, topk_indices)`` with
the fold left to the next layer, so the stream after layer i is
``hidden + residual``; the edit is the Qwen pair edit with ``topk_indices``
passed through.

- Fakes (the first section): the hooked fake against the edit written out
  by hand, the pipeline ranks, the TP > 1 output forms (the partial-sum
  attribute, a deferred MoE handoff, SGLang's ``UnreducedOutput`` over a
  fake two-rank group) and every boot refusal the row relies on.
- With WEIGHTLESS_TEST_GLP_DIR set, the published GLP-77 file on a fake of
  the checkpoint's depth and width (78 x 6144); with
  WEIGHTLESS_TEST_CONFIG_DIR set to a folder of ``<org>__<repo>/config.json``,
  the row's width and the file against the checkpoint's config.
- ``RealGlm53``: SGLang's real ``DeepseekV2Model.forward``,
  ``DeepseekV2DecoderLayer.forward`` and a real ``LayerCommunicator`` inside a
  real ``GlmMoeDsaForCausalLM``, built with ``__new__`` (no weights, no GPU,
  no process group); the fusion decision is set by the test, so the real
  layer produces each of its TP > 1 output forms on CPU. Only the attention,
  the MLP and the norms are stubs.
- ``StructureGlm53``: a parse of the model files (the installed SGLang
  package and every package folder in WEIGHTLESS_TEST_SGLANG_TREES,
  os.pathsep separated) that pins the loop's call style, the tuple arity,
  where the last layer's output goes, the two-batch-overlap bypass and the
  output forms that decide the row's ``tp``.
"""
import ast
import contextlib
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

from test_archs import (_trees, _parse, _cls, _fn, _src, _calls, _entry, _returns, _loops,
                         _layer_calls, unit_dirs, proj_ok, OneRank, runner_of, spy_out)  # noqa: E402

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.archs import ARCH  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402
from weightless_steer.container import read_gguf_cvec  # noqa: E402
from weightless_steer.core import SteeringCore  # noqa: E402


# ---------------------------------------------------------------- fakes
# Each layer returns what SGLang's DeepseekV2DecoderLayer returns: (hidden,
# residual, topk_indices), the fold residual <- hidden + residual left to the
# next layer. ``form`` makes a layer return one of the TP > 1 output forms.
# The weights depend only on the layer id, so ranks built one at a time make
# one model, and a hooked and an unhooked copy are comparable.


def _w(layer_id, width, lo=-1.0, hi=1.0, step=0.01):
    return nn.Parameter(torch.linspace(lo, hi, width) * (1 + step * layer_id), requires_grad=False)


def _topk(x, layer_id):
    return torch.full((x.shape[0], 2), layer_id, dtype=torch.int32)


class Handoff:
    """Stands for a deferred MoE finalize handoff (a non-tensor output)."""

    def __init__(self, t):
        self.t = t


class TwoRanks:
    """A process group of two identical ranks: all_reduce doubles."""

    def all_reduce(self, x):
        return x * 2


class _DsLayer(nn.Module):
    def __init__(self, layer_id, width, form="plain"):
        super().__init__()
        self.layer_id = layer_id
        self.w = _w(layer_id, width, 0.5, 1.5)
        self.form = form

    def forward(self, positions, hidden_states, forward_batch, residual, *args, **kwargs):
        residual = hidden_states if residual is None else hidden_states + residual
        out = torch.tanh(residual) * self.w
        topk = _topk(residual, self.layer_id)
        if self.form == "marker":
            out._sglang_needs_allreduce_fusion = True
        elif self.form == "handoff":
            out = Handoff(out)
        elif self.form == "unreduced":
            from sglang.srt.layers.communicator import UnreducedOutput
            out = UnreducedOutput(partial=out / 2, group=TwoRanks())
        return out, residual, topk


DeepseekV2DecoderLayer = type("DeepseekV2DecoderLayer", (_DsLayer,), {})
DsOtherLayer = type("DeepseekV4DecoderLayer", (_DsLayer,), {})


class DeepseekV2Model(nn.Module):
    """The DeepseekV2Model loop: ``h, r, topk = self.layers[i](...)`` over
    ``[start_layer, end_layer)``; PP hands on hidden_states and residual;
    the last rank folds them into the final norm (here: returns h + r)."""

    def __init__(self, num_layers=78, width=8, start=0, end=None, model_type="glm_moe_dsa",
                 layer_cls=None, forms=None):
        super().__init__()
        end = num_layers if end is None else end
        self.config = types.SimpleNamespace(hidden_size=width, num_hidden_layers=num_layers,
                                            model_type=model_type)
        forms = forms or {}
        self.layers = nn.ModuleList(
            (layer_cls or DeepseekV2DecoderLayer)(i, width, forms.get(i, "plain"))
            if start <= i < end else nn.Identity() for i in range(num_layers))
        self.start_layer, self.end_layer = start, end
        self.topk_seen = []

    def forward(self, x=None, proxy=None):
        h, r = (x, None) if proxy is None else (proxy["hidden_states"], proxy["residual"])
        self.topk_seen = []
        for i in range(self.start_layer, self.end_layer):
            h, r, topk = self.layers[i](None, h, None, r)
            self.topk_seen.append(topk)
        if self.end_layer < self.config.num_hidden_layers:
            return {"hidden_states": h, "residual": r}
        return h + r


def _wrapper(name):
    return type(name, (nn.Module,), {
        "__init__": lambda self, bb: (nn.Module.__init__(self), setattr(self, "model", bb))[0]})


GlmMoeDsaForCausalLM = _wrapper("GlmMoeDsaForCausalLM")


def runner(model, *, hf_model_type, tp_size=1, pp_rank=0, pp_size=1, tbo=False):
    return types.SimpleNamespace(
        model=model, is_draft_worker=False, tp_rank=0, tp_size=tp_size, pp_rank=pp_rank,
        pp_size=pp_size, server_args=types.SimpleNamespace(enable_two_batch_overlap=tbo),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf_model_type),
                                           dtype=torch.float32))


def glm53_runner(num_layers=78, width=8, **kw):
    run_kw = {k: kw.pop(k) for k in ("tp_size", "pp_rank", "pp_size", "tbo") if k in kw}
    return runner(GlmMoeDsaForCausalLM(DeepseekV2Model(num_layers, width, **kw)),
                  hf_model_type="glm_moe_dsa", **run_kw)


# ---------------------------------------------------------------- the row on the fakes


HOOK = "residual_stream_post_layer"
GLM53 = "GlmMoeDsaForCausalLM"
try:
    from sglang.srt.layers.communicator import UnreducedOutput  # noqa: F401
    HAVE_UNREDUCED = True
except Exception:
    HAVE_UNREDUCED = False

# The published file (huggingface.co/msuiche/<name without the suffix>).
GLM53_FILE = glpfiles.find_glp("GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf")
# config.json of each checkpoint, under WEIGHTLESS_TEST_CONFIG_DIR
CONFIG_DIR = os.environ.get("WEIGHTLESS_TEST_CONFIG_DIR", "").strip() or None


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)

    def write(self, layers, width, hint, alpha="1.0", name="v.gguf", **extra):
        m = good_meta(layers=layers, alpha=alpha)
        m["controlvector.model_hint"] = hint
        m.update(extra)
        path = os.path.join(self.tmp.name, name)
        write_gguf(path, m, unit_dirs(layers, width))
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        return path

    def install(self, run):
        with redirect_stderr(io.StringIO()):
            return install_steering(run, source="env")

    def refused(self, run, pattern, forward=None):
        with self.assertRaisesRegex(Exception, pattern):
            self.install(run)
            if forward is not None:
                forward(run)

    def hooked(self, run):
        return sorted(i for i, l in enumerate(run.model.model.layers) if l._forward_hooks)

    def spy(self, layer, store, key):
        """A forward hook that sees what the layer returns before the
        steering hook (registered first) or after it (registered after)."""
        return layer.register_forward_hook(lambda m, a, o: store.__setitem__(key, o))


class Glm53Fake(Base):
    L, W = 78, 8
    STEERED = tuple(range(1, 78))

    def setUp(self):
        super().setUp()
        self.write(self.STEERED, self.W, "glm_moe_dsa")

    def test_row(self):
        row = ARCH[GLM53]
        self.assertEqual(row.backbone, ("model",))
        self.assertEqual(row.layers, frozenset({"DeepseekV2DecoderLayer"}))
        self.assertEqual((row.arity, row.hidden_index, row.residual_index), (3, 0, 1))
        self.assertEqual((row.install, row.exec_id, row.tp, row.per_stream), ("hook", "layer", "ok", False))
        self.assertEqual(row.hooks, frozenset({HOOK}))
        self.assertEqual(row.hint, frozenset({"glm_moe_dsa"}))
        self.assertEqual(row.width(DeepseekV2Model(2, 6144).config), 6144)

    def test_hooked_set_stack_and_manifest(self):
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = glm53_runner(self.L, self.W)
        rec = self.install(run)
        self.assertEqual(self.hooked(run), list(self.STEERED))
        self.assertEqual(rec["local_layer_ids"], list(self.STEERED))
        self.assertEqual((rec["row"], rec["install"], rec["width"], rec["hook_point"]),
                         (GLM53, "hook", self.W, HOOK))
        self.assertEqual(rec["model_types"], ["glm_moe_dsa"])
        self.assertEqual(tuple(run.model.model._steer_stack.shape), (self.L, 1, self.W))
        with open(rec["manifest"]) as f:
            self.assertEqual(json.load(f)["row"], GLM53)

    def test_forward_is_the_pair_edit_and_topk_passes_through(self):
        alpha = 1.0
        run = glm53_runner(self.L, self.W)
        bb = run.model.model.double()
        raw, out = {}, {}
        for i in self.STEERED:
            self.spy(bb.layers[i], raw, i)  # before the steering hook
        self.install(run)
        for i in self.STEERED:
            self.spy(bb.layers[i], out, i)  # after it
        ref = glm53_runner(self.L, self.W).model.model.double()
        stack = bb._steer_stack.double()
        x = torch.randn(4, self.W, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        h, r = x, None
        for i in range(self.L):
            h, r, _ = ref.layers[i](None, h, None, r)
            if i in self.STEERED:
                d = stack[i, 0]
                s = h + r
                h = (s - alpha * (s @ d).unsqueeze(-1) * d) - r
        got = bb(x)
        self.assertTrue(torch.allclose(got, h + r, atol=1e-12))
        for i in self.STEERED:
            self.assertIs(out[i][2], raw[i][2])  # topk_indices: the same object
            self.assertIs(out[i][1], raw[i][1])  # the residual is read, never changed
            proj_ok(self, raw[i][0] + raw[i][1], out[i][0] + out[i][1], stack[i, 0], alpha)
        self.assertEqual(len(bb.topk_seen), self.L)

    def test_pipeline_ranks_equal_one_rank(self):
        x = torch.randn(3, self.W, generator=torch.Generator().manual_seed(4))
        one = glm53_runner(self.L, self.W)
        self.install(one)
        ref = one.model.model(x)
        self.assertFalse(torch.allclose(ref, glm53_runner(self.L, self.W).model.model(x)))
        for part in ((39, 39), (26, 26, 26)):
            with self.subTest(partition=part):
                out, start = None, 0
                for rank, n in enumerate(part):
                    run = glm53_runner(self.L, self.W, start=start, end=start + n,
                                          pp_rank=rank, pp_size=len(part))
                    rec = self.install(run)
                    self.assertEqual(rec["local_layer_ids"], list(range(max(1, start), start + n)))
                    bb = run.model.model
                    out = bb(x) if rank == 0 else bb(proxy=out)
                    if rank < len(part) - 1:
                        self.assertEqual(sorted(out), ["hidden_states", "residual"])
                    start += n
                self.assertTrue(torch.equal(out, ref))

    def test_tp2_steers_a_plain_reduced_output(self):
        run = glm53_runner(self.L, self.W, tp_size=2)
        self.assertEqual(self.install(run)["tp_size"], 2)
        one = glm53_runner(self.L, self.W)
        self.install(one)
        x = torch.randn(2, self.W)
        self.assertTrue(torch.equal(run.model.model(x), one.model.model(x)))

    def test_partial_sum_attribute_is_refused(self):
        run = glm53_runner(self.L, self.W, tp_size=2, forms={5: "marker"})
        self.refused(run, wcore.PARTIAL_SUM_MARKER, lambda r: r.model.model(torch.randn(2, self.W)))

    def test_deferred_moe_handoff_is_refused(self):
        run = glm53_runner(self.L, self.W, tp_size=2, forms={9: "handoff"})
        self.refused(run, r"decoder layer 9 output is Handoff \(for example a deferred MoE",
                     lambda r: r.model.model(torch.randn(2, self.W)))

    @unittest.skipUnless(HAVE_UNREDUCED, "this SGLang build has no UnreducedOutput (older trees mark "
                                         "partial sums with a tensor attribute instead)")
    def test_unreduced_output_is_reduced_then_steered(self):
        x = torch.randn(3, self.W, dtype=torch.float64)
        plain = glm53_runner(self.L, self.W)
        plain.model.model.double()
        self.install(plain)
        run = glm53_runner(self.L, self.W, tp_size=2, forms={i: "unreduced" for i in (5, 40, 77)})
        run.model.model.double()
        self.install(run)
        self.assertTrue(torch.equal(run.model.model(x), plain.model.model(x)))

    def test_two_batch_overlap_is_refused(self):
        self.refused(glm53_runner(self.L, self.W, tbo=True), "two-batch overlap")

    def test_a_layer_that_returns_two_slots_is_refused(self):
        run = glm53_runner(self.L, self.W)
        self.install(run)
        lay = run.model.model.layers[7]
        orig = lay.forward
        lay.forward = lambda *a, **k: orig(*a, **k)[:2]
        with self.assertRaisesRegex(Exception, "expected a 3-tuple"):
            run.model.model(torch.randn(2, self.W))

    def test_wrong_layer_class_is_refused(self):
        self.refused(glm53_runner(self.L, self.W, layer_cls=DsOtherLayer), "not one of")

    def test_file_refusals(self):
        cases = (
            (dict(hint="hy_v4"), "model_hint"),
            (dict(hint="glm_moe_dsa", **{"glp.hook_point": "ffn_out_pre_residual"}), "hook_point"),
            (dict(hint="glm_moe_dsa", **{"glp.structure": "per-execution-step"}), "per-execution-step"),
        )
        for kw, pattern in cases:
            with self.subTest(kw=kw):
                self.write(self.STEERED, self.W, **kw)
                self.refused(glm53_runner(self.L, self.W), pattern)
        self.write(tuple(range(1, 79)), self.W, "glm_moe_dsa")
        self.refused(glm53_runner(self.L, self.W), "out of range")
        self.write(self.STEERED, 16, "glm_moe_dsa")
        self.refused(glm53_runner(self.L, self.W), "width")

    def test_free_text_structure_is_accepted(self):
        self.write(self.STEERED, self.W, "glm_moe_dsa",
                   **{"glp.structure": "77 per-layer vectors (L1-77; layer 0 inexpressible)"})
        self.assertEqual(self.install(glm53_runner(self.L, self.W))["local_layer_ids"],
                         list(self.STEERED))

    def test_alpha_zero_is_bitwise_stock(self):
        self.write(self.STEERED, self.W, "glm_moe_dsa", alpha="0")
        x = torch.randn(5, self.W).bfloat16()
        stock = glm53_runner(self.L, self.W).model.model.bfloat16()(x)
        run = glm53_runner(self.L, self.W)
        run.model.model.bfloat16()
        self.assertEqual(self.install(run)["alpha"], 0.0)
        self.assertTrue(torch.equal(run.model.model(x), stock))


def _meta(path):
    meta, tensors = read_gguf_cvec(path)
    ids = sorted(int(k.split(".")[1]) for k in tensors)
    widths = {int(np.asarray(t).shape[-1]) for t in tensors.values()}
    return meta, ids, widths


class RealFiles(Base):
    """The row's published file on a fake of the checkpoint's shape."""

    def check_stack(self, run, path, num_layers, width, ids):
        bb = run.model.model
        core = SteeringCore.from_env(hook=HOOK, num_layers=num_layers, hidden_size=width)
        self.assertEqual(sorted(core.dirs), ids)
        self.assertEqual(tuple(bb._steer_stack.shape), (num_layers, 1, width))
        for i in range(num_layers):
            row = bb._steer_stack[i, 0]
            if i in core.dirs:
                self.assertTrue(torch.equal(row, core.dirs[i]), i)
            else:
                self.assertEqual(float(row.abs().sum()), 0.0, i)

    @unittest.skipUnless(glpfiles.have(GLM53_FILE), "set WEIGHTLESS_TEST_GLP_DIR to the folder with "
                                                  "GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf")
    def test_glm53_glp77(self):
        meta, ids, widths = _meta(GLM53_FILE)
        self.assertEqual((meta["controlvector.model_hint"], meta["glp.hook_point"], meta["glp.mode"]),
                         ("glm_moe_dsa", HOOK, "project"))
        self.assertEqual(float(meta["glp.alpha_default"]), 1.0)
        self.assertEqual((ids, widths), (list(range(1, 78)), {6144}))
        self.assertNotEqual(str(meta["glp.structure"]).strip().lower(), "per-execution-step")
        os.environ["WEIGHTLESS_STEER_PATH"] = GLM53_FILE
        run = glm53_runner(78, 6144)
        rec = self.install(run)  # free-text glp.structure: accepted on this row
        self.assertEqual((rec["local_layer_ids"], rec["alpha"], rec["width"]), (ids, 1.0, 6144))
        self.check_stack(run, GLM53_FILE, 78, 6144, ids)
        y = run.model.model(torch.randn(2, 6144))
        self.assertTrue(bool(torch.isfinite(y).all()))
        for part, want in (((39, 39), [list(range(1, 39)), list(range(39, 78))]),):
            start = 0
            for rank, n in enumerate(part):
                r = glm53_runner(78, 6144, start=start, end=start + n, pp_rank=rank, pp_size=2)
                self.assertEqual(self.install(r)["local_layer_ids"], want[rank])
                start += n


def _config(org_repo):
    path = os.path.join(CONFIG_DIR or "", org_repo, "config.json")
    if not (CONFIG_DIR and os.path.isfile(path)):
        return None
    with open(path) as f:
        return json.load(f)


@unittest.skipUnless(CONFIG_DIR, "set WEIGHTLESS_TEST_CONFIG_DIR to a folder of <org>__<repo>/config.json")
class Configs(unittest.TestCase):
    """The row each checkpoint's config.json resolves to, the width the row
    reads from it (through SGLang's own config class where this SGLang has
    one), the depth, and the published file against both."""

    def cfg_or_skip(self, org_repo):
        c = _config(org_repo)
        if c is None:
            self.skipTest(f"{org_repo}/config.json is not in {CONFIG_DIR}")
        return c

    def check_file(self, path, depth, width, hint):
        if not glpfiles.have(path):
            return
        meta, ids, widths = _meta(path)
        self.assertLess(max(ids), depth)
        self.assertEqual(widths, {width})
        self.assertEqual(meta["controlvector.model_hint"], hint)

    def test_glm53(self):
        c = self.cfg_or_skip("RadixArk__GLM-5.3-NVFP4")
        self.assertEqual(c["architectures"], [GLM53])
        self.assertIn(c["model_type"], {c["model_type"]} | ARCH[GLM53].hint)
        self.assertEqual(c["model_type"], "glm_moe_dsa")
        self.assertEqual(c["num_hidden_layers"], 78)
        cfg = type("Cfg", (), dict(c))
        self.assertEqual(ARCH[GLM53].width(cfg), 6144)
        self.check_file(GLM53_FILE, 78, 6144, "glm_moe_dsa")


# ---------------------------------------------------------------- SGLang's real classes (R)


def _try(fn):
    try:
        return fn(), None
    except Exception as e:  # no SGLang, or a build without the model
        return None, f"{type(e).__name__}: {e}"


def _ds():
    from sglang.srt.layers import communicator as COMM
    from sglang.srt.models import deepseek_v2 as D
    from sglang.srt.models import glm4_moe as G4
    return COMM, D, G4


DS, DS_WHY = _try(_ds)


def _pp_proxy():
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    return PPProxyTensors


class RealBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)

    def write(self, layers, width, hint, alpha):
        m = good_meta(layers=layers, alpha=alpha)
        m["controlvector.model_hint"] = hint
        path = os.path.join(self.tmp.name, f"v{alpha}.gguf")
        write_gguf(path, m, unit_dirs(layers, width))
        os.environ["WEIGHTLESS_STEER_PATH"] = path

    def install(self, run):
        with redirect_stderr(io.StringIO()):
            return install_steering(run, source="env")


class _Norm(nn.Module):
    """RMSNorm with SGLang's fused add: norm(x, residual) -> (norm(x + r), x + r)."""

    def __init__(self, h):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(h), requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, x, residual=None, post_residual_addition=None):
        if residual is not None:
            x = x + residual
            residual = x
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight.float()).to(x.dtype)
        return y if residual is None else (y, residual)


FB = types.SimpleNamespace(
    can_run_tbo=False, reuse_dsa_topk_indices=False, spec_info=None, mm_input_embeds=None,
    input_ids=None, forward_mode=types.SimpleNamespace(is_idle=lambda: False, is_extend=lambda **k: False))


DS_H, DS_NL = 8, 12
DS_STEERED = tuple(range(1, DS_NL))


class _DsAttn(nn.Module):
    """Stub MLA: a token-local map; returns (out, topk_indices) as the DSA
    attention does."""

    def __init__(self, i):
        super().__init__()
        self.i = i
        self.w = nn.Parameter(torch.linspace(0.5, 1.5, DS_H) * (1 + 0.02 * i), requires_grad=False)

    def maybe_use_decode_attn_tp(self, forward_batch):
        return contextlib.nullcontext()

    def forward(self, positions=None, hidden_states=None, forward_batch=None, zero_allocator=None,
                llama_4_scaling=None, layer_scatter_modes=None, prev_topk_indices=None):
        topk = torch.full((hidden_states.shape[0], 2), self.i, dtype=torch.int32)
        return torch.tanh(hidden_states * self.w), topk


class _DsMLP(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(-1, 1, DS_H) * (1 + 0.01 * i), requires_grad=False)
        self.handoff = False

    def forward(self, x, forward_batch=None, gemm_output_zero_allocator=None):
        out = x * self.w
        return types.SimpleNamespace(deferred=out) if self.handoff else out


def ds_layer(i, dtype):
    COMM, D, _ = DS
    cfg = types.SimpleNamespace(hidden_size=DS_H, num_hidden_layers=DS_NL, rms_norm_eps=1e-6)
    L = D.DeepseekV2DecoderLayer.__new__(D.DeepseekV2DecoderLayer)
    nn.Module.__init__(L)
    L.config, L.hidden_size, L.layer_id, L._gfx95_quant_format = cfg, DS_H, i, ""
    L.self_attn, L.mlp = _DsAttn(i).to(dtype), _DsMLP(i).to(dtype)
    L.input_layernorm, L.post_attention_layernorm = _Norm(DS_H).to(dtype), _Norm(DS_H).to(dtype)
    L.layer_scatter_modes = types.SimpleNamespace(
        is_last_layer=i == DS_NL - 1, is_layer_sparse=False, layer_output_mode=None, mlp_mode=None)
    C = COMM.LayerCommunicator.__new__(COMM.LayerCommunicator)
    C.layer_scatter_modes = L.layer_scatter_modes
    C.input_layernorm, C.post_attention_layernorm = L.input_layernorm, L.post_attention_layernorm
    C.allow_reduce_scatter, C.is_last_layer, C.qkv_latent_func = False, i == DS_NL - 1, None
    C.enable_fused_ar_quant = C.fused_ar_quant_keep_bf16 = C.force_layernorm_before_dp_gather = False
    C._context = types.SimpleNamespace(tp_size=1)
    C._communicate_simple_fn = COMM.CommunicateSimpleFn._trivial
    C._communicate_with_all_reduce_and_layer_norm_fn = COMM.CommunicateWithAllReduceAndLayerNormFn._simple
    C._communicate_summable_tensor_pair_fn = COMM.CommunicateSummableTensorPairFn._trivial
    C._postprocess_scatters_to_local_tokens, C._sp_variant, C._speculative_algo = False, None, None
    # The fusion decision reads the published parallel state; the test makes
    # it (False: the TP=1 path, where postprocess_layer completes the output).
    C.should_fuse_mlp_allreduce_with_next_layer = lambda forward_batch: False
    L.layer_communicator = C
    return L


def ds_runner(partition=(DS_NL,), rank=0, dtype=torch.float32, tp_size=1):
    _, D, G4 = DS
    start = sum(partition[:rank])
    end = start + partition[rank]
    cfg = types.SimpleNamespace(hidden_size=DS_H, num_hidden_layers=DS_NL, rms_norm_eps=1e-6,
                                model_type="glm_moe_dsa")
    M = D.DeepseekV2Model.__new__(D.DeepseekV2Model)
    nn.Module.__init__(M)
    M.config, M.use_dsa, M.first_k_dense_replace = cfg, False, 3
    M.pp_group = types.SimpleNamespace(is_first_rank=start == 0, is_last_rank=end == DS_NL)
    M.embed_tokens = nn.Identity()
    M.layers = nn.ModuleList(ds_layer(i, dtype) if start <= i < end else nn.Identity()
                             for i in range(DS_NL))
    M.start_layer, M.end_layer = start, end
    M.norm = _Norm(DS_H).to(dtype) if end == DS_NL else nn.Identity()
    M.gemm_output_zero_allocator_size, M.layers_to_capture = 0, []
    M.llama_4_scaling_config, M.next_full_attention_layer_id = None, {}
    W = G4.GlmMoeDsaForCausalLM.__new__(G4.GlmMoeDsaForCausalLM)
    nn.Module.__init__(W)
    W.model = M
    return runner_of(W, "glm_moe_dsa", tp_size=tp_size, pp_rank=rank, pp_size=len(partition),
                     dtype=dtype)


def ds_run(run, x=None, proxy=None, fb=FB):
    return run.model.model(None, None, fb, input_embeds=x, pp_proxy_tensors=proxy)


def _ds_has_ffn_exit():
    import inspect
    return "ffn_exit" in inspect.getsource(DS[1].DeepseekV2DecoderLayer.forward)


@unittest.skipUnless(DS, f"SGLang's DeepseekV2 / GlmMoeDsa model is not importable here ({DS_WHY})")
class RealGlm53(RealBase):
    def setUp(self):
        super().setUp()
        self.write(DS_STEERED, DS_H, "glm_moe_dsa", "1.0")

    def test_real_classes_install(self):
        run = ds_runner()
        rec = self.install(run)
        self.assertEqual((rec["row"], rec["install"]), ("GlmMoeDsaForCausalLM", "hook"))
        self.assertEqual(rec["local_layer_ids"], list(DS_STEERED))
        bb = run.model.model
        self.assertEqual([i for i in range(DS_NL) if bb.layers[i]._forward_hooks], list(DS_STEERED))

    def test_steered_pair_reaches_next_layer_and_last_layer(self):
        alpha = 1.0
        run = ds_runner(dtype=torch.float64)
        bb = run.model.model
        raw, out, seen = {}, {}, {}
        for i in DS_STEERED:
            spy_out(bb.layers[i], raw, i)
        for i in range(1, DS_NL):  # the loop calls layer(positions, hidden, fb, residual, ...)
            bb.layers[i].register_forward_pre_hook(
                lambda m, a, _i=i: seen.__setitem__(_i, (a[1], a[3])))
        self.install(run)
        for i in DS_STEERED:
            spy_out(bb.layers[i], out, i)
        x = torch.randn(4, DS_H, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        y = ds_run(run, x=x)
        stack = bb._steer_stack.double()
        for i in DS_STEERED:
            d = stack[i, 0]
            h, r, topk = raw[i]
            s = h + r
            s2 = s - alpha * (s @ d).unsqueeze(-1) * d  # vllm-plugin _steer_post_layer
            self.assertTrue(torch.allclose(out[i][0], s2 - r, atol=1e-12), i)
            self.assertIs(out[i][1], r)
            self.assertIs(out[i][2], topk)  # topk_indices passed through
            if i + 1 < DS_NL:  # the next layer receives the steered pair
                self.assertIs(seen[i + 1][0], out[i][0])
                self.assertIs(seen[i + 1][1], r)
                proj_ok(self, s, seen[i + 1][0] + seen[i + 1][1], d, alpha)
            else:  # the last layer: its steered pair is what the final norm folds
                self.assertTrue(torch.allclose(y, bb.norm(out[i][0], r)[0], atol=1e-12))
                proj_ok(self, s, out[i][0] + r, d, alpha)
        self.assertEqual(seen[1][0].shape, (4, DS_H))  # layer 0 is not steered

    def test_pipeline_ranks_equal_one_rank(self):
        x = torch.randn(3, DS_H, generator=torch.Generator().manual_seed(2))
        one = ds_runner()
        self.install(one)
        ref = ds_run(one, x=x)
        self.assertFalse(torch.allclose(ref, ds_run(ds_runner(), x=x)))
        for part in ((6, 6), (5, 4, 3)):
            with self.subTest(partition=part):
                out = None
                for rank in range(len(part)):
                    run = ds_runner(part, rank)
                    start = sum(part[:rank])
                    self.assertEqual(self.install(run)["local_layer_ids"],
                                     list(range(max(1, start), start + part[rank])))
                    out = ds_run(run, x=x) if rank == 0 else ds_run(run, proxy=out)
                    if rank < len(part) - 1:
                        self.assertIsInstance(out, _pp_proxy())
                        self.assertEqual(sorted(out.tensors), ["hidden_states", "residual"])
                self.assertTrue(torch.equal(out, ref))

    def test_alpha_zero_is_bitwise_stock(self):
        self.write(DS_STEERED, DS_H, "glm_moe_dsa", "0")
        x = torch.randn(5, DS_H, generator=torch.Generator().manual_seed(9)).bfloat16()
        stock = ds_run(ds_runner(dtype=torch.bfloat16), x=x)
        run = ds_runner(dtype=torch.bfloat16)
        self.install(run)
        self.assertTrue(torch.equal(ds_run(run, x=x), stock))

    def test_tp_output_form_when_the_allreduce_goes_to_the_next_layer(self):
        """The real layer, told to leave its FFN sum to the next layer (the
        TP>1 fusion), returns what this tree returns for that: older trees the
        partial-sum attribute (refused), newer trees an UnreducedOutput (the
        site completes it with SGLang's reduce_output, then steers)."""
        k = 5
        x = torch.randn(3, DS_H, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
        run = ds_runner(dtype=torch.float64, tp_size=2)
        bb = run.model.model
        bb.layers[k].layer_communicator.should_fuse_mlp_allreduce_with_next_layer = lambda fb: True
        raw = {}
        spy_out(bb.layers[k], raw, k)
        self.install(run)
        if not _ds_has_ffn_exit():  # older trees
            with self.assertRaisesRegex(wcore.SiteError, f"layer {k} returned a TP-partial sum"):
                ds_run(run, x=x)
            self.assertTrue(getattr(raw[k][0], wcore.PARTIAL_SUM_MARKER, False))
            return
        from sglang.srt.layers.communicator import UnreducedOutput
        from sglang.srt.runtime_context import get_parallel
        plain = ds_runner(dtype=torch.float64)
        self.install(plain)
        with get_parallel().override(tp_group=OneRank()):
            y = ds_run(run, x=x)
        self.assertIsInstance(raw[k][0], UnreducedOutput)
        self.assertTrue(torch.equal(y, ds_run(plain, x=x)))

    def test_deferred_moe_handoff_is_refused(self):
        if not _ds_has_ffn_exit():
            self.skipTest("this tree's DeepseekV2DecoderLayer has no deferred MoE handoff path (no FfnExit)")
        from sglang.srt.runtime_context import get_parallel
        k = 4
        run = ds_runner(tp_size=2)
        lay = run.model.model.layers[k]
        lay.layer_communicator.should_defer_moe_finalize = lambda fb, m=None: True
        lay.mlp.handoff = True
        self.install(run)
        with get_parallel().override(tp_group=OneRank()), \
                self.assertRaisesRegex(wcore.SiteError, f"decoder layer {k} output is SimpleNamespace "
                                                     r"\(for example a deferred MoE finalize handoff\)"):
            ds_run(run, x=torch.randn(2, DS_H))

    def test_fired_check_catches_the_two_batch_overlap_bypass(self):
        """With can_run_tbo the loop stops at first_k_dense_replace and the
        rest runs through model_forward_maybe_tbo, which calls layer ops, not
        the layer. Here it is replaced by a loop over ``layer.forward`` (no
        hook runs): the fired check fails the forward."""
        _, D, _ = DS
        run = ds_runner()
        self.install(run)
        bb = run.model.model

        def tbo(*, layers, positions, forward_batch, hidden_states, residual, zero_allocator, **kw):
            for layer in layers:
                hidden_states, residual, _ = layer.forward(positions, hidden_states, forward_batch,
                                                           residual, zero_allocator)
            return hidden_states, residual

        fb = types.SimpleNamespace(**{**vars(FB), "can_run_tbo": True})
        with mock.patch.object(D, "model_forward_maybe_tbo", tbo), \
                self.assertRaisesRegex(RuntimeError, r"did not run in this forward \(layers \[3, 4, 5"):
            ds_run(run, x=torch.randn(2, DS_H), fb=fb)
        ds_run(run, x=torch.randn(2, DS_H))  # the normal loop: every site fires


# ---------------------------------------------------------------- structure


GLM53_TREES = _trees("srt/models/glm4_moe.py")


@unittest.skipUnless(GLM53_TREES, "no SGLang package with models/glm4_moe.py found")
class StructureGlm53(unittest.TestCase):
    """GlmMoeDsaForCausalLM -> DeepseekV2ForCausalLM -> DeepseekV2Model ->
    DeepseekV2DecoderLayer, in every tree."""

    def trees(self):
        return [(t, _parse(t, "srt/models/glm4_moe.py"), _parse(t, "srt/models/deepseek_v2.py"))
                for t in GLM53_TREES]

    def test_served_class_and_backbone(self):
        for t, g4, ds in self.trees():
            with self.subTest(tree=t):
                c = _cls(g4, "GlmMoeDsaForCausalLM")
                self.assertEqual([_src(b) for b in c.bases], ["DeepseekV2ForCausalLM"])
                self.assertNotIn("__init__", [n.name for n in c.body if isinstance(n, ast.FunctionDef)])
                self.assertNotIn("forward", [n.name for n in c.body if isinstance(n, ast.FunctionDef)])
                self.assertIn("GlmMoeDsaForCausalLM", _entry(g4))
                init = _src(_fn(_cls(ds, "DeepseekV2ForCausalLM"), "__init__"))
                self.assertIn("self.model = DeepseekV2Model(", init)
                minit = _src(_fn(_cls(ds, "DeepseekV2Model"), "__init__"))
                self.assertIn("self.layers, self.start_layer, self.end_layer = make_layers(", minit)
                self.assertIn("lambda idx, prefix: DeepseekV2DecoderLayer(", minit)

    def test_model_loop_calls_each_layer_and_unpacks_three(self):
        for t, _, ds in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(ds, "DeepseekV2Model"), "forward")
                loops = _loops(fwd)
                self.assertEqual(len(loops), 1)
                self.assertEqual(_src(loops[0].iter), "range(normal_start_layer, normal_end_layer)")
                self.assertIn("layer = self.layers[i]", _src(loops[0]))
                calls = _layer_calls(loops[0])
                self.assertEqual(len(calls), 1)
                self.assertEqual(_src(calls[0].targets[0]), "(hidden_states, residual, topk_indices)")
                self.assertEqual(len(_calls(fwd, "layer")), 1)
                self.assertEqual(len(_calls(fwd, "forward")), 0)
                # the only other way through the layers: two-batch overlap
                self.assertEqual(len(_calls(fwd, "model_forward_maybe_tbo")), 1)
                self.assertIn("if forward_batch.can_run_tbo:", _src(fwd))
                s = _src(fwd)
                # no last-layer trap: the final norm folds the last layer's pair
                self.assertIn("hidden_states, _ = self.norm(hidden_states, residual)", s)
                # PP hands on the pair
                self.assertIn("'hidden_states': hidden_states, 'residual': residual", s)
                self.assertIn("residual = pp_proxy_tensors['residual']", s)

    def test_layer_returns_three_with_the_residual_second(self):
        for t, _, ds in self.trees():
            with self.subTest(tree=t):
                lay = _cls(ds, "DeepseekV2DecoderLayer")
                fwd = _fn(lay, "forward")
                self.assertEqual(_returns(fwd), ["(hidden_states, residual, topk_indices)"])
                self.assertIn("self.layer_id = layer_id", _src(_fn(lay, "__init__")))
                s = _src(fwd)
                self.assertIn("self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(", s)

    def test_output_forms_that_decide_tp(self):
        """At TP=1 the layer output is the reduced tensor postprocess_layer
        returns. At TP>1 it may instead be: older trees, the same tensor marked
        with the partial-sum attribute (the site refuses it); newer trees, an
        UnreducedOutput from FfnExit (the site runs reduce_output, the same
        call the next layer's prepare_attn makes) or a deferred MoE handoff,
        a non-tensor (the site refuses it)."""
        for t, _, ds in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(ds, "DeepseekV2DecoderLayer"), "forward")
                s = _src(fwd)
                comm = _parse(t, "srt/layers/communicator.py")
                if "ffn_exit" in s:
                    self.assertIn("hidden_states, residual = ffn_exit.finish(hidden_states, residual)", s)
                    self.assertNotIn(wcore.PARTIAL_SUM_MARKER, s)
                    fin = _fn(_cls(comm, "FfnExit"), "finish")
                    fs = _src(fin)
                    self.assertIn("if not isinstance(hidden_states, torch.Tensor):", fs)
                    self.assertIn("return (self._leave(hidden_states), residual)", fs)
                    self.assertIn("return self.communicator.postprocess_layer(", fs)
                    # exactly these three exits: a plain partial sum handed back by an
                    # extra return would be steered at TP > 1 without a refusal
                    self.assertEqual(sorted(_returns(fin)), sorted([
                        "(hidden_states, residual)",
                        "(self._leave(hidden_states), residual)",
                        "self.communicator.postprocess_layer(hidden_states, residual, "
                        "self.forward_batch)"]))
                    guard = [n for n in fin.body if isinstance(n, ast.If)
                             and _src(n.test) == "not isinstance(hidden_states, torch.Tensor)"]
                    self.assertEqual(len(guard), 1)  # the plain return is the non-tensor handoff's
                    self.assertEqual([_src(r.value) for r in ast.walk(guard[0])
                                      if isinstance(r, ast.Return)], ["(hidden_states, residual)"])
                    sel = _fn(_cls(comm, "LayerCommunicator"), "_select_ffn_completion")
                    leaves = [n for n in ast.walk(sel) if isinstance(n, ast.Assign)
                              and _src(n.targets[0]) == "leave"]
                    self.assertTrue(leaves)
                    for n in leaves:  # every leave is an UnreducedOutput, or None
                        v = _src(n.value)
                        self.assertTrue(v.startswith("partial(UnreducedOutput") or
                                        v == "None" or "partial(UnreducedOutput" in v, v)
                    red = _src(_fn(_cls(comm, "LayerCommunicator"), "prepare_attn"))
                    self.assertIn("reduce_output(", red)
                    fls = _src(_fn(_cls(comm, "LayerCommunicator"), "finish_layer_stack"))
                    self.assertIn("return (reduce_output(hidden_states), residual)", fls)
                    self.assertIn("finish_layer_stack(", _src(_fn(_cls(ds, "DeepseekV2Model"), "forward")))
                else:
                    self.assertIn(f"hidden_states.{wcore.PARTIAL_SUM_MARKER} = True", s)
                    self.assertIn("if fuse_mlp_allreduce:", s)
                    self.assertIn("if not fuse_mlp_allreduce:", s)
                    self.assertEqual(len(_calls(fwd, "postprocess_layer")), 1)
                    names = [n.name for n in comm.body if isinstance(n, ast.ClassDef)]
                    self.assertNotIn("UnreducedOutput", names)
                    pred = _src(_fn(_cls(comm, "LayerCommunicator"),
                                    "should_fuse_mlp_allreduce_with_next_layer"))
                    self.assertIn("not self.is_last_layer", pred)
                    self.assertIn("self._context.tp_size > 1", pred)


if __name__ == "__main__":
    unittest.main(verbosity=2)
