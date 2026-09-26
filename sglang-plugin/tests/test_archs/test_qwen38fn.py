"""The Qwen3.8-Flash-Next row (``Qwen4ExpForConditionalGeneration``), CPU
only.

The layers return ``([T, hc_count x hidden], None)``: the hyper-connection
streams combined inside the layer, flat and stream-major. The edit is on
that widened stream, with no residual.

- Fakes (the first section): the hooked fake against the edit written out
  by hand, the pipeline ranks, the exact class names (the layers subclass
  the Qwen3_5 ones) and every boot refusal the row relies on.
- With WEIGHTLESS_TEST_GLP_DIR set, the published GLP-47 file on a fake of
  the checkpoint's shape (48 x 4 x 2560); with WEIGHTLESS_TEST_CONFIG_DIR,
  the width through SGLang's own ``Qwen4ExpConfig``.
- ``RealQwen38fn``: SGLang's real ``Qwen4ExpModel.forward`` (through
  ``Qwen4ExpVLModel``), both ``Qwen4Exp*DecoderLayer.forward`` and the real
  ``GatedResidual`` mix / combine, inside a real
  ``Qwen4ExpForConditionalGeneration``, built with ``__new__``. Only the
  mixers and the MLPs are stubs.
- ``StructureQwen38fn``: a parse of ``models/qwen4_exp.py`` and what it
  relies on, in every SGLang tree found.
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
from test_archs import test_qwen38 as fakes  # noqa: E402  (the Qwen3.8-27B fake)

from test_archs import (_trees, _parse, _cls, _fn, _src, _calls, _entry, _returns, _loops,
                         _layer_calls, unit_dirs, proj_ok, OneRank, runner_of, spy_out)  # noqa: E402

from weightless_sglang.archs import ARCH  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402
from weightless_steer.container import read_gguf_cvec  # noqa: E402
from weightless_steer.core import SteeringCore  # noqa: E402


# ---------------------------------------------------------------- fakes
# Each layer returns ([T, hc_count x hidden], None), the streams combined inside
# the layer, stream-major. The weights depend only on the layer id.


def _w(layer_id, width, lo=-1.0, hi=1.0, step=0.01):
    return nn.Parameter(torch.linspace(lo, hi, width) * (1 + step * layer_id), requires_grad=False)


class _QxLayer(nn.Module):
    def __init__(self, layer_id, width, hc):
        super().__init__()
        self.layer_id, self.hc, self.hidden_size = layer_id, hc, width
        self.w = _w(layer_id, width, 0.5, 1.5)
        self.g = nn.Parameter(torch.linspace(0.3, 1.2, hc) * (1 + 0.02 * layer_id), requires_grad=False)

    def forward(self, positions=None, hidden_states=None, residual=None, forward_batch=None, **kw):
        if hidden_states.shape[-1] != self.hc * self.hidden_size:  # the widening of layer 0
            hidden_states = torch.cat([hidden_states] * self.hc, dim=-1)
        streams = hidden_states.unflatten(-1, (self.hc, self.hidden_size))
        out = torch.tanh(streams.mean(-2) * self.w)
        return (streams + out.unsqueeze(-2) * self.g.view(-1, 1)).flatten(-2), None


Qwen4ExpLinearDecoderLayer = type("Qwen4ExpLinearDecoderLayer", (_QxLayer,), {})
Qwen4ExpAttentionDecoderLayer = type("Qwen4ExpAttentionDecoderLayer", (_QxLayer,), {})
Qwen3_5LinearDecoderLayer = type("Qwen3_5LinearDecoderLayer", (_QxLayer,), {})
Qwen3_5AttentionDecoderLayer = type("Qwen3_5AttentionDecoderLayer", (_QxLayer,), {})


class Qwen4ExpVLModel(nn.Module):
    """The Qwen4ExpModel loop: ``hidden_states, residual = layer(...)`` over
    ``[start_layer, end_layer)``, keyword arguments; PP hands on the
    widened hidden_states only; the last rank mixes the streams down after
    the loop (here: their mean) and keeps the stream it mixed."""

    def __init__(self, num_layers=48, width=8, hc=4, start=0, end=None, qwen3_5_names=False,
                 model_type="qwen4_exp_text"):
        super().__init__()
        end = num_layers if end is None else end
        self.config = types.SimpleNamespace(hidden_size=width, hc_count=hc,
                                            num_hidden_layers=num_layers, model_type=model_type)
        lin, att = ((Qwen3_5LinearDecoderLayer, Qwen3_5AttentionDecoderLayer) if qwen3_5_names
                    else (Qwen4ExpLinearDecoderLayer, Qwen4ExpAttentionDecoderLayer))
        self.layers = nn.ModuleList(
            (att if i % 4 == 3 else lin)(i, width, hc) if start <= i < end else nn.Identity()
            for i in range(num_layers))
        self.start_layer, self.end_layer = start, end
        self.last_hc_hidden_states = None

    def forward(self, x=None, proxy=None):
        h, r = (x, None) if proxy is None else (proxy["hidden_states"], None)
        for i in range(self.start_layer, self.end_layer):
            h, r = self.layers[i](positions=None, hidden_states=h, residual=r, forward_batch=None)
        if self.end_layer < self.config.num_hidden_layers:
            return {"hidden_states": h}
        self.last_hc_hidden_states = h
        return h.unflatten(-1, (self.config.hc_count, self.config.hidden_size)).mean(-2)


def _wrapper(name):
    return type(name, (nn.Module,), {
        "__init__": lambda self, bb: (nn.Module.__init__(self), setattr(self, "model", bb))[0]})


Qwen4ExpForConditionalGeneration = _wrapper("Qwen4ExpForConditionalGeneration")
Qwen3_5ForConditionalGeneration = _wrapper("Qwen3_5ForConditionalGeneration")


def runner(model, *, hf_model_type, tp_size=1, pp_rank=0, pp_size=1, tbo=False):
    return types.SimpleNamespace(
        model=model, is_draft_worker=False, tp_rank=0, tp_size=tp_size, pp_rank=pp_rank,
        pp_size=pp_size, server_args=types.SimpleNamespace(enable_two_batch_overlap=tbo),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf_model_type),
                                           dtype=torch.float32))


def qwen38fn_runner(num_layers=48, width=8, cls=None, hf_model_type="qwen4_exp", **kw):
    run_kw = {k: kw.pop(k) for k in ("tp_size", "pp_rank", "pp_size", "tbo") if k in kw}
    return runner((cls or Qwen4ExpForConditionalGeneration)(Qwen4ExpVLModel(num_layers, width, **kw)),
                  hf_model_type=hf_model_type, **run_kw)


# ---------------------------------------------------------------- the row on the fakes


HOOK = "residual_stream_post_layer"
QX = "Qwen4ExpForConditionalGeneration"
try:
    from sglang.srt.layers.communicator import UnreducedOutput  # noqa: F401
    HAVE_UNREDUCED = True
except Exception:
    HAVE_UNREDUCED = False

# The published file (huggingface.co/msuiche/<name without the suffix>).
QX_FILE = glpfiles.find_glp("Qwen3.8-Flash-Next-abliterated-cyber-GLP-47-L1-47-a1.gguf")
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


class Qwen38fnFake(Base):
    L, H, HC = 48, 8, 4
    W = H * HC
    STEERED = tuple(range(1, 48))

    def setUp(self):
        super().setUp()
        self.write(self.STEERED, self.W, "qwen4_exp")

    def test_row(self):
        row = ARCH[QX]
        self.assertEqual(row.layers, frozenset({"Qwen4ExpLinearDecoderLayer", "Qwen4ExpAttentionDecoderLayer"}))
        self.assertEqual((row.arity, row.hidden_index, row.residual_index, row.per_stream), (2, 0, None, False))
        self.assertEqual((row.install, row.exec_id, row.tp), ("hook", "layer", "ok"))
        self.assertEqual(row.hint, frozenset({"qwen4_exp"}))
        self.assertEqual(row.width(Qwen4ExpVLModel(2, 2560).config), 10240)

    def test_forward_is_the_stream_edit_and_the_last_layer_is_steered(self):
        alpha = 1.0
        run = qwen38fn_runner(self.L, self.H)
        bb = run.model.model.double()
        raw, out = {}, {}
        for i in self.STEERED:
            self.spy(bb.layers[i], raw, i)
        rec = self.install(run)
        self.assertEqual(rec["full_attn_layers"], [i for i in self.STEERED if i % 4 == 3])
        self.assertEqual((rec["width"], rec["hidden_size"]), (self.W, self.H))
        for i in self.STEERED:
            self.spy(bb.layers[i], out, i)
        ref = qwen38fn_runner(self.L, self.H).model.model.double()
        stack = bb._steer_stack.double()
        x = torch.randn(3, self.H, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
        h = x
        for i in range(self.L):
            h, _ = ref.layers[i](hidden_states=h, residual=None)
            if i in self.STEERED:
                d = stack[i, 0]
                h = h - alpha * (h @ d).unsqueeze(-1) * d
        y = bb(x)
        self.assertTrue(torch.allclose(bb.last_hc_hidden_states, h, atol=1e-12))
        self.assertTrue(torch.allclose(y, h.unflatten(-1, (self.HC, self.H)).mean(-2), atol=1e-12))
        for i in self.STEERED:
            self.assertIsNone(out[i][1])
            proj_ok(self, raw[i][0], out[i][0], stack[i, 0], alpha)

    def test_pipeline_ranks_equal_one_rank(self):
        x = torch.randn(3, self.H, generator=torch.Generator().manual_seed(7))
        one = qwen38fn_runner(self.L, self.H)
        self.install(one)
        ref = one.model.model(x)
        out = None
        for rank, (s, e) in enumerate(((0, 24), (24, 48))):
            run = qwen38fn_runner(self.L, self.H, start=s, end=e, pp_rank=rank, pp_size=2)
            self.assertEqual(self.install(run)["local_layer_ids"], list(range(max(1, s), e)))
            out = run.model.model(x) if rank == 0 else run.model.model(proxy=out)
            if rank == 0:
                self.assertEqual(sorted(out), ["hidden_states"])  # the widened stream only
                self.assertEqual(tuple(out["hidden_states"].shape), (3, self.W))
        self.assertTrue(torch.equal(out, ref))

    def test_exact_class_names(self):
        # Qwen4Exp model, plain Qwen3_5 layer classes: refused
        self.refused(qwen38fn_runner(self.L, self.H, qwen3_5_names=True), "not one of")
        # Qwen4Exp layers inside a Qwen3_5 model: the Qwen3_5 row reads hidden_size,
        # so the widened file does not fit it, and a hidden-wide file meets unknown classes
        q35 = dict(cls=Qwen3_5ForConditionalGeneration, hf_model_type="qwen3_5")
        self.refused(qwen38fn_runner(self.L, self.H, **q35), "width")
        self.write(self.STEERED, self.H, "qwen3_5")
        self.refused(qwen38fn_runner(self.L, self.H, **q35), "not one of")

    def test_hidden_width_file_is_refused(self):
        self.write(self.STEERED, self.H, "qwen4_exp")
        self.refused(qwen38fn_runner(self.L, self.H), f"width {self.H} != {self.W}|width")

    def test_rank_without_a_language_model_is_refused(self):
        run = qwen38fn_runner(self.L, self.H)
        del run.model.model  # an encoder-only rank: Qwen3VL builds no language model
        self.refused(run, "none of the backbone paths")

    def test_config_with_one_stream_is_refused(self):
        self.refused(qwen38fn_runner(self.L, self.H, hc=1), "hc_count=1")

    def test_tp2_is_accepted(self):
        self.assertEqual(self.install(qwen38fn_runner(self.L, self.H, tp_size=2))["tp_size"], 2)

    def test_alpha_zero_is_bitwise_stock(self):
        self.write(self.STEERED, self.W, "qwen4_exp", alpha="0")
        x = torch.randn(5, self.H).bfloat16()
        stock = qwen38fn_runner(self.L, self.H).model.model.bfloat16()(x)
        run = qwen38fn_runner(self.L, self.H)
        run.model.model.bfloat16()
        self.install(run)
        self.assertTrue(torch.equal(run.model.model(x), stock))

    def test_file_refusals(self):
        for kw, pattern in ((dict(hint="qwen3_5"), "model_hint"),
                            (dict(hint="qwen4_exp", **{"glp.hook_point": "ffn_out_pre_residual"}), "hook_point"),
                            (dict(hint="qwen4_exp", **{"glp.structure": "per-execution-step"}), "per-execution-step")):
            with self.subTest(kw=kw):
                self.write(self.STEERED, self.W, **kw)
                self.refused(qwen38fn_runner(self.L, self.H), pattern)
        self.refused(qwen38fn_runner(self.L, self.H, tbo=True), "two-batch overlap")


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

    @unittest.skipUnless(glpfiles.have(QX_FILE), "set WEIGHTLESS_TEST_GLP_DIR to the folder with "
                                               "Qwen3.8-Flash-Next-abliterated-cyber-GLP-47-L1-47-a1.gguf")
    def test_qwen38fn_glp47(self):
        meta, ids, widths = _meta(QX_FILE)
        self.assertEqual((meta["controlvector.model_hint"], meta["glp.hook_point"], meta["glp.mode"]),
                         ("qwen4_exp", HOOK, "project"))
        self.assertEqual(float(meta["glp.alpha_default"]), 1.0)
        self.assertEqual(meta["glp.structure"], "per-layer")
        self.assertEqual((ids, widths), (list(range(1, 48)), {10240}))
        os.environ["WEIGHTLESS_STEER_PATH"] = QX_FILE
        run = qwen38fn_runner(48, 2560)
        rec = self.install(run)
        self.assertEqual((rec["local_layer_ids"], rec["alpha"], rec["width"]), (ids, 1.0, 10240))
        self.assertEqual(rec["full_attn_layers"], list(range(3, 48, 4)))
        self.check_stack(run, QX_FILE, 48, 10240, ids)
        y = run.model.model(torch.randn(2, 2560))
        self.assertEqual(tuple(run.model.model.last_hc_hidden_states.shape), (2, 10240))
        self.assertTrue(bool(torch.isfinite(y).all()))
        for rank, (s, e) in enumerate(((0, 24), (24, 48))):
            r = qwen38fn_runner(48, 2560, start=s, end=e, pp_rank=rank, pp_size=2)
            self.assertEqual(self.install(r)["local_layer_ids"], list(range(max(1, s), e)))
        # the plain Qwen3_5 row: the widened file does not fit a 5120 stream, and
        # at 10240 the Qwen3_5 model type does not match the file's hint
        self.refused(fakes.runner(width=5120), "width")
        self.refused(fakes.runner(num_layers=48, width=10240), "model_hint='qwen4_exp'")
        self.refused(qwen38fn_runner(48, 2560, qwen3_5_names=True), "not one of")
        if glpfiles.have(glpfiles.GLP49):  # the Qwen3.8-27B file on this row
            os.environ["WEIGHTLESS_STEER_PATH"] = glpfiles.GLP49
            self.refused(qwen38fn_runner(64, 2560), "width")


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

    def test_qwen38fn(self):
        c = self.cfg_or_skip("Qwen__Qwen3.8-Flash-Next")
        self.assertEqual(c["architectures"], [QX])
        self.assertEqual(c["model_type"], "qwen4_exp")
        try:
            from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
        except Exception as e:  # pragma: no cover
            self.skipTest(f"SGLang's Qwen4ExpConfig is not importable here: {e}")
        cfg = Qwen4ExpConfig(**{k: v for k, v in c.items() if k not in ("architectures", "model_type")})
        text = cfg.text_config  # what the backbone (Qwen4ExpVLModel) holds as .config
        self.assertEqual((text.model_type, text.num_hidden_layers, text.hidden_size, text.hc_count),
                         ("qwen4_exp_text", 48, 2560, 4))
        self.assertEqual(text.hc_mult, 4)  # attribute_map: hc_mult -> hc_count
        self.assertEqual(ARCH[QX].width(text), 10240)
        self.check_file(QX_FILE, text.num_hidden_layers, 10240, cfg.model_type)


# ---------------------------------------------------------------- SGLang's real classes (R)


def _try(fn):
    try:
        return fn(), None
    except Exception as e:  # no SGLang, or a build without the model
        return None, f"{type(e).__name__}: {e}"


def _qx():
    from sglang.srt.layers import hyperconnection as HC
    from sglang.srt.models import qwen3_5 as Q35
    from sglang.srt.models import qwen4_exp as Q
    return HC, Q35, Q


QXM, QX_WHY = _try(_qx)


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


FB = types.SimpleNamespace(
    can_run_tbo=False, reuse_dsa_topk_indices=False, spec_info=None, mm_input_embeds=None,
    input_ids=None, forward_mode=types.SimpleNamespace(is_idle=lambda: False, is_extend=lambda **k: False))


QX_H, QX_HC, QX_NL = 8, 4, 8
QX_W = QX_H * QX_HC
QX_STEERED = tuple(range(1, QX_NL))


@contextlib.contextmanager
def _cpu_current_device():
    """GatedResidual places its linears on the current accelerator; on a
    CPU-only process that is the CPU."""
    dm = types.SimpleNamespace(current_device=lambda: "cpu")
    with mock.patch.object(torch, "get_device_module", lambda *a, **k: dm), \
            mock.patch.object(torch.cuda, "current_device", lambda: "cpu"):
        yield


def qx_gated(use_combine, seed, dtype):
    HC = QXM[0]
    hcc = HC.HyperConnectionConfig(hc_count=QX_HC, hidden_size=QX_H, params_dtype=dtype, hc_lowrank=4,
                                   rms_norm_eps=1e-6, hc_per_branch_norm=True)
    with _cpu_current_device():
        g = HC.GatedResidual(hcc, use_mix=True, use_combine=use_combine)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in g.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    g.to(dtype)
    # the real math, run eagerly (torch.compile is not what is under test)
    for name in ("_mix_compute", "_combine_compute"):
        fn = getattr(g, name)
        setattr(g, name, getattr(fn, "_torchdynamo_orig_callable", fn))
    return g


class _QxMixer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(0.5, 1.5, QX_H) * (1 + 0.02 * i), requires_grad=False)

    def forward(self, hidden_states, forward_batch=None, positions=None):
        return torch.tanh(hidden_states * self.w)


class _QxMLP(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(-1, 1, QX_H) * (1 + 0.01 * i), requires_grad=False)

    def forward(self, x):
        return x * self.w


def qx_layer(i, dtype, plain_qwen3_5=False):
    _, Q35, Q = QXM
    cfg = types.SimpleNamespace(hidden_size=QX_H, hc_count=QX_HC, num_experts=0)
    if plain_qwen3_5:
        cls = Q35.Qwen3_5AttentionDecoderLayer if i % 4 == 3 else Q35.Qwen3_5LinearDecoderLayer
    else:
        cls = Q.Qwen4ExpAttentionDecoderLayer if i % 4 == 3 else Q.Qwen4ExpLinearDecoderLayer
    L = cls.__new__(cls)
    nn.Module.__init__(L)
    L.config, L.layer_id, L.hc_count, L.hidden_size, L.ple = cfg, i, QX_HC, QX_H, None
    L.attn_hyper_connection = qx_gated(True, 10 + i, dtype)
    L.mlp_hyper_connection = qx_gated(True, 100 + i, dtype)
    m = _QxMixer(i).to(dtype)
    if i % 4 == 3:  # the attention mixer is a method of the layer: stubbed on the instance
        L.attn_stub = m
        object.__setattr__(L, "self_attention",
                           lambda positions=None, hidden_states=None, forward_batch=None, _m=m:
                           _m(hidden_states))
    else:
        L.linear_attn = m
    L.mlp = _QxMLP(i).to(dtype)
    return L


def qx_runner(partition=(QX_NL,), rank=0, dtype=torch.float32, plain_qwen3_5=False, wrapper=None):
    _, Q35, Q = QXM
    start = sum(partition[:rank])
    end = start + partition[rank]
    cfg = types.SimpleNamespace(hidden_size=QX_H, hc_count=QX_HC, num_hidden_layers=QX_NL,
                                model_type="qwen4_exp_text")
    M = Q.Qwen4ExpVLModel.__new__(Q.Qwen4ExpVLModel)
    nn.Module.__init__(M)
    M.config, M._start_layer, M._end_layer = cfg, start, end
    M.hc_count, M.hidden_size = QX_HC, QX_H
    M.pp_group = types.SimpleNamespace(is_first_rank=start == 0, is_last_rank=end == QX_NL)
    M.embed_tokens = nn.Identity()
    M.layers = nn.ModuleList(qx_layer(i, dtype, plain_qwen3_5) if start <= i < end else nn.Identity()
                             for i in range(QX_NL))
    M.has_ple, M.ple_ngram_size, M.ple_ngram_eos_token_id = False, None, None
    M.hyper_connection_mixer = qx_gated(False, 999, dtype) if end == QX_NL else nn.Identity()
    M.last_hc_hidden_states = None
    cls = wrapper or Q.Qwen4ExpForConditionalGeneration
    W = cls.__new__(cls)
    nn.Module.__init__(W)
    W.model = M
    return runner_of(W, "qwen4_exp", pp_rank=rank, pp_size=len(partition), dtype=dtype)


def qx_run(run, x=None, proxy=None):
    from sglang.srt.runtime_context import get_parallel
    fb = types.SimpleNamespace(**vars(FB))
    with get_parallel().override(attn_tp_group=OneRank()):  # attn_tp_all_reduce at one rank
        return run.model.model(None, None, fb, input_embeds=x, pp_proxy_tensors=proxy)


@unittest.skipUnless(QXM, f"SGLang's qwen4_exp model is not importable here ({QX_WHY})")
class RealQwen38fn(RealBase):
    def setUp(self):
        super().setUp()
        self.write(QX_STEERED, QX_W, "qwen4_exp", "1.0")

    def test_real_classes_install(self):
        run = qx_runner()
        rec = self.install(run)
        self.assertEqual((rec["row"], rec["width"], rec["hidden_size"]),
                         ("Qwen4ExpForConditionalGeneration", QX_W, QX_H))
        self.assertEqual(rec["local_layer_ids"], list(QX_STEERED))
        self.assertEqual(rec["full_attn_layers"], [i for i in QX_STEERED if i % 4 == 3])

    def test_steered_stream_reaches_next_layer_and_the_last_layer(self):
        alpha = 1.0
        run = qx_runner(dtype=torch.float64)
        bb = run.model.model
        raw, out, seen = {}, {}, {}
        for i in QX_STEERED:
            spy_out(bb.layers[i], raw, i)
        for i in range(1, QX_NL):  # the loop calls layer(hidden_states=..., residual=..., ...)
            bb.layers[i].register_forward_pre_hook(
                lambda m, a, k, _i=i: seen.__setitem__(_i, (k["hidden_states"], k["residual"])),
                with_kwargs=True)
        self.install(run)
        for i in QX_STEERED:
            spy_out(bb.layers[i], out, i)
        x = torch.randn(3, QX_H, dtype=torch.float64, generator=torch.Generator().manual_seed(6))
        y = qx_run(run, x=x)
        stack = bb._steer_stack.double()
        for i in QX_STEERED:
            pre, post = raw[i][0], out[i][0]
            self.assertEqual(tuple(post.shape), (3, QX_W))
            self.assertIsNone(out[i][1])
            proj_ok(self, pre, post, stack[i, 0], alpha)
            if i + 1 < QX_NL:
                self.assertIs(seen[i + 1][0], post)
                self.assertIsNone(seen[i + 1][1])
        last = out[QX_NL - 1][0]
        self.assertIs(bb.last_hc_hidden_states, last)  # the stream the final mixer read
        self.assertTrue(torch.allclose(y, bb.hyper_connection_mixer.mix(last)[0], atol=1e-12))

    def test_pipeline_ranks_equal_one_rank(self):
        x = torch.randn(3, QX_H, generator=torch.Generator().manual_seed(2))
        one = qx_runner()
        self.install(one)
        ref = qx_run(one, x=x)
        self.assertFalse(torch.allclose(ref, qx_run(qx_runner(), x=x)))
        out = None
        for rank in range(2):
            run = qx_runner((4, 4), rank)
            self.assertEqual(self.install(run)["local_layer_ids"], [[1, 2, 3], [4, 5, 6, 7]][rank])
            out = qx_run(run, x=x) if rank == 0 else qx_run(run, proxy=out)
            if rank == 0:
                self.assertEqual(sorted(out.tensors), ["hidden_states"])  # the widened stream only
                self.assertEqual(tuple(out["hidden_states"].shape), (3, QX_W))
        self.assertTrue(torch.equal(out, ref))

    def test_alpha_zero_is_bitwise_stock(self):
        self.write(QX_STEERED, QX_W, "qwen4_exp", "0")
        x = torch.randn(5, QX_H, generator=torch.Generator().manual_seed(9)).bfloat16()
        stock = qx_run(qx_runner(dtype=torch.bfloat16), x=x)
        run = qx_runner(dtype=torch.bfloat16)
        self.install(run)
        self.assertTrue(torch.equal(qx_run(run, x=x), stock))

    def test_exact_class_names(self):
        _, Q35, _ = QXM
        # SGLang's plain Qwen3_5 layers (the Qwen4Exp layers' base classes) in the Qwen4Exp model
        with self.assertRaisesRegex(Exception, "not one of"):
            self.install(qx_runner(plain_qwen3_5=True))
        # the Qwen4Exp layers in a Qwen3_5 model: the Qwen3_5 row reads hidden_size
        run = qx_runner(wrapper=Q35.Qwen3_5ForConditionalGeneration)
        run.model_config.hf_config.model_type = "qwen3_5"
        with self.assertRaisesRegex(Exception, f"width {QX_W} != {QX_H}|width"):
            self.install(run)
        self.write(QX_STEERED, QX_H, "qwen3_5", "1.0")
        with self.assertRaisesRegex(Exception, "not one of"):
            self.install(run)


# ---------------------------------------------------------------- structure


QX_TREES = _trees("srt/models/qwen4_exp.py")


@unittest.skipUnless(QX_TREES, "no SGLang package with models/qwen4_exp.py found")
class StructureQwen38fn(unittest.TestCase):
    def trees(self):
        return [(t, _parse(t, "srt/models/qwen4_exp.py")) for t in QX_TREES]

    def test_served_class_backbone_and_layer_classes(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                self.assertEqual(_entry(q), "[Qwen4ExpForConditionalGeneration]")
                w = _cls(q, "Qwen4ExpForConditionalGeneration")
                self.assertEqual([_src(b) for b in w.bases], ["Qwen3VLForConditionalGeneration"])
                self.assertIn("language_model_cls=Qwen4ExpVLModel", _src(_fn(w, "__init__")))
                vl = _parse(t, "srt/models/qwen3_vl.py")
                self.assertIn("self.model = language_model_cls(",
                              _src(_fn(_cls(vl, "Qwen3VLForConditionalGeneration"), "__init__")))
                self.assertEqual([_src(b) for b in _cls(q, "Qwen4ExpVLModel").bases], ["Qwen4ExpModel"])
                self.assertEqual([_src(b) for b in _cls(q, "Qwen4ExpModel").bases], ["Qwen3_5ForCausalLM"])
                # the layers subclass the Qwen3_5 ones: the row must match exact names
                self.assertEqual([_src(b) for b in _cls(q, "Qwen4ExpLinearDecoderLayer").bases],
                                 ["Qwen4ExpLayerExtensionMixin", "Qwen3_5LinearDecoderLayer"])
                self.assertEqual([_src(b) for b in _cls(q, "Qwen4ExpAttentionDecoderLayer").bases],
                                 ["Qwen4ExpLayerExtensionMixin", "Qwen3_5AttentionDecoderLayer"])
                types_map = [n for n in q.body if isinstance(n, ast.Assign)
                             and _src(n.targets[0]) == "ALL_DECODER_LAYER_TYPES"]
                self.assertEqual(len(types_map), 1)
                self.assertEqual(set(ast.literal_eval(_src(types_map[0].value).replace(
                    "Qwen4ExpAttentionDecoderLayer", "'A'").replace("Qwen4ExpLinearDecoderLayer", "'L'")).values()),
                    {"A", "L"})

    def test_model_loop_pp_and_final_mixer(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(q, "Qwen4ExpModel"), "forward")
                loops = _loops(fwd)
                self.assertEqual(len(loops), 1)
                self.assertEqual(_src(loops[0].iter), "range(self.start_layer, self.end_layer)")
                self.assertIn("layer = self.layers[i]", _src(loops[0]))
                calls = _layer_calls(loops[0])
                self.assertEqual(len(calls), 1)
                self.assertEqual(_src(calls[0].targets[0]), "(hidden_states, residual)")
                s = _src(fwd)
                self.assertNotIn("tbo", s)
                self.assertIn("proxy_tensors = {'hidden_states': hidden_states}", s)
                self.assertIn("residual = None", s)
                # no last-layer trap: the final mixer runs after the loop on the stream
                self.assertIn("hc_hidden_states = hidden_states", s)
                self.assertIn("hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)", s)
                # the vLLM build adds deepstack embeddings to the stream inside the loop,
                # before its steering; SGLang's VL model takes them and drops them
                vlf = _fn(_cls(q, "Qwen4ExpVLModel"), "forward")
                body = "\n".join(_src(n) for n in vlf.body)
                self.assertNotIn("deepstack", body)
                self.assertNotIn("deepstack", s)

    def test_layers_return_the_combined_stream_and_none(self):
        for t, q in self.trees():
            with self.subTest(tree=t):
                for name in ("Qwen4ExpLinearDecoderLayer", "Qwen4ExpAttentionDecoderLayer"):
                    self.assertEqual(_returns(_fn(_cls(q, name), "forward")),
                                     ["self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)"])
                mix = _cls(q, "Qwen4ExpLayerExtensionMixin")
                post = _fn(mix, "_postprocess_qwen4_exp_layer")
                self.assertEqual(_src(post.body[0]),
                                 "hidden_states = self.mlp_hyper_connection.combine(hidden_states, residual)")
                self.assertEqual(_returns(post), ["(hidden_states, None)"])

    def test_no_reduction_left_for_the_next_layer(self):
        """The mixin deletes the layer communicator; the MLP is called with no
        deferred finalize; the next layer's input path runs no reduction and
        adds the PLE after the boundary (inside the next layer)."""
        for t, q in self.trees():
            with self.subTest(tree=t):
                mix = _cls(q, "Qwen4ExpLayerExtensionMixin")
                init = _src(_fn(mix, "_init_qwen4_exp_layer_extensions"))
                self.assertIn("'layer_communicator'", init)
                self.assertIn("delattr(self, attr_name)", init)
                run = _fn(mix, "_run_qwen4_exp_mlp")
                self.assertEqual(sorted(_src(c) for c in _calls(run, "mlp")),
                                 ["self.mlp(hidden_states)", "self.mlp(hidden_states, forward_batch)"])
                prep = _src(_fn(mix, "_prepare_qwen4_exp_attn"))
                self.assertNotIn("all_reduce", prep)
                self.assertNotIn("reduce_output", prep)
                self.assertIn("hidden_states = hidden_states + self.ple(", prep)
                self.assertIn("hidden_states, residual = self.attn_hyper_connection.mix(hidden_states)", prep)
                for name in ("Qwen4ExpLinearDecoderLayer", "Qwen4ExpAttentionDecoderLayer"):
                    self.assertNotIn("layer_communicator", _src(_fn(_cls(q, name), "forward")))

    def test_stream_layout_and_width_source(self):
        for t, _ in self.trees():
            with self.subTest(tree=t):
                hcm = _parse(t, "srt/layers/hyperconnection.py")
                gr = _src(_cls(hcm, "GatedResidual"))
                self.assertIn("R = residual.unflatten(-1, (hc, hs))", gr)  # stream-major
                self.assertIn("return (R + injection).flatten(-2)", gr)
                cfg = _parse(t, "srt/configs/qwen4_exp.py")
                tc = _src(_cls(cfg, "Qwen4ExpTextConfig"))
                self.assertIn("model_type = 'qwen4_exp_text'", tc)
                self.assertIn("attribute_map = {'hc_mult': 'hc_count'}", tc)
                self.assertIn("self.hc_count = hc_count", tc)
                self.assertIn("model_type = 'qwen4_exp'", _src(_cls(cfg, "Qwen4ExpConfig")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
