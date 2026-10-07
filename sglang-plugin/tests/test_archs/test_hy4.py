"""The Hy4-preview row (``HYV4ForCausalLM``), CPU only.

``HYV4DecoderLayer`` returns ``([T, hc_mult, hidden], topk_indices)``: the iHC
streams merged inside the layer, no residual. The GLP file holds one
hidden-wide direction per layer, applied to each stream on its own.

- Fakes (the first section): the hooked fake against the per-stream edit
  written out by hand and against the vLLM hy4 adapter's apply, and every
  boot refusal the row relies on.
- With WEIGHTLESS_TEST_GLP_DIR set, the published GLP-77 file on a fake of
  the checkpoint's shape (78 x 4 x 6144); with WEIGHTLESS_TEST_CONFIG_DIR,
  the width through SGLang's own ``HYV4Config``.
- ``RealHy4``: SGLang's real ``HYV4Model.forward`` and
  ``HYV4DecoderLayer.forward`` with the real ``HYV4HCLayer`` /
  ``HYV4HCPreLayer`` / ``HYV4HCHeadLayer`` torch paths, inside a real
  ``HYV4ForCausalLM``, built with ``__new__``. Only the attention, the MLP,
  the norms and the two ReplicatedLinear gates are stubs.
- ``StructureHy4``: a parse of ``models/hunyuan_v4.py`` in every SGLang tree
  found (the installed package and WEIGHTLESS_TEST_SGLANG_TREES).
"""
import ast
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import glpfiles  # noqa: E402  (sets up the import paths)
from glpfiles import good_meta, write_gguf  # noqa: E402

from test_archs import (_trees, _parse, _cls, _fn, _src, _calls, _entry, _returns, _loops,
                         _layer_calls, unit_dirs, proj_ok, runner_of, spy_out)  # noqa: E402

from weightless_sglang.archs import ARCH  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402
from weightless_steer.container import read_gguf_cvec  # noqa: E402
from weightless_steer.core import SteeringCore  # noqa: E402


# ---------------------------------------------------------------- fakes
# The first layer widens a 2-D input the way HYV4HCLayer.prepare_input does;
# each layer returns ([T, hc, H], topk). The weights depend only on the layer id.


def _w(layer_id, width, lo=-1.0, hi=1.0, step=0.01):
    return nn.Parameter(torch.linspace(lo, hi, width) * (1 + step * layer_id), requires_grad=False)


def _topk(x, layer_id):
    return torch.full((x.shape[0], 2), layer_id, dtype=torch.int32)


class _HyLayer(nn.Module):
    def __init__(self, layer_id, width, hc):
        super().__init__()
        self.hc = hc
        self.w = _w(layer_id, width, 0.5, 1.5)
        self.g = nn.Parameter(torch.linspace(0.2, 1.0, hc) * (1 + 0.03 * layer_id), requires_grad=False)
        self.layer_idx = layer_id

    flat_output = False  # a wrong layout: the layer returns its streams flattened
    contract = False  # a wrong layout: the layer returns the streams contracted to one

    def forward(self, positions, hidden_states, forward_batch, zero_allocator, prev_topk_indices=None):
        if hidden_states.dim() == 2:  # HYV4HCLayer.prepare_input: widen, or unflatten
            if hidden_states.shape[-1] == self.w.shape[0]:
                hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc, 1)
            else:
                hidden_states = hidden_states.reshape(-1, self.hc, self.w.shape[0])
        mixed = (hidden_states * self.g.view(1, -1, 1)).mean(1)
        out = hidden_states + self.g.view(1, -1, 1) * torch.tanh(mixed * self.w).unsqueeze(1)
        topk = _topk(out, self.layer_idx)
        if self.contract:
            return out.mean(1), topk
        return (out.flatten(1) if self.flat_output else out), topk


HYV4DecoderLayer = type("HYV4DecoderLayer", (_HyLayer,), {})


class HYV4Model(nn.Module):
    """The HYV4Model loop: every layer (``start_layer`` 0, ``end_layer`` N;
    SGLang refuses PP), ``hidden_states, topk = layer(...)``; the head
    contracts the streams after the loop (here: their mean)."""

    def __init__(self, num_layers=78, width=8, hc=4, enable_ihc=True, flat_output=False,
                 model_type="hy_v4"):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=width, hc_mult=hc, enable_ihc=enable_ihc,
                                            num_hidden_layers=num_layers, model_type=model_type)
        self.layers = nn.ModuleList(HYV4DecoderLayer(i, width, hc) for i in range(num_layers))
        for layer in self.layers:
            layer.flat_output = flat_output
        self.start_layer, self.end_layer = 0, num_layers
        self.topk_seen = []

    def forward(self, x):
        h = x
        self.topk_seen = []
        for layer in self.layers:
            h, topk = layer(None, h, None, None, None)
            self.topk_seen.append(topk)
        return h.mean(1) if h.dim() == 3 else h


def _wrapper(name):
    return type(name, (nn.Module,), {
        "__init__": lambda self, bb: (nn.Module.__init__(self), setattr(self, "model", bb))[0]})


HYV4ForCausalLM = _wrapper("HYV4ForCausalLM")


def runner(model, *, hf_model_type, tp_size=1, pp_rank=0, pp_size=1, tbo=False):
    return types.SimpleNamespace(
        model=model, is_draft_worker=False, tp_rank=0, tp_size=tp_size, pp_rank=pp_rank,
        pp_size=pp_size, server_args=types.SimpleNamespace(enable_two_batch_overlap=tbo),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf_model_type),
                                           dtype=torch.float32))


def hy4_runner(num_layers=78, width=8, **kw):
    run_kw = {k: kw.pop(k) for k in ("tp_size", "pp_rank", "pp_size", "tbo") if k in kw}
    return runner(HYV4ForCausalLM(HYV4Model(num_layers, width, **kw)), hf_model_type="hy_v4", **run_kw)


# ---------------------------------------------------------------- the row on the fakes


HOOK = "residual_stream_post_layer"
HY4 = "HYV4ForCausalLM"
try:
    from sglang.srt.layers.communicator import UnreducedOutput  # noqa: F401
    HAVE_UNREDUCED = True
except Exception:
    HAVE_UNREDUCED = False

# The published file (huggingface.co/msuiche/<name without the suffix>).
HY4_FILE = glpfiles.find_glp("Hy4-preview-abliterated-cyber-GLP-77-L1-77-a2.0.gguf")
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


class Hy4Fake(Base):
    L, W, HC = 78, 8, 4
    STEERED = tuple(range(1, 78))

    def setUp(self):
        super().setUp()
        self.write(self.STEERED, self.W, "hy_v4", alpha="2.0")

    def test_row(self):
        row = ARCH[HY4]
        self.assertEqual(row.layers, frozenset({"HYV4DecoderLayer"}))
        self.assertEqual((row.arity, row.hidden_index, row.residual_index), (2, 0, None))
        self.assertTrue(row.per_stream)
        self.assertEqual((row.install, row.exec_id, row.tp), ("hook", "layer", "ok"))
        self.assertEqual(row.hint, frozenset({"hy_v4"}))
        self.assertEqual(row.width(HYV4Model(2, 6144).config), 6144)  # per stream, not 4 x 6144

    def test_forward_is_the_per_stream_edit_and_topk_passes_through(self):
        alpha = 2.0
        run = hy4_runner(self.L, self.W)
        bb = run.model.model.double()
        raw, out = {}, {}
        for i in self.STEERED:
            self.spy(bb.layers[i], raw, i)
        rec = self.install(run)
        self.assertEqual(rec["alpha"], 2.0)
        self.assertEqual(rec["local_layer_ids"], list(self.STEERED))
        for i in self.STEERED:
            self.spy(bb.layers[i], out, i)
        ref = hy4_runner(self.L, self.W).model.model.double()
        stack = bb._steer_stack.double()
        x = torch.randn(3, self.W, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        h = x
        for i in range(self.L):
            h, _ = ref.layers[i](None, h, None, None)
            if i in self.STEERED:
                d = stack[i, 0]
                h = h - alpha * (h @ d).unsqueeze(-1) * d  # each of the hc streams on its own
        self.assertTrue(torch.allclose(bb(x), h.mean(1), atol=1e-12))
        # the vLLM hy4 adapter's apply (SteeringCore.apply) on the same input
        core = SteeringCore.from_env(hook=HOOK, num_layers=self.L, hidden_size=self.W)
        owner = torch.nn.Module()
        core.register_buffers(owner, torch.float64)
        owner._steer_alpha.fill_(alpha)
        for i in self.STEERED:
            self.assertEqual(tuple(out[i][0].shape), (3, self.HC, self.W))
            self.assertIs(out[i][1], raw[i][1])  # topk_indices
            self.assertTrue(torch.allclose(out[i][0], core.apply(i, raw[i][0]), atol=1e-12))
            for s in range(self.HC):
                proj_ok(self, raw[i][0][:, s], out[i][0][:, s], stack[i, 0], alpha)

    def test_flattened_output_is_refused(self):
        run = hy4_runner(self.L, self.W, flat_output=True)
        self.refused(run, r"expected \[T, streams, 8\]", lambda r: r.model.model(torch.randn(2, self.W)))

    def test_last_layer_contracting_its_streams_is_refused(self):
        """Were the head's contraction moved into the last layer, that layer
        would return one [T, H] stream: per_stream refuses it."""
        run = hy4_runner(self.L, self.W)
        run.model.model.layers[self.L - 1].contract = True
        self.refused(run, r"layer 77: slot 0 should hold the hidden stream, expected \[T, streams, "
                          r"8\] \(a floating-point tensor\), found a torch\.float32 tensor of "
                          r"shape \(2, 8\)",
                     lambda r: r.model.model(torch.randn(2, self.W)))

    def test_flattened_width_file_is_refused(self):
        self.write(self.STEERED, self.W * self.HC, "hy_v4")
        self.refused(hy4_runner(self.L, self.W), "width")

    def test_config_without_ihc_is_refused(self):
        self.refused(hy4_runner(self.L, self.W, enable_ihc=False), "no iHC")

    def test_tp2_is_accepted(self):
        self.assertEqual(self.install(hy4_runner(self.L, self.W, tp_size=2))["tp_size"], 2)

    def test_alpha_zero_is_bitwise_stock(self):
        self.write(self.STEERED, self.W, "hy_v4", alpha="0")
        x = torch.randn(5, self.W).bfloat16()
        stock = hy4_runner(self.L, self.W).model.model.bfloat16()(x)
        run = hy4_runner(self.L, self.W)
        run.model.model.bfloat16()
        self.install(run)
        self.assertTrue(torch.equal(run.model.model(x), stock))

    def test_file_refusals(self):
        for kw, pattern in ((dict(hint="glm_moe_dsa"), "model_hint"),
                            (dict(hint="hy_v4", **{"glp.hook_point": "ffn_out_pre_residual"}), "hook_point"),
                            (dict(hint="hy_v4", **{"glp.structure": "per-execution-step"}), "per-execution-step")):
            with self.subTest(kw=kw):
                self.write(self.STEERED, self.W, **kw)
                self.refused(hy4_runner(self.L, self.W), pattern)
        self.refused(hy4_runner(self.L, self.W, tbo=True), "two-batch overlap")


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

    @unittest.skipUnless(glpfiles.have(HY4_FILE), "set WEIGHTLESS_TEST_GLP_DIR to the folder with "
                                                "Hy4-preview-abliterated-cyber-GLP-77-L1-77-a2.0.gguf")
    def test_hy4_glp77(self):
        meta, ids, widths = _meta(HY4_FILE)
        self.assertEqual((meta["controlvector.model_hint"], meta["glp.hook_point"], meta["glp.mode"]),
                         ("hy_v4", HOOK, "project"))
        self.assertEqual(float(meta["glp.alpha_default"]), 2.0)
        self.assertEqual(meta["glp.structure"], "per-layer")
        self.assertEqual((ids, widths), (list(range(1, 78)), {6144}))
        os.environ["WEIGHTLESS_STEER_PATH"] = HY4_FILE
        run = hy4_runner(78, 6144)
        rec = self.install(run)
        self.assertEqual((rec["local_layer_ids"], rec["alpha"], rec["width"]), (ids, 2.0, 6144))
        self.check_stack(run, HY4_FILE, 78, 6144, ids)
        raw, out = {}, {}  # a second copy: the steering hook runs between two spies
        run2 = hy4_runner(78, 6144)
        bb = run2.model.model.double()
        for i in (1, 40, 77):
            self.spy(bb.layers[i], raw, i)
        self.install(run2)
        for i in (1, 40, 77):
            self.spy(bb.layers[i], out, i)
        bb(torch.randn(2, 6144, dtype=torch.float64))
        for i in (1, 40, 77):  # the last layer too: each of the 4 streams, alpha 2
            self.assertEqual(tuple(out[i][0].shape), (2, 4, 6144))
            for s in range(4):
                proj_ok(self, raw[i][0][:, s], out[i][0][:, s], bb._steer_stack[i, 0].double(), 2.0)


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

    def test_hy4(self):
        c = self.cfg_or_skip("tencent__Hy4-preview-FP8")
        self.assertEqual(c["architectures"], [HY4])
        self.assertEqual((c["model_type"], c["num_hidden_layers"], c["hc_mult"], c["enable_ihc"]),
                         ("hy_v4", 78, 4, True))
        try:
            from sglang.srt.configs.hy_v4 import HYV4Config
        except Exception as e:  # pragma: no cover
            self.skipTest(f"SGLang's HYV4Config is not importable here: {e}")
        cfg = HYV4Config(**{k: v for k, v in c.items() if k not in ("architectures", "model_type")})
        self.assertEqual(ARCH[HY4].width(cfg), 6144)
        self.check_file(HY4_FILE, cfg.num_hidden_layers, 6144, cfg.model_type)


# ---------------------------------------------------------------- SGLang's real classes (R)


def _try(fn):
    try:
        return fn(), None
    except Exception as e:  # no SGLang, or a build without the model
        return None, f"{type(e).__name__}: {e}"


def _hy():
    from sglang.srt.models import hunyuan_v4 as HY
    return HY


HYM, HY_WHY = _try(_hy)


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


HY_H, HY_HC, HY_NL = 8, 4, 10
HY_STEERED = tuple(range(1, HY_NL))


class _Lin(nn.Module):
    """Stub ReplicatedLinear: returns (x W^T, None)."""

    def __init__(self, o, i, g):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(o, i, generator=g) * 0.1, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype)), None


class _HyAttn(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.i = i
        self.w = nn.Parameter(torch.linspace(0.5, 1.5, HY_H) * (1 + 0.02 * i), requires_grad=False)
        self.prepare_qkv_latent = None

    def forward(self, positions, hidden_states, forward_batch, zero_allocator, prev_topk_indices=None):
        topk = torch.full((hidden_states.shape[0], 2), self.i, dtype=torch.int32)
        return torch.tanh(hidden_states * self.w), topk


class _HyMLP(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(-1, 1, HY_H) * (1 + 0.01 * i), requires_grad=False)

    def forward(self, x):
        return x * self.w


class _PlainNorm(_Norm):
    def forward(self, x):
        return super().forward(x)


def hy_hc_layer(g, dtype):
    HY = HYM
    P = HY.HYV4HCPreLayer.__new__(HY.HYV4HCPreLayer)
    nn.Module.__init__(P)
    P.hidden_size, P.hc_mult, P.magnitude, P.hc_eps, P.rms_norm_eps = HY_H, HY_HC, 2.0, 1e-6, 1e-6
    P.hc_fn = _Lin(2 * HY_HC, HY_HC * HY_H, g)
    P.hc_scale = nn.Parameter(torch.ones(2), requires_grad=False)
    P.hc_base = nn.Parameter(torch.randn(2 * HY_HC, generator=g) * 0.1, requires_grad=False)
    P._fused_ihc_pre_disabled = False
    L = HY.HYV4HCLayer.__new__(HY.HYV4HCLayer)
    nn.Module.__init__(L)
    L.hidden_size, L.hc_mult, L.hc_pre = HY_H, HY_HC, P
    L._fused_ihc_post_disabled = L._fused_ihc_post_pre_disabled = False
    return L


def hy_runner(dtype=torch.float32):
    HY = HYM
    cfg = types.SimpleNamespace(hidden_size=HY_H, hc_mult=HY_HC, enable_ihc=True, hc_eps=1e-6,
                                rms_norm_eps=1e-6, num_hidden_layers=HY_NL, model_type="hy_v4")
    layers = []
    for i in range(HY_NL):
        g = torch.Generator().manual_seed(100 + i)
        L = HY.HYV4DecoderLayer.__new__(HY.HYV4DecoderLayer)
        nn.Module.__init__(L)
        L.self_attn, L.mlp = _HyAttn(i).to(dtype), _HyMLP(i).to(dtype)
        L.input_layernorm, L.post_attention_layernorm = _PlainNorm(HY_H).to(dtype), _PlainNorm(HY_H).to(dtype)
        L.hc_attn_layer, L.hc_mlp_layer = hy_hc_layer(g, dtype), hy_hc_layer(g, dtype)
        layers.append(L)
    M = HY.HYV4Model.__new__(HY.HYV4Model)
    nn.Module.__init__(M)
    M.config, M.start_layer, M.end_layer = cfg, 0, HY_NL
    M.embed_tokens, M.layers = nn.Identity(), nn.ModuleList(layers)
    Hd = HY.HYV4HCHeadLayer.__new__(HY.HYV4HCHeadLayer)
    nn.Module.__init__(Hd)
    Hd.config, Hd.hc_head_fn = cfg, _Lin(HY_HC, HY_HC * HY_H, torch.Generator().manual_seed(7))
    Hd.hc_head_scale = nn.Parameter(torch.ones(1), requires_grad=False)
    Hd.hc_head_base = nn.Parameter(torch.zeros(HY_HC), requires_grad=False)
    Hd._fused_ihc_head_disabled = False
    M.hc_head, M.norm = Hd, _PlainNorm(HY_H).to(dtype)
    W = HY.HYV4ForCausalLM.__new__(HY.HYV4ForCausalLM)
    nn.Module.__init__(W)
    W.model = M
    return runner_of(W, "hy_v4", dtype=dtype)


def hy_run(run, x):
    return run.model.model(None, None, FB, input_embeds=x)


@unittest.skipUnless(HYM, f"SGLang's hunyuan_v4 model is not importable here ({HY_WHY})")
class RealHy4(RealBase):
    def setUp(self):
        super().setUp()
        self.write(HY_STEERED, HY_H, "hy_v4", "2.0")

    def test_real_classes_install(self):
        run = hy_runner()
        rec = self.install(run)
        self.assertEqual((rec["row"], rec["alpha"], rec["width"]), ("HYV4ForCausalLM", 2.0, HY_H))
        self.assertEqual(rec["local_layer_ids"], list(HY_STEERED))

    def test_each_stream_is_steered_and_reaches_the_next_layer(self):
        alpha = 2.0
        run = hy_runner(dtype=torch.float64)
        bb = run.model.model
        raw, out, seen = {}, {}, {}
        for i in HY_STEERED:
            spy_out(bb.layers[i], raw, i)
        for i in range(1, HY_NL):  # the loop calls layer(positions, hidden, fb, allocator, topk)
            bb.layers[i].register_forward_pre_hook(lambda m, a, _i=i: seen.__setitem__(_i, a[1]))
        self.install(run)
        for i in HY_STEERED:
            spy_out(bb.layers[i], out, i)
        x = torch.randn(3, HY_H, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
        y = hy_run(run, x)
        stack = bb._steer_stack.double()
        core = SteeringCore.from_env(hook=HOOK, num_layers=HY_NL, hidden_size=HY_H)
        owner = nn.Module()
        core.register_buffers(owner, torch.float64)  # the vLLM hy4 adapter's apply
        for i in HY_STEERED:
            pre, post = raw[i][0], out[i][0]
            self.assertEqual(tuple(post.shape), (3, HY_HC, HY_H))
            self.assertIs(out[i][1], raw[i][1])  # topk_indices passed through
            self.assertTrue(torch.allclose(post, core.apply(i, pre), atol=1e-12), i)
            for s in range(HY_HC):
                proj_ok(self, pre[:, s], post[:, s], stack[i, 0], alpha)
            if i + 1 < HY_NL:
                self.assertIs(seen[i + 1], post)  # prepare_input passes a 3-D stream through
        self.assertTrue(torch.allclose(y, bb.hc_head(out[HY_NL - 1][0], bb.norm), atol=1e-12))
        self.assertEqual(tuple(seen[1].shape), (3, HY_HC, HY_H))
        self.assertFalse(torch.equal(seen[2], raw[1][0]))  # what layer 2 reads is steered
        self.assertNotIn(0, raw)  # layer 0 is not steered

    def test_alpha_zero_is_bitwise_stock(self):
        self.write(HY_STEERED, HY_H, "hy_v4", "0")
        x = torch.randn(5, HY_H, generator=torch.Generator().manual_seed(9)).bfloat16()
        stock = hy_run(hy_runner(dtype=torch.bfloat16), x)
        run = hy_runner(dtype=torch.bfloat16)
        self.install(run)
        self.assertTrue(torch.equal(hy_run(run, x), stock))

    def test_flattened_width_file_is_refused(self):
        self.write(HY_STEERED, HY_H * HY_HC, "hy_v4", "2.0")
        with self.assertRaisesRegex(Exception, "width"):
            self.install(hy_runner())


# ---------------------------------------------------------------- structure


HY4_TREES = _trees("srt/models/hunyuan_v4.py")


@unittest.skipUnless(HY4_TREES, "no SGLang package with models/hunyuan_v4.py found")
class StructureHy4(unittest.TestCase):
    def trees(self):
        return [(t, _parse(t, "srt/models/hunyuan_v4.py")) for t in HY4_TREES]

    def test_served_class_backbone_and_no_pp(self):
        for t, hy in self.trees():
            with self.subTest(tree=t):
                self.assertEqual(_entry(hy), "[HYV4ForCausalLM]")
                self.assertIn("self.model = HYV4Model(", _src(_fn(_cls(hy, "HYV4ForCausalLM"), "__init__")))
                init = _src(_fn(_cls(hy, "HYV4Model"), "__init__"))
                self.assertIn("if get_parallel().pp_group.world_size != 1:", init)
                self.assertIn("raise ValueError('HYV4 pipeline parallelism is not supported')", init)
                self.assertIn("self.start_layer = 0", init)
                self.assertIn("self.end_layer = config.num_hidden_layers", init)
                self.assertIn("HYV4DecoderLayer(config, i", init)

    def test_model_loop_and_head(self):
        for t, hy in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(hy, "HYV4Model"), "forward")
                loops = _loops(fwd)
                self.assertEqual(len(loops), 1)
                self.assertEqual(_src(loops[0].target), "layer")
                self.assertEqual(_src(loops[0].iter), "self.layers")
                calls = _layer_calls(loops[0])
                self.assertEqual(len(calls), 1)
                self.assertEqual(_src(calls[0].targets[0]), "(hidden_states, topk_indices)")
                # no TBO path, and the head contracts the streams after the loop
                self.assertNotIn("tbo", _src(fwd))
                self.assertEqual(_returns(fwd), ["self.hc_head(hidden_states, self.norm)"])

    def test_layer_returns_the_merged_streams_and_topk(self):
        for t, hy in self.trees():
            with self.subTest(tree=t):
                lay = _cls(hy, "HYV4DecoderLayer")
                fwd = _fn(lay, "forward")
                self.assertEqual(_returns(fwd), ["(hidden_states, topk_indices)"])
                body = [n for n in fwd.body if isinstance(n, ast.Assign)]
                self.assertEqual(_src(body[-1]),
                                 "hidden_states = self.hc_mlp_layer.post(hidden_states, residual, post)")
                s = _src(fwd)
                # the MLP is called with no deferred-reduction arguments, and
                # no communicator leaves a sum for the next layer
                self.assertEqual(sorted(_src(c) for c in _calls(fwd, "mlp")),
                                 ["self.mlp(hidden_states)", "self.mlp(hidden_states, forward_batch)"])
                self.assertNotIn("layer_communicator", _src(lay))
                self.assertNotIn("fuse", s.replace("post_pre", ""))
                # the attention -> MLP fusion stays inside the layer
                pp = _calls(fwd, "post_pre")
                self.assertEqual(len(pp), 1)
                self.assertIn("self.hc_mlp_layer", _src(pp[0]))

    def test_stream_layout_and_no_reduction_at_the_next_layer_input(self):
        for t, hy in self.trees():
            with self.subTest(tree=t):
                hc = _cls(hy, "HYV4HCLayer")
                prep = _fn(hc, "prepare_input")
                self.assertEqual(_src(prep.body[0]), "if hidden_states.ndim == 3:\n    return hidden_states")
                post = _src(_fn(hc, "post"))
                self.assertIn("result = post.float().unsqueeze(-1) * output.float().unsqueeze(1)", post)
                self.assertIn("return (result + residual.float()).to(output.dtype)", post)
                for fn in (prep, _fn(hc, "pre"), _fn(_cls(hy, "HYV4HCPreLayer"), "forward")):
                    self.assertNotIn("all_reduce", _src(fn))
                    self.assertNotIn("reduce_output", _src(fn))
                cfg = _parse(t, "srt/configs/hy_v4.py")
                cs = _src(_cls(cfg, "HYV4Config"))
                self.assertIn("model_type = 'hy_v4'", cs)
                self.assertIn("if not self.enable_ihc or self.hc_mult <= 0:", cs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
