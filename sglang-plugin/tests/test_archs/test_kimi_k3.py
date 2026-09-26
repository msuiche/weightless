"""The Kimi-K3 rows (``KimiK3ForConditionalGeneration``,
``KimiK3LinearForCausalLM``), CPU only.

``KimiK3DecoderLayer`` returns ``(hidden, residual, sp_sharded)``. With
attention residuals on, the MLP folds the pending prefix sum into its output
and the layer returns ``(stream, None, bool)``: hidden is the whole
post-layer stream. Without them it returns ``(mlp_out, residual, False)``
and the stream is ``hidden + residual``.

- Fakes (the first section), both modes and both served classes, against
  the edit written out by hand; the pipeline ranks, a rank without the
  language model, TP and alpha 0.
- With WEIGHTLESS_TEST_GLP_DIR set, the published GLP-92 file on a fake of
  the checkpoint's depth and width (93 x 7168); with
  WEIGHTLESS_TEST_CONFIG_DIR, the text config and the attention pattern.
- ``RealKimi``: SGLang's real ``KimiK3LinearModel.forward``,
  ``KimiK3DecoderLayer`` in both modes, the real ``KimiK3MLP`` (which folds
  the prefix sum) and the real ``AttnResidual`` on its torch aggregation,
  built with ``__new__``; the block store of the next layers is compared
  too.
- ``StructureKimiK3``: a parse of ``models/kimi_k3.py`` and the Kimi Linear
  config in every SGLang tree found.
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

from test_archs import (_trees, _parse, _cls, _fn, _src, _calls_where, _returns_in_order,
                         _assign, unit_dirs)  # noqa: E402

from weightless_sglang.archs import ARCH  # noqa: E402
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


class KimiK3DecoderLayer(_PairLayer):
    """attn_res mode (``attn_res`` given): the input is the whole stream
    head (residual None between layers); the output is that head plus this
    layer's write (the MLP folds the prefix sum in), residual None. Plain
    mode: the pair convention."""

    def forward(self, positions, hidden_states, forward_batch, residual, attn_res,
                zero_allocator, input_sharded=False, keep_sharded=False):
        if attn_res is not None:
            head = self.fold(hidden_states, residual)
            return head + self.write(head), None, False
        residual = self.fold(hidden_states, residual)
        return self.write(residual), residual, False


KimiK3DeltaAttention = type("KimiK3DeltaAttention", (nn.Module,), {})
KimiK3MLAAttention = type("KimiK3MLAAttention", (nn.Module,), {})


def kimi_k3_kda(num_layers):
    """0-based ids of the KDA (linear) layers in K3's pattern: its config's
    1-based linear_attn_config full_attn_layers are 4, 8, ..., 92 and the
    last layer; SGLang's is_kda_layer(i) tests i + 1."""
    return {i for i in range(num_layers) if (i + 1) % 4 and i != num_layers - 1}


class KimiK3LinearModel(nn.Module):
    def __init__(self, num_layers, width, attn_res=True, start=0, end=None, kda=None):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=width, num_hidden_layers=num_layers,
                                            attn_res_block_size=12 if attn_res else None,
                                            model_type="kimi_linear")
        self.layers = nn.ModuleList(KimiK3DecoderLayer(i, width) if start <= i < (end or num_layers)
                                    else nn.Identity() for i in range(num_layers))
        self.start_layer, self.end_layer = start, num_layers if end is None else end
        kda = kimi_k3_kda(num_layers) if kda is None else set(kda)
        for i, layer in enumerate(self.layers):  # the class name is what the row's kind reads
            if isinstance(layer, KimiK3DecoderLayer):
                layer.self_attn = (KimiK3DeltaAttention if i in kda else KimiK3MLAAttention)()

    def forward(self, hidden_states=None, residual=None):
        attn_res = object() if self.config.attn_res_block_size is not None else None
        for i in range(self.start_layer, self.end_layer):
            hidden_states, residual, _ = self.layers[i](
                positions=None, hidden_states=hidden_states, forward_batch=None,
                residual=residual, attn_res=attn_res, zero_allocator=None,
                input_sharded=False, keep_sharded=False)
        return hidden_states, residual


KimiK3LinearForCausalLM = _wrapper("KimiK3LinearForCausalLM")


class KimiK3ForConditionalGeneration(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.language_model = text  # a KimiK3LinearForCausalLM, or None on an encoder-only rank

    def forward(self, *a, **k):
        return self.language_model.model.forward(*a, **k)


def kimi_runner(num_layers=13, width=8, attn_res=True, wrapper=True, **kw):
    bb = KimiK3LinearModel(num_layers, width, attn_res=attn_res,
                           **{k: kw.pop(k) for k in ("start", "end", "kda") if k in kw})
    text = KimiK3LinearForCausalLM(bb)
    model = KimiK3ForConditionalGeneration(text) if wrapper else text
    return _runner(model, "kimi_k3" if wrapper else "kimi_linear", **kw)


def kimi_backbone(run):
    m = run.model
    return m.language_model.model if hasattr(m, "language_model") else m.model


def kimi_reference(run, x, dirs, alpha):
    bb = kimi_backbone(run)
    attn_res = bb.config.attn_res_block_size is not None
    h, r = x, None
    for i in range(bb.start_layer, bb.end_layer):
        L = bb.layers[i]
        if attn_res:
            head = L.fold(h, r)
            h, r = head + L.write(head), None
        else:
            r = L.fold(h, r)
            h = L.write(r)
        if i in dirs:
            s = h.double() if r is None else (h + r).double()
            d = dirs[i].double()
            s2 = s - alpha * (s @ d).unsqueeze(-1) * d
            h = (s2 if r is None else s2 - r.double()).to(h.dtype)
    return h, r


# ---------------------------------------------------------------- the row on the fakes


HOOK = "residual_stream_post_layer"
KW, KL = "KimiK3ForConditionalGeneration", "KimiK3LinearForCausalLM"

# The published file (huggingface.co/msuiche/<repo name>).
KIMI_FILE = glpfiles.find_glp("glp.kimi-k3-GLP-92-L1-92-a1.gguf")
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


class KimiRows(Base):
    W, L = 16, 13

    def test_modes_and_classes_match_the_reference(self):
        for attn_res in (True, False):
            for wrapper in (True, False):
                with self.subTest(attn_res=attn_res, wrapper=wrapper):
                    path = self.write(range(1, self.L), self.W, hint="kimi_linear")
                    run = kimi_runner(self.L, self.W, attn_res=attn_res, wrapper=wrapper)
                    rec = self.install(run)
                    self.assertEqual(rec["row"], KW if wrapper else KL)
                    self.assertEqual(rec["install"], "hook")
                    self.assertEqual(rec["local_layer_ids"], list(range(1, self.L)))
                    # K3's attention pattern: MLA on 3, 7, 11 and the last layer
                    self.assertEqual(rec["full_attn_layers"], [3, 7, 11, 12])
                    x = torch.randn(5, self.W, generator=torch.Generator().manual_seed(2))
                    out = fwd(run, x)
                    if attn_res:
                        self.assertIsNone(out[1])
                    dirs, _ = dirs_of(path)
                    self.close(out, kimi_reference(run, x, dirs, 1.5))
                    stock = fwd(kimi_runner(self.L, self.W, attn_res=attn_res, wrapper=wrapper), x)
                    self.assertGreater(float((stream(out) - stream(stock)).abs().max()), 1e-2)

    def test_other_slots_pass_through(self):
        for attn_res in (True, False):
            with self.subTest(attn_res=attn_res):
                self.write(range(1, self.L), self.W)
                run = kimi_runner(self.L, self.W, attn_res=attn_res)
                self.install(run)
                layer = kimi_backbone(run).layers[2]
                x = torch.randn(3, self.W)
                out = layer(positions=None, hidden_states=x, forward_batch=None,
                            residual=None if attn_res else torch.randn(3, self.W),
                            attn_res=object() if attn_res else None, zero_allocator=None)
                self.assertEqual(len(out), 3)
                self.assertIs(out[2], False)
                if attn_res:
                    self.assertIsNone(out[1])

    def test_pipeline_ranks_equal_one_rank(self):
        for attn_res in (True, False):
            with self.subTest(attn_res=attn_res):
                self.write(range(1, self.L), self.W)
                x = torch.randn(4, self.W)
                one = kimi_runner(self.L, self.W, attn_res=attn_res)
                self.install(one)
                ref = fwd(one, x)
                r0 = kimi_runner(self.L, self.W, attn_res=attn_res, end=7, pp_size=2)
                r1 = kimi_runner(self.L, self.W, attn_res=attn_res, start=7, pp_rank=1, pp_size=2)
                self.assertEqual(self.install(r0)["local_layer_ids"], list(range(1, 7)))
                self.assertEqual(self.install(r1)["local_layer_ids"], list(range(7, self.L)))
                h, r = fwd(r0, x)
                self.close(kimi_backbone(r1).forward(hidden_states=h, residual=r), ref, atol=0)

    def test_a_rank_without_the_language_model_installs_nothing(self):
        self.write(range(1, self.L), self.W)
        run = kimi_runner(self.L, self.W)
        run.model.language_model = None
        self.assertIsNone(self.install(run))
        self.assertIn("has no backbone on this rank", self.log)

    def test_wrong_backbone_path_fails_closed(self):
        self.write(range(1, self.L), self.W)
        self.patch_row(KW, backbone=("model",))
        with self.assertRaisesRegex(RuntimeError, "none of the backbone paths"):
            self.install(kimi_runner(self.L, self.W))

    def test_tp_is_refused(self):
        self.write(range(1, self.L), self.W)
        for wrapper in (True, False):
            with self.subTest(wrapper=wrapper), \
                    self.assertRaisesRegex(RuntimeError, "not validated at TP=2"):
                self.install(kimi_runner(self.L, self.W, wrapper=wrapper, tp_size=2))

    def test_alpha_zero_is_bitwise_stock(self):
        for attn_res in (True, False):
            with self.subTest(attn_res=attn_res):
                self.write(range(1, self.L), self.W)
                run = kimi_runner(self.L, self.W, attn_res=attn_res)
                self.install(run, WEIGHTLESS_STEER_ALPHA="0")
                x = torch.randn(4, self.W)
                out, stock = fwd(run, x), fwd(kimi_runner(self.L, self.W, attn_res=attn_res), x)
                self.assertTrue(torch.equal(out[0], stock[0]))


@unittest.skipUnless(glpfiles.have(KIMI_FILE), "set WEIGHTLESS_TEST_GLP_DIR to the folder with "
                     "the Kimi-K3 GLP-92 file")
class PublishedFiles(Base):
    def run_file(self, run, path, ref, dirs, alpha, width, tokens=3):
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        rec = self.install(run)
        x = torch.randn(tokens, width, generator=torch.Generator().manual_seed(7))
        out = fwd(run, x)
        self.close(out, ref(run, x, dirs, alpha), atol=5e-5)
        return rec, out

    def test_metadata(self):
        want = {KIMI_FILE: ("kimi_linear", 92, 7168, "1.0", "per-layer", (KW, KL))}
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

    def test_kimi_glp92(self):
        depth, width, kda = 93, 7168, None
        cfg = config("moonshotai__Kimi-K3")
        if cfg is not None:
            t = cfg["text_config"]
            self.assertEqual((t["model_type"], t["hidden_size"], t["num_hidden_layers"],
                              t["attn_res_block_size"]), ("kimi_linear", width, depth, 12))
            kda = {i for i in range(depth) if i + 1 in t["linear_attn_config"]["kda_layers"]}
            self.assertEqual(kda, kimi_k3_kda(depth))
        dirs, alpha = dirs_of(KIMI_FILE)
        full = sorted(set(range(1, 93)) - kimi_k3_kda(depth))
        for attn_res in (True, False):
            for wrapper in (True, False):
                with self.subTest(attn_res=attn_res, wrapper=wrapper):
                    run = kimi_runner(depth, width, attn_res=attn_res, wrapper=wrapper, kda=kda)
                    rec, _ = self.run_file(run, KIMI_FILE, kimi_reference, dirs, alpha, width)
                    self.assertEqual(rec["local_layer_ids"], list(range(1, 93)))
                    self.assertEqual((rec["width"], rec["alpha"], rec["model_hint"]),
                                     (width, 1.0, "kimi_linear"))
                    self.assertEqual(rec["full_attn_layers"], full)


# ---------------------------------------------------------------- SGLang's real classes (R)


def _try(name):
    try:
        return importlib.import_module(name), None
    except Exception as e:  # no SGLang, or a build without this model
        return None, f"{type(e).__name__}: {e}"


KK, KK_WHY = _try("sglang.srt.models.kimi_k3")
AR, AR_WHY = _try("sglang.srt.layers.attn_residual")


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


class _KNorm(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(h), requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        y = rms(x)
        return y if residual is None else (y, residual)


class _KProj(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-1, 1, h).unsqueeze(0) * (0.1 + 0.01 * i),
                                   requires_grad=False)

    def forward(self, x):
        return x @ self.weight.t(), None


class _KAttn(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(0.5, 1.5, h) * (1 + .02 * i), requires_grad=False)

    def forward(self, hidden_states=None, positions=None, forward_batch=None, zero_allocator=None):
        return torch.tanh(hidden_states * self.w)


class _KLin(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = nn.Parameter(w, requires_grad=False)

    def forward(self, x, **kw):
        return x @ self.w, None


class _KAct(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, -1)
        return torch.nn.functional.silu(a) * b


def kimi_mlp(h, i):
    m = KK.KimiK3MLP.__new__(KK.KimiK3MLP)
    nn.Module.__init__(m)
    g = torch.Generator().manual_seed(i)
    m.gate_up_proj = _KLin(torch.randn(h, 2 * h, generator=g) * 0.3)
    m.down_proj = _KLin(torch.randn(h, h, generator=g) * 0.3)
    m.act_fn = _KAct()
    m._dp_attention = False
    m._dense_attn_tp = False
    return m


def kimi_layer(h, i, block):
    L = KK.KimiK3DecoderLayer.__new__(KK.KimiK3DecoderLayer)
    nn.Module.__init__(L)
    L.hidden_size, L.layer_idx, L.is_moe = h, i, False
    L._dp_attention = L._trim_padded_attn = L._sp_moe = L.all_reduce_fusion = False
    L._is_moe_layer = False
    L.self_attn = _KAttn(h, i)
    L.mlp = kimi_mlp(h, i)
    L.input_layernorm, L.post_attention_layernorm = _KNorm(h), _KNorm(h)
    L.use_attn_residuals = block is not None
    if block is not None:
        L.attn_res_block_size = block
        L.is_block_write_layer = i % block == 0
        L.prev_valid_blocks = KK._cdiv(i, block)
        L.self_attention_res_norm, L.mlp_res_norm = _KNorm(h), _KNorm(h)
        L.self_attention_res_proj, L.mlp_res_proj = _KProj(h, i), _KProj(h, i + 100)
    return L


def kimi_model(n, h, block):
    cfg = types.SimpleNamespace(hidden_size=h, num_hidden_layers=n, attn_res_block_size=block,
                                model_type="kimi_linear", rms_norm_eps=1e-6)
    M = KK.KimiK3LinearModel.__new__(KK.KimiK3LinearModel)
    nn.Module.__init__(M)
    M.config = cfg
    M.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1)
    M.dspark_layers_to_capture = None
    M._dp_attention = M._trim_padded_attn = False
    M.embed_tokens = nn.Identity()
    M.alt_streams = None
    M.layers = nn.ModuleList(kimi_layer(h, i, block) for i in range(n))
    M.start_layer, M.end_layer = 0, n
    M.norm = _KNorm(h)
    if block is not None:
        M.output_attn_res_norm = _KNorm(h)
        M.output_attn_res_proj = _KProj(h, 999)
    return M


@contextlib.contextmanager
def kimi_context(writes):
    """The torch aggregation of AttnResidual (the fast kernels need a GPU),
    PP 1, and a record of every block-store write."""
    from sglang.srt.runtime_context import get_parallel

    def agg(prefix_sum, bank, nvb, score_proj, score_norm, out_norm):
        return out_norm(AR.aggregate_stream_torch(prefix_sum, bank, nvb, score_proj, score_norm))

    orig_write = AR.AttnResidual.write

    def write(self, x, *a, **k):
        writes.append(x.detach().clone())
        return orig_write(self, x, *a, **k)

    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch.object(AR, "_FAST_SUPPORTED", False))
        st.enter_context(mock.patch.object(AR, "_aggregate_fused", agg))
        st.enter_context(mock.patch.object(AR.AttnResidual, "write", write))
        st.enter_context(get_parallel().override(pp_group=types.SimpleNamespace(
            is_first_rank=True, is_last_rank=True, world_size=1)))
        yield


def hand_hooks(M, dirs, alpha):
    """The edit written out by hand as forward hooks on a second model: the
    whole stream when the layer returns no residual, else the pair edit."""
    hs = []
    for i, d in dirs.items():
        def hook(mod, args, out, d=d):
            h, r, flag = out
            if r is None:
                return project(h.double(), d, alpha).to(h.dtype), r, flag
            s = project((h + r).double(), d, alpha)
            return (s - r.double()).to(h.dtype), r, flag
        hs.append(M.layers[i].register_forward_hook(hook))
    return hs


@unittest.skipUnless(KK is not None and AR is not None,
                     f"SGLang's Kimi-K3 model is not importable here ({KK_WHY or AR_WHY})")
class RealKimi(RealBase):
    N, H = 13, 8

    def forward(self, M, x, writes):
        fb = types.SimpleNamespace(forward_mode=types.SimpleNamespace(
            is_idle=lambda: False, is_extend=lambda: False), out_cache_loc=None)
        seen = {}

        def pre(mod, args, kwargs):
            seen[mod.layer_idx] = (kwargs["hidden_states"].clone(),
                                   None if kwargs["residual"] is None else kwargs["residual"].clone())

        hs = [l.register_forward_pre_hook(pre, with_kwargs=True) for l in M.layers]
        try:
            with kimi_context(writes):
                y = M(None, torch.arange(x.shape[0]), fb, inputs_embeds=x.clone())
        finally:
            for h in hs:
                h.remove()
        return y, seen

    def test_both_modes_both_classes(self):
        for block in (3, None):
            for wrapper in (True, False):
                with self.subTest(block=block, wrapper=wrapper):
                    path = self.write(range(1, self.N), self.H, hint="kimi_linear")
                    dirs, alpha = real_dirs_of(path)
                    M = kimi_model(self.N, self.H, block)
                    text = named("KimiK3LinearForCausalLM", "model")(M)
                    model = named("KimiK3ForConditionalGeneration", "language_model")(text) \
                        if wrapper else text
                    rec = self.install(runner(model, "kimi_k3" if wrapper else "kimi_linear",
                                              torch.float32))
                    self.assertEqual(rec["local_layer_ids"], list(range(1, self.N)))
                    # the real layers' attention classes are stubs: every layer counts as full
                    x = torch.randn(3, self.H, generator=torch.Generator().manual_seed(1))
                    w_out = []
                    out, seen = self.forward(M, x, w_out)

                    R = kimi_model(self.N, self.H, block)
                    hand_hooks(R, dirs, alpha)
                    w_ref = []
                    ref, seen_ref = self.forward(R, x, w_ref)
                    torch.testing.assert_close(out, ref, atol=2e-5, rtol=1e-5)
                    for i in range(1, self.N):  # the next layer receives the steered value
                        for a, b in zip(seen[i], seen_ref[i]):
                            if a is None or b is None:
                                self.assertIs(a, b)
                            else:
                                torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
                    if block is not None:  # and the block store holds the steered stream
                        self.assertEqual(len(w_out), len(w_ref))
                        self.assertGreaterEqual(len(w_out), 4)
                        for a, b in zip(w_out, w_ref):
                            torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-5)
                        for i in range(1, self.N):
                            self.assertIsNone(seen[i][1])

                    S = kimi_model(self.N, self.H, block)
                    w_stock = []
                    stock, _ = self.forward(S, x, w_stock)
                    self.assertGreater(float((out - stock).abs().max()), 1e-3)
                    if block is not None:
                        self.assertGreater(float((w_out[-1] - w_stock[-1]).abs().max()), 1e-3)

    def test_last_layer_alone_is_steered(self):
        for block in (3, None):
            with self.subTest(block=block):
                path = self.write([self.N - 1], self.H)
                dirs, alpha = real_dirs_of(path)
                M = kimi_model(self.N, self.H, block)
                self.install(runner(named("KimiK3LinearForCausalLM", "model")(M), "kimi_linear",
                                    torch.float32))
                x = torch.randn(3, self.H)
                out, _ = self.forward(M, x, [])
                R = kimi_model(self.N, self.H, block)
                hand_hooks(R, dirs, alpha)
                torch.testing.assert_close(out, self.forward(R, x, [])[0], atol=2e-5, rtol=1e-5)
                stock, _ = self.forward(kimi_model(self.N, self.H, block), x, [])
                self.assertGreater(float((out - stock).abs().max()), 1e-4)

    def test_tp_is_refused(self):
        self.write(range(1, self.N), self.H)
        M = kimi_model(self.N, self.H, 3)
        with self.assertRaisesRegex(RuntimeError, "not validated at TP=2"):
            self.install(runner(named("KimiK3LinearForCausalLM", "model")(M), "kimi_linear",
                                torch.float32, tp_size=2))


# ---------------------------------------------------------------- structure


MODEL_FILE = "srt/models/kimi_k3.py"


@unittest.skipUnless(_trees(MODEL_FILE), "no SGLang package with models/kimi_k3.py found")
class StructureKimiK3(unittest.TestCase):
    """Pins, per SGLang tree, the code paths the Kimi-K3 rows depend on."""

    def each(self, rel):
        return [(t, _parse(t, rel)) for t in _trees(MODEL_FILE)]

    def test_kimi_classes_and_backbones(self):
        wrap, text = ARCH["KimiK3ForConditionalGeneration"], ARCH["KimiK3LinearForCausalLM"]
        self.assertEqual((wrap.backbone, text.backbone), (("language_model.model",), ("model",)))
        for t, m in self.each("srt/models/kimi_k3.py"):
            with self.subTest(tree=t):
                self.assertEqual(_src(_assign(m, "EntryClass")),
                                 "[KimiK3ForConditionalGeneration, KimiK3LinearForCausalLM]")
                init = _src(_fn(_cls(m, "KimiK3ForConditionalGeneration"), "__init__"))
                self.assertIn("self.language_model = None", init)
                self.assertIn("self.language_model = KimiK3LinearForCausalLM(", init)
                self.assertIn("self.model = KimiK3LinearModel(",
                              _src(_fn(_cls(m, "KimiK3LinearForCausalLM"), "__init__")))
                minit = _src(_fn(_cls(m, "KimiK3LinearModel"), "__init__"))
                self.assertIn("self.layers, self.start_layer, self.end_layer = make_layers(", minit)
                self.assertIn("KimiK3DecoderLayer(", minit)
                self.assertEqual(set(wrap.layers), {"KimiK3DecoderLayer"})
                linit = _src(_fn(_cls(m, "KimiK3DecoderLayer"), "__init__"))
                # the row's kind reads the attention class
                self.assertRegex(linit, r"if config\.is_kda_layer\(layer_idx\):\s+self\.self_attn = "
                                        r"KimiK3DeltaAttention\(")
                self.assertIn("self.self_attn = KimiK3MLAAttention(", linit)
        for t in _trees(MODEL_FILE):
            with self.subTest(tree=t):
                cfg = _src(_fn(_cls(_parse(t, "srt/configs/kimi_linear.py"), "KimiLinearConfig"),
                               "is_kda_layer"))
                self.assertIn("layer_idx + 1 in self.linear_attn_config['kda_layers']", cfg)

    def test_kimi_loop_calls_each_layer_once_and_unpacks_three(self):
        self.assertEqual(ARCH["KimiK3LinearForCausalLM"].install, "hook")
        for t, m in self.each("srt/models/kimi_k3.py"):
            with self.subTest(tree=t):
                fwd = _fn(_cls(m, "KimiK3LinearModel"), "forward")
                loops = [n for n in ast.walk(fwd) if isinstance(n, ast.For)]
                self.assertEqual(len(loops), 1)
                self.assertEqual(_src(loops[0].iter), "range(self.start_layer, self.end_layer)")
                calls = _calls_where(fwd, lambda c: isinstance(c.func, ast.Subscript)
                               and _src(c.func.value) == "self.layers")
                self.assertEqual(len(calls), 1)
                self.assertEqual(_src(calls[0].func), "self.layers[i]")
                self.assertEqual([k.arg for k in calls[0].keywords],
                                 ["positions", "hidden_states", "forward_batch", "residual",
                                  "attn_res", "zero_allocator", "input_sharded", "keep_sharded"])
                self.assertIn("hidden_states, residual, sp_sharded = self.layers[i](", _src(loops[0]))
                self.assertEqual(len(_calls_where(fwd, lambda c: _src(c.func) in ("layer", "layer.forward"))),
                                 0)

    def test_kimi_layer_returns(self):
        row = ARCH["KimiK3LinearForCausalLM"]
        self.assertEqual((row.arity, row.hidden_index, row.residual_index), (3, 0, 1))
        for t, m in self.each("srt/models/kimi_k3.py"):
            with self.subTest(tree=t):
                c = _cls(m, "KimiK3DecoderLayer")
                fwd = _fn(c, "forward")
                self.assertEqual([a.arg for a in fwd.args.args],
                                 ["self", "positions", "hidden_states", "forward_batch", "residual",
                                  "attn_res", "zero_allocator", "input_sharded", "keep_sharded"])
                rets = _returns_in_order(fwd)
                self.assertEqual(rets[-1], "(hidden_states, residual, False)")
                self.assertEqual(len(rets), 2)
                self.assertTrue(rets[0].startswith("self._forward_attn_residual("))
                ar = _fn(c, "_forward_attn_residual")
                self.assertEqual(sorted(_returns_in_order(ar)), ["(out, None, False)", "(out, None, True)"])
                s = _src(ar)
                # the MLP folds the pending prefix sum: out is the whole stream
                self.assertIn("out = self.mlp(hidden_states, prefix_sum=prefix_sum, "
                              "forward_batch=forward_batch)", s)
                mlp = _src(_fn(_cls(m, "KimiK3MLP"), "forward"))
                self.assertIn("if prefix_sum is not None:\n        hidden_states = hidden_states + "
                              "prefix_sum", mlp)
                self.assertLess(mlp.index("self.down_proj("), mlp.index("hidden_states + prefix_sum"))
                moe = _fn(_cls(m, "KimiK3MoE"), "forward")
                self.assertIn("prefix_sum", [a.arg for a in moe.args.kwonlyargs])

    def test_kimi_tp_is_refused(self):
        for name in ("KimiK3ForConditionalGeneration", "KimiK3LinearForCausalLM"):
            self.assertEqual(ARCH[name].tp, "refuse")


if __name__ == "__main__":
    unittest.main(verbosity=2)
