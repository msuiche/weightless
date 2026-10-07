"""The GLM-5.3-Flash row (``Glm5NextForConditionalGeneration``), CPU only.

The site is ``layer.layer_communicator.mhc.mlp_combine``: the widened mHC
stream [T, hc_mult x hidden], steered with r=None on every steered layer,
the last one before its hc_contract. Under pipeline parallelism each rank
steers the file's layers in its own [start_layer, end_layer), by absolute
layer id, and hands the steered stream on.

- Fakes shaped like SGLang's classes (the first section) and the row on
  them: install, the partitions, the edit, refusals, diagnostics.
- With WEIGHTLESS_TEST_GLP_DIR set, the published GLP-44 file is installed
  on a 45-layer fake at the real width (16384) over the pipeline
  partitions. With WEIGHTLESS_TEST_GLM_CONFIG set to a GLM-5.3-Flash
  config.json, the file is also checked against it.
- The golden test: the handler's output bits, recorded.
- ``RealLayers``: SGLang's real ``Glm5NextDecoderLayer.forward``,
  ``MHCLayerCommunicator``, ``MHCState``, the real mHC torch math and the
  real ``Glm5NextModel.forward`` loop and ``PPProxyTensors``, on a small
  width. Only the attention, the MLP and the norms are stubs. Built with
  ``__new__`` so no weights, no GPU and no distributed setup are needed.
  Skips when this interpreter cannot import SGLang's GLM model.
- ``Structure``: a parse of ``models/glm5_next.py``,
  ``layers/communicator_mhc.py`` and ``layers/communicator.py`` that pins
  what the row relies on, in the installed SGLang package and every
  package folder listed in WEIGHTLESS_TEST_SGLANG_TREES (os.pathsep
  separated), without importing them.
"""
import ast
import dataclasses
import hashlib
import io
import json
import math
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

from test_archs import _trees, _parse, _cls, _fn, _src, _calls, unit_dirs  # noqa: E402

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.archs import ARCH, SPECIAL  # noqa: E402
from weightless_sglang.core import layer_verdict  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402


# ---------------------------------------------------------------- fakes shaped like SGLang's classes
# They follow the SGLang classes call for call: the backbone at .model (None on
# an encoder-only rank); a full-length layers list with stand-ins outside this
# rank's [start_layer, end_layer); the layer runs prepare_attn (hc_expand on
# layer 0, then hc_pre), the attention, prepare_mlp (hc_post then hc_pre), the
# MLP and postprocess_layer (mhc.mlp_combine, then hc_contract on the last
# layer) and returns a 3-tuple with residual None. The hc_pre / hc_post math is
# a small torch version of the real one, enough to make every stream matter
# and to keep the layout stream-major.


def hc_expand(x, n):
    return x.repeat(1, n)


def hc_contract(x, n):
    return x.unflatten(-1, (n, -1)).mean(dim=-2)


def is_kda_layer(i):
    """GLM-5.3-Flash: DSA (full attention) on layers 3, 7, ..., 43."""
    return i % 4 != 3


@dataclasses.dataclass
class MHCState:
    hc_mult: int
    hc_attn_pre: object
    hc_ffn_pre: object
    hc_post: object
    hc_ffn_post_pre: object = None
    h_res: object = None
    h_post: object = None

    def attn_split(self, hidden_states, out_norm=None):
        residual = hidden_states
        hidden_states, self.h_res, self.h_post = self.hc_attn_pre(hidden_states)
        return hidden_states, residual

    def attn_to_mlp(self, hidden_states, residual, out_norm=None):
        hidden_states = self.hc_post(hidden_states, residual, self.h_res, self.h_post)
        residual = hidden_states
        hidden_states, self.h_res, self.h_post = self.hc_ffn_pre(hidden_states)
        return hidden_states, residual

    def mlp_combine(self, hidden_states, residual):
        return self.hc_post(hidden_states, residual, self.h_res, self.h_post)

    def reset_aux(self):
        self.h_res = None
        self.h_post = None


class _Communicator:
    def __init__(self, mhc, is_first_layer, is_last_layer):
        self.mhc = mhc
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer
        self.fuse_with_next = False  # SGLang: always False for mHC

    def prepare_attn(self, hidden_states, residual, forward_batch):
        if self.is_first_layer:
            hidden_states = hc_expand(hidden_states, self.mhc.hc_mult)
        return self.mhc.attn_split(hidden_states)

    def prepare_mlp(self, hidden_states, residual, forward_batch):
        return self.mhc.attn_to_mlp(hidden_states, residual)

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        hidden_states = self.mhc.mlp_combine(hidden_states, residual)
        if self.is_last_layer:
            hidden_states = hc_contract(hidden_states, self.mhc.hc_mult)
        self.mhc.reset_aux()
        return hidden_states, None

    def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
        return self.fuse_with_next


MHCLayerCommunicator = type("MHCLayerCommunicator", (_Communicator,), {})
LayerCommunicator = type("LayerCommunicator", (_Communicator,), {})


def _vec(gen, n, lo, hi):
    return lo + (hi - lo) * torch.rand(n, generator=gen)


class _DecoderLayer(nn.Module):
    """Weights depend only on the layer id, so the ranks of a pipeline
    built one by one hold the same model."""

    def __init__(self, cfg, layer_id):
        super().__init__()
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.layer_id = layer_id
        self.is_nextn = False
        self.is_linear_attn = is_kda_layer(layer_id)
        H, n = cfg.hidden_size, cfg.hc_mult
        g = torch.Generator().manual_seed(1000 + layer_id)
        self.w_attn = nn.Parameter(_vec(g, H, 0.5, 1.5), requires_grad=False)
        self.w_mlp = nn.Parameter(_vec(g, H, -1.0, 1.0), requires_grad=False)
        self.pre_attn = nn.Parameter(_vec(g, n, -1.0, 1.0), requires_grad=False)
        self.pre_ffn = nn.Parameter(_vec(g, n, -1.0, 1.0), requires_grad=False)
        self.post = nn.Parameter(_vec(g, n, -1.0, 1.0), requires_grad=False)
        mix = torch.eye(n) * 0.7 + 0.3 / n  # doubly stochastic
        self.register_buffer("res_mix", mix[torch.randperm(n, generator=g)], persistent=False)
        mhc = MHCState(hc_mult=n, hc_attn_pre=self.hc_attn_pre, hc_ffn_pre=self.hc_ffn_pre,
                       hc_post=self.hc_post)
        comm_cls = MHCLayerCommunicator if cfg.mhc else LayerCommunicator
        self.layer_communicator = comm_cls(mhc, layer_id == 0,
                                           layer_id == cfg.num_hidden_layers - 1)

    def _hc_pre(self, pre, x):
        T = x.shape[0]
        n = self.config.hc_mult
        w = torch.softmax(pre.float(), 0)
        streams = x.float().view(T, n, -1)
        layer_input = (w.view(1, n, 1) * streams).sum(1).to(x.dtype)
        h_res = self.res_mix.reshape(1, n * n).expand(T, n * n)
        h_post = (2.0 * torch.sigmoid(self.post.float())).expand(T, n)
        return layer_input, h_res, h_post

    def hc_attn_pre(self, x):
        return self._hc_pre(self.pre_attn, x)

    def hc_ffn_pre(self, x):
        return self._hc_pre(self.pre_ffn, x)

    def hc_post(self, x, residual, h_res, h_post):
        T, H = x.shape
        n = self.config.hc_mult
        res = residual.float().view(T, n, H)
        out = h_post.view(T, n, 1) * x.float().unsqueeze(1) + (
            h_res.view(T, n, n).unsqueeze(-1) * res.unsqueeze(2)).sum(1)
        return out.to(x.dtype).view(T, n * H)

    def forward(self, positions, hidden_states, forward_batch, residual, zero_allocator=None,
                gemm_output_zero_allocator=None, prev_topk_indices=None):
        comm = self.layer_communicator
        hidden_states, residual = comm.prepare_attn(hidden_states, residual, forward_batch)
        hidden_states = torch.tanh(hidden_states * self.w_attn)
        topk_indices = None if self.is_linear_attn else torch.zeros(hidden_states.shape[0], 1)
        hidden_states, residual = comm.prepare_mlp(hidden_states, residual, forward_batch)
        hidden_states = hidden_states * self.w_mlp
        if not comm.should_fuse_mlp_allreduce_with_next_layer(forward_batch):
            hidden_states, residual = comm.postprocess_layer(hidden_states, residual,
                                                             forward_batch)
        return hidden_states, residual, topk_indices


Glm5NextDecoderLayer = type("Glm5NextDecoderLayer", (_DecoderLayer,), {})
DeepseekV2DecoderLayer = type("DeepseekV2DecoderLayer", (_DecoderLayer,), {})
PPMissingLayer = type("PPMissingLayer", (nn.Identity,), {})


class Glm5NextModel(nn.Module):
    def __init__(self, cfg, start, end, layer_cls=Glm5NextDecoderLayer):
        super().__init__()
        self.config = cfg
        L = cfg.num_hidden_layers
        self.layers = nn.ModuleList(layer_cls(cfg, i) if start <= i < end else PPMissingLayer()
                                    for i in range(L))
        self.start_layer, self.end_layer = start, end

    def forward(self, input_embeds=None, pp_hidden_states=None):
        first = self.start_layer == 0
        hidden_states = input_embeds if first else pp_hidden_states
        residual = None
        topk_indices = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states, residual, topk_indices = layer(
                None, hidden_states, None, residual, None, None, prev_topk_indices=topk_indices)
        if self.end_layer != self.config.num_hidden_layers:
            return {"hidden_states": hidden_states}  # PPProxyTensors: the stream only
        return hidden_states


class Glm5NextForConditionalGeneration(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone


def config(hidden=8, hc_mult=4, num_layers=45, mhc=True):
    return types.SimpleNamespace(hidden_size=hidden, hc_mult=hc_mult, mhc=mhc,
                                 num_hidden_layers=num_layers, model_type="glm5_next_text")


def stage_bounds(partition, rank):
    start = sum(partition[:rank])
    return start, start + partition[rank]


def runner(partition=(45,), rank=0, hidden=8, hc_mult=4, mhc=True, num_layers=None,
           layer_cls=Glm5NextDecoderLayer, dtype=torch.float32):
    L = sum(partition) if num_layers is None else num_layers
    cfg = config(hidden, hc_mult, L, mhc)
    start, end = stage_bounds(partition, rank)
    bb = Glm5NextModel(cfg, start, end, layer_cls).to(dtype)
    return types.SimpleNamespace(
        model=Glm5NextForConditionalGeneration(bb), is_draft_worker=False, tp_rank=0,
        tp_size=1, pp_rank=rank, pp_size=len(partition),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="glm5_next"),
                                           dtype=dtype))


def pipeline(runners, x):
    """Run the pipeline ranks in order: rank 0 takes the embeddings, each
    later rank takes the stream the previous one handed over."""
    out = None
    for n, run in enumerate(runners):
        bb = run.model.model
        out = bb(input_embeds=x) if n == 0 else bb(pp_hidden_states=out["hidden_states"])
    return out


# ---------------------------------------------------------------- the row on the fakes


GLM = "Glm5NextForConditionalGeneration"
STEERED = tuple(range(1, 45))  # GLP-44: layers 1-44 of 45; layer 0 is never steered
# Pipeline partitions the model may be served with, and the steered layers
# each rank must own.
PARTITIONS = {
    (15, 15, 15): [list(range(1, 15)), list(range(15, 30)), list(range(30, 45))],
    (16, 15, 14): [list(range(1, 16)), list(range(16, 31)), list(range(31, 45))],
    (17, 14, 14): [list(range(1, 17)), list(range(17, 31)), list(range(31, 45))],
}


class GlmBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)

    def write(self, layers=STEERED, width=32, alpha="2.0", hint="glm5_next", name="v.gguf",
              **extra):
        m = good_meta(layers=layers, alpha=alpha)
        m["controlvector.model_hint"] = hint
        m["glp.structure"] = "per-layer"
        m.update(extra)
        path = os.path.join(self.tmp.name, name)
        write_gguf(path, m, unit_dirs(layers, width))
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        return path

    def install(self, run):
        with redirect_stderr(io.StringIO()) as err:
            rec = install_steering(run, source="env")
        self.log = err.getvalue()
        return rec

    @staticmethod
    def wrapped(run):
        bb = run.model.model
        return sorted(i for i in range(bb.start_layer, bb.end_layer)
                      if "mlp_combine" in vars(bb.layers[i].layer_communicator.mhc))


class Row(unittest.TestCase):
    def test_row_is_declared(self):
        row = ARCH[GLM]
        self.assertEqual(row.backbone, ("model",))
        self.assertEqual(row.layers, frozenset({"Glm5NextDecoderLayer"}))
        self.assertEqual(row.hooks, frozenset({"residual_stream_post_layer"}))
        self.assertEqual((row.arity, row.hidden_index, row.residual_index), (3, 0, None))
        self.assertEqual(row.install, "special:glm_mhc_combine")
        self.assertEqual((row.exec_id, row.tp, row.per_stream), ("layer", "refuse", False))
        self.assertIn("glm5_next", row.hint)
        self.assertEqual(row.width(config(4096, 4)), 16384)
        with self.assertRaisesRegex(RuntimeError, "mhc off"):
            row.width(config(4096, 4, mhc=False))
        self.assertEqual(SPECIAL["glm_mhc_combine"][1], "mhc.mlp_combine")


class Install(GlmBase):
    def test_partitions_steer_exactly_the_local_layers(self):
        self.write()
        for part, expect in PARTITIONS.items():
            stacks = []
            for rank in range(3):
                with self.subTest(partition=part, rank=rank):
                    run = runner(part, rank)
                    rec = self.install(run)
                    self.assertEqual(rec["local_layer_ids"], expect[rank])
                    self.assertEqual(self.wrapped(run), expect[rank])
                    bb = run.model.model
                    self.assertEqual(tuple(bb._steer_stack.shape), (45, 1, 32))
                    nz = [int(i) for i in (bb._steer_stack.abs().sum((1, 2)) > 0).nonzero().flatten()]
                    self.assertEqual(nz, list(STEERED))  # every rank holds the whole stack
                    self.assertEqual(float(bb._steer_stack[0].abs().sum()), 0.0)
                    self.assertFalse(any(l._forward_hooks for l in bb.layers))
                    self.assertEqual((rec["pp_rank"], rec["pp_size"]), (rank, 3))
                    self.assertEqual(rec["site"], "mhc.mlp_combine")
                    self.assertEqual(rec["install"], "special:glm_mhc_combine")
                    self.assertEqual((rec["width"], rec["hidden_size"], rec["num_layers"]), (32, 8, 45))
                    self.assertEqual(rec["full_attn_layers"], [i for i in expect[rank] if i % 4 == 3])
                    n_full = len(rec["full_attn_layers"])
                    self.assertIn(f"({len(expect[rank])}; {n_full} full-attn, "
                                  f"{len(expect[rank]) - n_full} linear)", self.log)
                    self.assertIn("pp_rank=%d" % rank, self.log)
                    stacks.append(bb._steer_stack.clone())
            self.assertTrue(all(torch.equal(stacks[0], s) for s in stacks[1:]))
            self.assertEqual(sum(len(e) for e in expect), 44)

    def test_pipeline_equals_one_rank_and_steers(self):
        """Each partition's three ranks, run in turn, give the same bits as
        one rank holding every layer; and the steering changes the output."""
        self.write()
        x = torch.randn(5, 8, generator=torch.Generator().manual_seed(3))
        one = runner((45,), 0)
        self.install(one)
        ref = one.model.model(input_embeds=x)
        stock = runner((45,), 0).model.model(input_embeds=x)
        self.assertEqual(tuple(ref.shape), (5, 8))
        self.assertFalse(torch.allclose(ref, stock))
        for part in PARTITIONS:
            with self.subTest(partition=part):
                runs = [runner(part, r) for r in range(3)]
                for run in runs:
                    self.install(run)
                self.assertTrue(torch.equal(pipeline(runs, x), ref))

    def test_steered_stream_reaches_next_layer_and_last_before_contract(self):
        """Spy on each layer's hc_post (the stream before the edit) and on
        the next layer's hc_attn_pre (the stream it receives)."""
        alpha = 2.0
        self.write(alpha=str(alpha))
        run = runner((45,), 0, dtype=torch.float64)
        bb = run.model.model
        pre, seen = {}, {}
        for i in range(45):
            mhc = bb.layers[i].layer_communicator.mhc

            def spy_post(*a, _i=i, _f=mhc.hc_post):
                out = _f(*a)
                pre[_i] = out  # the last call per layer is mlp_combine's
                return out

            def spy_in(x, _i=i, _f=mhc.hc_attn_pre):
                seen[_i] = x
                return _f(x)

            mhc.hc_post, mhc.hc_attn_pre = spy_post, spy_in
        self.install(run)
        x = torch.randn(4, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        y = bb(input_embeds=x)
        stack = bb._steer_stack.double()
        for i in STEERED:
            d = stack[i, 0]
            post = seen[i + 1] if i < 44 else None
            if i == 44:  # contracted after the edit, inside the layer
                post = pre[44] - alpha * (pre[44] @ d).unsqueeze(-1) * d
                self.assertTrue(torch.allclose(hc_contract(post, 4), y, atol=1e-12))
            # the next layer receives the edited stream: (1 - alpha) along d, unchanged across d
            self.assertTrue(torch.allclose(post @ d, (1 - alpha) * (pre[i] @ d), atol=1e-9), i)
            e = torch.randn(32, dtype=torch.float64, generator=torch.Generator().manual_seed(i))
            e = e - (e @ d) * d
            self.assertTrue(torch.allclose(post @ e, pre[i] @ e, atol=1e-9), i)
        # layer 0 is not steered: layer 1 receives layer 0's stream as it came out
        self.assertTrue(torch.equal(seen[1], pre[0]))

    def test_alpha_zero_is_bitwise_stock_at_16384(self):
        """bf16, width 16384 (4 x 4096), 45 layers: alpha 0 gives the stock
        bits; alpha 2 does not."""
        self.write(width=16384, alpha="0")
        x = torch.randn(3, 4096, generator=torch.Generator().manual_seed(7)).bfloat16()
        stock = runner((45,), 0, hidden=4096, dtype=torch.bfloat16).model.model(input_embeds=x)
        run = runner((45,), 0, hidden=4096, dtype=torch.bfloat16)
        rec = self.install(run)
        self.assertEqual((rec["alpha"], rec["width"]), (0.0, 16384))
        self.assertTrue(torch.equal(run.model.model(input_embeds=x), stock))
        os.environ["WEIGHTLESS_STEER_ALPHA"] = "2.0"
        run2 = runner((45,), 0, hidden=4096, dtype=torch.bfloat16)
        self.install(run2)
        self.assertFalse(torch.equal(run2.model.model(input_embeds=x), stock))

    def test_fired_check_catches_a_skipped_combine(self):
        """SGLang skips postprocess_layer (and so mlp_combine) when the
        communicator fuses the all-reduce into the next layer; mHC says no
        today. If that changed, the boot's first forward must fail."""
        self.write()
        run = runner((45,), 0)
        self.install(run)
        bb = run.model.model
        x = torch.randn(2, 8)
        bb(input_embeds=x)  # all 44 sites fire
        bb.layers[44].layer_communicator.fuse_with_next = True
        with self.assertRaisesRegex(RuntimeError, r"did not run in this forward \(layers \[44\]"):
            bb(input_embeds=x)

    def test_encoder_only_rank_installs_nothing(self):
        self.write()
        run = runner((45,), 0)
        run.model.model = None
        self.assertIsNone(self.install(run))
        self.assertIn("has no backbone on this rank", self.log)

    def test_manifest(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = runner((17, 14, 14), 2)
        self.install(run)
        m = json.load(open(os.path.join(self.tmp.name, f"weightless-manifest-{os.getpid()}.json")))
        self.assertEqual(m["row"], GLM)
        self.assertEqual(m["model_class"], GLM)
        self.assertEqual(m["site"], "mhc.mlp_combine")
        self.assertEqual(m["install"], "special:glm_mhc_combine")
        self.assertEqual((m["width"], m["hidden_size"], m["pp_rank"], m["pp_size"]), (32, 8, 2, 3))
        self.assertEqual(m["local_layer_ids"], list(range(31, 45)))
        self.assertEqual(m["full_attn_layers"], [31, 35, 39, 43])
        self.assertEqual(m["kernel"], "torch")
        self.assertEqual(m["model_types"], ["glm5_next", "glm5_next_text"])
        self.assertEqual(m["fired_at_capture"], [])

    def test_undo_takes_the_wrap_off(self):
        self.write()
        run = runner((45,), 0)
        self.install(run)
        self.assertEqual(self.wrapped(run), list(STEERED))
        for undo in run._weightless_steer_handles:
            undo()
        self.assertEqual(self.wrapped(run), [])

    def test_layer_map_diag_on_every_rank(self):
        """WEIGHTLESS_STEER_DIAG=1 at alpha 2: each rank's diag file lists
        exactly its steered layers, and the alpha-aware rule passes on
        every one of them, layer 44 (before the contract) included."""
        self.write(alpha="2.0")
        x = torch.randn(6, 8, generator=torch.Generator().manual_seed(11))
        # One process holds all three ranks here, so each rank gets its own diag folder.
        for part, expect in PARTITIONS.items():
            ddirs = []
            runs = []
            for r in range(3):
                ddir = os.path.join(self.tmp.name, f"d{r}-" + "-".join(map(str, part)))
                os.environ.update(WEIGHTLESS_STEER_DIAG="1", WEIGHTLESS_STEER_DIAG_DIR=ddir)
                run = runner(part, r)
                self.install(run)
                runs.append(run)
                ddirs.append(ddir)
            pipeline(runs, x)
            for rank, ddir in enumerate(ddirs):
                with self.subTest(partition=part, rank=rank):
                    f = json.load(open(os.path.join(ddir, f"weightless-diag-{os.getpid()}.json")))
                    self.assertEqual(sorted(int(k) for k in f["layers"]), expect[rank])
                    self.assertEqual(f["alpha"], 2.0)
                    for k, st in f["layers"].items():
                        # fp32 here: the edit is exact to ~1e-7 of |h.d|, and the
                        # signed ratio is 1 - alpha = -1
                        self.assertLess(st["sum_abs_err"], 1e-5 * st["sum_abs_pre"], k)
                        self.assertAlmostEqual(st["sum_signed_post"] / st["sum_abs_pre"], -1.0,
                                               places=5)
                        # the rule used on the GPU run never fails a correct layer
                        # (it may call a narrow 32-wide fake "not decisive")
                        for floor in ("norm", "elementwise"):
                            v = layer_verdict(st, f["alpha"], floor=floor)
                            self.assertNotIn("fail", (v["edit"], v["sign"], v["probe"]), (k, v))
                    last = str(expect[rank][-1])
                    self.assertEqual(f["next_layer"].get(last), None)  # no next direction here


class Refusals(GlmBase):
    def test_boot_refusals(self):
        self.write()
        cases = []

        def case(name, pattern, mutate):
            cases.append((name, pattern, mutate))

        case("mhc off", "mhc off", lambda run: setattr(run.model.model.config, "mhc", False))
        case("tp 2", "TP=2", lambda run: setattr(run, "tp_size", 2))
        case("two-batch overlap", "two-batch overlap",
             lambda run: setattr(run, "server_args",
                                 types.SimpleNamespace(enable_two_batch_overlap=True)))
        case("plain communicator", "not MHCLayerCommunicator",
             lambda run: setattr(run.model.model.layers[7].layer_communicator, "__class__",
                                 LayerCommunicator))
        case("no mlp_combine", "mlp_combine is missing",
             lambda run: setattr(run.model.model.layers[7].layer_communicator, "mhc",
                                 types.SimpleNamespace(hc_mult=4)))
        case("nextn layer", "NEXTN",
             lambda run: setattr(run.model.model.layers[44], "is_nextn", True))
        case("layer id", "says layer_id=8",
             lambda run: setattr(run.model.model.layers[7], "layer_id", 8))
        case("hc_mult", "hc_mult 2",
             lambda run: setattr(run.model.model.layers[7].layer_communicator.mhc, "hc_mult", 2))

        def wrap_twice(run):
            mhc = run.model.model.layers[9].layer_communicator.mhc
            mhc.mlp_combine = mhc.mlp_combine

        case("already wrapped", "already wrapped", wrap_twice)
        for name, pattern, mutate in cases:
            with self.subTest(name):
                run = runner((45,), 0)
                mutate(run)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self.install(run)

    def test_wrong_layer_class(self):
        self.write()
        run = runner((45,), 0, layer_cls=DeepseekV2DecoderLayer)
        with self.assertRaisesRegex(RuntimeError, "not one of"):
            self.install(run)

    def test_file_refusals(self):
        for name, kw, pattern in (
                ("qwen hint", dict(hint="qwen3_5"), "model_hint"),
                ("ffn hook point", {"glp.hook_point": "ffn_out_pre_residual",
                                    "glp.derived_at": "ffn_out_pre_residual"}, "glp.hook_point"),
                ("looped file", {"glp.structure": "per-execution-step"}, "per-execution-step"),
                ("layer 45", dict(layers=(1, 45)), "45")):
            with self.subTest(name):
                self.write(**kw)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self.install(runner((45,), 0))

    def test_width_guard(self):
        # a hidden-wide (unwidened) file on the GLM row
        self.write(width=8)
        with self.assertRaisesRegex(RuntimeError, r"width 8 != 32"):
            self.install(runner((45,), 0))
        # the widened file on a model whose stream is hidden-wide
        self.write(width=32)
        with self.assertRaisesRegex(RuntimeError, r"width 32 != 8"):
            self.install(runner((45,), 0, hc_mult=1, hidden=8))
        run = runner((45,), 0)
        run.model.model.config.mhc = False
        with self.assertRaisesRegex(RuntimeError, "mhc off"):
            self.install(run)

    def test_width_guard_at_real_width(self):
        """A 4096-wide file on the 16384 stream, and the 16384 file on a
        4096-wide stream (hc_mult 1), both fail the boot."""
        self.write(width=4096, layers=(1, 2, 44))
        with self.assertRaisesRegex(RuntimeError, r"width 4096 != 16384"):
            self.install(runner((45,), 0, hidden=4096))
        self.write(width=16384, layers=(1, 2, 44))
        with self.assertRaisesRegex(RuntimeError, r"width 16384 != 4096"):
            self.install(runner((45,), 0, hidden=4096, hc_mult=1))


@unittest.skipUnless(glpfiles.have(glpfiles.GLM44),
                     "set WEIGHTLESS_TEST_GLP_DIR to a folder that holds the GLM-5.3-Flash GLP-44 file")
class RealFileGLP44(GlmBase):
    """The published GLM-5.3-Flash GLP-44 file (layers 1-44, width 16384)
    through install_steering on a 45-layer fake at the real width."""

    def setUp(self):
        super().setUp()
        os.environ["WEIGHTLESS_STEER_PATH"] = glpfiles.GLM44

    def test_metadata(self):
        from weightless_steer.container import read_gguf_cvec
        meta, tensors = read_gguf_cvec(glpfiles.GLM44)
        self.assertEqual(meta["controlvector.model_hint"], "glm5_next")
        self.assertEqual(meta["glp.hook_point"], "residual_stream_post_layer")
        self.assertEqual(meta["glp.mode"], "project")
        self.assertEqual(float(meta["glp.alpha_default"]), 2.0)
        ids = sorted(int(k.split(".")[1]) for k in tensors if k.startswith("direction."))
        self.assertEqual(ids, list(STEERED))
        self.assertTrue(all(np.asarray(v[0] if isinstance(v, tuple) else v).size == 16384
                            for v in tensors.values()))
        self.assertLess(max(ids), 45)  # GLM-5.3-Flash has 45 decoder layers (0-44)

    def test_install_over_the_partitions(self):
        from weightless_steer.core import SteeringCore
        core = SteeringCore.from_env(hook="residual_stream_post_layer", num_layers=45,
                                     hidden_size=16384)
        self.assertEqual(sorted(core.dirs), list(STEERED))
        self.assertEqual(core.alpha, 2.0)  # the file's glp.alpha_default
        x = torch.randn(2, 4096, generator=torch.Generator().manual_seed(1))
        one = runner((45,), 0, hidden=4096)
        self.install(one)
        ref = one.model.model(input_embeds=x)
        for part, expect in PARTITIONS.items():
            runs = [runner(part, r, hidden=4096) for r in range(3)]
            for rank, run in enumerate(runs):
                with self.subTest(partition=part, rank=rank):
                    rec = self.install(run)
                    self.assertEqual(rec["local_layer_ids"], expect[rank])
                    self.assertEqual(self.wrapped(run), expect[rank])
                    self.assertEqual((rec["width"], rec["alpha"]), (16384, 2.0))
                    stack = run.model.model._steer_stack
                    self.assertEqual(tuple(stack.shape), (45, 1, 16384))
                    self.assertEqual(float(stack[0].abs().sum()), 0.0)
                    for i in STEERED:
                        self.assertTrue(torch.equal(stack[i, 0], core.dirs[i]))
                    if rank == 2:
                        self.assertIn(44, rec["local_layer_ids"])
            with self.subTest(partition=part, check="pipeline"):
                self.assertTrue(torch.equal(pipeline(runs, x), ref))
        print(f"  GLP-44: 44 layers 1..44 at width 16384, sha256 {rec['file_sha256'][:16]}")


def _glm_config():
    p = os.environ.get("WEIGHTLESS_TEST_GLM_CONFIG", "").strip()
    if p and os.path.isdir(p):
        p = os.path.join(p, "config.json")
    return p if p and os.path.isfile(p) else None


@unittest.skipUnless(_glm_config(), "set WEIGHTLESS_TEST_GLM_CONFIG to a GLM-5.3-Flash config.json")
class CheckpointConfig(unittest.TestCase):
    """The served checkpoint's geometry against the row and the GLP file."""

    def test_config(self):
        cfg = json.load(open(_glm_config()))
        self.assertIn(GLM, cfg["architectures"])
        self.assertEqual(cfg["model_type"], "glm5_next")
        t = types.SimpleNamespace(**cfg["text_config"])
        self.assertEqual(t.model_type, "glm5_next_text")
        self.assertEqual((t.hidden_size, t.hc_mult, t.mhc, t.num_hidden_layers), (4096, 4, True, 45))
        self.assertEqual(ARCH[GLM].width(t), 16384)
        full = t.linear_attn_config["full_attn_layers"]
        self.assertEqual(full, [i for i in range(45) if not is_kda_layer(i)])
        if glpfiles.have(glpfiles.GLM44):
            from weightless_steer.container import read_gguf_cvec
            meta, tensors = read_gguf_cvec(glpfiles.GLM44)
            ids = sorted(int(k.split(".")[1]) for k in tensors if k.startswith("direction."))
            self.assertLess(max(ids), t.num_hidden_layers)
            self.assertIn(meta["controlvector.model_hint"], {cfg["model_type"]} | ARCH[GLM].hint)


def _fixed(n, mult, scale):
    """n float64 values in [-scale/2, scale/2) from an integer hash: the same
    on every machine and torch version (no random generator)."""
    k = torch.arange(n, dtype=torch.int64)
    return ((k * mult + 12345) % 4294967291).double() / 4294967291.0 * scale - scale / 2


class GoldenGLM(unittest.TestCase):
    """The GLM handler's output bits, recorded: a fixed [T, 16384] bf16
    stream through glm_mhc_combine (archs/glm5next.py) and the site install builds
    for this row (CPU, torch path, r=None), at alpha 0, 1 and 2, with T = 1
    and T = 600 (two slices of the bounded torch path). Any change to the
    shared math (dot_fixed, steer_delta, steer_rows, the site, the handler)
    changes the digest. The recorded values were taken on the code the
    GLM-5.3-Flash GPU run used."""

    W, HIDDEN, HC = 16384, 4096, 4
    LAYER = 7
    INPUT_SHA256 = "6bf8a8b0ddeacf5521be9b8f3e36c4614f7b0562450d6b06767bdb35526a4125"
    OUTPUT_SHA256 = "a75fc9a0acc3dfb25e849187e83f1216c2f3bd9ee005fc4abe92657338a9dfd5"

    def stream(self, T):
        x = _fixed(T * self.W, 2654435761, 8.0).view(T, self.W)
        for j in (7, self.W // 3, self.W - 5):  # a few massive stream values
            x[:, j] *= 400.0
        return x.float().bfloat16()  # float64 -> float32 -> bf16, each step IEEE

    def direction(self):
        v = _fixed(self.W, 40503, 2.0)
        norm = math.sqrt(math.fsum(float(t) * float(t) for t in v.tolist()))
        return (v / norm).float()

    def run_handler(self, x, alpha):
        handler, site_name = SPECIAL["glm_mhc_combine"]
        self.assertEqual(site_name, "mhc.mlp_combine")
        cfg = types.SimpleNamespace(hidden_size=self.HIDDEN, hc_mult=self.HC, mhc=True,
                                    num_hidden_layers=45)
        owner = torch.nn.Module()
        stack = torch.zeros(45, 1, self.W)
        stack[self.LAYER, 0] = self.direction()
        owner.register_buffer("_steer_stack", stack, persistent=False)
        owner.register_buffer("_steer_alpha", torch.tensor(float(alpha)), persistent=False)
        owner.config = cfg
        mhc = MHCState(hc_mult=self.HC, hc_attn_pre=None, hc_ffn_pre=None,
                          hc_post=lambda h, r, a, b: h)
        layer = types.SimpleNamespace(layer_communicator=MHCLayerCommunicator(mhc, False, False),
                                      hidden_size=self.HIDDEN, layer_id=self.LAYER, is_nextn=False)
        site = wcore.make_site(owner, self.LAYER, kernel="torch", width=self.W)
        undo = handler(layer, site, layer_id=self.LAYER, backbone=owner, row=ARCH[GLM])
        try:
            return mhc.mlp_combine(x, None)
        finally:
            undo()

    def test_recorded_bits(self):
        h_in, h_out = hashlib.sha256(), hashlib.sha256()
        for T in (1, 600):
            x = self.stream(T)
            h_in.update(x.view(torch.int16).numpy().tobytes())
            for alpha in (0.0, 1.0, 2.0):
                out = self.run_handler(x, alpha)
                self.assertEqual((out.dtype, tuple(out.shape)), (torch.bfloat16, (T, self.W)))
                if alpha == 0.0:
                    self.assertTrue(torch.equal(out.view(torch.int16), x.view(torch.int16)))
                h_out.update(out.view(torch.int16).numpy().tobytes())
        print(f"  GLM golden: input sha256 {h_in.hexdigest()}, output sha256 {h_out.hexdigest()}")
        self.assertEqual(h_in.hexdigest(), self.INPUT_SHA256, "the fixed input changed")
        self.assertEqual(h_out.hexdigest(), self.OUTPUT_SHA256,
                         "the GLM handler's output bits changed")


# ---------------------------------------------------------------- SGLang's real classes

# The mHC kernels have torch paths; the TileLang ones need a GPU.
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "0")

try:
    from sglang.srt.layers import communicator as COMM
    from sglang.srt.layers import communicator_mhc as CM
    from sglang.srt.models import glm5_next as G
    from sglang.kernels.ops.layernorm.mhc import hc_contract as sglang_hc_contract
    HAVE_SGLANG = True
except Exception as e:  # no SGLang, or a build without the GLM model
    HAVE_SGLANG, WHY = False, f"{type(e).__name__}: {e}"

H, N, NL = 8, 4, 45  # STEERED and PARTITIONS as above


def _cfg():
    return types.SimpleNamespace(hidden_size=H, hc_mult=N, mhc=True, rms_norm_eps=1e-6,
                                 hc_eps=1e-6, hc_sinkhorn_iters=20, num_hidden_layers=NL,
                                 model_type="glm5_next_text")


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H), requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, x, residual=None):
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight).to(x.dtype)
        return y if residual is None else (y, residual)


class _Attn(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(0.5, 1.5, H) * (1 + 0.02 * i), requires_grad=False)

    def forward(self, positions=None, hidden_states=None, forward_batch=None, zero_allocator=None,
                layer_scatter_modes=None, prev_topk_indices=None):
        return torch.tanh(hidden_states * self.w)


class _MLP(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(-1, 1, H) * (1 + 0.01 * i), requires_grad=False)

    def forward(self, x, forward_batch=None, gemm_output_zero_allocator=None):
        return x * self.w


def real_layer(i, cfg, dtype):
    """A real Glm5NextDecoderLayer with a real MHCLayerCommunicator."""
    g = torch.Generator().manual_seed(100 + i)
    L = G.Glm5NextDecoderLayer.__new__(G.Glm5NextDecoderLayer)
    nn.Module.__init__(L)
    L.config, L.hidden_size, L.layer_id, L.is_nextn = cfg, H, i, False
    L.is_linear_attn = i % 4 != 3
    mix = (2 + N) * N
    for p in ("attn", "ffn"):
        setattr(L, f"hc_{p}_base", nn.Parameter(torch.randn(mix, generator=g) * 0.1, requires_grad=False))
        setattr(L, f"hc_{p}_scale", nn.Parameter(torch.ones(3), requires_grad=False))
        setattr(L, f"hc_{p}_fn", nn.Parameter(torch.randn(mix, N * H, generator=g) * 0.05,
                                             requires_grad=False))
    L.self_attn, L.mlp, L.layer_scatter_modes = _Attn(i).to(dtype), _MLP(i).to(dtype), None
    L.input_layernorm, L.post_attention_layernorm = _Norm().to(dtype), _Norm().to(dtype)
    C = CM.MHCLayerCommunicator.__new__(CM.MHCLayerCommunicator)
    C.is_first_layer = i == 0
    C.mhc = CM.MHCState(hc_mult=N, hc_attn_pre=L.hc_attn_pre, hc_ffn_pre=L.hc_ffn_pre,
                        hc_post=L.hc_post, hc_ffn_post_pre=None)
    C.input_layernorm, C.post_attention_layernorm = L.input_layernorm, L.post_attention_layernorm
    C.allow_reduce_scatter, C.is_last_layer, C.qkv_latent_func = False, i == NL - 1, None
    C._context = types.SimpleNamespace()
    C._communicate_simple_fn = COMM.CommunicateSimpleFn._trivial
    C._communicate_with_all_reduce_and_layer_norm_fn = CM.MHCCommunicateWithAllReduceAndLayerNormFn._simple
    C._communicate_summable_tensor_pair_fn = CM.MHCCommunicateSummableTensorPairFn._trivial
    # attributes that newer trees read in the FFN exit (not set by __new__)
    C._postprocess_scatters_to_local_tokens = False
    C._sp_variant = None
    L.layer_communicator = C
    return L  # the hc_* parameters stay fp32, as SGLang makes them


def real_runner(partition=(NL,), rank=0, dtype=torch.float32):
    cfg = _cfg()
    start = sum(partition[:rank])
    end = start + partition[rank]
    M = G.Glm5NextModel.__new__(G.Glm5NextModel)
    nn.Module.__init__(M)
    M.config, M.first_k_dense_replace = cfg, 3
    M.pp_group = types.SimpleNamespace(is_first_rank=start == 0, is_last_rank=end == NL)
    M.embed_tokens = nn.Identity()
    M.layers = nn.ModuleList(real_layer(i, cfg, dtype) if start <= i < end else nn.Identity()
                             for i in range(NL))
    M.start_layer, M.end_layer = start, end
    M.norm = _Norm().to(dtype) if end == NL else nn.Identity()
    M.gemm_output_zero_allocator_size = 0
    M.layers_to_capture, M.dflash_capture, M.enable_a2a_moe = [], False, False
    W = G.Glm5NextForConditionalGeneration.__new__(G.Glm5NextForConditionalGeneration)
    nn.Module.__init__(W)
    W.model = M
    return types.SimpleNamespace(
        model=W, is_draft_worker=False, tp_rank=0, tp_size=1, pp_rank=rank,
        pp_size=len(partition),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="glm5_next"),
                                           dtype=dtype))


FB = types.SimpleNamespace(can_run_tbo=False,
                           forward_mode=types.SimpleNamespace(is_idle=lambda: False))


def run_model(run, x=None, proxy=None):
    return run.model.model(None, None, FB, input_embeds=x, pp_proxy_tensors=proxy)


def real_pipeline(runs, x):
    out = None
    for n, run in enumerate(runs):
        out = run_model(run, x=x) if n == 0 else run_model(run, proxy=out)
    return out


@unittest.skipUnless(HAVE_SGLANG, "SGLang's GLM model is not importable here")
class RealLayers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        self.write()

    def write(self, alpha="2.0"):
        m = good_meta(layers=STEERED, alpha=alpha)
        m["controlvector.model_hint"] = "glm5_next"
        rng = np.random.default_rng(0)
        t = {}
        for i in STEERED:
            v = rng.standard_normal(N * H)
            t[f"direction.{i}"] = ((v / np.linalg.norm(v)).astype(np.float32), 0)
        path = os.path.join(self.tmp.name, f"v{alpha}.gguf")
        write_gguf(path, m, t)
        os.environ["WEIGHTLESS_STEER_PATH"] = path

    def install(self, run):
        with redirect_stderr(io.StringIO()):
            return install_steering(run, source="env")

    def test_real_classes_install(self):
        run = real_runner()
        rec = self.install(run)
        self.assertEqual(rec["row"], "Glm5NextForConditionalGeneration")
        self.assertEqual(rec["local_layer_ids"], list(STEERED))
        self.assertEqual(rec["full_attn_layers"], [i for i in STEERED if i % 4 == 3])
        self.assertEqual(rec["width"], N * H)
        bb = run.model.model
        wrapped = [i for i in range(NL) if "mlp_combine" in vars(bb.layers[i].layer_communicator.mhc)]
        self.assertEqual(wrapped, list(STEERED))

    def test_steered_stream_reaches_next_layer_and_last_before_contract(self):
        alpha = 2.0
        run = real_runner(dtype=torch.float64)
        bb = run.model.model
        pre, seen = {}, {}
        for i in range(NL):
            mhc = bb.layers[i].layer_communicator.mhc

            def spy_post(*a, _i=i, _f=mhc.hc_post):
                out = _f(*a)
                pre[_i] = out  # attn_to_mlp calls it first, mlp_combine last
                return out

            def spy_in(x, *a, _i=i, _f=mhc.hc_attn_pre):
                seen[_i] = x
                return _f(x, *a)

            mhc.hc_post, mhc.hc_attn_pre = spy_post, spy_in
        self.install(run)
        x = torch.randn(4, H, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        y = run_model(run, x=x)
        stack = bb._steer_stack.double()
        for i in STEERED:
            d = stack[i, 0]
            if i < NL - 1:
                post = seen[i + 1]
            else:
                post = pre[i] - alpha * (pre[i] @ d).unsqueeze(-1) * d
                self.assertTrue(torch.allclose(bb.norm(sglang_hc_contract(post, N)), y, atol=1e-12))
            self.assertTrue(torch.allclose(post @ d, (1 - alpha) * (pre[i] @ d), atol=1e-9), i)
            e = torch.randn(N * H, dtype=torch.float64, generator=torch.Generator().manual_seed(i))
            e = e - (e @ d) * d
            self.assertTrue(torch.allclose(post @ e, pre[i] @ e, atol=1e-9), i)
        self.assertTrue(torch.equal(seen[1], pre[0]))  # layer 0 is not steered

    def test_pipeline_ranks_equal_one_rank(self):
        x = torch.randn(3, H, generator=torch.Generator().manual_seed(2))
        one = real_runner()
        self.install(one)
        ref = run_model(one, x=x)
        stock = run_model(real_runner(), x=x)
        self.assertFalse(torch.allclose(ref, stock))
        for part, expect in PARTITIONS.items():
            with self.subTest(partition=part):
                runs = [real_runner(part, r) for r in range(3)]
                for r, run in enumerate(runs):
                    self.assertEqual(self.install(run)["local_layer_ids"], expect[r])
                mid = run_model(runs[0], x=x)
                self.assertEqual(sorted(mid.tensors), ["hidden_states"])  # the stream only
                self.assertEqual(tuple(mid["hidden_states"].shape), (3, N * H))
                self.assertTrue(torch.equal(real_pipeline(runs, x), ref))

    def test_alpha_zero_is_bitwise_stock(self):
        self.write(alpha="0")
        x = torch.randn(5, H, generator=torch.Generator().manual_seed(9)).bfloat16()
        stock = run_model(real_runner(dtype=torch.bfloat16), x=x)
        run = real_runner(dtype=torch.bfloat16)
        self.assertEqual(self.install(run)["alpha"], 0.0)
        self.assertTrue(torch.equal(run_model(run, x=x), stock))

    def test_empty_batch(self):
        """A forward with no tokens goes through every steered site."""
        run = real_runner()
        self.install(run)
        y = run_model(run, x=torch.empty(0, H))
        self.assertEqual(tuple(y.shape), (0, H))

    def test_fired_check_when_the_last_layer_skips_its_combine(self):
        run = real_runner()
        self.install(run)
        x = torch.randn(2, H)
        run_model(run, x=x)
        comm = run.model.model.layers[NL - 1].layer_communicator
        if hasattr(comm, "ffn_exit"):  # newer trees: the FFN exit leaves the sum for later
            from sglang.srt.layers.communicator import FfnCompletion
            comm._select_ffn_completion = lambda fb: FfnCompletion(
                defer_moe_finalize=False, fuse_mlp_allreduce=False, mlp_reduce_scatter=False,
                leave=lambda h: h)
        else:  # older trees: the all-reduce fused into the next layer skips postprocess
            comm.should_fuse_mlp_allreduce_with_next_layer = lambda fb: True
        with self.assertRaisesRegex(RuntimeError, r"did not run in this forward \(layers \[44\]"):
            run_model(run, x=x)


# ---------------------------------------------------------------- structure


MODEL_FILE = "srt/models/glm5_next.py"


@unittest.skipUnless(_trees(MODEL_FILE), "no SGLang package with models/glm5_next.py found")
class Structure(unittest.TestCase):
    """Pins, per SGLang tree, the code paths the GLM row depends on. A change
    in any of them fails here, before a GPU is touched."""

    def trees(self):
        return [(t, _parse(t, "srt/models/glm5_next.py"), _parse(t, "srt/layers/communicator_mhc.py"))
                for t in _trees(MODEL_FILE)]

    def test_model_loop_calls_each_layer_and_unpacks_three(self):
        for t, g, _ in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(g, "Glm5NextModel"), "forward")
                loops = [n for n in ast.walk(fwd) if isinstance(n, ast.For)]
                self.assertEqual(len(loops), 1)
                loop = loops[0]
                self.assertEqual(_src(loop.iter), "range(normal_start_layer, normal_end_layer)")
                assigns = [n for n in ast.walk(loop) if isinstance(n, ast.Assign)
                           and isinstance(n.value, ast.Call) and _src(n.value.func) == "layer"]
                self.assertEqual(len(assigns), 1)
                self.assertEqual(_src(assigns[0].targets[0]), "(hidden_states, residual, topk_indices)")
                self.assertIn("layer = self.layers[i]", _src(loop))
                # the only other way through the layers is two-batch overlap
                tbo = _calls(fwd, "model_forward_maybe_tbo")
                self.assertEqual(len(tbo), 1)
                self.assertIn("forward_batch.can_run_tbo", _src(fwd))
                self.assertEqual(len(_calls(fwd, "layer")), 1)

    def test_pipeline_carries_the_widened_stream_only(self):
        for t, g, _ in self.trees():
            with self.subTest(tree=t):
                s = _src(_fn(_cls(g, "Glm5NextModel"), "forward"))
                self.assertIn("residual = None if self.config.mhc else pp_proxy_tensors['residual']", s)
                self.assertIn("return PPProxyTensors({'hidden_states': hidden_states})", s)

    def test_backbone_and_layer_classes(self):
        for t, g, _ in self.trees():
            with self.subTest(tree=t):
                init = _src(_fn(_cls(g, "Glm5NextForConditionalGeneration"), "__init__"))
                self.assertIn("self.model = None", init)
                self.assertIn("self.model = Glm5NextModel(", init)
                model_init = _src(_fn(_cls(g, "Glm5NextModel"), "__init__"))
                self.assertIn("make_layers(", model_init)
                self.assertIn("Glm5NextDecoderLayer(", model_init)
                layer_init = _src(_fn(_cls(g, "Glm5NextDecoderLayer"), "__init__"))
                self.assertIn("self.layer_id = layer_id", layer_init)
                self.assertIn("self.is_linear_attn = config.is_kda_layer(layer_id)", layer_init)
                if "is_last_layer=" in layer_init:  # older trees: passed to the communicator
                    self.assertIn("is_last_layer=is_nextn or self.layer_id == "
                                  "self.config.num_hidden_layers - 1", layer_init)
                else:  # newer trees: the scatter modes carry it
                    self.assertIn("num_layers=1 if is_nextn else config.num_hidden_layers", layer_init)
                    comm = _parse(t, "srt/layers/communicator.py")
                    modes = _src(_cls(comm, "LayerScatterModes"))
                    self.assertIn("is_last_layer=context.layer_id == context.num_layers - 1", modes)
                    self.assertIn("self.is_last_layer = layer_scatter_modes.is_last_layer",
                                  _src(_fn(_cls(comm, "LayerCommunicator"), "__init__")))
                self.assertIn("self.layer_communicator = MHCLayerCommunicator(", layer_init)
                self.assertIn("hc_post=self.hc_post", layer_init)

    def test_layer_forward_returns_three_and_reaches_postprocess(self):
        for t, g, _ in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(g, "Glm5NextDecoderLayer"), "forward")
                rets = [n for n in ast.walk(fwd) if isinstance(n, ast.Return)]
                self.assertEqual([_src(r.value) for r in rets], ["(hidden_states, residual, topk_indices)"])
                s = _src(fwd)
                if "ffn_exit" in s:
                    # newer trees: FfnExit.finish runs postprocess_layer unless it
                    # leaves the sum to the next layer or hands off a deferred MoE
                    self.assertIn("hidden_states, residual = ffn_exit.finish(hidden_states, residual)", s)
                    comm = _parse(t, "srt/layers/communicator.py")
                    fin = _src(_fn(_cls(comm, "FfnExit"), "finish"))
                    self.assertIn("if not isinstance(hidden_states, torch.Tensor):", fin)
                    self.assertIn("if self._leave is not None:", fin)
                    self.assertIn("return self.communicator.postprocess_layer(", fin)
                else:
                    # older trees: postprocess_layer is skipped only when the all-reduce is
                    # fused into the next layer, which mHC turns off
                    self.assertIn("if not should_allreduce_fusion:", s)
                    self.assertEqual(len(_calls(fwd, "postprocess_layer")), 1)
                    self.assertIn("self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(",
                                  s)

    def test_mhc_state_and_combine_order(self):
        for t, _, c in self.trees():
            with self.subTest(tree=t):
                st = _cls(c, "MHCState")
                self.assertEqual([_src(d) for d in st.decorator_list], ["dataclass"])  # no slots
                self.assertEqual(_src(_fn(st, "mlp_combine")).splitlines()[-1].strip(),
                                 "return self.hc_post(hidden_states, residual, self.h_res, self.h_post)")
                comm = _cls(c, "MHCLayerCommunicator")
                post = _fn(comm, "postprocess_layer")
                call = _calls(post, "_communicate_summable_tensor_pair_fn")
                self.assertEqual(len(call), 1)
                kws = {k.arg: _src(k.value) for k in call[0].keywords}
                self.assertEqual(kws.get("mhc"), "self.mhc")
                self.assertEqual(kws.get("is_last_layer"), "self.is_last_layer")
                self.assertEqual(_src(_fn(comm, "should_fuse_mlp_allreduce_with_next_layer").body[-1]),
                                 "return False")
                names = [n.name for n in comm.body if isinstance(n, ast.FunctionDef)]
                if "should_defer_ffn_reduction" in names:
                    self.assertEqual(_src(_fn(comm, "should_defer_ffn_reduction").body[-1]), "return False")
                pair = _cls(c, "MHCCommunicateSummableTensorPairFn")
                combining = set()
                for fn in pair.body:
                    if not isinstance(fn, ast.FunctionDef):
                        continue
                    mc = _calls(fn, "mlp_combine")
                    hc = _calls(fn, "hc_contract")
                    if mc:
                        combining.add(fn.name)
                    for k in hc:  # every contract comes after the combine
                        self.assertTrue(mc and mc[0].lineno < k.lineno, (fn.name, k.lineno))
                self.assertEqual(combining, {"_trivial", "_scatter_hidden_states", "_gather", "_scatter"})
                # nothing else in the model calls hc_post or mlp_combine around the communicator
                self.assertEqual(len(_calls(c, "mlp_combine")), 4)

    def test_no_fusion_across_the_layer_boundary(self):
        """hc_ffn_post_pre fuses attention hc_post with the FFN hc_pre INSIDE a
        layer; nothing fuses a layer's mlp_combine into the next layer."""
        for t, g, c in self.trees():
            with self.subTest(tree=t):
                st = _cls(c, "MHCState")
                users = [fn.name for fn in st.body if isinstance(fn, ast.FunctionDef)
                         and "hc_ffn_post_pre" in _src(fn)]
                self.assertEqual(users, ["attn_to_mlp"])
                prep = _src(_fn(_cls(c, "MHCLayerCommunicator"), "prepare_attn"))
                self.assertIn("self.mhc.attn_split(", prep)
                self.assertNotIn("post", prep.replace("postprocess", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
