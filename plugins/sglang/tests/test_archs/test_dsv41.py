"""The DeepSeek-V4.1 row (``DeepseekV4ForCausalLM`` with model_type
deepseek_v41, row ``DeepseekV4ForCausalLM:deepseek_v41``), CPU only.

The site is the HcState the FFN post of each steered layer returns (the
post-layer hyper-connection streams [T, 4, hidden]), edited before any
reader sees it: the next layer's attention read, the Engram seam's
materialize, and the FinalRead after the rank's last layer.

- The edit (``archs/dsv41.py``): the mean-coefficient form against a float64
  reference, the mean projected and the copy differences kept, alpha 0
  bit-identical, the slices equal to one pass, and the "each" form. The
  fused kernel (fused.py ``_glp_hc_mean_kernel``) runs under Triton's CPU
  interpreter in a child process: bit-identical to the torch path on fp32
  streams; on bf16 streams the interpreter's own store rounding differs from
  the GPU's round-to-nearest-even (the existing row kernel shows the same),
  so there it is checked to one bf16 step. The bitwise bf16 check is the
  start-up self-check on the GPU (``self_check_hc_mean``).
- Fakes shaped like SGLang's classes, call for call with
  models/deepseek_v4.py ``_forward_layers_hc_pre_from_prev`` and
  deepseek_v4_boundary.py: decoder layers named ``DeepseekV4CudaDecoderLayer``
  built from two ``StageBoundary`` objects whose declared updates are
  ``HcUpdate`` objects with ``post(output, state, *, next=None)``; the post
  rebuilds the streams from coefficients computed from the streams, and with
  ``next`` also hands over the next input and statistics precomputed from
  them; the model loop materializes the pending post before an Engram layer
  and calls ``engram(state.residual)``, runs the layers, and reads the last
  post with a FinalRead; the pipeline hand-over carries the streams and
  ``pre``. The row on them: install, the edit at every steered layer, the
  pre-Engram order, the dropped shortcut, alpha 0, pipeline ranks, the fired
  check, the refusals, the diagnostics, the published files.
- ``RealSGLang``: SGLang's own HcUpdate.post, HcState, ReadState,
  materialize, read_output with FinalRead and ResidualStream (the installed
  package), on CPU, with the two TileLang kernels they call given SGLang's
  own torch reference (test/kernels/mhc.py) and the one-line torch combine.
- ``Structure``: a parse of models/deepseek_v4.py, deepseek_v4_boundary.py,
  deepseek_v4_mhc.py and configs/deepseek_v41.py in every SGLang tree (the
  installed one and WEIGHTLESS_TEST_SGLANG_TREES) that pins what the row
  relies on.
"""
import ast
import io
import json
import os
import subprocess
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

from test_archs import _trees, _parse, _cls, _fn, _src, _flat, unit_dirs  # noqa: E402

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.archs import ARCH, EDIT, PREFLIGHT, REFUSED, SPECIAL, VARIANTS  # noqa: E402
from weightless_sglang.archs import dsv41 as A  # noqa: E402
from weightless_sglang.install import install_steering, resolve_row  # noqa: E402

ROW = A.ROW_NAME
RES = "residual_stream_post_layer"
FFN = "ffn_out_pre_residual"
GLP1017 = glpfiles.find_glp("glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf")
GLP0909 = glpfiles.find_glp("glp.deepseek-v41-flash-GLP-39-L1-39-a0.5.gguf")
DSV4_FFN = glpfiles.find_glp("DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4.gguf")


# ---------------------------------------------------------------- the edit


def _streams(T=7, S=4, W=64, seed=0, dtype=torch.bfloat16, big=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, S, W, generator=g) * 4.0
    if big:
        x[:, :, :: max(1, W // 3)] *= 400.0  # a few very large values, as mHC streams have
    return x.to(dtype)


def _unit(W, seed=1):
    d = torch.randn(W, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    return (d / d.norm()).float()


def bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


class Edit(unittest.TestCase):
    def test_mean_form_against_float64(self):
        """x'k = x_k - alpha (m.d) d for every copy k, m the mean of the copies:
        the mean of the result is the projected mean, and the differences
        between copies are untouched (to the bf16 rounding of the result)."""
        x, d = _streams(), _unit(64)
        for a in (0.0, 0.5, 0.75, 1.0, 2.0):
            with self.subTest(alpha=a):
                al = torch.tensor(a)
                got = A.hc_mean_rows(x, d, al).double()
                xd, dd = x.double(), d.double()
                m = xd.mean(1)
                want = xd - a * (m @ dd)[:, None, None] * dd
                ulp = 2.0 ** -8 * xd.abs().amax().item()  # one bf16 step at the largest value
                self.assertLess((got - want).abs().max().item(), ulp)
                m2 = got.mean(1)
                bound = 2.0 ** -8 * float((got.abs().mean(1) @ dd.abs()).max())
                self.assertLess(((m2 @ dd) - (1 - a) * (m @ dd)).abs().max().item(), bound + 1e-9)
                diff_in = xd[:, 1:] - xd[:, :1]
                diff_out = got[:, 1:] - got[:, :1]
                self.assertLess((diff_out - diff_in).abs().max().item(), 2 * ulp)

    def test_alpha_zero_is_bit_identical(self):
        x, d = _streams(), _unit(64)
        self.assertTrue(torch.equal(bits(A.hc_mean_rows(x, d, torch.tensor(0.0))), bits(x)))

    def test_slices_equal_one_pass(self):
        x, d, al = _streams(T=37), _unit(64), torch.tensor(0.75)
        one = A.hc_mean_delta(x, d, al)
        with mock.patch.object(A, "TEMP_BUDGET_BYTES", 64 * (10 * 4 + 28) * 4):  # 4 tokens a slice
            self.assertEqual(A.hc_chunk_tokens(4, 64), 4)
            sliced = A.hc_mean_rows(x, d, al)
        self.assertTrue(torch.equal(bits(one), bits(sliced)))
        self.assertEqual(A.hc_chunk_tokens(4, 5120), 512)

    def test_sum_order_is_fixed(self):
        """The copies are added in index order in fp32: the same as an explicit
        ((x0 + x1) + x2) + x3."""
        x = _streams().float()
        want = ((x[:, 0] + x[:, 1]) + x[:, 2]) + x[:, 3]
        self.assertTrue(torch.equal(A.copies_sum(x), want))

    def test_each_form_projects_every_copy(self):
        x, d, al = _streams(), _unit(64), torch.tensor(0.75)
        e = A.HcEdit("each")
        got = e(x, None, d, al, "torch")
        self.assertTrue(torch.equal(bits(got), bits(wcore.steer_rows(x, None, d, al))))
        # its mean is the projected mean too
        m, m2, dd = x.double().mean(1), got.double().mean(1), d.double()
        bound = 2.0 ** -8 * float((got.double().abs().mean(1) @ dd.abs()).max())
        self.assertLess(((m2 @ dd) - 0.25 * (m @ dd)).abs().max().item(), bound + 1e-9)

    def test_modes_and_their_labels(self):
        with mock.patch.dict(os.environ, {A.COPIES_ENV: ""}):
            self.assertEqual(A.make_edit().mode, "mean")
        with mock.patch.dict(os.environ, {A.COPIES_ENV: "EACH"}):
            self.assertEqual(A.make_edit().mode, "each")
        with mock.patch.dict(os.environ, {A.COPIES_ENV: "sum"}):
            with self.assertRaisesRegex(RuntimeError, "WEIGHTLESS_STEER_HC_COPIES='sum' is not one of"):
                A.make_edit()
        self.assertIn("dsv41_hc_post", EDIT)
        self.assertIs(EDIT["dsv41_hc_post"], A.make_edit)

    def test_edit_refuses_a_residual(self):
        x, d = _streams(), _unit(64)
        with self.assertRaisesRegex(RuntimeError, "has no residual"):
            A.HcEdit("mean")(x, x, d, torch.tensor(1.0), "torch")

    def test_diag_view_is_the_mean_stream(self):
        x, d = _streams(), _unit(64)
        e = A.HcEdit("mean")
        xn = e(x, None, d, torch.tensor(1.0), "torch")
        a, b = e.diag_view(x, xn)
        self.assertEqual(tuple(a.shape), (7, 64))
        self.assertTrue(torch.equal(a, A.copies_sum(x.float()) * 0.25))
        self.assertTrue(torch.equal(b, A.copies_sum(xn.float()) * 0.25))
        x2, xn2 = A.HcEdit("each").diag_view(x, xn)
        self.assertIs(x2, x)


_INTERP = r'''
import sys, json
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import torch
from weightless_sglang import fused, core
from weightless_sglang.archs import dsv41
out = {}
T, S = 9, 4
for W in (5120, 72):
    g = torch.Generator().manual_seed(W)
    x = torch.randn(T, S, W, generator=g) * 4.0
    x[:, :, ::max(1, W // 3)] *= 400.0
    x[0].zero_()
    d = torch.randn(W, dtype=torch.float64, generator=g); d = (d / d.norm()).float()
    for a in (0.0, 0.75, 2.0):
        al = torch.tensor(a)
        for name, xs in (("fp32", x), ("bf16", x.bfloat16())):
            want = dsv41.hc_mean_rows(xs, d, al)
            o = torch.empty_like(xs)
            fused._glp_hc_mean_kernel[(T,)](xs, d, al, o, W, xs.stride(0), xs.stride(1), o.stride(0),
                                            o.stride(1), core.fixed_point_bits(W), 1.0 / S, S=S,
                                            BLOCK=1 << (W - 1).bit_length())
            same = bool(torch.equal(o.view(-1).view(torch.uint8), want.view(-1).view(torch.uint8)))
            step = float(((o.float() - want.float()).abs() / want.float().abs().clamp_min(1e-30)).max())
            out[f"{W}/{a}/{name}"] = [same, step]
print(json.dumps(out))
'''


class FusedUnderTheInterpreter(unittest.TestCase):
    """The fused kernel's steps on CPU, through Triton's interpreter."""

    def test_kernel_matches_the_torch_path(self):
        try:
            import triton  # noqa: F401
        except Exception as e:
            self.skipTest(f"no Triton here ({type(e).__name__})")
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([sys.executable, "-c", _INTERP, here,
                            os.path.join(os.path.dirname(here), "vllm")],
                           capture_output=True, text=True, env=env, timeout=600)
        if p.returncode != 0:
            self.fail(p.stderr[-2000:])
        res = json.loads(p.stdout.strip().splitlines()[-1])
        for key, (same, step) in sorted(res.items()):
            with self.subTest(key):
                if key.endswith("fp32") or key.split("/")[1] == "0.0":
                    self.assertTrue(same, f"{key}: max relative step {step}")
                else:  # the interpreter's bf16 store is not round-to-nearest-even
                    self.assertLessEqual(step, 2.0 ** -7)

    def test_pick_kernel_runs_the_rows_own_check(self):
        """install.pick_kernel calls the edit's own start-up check (not the row
        kernel's) and names it in the reason."""
        from weightless_sglang import fused
        from weightless_sglang.install import pick_kernel
        e = A.HcEdit("mean")
        seen = []

        def check(device, dtype, width, ok=True):
            seen.append((device, dtype, width))
            return ok
        with mock.patch.object(fused, "available", lambda: True), \
                mock.patch.object(fused, "self_check", side_effect=AssertionError("row kernel")):
            k, why = pick_kernel("auto", "cuda:0", torch.bfloat16, 5120, with_residual=False,
                                 check=check, form=e.form)
            self.assertEqual(k, "triton")
            self.assertEqual(why, "auto: fused Triton kernel, start-up self-check passed "
                                  "(hc copies: mean coefficient, width 5120)")
            self.assertEqual(seen, [("cuda:0", torch.bfloat16, 5120)])
            k, why = pick_kernel("auto", "cuda:0", torch.bfloat16, 5120, with_residual=False,
                                 check=lambda *a, **kw: False, form=e.form)
            self.assertEqual(k, "torch")
            self.assertIn("did not give the same bits", why)
            with self.assertRaisesRegex(RuntimeError, "WEIGHTLESS_STEER_KERNEL=triton but"):
                pick_kernel("triton", "cuda:0", torch.bfloat16, 5120, with_residual=False,
                            check=lambda *a, **kw: False, form=e.form)

    def test_self_check_refuses_rows_wider_than_a_block(self):
        from weightless_sglang import fused
        self.assertFalse(fused.self_check_hc_mean("cpu", width=fused.MAX_BLOCK + 1))


# ---------------------------------------------------------------- fakes shaped like SGLang's classes


S_COPIES = 4


def _vec(g, n, lo, hi):
    return lo + (hi - lo) * torch.rand(n, generator=g)


class HcState:
    """deepseek_v4_mhc.HcState: the streams, the ``pre`` the next combine
    uses, and the shortcut a fused post may leave (``input``, ``stats``).
    ``with_residual`` drops the shortcut and keeps ``pre``."""

    def __init__(self, streams, pre=None, input=None, stats=None):
        self.streams, self.pre, self.input, self.stats = streams, pre, input, stats

    @property
    def residual(self):
        assert isinstance(self.streams, torch.Tensor)
        return self.streams

    def with_residual(self, residual):
        return HcState(residual, self.pre)


class ReadState:
    def __init__(self, streams):
        self.streams = streams


def combine(R, pre):
    """mhc.combine: the sublayer input, sum_k pre[k] R[k] (the first copy
    when there is no pre yet)."""
    if pre is None:
        return R[:, 0, :]
    return (pre.float().unsqueeze(-1) * R.float()).sum(1).to(R.dtype)


class HcSubLayer(nn.Module):
    """The coefficients of one hyper-connection, computed from the streams
    it reads (mix_stats): pre, post and comb, so a steered stream changes
    every later coefficient."""

    def __init__(self, W, S, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("fn", 0.05 * torch.randn(S * W, (2 + S) * S, generator=g),
                             persistent=False)

    def triplet(self, R):
        T, S, W = R.shape
        mix = torch.tanh(R.float().reshape(T, S * W) @ self.fn.float() / (S * W) ** 0.5)
        pre = torch.sigmoid(mix[:, :S]) + 1e-6
        post = 2.0 * torch.sigmoid(mix[:, S:2 * S])
        comb = torch.softmax(mix[:, 2 * S:].reshape(T, S, S), -1)
        return pre, post, comb


def post_formula(y, R, post, comb):
    """mhc.post (its torch form): R'[i] = post[i] y + sum_j comb[j][i] R[j]."""
    return (post.unsqueeze(-1) * y.float().unsqueeze(1)
            + (comb.unsqueeze(-1) * R.float().unsqueeze(2)).sum(dim=1)).to(R.dtype)


class HcUpdate:
    """deepseek_v4_boundary.HcUpdate: ``post(output, state, *, next=None)``
    runs the sublayer's hyper-connection post and returns an HcState; with
    ``next`` it also precomputes the next input and statistics from the new
    streams (the fused kernel's shortcut). ``update`` calls ``self.post``."""

    def __init__(self, layer, attention):
        self.layer, self.attention = layer, attention
        self.calls = 0
        self.last = None  # spy: the HcState the post built (before any wrapper)

    def post(self, output, state, *, next=None):
        self.calls += 1
        hc = self.layer.attn_hc if self.attention else self.layer.ffn_hc
        pre, post, comb = hc.triplet(state.streams)
        R = post_formula(output, state.streams, post, comb)
        out = HcState(R, pre)
        if next is not None:
            out = HcState(R, pre, input=combine(R, pre), stats=next.triplet(R))
        self.last = out
        return out

    def update(self, output, state):
        return self.post(output, state)


class StageBoundary:
    def __init__(self, update):
        self.declaration = types.SimpleNamespace(update=update)


class Stream:
    """forward_batch.residual_stream: a written HcState, or a pending
    (update, output, read state) the next reader runs."""

    def __init__(self, written=None):
        self.written, self.pending = written, None


def take(fb, next=None):
    """The next stage's read of a pending post (HcRead.update_and_read): run
    the producer's post, write the stream."""
    st = fb.residual_stream
    if st.pending is not None:
        upd, y, read = st.pending
        st.pending = None
        st.written = upd.post(y, read, next=next)
    return st.written


def materialize(fb):
    """hc_boundary.materialize: run the pending post (no fused next input)."""
    return take(fb, next=None)


def final_read(fb):
    """FinalRead.update_and_read: ``update.update(value, residual)``."""
    st = fb.residual_stream
    if st.pending is not None:
        upd, y, read = st.pending
        st.pending = None
        st.written = upd.update(y, read)
    return st.written


class _DecoderLayer(nn.Module):
    """DeepseekV4CudaDecoderLayer.forward through the stream: the attention
    reads the previous layer's pending post (with the fused next input, as a
    layer without Engram is given), runs, and its own post (with the FFN
    read's shortcut); the FFN reads, runs and leaves its post pending."""

    def __init__(self, cfg, layer_id, engram=False):
        super().__init__()
        self.config = cfg
        self.layer_id = layer_id
        self.hidden_size = W = cfg.hidden_size
        self.hc_mult = S = cfg.hc_mult
        self.boundary_pre_from_prev = bool(cfg.hc_pre_from_prev_sublayer)
        self.attn_hc = HcSubLayer(W, S, 5000 + layer_id)
        self.ffn_hc = HcSubLayer(W, S, 6000 + layer_id)
        g = torch.Generator().manual_seed(2000 + layer_id)
        self.w_attn = nn.Parameter(_vec(g, W, 0.5, 1.5), requires_grad=False)
        self.w_mlp = nn.Parameter(_vec(g, W, -1.0, 1.0), requires_grad=False)
        self.attn_boundary = StageBoundary(HcUpdate(self, True))
        self.ffn_boundary = StageBoundary(HcUpdate(self, False))
        self.engram = Engram(cfg, layer_id) if engram else None

    def forward(self, positions, hidden_states, input_ids, forward_batch, input_ids_global):
        fb = forward_batch
        nxt = None if self.engram is not None else self.attn_hc  # HcReadCall.next_boundary
        state = take(fb, next=nxt)
        x = state.input if state.input is not None else combine(state.residual, state.pre)
        a = torch.tanh(x.float() * self.w_attn).to(x.dtype)
        mid = self.attn_boundary.declaration.update.post(a, ReadState(state.residual),
                                                         next=self.ffn_hc)
        x2 = mid.input if mid.input is not None else combine(mid.residual, mid.pre)
        y = (x2.float() * self.w_mlp).to(x2.dtype)
        fb.residual_stream.written = None
        fb.residual_stream.pending = (self.ffn_boundary.declaration.update, y,
                                      ReadState(mid.residual))
        return y


DeepseekV4CudaDecoderLayer = type("DeepseekV4CudaDecoderLayer", (_DecoderLayer,), {})
PPMissingLayer = type("PPMissingLayer", (nn.Identity,), {})


class Engram(nn.Module):
    """Adds a deterministic function of the residual (layers.engram.Engram);
    records the streams it was given."""

    def __init__(self, cfg, layer_id):
        super().__init__()
        g = torch.Generator().manual_seed(3000 + layer_id)
        self.register_buffer("w", _vec(g, cfg.hidden_size, -1.0, 1.0), persistent=False)
        self.seen = []

    def forward(self, residual):
        self.seen.append(residual.clone())
        return (residual.float() + 0.1 * torch.tanh(residual.float() * self.w)).to(residual.dtype)


class DeepseekV4Model(nn.Module):
    """``_forward_impl`` + ``_forward_layers_hc_pre_from_prev`` (CUDA): the
    embedding broadcast to the copies on the first rank, the stream started
    as written, then per layer: materialize and Engram on an Engram layer,
    the layer call; after the loop the FinalRead, then the pipeline hand-over
    (streams and pre) or the head (combine with the last pre)."""

    def __init__(self, cfg, start, end, engram_ids=(1, 14)):
        super().__init__()
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.hc_mult = cfg.hc_mult
        self.hc_pre_from_prev_sublayer = cfg.hc_pre_from_prev_sublayer
        self.decoder_replay_tail = None
        L = cfg.num_hidden_layers
        self.layers = nn.ModuleList(DeepseekV4CudaDecoderLayer(cfg, i, i in engram_ids)
                                    if start <= i < end else PPMissingLayer() for i in range(L))
        self.start_layer, self.end_layer = start, end
        self.layers_run = []

    def forward(self, input_embeds=None, pp_proxy_tensors=None):
        if self.start_layer == 0:
            streams = input_embeds.unsqueeze(1).repeat(1, self.hc_mult, 1)  # hc_broadcast
            pre = None
        else:
            h = pp_proxy_tensors["hidden_states"]
            streams = h.view(h.shape[0], self.hc_mult, self.hidden_size)
            pre = pp_proxy_tensors["pre"]
        state = HcState(streams, pre)
        fb = types.SimpleNamespace(residual_stream=Stream(state))
        hidden_states = streams
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if layer.engram is not None:
                hidden_states = state = materialize(fb)
                hidden_states = layer.engram(state.residual)
                state = state.with_residual(hidden_states)
                fb.residual_stream = Stream(state)  # residual_batch.set_written
            self.layers_run.append(i)
            hidden_states = layer(positions=None, hidden_states=hidden_states, input_ids=None,
                                  forward_batch=fb, input_ids_global=None)
        state = final_read(fb)
        if self.end_layer != self.config.num_hidden_layers:
            return {"hidden_states": state.residual.flatten(1), "pre": state.pre}
        return combine(state.residual, state.pre)


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone
        self.config = backbone.config


def config(hidden=16, hc_mult=4, num_layers=40, model_type="deepseek_v41", v41=True):
    return types.SimpleNamespace(hidden_size=hidden, hc_mult=hc_mult, num_hidden_layers=num_layers,
                                 model_type=model_type, hc_pre_from_prev_sublayer=v41)


def stage_bounds(partition, rank):
    start = sum(partition[:rank])
    return start, start + partition[rank]


def runner(partition=(40,), rank=0, hidden=16, cfg=None, hf_model_type="deepseek_v41",
           dtype=torch.bfloat16, tp_size=1):
    cfg = cfg if cfg is not None else config(hidden, 4, sum(partition))
    start, end = stage_bounds(partition, rank)
    bb = DeepseekV4Model(cfg, start, end)
    bb.to(dtype)
    return types.SimpleNamespace(
        model=DeepseekV4ForCausalLM(bb), is_draft_worker=False, tp_rank=0, tp_size=tp_size,
        pp_rank=rank, pp_size=len(partition),
        server_args=types.SimpleNamespace(enable_two_batch_overlap=False),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(
            model_type=hf_model_type), dtype=dtype))


def embeds(T=5, W=16, seed=0, dtype=torch.bfloat16):
    return (torch.randn(T, W, generator=torch.Generator().manual_seed(seed)) * 2.0).to(dtype)


def run_pipeline(partition, x, hidden=16):
    out = None
    for rank in range(len(partition)):
        run = runner(partition, rank, hidden=hidden)
        install_steering(run, source="env")
        bb = run.model.model
        out = bb(input_embeds=x) if rank == 0 else bb(pp_proxy_tensors=out)
    return out


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)

    def write(self, layers=tuple(range(1, 40)), width=16, alpha="0.75", hint="deepseek_v41",
              hook=RES, seed=0):
        m = good_meta(layers=layers, alpha=alpha)
        m["controlvector.model_hint"] = hint
        m["glp.hook_point"] = hook
        path = os.path.join(self.tmp.name, f"v{seed}.gguf")
        write_gguf(path, m, unit_dirs(layers, width, seed))
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
                      if "post" in vars(bb.layers[i].ffn_boundary.declaration.update))


class Row(Base):
    def test_row_and_variant(self):
        row = ARCH[ROW]
        self.assertEqual(row.layers, frozenset({"DeepseekV4CudaDecoderLayer"}))
        self.assertEqual(row.hooks, frozenset({RES, "hc_fold_pre_engram"}))  # spec 3: test_dsv41_fold.py
        self.assertEqual(row.hint, frozenset({"deepseek_v41"}))
        self.assertEqual((row.install, row.exec_id, row.tp, row.boundary, row.per_stream),
                         ("special:dsv41_hc_post", "layer", "ok", True, True))
        self.assertIn("dsv41_hc_post", SPECIAL)
        self.assertIn("dsv41_hc_post", PREFLIGHT)
        self.assertEqual(VARIANTS["DeepseekV4ForCausalLM"], ((A.selects, ROW),))
        self.assertNotIn("DeepseekV41ForCausalLM", ARCH)
        self.assertIn("DeepseekV41ForCausalLM", REFUSED)

    def test_resolve_picks_the_variant_by_model_type(self):
        run = runner()
        self.assertEqual(resolve_row(run.model, run)[0], ROW)
        for name, mutate in (
                ("only the hf config", lambda r: setattr(r.model.model.config, "model_type",
                                                         "deepseek_v4")),
                ("only the flag", lambda r: (setattr(r.model.model.config, "model_type",
                                                     "deepseek_v4"),
                                             setattr(r.model_config.hf_config, "model_type",
                                                     "deepseek_v4")))):
            with self.subTest(name):
                r = runner()
                mutate(r)
                self.assertEqual(resolve_row(r.model, r)[0], ROW)
        # DeepSeek-V4 (no V4.1 sign anywhere) keeps its own row
        r = runner(cfg=config(model_type="deepseek_v4", v41=False), hf_model_type="deepseek_v4")
        self.assertEqual(resolve_row(r.model, r)[0], "DeepseekV4ForCausalLM")

    def test_install_wraps_the_ffn_post_of_the_file_layers(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = runner()
        rec = self.install(run)
        self.assertEqual(self.wrapped(run), list(range(1, 40)))
        bb = run.model.model
        for i in range(40):  # the attention updates and the forward hooks stay untouched
            self.assertNotIn("post", vars(bb.layers[i].attn_boundary.declaration.update))
            self.assertFalse(bb.layers[i]._forward_hooks)
        self.assertEqual(rec["row"], ROW)
        self.assertEqual(rec["model_class"], "DeepseekV4ForCausalLM")
        self.assertEqual(rec["local_layer_ids"], list(range(1, 40)))
        self.assertEqual((rec["width"], rec["hook_point"], rec["alpha"]), (16, RES, 0.75))
        self.assertEqual(rec["layout"], "stage-boundary")
        self.assertEqual(rec["edit"], "hc copies: mean coefficient")
        self.assertEqual(rec["site"], A.SITE + " [hc copies: mean coefficient]")
        self.assertIn(f"row={ROW} install=special:dsv41_hc_post site={A.SITE} [hc copies: mean "
                      f"coefficient] width=16 layout=stage-boundary", self.log)
        self.assertTrue(os.path.exists(rec["manifest"]))

    def test_install_twice_is_refused_by_the_handler(self):
        self.write()
        run = runner()
        self.install(run)
        layer = run.model.model.layers[5]
        site = wcore.make_site(run.model.model, 5, width=16, per_stream=True)
        with self.assertRaisesRegex(RuntimeError, "already wrapped on this instance"):
            SPECIAL["dsv41_hc_post"][0](layer, site, layer_id=5, backbone=run.model.model,
                                        row=ARCH[ROW])


class Steering(Base):
    def spy(self, run):
        """Record, per steered layer, what the wrapped post returns."""
        bb, out = run.model.model, {}
        for i in range(bb.start_layer, bb.end_layer):
            upd = bb.layers[i].ffn_boundary.declaration.update
            inner = upd.post

            def post(*a, _i=i, _inner=inner, **k):
                s = _inner(*a, **k)
                out[_i] = s
                return s
            upd.post = post
        return out

    def test_each_steered_layer_gets_the_mean_edit(self):
        self.write()
        run = runner()
        self.install(run)
        got = self.spy(run)
        bb = run.model.model
        bb(input_embeds=embeds())
        self.assertEqual(sorted(got), list(range(40)))
        for i in range(1, 40):
            with self.subTest(layer=i):
                upd = bb.layers[i].ffn_boundary.declaration.update
                before, after = upd.last, got[i]
                d = bb._steer_stack[i, 0]
                want = A.hc_mean_rows(before.residual, d, bb._steer_alpha)
                self.assertTrue(torch.equal(bits(after.residual), bits(want)))
                self.assertIs(after.pre, before.pre)  # pre weights the copies: kept
        # layer 0 has no direction: its post passes through untouched
        self.assertIs(got[0], bb.layers[0].ffn_boundary.declaration.update.last)

    def test_the_fused_shortcut_is_dropped(self):
        """The post hands the next layer an input and statistics computed from
        the unedited streams; the steered state carries neither, so the next
        read recomputes them from the edited streams."""
        self.write()
        run = runner()
        self.install(run)
        got = self.spy(run)
        bb = run.model.model
        bb(input_embeds=embeds())
        n_short = 0
        for i in range(1, 39):
            before = bb.layers[i].ffn_boundary.declaration.update.last
            n_short += before.input is not None
            self.assertIsNone(got[i].input)
            self.assertIsNone(got[i].stats)
        self.assertGreater(n_short, 30)  # the shortcut was there to drop (not before Engram)

    def test_engram_reads_the_steered_streams(self):
        """Pre-Engram: layer 14's Engram is given layer 13's edited streams;
        layer 1's is given layer 0's unedited ones (the file has no direction.0)."""
        self.write()
        run = runner()
        self.install(run)
        got = self.spy(run)
        bb = run.model.model
        bb(input_embeds=embeds())
        e14, e1 = bb.layers[14].engram.seen, bb.layers[1].engram.seen
        self.assertEqual((len(e14), len(e1)), (1, 1))
        self.assertTrue(torch.equal(bits(e14[0]), bits(got[13].residual)))
        self.assertFalse(torch.equal(bits(e14[0]),
                                     bits(bb.layers[13].ffn_boundary.declaration.update.last.residual)))
        self.assertTrue(torch.equal(bits(e1[0]), bits(got[0].residual)))

    def test_alpha_zero_is_stock_and_steering_changes_the_output(self):
        x = embeds()
        stock = runner().model.model(input_embeds=x)
        self.write(alpha="0.0")
        run = runner()
        self.install(run)
        self.assertTrue(torch.equal(bits(run.model.model(input_embeds=x)), bits(stock)))
        self.write(alpha="0.75", seed=1)
        run = runner()
        self.install(run)
        steered = run.model.model(input_embeds=x)
        self.assertFalse(torch.equal(bits(steered), bits(stock)))

    def test_last_layer_is_steered_before_the_head(self):
        """Layer 39's post runs in the FinalRead; its edited streams reach the head."""
        self.write(layers=(39,))
        run = runner()
        self.install(run)
        got = self.spy(run)
        bb = run.model.model
        out = bb(input_embeds=embeds())
        self.assertTrue(torch.equal(bits(out), bits(combine(got[39].residual, got[39].pre))))
        before = bb.layers[39].ffn_boundary.declaration.update.last
        self.assertFalse(torch.equal(bits(got[39].residual), bits(before.residual)))

    def test_pipeline_ranks_steer_their_own_layers_and_equal_one_rank(self):
        self.write()
        x = embeds()
        one = run_pipeline((40,), x)
        for part in ((14, 13, 13), (20, 20), (1, 38, 1)):
            with self.subTest(partition=part):
                for rank in range(len(part)):
                    run = runner(part, rank)
                    rec = self.install(run)
                    s, e = stage_bounds(part, rank)
                    want = [i for i in range(1, 40) if s <= i < e]
                    self.assertEqual(rec["local_layer_ids"], want)
                    self.assertEqual(self.wrapped(run), want)
                    self.assertEqual(rec["pp_rank"], rank)
                self.assertTrue(torch.equal(bits(run_pipeline(part, x)), bits(one)))

    def test_each_mode(self):
        self.write()
        os.environ[A.COPIES_ENV] = "each"
        run = runner()
        rec = self.install(run)
        self.assertEqual(rec["edit"], "hc copies: each projected")
        got = self.spy(run)
        bb = run.model.model
        bb(input_embeds=embeds())
        for i in (1, 20, 39):
            before = bb.layers[i].ffn_boundary.declaration.update.last
            want = wcore.steer_rows(before.residual, None, bb._steer_stack[i, 0], bb._steer_alpha)
            self.assertTrue(torch.equal(bits(got[i].residual), bits(want)))

    def test_fired_check(self):
        """A post the loop reaches without the instance attribute (the class's
        method) skips the edit: the forward fails closed. A post run twice
        fails closed too."""
        self.write()
        run = runner()
        self.install(run)
        bb = run.model.model
        upd = bb.layers[7].ffn_boundary.declaration.update
        wrapper = upd.post
        upd.post = types.MethodType(HcUpdate.post, upd)
        with self.assertRaisesRegex(RuntimeError, r"1 steered .* did not run .*layers \[7\]"):
            bb(input_embeds=embeds())
        upd.post = wrapper
        bb(input_embeds=embeds())  # back to normal

        def twice(*a, **k):
            wrapper(*a, **k)
            return wrapper(*a, **k)
        upd.post = twice
        with self.assertRaisesRegex(RuntimeError, r"ran more than once .*layers \[7\]"):
            bb(input_embeds=embeds())

    def test_post_must_return_the_hc_state(self):
        """The wrapper checks what the original post returned, at every call."""
        self.write()
        run = runner()
        self.install(run)
        bb = run.model.model
        for name, bad in (("a tensor", torch.zeros(5, 4, 16)),
                          ("pending streams", HcState(object())),
                          ("wrong width", HcState(torch.zeros(5, 4, 8))),
                          ("wrong copies", HcState(torch.zeros(5, 2, 16)))):
            with self.subTest(name):
                cls = type("HcUpdate", (HcUpdate,), {"post": lambda self, *a, _b=bad, **k: _b})
                layer = bb.layers[3]
                layer.ffn_boundary = StageBoundary(cls(layer, False))
                site = wcore.make_site(bb, 3, width=16, per_stream=True)
                SPECIAL["dsv41_hc_post"][0](layer, site, layer_id=3, backbone=bb, row=ARCH[ROW])
                with self.assertRaises(A.Dsv41SiteError):
                    layer.ffn_boundary.declaration.update.post(None, None)

    def test_diagnostics_judge_the_mean_stream(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_DIAG"] = "1"
        os.environ["WEIGHTLESS_STEER_DIAG_DIR"] = self.tmp.name
        run = runner(dtype=torch.float32)
        rec = self.install(run)
        run.model.model(input_embeds=embeds(T=9, dtype=torch.float32))
        diag = json.load(open(rec["diag"]))
        for i in (1, 13, 20, 39):
            st = diag["layers"][str(i)]
            v = wcore.layer_verdict(st, 0.75, floor="elementwise")
            with self.subTest(layer=i):
                self.assertEqual(v["edit"], "pass")
                self.assertEqual(st["tokens"], 9)  # one row per token: the mean stream

    def test_published_files(self):
        for path, alpha in ((GLP1017, 0.75), (GLP0909, 0.5)):
            with self.subTest(path=path):
                if path is None:
                    self.skipTest("the DeepSeek-V4.1 GLP files are not under WEIGHTLESS_TEST_GLP_DIR")
                os.environ["WEIGHTLESS_STEER_PATH"] = path
                run = runner(hidden=5120)
                rec = self.install(run)
                self.assertEqual(rec["local_layer_ids"], list(range(1, 40)))
                self.assertEqual((rec["width"], rec["alpha"], rec["hook_point"]), (5120, alpha, RES))
                self.assertEqual(rec["model_hint"], "deepseek_v41")
                bb = run.model.model
                out = bb(input_embeds=embeds(T=2, W=5120))
                self.assertEqual(tuple(out.shape), (2, 5120))
                self.assertTrue(bool(torch.isfinite(out.float()).all()))
                print(f"  {os.path.basename(path)} -> {ROW}: layers 1..39 width 5120 "
                      f"alpha {alpha}, one forward fired {len(rec['local_layer_ids'])} sites")


class Refusals(Base):
    def assertRefused(self, run, regex):
        with self.assertRaisesRegex(RuntimeError, regex):
            self.install(run)
        self.assertEqual(self.wrapped(run), [])
        self.assertFalse(hasattr(run.model.model, "_steer_stack"))

    def test_model_refusals(self):
        self.write()
        cases = (
            ("text model type", r"serves model_type 'deepseek_v41' only.*'deepseek_v41_text'",
             lambda r: setattr(r.model_config.hf_config, "model_type", "deepseek_v41_text")),
            ("V4 model with the V4.1 flag", r"serves model_type 'deepseek_v41' only.*'deepseek_v4'",
             lambda r: (setattr(r.model_config.hf_config, "model_type", "deepseek_v4"),
                        setattr(r.model.model.config, "model_type", "deepseek_v4"))),
            ("no flag on the backbone", r"hc_pre_from_prev_sublayer=True",
             lambda r: setattr(r.model.model, "hc_pre_from_prev_sublayer", False)),
            ("hc_mult 3", r"hc_mult=3 is not a power of two",
             lambda r: setattr(r.model.model.config, "hc_mult", 3)),
            ("decoder tail graph", r"decoder tail graph is on.*"
                                   r"SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=0",
             lambda r: setattr(r.model.model, "decoder_replay_tail", object())),
            ("copies mode", r"WEIGHTLESS_STEER_HC_COPIES='sum'",
             lambda r: os.environ.__setitem__(A.COPIES_ENV, "sum")),
            ("two-batch overlap", r"two-batch overlap",
             lambda r: setattr(r.server_args, "enable_two_batch_overlap", True)),
        )
        for name, regex, mutate in cases:
            with self.subTest(name):
                os.environ.pop(A.COPIES_ENV, None)
                run = runner()
                mutate(run)
                self.assertRefused(run, regex)

    def test_file_refusals(self):
        run = runner()
        self.write(hook=FFN, hint="deepseek_v41")
        self.assertRefused(run, r"glp.hook_point='ffn_out_pre_residual', which the DeepSeek-V4.1 "
                                r"row does not serve: that is the FFN write alone")
        self.write(hint="deepseek_v4", seed=2)
        self.assertRefused(runner(), r"is a DeepSeek-V4 file \(model_hint='deepseek_v4'\)")
        self.write(hint="qwen3_5", seed=3)
        self.assertRefused(runner(), r"model_hint='qwen3_5'")
        self.write(width=8, seed=4)
        self.assertRefused(runner(), r"width")
        if DSV4_FFN is not None:  # the published DeepSeek-V4-0731 GLP-29 file
            os.environ["WEIGHTLESS_STEER_PATH"] = DSV4_FFN
            self.assertRefused(runner(hidden=4096, cfg=config(4096, 4, 43)),
                               r"ffn_out_pre_residual.*does not serve")

    def test_layer_refusals(self):
        self.write()

        def update_is_attention(r):
            layer = r.model.model.layers[9]
            layer.ffn_boundary = StageBoundary(layer.attn_boundary.declaration.update)

        def not_stage_built(r):
            for layer in r.model.model.layers:
                if hasattr(layer, "ffn_boundary"):
                    del layer.attn_boundary, layer.ffn_boundary

        def one_not_stage_built(r):
            layer = r.model.model.layers[9]
            del layer.attn_boundary, layer.ffn_boundary

        def other_layer(r):
            layer = r.model.model.layers[9]
            layer.ffn_boundary.declaration.update.layer = r.model.model.layers[8]

        def nextn(r):
            r.model.model.layers[9].boundary_pre_from_prev = False

        def wrong_id(r):
            r.model.model.layers[9].layer_id = 10

        for name, regex, mutate in (
                ("the FFN update is the attention one", r"layer 9: the FFN boundary's update",
                 update_is_attention),
                ("not stage-built", r"not built from SGLang's stage boundaries", not_stage_built),
                ("one layer not stage-built", r"mixes SGLang layer layouts", one_not_stage_built),
                ("another layer's update", r"layer 9: the FFN boundary's update", other_layer),
                ("boundary_pre_from_prev off", r"boundary_pre_from_prev is not True", nextn),
                ("layer_id", r"says layer_id=10", wrong_id)):
            with self.subTest(name):
                run = runner()
                mutate(run)
                with self.assertRaisesRegex(RuntimeError, regex):
                    self.install(run)

    def test_layer_class_is_exact(self):
        self.write()
        run = runner()
        bb = run.model.model
        other = type("DeepseekV4PlatformDecoderLayer", (_DecoderLayer,), {})
        bb.layers[20] = other(bb.config, 20)
        self.assertRefused(run, r"layer 20 is DeepseekV4PlatformDecoderLayer, not one of "
                                r"\['DeepseekV4CudaDecoderLayer'\]")


def _dsv41_config():
    p = os.environ.get("WEIGHTLESS_TEST_DSV41_CONFIG", "").strip()
    if p and os.path.isdir(p):
        p = os.path.join(p, "config.json")
    return p if p and os.path.isfile(p) else None


@unittest.skipUnless(_dsv41_config(), "set WEIGHTLESS_TEST_DSV41_CONFIG to the DeepSeek-V4.1 config.json")
class RealConfig(Base):
    """The checkpoint's config.json through SGLang's own config class: served
    as DeepseekV4ForCausalLM, model_type deepseek_v41, the V4.1 hc scheme;
    the row is picked and installs at the checkpoint's depth and width."""

    def test_real_config_picks_the_row_and_installs(self):
        raw = json.load(open(_dsv41_config()))
        self.assertEqual(raw["architectures"], ["DeepseekV41ForCausalLM"])
        try:
            from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
        except Exception as e:
            self.skipTest(f"SGLang's DeepSeek-V4.1 config class is not importable ({e})")
        cfg = DeepseekV41Config(**raw)
        self.assertEqual(cfg.architectures, ["DeepseekV4ForCausalLM"])
        self.assertEqual(cfg.model_type, "deepseek_v41")
        self.assertIs(cfg.hc_pre_from_prev_sublayer, True)
        self.assertEqual((cfg.num_hidden_layers, cfg.hidden_size, cfg.hc_mult), (40, 5120, 4))
        self.assertEqual(list(cfg.engram_layer_ids), [1, 14])
        if GLP1017 is not None:
            os.environ["WEIGHTLESS_STEER_PATH"] = GLP1017
        else:
            self.write(width=5120)
        bb_cfg = config(cfg.hidden_size, cfg.hc_mult, cfg.num_hidden_layers, cfg.model_type,
                        cfg.hc_pre_from_prev_sublayer)
        run = runner(cfg=bb_cfg)
        run.model_config.hf_config = cfg
        self.assertEqual(resolve_row(run.model, run)[0], ROW)
        rec = self.install(run)
        self.assertEqual(rec["local_layer_ids"], list(range(1, 40)))


# ---------------------------------------------------------------- SGLang's real mHC boundary code

os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "0")

try:
    from sglang.srt.models import deepseek_v4_boundary as RB
    from sglang.srt.models import deepseek_v4_mhc as RM
    from sglang.srt.layers.layer_boundary.residual import batch as RBATCH
    from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream as RStream
    from sglang.test.kernels.mhc import hc_split_sinkhorn_torch
    import sglang.kernels.ops.layernorm.mhc as RKERN
    HAVE_REAL = True
except Exception as e:  # no SGLang, or one without the V4.1 mHC boundary
    HAVE_REAL, WHY_REAL = False, f"{type(e).__name__}: {e}"


def _torch_mhc_combine(x_flat, pre, hc, dtype):
    T = x_flat.shape[0]
    return (pre.float().unsqueeze(-1) * x_flat.float().view(T, hc, -1)).sum(1).to(dtype)


@unittest.skipUnless(HAVE_REAL, "SGLang's DeepSeek-V4.1 mHC boundary is not importable here: "
                     + (WHY_REAL if not HAVE_REAL else ""))
class RealSGLang(Base):
    """SGLang's HcUpdate.post (run_moe_post, _compute_triplet, _post_fusion,
    post), HcState, ReadState, materialize, read_output with FinalRead and
    ResidualStream, on CPU. The TileLang sinkhorn split and the combine are
    given SGLang's torch reference (test/kernels/mhc.py) and the one-line
    torch combine."""

    W, S, T = 64, 4, 5

    def setUp(self):
        super().setUp()
        for target, value in (("hc_split_sinkhorn", hc_split_sinkhorn_torch),
                              ("mhc_combine", _torch_mhc_combine)):
            p = mock.patch.object(RKERN, target, value)
            p.start()
            self.addCleanup(p.stop)

    def layer(self, layer_id=3):
        W, S = self.W, self.S
        from sglang.srt.layers.layernorm import RMSNorm
        cfg = RM.HcConfig(mult=S, sinkhorn_iters=20, eps=1e-6, rms_eps=1e-6, hidden=W,
                          pre_from_prev=True, cp_prefill=False)
        g = torch.Generator().manual_seed(layer_id)

        def sub():
            return RM.HcSubLayer(cfg, 0.05 * torch.randn((2 + S) * S, S * W, generator=g),
                                 1.0 + 0.1 * torch.randn(3, generator=g),
                                 0.1 * torch.randn((2 + S) * S, generator=g), RMSNorm(W, eps=1e-6))
        layer = nn.Module()
        layer.layer_id, layer.hidden_size, layer.hc_mult = layer_id, W, S
        layer.boundary_pre_from_prev, layer.hc_cfg = True, cfg
        layer.attn_hc, layer.ffn_hc = sub(), sub()
        layer.mlp = types.SimpleNamespace(tp_size=1)
        layer.attn_boundary = StageBoundary(RB.HcUpdate(layer, True))
        layer.ffn_boundary = StageBoundary(RB.HcUpdate(layer, False))
        return layer

    def owner(self, layer_id, alpha):
        o = nn.Module()
        stack = torch.zeros(8, 1, self.W)
        stack[layer_id, 0] = _unit(self.W, 9)
        o.register_buffer("_steer_stack", stack)
        o.register_buffer("_steer_alpha", torch.tensor(alpha))
        o.config = types.SimpleNamespace(hidden_size=self.W, hc_mult=self.S)
        return o

    def steer(self, layer, alpha=0.75, check=None):
        o = self.owner(layer.layer_id, alpha)
        site = wcore.make_site(o, layer.layer_id, width=self.W, per_stream=True, check=check,
                               edit=A.HcEdit("mean"))
        undo = SPECIAL["dsv41_hc_post"][0](layer, site, layer_id=layer.layer_id, backbone=o,
                                           row=ARCH[ROW])
        return o, undo

    def inputs(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        R = torch.randn(self.T, self.S, self.W, generator=g).bfloat16()
        y = torch.randn(self.T, self.W, generator=g).bfloat16()
        return R, y

    def test_the_wrapped_post_returns_the_edited_real_state(self):
        layer = self.layer()
        upd = layer.ffn_boundary.declaration.update
        R, y = self.inputs()
        stock = upd.post(y, RB.ReadState(R, call=RB.LayerCall()))
        self.assertIsInstance(stock, RM.HcState)
        o, undo = self.steer(layer)
        got = upd.post(y, RB.ReadState(R, call=RB.LayerCall()))
        self.assertIsInstance(got, RM.HcState)
        want = A.hc_mean_rows(stock.residual, o._steer_stack[3, 0], o._steer_alpha)
        self.assertTrue(torch.equal(bits(got.residual), bits(want)))
        self.assertTrue(torch.equal(got.pre, stock.pre))
        self.assertIsNone(got.input)
        self.assertIsNone(got.stats)
        undo()
        self.assertNotIn("post", vars(upd))

    def test_materialize_and_the_final_read_run_the_wrapper_once(self):
        layer = self.layer()
        check = wcore.FiredCheck([3])
        o, _ = self.steer(layer, check=check)
        upd = layer.ffn_boundary.declaration.update
        R, y = self.inputs()
        for name in ("materialize", "final read"):
            with self.subTest(name):
                check.begin()
                fb = types.SimpleNamespace(residual_stream=None)
                RBATCH.start(fb)
                stream = RBATCH.stream_of(fb)
                stream.residual = RB.ReadState(R, call=RB.LayerCall())
                h = stream.record(y, upd)
                if name == "materialize":
                    st = RB.materialize(h, fb)
                else:
                    st = RBATCH.read_output(h, fb, RB.FinalRead(), None)
                check.end()
                self.assertEqual(check.fired[3], 1)
                self.assertIsInstance(st, RM.HcState)
                self.assertIsNone(st.input)

    def test_the_attention_update_is_left_alone(self):
        layer = self.layer()
        self.steer(layer)
        self.assertNotIn("post", vars(layer.attn_boundary.declaration.update))
        self.assertIn("post", vars(layer.ffn_boundary.declaration.update))

    def test_with_residual_keeps_pre_and_drops_the_shortcut(self):
        R, _ = self.inputs()
        s = RM.HcState(R, pre=torch.ones(self.T, self.S), input=RM.HcNormed(R[:, 0]), stats=(R,))
        t = s.with_residual(R + 1)
        self.assertIs(t.pre, s.pre)
        self.assertIsNone(t.input)
        self.assertIsNone(t.stats)


# ---------------------------------------------------------------- the SGLang trees

MODEL = "srt/models/deepseek_v4.py"
BOUNDARY = "srt/models/deepseek_v4_boundary.py"
MHC = "srt/models/deepseek_v4_mhc.py"
CONFIG = "srt/configs/deepseek_v41.py"


@unittest.skipUnless(_trees(MODEL, BOUNDARY, MHC, CONFIG),
                     "no SGLang package with the DeepSeek-V4.1 hc boundary found")
class Structure(unittest.TestCase):
    """Pins, per SGLang tree, what the row relies on."""

    def assertIn(self, member, container, msg=None):  # noqa: N802
        if isinstance(member, str) and isinstance(container, str):
            member, container = _flat(member), _flat(container)
        super().assertIn(member, container, msg)

    def trees(self):
        return _trees(MODEL, BOUNDARY, MHC, CONFIG)

    def test_v41_is_served_through_the_v4_class(self):
        for t in self.trees():
            with self.subTest(tree=t):
                src = _src(_parse(t, CONFIG))
                self.assertIn("if values.get('architectures') == ['DeepseekV41ForCausalLM']: "
                              "values['architectures'] = ['DeepseekV4ForCausalLM']", src)
                cfg = _cls(_parse(t, CONFIG), "DeepseekV41Config")
                self.assertIn("hc_pre_from_prev_sublayer = True", _src(cfg))
                models = os.path.join(t, "srt", "models")
                for f in sorted(os.listdir(models)):
                    if f.endswith(".py"):
                        with open(os.path.join(models, f)) as fh:
                            self.assertNotIn("class DeepseekV41ForCausalLM", fh.read(), f)

    def test_the_layer_declares_the_ffn_hc_update(self):
        for t in self.trees():
            with self.subTest(tree=t):
                m = _parse(t, MODEL)
                layer = _cls(m, "DeepseekV4CudaDecoderLayer")
                init = _src(_fn(layer, "__init__"))
                self.assertIn("declarations = hc_boundary.stages(self.config, self.layer_id, "
                              "layer=self)", init)
                self.assertIn("self.attn_boundary, self.ffn_boundary = append_stages(", init)
                fwd = _src(_fn(layer, "forward"))
                self.assertIn("with self.ffn_boundary.exit(forward_batch) as output:", fwd)
                self.assertIn("return output.finish(hidden_states)", fwd)
                b = _parse(t, BOUNDARY)
                stages = _src([n for n in b.body if isinstance(n, ast.FunctionDef)
                               and n.name == "stages"][0])
                self.assertIn("fu = HcUpdate(layer, False, applied_at_exit=applied)", stages)
                self.assertIn("update=fu,", stages)
                self.assertIn("applied = False if lagged else not fused", stages)
                base = _cls(m, "DeepseekV4DecoderLayerBase")
                self.assertIn("config.hc_pre_from_prev_sublayer and (not is_nextn)",
                              _src(_fn(base, "__init__")))
                alias = _src(m)
                self.assertIn("DeepseekV4DecoderLayer = DeepseekV4CudaDecoderLayer if _is_cuda "
                              "else DeepseekV4PlatformDecoderLayer", alias)

    def test_every_reader_runs_the_post_through_the_instance(self):
        """HcUpdate.update calls self.post; the attention read, the completing
        fusion, materialize and the FinalRead call update.post on the pending
        update object. The V4.1 post returns an HcState from run_moe_post."""
        for t in self.trees():
            with self.subTest(tree=t):
                b = _parse(t, BOUNDARY)
                upd = _cls(b, "HcUpdate")
                post = _fn(upd, "post")
                self.assertEqual([a.arg for a in post.args.args], ["self", "output", "state"])
                self.assertEqual([a.arg for a in post.args.kwonlyargs], ["next"])
                self.assertIn("return mhc.run_moe_post(hc, output, state.streams,", _src(post))
                self.assertIn("hc = self.layer.attn_hc if self.attention else self.layer.ffn_hc",
                              _src(post))
                self.assertEqual(_src(_fn(upd, "update")).split("\n")[-1].strip(),
                                 "return self.post(output, state)")
                src = _src(b)
                self.assertIn("update.post(output, state, next=read.next_boundary)", src)
                self.assertIn("update.post(value, residual, next=self.next_boundary)", src)
                self.assertIn("return update.post(output.pieces, residual)", src)
                mat = _src([n for n in b.body if isinstance(n, ast.FunctionDef)
                            and n.name == "materialize"][0])
                self.assertIn("result = pending.update.post(pending.complete(), stream.residual, "
                              "next=next)", mat)
                self.assertIn("stream.write(result)", mat)
                fr = _cls(b, "FinalRead")
                self.assertIn("return update.update(value, residual)",
                              _src(_fn(fr, "update_and_read")))
                calls = [n for n in ast.walk(b) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "post"]
                for c in calls:  # never through the class
                    self.assertNotIn(_src(c.func.value), ("HcUpdate", "type(update)"))

    def test_with_residual_drops_the_shortcut(self):
        for t in self.trees():
            with self.subTest(tree=t):
                st = _cls(_parse(t, MHC), "HcState")
                self.assertEqual(_src(_fn(st, "with_residual")).split("\n")[-1].strip(),
                                 "return HcState(residual, self.pre)")
                self.assertIn("def residual(self) -> torch.Tensor:", _src(st))
                self.assertIn("next=next if state.call.allow_cross_layer_fusion else None",
                              _src(_fn(_cls(_parse(t, BOUNDARY), "HcUpdate"), "post")))

    def test_engram_is_added_after_the_pending_post_is_materialized(self):
        """In the V4.1 loop, an Engram layer's iteration materializes the
        pending post (the previous layer's, through update.post) and only then
        calls engram(state.residual, ...) on that state; the loop calls each
        layer as layer(...); the last post is read by FinalRead after the loop."""
        for t in self.trees():
            with self.subTest(tree=t):
                m = _parse(t, MODEL)
                fn = _fn(_cls(m, "DeepseekV4Model"), "_forward_layers_hc_pre_from_prev")
                loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
                         and _src(n.iter) == "range(self.start_layer, self.end_layer)"]
                self.assertEqual(len(loops), 1)
                body = _flat(_src(loops[0]))
                i_mat = body.index("hidden_states = state = hc_boundary.materialize(")
                i_eng = body.index("hidden_states = engram(state.residual, "
                                   "hash_ids[:, engram.layer_hash_index]")
                i_with = body.index("state = state.with_residual(hidden_states)")
                i_call = body.index("hidden_states = self.layers[i](")
                self.assertLess(i_mat, i_eng)
                self.assertLess(i_eng, i_with)
                self.assertLess(i_with, i_call)
                self.assertIn("or layer.engram is not None", body)
                whole = _flat(_src(fn))
                after = whole[whole.index(body) + len(body):]
                self.assertIn("state = residual_batch.read_output(hidden_states, forward_batch, "
                              "hc_boundary.FinalRead(), None)", after)
                # no Engram on the next layer's read: the post gets no shortcut there
                nb = _src(_fn(_cls(_parse(t, BOUNDARY), "HcReadCall"), "next_boundary"))
                self.assertIn("self.attention and self.layer.engram is not None", nb)

    def test_the_pipeline_hand_over_and_the_pre_mix(self):
        """The rank's streams go to the next rank flattened after the loop. A tree
        without the V4.1 pipeline wiring refuses PP in the loop (upstream main); a
        tree with it (the Thireus/sglang fork) hands the pre-mix over with the
        streams, and the next rank's HcState starts from them. On both, the head
        combines the last streams with the last pre."""
        for t in self.trees():
            with self.subTest(tree=t):
                m = _parse(t, MODEL)
                fn = _src(_fn(_cls(m, "DeepseekV4Model"), "_forward_layers_hc_pre_from_prev"))
                impl = _src(_fn(_cls(m, "DeepseekV4Model"), "_forward_impl"))
                if "pre-mix hand-off across PP is not wired" in fn:
                    self.assertIn("return PPProxyTensors({'hidden_states': hidden_states.flatten(1)})",
                                  impl)
                else:
                    self.assertIn("state = mhc.HcState(hidden_states, prev_pre)", fn)
                    self.assertIn("tensors = {'hidden_states': hidden_states.flatten(1)}", impl)
                    self.assertIn("self._pack_stage_outputs(tensors, last_pre, received_sources)", impl)
                # the head combines the last streams with the last pre (the cast of the
                # streams differs between trees)
                self.assertRegex(impl, r"hidden_states = mhc_combine\(pre_hc_head(\.float\(\))?, "
                                       r"last_pre, self\.hc_mult, hidden_states\.dtype\)")

    def test_the_decoder_tail_graph_is_an_attribute_of_the_backbone(self):
        for t in self.trees():
            with self.subTest(tree=t):
                m = _parse(t, MODEL)
                init = _src(_fn(_cls(m, "DeepseekV4Model"), "__init__"))
                self.assertIn("self.decoder_replay_tail = DecoderReplayTail.maybe_create(self)",
                              init)
                fn = _src(_fn(_cls(m, "DeepseekV4Model"), "_forward_layers_hc_pre_from_prev"))
                self.assertIn("if self.decoder_replay_tail is not None:", fn)

    def test_the_backbone_forward_is_called_on_the_instance(self):
        """The fired check wraps the backbone instance's forward: the served
        class calls self.model.forward(...)."""
        for t in self.trees():
            with self.subTest(tree=t):
                m = _parse(t, MODEL)
                fwd = _src(_fn(_cls(m, "DeepseekV4ForCausalLM"), "forward"))
                self.assertIn("hidden_states = self.model.forward(input_ids, positions, "
                              "forward_batch, input_embeds, pp_proxy_tensors)", fwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
