"""The generic core, CPU always (the CUDA cases run too when a GPU is
visible): the golden bits against the first release, the edit against the
vLLM and HF reference implementations, the fixed-point dot product, the
fused kernel against the torch path, the kernel choice, the tuple codec and
its refusals, the slot check, the fired check, bounded memory, the
alpha-aware layer-map statistics, and install_steering on declarative rows.

References for the math (this checkout):
- vLLM path: SteeringCore.apply (vllm-plugin/weightless_steer/core.py) via
  SteeredModelMixin._steer_post_layer (archs/base.py): h = x + r;
  apply(h) - r.
- HF path: tools/captain-vector/apply_transformers.py _make_hook:
  t - ((t @ D.T) * a) @ D on the full stream t.
The SGLang site evaluates the delta form x' = x - alpha ((x + r).d) d.
"""
import inspect
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/
import glpfiles  # noqa: E402  (sets up the import paths)
from glpfiles import good_meta, good_tensors, write_gguf  # noqa: E402
from test_archs import test_qwen38 as fakes  # noqa: E402  (the fake qwen3_5 runner)

import apply_transformers as hf_ref  # noqa: E402
from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.archs import ARCH  # noqa: E402
from weightless_sglang.archs import base as archs_base  # noqa: E402
from weightless_sglang.archs.base import HOOK_POINT, ArchRow  # noqa: E402
from weightless_sglang.core import (LayerDiag, chunk_rows, layer_verdict,  # noqa: E402
                                    make_post_layer_hook, make_site, steer_delta, steer_rows)
from weightless_sglang.install import install_steering  # noqa: E402
from weightless_steer.archs.base import SteeredModelMixin  # noqa: E402
from weightless_steer.container import load_control_vector  # noqa: E402
from weightless_steer.core import SteeringCore  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
LAYER = 30
NEXT = 31


def unit(w, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(w, generator=g, dtype=torch.float64)
    return (v / v.norm()).float()


def rows(T, W, dtype, device, seed=0, residual=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, W, generator=g, dtype=torch.float64)
    r = torch.randn(T, W, generator=g, dtype=torch.float64) * 4.0
    for j in (7, W // 3, W - 5):
        r[:, j] *= 400.0  # a few massive residual values (as the model cards describe)
    if not residual:
        x = x + r  # a stream site: the stream carries them itself
    return x.to(dtype).to(device), (r.to(dtype).to(device) if residual else None)


class Owner(nn.Module):
    def __init__(self, W, device, alpha, dirs):
        super().__init__()
        stack = torch.zeros(64, 1, W)
        for i, d in dirs.items():
            stack[i, 0] = d
        self.register_buffer("_steer_stack", stack.to(device), persistent=False)
        self.register_buffer("_steer_alpha", torch.tensor(float(alpha), device=device),
                             persistent=False)


def same_bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8))


# The edit math exactly as the first release of the plugin shipped it
# (fixed_point_bits, dot_fixed and steer_delta of core.py, copied unchanged).
# The golden tests compare the current code against it bit for bit, so a
# refactor cannot change what a Qwen3.8-27B server computes.


def first_release_fixed_point_bits(width: int) -> int:
    """Magnitude budget, in bits, of one term of the fixed-point dot product.

    Each term is scaled below 2**bits, so that ``width`` of them add up in an
    int64 without overflow (5120 wide: 49 bits).
    """
    return 62 - (int(width) - 1).bit_length()


def first_release_dot_fixed(h: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Row-wise ``h . d`` in fp32 whose result does not depend on the
    order of the additions.

    h: fp32 [..., H]; d: fp32 [H]. Steps, per row:
      p     = h * d                        (fp32, one rounding per term)
      e     = the fp32 exponent of max |p|
      s     = bits + 126 - e, kept in [-100, 100]
      q     = int64(p * 2**s)              (exact scale, cut toward zero)
      c     = f32(sum q) * 2**-s           (integer sum: exact, any order)
    Every term is below 2**bits after scaling, so the int64 sum cannot
    overflow, and the cut loses at most 2**-bits of the largest term per
    term: far below fp32 precision. fused.py runs the same steps, which
    is why the two paths agree bit for bit.
    """
    bits = first_release_fixed_point_bits(h.shape[-1])
    p = h * d
    pm = p.abs().amax(-1, keepdim=True)
    e = (pm.view(torch.int32) >> 23) & 0xFF
    s = (bits + 126 - e).clamp(-100, 100)
    scale = ((s + 127) << 23).view(torch.float32)
    inv = ((127 - s) << 23).view(torch.float32)
    q = (p * scale).to(torch.int64)
    acc = q.sum(-1, keepdim=True)
    return (acc.to(torch.float32) * inv).squeeze(-1)


def first_release_steer_delta(x: torch.Tensor, r, d: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Return x' such that x' + r == (x + r) - alpha * ((x + r) . d) d.

    x, r: [..., H] (r may be None: then the stream is x itself).
    d: fp32 [H] unit direction. alpha: fp32 0-d tensor. The math runs in
    fp32 and rounds once to x.dtype. float64 inputs (used by the tests to
    compare against the exact form) use a plain float64 dot product.
    """
    ct = torch.promote_types(x.dtype, torch.float32)  # fp32, or fp64 in tests
    xf = x.to(ct)
    d = d.to(ct)
    h = xf if r is None else xf + r.to(ct)
    c = first_release_dot_fixed(h, d) if ct == torch.float32 else h @ d  # [...]
    t = (alpha.to(ct) * c).unsqueeze(-1)
    return (xf - t * d).to(x.dtype)


ref = types.SimpleNamespace(fixed_point_bits=first_release_fixed_point_bits,
                            dot_fixed=first_release_dot_fixed,
                            steer_delta=first_release_steer_delta)


H = 5120
WIDTHS = (5120, 16384)  # Qwen3.8-27B's hidden size, GLM-5.3-Flash's 4-stream mHC width
T = 37
ALPHAS = (0.0, 0.5, 1.0, 2.0)
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)} (visible: {torch.cuda.device_count()})")


def directions():
    g = torch.Generator().manual_seed(1)
    rnd = torch.randn(H, generator=g, dtype=torch.float64)
    out = {"random": (rnd / rnd.norm()).float()}
    if glpfiles.have(glpfiles.GLP49):
        _, dirs = load_control_vector(glpfiles.GLP49)
        d = dirs[LAYER].double()
        out["glp49_l30"] = (d / d.norm()).float()
    return out


def _unit(w, seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(w, generator=g, dtype=torch.float64)
    return (v / v.norm()).float()


def inputs(dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g, dtype=torch.float64)
    r = torch.randn(T, H, generator=g, dtype=torch.float64) * 4.0
    # massive activations: a few residual dims 400x the rest (as the model cards describe)
    for j in (7, 1111, 4095):
        r[:, j] *= 400.0
    return x.to(dtype).to(device), r.to(dtype).to(device)


class _RefModel(SteeredModelMixin, nn.Module):
    """The vLLM-path mixin with a hand-built core (no env, no file)."""

    def __init__(self, d, alpha, dtype, device):
        nn.Module.__init__(self)
        core = SteeringCore({LAYER: d.float().cpu()}, alpha,
                            "residual_stream_post_layer", 64, H)
        with torch.device(device):
            core.register_buffers(self, dtype)
        self._steer_core = core


class _Owner(nn.Module):
    def __init__(self, d, alpha, device, dtype=torch.float32):
        super().__init__()
        stack = torch.zeros(64, 1, H, dtype=dtype)
        stack[LAYER, 0] = d.to(dtype)
        self.register_buffer("_steer_stack", stack.to(device), persistent=False)
        self.register_buffer("_steer_alpha", torch.tensor(alpha, dtype=dtype, device=device),
                             persistent=False)


def ours(x, r, d, alpha, dtype=torch.float32):
    owner = _Owner(d, alpha, x.device, dtype)
    hook = make_post_layer_hook(owner, LAYER)
    xn, rn = hook(None, (), (x, r))
    return xn, rn


class GoldenAgainstFirstRelease(unittest.TestCase):
    """The refactor leaves the Qwen path bit-identical: the codec output
    equals the first release's steer_delta, byte for byte."""

    def test_codec_equals_first_release(self):
        W = 5120
        d = unit(W, 1)
        for dev in DEVICES:
            for dt in (torch.bfloat16, torch.float32):
                # 37 rows, and 1100 rows (two slices of the bounded torch path)
                for T in (0, 1, 37, 1100):
                    x, r = rows(T, W, dt, dev, seed=T)
                    for a in (0.0, 0.5, 1.0, 2.0):
                        owner = Owner(W, dev, a, {LAYER: d})
                        want = ref.steer_delta(x, r, owner._steer_stack[LAYER, 0], owner._steer_alpha)
                        got, rn = make_post_layer_hook(owner, LAYER, kernel="torch")(None, (), (x, r))
                        with self.subTest(dev=dev, dtype=dt, T=T, alpha=a):
                            self.assertTrue(same_bits(got, want))
                            self.assertIs(rn, r)
                        for rr in (r, None):
                            got = steer_rows(x, rr, owner._steer_stack[LAYER, 0], owner._steer_alpha)
                            want = ref.steer_delta(x, rr, owner._steer_stack[LAYER, 0],
                                                   owner._steer_alpha)
                            with self.subTest(dev=dev, dtype=dt, T=T, alpha=a, residual=rr is not None):
                                self.assertTrue(same_bits(got, want))

    def test_float64_path_unchanged(self):
        W = 5120
        d = unit(W, 2)
        x, r = rows(37, W, torch.float64, "cpu")
        for a in (0.0, 1.0, 2.0):
            al = torch.tensor(a)
            self.assertTrue(same_bits(wcore.steer_delta(x, r, d.double(), al),
                                      ref.steer_delta(x, r, d.double(), al)))

    def test_fused_interpreter_equals_first_release(self):
        """The fused kernel on CPU through Triton's interpreter (fp32, where
        the interpreter's final rounding is a no-op), at 5120 (one block)
        and 16384 (blocks), with and without a residual, 2-D and 3-D."""
        try:
            import triton  # noqa: F401
        except Exception:
            self.skipTest("triton is not installed")
        here = os.path.dirname(os.path.abspath(__file__))
        code = f"""
import sys, torch
sys.path[:0] = {[glpfiles.PLUGIN, here]!r}
from weightless_sglang.fused import steer_delta_fused, launch_shape
from test_core import ref
g = torch.Generator().manual_seed(11)
bad = 0
assert launch_shape(5120) == (8192, 1) and launch_shape(16384) == (8192, 2), (launch_shape(5120), launch_shape(16384))
for H, shape in ((5120, (7,)), (16384, (5,)), (16384, (2, 4)), (1000, (3,))):
    x = torch.randn(*shape, H, generator=g)
    r = torch.randn(*shape, H, generator=g) * 4.0
    r[..., ::97] *= 400.0
    d = torch.randn(H, generator=g, dtype=torch.float64)
    d = (d / d.norm()).float()
    for a in (0.0, 0.5, 1.0, 2.0):
        al = torch.tensor(a)
        for rr in (r, None):
            got, want = steer_delta_fused(x, rr, d, al), ref.steer_delta(x, rr, d, al)
            bad += got.shape != want.shape or not torch.equal(got.view(torch.int32), want.view(torch.int32))
print("mismatches", bad)
sys.exit(1 if bad else 0)
"""
        env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                           timeout=1200)
        self.assertEqual(p.returncode, 0, p.stdout[-2000:] + p.stderr[-4000:])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_golden_cuda_fused_equals_first_release(self):
        from weightless_sglang.fused import steer_delta_fused
        for W in (5120, 16384):
            d = unit(W, 3).cuda()
            for dt in (torch.bfloat16, torch.float16, torch.float32):
                for T in (0, 1, 37, 1100):
                    for residual in (True, False):
                        x, r = rows(T, W, dt, "cuda", seed=T, residual=residual)
                        for a in (0.0, 1.0, 2.0):
                            al = torch.tensor(a, device="cuda")
                            with self.subTest(W=W, dtype=dt, T=T, residual=residual, alpha=a):
                                self.assertTrue(same_bits(steer_delta_fused(x, r, d, al),
                                                          ref.steer_delta(x, r, d, al)))


class DiagDoesNotChangeBits(unittest.TestCase):
    def test_hook_with_and_without_diag(self):
        for dev in DEVICES:
            for W, residual in ((5120, True), (16384, False)):
                d, dn = unit(W, 4), unit(W, 5)
                x, r = rows(37, W, torch.bfloat16, dev, residual=residual)
                owner = Owner(W, dev, 2.0, {LAYER: d, NEXT: dn})
                kw = dict(residual_index=1 if residual else None)
                plain = make_post_layer_hook(owner, LAYER, next_layer_id=NEXT, **kw)
                with tempfile.TemporaryDirectory() as tmp:
                    diag = LayerDiag(tmp, [LAYER])
                    with_diag = make_post_layer_hook(owner, LAYER, next_layer_id=NEXT, diag=diag, **kw)
                    out = (x, r) if residual else (x, None)
                    a, b = plain(None, (), out)[0], with_diag(None, (), out)[0]
                    self.assertTrue(os.path.exists(diag.path))
                with self.subTest(dev=dev, W=W):
                    self.assertTrue(same_bits(a, b))

    def test_installed_model_with_and_without_diag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "v.gguf")
            layers = (10, 11, 30)
            m = good_meta(layers=layers, alpha="2.0")
            m["controlvector.model_hint"] = "qwen3_5"
            write_gguf(path, m, good_tensors(layers))
            x = torch.randn(9, 8).to(torch.bfloat16)
            outs = []
            for diag in ("0", "1"):
                env = {"WEIGHTLESS_STEER_PATH": path, "WEIGHTLESS_STEER_DIAG": diag,
                       "WEIGHTLESS_STEER_DIAG_DIR": tmp}
                with mock.patch.dict(os.environ, env, clear=False):
                    run = fakes.runner()
                    run.model.model.to(torch.bfloat16)
                    install_steering(run, source="env")
                    outs.append(run.model.model(x))
            self.assertTrue(same_bits(outs[0], outs[1]))
            self.assertTrue(any(f.startswith("weightless-diag-") for f in os.listdir(tmp)))


class Codec(unittest.TestCase):
    W = 64

    def owner(self, alpha=1.0):
        return Owner(self.W, "cpu", alpha, {LAYER: unit(self.W, 6)})

    def test_three_tuple_other_slots_pass_through(self):
        x, r = rows(5, self.W, torch.bfloat16, "cpu")
        topk = torch.arange(10)
        o = self.owner()
        out = make_post_layer_hook(o, LAYER, arity=3, hidden_index=0, residual_index=1)(
            None, (), (x, r, topk))
        self.assertEqual(len(out), 3)
        self.assertIs(out[1], r)
        self.assertIs(out[2], topk)
        self.assertTrue(same_bits(out[0], ref.steer_delta(x, r, o._steer_stack[LAYER, 0], o._steer_alpha)))

    def test_no_residual_slot(self):
        x, _ = rows(5, self.W, torch.bfloat16, "cpu", residual=False)
        o = self.owner()
        out = make_post_layer_hook(o, LAYER, arity=3, hidden_index=0, residual_index=None)(
            None, (), (x, None, "keep"))
        self.assertIsNone(out[1])
        self.assertEqual(out[2], "keep")
        self.assertTrue(same_bits(out[0], ref.steer_delta(x, None, o._steer_stack[LAYER, 0], o._steer_alpha)))

    def test_hidden_not_first(self):
        x, r = rows(5, self.W, torch.bfloat16, "cpu")
        o = self.owner()
        out = make_post_layer_hook(o, LAYER, arity=2, hidden_index=1, residual_index=0)(None, (), (r, x))
        self.assertIs(out[0], r)
        self.assertTrue(same_bits(out[1], ref.steer_delta(x, r, o._steer_stack[LAYER, 0], o._steer_alpha)))

    def test_wrong_arity_and_width_fail(self):
        x, r = rows(5, self.W, torch.bfloat16, "cpu")
        h = make_post_layer_hook(self.owner(), LAYER, arity=3)
        with self.assertRaisesRegex(RuntimeError, "3-tuple"):
            h(None, (), (x, r))
        h = make_post_layer_hook(self.owner(), LAYER)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            h(None, (), (x[:, :32], r[:, :32]))

    def test_per_stream_needs_3d(self):
        o = self.owner()
        site = make_site(o, LAYER, per_stream=True)
        x3 = torch.randn(5, 4, self.W)
        self.assertEqual(site(x3).shape, x3.shape)
        with self.assertRaisesRegex(RuntimeError, "streams"):
            site(torch.randn(5, self.W))
        # per stream = the same edit on each [W] row
        want = ref.steer_delta(x3.reshape(-1, self.W), None, o._steer_stack[LAYER, 0],
                               o._steer_alpha).reshape(x3.shape)
        self.assertTrue(same_bits(site(x3), want))


class Refusals(unittest.TestCase):
    W = 64

    def site(self):
        return make_site(Owner(self.W, "cpu", 1.0, {LAYER: unit(self.W, 7)}), LAYER)

    def test_partial_sum_marker(self):
        x = torch.randn(3, self.W)
        x._sglang_needs_allreduce_fusion = True
        with self.assertRaisesRegex(RuntimeError, "_sglang_needs_allreduce_fusion"):
            self.site()(x)
        x._sglang_needs_allreduce_fusion = False
        self.site()(x)  # the flag cleared (the next layer consumed it): fine

    def test_deferred_moe_handoff(self):
        Handoff = type("MoeFinalizeHandoff", (), {})
        with self.assertRaisesRegex(RuntimeError, "MoeFinalizeHandoff"):
            self.site()(Handoff())


class SlotGuard(unittest.TestCase):
    """The slot check at a steered layer's first forward (hook.check_slot):
    a slot that should hold the hidden or the residual stream but holds a
    bool, None, an integer tensor or a stream of another width fails the
    boot with a SiteError that names the row, the layer, the slot and what
    was found. Right slots give the same bits as before, and the check runs
    once per layer only."""

    W = 64

    def hook(self, **kw):
        owner = Owner(self.W, "cpu", 1.0, {LAYER: unit(self.W, 8)})
        return owner, make_post_layer_hook(owner, LAYER, row="TestRow", **kw)

    def want(self, owner, x, r):
        return ref.steer_delta(x, r, owner._steer_stack[LAYER, 0], owner._steer_alpha)

    def test_bool_or_none_in_the_residual_slot(self):
        x, _ = rows(5, self.W, torch.bfloat16, "cpu")
        _, h = self.hook()
        with self.assertRaisesRegex(wcore.SiteError, r"row TestRow, decoder layer 30: slot 1 should "
                                                  r"hold the residual stream, expected "
                                                  r"\[\.\.\., 64\] \(a floating-point tensor\), "
                                                  r"found a bool"):
            h(None, (), (x, True))
        _, h = self.hook()
        with self.assertRaisesRegex(wcore.SiteError, "slot 1 should hold the residual stream.*found None"):
            h(None, (), (x, None))
        # a row that declares the residual slot may be None (the whole stream in hidden)
        owner, h = self.hook(residual_may_be_none=True)
        self.assertTrue(same_bits(h(None, (), (x, None))[0], self.want(owner, x, None)))

    def test_integer_tensor_in_a_slot(self):
        x, r = rows(5, self.W, torch.bfloat16, "cpu")
        topk = torch.zeros(5, self.W, dtype=torch.int32)
        # a three-tuple row that names the index slot as the residual
        _, h = self.hook(arity=3, hidden_index=0, residual_index=2)
        with self.assertRaisesRegex(wcore.SiteError, r"slot 2 should hold the residual stream.*found "
                                                  r"a torch\.int32 tensor of shape \(5, 64\)"):
            h(None, (), (x, r, topk))
        _, h = self.hook()
        with self.assertRaisesRegex(wcore.SiteError, r"slot 0 should hold the hidden stream.*found "
                                                  r"a torch\.int64 tensor"):
            h(None, (), (torch.zeros(5, self.W, dtype=torch.int64), r))
        _, h = self.hook()
        with self.assertRaisesRegex(wcore.SiteError, r"decoder layer 30 output is bool .* in slot 0 "
                                                  r"of row TestRow"):
            h(None, (), (False, r))

    def test_fp8_tensor_in_a_slot(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("this torch has no float8_e4m3fn")
        x, r = rows(5, self.W, torch.bfloat16, "cpu")
        _, h = self.hook()
        with self.assertRaisesRegex(wcore.SiteError, r"slot 1 should hold the residual stream.*found "
                                                  r"a torch\.float8_e4m3fn tensor of shape \(5, 64\)"):
            h(None, (), (x, r.to(torch.float8_e4m3fn)))
        _, h = self.hook()
        with self.assertRaisesRegex(wcore.SiteError, r"slot 0 should hold the hidden stream.*found "
                                                  r"a torch\.float8_e4m3fn tensor"):
            h(None, (), (x.to(torch.float8_e4m3fn), r))

    def test_hidden_slot_of_the_wrong_width(self):
        x, r = rows(5, self.W, torch.bfloat16, "cpu")
        _, h = self.hook()
        with self.assertRaisesRegex(wcore.SiteError, r"row TestRow, decoder layer 30: slot 0 should "
                                                  r"hold the hidden stream, expected "
                                                  r"\[\.\.\., 64\] \(a floating-point tensor\), "
                                                  r"found a torch\.bfloat16 tensor of shape "
                                                  r"\(5, 32\)"):
            h(None, (), (x[:, :32], r[:, :32]))
        # a per-stream row ([T, streams, W]) handed one flat stream
        _, h = self.hook(residual_index=None, per_stream=True)
        with self.assertRaisesRegex(wcore.SiteError, r"slot 0 should hold the hidden stream, expected "
                                                  r"\[T, streams, 64\] \(a floating-point tensor\), "
                                                  r"found a torch\.bfloat16 tensor of shape "
                                                  r"\(5, 64\)"):
            h(None, (), (x, None))

    def test_right_slots_same_bits_and_checked_once(self):
        x, r = rows(37, self.W, torch.bfloat16, "cpu")
        owner, h = self.hook()
        with mock.patch.object(wcore, "check_slot", wraps=wcore.check_slot) as spy:
            got, rn = h(None, (), (x, r))
            self.assertEqual(spy.call_count, 2)  # hidden and residual, at the first forward
            self.assertTrue(same_bits(got, self.want(owner, x, r)))
            self.assertIs(rn, r)
            got2, _ = h(None, (), (x, r))
            self.assertEqual(spy.call_count, 2)  # later forwards skip it
            self.assertTrue(same_bits(got2, got))

    def test_only_kimi_declares_a_residual_slot_that_may_be_none(self):
        self.assertEqual(sorted(k for k, v in ARCH.items() if v.residual_may_be_none),
                         ["KimiK3ForConditionalGeneration", "KimiK3LinearForCausalLM"])


class SlotGuardOnTheFakes(unittest.TestCase):
    """The same check through install_steering on the fake qwen3_5 model:
    one steered layer's output slot is broken after the install."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = os.path.join(self.tmp.name, "v.gguf")
        m = good_meta(layers=(10, 11, 30), alpha="1.0")
        m["controlvector.model_hint"] = "qwen3_5"
        write_gguf(path, m, good_tensors((10, 11, 30)))
        p = mock.patch.dict(os.environ, {"WEIGHTLESS_STEER_PATH": path}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_") and k != "WEIGHTLESS_STEER_PATH"]:
            os.environ.pop(k)

    def installed(self):
        run = fakes.runner()
        run.model.model.to(torch.bfloat16)
        install_steering(run, source="env")
        bb = run.model.model
        steered = [i for i in range(len(bb.layers)) if bool(bb._steer_stack[i, 0].abs().sum() > 0)]
        self.assertEqual(len(steered), 3)
        return bb, steered

    def break_layer(self, bb, i, fn):
        orig = bb.layers[i].forward
        bb.layers[i].forward = lambda *a, **k: fn(*orig(*a, **k))

    def test_bool_in_the_residual_slot(self):
        bb, steered = self.installed()
        k = steered[0]
        self.break_layer(bb, k, lambda h, r: (h, True))
        with self.assertRaisesRegex(wcore.SiteError, f"row Qwen3_5ForConditionalGeneration, decoder "
                                                  f"layer {k}: slot 1 should hold the residual "
                                                  f"stream.*found a bool"):
            bb(torch.randn(9, 8).to(torch.bfloat16))

    def test_hidden_slot_of_the_wrong_width(self):
        bb, steered = self.installed()
        k = steered[1]
        self.break_layer(bb, k, lambda h, r: (h[:, :4], r))
        with self.assertRaisesRegex(wcore.SiteError, f"row Qwen3_5ForConditionalGeneration, decoder "
                                                  rf"layer {k}: slot 0 should hold the hidden "
                                                  rf"stream, expected \[\.\.\., 8\] \(a "
                                                  rf"floating-point tensor\), found a torch\.bfloat16 "
                                                  rf"tensor of shape \(9, 4\)"):
            bb(torch.randn(9, 8).to(torch.bfloat16))

    def test_right_slots_give_the_same_bits_as_the_edit_by_hand(self):
        bb, steered = self.installed()
        x = torch.randn(9, 8, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
        outs = [bb(x), bb(x)]  # the first forward checks the slots, the second does not
        stock = fakes.runner().model.model.to(torch.bfloat16)
        h, r = x, None
        for i in range(len(stock.layers)):
            h, r = stock.layers[i](hidden_states=h, residual=r)
            if i in steered:
                h = ref.steer_delta(h, r, bb._steer_stack[i, 0], bb._steer_alpha)
        for out in outs:
            self.assertTrue(same_bits(out, h + r))


class Rows(unittest.TestCase):
    """install_steering against declarative rows (fake models)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "v.gguf")
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        self.arch = mock.patch.dict(ARCH, {}, clear=False)
        self.arch.start()
        self.addCleanup(self.arch.stop)

    def write(self, layers=(10, 11, 30), width=8, **extra):
        m = good_meta(layers=layers, alpha="1.0")
        m["controlvector.model_hint"] = "qwen3_5"
        m.update(extra)
        t = good_tensors(layers)
        if width != 8:
            import numpy as np
            t = {k: (np.eye(width, dtype=np.float32)[i % width], 0) for i, k in enumerate(t)}
        write_gguf(self.path, m, t)
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        return self.path

    def fake_row(self, **kw):
        base = dict(ARCH["Qwen3_5ForConditionalGeneration"].__dict__)
        base.update(kw)
        row = ArchRow(**base)
        ARCH["Qwen3_5ForConditionalGeneration"] = row
        return row

    def test_two_batch_overlap_refused(self):
        self.write()
        run = fakes.runner()
        run.server_args = types.SimpleNamespace(enable_two_batch_overlap=True)
        with self.assertRaisesRegex(RuntimeError, "two-batch overlap"):
            install_steering(run, source="env")
        run.server_args.enable_two_batch_overlap = False
        self.assertIsNotNone(install_steering(run, source="env"))

    def test_tp_refuse_row(self):
        self.write()
        self.fake_row(tp="refuse")
        run = fakes.runner()
        run.tp_size = 2
        with self.assertRaisesRegex(RuntimeError, "TP=2"):
            install_steering(run, source="env")
        run.tp_size = 1
        self.assertIsNotNone(install_steering(run, source="env"))

    def test_tp_ok_row_still_installs(self):
        self.write()
        run = fakes.runner()
        run.tp_size = 2
        self.assertIsNotNone(install_steering(run, source="env"))

    def test_hook_point_not_served(self):
        self.write(**{"glp.hook_point": "ffn_out_pre_residual"})
        with self.assertRaisesRegex(RuntimeError, "glp.hook_point='ffn_out_pre_residual'"):
            install_steering(fakes.runner(), source="env")

    def test_hook_point_passed_to_core(self):
        """A row that serves another hook point: the file's hook point
        reaches SteeringCore (which re-checks it against the file)."""
        self.write(**{"glp.hook_point": "ffn_out_pre_residual"})
        self.fake_row(hooks=frozenset({"ffn_out_pre_residual"}))
        rec = install_steering(fakes.runner(), source="env")
        self.assertEqual(rec["hook_point"], "ffn_out_pre_residual")

    def test_structure(self):
        # a looped model's file on a normal row
        self.write(**{"glp.structure": "per-execution-step"})
        with self.assertRaisesRegex(RuntimeError, "per-execution-step"):
            install_steering(fakes.runner(), source="env")
        # missing, per-layer and free text are fine on a normal row
        for s in (None, "per-layer", "77 per-layer vectors (L1-77; layer 0 inexpressible)"):
            extra = {} if s is None else {"glp.structure": s}
            self.write(**extra)
            with self.subTest(structure=s):
                self.assertIsNotNone(install_steering(fakes.runner(), source="env"))
        # a looped row needs per-execution-step
        self.fake_row(exec_id="looped")
        for s in (None, "per-layer", "free text"):
            extra = {} if s is None else {"glp.structure": s}
            self.write(**extra)
            with self.subTest(looped=s), self.assertRaisesRegex(RuntimeError, "looped"):
                install_steering(fakes.runner(), source="env")
        self.write(**{"glp.structure": "per-execution-step"})
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            install_steering(fakes.runner(), source="env")

    def test_row_hint_accepted(self):
        self.write(**{"controlvector.model_hint": "other_name"})
        with self.assertRaisesRegex(RuntimeError, "model_hint"):
            install_steering(fakes.runner(), source="env")
        self.fake_row(hint=frozenset({"other_name"}))
        self.assertIsNotNone(install_steering(fakes.runner(), source="env"))

    def test_backbone_candidates_and_absent_backbone(self):
        self.write()
        self.fake_row(backbone=("language_model.model", "model"))
        rec = install_steering(fakes.runner(), source="env")
        self.assertEqual(rec["hooked_layers"], [10, 11, 30])
        run = fakes.runner()
        run.model.model = None  # e.g. an encoder-only rank
        self.assertIsNone(install_steering(run, source="env"))
        self.fake_row(backbone=("nothing_here",))
        with self.assertRaisesRegex(RuntimeError, "backbone paths"):
            install_steering(fakes.runner(), source="env")

    def test_width_from_row(self):
        self.write(width=32)
        self.fake_row(width=lambda cfg: cfg.hidden_size * 4)  # a 4-stream site
        rec = install_steering(fakes.runner(width=8), source="env")
        self.assertEqual(rec["width"], 32)
        self.assertEqual(rec["hidden_size"], 8)
        self.assertEqual(tuple(fakes.runner().model.model.layers[0].w.shape), (8,))

    def test_unknown_special_and_forward_wrap_fail_closed(self):
        self.write()
        self.fake_row(install="special:nope")
        with self.assertRaisesRegex(RuntimeError, "special:nope"):
            install_steering(fakes.runner(), source="env")
        # forward_wrap is implemented: it wraps the instance
        # forward instead of registering a hook (test_archs/test_nemotron_h.py)
        self.fake_row(install="forward_wrap")
        run = fakes.runner()
        rec = install_steering(run, source="env")
        bb = run.model.model
        self.assertEqual(rec["install"], "forward_wrap")
        self.assertEqual(sorted(i for i, l in enumerate(bb.layers) if "forward" in vars(l)),
                         [10, 11, 30])
        self.assertFalse(any(l._forward_hooks for l in bb.layers))
        self.fake_row(install="nope")
        with self.assertRaisesRegex(RuntimeError, "unknown install mode 'nope'"):
            install_steering(fakes.runner(), source="env")

    def test_manifest_fields(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = fakes.runner(start=0, end=20)
        run.pp_rank, run.pp_size = 0, 3
        install_steering(run, source="env")
        m = json.load(open(os.path.join(self.tmp.name, f"weightless-manifest-{os.getpid()}.json")))
        for k in ("kernel", "kernel_reason", "row", "install", "site", "width", "pp_rank",
                  "local_layer_ids", "fired_at_capture", "file_sha256", "alpha"):
            self.assertIn(k, m)
        self.assertEqual(m["row"], "Qwen3_5ForConditionalGeneration")
        self.assertEqual(m["install"], "hook")
        self.assertEqual(m["width"], 8)
        self.assertEqual(m["pp_rank"], 0)
        self.assertEqual(m["local_layer_ids"], [10, 11])
        self.assertEqual(m["fired_at_capture"], [])


class _BypassBackbone(fakes.Backbone):
    """A loop that reaches some layers without calling them (as a
    two-batch-overlap path or a changed call style would)."""

    def forward(self, x, skip=()):
        h, r = x, None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            h, r = (layer.forward if i in skip else layer)(hidden_states=h, residual=r)
        return h + r


class FiredCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "v.gguf")
        m = good_meta(layers=(10, 11, 30), alpha="1.0")
        m["controlvector.model_hint"] = "qwen3_5"
        write_gguf(self.path, m, good_tensors((10, 11, 30)))
        p = mock.patch.dict(os.environ, {"WEIGHTLESS_STEER_PATH": self.path}, clear=False)
        p.start()
        self.addCleanup(p.stop)

    def runner(self):
        bb = _BypassBackbone(64, 8)
        return types.SimpleNamespace(
            model=fakes.Qwen3_5ForConditionalGeneration(bb), is_draft_worker=False, tp_rank=0,
            tp_size=1, model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(model_type="qwen3_5")))

    def test_all_fire(self):
        run = self.runner()
        install_steering(run, source="env")
        run.model.model(torch.randn(3, 8))
        run.model.model.forward(torch.randn(3, 8))  # a direct .forward call is checked too
        self.assertEqual(run._weightless_steer_check.forwards, 2)

    def test_bypass_fails(self):
        run = self.runner()
        install_steering(run, source="env")
        with self.assertRaisesRegex(RuntimeError, r"did not run in this forward \(layers \[11\]\)"):
            run.model.model(torch.randn(3, 8), skip=(11,))
        with self.assertRaisesRegex(RuntimeError, "did not run"):
            run.model.model.forward(torch.randn(3, 8), skip=(10, 30))
        run.model.model(torch.randn(3, 8))  # the next full forward is fine again

    def test_unsteered_layers_may_be_bypassed(self):
        run = self.runner()
        install_steering(run, source="env")
        run.model.model(torch.randn(3, 8), skip=(12, 13))

    def test_signature_kept_and_undo(self):
        run = self.runner()
        bb = run.model.model
        before = inspect.signature(bb.forward)
        install_steering(run, source="env")
        self.assertIn("forward", bb.__dict__)
        self.assertEqual(inspect.signature(bb.forward), before)
        run._weightless_steer_undo_check()
        self.assertNotIn("forward", bb.__dict__)

    def test_capture_is_recorded_in_manifest(self):
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = self.runner()
        install_steering(run, source="env")
        with mock.patch.object(wcore, "_sglang_capture_mode", return_value=True):
            for bs in (1, 2, 4):
                run.model.model(torch.randn(bs, 8))
            with self.assertRaises(RuntimeError):
                run.model.model(torch.randn(8, 8), skip=(30,))
        run.model.model(torch.randn(5, 8))  # not a capture: not recorded
        m = json.load(open(os.path.join(self.tmp.name, f"weightless-manifest-{os.getpid()}.json")))
        caps = m["fired_at_capture"]
        self.assertEqual([c["tokens"] for c in caps], [1, 2, 4, 8])
        self.assertEqual([c["fired"] for c in caps], [3, 3, 3, 2])
        self.assertTrue(all(c["expected"] == 3 for c in caps))


class SpecialInstall(unittest.TestCase):
    """A special:<name> row: the handler wraps an instance attribute and
    calls the site with r=None (the shape of the GLM mHC site)."""

    def test_handler_gets_a_working_site(self):
        class Combine(nn.Module):
            def forward(self, x):
                return torch.cat([x, x], dim=-1)  # a 2-stream widening

        class Layer(nn.Module):
            def __init__(self, i):
                super().__init__()
                self.i = i
                self.comb = Combine()

            def forward(self, hidden_states=None, residual=None, **kw):
                s = self.comb(hidden_states if hidden_states.shape[-1] == 8 else hidden_states[..., :8])
                return s, None, "topk"

        StreamLayer = type("StreamDecoderLayer", (Layer,), {})

        class BB(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(hidden_size=8, model_type="qwen3_5")
                self.layers = nn.ModuleList(StreamLayer(i) for i in range(40))
                self.start_layer, self.end_layer = 0, 40

            def forward(self, x):
                h = x
                for layer in self.layers:
                    h, _, _ = layer(hidden_states=h)
                return h

        calls = []

        def handler(layer, site, *, layer_id, backbone, row):
            orig = layer.comb.forward

            def wrapped(*a, **k):
                calls.append(layer_id)
                return site(orig(*a, **k), None)

            layer.comb.forward = wrapped
            return lambda: layer.comb.__dict__.pop("forward", None)

        with mock.patch.dict(archs_base.SPECIAL, {}, clear=False), \
                mock.patch.dict(ARCH, {}, clear=False), \
                tempfile.TemporaryDirectory() as tmp:
            archs_base.register_special("test_stream", site="comb")(handler)
            ModelCls = type("StreamModel", (nn.Module,), {"__init__": lambda self, bb: (
                nn.Module.__init__(self), setattr(self, "model", bb))[0]})
            ARCH["StreamModel"] = ArchRow(
                backbone=("model",), layers=frozenset({"StreamDecoderLayer"}),
                hooks=frozenset({HOOK_POINT}), width=lambda c: 2 * c.hidden_size,
                arity=3, hidden_index=0, residual_index=None, install="special:test_stream",
                tp="refuse", hint=frozenset({"qwen3_5"}))
            path = os.path.join(tmp, "v.gguf")
            import numpy as np
            layers = (5, 6, 39)
            m = good_meta(layers=layers, alpha="2.0")
            m["controlvector.model_hint"] = "qwen3_5"
            write_gguf(path, m, {f"direction.{i}": (np.eye(16, dtype=np.float32)[i % 16], 0)
                                 for i in layers})
            run = types.SimpleNamespace(model=ModelCls(BB()), is_draft_worker=False, tp_rank=0,
                                        tp_size=1, model_config=None)
            with mock.patch.dict(os.environ, {"WEIGHTLESS_STEER_PATH": path}, clear=False):
                rec = install_steering(run, source="env")
            self.assertEqual(rec["site"], "comb")
            self.assertEqual(rec["width"], 16)
            self.assertEqual(rec["install"], "special:test_stream")
            x = torch.ones(3, 8)
            out = run.model.model(x)
            self.assertEqual(sorted(set(calls)), [5, 6, 39])
            # the last layer (39) is steered: its output's component along e_(39 % 16) is reflected
            self.assertTrue(torch.all(out[:, 39 % 16] < 0))
            with self.assertRaisesRegex(RuntimeError, "did not run"):
                run.model.model.layers[6].comb.__dict__.pop("forward")
                run.model.model(x)


class BoundedMemory(unittest.TestCase):
    """The torch path's temporaries, measured from what it allocates (on the
    meta device: no memory is used), stay under the 256 MB budget."""

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.bytes = 0

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            rets = out if isinstance(out, (tuple, list)) else (out,)
            for ret, spec in zip(rets, func._schema.returns):
                if isinstance(ret, torch.Tensor) and spec.alias_info is None:
                    self.bytes += ret.numel() * ret.element_size()
            return out

    def test_chunk_rows(self):
        self.assertEqual(chunk_rows(5120), 1024)
        self.assertEqual(chunk_rows(16384), 512)

    def test_temporaries_under_budget(self):
        T = 8192
        for W in (5120, 16384):
            N = chunk_rows(W)
            for residual in (True, False):
                x = torch.empty(T, W, dtype=torch.bfloat16, device="meta")
                r = torch.empty(T, W, dtype=torch.bfloat16, device="meta") if residual else None
                d = torch.empty(W, dtype=torch.float32, device="meta")
                a = torch.empty((), dtype=torch.float32, device="meta")
                with self.Count() as one:
                    wcore.steer_delta(x[:N], None if r is None else r[:N], d, a)
                with self.Count() as full:
                    out = steer_rows(x, r, d, a)
                slices = -(-T // N)
                per_slice = (full.bytes - out.numel() * out.element_size()) / slices
                with self.subTest(W=W, residual=residual):
                    self.assertEqual(one.bytes, per_slice)
                    self.assertLessEqual(per_slice, wcore.TEMP_BUDGET_BYTES)
                    # the per-element constant used to size the slices covers it
                    # (plus a few [N, 1] per-row scalars)
                    self.assertLessEqual(per_slice, wcore.TEMP_BYTES_PER_ELEM * W * N + 64 * N)
                    print(f"  torch path W={W} residual={residual}: {N} rows per slice, "
                          f"{per_slice / 1e6:.1f} MB of temporaries per slice")


class LayerMap(unittest.TestCase):
    """The alpha-aware layer-map statistics (LayerDiag and layer_verdict)."""

    def run_hook(self, W, residual, alpha, dev, cos=0.9, mode="edit"):
        d = unit(W, 8)
        e = unit(W, 9)
        e = e - (e.double() @ d.double()).float() * d
        e = e / e.norm()
        dn = (cos * d + (1 - cos * cos) ** 0.5 * e).float()
        dn = (dn / dn.norm()).float()
        x, r = rows(37, W, torch.bfloat16, dev, seed=W + int(alpha * 10), residual=residual)
        owner = Owner(W, dev, alpha, {LAYER: d, NEXT: dn})
        with tempfile.TemporaryDirectory() as tmp:
            diag = LayerDiag(tmp, [LAYER])
            a = owner._steer_alpha
            if mode == "edit":
                make_site(owner, LAYER, next_layer_id=NEXT, diag=diag)(x, r)
            else:
                x_new = x.clone() if mode == "unedited" else wcore.steer_rows(x, r, owner._steer_stack[NEXT, 0], a)
                diag.record(LAYER, x, r, x_new, owner._steer_stack[LAYER, 0],
                            owner._steer_stack[NEXT, 0], alpha=a, next_layer_id=NEXT)
            got = json.load(open(diag.path))
        return got, got["layers"][str(LAYER)]

    def test_alphas(self):
        for dev in DEVICES:
            for W, residual in ((5120, True), (16384, False)):
                for alpha in (0.0, 1.0, 2.0):
                    got, st = self.run_hook(W, residual, alpha, dev)
                    floor = 2.0 ** -8 * st["sum_norm_xnew"]
                    ratio = st["sum_signed_post"] / st["sum_abs_pre"]
                    v = layer_verdict(st, alpha)
                    print(f"  diag {dev} W={W} alpha={alpha}: err/pre={st['sum_abs_err'] / st['sum_abs_pre']:.2e} "
                          f"floor/pre={floor / st['sum_abs_pre']:.2e} "
                          f"elementwise/pre={st['sum_round_bound'] / st['sum_abs_pre']:.2e} signed={ratio:+.4f} "
                          f"probe={v.get('probe_rel', 0):.3f} pred={v.get('probe_pred_rel', 0):.3f} {v['edit']}/{v['sign']}/{v['probe']}")
                    with self.subTest(dev=dev, W=W, alpha=alpha):
                        self.assertEqual(got["alpha"], alpha)
                        self.assertAlmostEqual(got["cos_next"][str(LAYER)], 0.9, places=5)
                        self.assertEqual(got["next_layer"][str(LAYER)], NEXT)
                        self.assertAlmostEqual(got["shift_scale"][str(LAYER)], alpha * (1 - 0.81) ** 0.5, places=5)
                        self.assertLessEqual(st["sum_abs_err"], floor)
                        # the elementwise rounding bound holds too, and is the tighter one
                        self.assertLessEqual(st["sum_abs_err"], st["sum_round_bound"] * (1 + 2 ** -7) + 1e-9)
                        self.assertLessEqual(st["sum_round_bound"], floor)
                        ve = layer_verdict(st, alpha, floor="elementwise")
                        self.assertNotIn("fail", (ve["edit"], ve["sign"], ve["probe"]))
                        if alpha and residual:
                            self.assertEqual((ve["edit"], ve["sign"], ve["probe"]), ("pass",) * 3)
                        self.assertLess(abs(ratio - (1 - alpha)), 0.01)
                        if alpha == 1.0:  # the Qwen rule is the alpha-1 case
                            self.assertAlmostEqual(st["sum_abs_err"], st["sum_abs_post"],
                                                   delta=1e-9 * st["sum_abs_pre"])
                        self.assertNotEqual(v["sign"], "fail")
                        self.assertNotEqual(v["edit"], "fail")
                        self.assertNotEqual(v["probe"], "fail")
                        if alpha == 0:
                            self.assertEqual(v["edit"], "not decisive")
                        else:
                            # a correct site: sum_abs_err_shift = alpha x sum_abs_pred_shift up to rounding
                            self.assertLessEqual(abs(st["sum_abs_err_shift"] - alpha * st["sum_abs_pred_shift"]),
                                                 floor)
                            # the edit is under 1e-2 x pre on both streams, so it
                            # decides whatever the floor (1e-2 first, the floor only after a miss)
                            self.assertEqual((v["edit"], v["sign"]), ("pass", "pass"))
                            if residual:  # plain stream, large values only in r: every check decides
                                self.assertEqual(v["edit"], "pass")
                                self.assertEqual(v["sign"], "pass")
                                self.assertEqual(v["probe"], "pass")

    def test_unedited_layer(self):
        for W, residual in ((5120, True), (16384, False)):
            for alpha in (0.0, 1.0, 2.0):
                _, st = self.run_hook(W, residual, alpha, "cpu", mode="unedited")
                with self.subTest(W=W, alpha=alpha):
                    self.assertAlmostEqual(st["sum_abs_err"] / st["sum_abs_pre"], alpha, delta=1e-9)
                    self.assertAlmostEqual(st["sum_signed_post"] / st["sum_abs_pre"], 1.0, delta=1e-9)
                    if alpha:
                        v = layer_verdict(st, alpha)
                        self.assertNotEqual(v["edit"], "pass")
                        self.assertNotEqual(v["sign"], "pass")
                        if residual:
                            self.assertEqual(v["edit"], "fail")
                            self.assertEqual(v["sign"], "fail")
                        ve = layer_verdict(st, alpha, floor="elementwise")
                        self.assertNotIn("pass", (ve["edit"], ve["sign"]))
                        if residual:
                            self.assertEqual((ve["edit"], ve["sign"]), ("fail", "fail"))

    def test_next_direction_applied(self):
        """The site applied d_next at this layer: the edit error is well
        above the floor and the shift probe sits at it."""
        for W, residual in ((5120, True), (16384, False)):
            for alpha in (1.0, 2.0):
                _, st = self.run_hook(W, residual, alpha, "cpu", cos=0.5, mode="shifted")
                _, ok = self.run_hook(W, residual, alpha, "cpu", cos=0.5)
                floor = 2.0 ** -8 * st["sum_norm_xnew"]
                v = layer_verdict(st, alpha)
                with self.subTest(W=W, alpha=alpha):
                    self.assertGreater(st["sum_abs_err"], 100 * ok["sum_abs_err"])
                    self.assertLessEqual(st["sum_abs_err_shift"], floor)
                    self.assertNotEqual(v["edit"], "pass")
                    self.assertNotEqual(v["probe"], "pass")
                    ve = layer_verdict(st, alpha, floor="elementwise")
                    self.assertNotIn("pass", (ve["edit"], ve["probe"]))
                    if residual:  # the floor is tight here: well above it
                        self.assertEqual((ve["edit"], ve["probe"]), ("fail", "fail"))
                        self.assertGreater(st["sum_abs_err"], 10 * floor)
                        self.assertEqual(v["edit"], "fail")
                        self.assertEqual(v["probe"], "fail")

    def test_probe_not_decisive_when_directions_nearly_agree(self):
        _, st = self.run_hook(5120, True, 2.0, "cpu", cos=0.99999)
        self.assertEqual(layer_verdict(st, 2.0)["probe"], "not decisive")

    def test_floor_rules(self):
        base = {"sum_abs_pre": 100.0, "sum_norm_xnew": 2.0 ** 8 * 3.0, "sum_abs_err": 5.0,
                "sum_signed_post": -97.0, "sum_abs_pre_next": 0.0}
        v = layer_verdict(base, 2.0)  # floor 3 > 1 (1e-2 x pre): judged on 2 x floor = 6
        self.assertTrue(v["on_floor"])
        self.assertEqual(v["edit"], "pass")
        self.assertAlmostEqual(v["sign_tol"], 0.06)
        self.assertEqual(v["sign"], "pass")
        v = layer_verdict(dict(base, sum_norm_xnew=2.0 ** 8 * 30.0), 2.0)  # 2 x floor = 60 > 0.25 x 2 x 100
        self.assertEqual(v["edit"], "not decisive")
        self.assertEqual(v["probe"], "none")


class VerdictOrder(unittest.TestCase):
    """layer_verdict follows the layer-map rule's order: 1e-2 x pre first,
    the bf16 floor only for a layer that misses it (and only while 2 x
    floor <= 0.25 x alpha x pre), then the sign, then the shift probe with
    its cosine caveat. Synthetic sums at alpha 2.0, pre = 100."""

    @staticmethod
    def sums(err_rel, floor_rel, signed_rel, pre=100.0):
        return {"sum_abs_pre": pre, "sum_norm_xnew": 2.0 ** 8 * floor_rel * pre,
                "sum_abs_err": err_rel * pre, "sum_signed_post": signed_rel * pre,
                "sum_round_bound": floor_rel * pre, "sum_abs_pre_next": 0.0}

    def test_correct_edit_passes_whatever_the_floor(self):
        # err/pre 0.005 is under 1e-2: decisive, even with the floor at
        # 0.30 x pre (2 x floor = 0.6 > 0.25 x 2.0 = 0.5, the cap)
        for floor_rel in (0.30, 0.10):
            for fl in ("norm", "elementwise"):
                v = layer_verdict(self.sums(0.005, floor_rel, -0.999), 2.0, floor=fl)
                with self.subTest(floor_rel=floor_rel, floor=fl):
                    self.assertEqual((v["edit"], v["sign"]), ("pass", "pass"))
                    self.assertFalse(v["on_floor"])
                    self.assertAlmostEqual(v["bound_rel"], 1e-2)
                    self.assertAlmostEqual(v["sign_tol"], 0.02)
                    self.assertAlmostEqual(v["floor_rel"], floor_rel)

    def test_unedited_layer_fails(self):
        # an unedited layer: err = alpha x pre, signed ratio +1
        for floor_rel in (0.30, 0.10, 1e-4):
            v = layer_verdict(self.sums(2.0, floor_rel, 1.0), 2.0)
            with self.subTest(floor_rel=floor_rel):
                self.assertEqual(v["edit"], "fail")
                self.assertNotEqual(v["sign"], "pass")
                if 2 * floor_rel <= 0.5:
                    self.assertEqual(v["sign"], "fail")

    def test_floor_fallback_only_after_a_miss(self):
        # misses 1e-2, floor 0.03 x pre: judged on 2 x floor = 0.06
        v = layer_verdict(self.sums(0.05, 0.03, -0.97), 2.0)
        self.assertTrue(v["on_floor"])
        self.assertEqual((v["edit"], v["sign"]), ("pass", "pass"))
        self.assertAlmostEqual(v["bound_rel"], 0.06)
        # within 2 x floor, but 2 x floor above the cap: cannot be told from unedited
        v = layer_verdict(self.sums(0.05, 0.30, -0.97), 2.0)
        self.assertEqual((v["edit"], v["sign"]), ("not decisive", "not decisive"))
        # misses 1e-2 with the floor under it: a real miss
        v = layer_verdict(self.sums(0.05, 0.001, -0.97), 2.0)
        self.assertFalse(v["on_floor"])
        self.assertEqual(v["edit"], "fail")
        # alpha 0: the exact edit is the identity, and a pass is never decisive
        v = layer_verdict(self.sums(0.0, 0.001, 1.0), 0.0)
        self.assertEqual((v["edit"], v["sign"]), ("not decisive", "not decisive"))

    def test_probe_cosine_caveat(self):
        st = dict(self.sums(0.005, 0.001, -0.999), sum_abs_pre_next=100.0,
                  sum_abs_pred_shift=40.0, sum_abs_err_shift=80.0)
        self.assertEqual(layer_verdict(st, 2.0)["probe"], "pass")
        self.assertEqual(layer_verdict(st, 2.0, cos_next=0.5)["probe"], "pass")
        v = layer_verdict(st, 2.0, cos_next=-0.995)
        self.assertEqual(v["probe"], "not decisive")
        self.assertEqual(v["cos_next"], -0.995)
        self.assertEqual(layer_verdict(dict(st, sum_abs_err_shift=0.01), 2.0)["probe"], "fail")


class Float64Equivalence(unittest.TestCase):
    """Check 1: the delta form equals the reference h' - r in float64."""

    def test_matches_vllm_path_and_hf_path(self):
        for dev in DEVICES:
            for name, d in directions().items():
                for a in ALPHAS:
                    x, r = inputs(torch.float64, dev)
                    xn, _ = ours(x, r, d.double(), a, torch.float64)
                    ref = _RefModel(d.double(), a, torch.float64, dev)
                    xr = ref._steer_post_layer(LAYER, x, r)
                    scale = (x + r).abs().max()
                    rel_v = float((xn - xr).abs().max() / scale)
                    D = d.double().to(dev).unsqueeze(0)
                    hf_hook = hf_ref._make_hook(D, torch.tensor([a], dtype=torch.float64))
                    t = hf_hook(None, (), (x + r,))[0]
                    rel_h = float(((xn + r) - t).abs().max() / scale)
                    with self.subTest(dev=dev, d=name, alpha=a):
                        self.assertLess(rel_v, 1e-12)
                        self.assertLess(rel_h, 1e-12)


class Bf16Rounding(unittest.TestCase):
    """Check 2: in bf16 the folded stream is no worse than the reference's."""

    def test_error_vs_float64_truth(self):
        eps = 2.0 ** -8  # bf16 unit roundoff
        for dev in DEVICES:
            for name, d in directions().items():
                for a in ALPHAS[1:]:
                    x, r = inputs(torch.bfloat16, dev)
                    x64, r64, d64 = x.double(), r.double(), d.double().to(dev)
                    h64 = x64 + r64
                    truth = h64 - a * (h64 @ d64).unsqueeze(-1) * d64
                    xn, _ = ours(x, r, d, a)
                    err_ours = (xn.float() + r.float()).double() - truth
                    ref = _RefModel(d, a, torch.bfloat16, dev)
                    xr = ref._steer_post_layer(LAYER, x, r)
                    err_ref = (xr.float() + r.float()).double() - truth
                    # bound: one bf16 rounding of x', the fp32 error in c, and
                    # the fp32 roundings of x - a c d and of the next fold x' + r
                    xprime64 = x64 - a * (h64 @ d64).unsqueeze(-1) * d64
                    dc = H * 2.0 ** -24 * (h64.abs() @ d64.abs())
                    bound = (eps * xprime64.abs() + a * dc.unsqueeze(-1) * d64.abs()
                             + 2.0 ** -23 * (xprime64.abs() + r64.abs() + x64.abs()) + 1e-30)
                    ratio = float((err_ours.abs() / bound).max())
                    mo, mr = float(err_ours.abs().max()), float(err_ref.abs().max())
                    print(f"  bf16 {dev} {name} alpha={a}: max|err| ours={mo:.3e} "
                          f"reference={mr:.3e} ours/bound={ratio:.3f}")
                    with self.subTest(dev=dev, d=name, alpha=a):
                        self.assertLessEqual(mo, mr)
                        self.assertLessEqual(ratio, 1.0)


class Semantics(unittest.TestCase):
    def test_removal_and_reflection(self):
        """Check 3: alpha=1 removes the component, alpha=2 reflects it."""
        for dev in DEVICES:
            for name, d in directions().items():
                x, r = inputs(torch.float64, dev)
                d64 = d.double().to(dev)
                pre = (x + r) @ d64
                xn, _ = ours(x, r, d.double(), 1.0, torch.float64)
                post = (xn + r) @ d64
                self.assertLessEqual(float(post.abs().max()), 1e-3 * float(pre.abs().max()))
                self.assertTrue(bool((post.abs() <= 1e-3 * pre.abs() + 1e-9).all()))
                xn2, _ = ours(x, r, d.double(), 2.0, torch.float64)
                post2 = (xn2 + r) @ d64
                # d is the f32 direction (|d|^2 = 1 +- 1e-8), so reflection is exact to ~1e-8 relative
                torch.testing.assert_close(post2, -pre, rtol=1e-6, atol=1e-9)

    def test_alpha_zero_is_identity_in_bf16(self):
        """Check 4: alpha=0 leaves x value- and bit-identical."""
        for dev in DEVICES:
            for name, d in directions().items():
                x, r = inputs(torch.bfloat16, dev)
                xn, _ = ours(x, r, d, 0.0)
                self.assertTrue(torch.equal(xn, x))
                self.assertTrue(torch.equal(xn.view(torch.int16), x.view(torch.int16)))

    def test_residual_untouched_and_dtype_kept(self):
        """Check 5."""
        for dev in DEVICES:
            d = directions()["random"]
            for dt in (torch.bfloat16, torch.float16, torch.float32):
                x, r = inputs(dt, dev)
                r0 = r.clone()
                xn, rn = ours(x, r, d, 1.0)
                self.assertIs(rn, r)
                self.assertTrue(torch.equal(r, r0))
                self.assertEqual(xn.dtype, dt)
                self.assertEqual(xn.shape, x.shape)

    def test_unreduced_output_is_reduced_first(self):
        """Check 6: a TP>1 deferred all-reduce output (identity group)."""
        try:
            from sglang.srt.layers.communicator import UnreducedOutput
        except Exception as e:  # pragma: no cover
            self.skipTest(f"sglang not importable: {e}")

        class _Group:
            calls = 0

            def all_reduce(self, t):
                _Group.calls += 1
                return t

        d = directions()["random"]
        x, r = inputs(torch.bfloat16, "cpu")
        want, _ = ours(x, r, d, 1.0)
        owner = _Owner(d, 1.0, "cpu")
        hook = make_post_layer_hook(owner, LAYER)
        got, _ = hook(None, (), (UnreducedOutput(partial=x, group=_Group()), r))
        self.assertEqual(_Group.calls, 1)
        self.assertTrue(torch.equal(got, want))

    def test_unknown_output_fails_closed(self):
        d = directions()["random"]
        owner = _Owner(d, 1.0, "cpu")
        hook = make_post_layer_hook(owner, LAYER)
        x, r = inputs(torch.bfloat16, "cpu")
        with self.assertRaises(RuntimeError):
            hook(None, (), x)
        with self.assertRaises(RuntimeError):
            hook(None, (), (object(), r))
        with self.assertRaises(RuntimeError):
            hook(None, (), (x, r[:, :100]))

    def test_padded_rows_and_empty_batch(self):
        d = directions()["random"]
        owner = _Owner(d, 1.0, "cpu")
        hook = make_post_layer_hook(owner, LAYER)
        e = torch.zeros(0, H, dtype=torch.bfloat16)
        xn, _ = hook(None, (), (e, e))
        self.assertEqual(xn.shape, (0, H))


class FixedPointDot(unittest.TestCase):
    """The dot product used by both paths: exact to fp32 and independent of
    the order of the additions."""

    def test_order_does_not_change_the_bits(self):
        from weightless_sglang.core import dot_fixed
        x, r = inputs(torch.bfloat16, "cpu")
        h = x.float() + r.float()
        d = directions()["random"]
        c = dot_fixed(h, d)
        for seed in (1, 2, 3):
            perm = torch.randperm(H, generator=torch.Generator().manual_seed(seed))
            self.assertTrue(torch.equal(dot_fixed(h[:, perm], d[perm]).view(torch.int32),
                                        c.view(torch.int32)))
        # one row at a time gives the same bits as the whole batch
        for t in (0, 5, T - 1):
            self.assertTrue(torch.equal(dot_fixed(h[t:t + 1], d), c[t:t + 1]))

    def test_close_to_float64(self):
        from weightless_sglang.core import dot_fixed
        x, r = inputs(torch.bfloat16, "cpu")
        h = x.float() + r.float()
        for name, d in directions().items():
            truth = h.double() @ d.double()
            scale = (h.double().abs() @ d.double().abs())
            err = float(((dot_fixed(h, d).double() - truth).abs() / scale).max())
            with self.subTest(d=name):
                self.assertLess(err, 2.0 ** -20)
        z = torch.zeros(3, H)
        self.assertTrue(torch.equal(dot_fixed(z, directions()["random"]), torch.zeros(3)))


class FusedKernelInterpreter(unittest.TestCase):
    """The fused kernel's math on CPU, through Triton's interpreter
    (TRITON_INTERPRET=1, in a child process). fp32 inputs, so the final
    rounding (which the interpreter does not do like the GPU) is a no-op."""

    def test_same_bits_as_torch_path(self):
        import subprocess
        import sys
        try:
            import triton  # noqa: F401
        except Exception:
            self.skipTest("triton is not installed")
        code = f"""
import sys, torch
sys.path[:0] = {[glpfiles.PLUGIN]!r}
from weightless_sglang.fused import steer_delta_fused
from weightless_sglang.core import steer_delta
g = torch.Generator().manual_seed(7)
bad = 0
for H in (5120, 1000):
    x = torch.randn(9, H, generator=g)
    r = torch.randn(9, H, generator=g) * 4.0
    r[:, ::97] *= 400.0
    d = torch.randn(H, generator=g, dtype=torch.float64)
    d = (d / d.norm()).float()
    for a in (0.0, 0.5, 1.0, 2.0):
        al = torch.tensor(a)
        for rr in (r, None):
            got, want = steer_delta_fused(x, rr, d, al), steer_delta(x, rr, d, al)
            bad += not torch.equal(got.view(torch.int32), want.view(torch.int32))
print("mismatches", bad)
sys.exit(1 if bad else 0)
"""
        env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                           timeout=600)
        self.assertEqual(p.returncode, 0, p.stdout[-2000:] + p.stderr[-4000:])


@unittest.skipUnless(torch.cuda.is_available(), "the fused kernel needs CUDA")
class FusedKernel(unittest.TestCase):
    """The fused Triton kernel against the torch path: the same bits."""

    def test_fused_equals_torch_path(self):
        from weightless_sglang.fused import steer_delta_fused
        for W in WIDTHS:
            dirs = directions() if W == H else {"random": _unit(W, 1)}
            for name, d in dirs.items():
                for dt in (torch.bfloat16, torch.float16, torch.float32):
                    for a in ALPHAS:
                        for T_ in (0, 1, 37, 4096):
                            g = torch.Generator().manual_seed(T_)
                            x = torch.randn(T_, W, generator=g)
                            r = torch.randn(T_, W, generator=g) * 4
                            r[:, (7, 1111, 4095)] *= 400.0
                            x, r = x.to(dt).cuda(), r.to(dt).cuda()
                            dd = d.cuda()
                            al = torch.tensor(a, device="cuda")
                            for rr in (r, None):
                                xf = steer_delta_fused(x, rr, dd, al)
                                xt = steer_delta(x, rr, dd, al)
                                with self.subTest(W=W, d=name, dtype=dt, alpha=a, T=T_,
                                                  residual=rr is not None):
                                    self.assertEqual(xf.dtype, x.dtype)
                                    self.assertTrue(torch.equal(xf.view(torch.uint8), xt.view(torch.uint8)))
                                    if a == 0.0:
                                        self.assertTrue(torch.equal(xf, x))

    def test_fused_3d_per_stream_cuda(self):
        from weightless_sglang.fused import steer_delta_fused
        W = 6144
        d = _unit(W, 2).cuda()
        x = torch.randn(37, 4, W, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16).cuda()
        al = torch.tensor(2.0, device="cuda")
        xf = steer_delta_fused(x, None, d, al)
        self.assertEqual(xf.shape, x.shape)
        self.assertTrue(torch.equal(xf.view(torch.uint8), steer_delta(x, None, d, al).view(torch.uint8)))

    def test_hook_gives_the_same_bits_with_either_kernel(self):
        d = directions()["random"]
        x, r = inputs(torch.bfloat16, "cuda")
        owner = _Owner(d, 1.0, "cuda")
        xt, _ = make_post_layer_hook(owner, LAYER, kernel="torch")(None, (), (x, r))
        xf, _ = make_post_layer_hook(owner, LAYER, kernel="triton")(None, (), (x, r))
        self.assertTrue(torch.equal(xt.view(torch.int16), xf.view(torch.int16)))

    def test_self_check_passes(self):
        from weightless_sglang.fused import self_check
        for W in WIDTHS:
            for with_residual in (True, False):
                for dt in (torch.bfloat16, torch.float16):
                    with self.subTest(W=W, residual=with_residual, dtype=dt):
                        self.assertTrue(self_check("cuda", dtype=dt, width=W, with_residual=with_residual))


class KernelChoice(unittest.TestCase):
    """WEIGHTLESS_STEER_KERNEL: auto (default), triton, torch."""

    def test_torch_is_always_possible(self):
        from weightless_sglang.install import pick_kernel
        self.assertEqual(pick_kernel("torch", "cpu", torch.bfloat16, H)[0], "torch")

    def test_bad_value_fails(self):
        from weightless_sglang.install import pick_kernel
        with self.assertRaises(RuntimeError):
            pick_kernel("cuda", "cpu", torch.bfloat16, H)

    def test_auto_on_cpu_falls_back_to_torch_and_triton_fails(self):
        from weightless_sglang.install import pick_kernel
        k, why = pick_kernel("", "cpu", torch.bfloat16, H)
        self.assertEqual(k, "torch")
        self.assertIn("auto", why)
        with self.assertRaises(RuntimeError):
            pick_kernel("triton", "cpu", torch.bfloat16, H)

    def test_auto_picks_fused_when_it_can_run(self):
        from unittest import mock
        from weightless_sglang import fused
        from weightless_sglang.install import pick_kernel
        with mock.patch.object(fused, "available", return_value=True), \
                mock.patch.object(fused, "self_check", return_value=True):
            self.assertEqual(pick_kernel("auto", "cuda:0", torch.bfloat16, H)[0], "triton")
        with mock.patch.object(fused, "available", return_value=True), \
                mock.patch.object(fused, "self_check", return_value=False):
            self.assertEqual(pick_kernel("auto", "cuda:0", torch.bfloat16, H)[0], "torch")
            with self.assertRaises(RuntimeError):
                pick_kernel("triton", "cuda:0", torch.bfloat16, H)
        with mock.patch.object(fused, "available", return_value=True), \
                mock.patch.object(fused, "self_check", side_effect=RuntimeError("no compiler")):
            k, why = pick_kernel("auto", "cuda:0", torch.bfloat16, H)
            self.assertEqual(k, "torch")
            self.assertIn("no compiler", why)

    def test_auto_checks_the_rows_width_and_residual_form(self):
        from unittest import mock
        from weightless_sglang import fused
        from weightless_sglang.install import pick_kernel
        with mock.patch.object(fused, "available", return_value=True), \
                mock.patch.object(fused, "self_check", return_value=True) as sc:
            k, why = pick_kernel("auto", "cuda:0", torch.bfloat16, 16384, with_residual=False)
        self.assertEqual(k, "triton")
        self.assertEqual(sc.call_args.kwargs["width"], 16384)
        self.assertIs(sc.call_args.kwargs["with_residual"], False)
        self.assertIn("width 16384, no residual", why)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_auto_picks_fused_on_this_gpu(self):
        from weightless_sglang.install import pick_kernel
        for W in WIDTHS:
            for with_residual in (True, False):
                with self.subTest(W=W, residual=with_residual):
                    self.assertEqual(pick_kernel("auto", "cuda", torch.bfloat16, W,
                                                 with_residual=with_residual)[0], "triton")


class Diag(unittest.TestCase):
    def test_layer_diag_records_removal(self):
        import json
        import tempfile
        from weightless_sglang.core import LayerDiag
        d = directions()["random"]
        owner = _Owner(d, 1.0, "cpu")
        with tempfile.TemporaryDirectory() as tmp:
            diag = LayerDiag(tmp, [LAYER])
            hook = make_post_layer_hook(owner, LAYER, diag=diag)
            x, r = inputs(torch.bfloat16, "cpu")
            hook(None, (), (x, r))
            got = json.load(open(diag.path))
        st = got["layers"][str(LAYER)]
        self.assertEqual(got["forwards"], 1)
        self.assertEqual(st["tokens"], T)
        self.assertGreater(st["sum_abs_pre"], 0)
        # post-edit |h.d| is at the bf16 rounding floor of x'
        floor = 2.0 ** -8 * st["sum_norm_xnew"]
        self.assertLess(st["sum_abs_post"], floor)
        print(f"  diag: pre={st['sum_abs_pre']:.3e} post={st['sum_abs_post']:.3e} floor~{floor:.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
