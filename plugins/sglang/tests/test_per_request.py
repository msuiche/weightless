"""Per-request GLP alpha in the plugin: one strength per token row.

The engine (an SGLang build with per-request GLP alpha, launched with
--glp-alpha-per-request) puts a float32 alpha per token row on each forward
and namespaces each alpha's KV by itself; these tests pin the plugin's half,
CPU only:

- the per-row lane at alpha a gives the bits of the scalar lane (the one
  scalar mode has always run) at a, row by row: every reference takes a 0-d
  alpha. Through both torch paths, both DeepSeek-V4.1 edits, a per-stream
  [T, S, W] stream and the site of every architecture row;
- a row at alpha 0 comes back bit for bit as it went in, -0.0 included, even
  when its coefficient is not finite (a zero-padded graph row holds garbage);
- an eager forward with every row at 0 skips the edit, a capture always
  edits and refuses to run without the vector, and a forward without the
  vector fails closed unless it has no token rows or is SGLang's autotune
  warm-up; so does a backbone forward whose forward batch cannot be found;
- a row vector of the wrong length, layout, dtype or device fails closed, in
  the sites and in the fused kernels' wrappers (before anything is launched);
- the codecs in per-request mode: the tuple codec, the forward wrapper, the
  residual-stream codec, the DeepSeek-V4.1 fold (an all-zero forward hands
  back SGLang's own HcState) and its companion writer;
- install_steering with a stand-in for the engine's modules: every pairing
  of the engine flag and the plugin mode, the boot refusals, arming last and
  once, the manifest, the Hy4 and DeepSeek-V4.1 rows end to end, and the
  kernel self-check's per-row lane only in per-request mode;
- where the per-request engine is importable, the pairings and the names
  the plugin relies on against it.

The fused kernels' per-row lane runs under Triton's interpreter in
test_archs/test_dsv41.py and, on a GPU, in the start-up self-checks
(fused.self_check / self_check_hc_mean with per_row=True).
"""
import ast
import dataclasses
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/
import glpfiles  # noqa: E402,F401  (sets up the import paths)
from glpfiles import good_meta, good_tensors, write_gguf  # noqa: E402
from test_archs import _cls, _parse, _trees, unit_dirs  # noqa: E402
from test_archs import test_dsv41 as T41  # noqa: E402  (the DeepSeek-V4.1 fakes)
from test_archs import test_hy4 as H4  # noqa: E402  (the Hy4 fakes)
from test_archs import test_qwen38 as fakes  # noqa: E402  (the qwen3_5 fakes)

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang import fused  # noqa: E402
from weightless_sglang import install as winstall  # noqa: E402
from weightless_sglang.archs import ARCH, EDIT  # noqa: E402
from weightless_sglang.archs import dsv41 as A  # noqa: E402

# One alpha per row: 0 (an unsteered conversation), the engine's ceiling 4.0,
# and repeats, so that every value has several rows.
ALPHAS = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0, 0.5, 2.0, 1.0, 0.0])
# The two ForwardBatch fields the engine carries the rows in (the contract).
FIELDS = ("glp_alpha_rows", "glp_any_steered")


def _unit(width, seed=1):
    d = torch.randn(width, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    return (d / d.norm()).float()


def bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def _rows(T, width=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, width, generator=g) * 4.0


def _hard_zero_rows(x, d, rows=(1, 2, 4)):
    """A copy of x with three rows that alpha 0 must hand back untouched:
    -0.0 elements, values so large that the coefficient overflows to inf, and
    an inf element."""
    x = x.clone()
    a, b, c = rows
    x[a, ..., ::7] = -0.0
    x[b] = (3.0e38 * torch.sign(d)).to(x.dtype)
    x[c].view(-1)[3] = float("inf")
    return x


def same_as_scalar(test, got, alphas, scalar):
    """``got``, the per-row lane over a batch at ``alphas``, against
    ``scalar(a)``, the scalar lane over the same batch at the 0-d alpha a:
    the same bits on every row at a."""
    for a in sorted(set(alphas.tolist())):
        idx = (alphas == a).nonzero().flatten()
        want = scalar(torch.tensor(a))
        test.assertTrue(torch.equal(bits(got[idx]), bits(want[idx])), f"alpha {a}")


def bound(rows, steered=None):
    b = wcore.AlphaRowsBinding()
    b.rows = rows
    b.any_steered = bool((rows != 0).any()) if steered is None else steered
    return b


def eager():
    """Not capturing, not SGLang's autotune warm-up (the CPU default, pinned)."""
    return _patches(capturing=False, capture_mode=False, autotune=False)


class _patches:
    def __init__(self, *, capturing, capture_mode, autotune):
        self.p = [mock.patch.object(wcore, "_capturing", return_value=capturing),
                  mock.patch.object(wcore, "_sglang_capture_mode", return_value=capture_mode),
                  mock.patch.object(wcore, "_sglang_autotune_dummy_run", return_value=autotune)]

    def __enter__(self):
        for p in self.p:
            p.start()

    def __exit__(self, *exc):
        for p in reversed(self.p):
            p.stop()
        return False


class _Owner:
    """The two buffers make_site reads, as register_buffers leaves them."""

    def __init__(self, width=64, layers=2, alpha=0.0):
        self._steer_stack = torch.zeros(layers, 1, width)
        for i in range(layers):
            self._steer_stack[i, 0] = _unit(width, seed=1 + i)
        self._steer_alpha = torch.tensor(alpha)


# ---------------------------------------------------------------- the edits


class TorchPaths(unittest.TestCase):
    """On CPU: the per-row lane at alpha a gives the scalar lane's bits at
    a, row by row."""

    def test_row_edit_equals_the_scalar_lane(self):
        d = _unit(64)
        for dtype in (torch.float32, torch.bfloat16):
            for residual in (False, True):
                with self.subTest(dtype=dtype, residual=residual):
                    x = _rows(9).to(dtype)
                    r = (_rows(9, seed=5) * 100.0).to(dtype) if residual else None
                    got = wcore.steer_rows(x, r, d, ALPHAS)
                    same_as_scalar(self, got, ALPHAS, lambda a: wcore.steer_rows(x, r, d, a))
                    for i in range(9):  # and each row on its own, at its own scalar alpha
                        want = wcore.steer_delta(x[i:i + 1], None if r is None else r[i:i + 1],
                                                 d, ALPHAS[i])
                        self.assertTrue(torch.equal(bits(got[i:i + 1]), bits(want)), f"row {i}")

    def test_row_vector_slices_equal_one_pass(self):
        x, d, al = _rows(37), _unit(64), torch.linspace(0.0, 4.0, 37)
        one = wcore.steer_rows(x, None, d, al)
        with mock.patch.object(wcore, "chunk_rows", lambda width: 4):
            sliced = wcore.steer_rows(x, None, d, al)
        self.assertTrue(torch.equal(bits(one), bits(sliced)))

    def test_zero_rows_come_back_bit_for_bit(self):
        d = _unit(64)
        alphas = torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 4.0, 0.5, 1.5])
        zero = alphas == 0
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for residual in (False, True):
                with self.subTest(dtype=dtype, residual=residual):
                    x = _hard_zero_rows(_rows(9).to(dtype), d)
                    r = _rows(9, seed=5).to(dtype) if residual else None
                    if dtype != torch.float16:  # the values stay finite, the coefficient does not
                        self.assertFalse(bool(torch.isfinite(wcore.dot_fixed(x[2:3].float(), d))))
                    got = wcore.steer_rows(x, r, d, alphas)
                    self.assertTrue(torch.equal(bits(got[zero]), bits(x[zero])))
                    neg0 = (x[1] == 0) & torch.signbit(x[1])
                    self.assertTrue(bool(neg0.any()) and bool(torch.signbit(got[1][neg0]).all()))
                    nz = ~zero
                    same_as_scalar(self, got[nz], alphas[nz], lambda a: wcore.steer_rows(
                        x[nz], None if r is None else r[nz], d, a))

    def test_zero_row_survives_a_non_finite_coefficient(self):
        """0 * inf is NaN; an alpha-0 row comes back as it was whatever its
        coefficient (here every coefficient is inf)."""
        x, d = _rows(2), _unit(64)
        with mock.patch.object(wcore, "dot_fixed",
                               side_effect=lambda h, dd: torch.full(
                                   (h.shape[0],), float("inf"))):
            got = wcore.steer_rows(x, None, d, torch.tensor([0.0, 1.0]))
        self.assertTrue(torch.equal(bits(got[0]), bits(x[0])))

    def test_row_count_mismatch_fails_closed(self):
        x, d = _rows(5), _unit(64)
        with self.assertRaisesRegex(wcore.SiteError, "5 rows"):
            wcore.steer_rows(x, None, d, torch.zeros(4))

    def test_hc_mean_equals_the_scalar_lane(self):
        d = _unit(64)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = T41._streams(T=9, seed=3, dtype=dtype)
                got = A.hc_mean_rows(x, d, ALPHAS)
                same_as_scalar(self, got, ALPHAS, lambda a: A.hc_mean_rows(x, d, a))
                for i in range(9):
                    want = A.hc_mean_delta(x[i:i + 1], d, ALPHAS[i])
                    self.assertTrue(torch.equal(bits(got[i:i + 1]), bits(want)), f"token {i}")

    def test_hc_mean_zero_tokens_come_back_bit_for_bit(self):
        d = _unit(64)
        alphas = torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 4.0, 0.5, 1.5])
        zero = alphas == 0
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = _hard_zero_rows(T41._streams(T=9, seed=3, big=False, dtype=dtype), d)
                got = A.hc_mean_rows(x, d, alphas)
                self.assertTrue(torch.equal(bits(got[zero]), bits(x[zero])))
                nz = ~zero
                same_as_scalar(self, got[nz], alphas[nz], lambda a: A.hc_mean_rows(x[nz], d, a))

    def test_hc_mean_slices_a_per_token_alpha_with_its_tokens(self):
        x, d, al = T41._streams(T=37), _unit(64), torch.linspace(0.0, 4.0, 37)
        one = A.hc_mean_delta(x, d, al)
        with mock.patch.object(A, "hc_chunk_tokens", lambda S, W: 4):
            sliced = A.hc_mean_rows(x, d, al)
        self.assertTrue(torch.equal(bits(one), bits(sliced)))

    def test_both_copies_modes_equal_the_scalar_lane(self):
        x, d = T41._streams(T=9, seed=4), _unit(64)
        for mode in A.COPIES_MODES:
            with self.subTest(mode=mode):
                e = A.HcEdit(mode)
                got = e(x, None, d, ALPHAS, "torch")
                same_as_scalar(self, got, ALPHAS, lambda a: e(x, None, d, a, "torch"))
                if mode == "each":  # each copy as a row, at its token's scalar alpha
                    for t in range(9):
                        want = wcore.steer_rows(x[t:t + 1], None, d, ALPHAS[t])
                        self.assertTrue(torch.equal(bits(got[t:t + 1]), bits(want)), f"token {t}")

    def test_a_per_token_vector_must_be_one_value_per_token(self):
        x, d = T41._streams(T=5), _unit(64)
        for al in (torch.zeros(4), torch.zeros(5, 1), torch.zeros(20)):
            with self.subTest(shape=tuple(al.shape)), \
                    self.assertRaisesRegex(wcore.SiteError, "for 5 tokens"):
                A.hc_mean_rows(x, d, al)


class FusedWrapperRefusals(unittest.TestCase):
    """The fused kernels read a vector alpha linearly: their wrappers refuse
    any other vector before anything is launched (so this runs on CPU)."""

    @staticmethod
    def bad(n):
        return {"length": torch.zeros(n + 1), "float64": torch.zeros(n, dtype=torch.float64),
                "2-D": torch.zeros(n, 1), "strided": torch.zeros(2 * n)[::2],
                "device": torch.zeros(n, device="meta")}

    def test_row_kernel_wrapper(self):
        x, d = _rows(5), _unit(64)
        for what, al in self.bad(5).items():
            with self.subTest(what), self.assertRaisesRegex(
                    wcore.SiteError,
                    r"per-row alpha needs a contiguous float32 \[5\] vector on cpu"):
                fused.steer_delta_fused(x, None, d, al)

    def test_hc_mean_kernel_wrapper(self):
        x, d = T41._streams(T=5), _unit(64)
        for what, al in self.bad(5).items():
            with self.subTest(what), self.assertRaisesRegex(
                    wcore.SiteError,
                    r"per-token alpha needs a contiguous float32 \[5\] vector on cpu"):
                fused.steer_hc_mean_fused(x, d, al)

    def test_the_message_names_the_layout_and_the_device(self):
        with self.assertRaisesRegex(wcore.SiteError, r"shape \(5,\), stride \(2,\), torch.float32 "
                                                     r"on cpu"):
            fused.steer_delta_fused(_rows(5), None, _unit(64), torch.zeros(10)[::2])
        with self.assertRaisesRegex(wcore.SiteError, r"torch.float32 on meta"):
            fused.steer_delta_fused(_rows(5), None, _unit(64), torch.zeros(5, device="meta"))


class SelfCheckLanes(unittest.TestCase):
    """The start-up self-checks run the per-row lane only in a per-request
    boot, so a scalar boot's kernel choice cannot depend on it. On CPU the
    fused wrappers are stood in for by the torch paths: what is pinned is
    which lanes each check runs, and that the per-row lane's alpha-0 rows
    (-0.0, a coefficient that overflows) come back as they went in."""

    def lanes(self, fn, **kw):
        seen = []

        def row(x, r, d, alpha):
            seen.append(alpha.dim())
            return wcore.steer_delta(x, r, d, alpha)

        def hc(x, d, alpha):
            seen.append(alpha.dim())
            return A.hc_mean_rows(x, d, alpha)

        with mock.patch.object(fused, "steer_delta_fused", row), \
                mock.patch.object(fused, "steer_hc_mean_fused", hc):
            ok = fn("cpu", dtype=torch.bfloat16, width=64, **kw)
        return ok, seen

    def test_the_row_kernel_check(self):
        for with_residual in (True, False):
            with self.subTest(with_residual=with_residual):
                self.assertEqual(self.lanes(fused.self_check, with_residual=with_residual),
                                 (True, [0, 0, 0, 0]))
                self.assertEqual(self.lanes(fused.self_check, with_residual=with_residual,
                                            per_row=True), (True, [0, 0, 0, 0, 1]))

    def test_the_hc_mean_check(self):
        self.assertEqual(self.lanes(fused.self_check_hc_mean), (True, [0] * 5))
        self.assertEqual(self.lanes(fused.self_check_hc_mean, per_row=True), (True, [0] * 5 + [1]))

    def test_the_edit_passes_the_lane_on(self):
        with mock.patch.object(fused, "self_check", return_value=True) as sc, \
                mock.patch.object(fused, "self_check_hc_mean", return_value=True) as sh:
            for per_row in (False, True):
                A.HcEdit("each").self_check("cuda:0", torch.bfloat16, 5120, per_row=per_row)
                A.HcEdit("mean").self_check("cuda:0", torch.bfloat16, 5120, per_row=per_row)
        self.assertEqual([c.kwargs["per_row"] for c in sc.call_args_list], [False, True])
        self.assertEqual([c.kwargs["per_row"] for c in sh.call_args_list], [False, True])

    def test_pick_kernel_runs_the_per_row_lane_only_when_asked(self):
        with mock.patch.object(fused, "available", return_value=True), \
                mock.patch.object(fused, "self_check", return_value=True) as sc:
            k, why = winstall.pick_kernel("auto", "cuda:0", torch.bfloat16, 5120)
            self.assertEqual(sc.call_args.kwargs, {"dtype": torch.bfloat16, "width": 5120,
                                                   "with_residual": True})
            self.assertNotIn("per-row", why)
            k, why = winstall.pick_kernel("auto", "cuda:0", torch.bfloat16, 5120, per_row=True)
            self.assertIs(sc.call_args.kwargs["per_row"], True)
            self.assertIn("(width 5120, with residual, per-row alpha)", why)
        seen = []

        def check(device, dtype, width, **kw):
            seen.append(kw)
            return True
        with mock.patch.object(fused, "available", return_value=True):
            winstall.pick_kernel("auto", "cuda:0", torch.bfloat16, 5120, check=check, form="f")
            winstall.pick_kernel("auto", "cuda:0", torch.bfloat16, 5120, check=check, form="f",
                                 per_row=True)
        self.assertEqual(seen, [{}, {"per_row": True}])

    def test_a_failing_per_row_lane_cannot_change_a_scalar_boot(self):
        def check(device, dtype, width, per_row=False):
            return not per_row

        def row_check(device, dtype, width, with_residual, per_row=False):
            if per_row:
                raise RuntimeError("the per-row lane failed")
            return True
        with mock.patch.object(fused, "available", return_value=True):
            for choice in ("auto", "triton"):
                self.assertEqual(winstall.pick_kernel(choice, "cuda:0", torch.bfloat16, 5120,
                                                      check=check, form="f")[0], "triton")
            self.assertEqual(winstall.pick_kernel("auto", "cuda:0", torch.bfloat16, 5120,
                                                  check=check, form="f", per_row=True)[0], "torch")
            with mock.patch.object(fused, "self_check", side_effect=row_check):
                self.assertEqual(winstall.pick_kernel("triton", "cuda:0", torch.bfloat16,
                                                      5120)[0], "triton")
                with self.assertRaisesRegex(RuntimeError, "the per-row lane failed"):
                    winstall.pick_kernel("triton", "cuda:0", torch.bfloat16, 5120, per_row=True)


# ---------------------------------------------------------------- the sites


class SiteBinding(unittest.TestCase):
    def _site(self, owner, **kw):
        return wcore.make_site(owner, 0, kernel="torch", **kw)

    def test_site_uses_the_bound_rows(self):
        owner, x = _Owner(), _rows(9)
        site = self._site(owner)
        owner._weightless_alpha_rows = bound(ALPHAS)
        got = site(x)
        owner._weightless_alpha_rows = None  # scalar mode, the same site
        same_as_scalar(self, got, ALPHAS, lambda a: (setattr(owner, "_steer_alpha", a), site(x))[1])

    def test_eager_all_zero_forward_skips_the_edit(self):
        owner, x = _Owner(), _rows(6)
        site = self._site(owner)
        owner._weightless_alpha_rows = bound(torch.zeros(6))
        with eager():
            self.assertFalse(site.edits_this_forward(x))
            self.assertIs(site(x), x)

    def test_a_forward_without_rows_fails_closed(self):
        """rows None: only a forward with no token rows, or SGLang's autotune
        warm-up (a dummy batch without requests), may run; it runs unedited.
        Anything else fails closed: eager, and inside a capture."""
        owner = _Owner()
        site = self._site(owner)
        owner._weightless_alpha_rows = wcore.AlphaRowsBinding()
        x = _rows(3)
        with eager():
            with self.assertRaisesRegex(wcore.SiteError, "forward of 3 token rows carried no "
                                                         "per-request alpha vector"):
                site(x)
            empty = _rows(0)
            self.assertIs(site(empty), empty)
        with _patches(capturing=False, capture_mode=True, autotune=True):
            self.assertFalse(site.edits_this_forward(x))
            self.assertIs(site(x), x)
        with _patches(capturing=True, capture_mode=True, autotune=True), \
                self.assertRaisesRegex(wcore.SiteError, "capture reached this site without"):
            site(x)

    def test_capture_always_edits(self):
        owner, x = _Owner(), _rows(4)
        site = self._site(owner)
        owner._weightless_alpha_rows = bound(torch.zeros(4), steered=False)
        for cap, mode in ((True, False), (False, True)):
            with _patches(capturing=cap, capture_mode=mode, autotune=False):
                self.assertTrue(site.edits_this_forward(x))

    def test_row_count_mismatch_fails_closed(self):
        owner, x = _Owner(), _rows(4)
        site = self._site(owner)
        owner._weightless_alpha_rows = bound(torch.ones(3))
        with self.assertRaisesRegex(wcore.SiteError, "3 .*4 token rows"):
            site(x)

    def test_vector_dtype_shape_and_device_are_checked(self):
        owner, x = _Owner(), _rows(4)
        site = self._site(owner)
        for rows, what in ((torch.ones(4, dtype=torch.float64), "torch.float64 tensor"),
                           (torch.ones(4, 1), r"shape \(4, 1\)"),
                           (torch.ones(4, device="meta"), "on meta, expected a float32 vector on "
                                                          "cpu")):
            owner._weightless_alpha_rows = bound(rows, steered=True)
            with self.subTest(what), self.assertRaisesRegex(wcore.SiteError, what):
                site(x)

    def test_bind_refuses_a_missing_batch_or_field(self):
        b = wcore.AlphaRowsBinding()
        for fb, what in ((None, "found no forward batch"),
                         (types.SimpleNamespace(residual_stream=None), "without glp_alpha_rows, "
                                                                       "glp_any_steered"),
                         (types.SimpleNamespace(glp_alpha_rows=None), "without glp_any_steered")):
            with self.subTest(what), self.assertRaisesRegex(wcore.SiteError, what):
                b.bind(fb)
        self.assertIsNone(b.rows)
        prev = b.bind(types.SimpleNamespace(glp_alpha_rows=ALPHAS, glp_any_steered=True))
        self.assertEqual(prev, (None, False))
        self.assertIs(b.rows, ALPHAS)


class PerStreamRows(unittest.TestCase):
    """A per-stream [T, S, W] stream without an edit of its own (the Hy4 row):
    one alpha per token, given to each of its streams."""

    def hook(self, owner):
        return wcore.make_post_layer_hook(owner, 0, arity=2, hidden_index=0, residual_index=None,
                                          per_stream=True, width=16, kernel="torch")

    def test_each_stream_of_a_token_is_edited_at_the_tokens_alpha(self):
        owner = _Owner(width=16)
        d = owner._steer_stack[0, 0]
        x = torch.randn(9, 4, 16, generator=torch.Generator().manual_seed(9)) * 3.0
        topk = torch.zeros(9, 2, dtype=torch.int32)
        hook = self.hook(owner)
        owner._weightless_alpha_rows = bound(ALPHAS)
        out, k = hook(None, (), (x, topk))
        self.assertIs(k, topk)
        same_as_scalar(self, out, ALPHAS, lambda a: wcore.steer_rows(x, None, d, a))
        self.assertTrue(torch.equal(bits(out), bits(wcore.steer_rows(
            x, None, d, ALPHAS.repeat_interleave(4)))))

    def test_a_vector_with_one_value_per_stream_row_is_refused(self):
        owner = _Owner(width=16)
        hook = self.hook(owner)
        owner._weightless_alpha_rows = bound(torch.ones(20))
        with self.assertRaisesRegex(wcore.SiteError, "20 values for a stream of 5 token rows"):
            hook(None, (), (torch.randn(5, 4, 16), torch.zeros(5, 2)))


class EveryRow(unittest.TestCase):
    """The site of every architecture row, as the row declares it (its
    stream shape, residual and edit), in per-request mode: the per-row lane
    at alpha a gives the scalar lane's bits at a, row by row. No row is
    left to fail at its first forward."""

    def test_every_row(self):
        self.assertEqual(sorted(EDIT), ["dsv41_hc_post"])
        self.assertIs(EDIT["dsv41_hc_post"], A.make_edit)
        W, T, S = 16, 9, 4
        g = torch.Generator().manual_seed(11)
        for name, row in sorted(ARCH.items()):
            special = row.install.split(":", 1)[1] if row.install.startswith("special:") else None
            edits = ([A.HcEdit(m) for m in A.COPIES_MODES] if special in EDIT else [None])
            for edit in edits:
                with self.subTest(row=name, edit=getattr(edit, "mode", None)):
                    x = torch.randn(*((T, S, W) if row.per_stream else (T, W)), generator=g) * 3.0
                    r = (torch.randn(T, W, generator=g) * 50.0
                         if row.residual_index is not None and not row.per_stream else None)
                    owner = _Owner(width=W)
                    site = wcore.make_site(owner, 0, kernel="torch", width=W,
                                           per_stream=row.per_stream, edit=edit)
                    owner._weightless_alpha_rows = bound(ALPHAS)
                    got = site(x, r)
                    owner._weightless_alpha_rows = None
                    same_as_scalar(self, got, ALPHAS, lambda a: (
                        setattr(owner, "_steer_alpha", a), site(x, r))[1])


class SkippedVisit(unittest.TestCase):
    """A site that decides not to edit still has to be counted.

    DeepSeek-V4.1's fold returns the untouched HcState for an all-zero eager
    forward, so it never calls the site; the fired check would then read the
    forward as one that stopped reaching the site and fail it. The site offers
    ``note`` for exactly that branch (the fold itself: Codecs below).
    """

    def test_note_counts_a_skipped_visit(self):
        owner, x = _Owner(), _rows(3)
        binding = bound(torch.zeros(3))
        check = wcore.FiredCheck([0])
        check.binding = binding
        site = wcore.make_site(owner, 0, kernel="torch", check=check)
        owner._weightless_alpha_rows = binding
        with eager():
            check.begin()
            if not site.edits_this_forward(x):
                site.note(x)
            check.end()  # would raise if the visit had not been counted
        self.assertEqual(check.forwards, 1)

    def test_editing_and_noting_twice_fails_closed(self):
        owner, x = _Owner(), _rows(3)
        binding = bound(torch.ones(3))
        check = wcore.FiredCheck([0])
        check.binding = binding
        site = wcore.make_site(owner, 0, kernel="torch", check=check)
        owner._weightless_alpha_rows = binding
        with self.assertRaisesRegex(RuntimeError, "ran more than once"):
            check.begin()
            site(x)
            site.note(x)
            check.end()


# ---------------------------------------------------------------- the codecs


class _Pair(nn.Module):
    """A layer that returns (hidden, residual), as SGLang's qwen3_5 layers do."""

    def __init__(self, x, r):
        super().__init__()
        self.x, self.r = x, r

    def forward(self, *args, **kwargs):
        return self.x.clone(), self.r


class Codecs(unittest.TestCase):
    """Every codec in per-request mode against the scalar lane, and the
    all-zero eager forward."""

    def setUp(self):
        self.owner = _Owner(width=16, layers=12)
        self.T = 9
        g = torch.Generator().manual_seed(21)
        self.x = torch.randn(self.T, 16, generator=g) * 3.0
        self.r = torch.randn(self.T, 16, generator=g) * 40.0

    def scalar(self, fn):
        """fn() in scalar mode at a, for same_as_scalar."""
        def run(a):
            self.owner._weightless_alpha_rows = None
            self.owner._steer_alpha = a
            return fn()
        return run

    def test_the_tuple_codec(self):
        hook = wcore.make_post_layer_hook(self.owner, 3, arity=3, hidden_index=0, residual_index=1,
                                          width=16, kernel="torch")
        extra = object()
        self.owner._weightless_alpha_rows = bound(ALPHAS)
        out, r, e = hook(None, (), (self.x, self.r, extra))
        self.assertIs(r, self.r)
        self.assertIs(e, extra)
        same_as_scalar(self, out, ALPHAS, self.scalar(lambda: hook(None, (), (self.x, self.r,
                                                                               extra))[0]))
        self.owner._weightless_alpha_rows = bound(torch.zeros(self.T))
        with eager():
            out, r, e = hook(None, (), (self.x, self.r, extra))
        self.assertIs(out, self.x)

    def test_the_forward_wrapper(self):
        layer = _Pair(self.x, self.r)
        hook = wcore.make_post_layer_hook(self.owner, 3, width=16, kernel="torch")
        wcore.wrap_forward(layer, hook, layer_id=3)
        self.owner._weightless_alpha_rows = bound(ALPHAS)
        out, r = layer.forward(None, None)
        self.assertIs(r, self.r)
        same_as_scalar(self, out, ALPHAS, self.scalar(lambda: layer(None, None)[0]))

    def test_the_residual_stream_codec(self):
        calls = []

        def complete_output(output, fb):
            calls.append("complete")
            return output

        batch = types.SimpleNamespace(complete_output=complete_output,
                                      written_residual=lambda fb: fb.r)
        with mock.patch.object(wcore, "_residual_batch", return_value=batch):
            hook = wcore.make_stream_hook(self.owner, 3, width=16, kernel="torch")
        upd = types.SimpleNamespace(is_plain_add=True, applied_at_exit=False)
        fb = types.SimpleNamespace(r=self.r, residual_stream=types.SimpleNamespace(
            pending=types.SimpleNamespace(update=upd, owed=None)))

        def run():
            return hook(None, (), {"forward_batch": fb}, self.x.clone())
        self.owner._weightless_alpha_rows = bound(ALPHAS)
        out = run()
        same_as_scalar(self, out, ALPHAS, self.scalar(run))
        # All zero, eager: no edit, the output keeps its bits; the codec still
        # completed the output first (the README says so).
        self.owner._weightless_alpha_rows = bound(torch.zeros(self.T))
        calls.clear()
        with eager():
            out = run()
        self.assertTrue(torch.equal(bits(out), bits(self.x)))
        self.assertEqual(calls, ["complete"])

    def _dsv41_layer(self, layer_id=5):
        cfg = T41.config(hidden=16, hc_mult=4, num_layers=8)
        bb = T41.DeepseekV4Model(cfg, 0, 8, engram_ids=())
        bb.to(torch.float32)
        return bb, bb.layers[layer_id]

    def _streams(self, T=9):
        g = torch.Generator().manual_seed(22)
        return torch.randn(T, 4, 16, generator=g) * 2.0, torch.randn(T, 16, generator=g)

    def test_the_dsv41_fold(self):
        bb, layer = self._dsv41_layer()
        check = wcore.FiredCheck([5])
        site = wcore.make_site(self.owner, 5, kernel="torch", width=16, per_stream=True,
                               check=check, row=A.ROW_NAME, hidden_slot=A.SITE,
                               edit=A.HcEdit("mean"))
        A.dsv41_hc_post(layer, site, layer_id=5, backbone=bb, row=ARCH[A.ROW_NAME])
        upd = layer.ffn_boundary.declaration.update
        R, y = self._streams()

        def post(rows, steered=None):
            self.owner._weightless_alpha_rows = None if rows is None else bound(rows, steered)
            check.begin()
            out = upd.post(y, T41.ReadState(R), next=layer.attn_hc)
            check.end()
            self.assertEqual(check.fired, {5: 1})
            return out

        with eager():
            # every row 0: SGLang's own HcState comes back, its shortcut kept
            out = post(torch.zeros(9))
            self.assertIs(out, upd.last)
            self.assertIsNotNone(out.input)
            # steered: the edited copies, the shortcut dropped, pre kept
            out = post(ALPHAS)
            self.assertIsNot(out, upd.last)
            self.assertIsNone(out.input)
            self.assertIs(out.pre, upd.last.pre)
            d = self.owner._steer_stack[5, 0]
            same_as_scalar(self, out.streams, ALPHAS,
                           lambda a: A.hc_mean_rows(upd.last.streams, d, a))
            # no vector, not the warm-up: fails closed inside the post
            self.owner._weightless_alpha_rows = wcore.AlphaRowsBinding()
            with self.assertRaisesRegex(wcore.SiteError, "carried no per-request alpha"):
                upd.post(y, T41.ReadState(R), next=layer.attn_hc)
        with _patches(capturing=False, capture_mode=False, autotune=True):
            check.begin()
            out = upd.post(y, T41.ReadState(R), next=layer.attn_hc)
            check.end()
            self.assertIs(out, upd.last)

    def test_the_dsv41_companion_writer(self):
        bb, layer = self._dsv41_layer()
        site = wcore.make_site(self.owner, 6, kernel="torch", width=16, row=A.ROW_NAME,
                               hidden_slot=A.WRITER_SITE)
        A.dsv41_attn_writer(layer, site, layer_id=5, row_id=6, backbone=bb, row=ARCH[A.ROW_NAME])
        aupd = layer.attn_boundary.declaration.update
        R, y = self._streams()
        d = self.owner._steer_stack[6, 0]
        self.owner._weightless_alpha_rows = bound(ALPHAS)
        out = aupd.post(y, T41.ReadState(R))
        # the post folded the attention write edited per row: the same as
        # folding the write that the scalar lane edits row by row
        written = wcore.steer_rows(y, None, d, ALPHAS)
        same_as_scalar(self, written, ALPHAS, lambda a: wcore.steer_rows(y, None, d, a))
        want = type(aupd).post(aupd, written, T41.ReadState(R))
        self.assertTrue(torch.equal(bits(out.streams), bits(want.streams)))


# ---------------------------------------------------------------- the backbone wrapper


class EndToEnd(unittest.TestCase):
    """Rows -> binding -> backbone wrapper -> site, as the engine's forward
    feeds them: each token row is edited at its own strength (the scalar
    lane's bits at that strength), and an alpha-0 row comes out
    bit-identical."""

    class _Module:
        def __init__(self, site, x):
            self.site = site
            self.x = x
            self.out = None

        def forward(self, *args, **kwargs):
            self.out = self.site(self.x)
            return self.out

    def test_each_row_is_edited_at_its_own_strength(self):
        width, T = 64, 5
        x = _rows(T, width)
        owner = _Owner(width=width)
        d = owner._steer_stack[0, 0]
        binding = wcore.AlphaRowsBinding()
        check = wcore.FiredCheck([0])
        check.binding = binding
        site = wcore.make_site(owner, 0, kernel="torch", check=check)
        # install_steering puts the binding on the backbone the sites read.
        owner._weightless_alpha_rows = binding
        module = self._Module(site, x)
        check.wrap_forward(module, binding=binding)
        alphas = torch.tensor([0.0, 1.0, 2.0, 0.0, 3.5])
        fb = mock.Mock()
        fb.glp_alpha_rows = alphas
        fb.glp_any_steered = True

        module.forward(forward_batch=fb)

        self.assertTrue(torch.equal(bits(module.out[0]), bits(x[0])))
        self.assertTrue(torch.equal(bits(module.out[3]), bits(x[3])))
        for i in (1, 2, 4):
            want = wcore.steer_delta(x[i:i + 1], None, d, alphas[i])
            self.assertTrue(torch.equal(bits(module.out[i:i + 1]), bits(want)), f"row {i}")
        # The binding is restored after the forward.
        self.assertIsNone(binding.rows)


class CaptureGuard(unittest.TestCase):
    class _Module:
        """A backbone whose forward reaches its steered layer (the bump)."""

        def __init__(self):
            self.reached = lambda: None

        def forward(self, *args, **kwargs):
            self.reached()
            return "out"

    def _check(self, rows, capturing, capture_mode=False, fb=True):
        check = wcore.FiredCheck([0])
        binding = wcore.AlphaRowsBinding()
        check.binding = binding
        module = self._Module()
        module.reached = lambda: check.bump(0, torch.zeros(4))
        check.wrap_forward(module, binding=binding)
        self.check, self.binding = check, binding
        kw = {}
        if fb:
            kw["forward_batch"] = types.SimpleNamespace(glp_alpha_rows=rows,
                                                        glp_any_steered=rows is not None)
        with _patches(capturing=capturing, capture_mode=capture_mode, autotune=False):
            return module.forward(**kw)

    def test_a_capture_without_rows_refuses_to_record(self):
        with self.assertRaisesRegex(RuntimeError, "without the per-request alpha vector"):
            self._check(rows=None, capturing=True)
        self.assertEqual(self.check.fired_at_capture, [])

    def test_a_capture_with_rows_records_and_reports_them(self):
        out = self._check(rows=torch.zeros(4), capturing=True)
        self.assertEqual(out, "out")
        cap = self.check.fired_at_capture[-1]
        self.assertEqual((cap["rows"], cap["per_request"], cap["stream_capturing"], cap["fired"]),
                         (4, True, True, 1))
        self.assertNotIn("per_request_armed", cap)

    def test_a_warm_up_in_capture_mode_is_not_a_capture(self):
        """SGLang runs a recapture's warm-up inside its capture mode but not
        inside a stream capture, on a dummy batch without the vector: nothing
        is recorded, so the backbone wrapper lets the sites judge it."""
        self.assertEqual(self._check(rows=None, capturing=False, capture_mode=True), "out")
        cap = self.check.fired_at_capture[-1]
        self.assertEqual((cap["rows"], cap["stream_capturing"]), (None, False))

    def test_an_eager_forward_without_rows_reaches_its_sites(self):
        self.assertEqual(self._check(rows=None, capturing=False), "out")

    def test_no_forward_batch_fails_closed_and_leaves_the_check_clean(self):
        with self.assertRaisesRegex(wcore.SiteError, "found no forward batch"):
            self._check(rows=None, capturing=False, fb=False)
        self.assertEqual(self.check.depth, 0)
        self.assertIsNone(self.binding.rows)


# ---------------------------------------------------------------- install


class _Engine:
    """A stand-in for what per-request mode reads from SGLang, put in
    sys.modules for one test and taken out again key by key: the
    ``sglang.srt.glp_alpha`` module (its arming recorded), the runtime
    context and its serving config (``--glp-alpha-per-request``), the
    resolved prefill CUDA-graph backend, the ragged verify mode and the
    ForwardBatch fields. ``glp=False`` is an SGLang without the module,
    ``published=False`` a process where no engine published its config."""

    KEYS = ("sglang", "sglang.srt", "sglang.srt.glp_alpha", "sglang.srt.runtime_context",
            "sglang.srt.model_executor", "sglang.srt.model_executor.cuda_graph_config",
            "sglang.srt.model_executor.forward_batch_info", "sglang.srt.speculative",
            "sglang.srt.speculative.ragged_verify")

    def __init__(self, *, flag=True, glp=True, published=True, prefill="disabled",
                 ragged="static", fields=FIELDS):
        self.armed = []
        m = {k: types.ModuleType(k) for k in self.KEYS}
        for k in self.KEYS:  # the package attributes an import would set
            if "." in k:
                parent, child = k.rsplit(".", 1)
                setattr(m[parent], child, m[k])
        ga = m["sglang.srt.glp_alpha"]
        ga.GLP_ALPHA_MAX = 4.0
        ga.mark_per_request_armed = lambda *, file_sha256, max_alpha=4.0: self.armed.append(
            file_sha256)
        serving = types.SimpleNamespace(glp_alpha_per_request=flag)
        graphs = types.SimpleNamespace(prefill=types.SimpleNamespace(backend=prefill))
        execs = types.SimpleNamespace(graph=types.SimpleNamespace(cuda_graph_config=graphs))

        def bag(value):
            def get():
                if not published:
                    raise ValueError("config namespace not published")
                return value
            return get
        rc = m["sglang.srt.runtime_context"]
        rc.get_context = lambda: types.SimpleNamespace(
            is_config_namespace_published=lambda name: published)
        rc.get_serving, rc.get_exec = bag(serving), bag(execs)
        cg = m["sglang.srt.model_executor.cuda_graph_config"]
        cg.Backend = types.SimpleNamespace(DISABLED="disabled", BREAKABLE="breakable")
        cg.default_cuda_graph_config = lambda: graphs
        m["sglang.srt.speculative.ragged_verify"].read_ragged_verify_mode = (
            lambda: types.SimpleNamespace(value=ragged))
        m["sglang.srt.model_executor.forward_batch_info"].ForwardBatch = dataclasses.make_dataclass(
            "ForwardBatch", [(f, object, dataclasses.field(default=None)) for f in fields])
        if not glp:
            del m["sglang.srt.glp_alpha"]
            del m["sglang.srt"].glp_alpha
        self.mods = m

    def __enter__(self):
        self.saved = {k: sys.modules.get(k) for k in self.KEYS}
        for k in self.KEYS:
            if k in self.mods:
                sys.modules[k] = self.mods[k]
            else:
                sys.modules.pop(k, None)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        return False


class _Backbone(fakes.Backbone):
    """The fake qwen3_5 backbone, called with SGLang's forward batch."""

    def forward(self, x, forward_batch=None):
        return super().forward(x)


def _batch(rows):
    return types.SimpleNamespace(glp_alpha_rows=rows, glp_any_steered=bool((rows != 0).any()),
                                 residual_stream=None)


class InstallBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        os.environ["WEIGHTLESS_STEER_KERNEL"] = "torch"
        self.lines = []
        e = mock.patch.object(winstall, "_emit", side_effect=self.lines.append)
        e.start()
        self.addCleanup(e.stop)
        self.write()

    def write(self, layers=(10, 11, 30), width=8, hint="qwen3_5", alpha="1.0", tensors=None):
        self.path = os.path.join(self.tmp.name, f"v{len(os.listdir(self.tmp.name))}.gguf")
        m = good_meta(layers=layers, alpha=alpha)
        m["controlvector.model_hint"] = hint
        write_gguf(self.path, m, tensors if tensors is not None else good_tensors(layers, width))
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        return self.path

    @staticmethod
    def mode(per_request, alpha="0"):
        os.environ[winstall.PER_REQUEST_ENV] = "1" if per_request else "0"
        if alpha is None:
            os.environ.pop("WEIGHTLESS_STEER_ALPHA", None)
        else:
            os.environ["WEIGHTLESS_STEER_ALPHA"] = alpha

    @staticmethod
    def runner(width=8):
        bb = _Backbone(64, width)
        return types.SimpleNamespace(
            model=fakes.Qwen3_5ForConditionalGeneration(bb), is_draft_worker=False, tp_rank=0,
            tp_size=1, pp_rank=0, pp_size=1, model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(model_type="qwen3_5")))

    @staticmethod
    def hooked(run):
        return sorted(i for i, layer in enumerate(run.model.model.layers) if layer._forward_hooks)


class Install(InstallBase):
    """install_steering against the stand-in engine: the pairings of the
    engine flag and the plugin mode, every boot refusal (each before
    anything is hooked, and never armed), arming last and once."""

    def refused(self, engine, pattern, run=None, **kw):
        run = run or self.runner()
        with engine, self.assertRaisesRegex(RuntimeError, pattern):
            winstall.install_steering(run, **kw)
        self.assertEqual(self.hooked(run), [])
        self.assertEqual(engine.armed, [])
        self.assertNotIn("forward", vars(run.model.model))

    def test_scalar_plugin_on_a_scalar_engine(self):
        self.mode(False, alpha="1.0")
        run = self.runner()
        with _Engine(flag=False) as engine:
            rec = winstall.install_steering(run)
        self.assertEqual(engine.armed, [])
        self.assertIs(rec["per_request"], False)
        self.assertNotIn("per_request_armed", rec)
        self.assertIsNone(run.model.model._weightless_alpha_rows)
        self.assertIsNone(run._weightless_steer_check.binding)
        self.assertEqual(self.hooked(run), [10, 11, 30])
        self.assertIn(" per_request=0 ", self.lines[0])
        self.assertFalse(any("armed" in line for line in self.lines))
        run.model.model(torch.randn(3, 8))  # scalar mode needs no forward batch

    def test_scalar_plugin_on_a_per_request_engine_is_refused(self):
        self.mode(False, alpha="1.0")
        self.refused(_Engine(flag=True), "runs --glp-alpha-per-request but "
                                         "WEIGHTLESS_STEER_PER_REQUEST is not set")

    def test_per_request_plugin_on_a_scalar_engine_is_refused(self):
        self.mode(True)
        self.refused(_Engine(flag=False), "launched without --glp-alpha-per-request")

    def test_per_request_plugin_on_a_per_request_engine(self):
        self.mode(True)
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = self.runner()
        with _Engine(flag=True) as engine:
            rec = winstall.install_steering(run)
        self.assertEqual(engine.armed, [rec["file_sha256"]])
        self.assertIs(rec["per_request"], True)
        self.assertIs(rec["per_request_armed"], True)
        self.assertIs(run.model.model._weightless_alpha_rows, run._weightless_steer_check.binding)
        self.assertEqual(len(self.lines), 2)
        self.assertIn(" per_request=1 ", self.lines[0])
        self.assertTrue(self.lines[1].startswith("weightless per-request alpha armed (pp_rank=0 "
                                                 "tp_rank=0; "))
        with open(rec["manifest"]) as f:
            m = json.load(f)
        self.assertEqual((m["per_request"], m["per_request_armed"]), (True, True))

    def test_no_file_installs_nothing_and_arms_nothing(self):
        self.mode(True)
        os.environ.pop("WEIGHTLESS_STEER_PATH")
        with _Engine(flag=True) as engine:
            self.assertIsNone(winstall.install_steering(self.runner()))
        self.assertEqual(engine.armed, [])

    def test_an_sglang_without_per_request_alpha(self):
        self.mode(False, alpha="1.0")
        with _Engine(glp=False) as engine:
            self.assertIsNotNone(winstall.install_steering(self.runner()))
        self.assertEqual(engine.armed, [])
        self.mode(True)
        self.refused(_Engine(glp=False), "has no sglang.srt.glp_alpha")

    def test_an_unpublished_serving_config(self):
        """A process where no engine published its config (a stand-in runner,
        this suite): scalar mode installs as it always did; per-request mode
        cannot read the flag and refuses."""
        self.mode(False, alpha="1.0")
        with _Engine(flag=True, published=False) as engine:
            self.assertIsNotNone(winstall.install_steering(self.runner()))
        self.assertEqual(engine.armed, [])
        self.mode(True)
        self.refused(_Engine(flag=True, published=False), "serving config is not published")

    def test_boot_refusals(self):
        self.mode(True)
        for kw, pattern in (({"prefill": "breakable"}, "not into prefill CUDA graphs"),
                            ({"ragged": "compact"}, "SGLANG_RAGGED_VERIFY_MODE=static, not "
                                                    "'compact'"),
                            ({"fields": ()}, "does not declare glp_alpha_rows, glp_any_steered"),
                            ({"fields": ("glp_alpha_rows",)}, "does not declare glp_any_steered")):
            with self.subTest(**kw):
                self.refused(_Engine(**kw), pattern)
        os.environ["WEIGHTLESS_STEER_DIAG"] = "1"
        self.refused(_Engine(), "WEIGHTLESS_STEER_DIAG reports one alpha per forward")

    def test_the_server_alpha_refusal_names_its_source(self):
        self.mode(True, alpha="2")
        self.refused(_Engine(),
                     r"but WEIGHTLESS_STEER_ALPHA=2 is set\. Set WEIGHTLESS_STEER_ALPHA=0")
        self.mode(True, alpha=None)
        self.refused(_Engine(), r"WEIGHTLESS_STEER_ALPHA is unset, so the file's glp.alpha_default "
                                r"\(1\) applies")
        self.refused(_Engine(), r"alpha=2 was passed with the path", source="server_args",
                     path=self.path, alpha=2.0)
        self.refused(_Engine(), r"the file's glp.alpha_default \(1\) applies",
                     source="server_args", path=self.path)

    def test_a_rank_without_a_backbone(self):
        run = self.runner()
        run.model.model = None
        self.mode(False, alpha="1.0")
        with _Engine(flag=False):
            self.assertIsNone(winstall.install_steering(run))
        self.mode(True)
        with _Engine(flag=True) as engine, \
                self.assertRaisesRegex(RuntimeError, "no backbone on this rank .* cannot arm"):
            winstall.install_steering(run)
        self.assertEqual(engine.armed, [])

    def test_arming_comes_last(self):
        """A per-request install that fails after its own checks passed arms
        nothing: the engine keeps refusing steered requests."""
        self.mode(True)
        self.write(hint="some_other_model")
        self.refused(_Engine(), "model_hint")
        self.write()
        os.environ["WEIGHTLESS_STEER_KERNEL"] = "triton"
        self.refused(_Engine(), "WEIGHTLESS_STEER_KERNEL=triton but")

    def test_the_kernel_choice_checks_the_per_row_lane_only_per_request(self):
        seen = []

        def pick(*args, **kw):
            seen.append(kw["per_row"])
            return "torch", "recorded"
        with mock.patch.object(winstall, "pick_kernel", side_effect=pick):
            self.mode(False, alpha="1.0")
            with _Engine(flag=False):
                winstall.install_steering(self.runner())
            self.mode(True)
            with _Engine(flag=True):
                winstall.install_steering(self.runner())
        self.assertEqual(seen, [False, True])


class InstalledForwards(InstallBase):
    """Forwards through an installed per-request plugin: the backbone
    wrapper, the sites, the fired check and the manifest."""

    def install(self, alpha="0", per_request=True, run=None, **engine):
        self.mode(per_request, alpha=alpha)
        run = run or self.runner()
        with _Engine(flag=per_request, **engine):
            winstall.install_steering(run)
        return run

    def test_mixed_rows_equal_scalar_installs_token_by_token(self):
        x = torch.randn(9, 8, generator=torch.Generator().manual_seed(31)) * 2.0
        run = self.install()
        with eager():
            got = run.model.model(x, forward_batch=_batch(ALPHAS))

        def scalar(a):
            r = self.install(alpha=f"{float(a)!r}", per_request=False)
            return r.model.model(x)
        same_as_scalar(self, got, ALPHAS, scalar)
        # every row at 0, eager: the stock model's bits
        stock = self.runner()
        with eager():
            zero = run.model.model(x, forward_batch=_batch(torch.zeros(9)))
        self.assertTrue(torch.equal(bits(zero), bits(stock.model.model(x))))

    def test_a_forward_must_carry_the_vector(self):
        run = self.install()
        bb, check = run.model.model, run._weightless_steer_check
        with eager():
            with self.assertRaisesRegex(wcore.SiteError, "found no forward batch"):
                bb(torch.randn(3, 8))
            self.assertEqual(check.depth, 0)
            with self.assertRaisesRegex(wcore.SiteError, "carried no per-request alpha vector"):
                bb(torch.randn(3, 8), forward_batch=types.SimpleNamespace(
                    glp_alpha_rows=None, glp_any_steered=False))
            self.assertEqual(check.depth, 0)
            bb(torch.randn(3, 8), forward_batch=_batch(torch.ones(3)))  # and the next one runs
        with _patches(capturing=False, capture_mode=False, autotune=True):
            bb(torch.randn(3, 8), forward_batch=types.SimpleNamespace(
                glp_alpha_rows=None, glp_any_steered=False))  # the autotune warm-up

    def test_the_manifest_records_each_capture(self):
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = self.install()
        bb = run.model.model
        with _patches(capturing=True, capture_mode=True, autotune=False):
            for bs in (1, 2, 4):
                bb(torch.randn(bs, 8), forward_batch=_batch(torch.zeros(bs)))
            with self.assertRaisesRegex(wcore.SiteError, "capture reached this site without"):
                bb(torch.randn(8, 8), forward_batch=types.SimpleNamespace(
                    glp_alpha_rows=None, glp_any_steered=False))
        with open(run._weightless_steer_installed["manifest"]) as f:
            m = json.load(f)
        self.assertTrue(m["per_request_armed"])
        self.assertEqual([(c["tokens"], c["rows"], c["per_request"], c["fired"])
                          for c in m["fired_at_capture"]],
                         [(1, 1, True, 3), (2, 2, True, 3), (4, 4, True, 3)])

    def test_the_hy4_row(self):
        """The Hy4 row's [T, 4, W] streams in per-request mode, end to end:
        each token at its own alpha gives the scalar install's bits at that
        alpha."""
        class Hy4(H4.HYV4Model):
            def forward(self, x, forward_batch=None):
                return super().forward(x)

        def hy4():
            return H4.runner(H4.HYV4ForCausalLM(Hy4(6, 8, 4)), hf_model_type="hy_v4")
        layers = (1, 2, 4)
        self.write(layers=layers, hint="hy_v4", tensors=unit_dirs(layers, 8))
        x = torch.randn(9, 8, generator=torch.Generator().manual_seed(32))
        run = self.install(run=hy4())
        self.assertEqual(run._weightless_steer_installed["row"], "HYV4ForCausalLM")
        with eager():
            got = run.model.model(x, forward_batch=_batch(ALPHAS))
        same_as_scalar(self, got, ALPHAS, lambda a: self.install(
            alpha=f"{float(a)!r}", per_request=False, run=hy4()).model.model(x))

    def test_the_dsv41_row(self):
        """DeepSeek-V4.1 in per-request mode, end to end on the fakes: mixed
        rows give the scalar installs' bits token by token, and an all-zero
        eager forward gives the stock model's bits (SGLang's own HcStates)."""
        class V41(T41.DeepseekV4Model):
            def forward(self, input_embeds=None, pp_proxy_tensors=None, forward_batch=None):
                return super().forward(input_embeds=input_embeds,
                                       pp_proxy_tensors=pp_proxy_tensors)

        def v41():
            run = T41.runner(hidden=16)
            bb = V41(T41.config(16, 4, 40), 0, 40)
            bb.to(torch.bfloat16)
            run.model = T41.DeepseekV4ForCausalLM(bb)
            return run
        layers = tuple(range(1, 40))
        self.write(layers=layers, width=16, hint="deepseek_v41", alpha="0.75",
                   tensors=unit_dirs(layers, 16))
        x = T41.embeds(T=9)
        run = self.install(run=v41())
        with eager():
            got = run.model.model(input_embeds=x, forward_batch=_batch(ALPHAS))
            zero = run.model.model(input_embeds=x, forward_batch=_batch(torch.zeros(9)))
        same_as_scalar(self, got, ALPHAS, lambda a: self.install(
            alpha=f"{float(a)!r}", per_request=False, run=v41()).model.model(input_embeds=x))
        self.assertTrue(torch.equal(bits(zero), bits(v41().model.model(input_embeds=x))))


# ---------------------------------------------------------------- the engine itself


_GLP = "srt/glp_alpha.py"
_FBI = "srt/model_executor/forward_batch_info.py"


def _engine_trees():
    return _trees(_GLP, _FBI)


@unittest.skipUnless(_engine_trees(),
                     "no SGLang tree with per-request GLP alpha (srt/glp_alpha.py)")
class EngineNames(unittest.TestCase):
    """The names the plugin reads in an SGLang that carries per-request GLP
    alpha, by a parse of its files (the installed package first, then
    WEIGHTLESS_TEST_SGLANG_TREES)."""

    def test_the_names(self):
        for tree in _engine_trees():
            with self.subTest(tree=tree):
                glp = _parse(tree, _GLP)
                fns = {n.name: n for n in glp.body if isinstance(n, ast.FunctionDef)}
                arm = fns["mark_per_request_armed"]
                self.assertIn("file_sha256", [a.arg for a in arm.args.kwonlyargs])
                self.assertTrue(any(
                    isinstance(n, ast.Assign)
                    and any(getattr(t, "id", None) == "GLP_ALPHA_MAX" for t in n.targets)
                    for n in glp.body))
                fb = _cls(_parse(tree, _FBI), "ForwardBatch")
                declared = {n.target.id for n in fb.body if isinstance(n, ast.AnnAssign)}
                self.assertEqual(wcore.AlphaRowsBinding.FIELDS, FIELDS)
                self.assertLessEqual(set(FIELDS), declared)
                rc = _cls(_parse(tree, "srt/runtime_context.py"), "RuntimeContext")
                self.assertIn("is_config_namespace_published",
                              [n.name for n in rc.body if isinstance(n, ast.FunctionDef)])
                lp = _parse(tree, "srt/layers/logits_processor.py")
                self.assertIn("get_in_autotune_dummy_run",
                              [n.name for n in lp.body if isinstance(n, ast.FunctionDef)])
                serving = _parse(tree, "srt/arg_groups/fields/serving.py")
                self.assertIn("glp_alpha_per_request", ast.dump(serving))


def _engine_importable():
    try:
        return importlib.util.find_spec("sglang.srt.glp_alpha") is not None
    except Exception:
        return False


@unittest.skipUnless(_engine_importable(), "this SGLang has no sglang.srt.glp_alpha")
class RealEngine(InstallBase):
    """The importable per-request engine, with no engine context published in
    this process (as in the rest of this suite): a scalar install is today's
    install; a per-request one refuses, since the flag cannot be read."""

    def test_scalar_installs_and_per_request_refuses(self):
        from sglang.srt.runtime_context import get_context
        self.assertFalse(get_context().is_config_namespace_published("serving"))
        self.mode(False, alpha="1.0")
        run = self.runner()
        rec = winstall.install_steering(run)
        self.assertEqual(self.hooked(run), [10, 11, 30])
        self.assertIs(rec["per_request"], False)
        self.mode(True)
        run = self.runner()
        with self.assertRaisesRegex(RuntimeError, "serving config is not published"):
            winstall.install_steering(run)
        self.assertEqual(self.hooked(run), [])


if __name__ == "__main__":
    unittest.main()
