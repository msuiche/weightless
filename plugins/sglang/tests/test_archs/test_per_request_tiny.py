"""Per-request GLP alpha on SGLang's real model classes: one forward that
mixes strengths, CPU only, bit for bit.

SGLang's real Qwen3_5ForConditionalGeneration (the Qwen3.8-27B row) and
Glm5NextForConditionalGeneration (the GLM-5.3-Flash row), built by their own
__init__ at width 64 by sglang_tiny.py (the attention and the MLP/MoE are
token-wise stand-ins; the decoder layers, the residual plumbing, the norms,
the mHC math and the model loop are SGLang's own), with the plugin installed
in per-request mode on the engine this interpreter imports, which must carry
per-request GLP alpha (sglang.srt.glp_alpha):

- the engine context is SGLang's own (``override_server_args``) with
  ``glp_alpha_per_request`` on and prefill CUDA graphs off, so
  install_steering runs every per-request boot check against the real engine
  and arms the engine's own flag (restored after each class);
- each forward's ``glp_alpha_rows`` / ``glp_any_steered`` come from the
  engine's own builder, ``glp_alpha.forward_alpha_rows``, over the
  ScheduleBatch fields it reads (stood in as the engine's own
  test_glp_alpha.py stands them in), from each request's token rows and its
  glp_alpha as the engine parses it (``read_request_alpha``, the parse
  ``Req.__init__`` runs), and go on the forward batch as
  ``ForwardBatch.init_new`` puts them there;
- the references are the same model built again (same weights: everything
  the harness generates depends on the layer, the parameter name and the
  seed) without the plugin (stock) or with the scalar plugin at one alpha,
  on an engine without the per-request flag; the GLP file is synthetic, at
  the tiny width (no published file is needed).

The claim: in one forward that carries several requests at different
strengths, every request's rows (the backbone's output and every layer's
post-layer stream) come out bit for bit as when that request runs alone at
its strength, and as the scalar plugin at that strength gives them; the rows
of a request at alpha 0 come out as stock.

What each case compares, per dtype (bf16, fp32), per request:

- same batch: its rows in the mixed forward against the same batch through
  the scalar plugin at its alpha (stock for alpha 0), where every other row
  runs at that alpha too. A direct comparison, no assumption on the
  backbone's numerics: the other requests' strengths do not reach its rows,
  and its rows get exactly its strength;
- alone: the request alone through the per-request plugin (its one-request
  vector from the same builder) against the scalar plugin at its alpha alone
  (stock for alpha 0);
- across batches: its rows in the mixed forward against its rows alone,
  through both plugins, asserted wherever the reference is row-local, i.e.
  where the scalar plugin at that alpha (or stock) itself gives the request
  the same bits in the batch as alone. Where it does not, no plugin can make
  the rows equal and the first two comparisons carry the claim. On the CPU
  this file was written on, the GLM row's own fp32 mHC math gives a request
  other bits in a batch than alone (F.linear in fp32, for one, gives a row
  other bits at other batch sizes): every request in fp32, some in bf16,
  while the stand-ins are row-local and the Qwen row is row-local
  throughout. The CPU analogue of the GPU's batch-size numerics. Each case
  prints which requests the direct comparison covered;
- the steered requests' rows differ from stock (no case is vacuous), and the
  vector holds each request's alpha on exactly that request's rows.

Cases:

- A: four requests of 5, 1, 7 and 3 rows at 0, 1.0, 2.25 and 0.5 in one
  EXTEND forward; the same four in one MIXED forward (three prefill chunks
  and the one-row request as the decode request, as mixed chunked prefill
  runs them) and in one DECODE forward, one row each; five requests with a
  repeated alpha, the engine's ceiling 4.0 and 1.37 (no binary fraction: the
  engine's two-decimal parse and the scalar plugin's WEIGHTLESS_STEER_ALPHA
  give the same fp32) in one EXTEND forward.
- B: every request at 0 (no field, 0, 0.0, -0.0) in a mixed-length batch,
  in each of the three modes: stock's bits, with nothing steered (the plugin
  skips the edit) and with the same zero vector marked steered (the edit
  runs on every row at 0, as in a captured decode graph).
- C: the same requests in another batch.reqs order, in each of the three
  modes: each request's rows are the ones it had before, and the vector
  built for the old order (rows that follow positions instead of requests)
  steers them wrong.

The stand-ins are checked to be row-local first. Skips when this SGLang
cannot build the tiny models or has no per-request GLP alpha, and without
WEIGHTLESS_TEST_QWEN_CONFIG / WEIGHTLESS_TEST_GLM_CONFIG.
"""
import collections
import contextlib
import io
import os
import sys
import tempfile
import types
import unittest
import zlib
from contextlib import redirect_stderr
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import glpfiles  # noqa: E402  (sets up the import paths)

from test_archs import unit_dirs  # noqa: E402
from test_archs import sglang_tiny as tiny  # noqa: E402

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402


def _engine_or_why():
    """None when this SGLang builds the tiny models and carries per-request
    GLP alpha, else the reason."""
    if tiny.WHY:
        return f"this SGLang cannot build the tiny models here ({tiny.WHY})"
    try:
        from sglang.srt import glp_alpha
        from sglang.srt.model_executor.cuda_graph_config import (  # noqa: F401
            Backend, Phase, default_cuda_graph_config, with_phase)
        from sglang.srt.model_executor.forward_batch_info import (  # noqa: F401
            ForwardBatch, ForwardMode)
        from sglang.srt.server_args import ServerArgs
    except ImportError as e:
        return f"this SGLang has no per-request GLP alpha ({type(e).__name__}: {e})"
    for name in ("forward_alpha_rows", "read_request_alpha", "per_request_armed"):
        if not callable(getattr(glp_alpha, name, None)):
            return f"this SGLang's sglang.srt.glp_alpha has no {name}"
    declared = getattr(ForwardBatch, "__dataclass_fields__", {})
    missing = [f for f in wcore.AlphaRowsBinding.FIELDS if f not in declared]
    if missing:
        return f"this SGLang's ForwardBatch does not declare {', '.join(missing)}"
    fields = getattr(ServerArgs, "__struct_fields__", ())
    for f in ("glp_alpha_per_request", "cuda_graph_config"):
        if f not in fields:
            return f"this SGLang's ServerArgs has no {f}"
    return None


WHY = _engine_or_why()
QWEN_CFG = tiny.config_path("qwen")
GLM_CFG = tiny.config_path("glm")

HID = 64
DTYPES = (torch.bfloat16, torch.float32)
# The forward modes: extend and mixed take each request's extend rows,
# decode one row per request.
KINDS = ("extend", "mixed", "decode")
STOCK, PER_REQUEST = "stock", "per-request"

Arch = collections.namedtuple("Arch", "row hf hint layers width seed")
ARCHS = {
    # like GLP-63 on Qwen3.8-27B: layers 1..63 of 64, on the residual stream
    "qwen": Arch("Qwen3_5ForConditionalGeneration", "qwen3_5", "qwen3_5", tuple(range(1, 64)),
                 HID, 5),
    # like GLP-44 on GLM-5.3-Flash: layers 1..44 of 45, on the 4 x 64 mHC stream
    "glm": Arch("Glm5NextForConditionalGeneration", "glm5_next", "glm5_next", tuple(range(1, 45)),
                4 * HID, 3),
}

# (name, extend rows, the request's glp_alpha field; None: no field)
CASE_A = (("r0", 5, 0), ("r1", 1, 1.0), ("r2", 7, 2.25), ("r3", 3, 0.5))
CASE_A5 = (("s0", 4, 2.25), ("s1", 6, 0), ("s2", 2, 4.0), ("s3", 3, 2.25), ("s4", 5, 1.37))
CASE_B = (("z0", 5, None), ("z1", 1, 0), ("z2", 7, 0.0), ("z3", 3, -0.0))
PERM = (2, 0, 3, 1)  # case C: batch.reqs in this order of CASE_A


def bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))


def _name(dtype):
    return {torch.bfloat16: "bf16", torch.float32: "fp32"}[dtype]


@contextlib.contextmanager
def _steer_env(env):
    """os.environ with no WEIGHTLESS_STEER_* but ``env``, restored after."""
    with mock.patch.dict(os.environ, {}, clear=False):
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            del os.environ[k]
        os.environ.update(env)
        yield


def _graph_config():
    """Prefill CUDA graphs off, as a per-request engine must run
    (--disable-prefill-cuda-graph). Without it the dummy context leaves the
    setting unresolved (None), and the plugin's boot check reads its prefill
    backend."""
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend, Phase, default_cuda_graph_config, with_phase)
    return with_phase(default_cuda_graph_config(), Phase.PREFILL, backend=Backend.DISABLED)


class Req:
    """One request: its token rows, its embeddings (seeded by its name, so a
    request brings the same rows wherever it sits in a batch; ``x1`` is its
    one decode row) and the glp_alpha the engine reads from its body."""

    def __init__(self, name, rows, value):
        from sglang.srt.glp_alpha import read_request_alpha
        self.name, self.rows = name, rows
        self.glp_alpha, err = read_request_alpha(None if value is None else {"glp_alpha": value})
        if err is not None:
            raise AssertionError(f"{name}: the engine refuses glp_alpha={value!r}: {err}")
        g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        self.x = torch.randn(rows, HID, generator=g) * 2.0
        self.x1 = torch.randn(1, HID, generator=g) * 2.0

    def __repr__(self):
        return f"{self.name} ({self.rows} rows at {self.glp_alpha:g})"


def requests(case):
    return [Req(*c) for c in case]


class Layout:
    """The requests of one forward in batch.reqs order: the input rows, each
    request's span, and the engine's row vector for it."""

    def __init__(self, reqs, kind, dtype):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        self.reqs, self.kind = list(reqs), kind
        self.mode = {"extend": ForwardMode.EXTEND, "mixed": ForwardMode.MIXED,
                     "decode": ForwardMode.DECODE}[kind]
        parts = [(r.x1 if kind == "decode" else r.x).to(dtype) for r in self.reqs]
        self.x = torch.cat(parts)
        self.spans, s = {}, 0
        for r, p in zip(self.reqs, parts):
            self.spans[r.name] = (s, s + p.shape[0])
            s += p.shape[0]
        self.T = s

    def span(self, r):
        return self.spans[r.name]

    def engine_rows(self):
        """``forward_alpha_rows`` for this forward: one alpha per token row in
        batch.reqs order, and whether any request is steered."""
        from sglang.srt.glp_alpha import forward_alpha_rows
        extend = self.kind != "decode"
        batch = types.SimpleNamespace(
            forward_mode=self.mode,
            reqs=[types.SimpleNamespace(glp_alpha=r.glp_alpha) for r in self.reqs],
            extend_lens=[r.rows for r in self.reqs] if extend else None,
            spec_info=None, beam_tail=None,
            seq_lens=torch.ones(len(self.reqs), dtype=torch.int64),
            input_ids=torch.zeros(self.T, dtype=torch.int64))
        return forward_alpha_rows(batch, device=torch.device("cpu"))


class Run:
    """One forward: the backbone's output and every layer's post-layer stream
    (sglang_tiny.capture_streams, registered after the steering)."""

    def __init__(self, out, streams):
        self.out, self.streams = out, streams

    def rows(self, span):
        s, e = span
        yield "output", self.out[s:e]
        for i in sorted(self.streams):
            x, r = self.streams[i]
            yield f"layer {i} stream", x[s:e]
            if r is not None:
                yield f"layer {i} residual", r[s:e]


def rows_equal(a, sa, b, sb):
    pa, pb = list(a.rows(sa)), list(b.rows(sb))
    return ([n for n, _ in pa] == [n for n, _ in pb]
            and all(same_bits(x, y) for (_, x), (_, y) in zip(pa, pb)))


class Model:
    """One tiny model of ``arch`` in ``dtype``: ``STOCK`` (no plugin),
    ``PER_REQUEST`` (the plugin with WEIGHTLESS_STEER_PER_REQUEST=1 and the
    server alpha 0, on a --glp-alpha-per-request engine) or a float, the
    scalar plugin at that alpha (today's boot, on an engine without the
    flag). Every one has the same graph settings and the same weights."""

    def __init__(self, arch, dtype, key, path):
        self.arch, self.dtype, self.key = arch, dtype, key
        ctx = tiny.Context()
        ctx.sa["cuda_graph_config"] = _graph_config()
        if key == PER_REQUEST:
            ctx.sa["glp_alpha_per_request"] = True
        model, self.bb, self.ctx = tiny.build(arch, tiny.config_path(arch), hidden=HID,
                                              dtype=dtype, ctx=ctx)
        self.runner = tiny.runner_of(model, ARCHS[arch].hf, dtype,
                                     ranks=None if tiny.NEW_LAYOUT else (0, 1, 0, 1))
        self.rec, self.log = None, ""
        if key != STOCK:
            env = {"WEIGHTLESS_STEER_PATH": path}
            if key == PER_REQUEST:
                env.update(WEIGHTLESS_STEER_ALPHA="0", WEIGHTLESS_STEER_PER_REQUEST="1")
            else:
                env["WEIGHTLESS_STEER_ALPHA"] = repr(float(key))
            with _steer_env(env), self.ctx, redirect_stderr(io.StringIO()) as err:
                self.rec = install_steering(self.runner, source="env")
            self.log = err.getvalue()
        self.store = {}
        tiny.capture_streams(arch, self.bb, self.store)

    def forward(self, lay, rows=None, steered=False):
        fb = tiny.forward_batch(lay.T)
        fb.forward_mode = lay.mode
        # The two fields ForwardBatch.init_new fills from forward_alpha_rows.
        fb.glp_alpha_rows, fb.glp_any_steered = rows, steered
        self.store.clear()
        with self.ctx, torch.no_grad():
            out = self.bb(None, None, fb, input_embeds=lay.x, pp_proxy_tensors=None)
        run = Run(out, dict(self.store))
        if sorted(run.streams) != list(range(self.bb.start_layer, self.bb.end_layer)):
            raise AssertionError(f"{self.arch}: the streams of layers {sorted(run.streams)} were "
                                 f"captured")
        if self.rec is not None:
            fired = self.runner._weightless_steer_check.fired
            if sorted(fired) != self.rec["local_layer_ids"] or set(fired.values()) != {1}:
                raise AssertionError(f"{self.arch} {self.key}: the sites fired {fired}")
        if self.key == PER_REQUEST and self.bb._weightless_alpha_rows.rows is not None:
            raise AssertionError("the forward's alpha vector is still bound after the forward")
        return run


class _MixedStrengths:
    """The cases, for the architecture row ``ARCH`` (module text)."""

    ARCH = None
    ROW_LOCAL = False   # set by an arch whose stock rows are batch-invariant on this CPU

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from sglang.srt import glp_alpha
        cls.ga = glp_alpha
        # The per-request installs arm the engine's per-process flag: back to
        # what it was once this class is done.
        armed = mock.patch.object(glp_alpha, "_ARMED", False)
        armed.start()
        cls.addClassCleanup(armed.stop)
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        spec = ARCHS[cls.ARCH]
        meta = glpfiles.good_meta(layers=list(spec.layers), alpha="1.0")
        meta["controlvector.model_hint"] = spec.hint
        cls.path = os.path.join(tmp.name, f"{cls.ARCH}-tiny.gguf")
        glpfiles.write_gguf(cls.path, meta, unit_dirs(spec.layers, spec.width, seed=spec.seed))
        cls.models = {}

    @classmethod
    def model(cls, dtype, key):
        if (dtype, key) not in cls.models:
            cls.models[dtype, key] = Model(cls.ARCH, dtype, key, cls.path)
        return cls.models[dtype, key]

    def reference(self, dtype, r):
        """Stock for a request at alpha 0, else the scalar plugin at its alpha,
        whose fp32 strength is the one the engine's vector holds for it."""
        if r.glp_alpha == 0:
            return self.model(dtype, STOCK)
        m = self.model(dtype, r.glp_alpha)
        self.assertTrue(same_bits(m.bb._steer_alpha.reshape(1),
                                  torch.tensor([r.glp_alpha], dtype=torch.float32)),
                        f"{r}: the scalar plugin holds {float(m.bb._steer_alpha)!r}")
        return m

    def say(self, dtype, label, text):
        print(f"  {self.ARCH} {_name(dtype)} {label}: {text}", flush=True)

    def assertRowsEqual(self, a, sa, b, sb, what):
        pa, pb = list(a.rows(sa)), list(b.rows(sb))
        self.assertEqual([n for n, _ in pa], [n for n, _ in pb], what)
        for (n, x), (_, y) in zip(pa, pb):
            if not same_bits(x, y):
                self.fail(f"{what}: the {n} differs (max abs "
                          f"{float((x.float() - y.float()).abs().max()):.3g})")
        return len(pa)

    def assertVector(self, rows, steered, lay):
        """The engine's vector: fp32, one value per token row, each request's
        alpha on exactly its rows, steered when any request is."""
        self.assertIsNotNone(rows)
        self.assertEqual((rows.dtype, tuple(rows.shape), rows.device.type),
                         (torch.float32, (lay.T,), "cpu"))
        for r in lay.reqs:
            s, e = lay.span(r)
            self.assertTrue(same_bits(rows[s:e], torch.full((e - s,), r.glp_alpha,
                                                            dtype=torch.float32)),
                            f"{r}: its rows hold {rows[s:e].tolist()}")
        self.assertIs(steered, any(r.glp_alpha != 0 for r in lay.reqs))

    def mixed(self, dtype, reqs, kind, label):
        """One forward that carries ``reqs`` at their strengths, against the
        references (module text). Returns (layout, run, rows, steered)."""
        pr, stock = self.model(dtype, PER_REQUEST), self.model(dtype, STOCK)
        lay = Layout(reqs, kind, dtype)
        rows, steered = lay.engine_rows()
        self.assertVector(rows, steered, lay)
        got = pr.forward(lay, rows, steered)
        stock_run = stock.forward(lay)
        refs, direct, other, n = {}, [], [], 0
        for r in lay.reqs:
            span, what = lay.span(r), f"{self.ARCH} {_name(dtype)} {label}, {r}"
            ref = self.reference(dtype, r)
            if r.glp_alpha not in refs:
                refs[r.glp_alpha] = ref.forward(lay)
            same = refs[r.glp_alpha]
            n = self.assertRowsEqual(got, span, same, span,
                                     f"{what}: the mixed forward against the same batch with "
                                     f"every row at {r.glp_alpha:g}")
            if r.glp_alpha != 0:
                self.assertFalse(rows_equal(got, span, stock_run, span),
                                 f"{what}: its steered rows are stock's")
            one = Layout([r], kind, dtype)
            rows1, steered1 = one.engine_rows()
            self.assertVector(rows1, steered1, one)
            alone, ref_alone = pr.forward(one, rows1, steered1), ref.forward(one)
            self.assertRowsEqual(alone, one.span(r), ref_alone, one.span(r),
                                 f"{what}: alone, the per-request plugin against "
                                 f"{'stock' if r.glp_alpha == 0 else 'the scalar plugin'}")
            if rows_equal(same, span, ref_alone, one.span(r)):
                self.assertRowsEqual(got, span, alone, one.span(r),
                                     f"{what}: the mixed forward against the request alone")
                self.assertRowsEqual(got, span, ref_alone, one.span(r),
                                     f"{what}: the mixed forward against the request alone "
                                     f"through the reference")
                direct.append(r.name)
            else:
                stock_one = stock.forward(one)
                local = rows_equal(stock_run, span, stock_one, one.span(r))
                other.append(f"{r.name} (stock {'is' if local else 'is not'} row-local there)")
        k = len(lay.reqs)
        if self.ROW_LOCAL:
            self.assertEqual(len(direct), k,
                             f"{self.ARCH} {_name(dtype)} {label}: every request must be "
                             f"judged mixed against alone on this arch, not judged: "
                             f"{', '.join(other)}")
        self.say(dtype, label,
                 f"{k} requests, {lay.T} rows, alphas "
                 f"{', '.join(f'{r.glp_alpha:g}' for r in lay.reqs)}; {n} row blocks per request "
                 f"(the output and every layer's stream) equal bit for bit: the same batch at "
                 f"its alpha {k}/{k}, alone through both plugins {k}/{k}, mixed against alone "
                 f"{len(direct)}/{k}"
                 + (f"; not judged across batches, the reference itself not being row-local "
                    f"there: {', '.join(other)}" if other else ""))
        return lay, got, rows, steered

    # ------------------------------------------------------------ the harness

    def test_the_stand_ins_are_row_local(self):
        """Each layer runs exactly one attention and one MLP stand-in, and each
        gives every row of a batch the bits that row gives alone, whatever
        the other rows hold: anything that ties a row's bits to its batch is
        SGLang's own code. Whether the whole stock backbone is row-local is
        measured on the case A batch and printed (the cases judge it per
        request)."""
        g = torch.Generator().manual_seed(77)
        x = torch.randn(13, HID, generator=g) * 2.0
        y = x.clone()
        y[7:] = torch.randn(6, HID, generator=g) * 3.0
        for dtype in DTYPES:
            with self.subTest(dtype=_name(dtype)):
                m = self.model(dtype, STOCK)
                xd, yd = x.to(dtype), y.to(dtype)
                for i in range(m.bb.start_layer, m.bb.end_layer):
                    mods = [mod for mod in m.bb.layers[i].modules()
                            if isinstance(mod, (tiny.StandInAttn, tiny.StandInMLP))]
                    self.assertEqual(sorted(type(mod).__name__ for mod in mods),
                                     ["StandInAttn", "StandInMLP"], f"layer {i}")
                    with torch.no_grad():
                        for mod in mods:
                            fx, fy = mod(xd), mod(yd)
                            self.assertTrue(same_bits(fx[:7], fy[:7]), f"layer {i} {mod}")
                            for t in range(13):
                                self.assertTrue(same_bits(fx[t:t + 1], mod(xd[t:t + 1])),
                                                f"layer {i} {type(mod).__name__} row {t}")
                lay = Layout(requests(CASE_A), "extend", dtype)
                run = m.forward(lay)
                local = [r.name for r in lay.reqs
                         if rows_equal(run, lay.span(r), m.forward(Layout([r], "extend", dtype)),
                                       (0, r.rows))]
                self.say(dtype, "stand-ins", f"row-local in all {m.bb.end_layer} layers; the "
                         f"stock backbone gives {len(local)}/{len(lay.reqs)} requests of the "
                         f"case A batch the bits they have alone")

    def test_the_install_is_per_request_on_this_engine(self):
        """install_steering in per-request mode on this engine: its record, its
        boot lines, the engine's own armed flag and the forward binding; the
        scalar references install as today's boot does."""
        spec = ARCHS[self.ARCH]
        for dtype in DTYPES:
            with self.subTest(dtype=_name(dtype)):
                pr = self.model(dtype, PER_REQUEST)
                rec = pr.rec
                self.assertEqual((rec["row"], rec["width"], rec["local_layer_ids"]),
                                 (spec.row, spec.width, list(spec.layers)))
                self.assertEqual((rec["per_request"], rec["per_request_armed"], rec["alpha"],
                                  rec["kernel"]), (True, True, 0.0, "torch"))
                self.assertTrue(self.ga.per_request_armed())  # the engine's own flag
                self.assertIn(" per_request=1 ", pr.log)
                self.assertIn("weightless per-request alpha armed", pr.log)
                self.assertIs(pr.bb._weightless_alpha_rows,
                              pr.runner._weightless_steer_check.binding)
                sc = self.model(dtype, 2.25)
                self.assertEqual((sc.rec["per_request"], sc.rec["alpha"], sc.rec["kernel"]),
                                 (False, 2.25, "torch"))
                self.assertIsNone(sc.bb._weightless_alpha_rows)
                self.assertIn(" per_request=0 ", sc.log)

    # ------------------------------------------------------------ A

    def test_a_mixed_strengths_in_one_extend_forward(self):
        for dtype in DTYPES:
            with self.subTest(dtype=_name(dtype)):
                self.mixed(dtype, requests(CASE_A), "extend", "A extend")

    def test_a_repeats_the_ceiling_and_a_two_decimal_alpha(self):
        for dtype in DTYPES:
            with self.subTest(dtype=_name(dtype)):
                self.mixed(dtype, requests(CASE_A5), "extend", "A extend, five requests")

    def test_a_mixed_strengths_in_one_mixed_mode_forward(self):
        for dtype in DTYPES:
            with self.subTest(dtype=_name(dtype)):
                self.mixed(dtype, requests(CASE_A), "mixed", "A mixed")

    def test_a_mixed_strengths_in_one_decode_forward(self):
        for dtype in DTYPES:
            with self.subTest(dtype=_name(dtype)):
                self.mixed(dtype, requests(CASE_A), "decode", "A decode")

    # ------------------------------------------------------------ B

    def test_b_every_request_at_zero_is_stock(self):
        for dtype in DTYPES:
            for kind in KINDS:
                with self.subTest(dtype=_name(dtype), kind=kind):
                    lay = Layout(requests(CASE_B), kind, dtype)
                    rows, steered = lay.engine_rows()
                    self.assertVector(rows, steered, lay)
                    self.assertFalse(steered)
                    whole = (0, lay.T)
                    stock = self.model(dtype, STOCK).forward(lay)
                    pr = self.model(dtype, PER_REQUEST)
                    n = self.assertRowsEqual(pr.forward(lay, rows, steered), whole, stock, whole,
                                             f"{kind}: every request at 0, nothing steered")
                    self.assertRowsEqual(pr.forward(lay, rows, True), whole, stock, whole,
                                         f"{kind}: every request at 0, the edit run on every row")
                    self.say(dtype, f"B {kind}", f"{len(lay.reqs)} requests at 0, {lay.T} rows: "
                             f"stock's bits in all {n} row blocks, with the edit skipped and "
                             f"with it run at 0 on every row")

    # ------------------------------------------------------------ C

    def test_c_the_rows_follow_the_requests(self):
        for dtype in DTYPES:
            for kind in KINDS:
                with self.subTest(dtype=_name(dtype), kind=kind):
                    reqs = requests(CASE_A)
                    lay, got, rows, steered = self.mixed(dtype, reqs, kind, f"C {kind}, in order")
                    order = [reqs[k] for k in PERM]
                    lay_p, got_p, _, _ = self.mixed(dtype, order, kind, f"C {kind}, permuted")
                    moved = []
                    for r in reqs:
                        ref = self.reference(dtype, r)
                        if rows_equal(ref.forward(lay_p), lay_p.span(r), ref.forward(lay),
                                      lay.span(r)):
                            self.assertRowsEqual(got_p, lay_p.span(r), got, lay.span(r),
                                                 f"{kind} {r}: permuted against in order")
                            moved.append(r.name)
                    # The vector built for the old order, on the new one: rows
                    # that follow positions instead of requests.
                    wrong = self.model(dtype, PER_REQUEST).forward(lay_p, rows, steered)
                    mis = []
                    for r in reqs:
                        s, e = lay_p.span(r)
                        if bool((rows[s:e] != r.glp_alpha).any()):
                            mis.append(r.name)
                            self.assertFalse(rows_equal(wrong, (s, e), got_p, (s, e)),
                                             f"{kind} {r}: steered at the wrong rows' alphas, "
                                             f"its rows did not move")
                    self.assertEqual(len(mis), len(reqs))
                    if self.ROW_LOCAL:
                        self.assertEqual(len(moved), len(reqs),
                                         f"{kind}: every request must be judged across the "
                                         f"two orders")
                    self.say(dtype, f"C {kind}", f"batch.reqs order {[r.name for r in order]}: "
                             f"{len(moved)}/{len(reqs)} requests keep their rows bit for bit "
                             f"across the two orders; the old order's vector mis-steers "
                             f"{len(mis)}/{len(reqs)}")


@unittest.skipIf(WHY, WHY or "")
@unittest.skipUnless(QWEN_CFG, "set WEIGHTLESS_TEST_QWEN_CONFIG to the Qwen3.8-27B config.json")
class Qwen38(_MixedStrengths, unittest.TestCase):
    """The Qwen3.8-27B row (the architecture of the RTX 5090 runs)."""

    ARCH = "qwen"
    # its stock rows do not depend on the batch: every request is judged across batches
    ROW_LOCAL = True


@unittest.skipIf(WHY, WHY or "")
@unittest.skipUnless(GLM_CFG, "set WEIGHTLESS_TEST_GLM_CONFIG to a GLM-5.3-Flash config.json")
class Glm53Flash(_MixedStrengths, unittest.TestCase):
    """The GLM-5.3-Flash row: the edit on the widened mHC stream."""

    ARCH = "glm"
    # its fp32 mHC math changes with the batch size: judged within one batch only
    ROW_LOCAL = False


if __name__ == "__main__":
    unittest.main(verbosity=2)
