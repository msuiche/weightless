"""The GLP edit at one steering site, the tuple codec around it, the
per-forward check that every site fired, and the forward wrapper used where
a model loop calls ``layer.forward(...)``. Pure torch.

Many SGLang decoder layers return ``(hidden, residual)``. The add of the two
is left to the next layer's fused add+RMSNorm, so the stream after layer i
is ``h = hidden + residual``. The vLLM plugin (vllm-plugin archs/base.py,
_steer_post_layer) projects h and writes back ``hidden <- h' - residual``.
Here the same edit is written in its delta form, computed in fp32 and
rounded once:

    c       = (f32(x) + f32(r)) . d
    x'      = round(f32(x) - (alpha * c) d)       # r is returned untouched

In exact arithmetic ``x' = h' - r``. Written this way, alpha = 0 gives back
x exactly (``round(f32(x) - 0) == x``), and ``x + r`` is never rounded to
bf16 first. On models with very large residual values that early rounding
would lose the low bits of x. When a site has no residual (``r=None``) the
stream is x itself and the same steps run with h = f32(x).

The dot product c uses a fixed-point sum (see ``dot_fixed``): its result
does not depend on the order of the additions. The fused Triton kernel in
fused.py does the same steps, so the torch path and the fused path give
the same bits, on any batch size.

The torch path works on at most ``chunk_rows(width)`` rows at a time, so
its temporaries stay under ``TEMP_BUDGET_BYTES`` at any width. Every row is
computed on its own, so the slices give the same bits as one pass.

No host sync, no data-dependent Python and no data-dependent allocation:
the Python below runs once per captured shape at CUDA-graph capture, and
its tensor ops replay with the graph.
"""
from __future__ import annotations

import functools
import json
import math
import os
import threading
import time

import torch


def fixed_point_bits(width: int) -> int:
    """Magnitude budget, in bits, of one term of the fixed-point dot product.

    Each term is scaled below 2**bits, so that ``width`` of them add up in an
    int64 without overflow (5120 wide: 49 bits).
    """
    return 62 - (int(width) - 1).bit_length()


def dot_fixed(h: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
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
    bits = fixed_point_bits(h.shape[-1])
    p = h * d
    pm = p.abs().amax(-1, keepdim=True)
    e = (pm.view(torch.int32) >> 23) & 0xFF
    s = (bits + 126 - e).clamp(-100, 100)
    scale = ((s + 127) << 23).view(torch.float32)
    inv = ((127 - s) << 23).view(torch.float32)
    q = p.mul_(scale).to(torch.int64)  # p is our own temporary: scaled in place
    acc = q.sum(-1, keepdim=True)
    return (acc.to(torch.float32) * inv).squeeze(-1)


def steer_delta(x: torch.Tensor, r, d: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Return x' such that x' + r == (x + r) - alpha * ((x + r) . d) d.

    x, r: [..., H] (r may be None: then the stream is x itself).
    d: fp32 [H] unit direction. alpha: fp32 0-d tensor. The math runs in
    fp32 and rounds once to x.dtype. float64 inputs (used by the tests to
    compare against the exact form) use a plain float64 dot product.
    This is one pass over all rows; ``steer_rows`` is the bounded-memory
    entry point that the sites call.
    """
    ct = torch.promote_types(x.dtype, torch.float32)  # fp32, or fp64 in tests
    xf = x.to(ct)
    d = d.to(ct)
    # xf + r promotes r to ct element by element: the same values as
    # xf + r.to(ct), without a converted copy of r.
    h = xf if r is None else xf + r
    if h.dtype != ct:  # r wider than x (never the case in SGLang)
        h = xf + r.to(ct)
    c = dot_fixed(h, d) if ct == torch.float32 else h @ d  # [...]
    del h
    t = (alpha.to(ct) * c).unsqueeze(-1)
    td = t * d
    return torch.sub(xf, td, out=td).to(x.dtype)  # xf - t d, written over td


# Bounded memory for the torch path. Per row element, one pass allocates at
# most (bf16 input, fp32 math): f32(x) 4, h 4, p 4, |p| 4, int64 q 8, t*d 4,
# and the rounded result 2: 30 bytes. The unit test measures this from the
# tensors a pass really allocates.
TEMP_BYTES_PER_ELEM = 30
TEMP_BUDGET_BYTES = 256 * 10 ** 6  # 256 MB


def chunk_rows(width: int, budget: int = TEMP_BUDGET_BYTES) -> int:
    """Rows per slice of the torch path: the largest power of two whose
    temporaries fit in ``budget`` (1024 at width 5120, 512 at 16384)."""
    n = max(1, int(budget) // (TEMP_BYTES_PER_ELEM * max(1, int(width))))
    return 1 << (n.bit_length() - 1)


def steer_rows(x: torch.Tensor, r, d: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """``steer_delta`` over x's rows (all leading dims flattened), at most
    ``chunk_rows(W)`` rows at a time. The same bits as one pass, because
    every row is computed on its own. The slice count depends only on x's
    shape, so the loop is fixed for a captured shape."""
    W = x.shape[-1]
    n = x.numel() // W if W else 0
    step = chunk_rows(W)
    if n <= step:
        return steer_delta(x, r, d, alpha)
    x2 = x.reshape(n, W)
    r2 = None if r is None else r.reshape(n, W)
    out = torch.empty_like(x2)
    for s in range(0, n, step):
        e = min(n, s + step)
        out[s:e].copy_(steer_delta(x2[s:e], None if r2 is None else r2[s:e], d, alpha))
    return out.reshape(x.shape)


def _unreduced():
    """SGLang's deferred-all-reduce layer output type, if SGLang has it."""
    try:
        from sglang.srt.layers.communicator import UnreducedOutput, reduce_output
    except Exception:  # unit tests without sglang, or builds without the type
        return None, None
    return UnreducedOutput, reduce_output


# The attribute some SGLang builds set on a TP-partial layer output whose
# all-reduce is deferred into the next layer's fused add+norm. Steering a
# partial sum would edit one rank's share only, so a site that sees it
# refuses.
PARTIAL_SUM_MARKER = "_sglang_needs_allreduce_fusion"


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


_CAPTURE_MODE = []


def _sglang_capture_mode() -> bool:
    """True inside SGLang's graph capture (its warmup runs included), when
    this SGLang build says so."""
    if not _CAPTURE_MODE:
        try:
            from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
        except Exception:
            get_is_capture_mode = None
        _CAPTURE_MODE.append(get_is_capture_mode)
    fn = _CAPTURE_MODE[0]
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


class FiredCheck:
    """Proof that every steered local site ran, per forward.

    Each site bumps a Python counter when it runs. ``wrap_forward`` puts a
    wrapper on the backbone INSTANCE's ``forward`` (SGLang also calls
    ``backbone.forward(...)`` directly, where a registered forward hook
    would not run). After each forward that runs Python (the warmup and
    every CUDA-graph capture at boot, and every eager forward) it raises if
    a steered local site did not fire. A graph replay runs no Python, so
    the check costs nothing there. Each forward inside a capture adds one
    entry to ``fired_at_capture``: the token count, the sites that fired
    and the sites expected.
    """

    def __init__(self, layer_ids, what="steered layer"):
        self.layer_ids = sorted(int(i) for i in layer_ids)
        self.what = what
        self.fired = dict.fromkeys(self.layer_ids, 0)
        self.tokens = None
        self.depth = 0
        self.forwards = 0
        self.fired_at_capture = []
        self.on_capture = None  # callable(), e.g. rewrite the manifest

    def bump(self, layer_id, x):
        self.fired[layer_id] = self.fired.get(layer_id, 0) + 1
        if self.tokens is None:
            self.tokens = int(x.shape[0])

    def begin(self):
        self.depth += 1
        if self.depth == 1:
            for k in self.fired:
                self.fired[k] = 0
            self.tokens = None

    def end(self):
        self.depth -= 1
        if self.depth:
            return
        self.forwards += 1
        missing = [i for i in self.layer_ids if not self.fired.get(i)]
        stream_cap = _capturing()
        if stream_cap or _sglang_capture_mode():
            self.fired_at_capture.append({
                "tokens": self.tokens,
                "fired": len(self.layer_ids) - len(missing),
                "expected": len(self.layer_ids),
                "stream_capturing": bool(stream_cap),
            })
            if self.on_capture is not None:
                self.on_capture()
        if missing:
            more = " ..." if len(missing) > 8 else ""
            raise RuntimeError(
                f"weightless: {len(missing)} {self.what}(s) did not run in this forward "
                f"(layers {missing[:8]}{more}); the model reached them another way (for "
                f"example two-batch overlap or a changed call style), so the steering would "
                f"be skipped. Failing closed."
            )

    def wrap_forward(self, module):
        """Wrap ``module.forward`` on the instance. Returns an undo callable."""
        orig = module.forward
        check = self

        @functools.wraps(orig)
        def forward(*args, **kwargs):
            check.begin()
            try:
                out = orig(*args, **kwargs)
            except BaseException:
                check.depth -= 1
                raise
            check.end()
            return out

        forward._weightless_fired_check = self
        module.forward = forward

        def undo():
            if module.__dict__.get("forward") is forward:
                del module.forward

        return undo

    def manifest(self):
        return {"expected_sites": len(self.layer_ids), "forwards_checked": self.forwards,
                "captures": list(self.fired_at_capture)}


class LayerDiag:
    """Eager-only per-layer statistics (WEIGHTLESS_STEER_DIAG=1).

    It reads tensors back to the host, so it is skipped while a CUDA graph
    is being captured; use it with --disable-cuda-graph. All sums are over
    rows (tokens, or token-streams on a per-stream site), in float64, a
    slice of rows at a time.

    With h the stream before the edit, h2 after it, d this layer's
    direction, d_next the next steered layer's direction, alpha the alpha in
    force and c = cos(d, d_next), per steered layer:

    - sum_abs_pre        |h.d|
    - sum_abs_post       |h2.d|                    (0 at alpha 1)
    - sum_rel_pre        |h.d| / |h|
    - sum_abs_shift_next |h2.d_next|
    - tokens             rows seen
    - sum_norm_xnew      |x'| (2^-8 of it bounds the bf16 rounding of x')
    - sum_abs_err        |h2.d - (1 - alpha)(h.d)|: the distance from the
                         exact edit. Near 0 on a steered layer at any
                         alpha; alpha x sum_abs_pre on a layer left
                         unedited. At alpha 1 it equals sum_abs_post.
    - sum_signed_post    sign(h.d) (h2.d). Over sum_abs_pre it is
                         (1 - alpha) on a steered layer (-1 at alpha 2) and
                         +1 on an unedited one.
    - sum_abs_pre_next   |h.d_next|
    - sum_abs_err_shift  |h2.d_next - (1 - alpha)(h.d_next)|
    - sum_abs_pred_shift |h.d_next - c (h.d)| = sqrt(1 - c^2) |h.e|, with e
                         the unit part of d_next orthogonal to d.
    - sum_round_bound    2^-8 sum_k |x'_k| |d_k|: a bound on what rounding x'
                         to bf16 can add to |h2.d| (elementwise, so much
                         tighter than 2^-8 |x'| when a few values are large)
    - sum_round_bound_next the same with d_next

    The one-layer shift probe. With the right direction applied,
    sum_abs_err_shift = alpha x sum_abs_pred_shift (up to rounding). Had
    the site applied d_next instead of d, it would sit at the rounding
    floor. So the probe separates the two only where alpha x
    sum_abs_pred_shift, which scales with alpha sqrt(1 - c^2), is well
    above that floor (``layer_verdict``). It catches a slip inside the site
    (the next stack row read at this layer). It cannot catch a shift in the
    loader (``direction.N`` stored at the wrong stack row), because the edit
    and this record read the same stack: that case is caught by checking
    the steered layer range against the file (the real-file tests).

    Top level of the file: ``alpha``, and per layer ``cos_next``,
    ``next_layer`` and ``shift_scale`` = alpha sqrt(1 - c^2).
    """

    FIELDS = ("sum_abs_pre", "sum_abs_post", "sum_rel_pre", "sum_abs_shift_next", "tokens",
              "sum_norm_xnew", "sum_abs_err", "sum_signed_post", "sum_abs_pre_next",
              "sum_abs_err_shift", "sum_abs_pred_shift", "sum_round_bound", "sum_round_bound_next")

    def __init__(self, out_dir: str, layer_ids):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"weightless-diag-{os.getpid()}.json")
        self.layer_ids = sorted(layer_ids)
        self.last = self.layer_ids[-1]
        self.stats = {i: dict.fromkeys(self.FIELDS, 0.0) for i in self.layer_ids}
        for s in self.stats.values():
            s["tokens"] = 0
        self.cos_next = {}
        self.next_layer = {}
        self.alpha = None
        self.forwards = 0
        self._lock = threading.Lock()

    def record(self, layer_id, x, r, x_new, d, d_next, alpha=None, next_layer_id=None):
        if _capturing():
            return
        if x.numel() == 0:
            return
        a = 1.0 if alpha is None else float(alpha)  # host read: eager only
        self.alpha = a
        W = x.shape[-1]
        n = x.numel() // W
        x2, xn2 = x.reshape(n, W), x_new.reshape(n, W)
        r2 = None if r is None else r.reshape(n, W)
        dd = d.double()
        dn = d_next.double() if d_next is not None else None
        c = float(dd @ dn) if dn is not None else 0.0
        if dn is not None:
            self.cos_next[layer_id] = c
            self.next_layer[layer_id] = next_layer_id
        step = chunk_rows(2 * W)  # float64 rows: twice the bytes of fp32
        with torch.no_grad():
            acc = torch.zeros(len(self.FIELDS), dtype=torch.float64, device=x.device)
            for s in range(0, n, step):
                e = min(n, s + step)
                if r2 is None:
                    h, h2 = x2[s:e].double(), xn2[s:e].double()
                else:
                    rr = r2[s:e].double()
                    h, h2 = x2[s:e].double() + rr, xn2[s:e].double() + rr
                    del rr
                pc, qc = h @ dd, h2 @ dd
                pre = pc.abs()
                rel = pre / h.norm(dim=-1).clamp_min(1e-30)
                xnorm = xn2[s:e].double().norm(dim=-1)
                err = (qc - (1.0 - a) * pc).abs()
                signed = torch.sign(pc) * qc
                if dn is not None:
                    pn, qn = h @ dn, h2 @ dn
                    shift, pre_n = qn.abs(), pn.abs()
                    err_s = (qn - (1.0 - a) * pn).abs()
                    pred_s = (pn - c * pc).abs()
                else:
                    shift = pre_n = err_s = pred_s = torch.zeros_like(pre)
                xa = xn2[s:e].double().abs()
                rb = xa @ dd.abs()
                rb_n = xa @ dn.abs() if dn is not None else torch.zeros_like(pre)
                acc += torch.stack([
                    pre.sum(), qc.abs().sum(), rel.sum(), shift.sum(),
                    torch.tensor(float(e - s), dtype=torch.float64, device=x.device),
                    xnorm.sum(), err.sum(), signed.sum(), pre_n.sum(), err_s.sum(), pred_s.sum(),
                    2.0 ** -8 * rb.sum(), 2.0 ** -8 * rb_n.sum()])
            st = self.stats[layer_id]
            for k, v in zip(self.FIELDS, acc.tolist()):
                st[k] += int(v) if k == "tokens" else v
        if layer_id == self.last:
            self.forwards += 1
            self.dump()  # every forward: eager-only diagnostics, cost is irrelevant

    def dump(self):
        with self._lock:
            a = self.alpha
            payload = {
                "pid": os.getpid(),
                "forwards": self.forwards,
                "at": time.time(),
                "alpha": a,
                "cos_next": {str(i): c for i, c in self.cos_next.items()},
                "next_layer": {str(i): n for i, n in self.next_layer.items()},
                "shift_scale": {
                    str(i): (abs(a) * math.sqrt(max(0.0, 1.0 - c * c)) if a is not None else None)
                    for i, c in self.cos_next.items()},
                "layers": {str(i): dict(s) for i, s in self.stats.items()},
            }
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, self.path)


def layer_verdict(st, alpha, rel_bound=1e-2, sign_tol=0.02, probe_factor=10.0, floor="norm",
                  cos_next=None, cos_limit=0.99):
    """Judge one layer's diag sums (a dict as LayerDiag writes it) at
    ``alpha``. Returns a dict of numbers and verdicts.

    ``floor`` picks the bf16 rounding floor: "norm" is 2^-8 x
    sum_norm_xnew (the rule of the first release's unit test);
    "elementwise" is sum_round_bound (and sum_round_bound_next for the
    probe), which is also a strict bound and much tighter on streams with a
    few very large values.

    The checks, in this order:

    - edit, first against rel_bound: sum_abs_err <= rel_bound x sum_abs_pre
      passes whatever the floor is. A layer left unedited gives alpha x
      sum_abs_pre, so this pass is decisive while rel_bound <= 0.25 x
      |alpha| (alpha 0 is never decisive).
    - edit, the bf16 floor fallback, only for a layer that misses
      rel_bound and only when the floor is above rel_bound x sum_abs_pre
      (``on_floor``): the layer is judged against 2 x the floor. It passes
      when sum_abs_err <= 2 x floor and 2 x floor <= 0.25 x |alpha| x
      sum_abs_pre, and is "not decisive" when it is within 2 x floor but
      2 x floor is above that cap (such an error cannot be told from an
      unedited layer). A layer above both rel_bound and 2 x floor fails:
      rounding cannot explain its error.
    - sign: sum_signed_post / sum_abs_pre within max(sign_tol, bound /
      sum_abs_pre) of (1 - alpha), bound being the one the edit was judged
      on (rel_bound x sum_abs_pre, or 2 x floor on the floor). Its distance
      from (1 - alpha) is at most sum_abs_err / sum_abs_pre. It is "not
      decisive" whenever that bound is above 0.25 x |alpha| x sum_abs_pre
      (an unedited layer gives +1, only |alpha| away from 1 - alpha).
    - probe (the one-layer shift probe): pred = |alpha| x
      sum_abs_pred_shift is what a correct site gives; it scales with
      alpha sqrt(1 - c^2). The probe's bound is rel_bound x
      sum_abs_pre_next, or 2 x the floor when that floor is above it. The
      probe is decisive when pred >= probe_factor x that bound, and passes
      when sum_abs_err_shift >= 0.5 x pred (a site that applied d_next gives
      about 0). With ``cos_next`` (the file's per-layer cosine between d and
      d_next) above ``cos_limit`` in magnitude, the two directions cannot be
      told apart: the probe is "not decisive" and the cosine is reported.
    """
    if floor not in ("norm", "elementwise"):
        raise ValueError(f"floor={floor!r}")
    a = abs(float(alpha))
    pre = float(st["sum_abs_pre"])
    norm_floor = 2.0 ** -8 * float(st["sum_norm_xnew"])
    fl = norm_floor if floor == "norm" else float(st["sum_round_bound"])
    fl_n = norm_floor if floor == "norm" else float(st.get("sum_round_bound_next", 0.0))
    err = float(st["sum_abs_err"])
    rel = rel_bound * pre
    cap = 0.25 * a * pre

    on_floor = err > rel and fl > rel
    bound = 2.0 * fl if on_floor else rel
    out = {"err_rel": err / pre if pre else None, "bound_rel": bound / pre if pre else None,
           "floor_rel": fl / pre if pre else None, "on_floor": on_floor}
    decisive = pre > 0 and bound <= cap
    if pre <= 0:
        out["edit"] = "not decisive"
    elif err > bound:
        out["edit"] = "fail"
    else:
        out["edit"] = "pass" if decisive else "not decisive"
    if pre > 0:
        ratio = float(st["sum_signed_post"]) / pre
        tol = max(sign_tol, bound / pre)
        sign = "pass" if abs(ratio - (1.0 - float(alpha))) <= tol else "fail"
        out.update(signed_ratio=ratio, sign_tol=tol, sign=sign if decisive else "not decisive")
    else:
        out.update(signed_ratio=None, sign_tol=None, sign="not decisive")
    pre_n = float(st.get("sum_abs_pre_next", 0.0))
    if pre_n > 0:
        b_s = rel_bound * pre_n
        if fl_n > b_s:
            b_s = 2.0 * fl_n
        pred = a * float(st["sum_abs_pred_shift"])
        got = float(st["sum_abs_err_shift"])
        out.update(probe_pred_rel=pred / pre_n, probe_rel=got / pre_n, probe_bound_rel=b_s / pre_n)
        if cos_next is not None:
            out["cos_next"] = float(cos_next)
        if cos_next is not None and abs(float(cos_next)) > cos_limit:
            out["probe"] = "not decisive"
        elif pred >= probe_factor * b_s:
            out["probe"] = "pass" if got >= 0.5 * pred else "fail"
        else:
            out["probe"] = "not decisive"
    else:
        out["probe"] = "none"  # last steered layer on this rank: no next direction
    return out


class SiteError(RuntimeError):
    pass


def _found(value) -> str:
    """What a slot holds, for an error message: its type, or a tensor's
    dtype and shape (never its values)."""
    if isinstance(value, torch.Tensor):
        return f"a {value.dtype} tensor of shape {tuple(value.shape)}"
    return "None" if value is None else f"a {type(value).__name__}"


def check_slot(value, *, row, layer_id, slot, role, width, per_stream=False):
    """Raise SiteError unless ``value`` is a floating-point tensor whose last
    dimension is ``width`` (and, with ``per_stream``, that is 3-D: [T,
    streams, width]), the stream the row says this slot holds. An 8-bit
    float (fp8) is refused too: the edit needs 16 bits or more.

    A site runs it once per steered layer, at the layer's first forward (the
    warmup at boot). A row whose tuple layout does not match the layer (a
    bool, None or an integer tensor where the row expects the hidden or the
    residual stream, or a stream of another width) then stops the boot
    instead of editing the wrong value. It reads only the value's type,
    dtype and shape: no host sync, nothing recorded into a CUDA graph.
    """
    if (isinstance(value, torch.Tensor) and value.is_floating_point() and value.element_size() >= 2
            and value.dim() >= 1
            and value.shape[-1] == width and (not per_stream or value.dim() == 3)):
        return
    want = f"[T, streams, {width}]" if per_stream else f"[..., {width}]"
    raise SiteError(
        f"weightless: row {row or 'unnamed'}, decoder layer {layer_id}: slot {slot} should hold "
        f"the {role} stream, expected {want} (a floating-point tensor), found "
        f"{_found(value)}. The layer's output does not match the row, so the edit would land "
        f"on the wrong value. Failing closed."
    )


def make_site(owner, layer_id: int, *, next_layer_id=None, diag=None, kernel="torch",
              width=None, per_stream=False, check=None, row=None, hidden_slot="hidden",
              residual_slot=None, residual_may_be_none=False):
    """The steering site for global layer ``layer_id``: ``site(x, r=None)``
    returns the edited x. ``r``, when given, is the residual that the next
    layer adds to x; it is read, never changed.

    ``owner`` carries ``_steer_stack`` [L, 1, W] fp32 and ``_steer_alpha``
    (0-d fp32). Both are read through ``owner`` on every call (a view, no
    kernel), so a later in-place ``copy_`` into the buffers reaches
    captured graphs. ``kernel="triton"`` uses the fused kernel for CUDA
    tensors, ``"torch"`` the bounded torch path; both give the same bits.
    ``check`` is the FiredCheck this site reports to.

    ``row`` (the row name), ``hidden_slot`` and ``residual_slot`` (where the
    two streams come from: a tuple index, or a name) are used by the check
    at the first call: x must be a floating-point tensor of the stream
    width, and so must r when ``residual_slot`` is set (None is accepted
    there only with ``residual_may_be_none``) or when r is given. Later
    calls skip that check (a flag per site, so per steered layer).
    """
    UnreducedOutput, reduce_output = _unreduced()
    rowname = row or "unnamed"
    unchecked = [True]

    def site(x, r=None):
        if UnreducedOutput is not None and isinstance(x, UnreducedOutput):
            # TP>1 deferred all-reduce: settle it first (the deepstack
            # pattern in qwen3_5.py). Same collective the next layer would
            # have run when all-reduce fusion is off.
            x = reduce_output(x)
        elif not isinstance(x, torch.Tensor):
            raise SiteError(
                f"weightless: decoder layer {layer_id} output is "
                f"{type(x).__name__} (for example a deferred MoE finalize handoff) in slot "
                f"{hidden_slot} of row {rowname}; this steering site cannot edit it. "
                f"Failing closed."
            )
        if getattr(x, PARTIAL_SUM_MARKER, False):
            raise SiteError(
                f"weightless: decoder layer {layer_id} returned a TP-partial sum whose "
                f"all-reduce is deferred to the next layer ({PARTIAL_SUM_MARKER}); "
                f"steering it would edit one rank's share only. Failing closed."
            )
        if unchecked[0]:  # this layer's first forward: the slots hold what the row says
            w0 = owner._steer_stack.shape[-1] if width is None else width
            check_slot(x, row=rowname, layer_id=layer_id, slot=hidden_slot, role="hidden",
                       width=w0, per_stream=per_stream)
            if r is not None or (residual_slot is not None and not residual_may_be_none):
                check_slot(r, row=rowname, layer_id=layer_id,
                           slot="residual" if residual_slot is None else residual_slot,
                           role="residual", width=w0)
            unchecked[0] = False
        if r is not None and r.shape != x.shape:
            raise SiteError(
                f"weightless: layer {layer_id} hidden {tuple(x.shape)} and "
                f"residual {tuple(r.shape)} differ in shape; refusing"
            )
        w = owner._steer_stack.shape[-1] if width is None else width
        if x.shape[-1] != w or (per_stream and x.dim() != 3):
            want = f"[T, streams, {w}]" if per_stream else f"[..., {w}]"
            raise SiteError(
                f"weightless: layer {layer_id} stream is {tuple(x.shape)}, expected {want}; "
                f"refusing"
            )
        if check is not None:
            check.bump(layer_id, x)
        d = owner._steer_stack[layer_id, 0]
        if kernel == "triton" and x.is_cuda:
            from .fused import steer_delta_fused
            x_new = steer_delta_fused(x, r, d, owner._steer_alpha)
        else:
            x_new = steer_rows(x, r, d, owner._steer_alpha)
        if diag is not None:
            d_next = owner._steer_stack[next_layer_id, 0] if next_layer_id is not None else None
            diag.record(layer_id, x, r, x_new, d, d_next, alpha=owner._steer_alpha,
                        next_layer_id=next_layer_id)
        return x_new

    return site


def make_post_layer_hook(owner, layer_id: int, next_layer_id=None, diag=None, kernel="torch", *,
                         arity=2, hidden_index=0, residual_index=1, width=None, per_stream=False,
                         check=None, row=None, residual_may_be_none=False):
    """A forward hook for decoder layer ``layer_id`` (global id): the tuple
    codec around ``make_site``.

    The layer returns a tuple of ``arity`` slots. The edit goes into
    ``output[hidden_index]`` with ``r = output[residual_index]`` (or None
    when ``residual_index`` is None); every other slot passes through
    unchanged. Any other output fails closed. At the layer's first forward
    the site checks that both slots hold a floating-point stream of the
    row's width (``check_slot``); ``residual_may_be_none`` lets the residual
    slot be None (a row whose layer returns the whole stream there).
    """
    site = make_site(owner, layer_id, next_layer_id=next_layer_id, diag=diag, kernel=kernel,
                     width=width, per_stream=per_stream, check=check, row=row,
                     hidden_slot=hidden_index, residual_slot=residual_index,
                     residual_may_be_none=residual_may_be_none)

    def hook(module, args, output):
        if not isinstance(output, tuple) or len(output) != arity:
            n = f" of {len(output)}" if isinstance(output, tuple) else ""
            raise SiteError(
                f"weightless: decoder layer {layer_id} returned {type(output).__name__}{n}, "
                f"expected a {arity}-tuple (hidden at {hidden_index}, residual at "
                f"{residual_index}); refusing to steer an unknown layer output"
            )
        x = output[hidden_index]
        r = output[residual_index] if residual_index is not None else None
        out = list(output)
        out[hidden_index] = site(x, r)
        return tuple(out)

    hook.site = site
    return hook

# The forward_wrap install mode. A forward hook registered with
# register_forward_hook runs only when the layer is called as layer(...)
# (nn.Module.__call__). Some SGLang model loops call layer.forward(...)
# instead (Nemotron-H), where such a hook never runs. So this mode puts a
# wrapper on the layer INSTANCE's forward attribute (never on the class):
# layer.forward(...) finds it directly, and layer(...) reaches it through
# __call__. The wrapper runs the original forward and passes its output
# through the same tuple codec a hook would (make_post_layer_hook), so the
# edit, the refusals, the fired count and the diagnostics are those of the
# hook mode. It is installed before CUDA-graph capture, so the edit is
# recorded into every captured graph.


class Undo:
    """Takes a wrapper off again. Callable, and also has ``.remove()`` like
    the handle ``register_forward_hook`` returns."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self):
        self._fn()

    remove = __call__


def wrap_forward(layer, codec, *, layer_id, with_kwargs=False):
    """Wrap ``layer.forward`` on the instance so that its output goes
    through ``codec(module, args, output)`` (a hook from
    ``make_post_layer_hook``), or ``codec(module, args, kwargs,
    output)`` with ``with_kwargs`` (the looped codec, which reads the loop
    index from the call). Returns an ``Undo``.

    Fails closed when the layer's forward is already wrapped by this plugin
    (steering twice) or when the layer takes no instance attributes.
    """
    where = f"weightless: decoder layer {layer_id}"
    if not hasattr(layer, "__dict__"):
        raise RuntimeError(f"{where} ({type(layer).__name__}) takes no instance attributes, so "
                           f"its forward cannot be wrapped. Failing closed.")
    current = vars(layer).get("forward")
    if current is not None and getattr(current, "_weightless_layer_id", None) is not None:
        raise RuntimeError(f"{where}: forward is already wrapped by the steering on this "
                           f"instance; refusing to steer twice. Failing closed.")
    orig = layer.forward  # the bound method, or an instance forward someone else set

    @functools.wraps(orig)
    def forward(*args, **kwargs):
        if with_kwargs:
            return codec(layer, args, kwargs, orig(*args, **kwargs))
        return codec(layer, args, orig(*args, **kwargs))

    forward._weightless_layer_id = int(layer_id)
    layer.forward = forward

    def undo():
        if vars(layer).get("forward") is forward:
            if current is None:
                del layer.forward
            else:
                layer.forward = current

    return Undo(undo)
