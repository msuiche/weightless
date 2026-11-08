"""Fused GLP post-layer edit (Triton). The default on CUDA devices.

One Triton program per token row. For a row up to MAX_BLOCK wide it reads
x and r once, writes x' once, and keeps the whole row in registers. A wider
row is done in blocks, in three passes over it. The torch path
(core.steer_delta) instead makes several passes over fp32 copies of the
rows.

It runs exactly the steps of core.steer_delta, in the same order:

    h  = f32(x) + f32(r)
    c  = dot_fixed(h, d)          # the order-independent fixed-point sum
    x' = round(f32(x) - (alpha * c) * d)

It is compiled with ``enable_fp_fusion=False``, so the multiply and the
subtract are two separate roundings, as they are in torch. With that, the
fused kernel and the torch path give the same bits (checked at start-up
by ``self_check`` and by the unit tests).

A second kernel, ``_glp_hc_mean_kernel``, is the DeepSeek-V4.1 row's edit of
the 4 hyper-connection copies with the coefficient of their mean
(archs/dsv41.py ``hc_mean_delta``, the same steps), with its own start-up
check, ``self_check_hc_mean``.

WEIGHTLESS_STEER_KERNEL picks the path: ``auto`` (the default: this kernel
when Triton and a CUDA device are there and the self-check passes,
otherwise torch), ``triton`` or ``torch``.
"""
from __future__ import annotations

import torch

from .core import fixed_point_bits, steer_delta

try:
    import triton
    import triton.language as tl
except Exception:  # CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def _glp_row_kernel(x_ptr, r_ptr, d_ptr, a_ptr, o_ptr, H, stride_x, stride_r, stride_o, BITS,
                        HAS_R: tl.constexpr, BLOCK: tl.constexpr, NBLK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        if NBLK == 1:
            # the whole row in registers: one read of x and r, one write
            m = offs < H
            x = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0).to(tl.float32)
            d = tl.load(d_ptr + offs, mask=m, other=0.0)
            h = x
            if HAS_R:
                h = x + tl.load(r_ptr + row * stride_r + offs, mask=m, other=0.0).to(tl.float32)
            # dot_fixed (core.py), step by step
            p = h * d
            pm = tl.max(tl.abs(p), axis=0)
            e = (pm.to(tl.int32, bitcast=True) >> 23) & 0xFF
            s = tl.minimum(tl.maximum(BITS + 126 - e, -100), 100)
            scale = ((s + 127) << 23).to(tl.float32, bitcast=True)
            inv = ((127 - s) << 23).to(tl.float32, bitcast=True)
            q = (p * scale).to(tl.int64)
            acc = tl.sum(q, axis=0)
            c = acc.to(tl.float32) * inv
            a = tl.load(a_ptr)
            t = a * c
            out = x - t * d
            tl.store(o_ptr + row * stride_o + offs, out.to(o_ptr.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m)
        else:
            # a row wider than BLOCK, in NBLK blocks and three passes: max |p|,
            # the integer sum, the write. The max and the int64 sum are exact,
            # so the blocks give the same bits as one pass.
            m = offs < H
            x = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0).to(tl.float32)
            d = tl.load(d_ptr + offs, mask=m, other=0.0)
            h = x
            if HAS_R:
                h = x + tl.load(r_ptr + row * stride_r + offs, mask=m, other=0.0).to(tl.float32)
            pm = tl.max(tl.abs(h * d), axis=0)
            for b in tl.static_range(1, NBLK):
                o = b * BLOCK + offs
                m = o < H
                x = tl.load(x_ptr + row * stride_x + o, mask=m, other=0.0).to(tl.float32)
                d = tl.load(d_ptr + o, mask=m, other=0.0)
                h = x
                if HAS_R:
                    h = x + tl.load(r_ptr + row * stride_r + o, mask=m, other=0.0).to(tl.float32)
                pm = tl.maximum(pm, tl.max(tl.abs(h * d), axis=0))
            e = (pm.to(tl.int32, bitcast=True) >> 23) & 0xFF
            s = tl.minimum(tl.maximum(BITS + 126 - e, -100), 100)
            scale = ((s + 127) << 23).to(tl.float32, bitcast=True)
            inv = ((127 - s) << 23).to(tl.float32, bitcast=True)
            m = offs < H
            x = tl.load(x_ptr + row * stride_x + offs, mask=m, other=0.0).to(tl.float32)
            d = tl.load(d_ptr + offs, mask=m, other=0.0)
            h = x
            if HAS_R:
                h = x + tl.load(r_ptr + row * stride_r + offs, mask=m, other=0.0).to(tl.float32)
            acc = tl.sum((h * d * scale).to(tl.int64), axis=0)
            for b in tl.static_range(1, NBLK):
                o = b * BLOCK + offs
                m = o < H
                x = tl.load(x_ptr + row * stride_x + o, mask=m, other=0.0).to(tl.float32)
                d = tl.load(d_ptr + o, mask=m, other=0.0)
                h = x
                if HAS_R:
                    h = x + tl.load(r_ptr + row * stride_r + o, mask=m, other=0.0).to(tl.float32)
                p = h * d
                acc += tl.sum((p * scale).to(tl.int64), axis=0)
            c = acc.to(tl.float32) * inv
            a = tl.load(a_ptr)
            t = a * c
            for b in tl.static_range(NBLK):
                o = b * BLOCK + offs
                m = o < H
                x = tl.load(x_ptr + row * stride_x + o, mask=m, other=0.0).to(tl.float32)
                d = tl.load(d_ptr + o, mask=m, other=0.0)
                out = x - t * d
                tl.store(o_ptr + row * stride_o + o, out.to(o_ptr.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m)


    @triton.jit
    def _glp_hc_mean_kernel(x_ptr, d_ptr, a_ptr, o_ptr, H, stride_xt, stride_xs, stride_ot,
                            stride_os, BITS, INV_S, S: tl.constexpr, BLOCK: tl.constexpr):
        # One program per token: its S hyper-connection copies, each H wide
        # (H <= BLOCK, the whole row in registers). The steps of
        # archs/dsv41.hc_mean_delta, in its order: the fp32 sum of the copies
        # in index order, times 1/S (exact, S a power of two), dot_fixed of
        # that mean with d, then every copy minus (alpha * c) * d.
        tok = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        m = offs < H
        d = tl.load(d_ptr + offs, mask=m, other=0.0)
        base = x_ptr + tok * stride_xt
        acc = tl.load(base + offs, mask=m, other=0.0).to(tl.float32)
        for k in tl.static_range(1, S):
            acc = acc + tl.load(base + k * stride_xs + offs, mask=m, other=0.0).to(tl.float32)
        h = acc * INV_S
        p = h * d
        pm = tl.max(tl.abs(p), axis=0)
        e = (pm.to(tl.int32, bitcast=True) >> 23) & 0xFF
        s = tl.minimum(tl.maximum(BITS + 126 - e, -100), 100)
        scale = ((s + 127) << 23).to(tl.float32, bitcast=True)
        inv = ((127 - s) << 23).to(tl.float32, bitcast=True)
        q = (p * scale).to(tl.int64)
        c = tl.sum(q, axis=0).to(tl.float32) * inv
        a = tl.load(a_ptr)
        t = a * c
        td = t * d
        obase = o_ptr + tok * stride_ot
        for k in tl.static_range(S):
            x = tl.load(base + k * stride_xs + offs, mask=m, other=0.0).to(tl.float32)
            out = x - td
            tl.store(obase + k * stride_os + offs,
                     out.to(o_ptr.dtype.element_ty, fp_downcast_rounding="rtne"), mask=m)


def available() -> bool:
    """Triton is importable and a CUDA device is visible."""
    return triton is not None and torch.cuda.is_available()


# Widest row kept whole in registers (Qwen3.8-27B's 5120 fits); wider rows
# (GLM-5.3-Flash's 16384 mHC stream) run in blocks of this size.
MAX_BLOCK = 8192


def launch_shape(width: int):
    """(BLOCK, NBLK) for a row of ``width`` elements."""
    block = min(triton.next_power_of_2(int(width)), MAX_BLOCK) if triton is not None else MAX_BLOCK
    return block, -(-int(width) // block)


def steer_delta_fused(x: torch.Tensor, r, d: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Same contract and same bits as core.steer_delta, for CUDA tensors.
    x may have any number of leading dims (a 3-D per-stream [T, S, H] is
    H-wide rows too); the edit is along the last one."""
    H = x.shape[-1]
    x2 = x.reshape(-1, H)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    r2 = None
    if r is not None:
        r2 = r.reshape(-1, H)
        if r2.stride(-1) != 1:
            r2 = r2.contiguous()
    out = torch.empty_like(x2)
    n = x2.shape[0]
    if n:
        rr = r2 if r2 is not None else x2
        block, nblk = launch_shape(H)
        _glp_row_kernel[(n,)](x2, rr, d, alpha, out, H, x2.stride(0), rr.stride(0), out.stride(0),
                              fixed_point_bits(H), HAS_R=r2 is not None,
                              BLOCK=block, NBLK=nblk, num_warps=8, enable_fp_fusion=False)
    return out.reshape(x.shape)


def self_check(device, dtype=torch.bfloat16, width=5120, rows=33, with_residual=True) -> bool:
    """Run both paths on the same random rows on ``device``. Return True
    when they give the same bits.

    ``width`` and ``with_residual`` must be the ones the model's sites run
    (the stream width, and whether the site passes a residual): the kernel
    is compiled per residual form and per block layout. It uses its own
    random generator, so the global random state is not touched. It also
    compiles the kernel for ``dtype`` and ``width`` before any CUDA graph is
    captured.
    """
    g = torch.Generator().manual_seed(20260925)
    x = torch.randn(rows, width, generator=g)
    r = torch.randn(rows, width, generator=g) * 4.0
    r[:, :: max(1, width // 3)] *= 400.0  # a few very large residual values
    if not with_residual:
        x = x + r  # the stream itself carries the large values
    x[0].zero_()
    r[0].zero_()  # one all-zero row
    d = torch.randn(width, generator=g, dtype=torch.float64)
    d = (d / d.norm()).float()
    x, r, d = x.to(dtype).to(device), r.to(dtype).to(device), d.to(device)
    rr = r if with_residual else None
    for a in (0.0, 0.5, 1.0, 2.0):
        al = torch.tensor(a, dtype=torch.float32, device=device)
        want = steer_delta(x, rr, d, al)
        got = steer_delta_fused(x, rr, d, al)
        if not torch.equal(got.view(-1).view(torch.uint8), want.view(-1).view(torch.uint8)):
            return False
    return True


def steer_hc_mean_fused(x: torch.Tensor, d: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """The DeepSeek-V4.1 mean-coefficient edit (archs/dsv41.hc_mean_delta) for
    CUDA tensors: x [T, S, W] (S a power of two, W <= MAX_BLOCK); the same
    bits as the torch path (checked at start-up by ``self_check_hc_mean``)."""
    T, S, W = x.shape
    if S < 1 or S & (S - 1) or W > MAX_BLOCK:
        raise RuntimeError(f"weightless: the fused hc-mean kernel takes S a power of two and "
                           f"W <= {MAX_BLOCK}, not S={S}, W={W}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty((T, S, W), dtype=x.dtype, device=x.device)
    if T:
        block = triton.next_power_of_2(int(W))
        _glp_hc_mean_kernel[(T,)](x, d, alpha, out, W, x.stride(0), x.stride(1), out.stride(0),
                                  out.stride(1), fixed_point_bits(W), 1.0 / S, S=S, BLOCK=block,
                                  num_warps=8, enable_fp_fusion=False)
    return out


def self_check_hc_mean(device, dtype=torch.bfloat16, width=5120, rows=33, copies=4) -> bool:
    """``self_check`` for the mean-coefficient edit: random [rows, copies,
    width] streams with a few very large values and one all-zero token, both
    paths at alpha 0, 0.5, 0.75, 1 and 2. True when they give the same bits.
    Also compiles the kernel for ``dtype`` and ``width`` before any CUDA graph
    is captured."""
    from .archs.dsv41 import hc_mean_rows
    if int(width) > MAX_BLOCK:
        return False
    g = torch.Generator().manual_seed(20261025)
    x = torch.randn(rows, copies, width, generator=g) * 4.0
    x[:, :, :: max(1, width // 3)] *= 400.0
    x[0].zero_()
    d = torch.randn(width, generator=g, dtype=torch.float64)
    d = (d / d.norm()).float()
    x, d = x.to(dtype).to(device), d.to(device)
    for a in (0.0, 0.5, 0.75, 1.0, 2.0):
        al = torch.tensor(a, dtype=torch.float32, device=device)
        want = hc_mean_rows(x, d, al)
        got = steer_hc_mean_fused(x, d, al)
        if not torch.equal(got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8)):
            return False
    return True
