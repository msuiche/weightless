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
