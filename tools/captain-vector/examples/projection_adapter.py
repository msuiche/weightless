#!/usr/bin/env python3
"""Reference implementation: the rank-1 projection adapter (PEFT-style).

Companion to docs/peft-contribution-draft.md -- the "computed adapter" that
proposal describes, as a minimal adapter class plus a numeric verification
harness on a tiny dense model with a synthetic direction.

The math. GLP steers at runtime by projecting the residual stream with a hook:

    h <- h - alpha * (h . d) d          d unit-norm, alpha a strength

`captain-vector bake` exports that same edit as a per-writer rank-1 weight
delta. At a residual writer h = Wx (o_proj, down_proj, ...):

    h - alpha*(h.d)d = Wx - alpha*d*(d^T W x) = (W - alpha*d d^T W) x
    =>  dW = -alpha * d d^T W = -alpha * d (W^T d)^T      (outer product, rank 1)
       dW = B A  with  B = d (out,1),  A = -alpha * d^T W (1,in),
       r = 1, lora_alpha = 1  (peft scaling 1.0 -- alpha is baked into A)

What this class is NOT:
  - Not a trained LoRA: the factors are computed algebraically from W, not
    learned; scaling is fixed at 1.0; the adapter is valid only for the exact
    base checkpoint A was computed from.
  - Not a llama.cpp control vector: CVC is activation ADDITION, h <- h + a*v,
    i.e. a bias b = a*v added to the layer output. The LoRA additive path has
    no bias term, so a CVC cannot be expressed as a weight delta at all; the
    projection edit can, because it is multiplicative in h.

Semantic scope (verified numerically in main()): the baked adapter projects
each writer's NEW contribution. A d-component already in the incoming
residual stream (embeddings, an unbaked writer) passes through untouched;
the runtime hook projects the ACCUMULATED stream and removes it regardless
of origin. Exact per writer on dense models; impractical on MoE, where the
writers are per-expert (hundreds of matrices per layer, each needing its
own W).

Usage:
    python3 projection_adapter.py        # needs torch + peft; runs the demo

Exit code 0 means every assertion passed; the printed max errors are the
float32 round-trip noise the proposal cites.
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionAdapterLayer(nn.Module):
    """A Linear residual writer wrapped with a baked rank-1 projection.

    PEFT-style: the base weight stays frozen and the edit rides an additive
    low-rank path, y = Wx + scaling * B(Ax). Unlike trained LoRA the factors
    are computed from W at construction, alpha is baked into A, and scaling
    is fixed at 1.0 (r = 1, lora_alpha = 1). merge() folds dW into a fresh
    Linear, giving W' = (I - alpha * d d^T) W.
    """

    scaling = 1.0  # alpha already lives in lora_A; anything else double-applies it

    def __init__(self, base: nn.Linear, direction: torch.Tensor, alpha: float):
        super().__init__()
        if base.bias is not None:
            raise ValueError("reference covers bias-free residual writers")
        d = direction.detach().to(torch.float32)
        d = d / d.norm()
        W = base.weight.detach().to(torch.float32)          # (out, in)
        if W.shape[0] != d.numel():
            raise ValueError(f"direction dim {d.numel()} != writer out dim "
                             f"{W.shape[0]} -- wrong base model")
        self.base = base
        self.alpha = float(alpha)
        self.register_buffer("direction", d)                # (out,)
        # peft names: lora_B (out, r), lora_A (r, in); r = 1.
        self.lora_B = nn.Parameter(d.unsqueeze(1), requires_grad=False)
        self.lora_A = nn.Parameter((-self.alpha * (d @ W)).unsqueeze(0),
                                   requires_grad=False)

    @property
    def delta_weight(self) -> torch.Tensor:
        """dW = -alpha * d d^T W, as (out, in)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merged_linear(self) -> nn.Linear:
        """A fresh Linear holding W' = W + dW = (I - alpha*d d^T) W."""
        out = nn.Linear(self.base.in_features, self.base.out_features,
                        bias=False)
        with torch.no_grad():
            out.weight.copy_(self.base.weight.to(torch.float32)
                             + self.delta_weight)
        return out


class TinyDenseModel(nn.Module):
    """Smallest thing with real residual writers: per block, o_proj writes
    the attention-ish contribution and down_proj writes the MLP one (up_proj
    feeds down_proj's input, so it is not itself a residual writer)."""

    def __init__(self, d_model=16, d_ff=32, n_layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.blocks = nn.ModuleList(nn.ModuleDict({
            "up_proj": nn.Linear(d_model, d_ff, bias=False),
            "o_proj": nn.Linear(d_model, d_model, bias=False),
            "down_proj": nn.Linear(d_ff, d_model, bias=False),
        }) for _ in range(n_layers))
        self.d_model = d_model

    def forward(self, h):
        for blk in self.blocks:
            h = h + blk["o_proj"](h)
            h = h + blk["down_proj"](F.gelu(blk["up_proj"](h)))
        return h


def _err(a, b):
    return float((a - b).detach().abs().max())


def main():
    torch.manual_seed(1)
    alpha = 0.5
    model = TinyDenseModel().eval()
    d = torch.randn(model.d_model)
    d = d / d.norm()

    # Wrap every residual writer with the computed adapter.
    adapters = {}
    for i, blk in enumerate(model.blocks):
        for suffix in ("o_proj", "down_proj"):
            adapters[(i, suffix)] = ProjectionAdapterLayer(blk[suffix], d,
                                                           alpha)

    # 1. Per-writer exactness: adapted output == closed-form projection of
    #    the writer's output, and merge() gives (I - alpha*d d^T) W exactly.
    print("== per-writer: adapted vs closed-form projection ==")
    worst = 0.0
    for (i, suffix), ad in adapters.items():
        x = torch.randn(4, ad.base.in_features)  # activations at this writer
        W = ad.base.weight
        h = ad.base(x)
        direct = h - alpha * (h @ d).unsqueeze(-1) * d      # runtime formula
        worst = max(worst, _err(ad(x), direct))
        merged = ad.merged_linear()
        worst = max(worst, _err(merged(x), direct))
        eye = torch.eye(W.shape[0])
        worst = max(worst, _err(merged.weight, (eye - alpha * torch.outer(d, d)) @ W))
        rank = torch.linalg.matrix_rank(ad.delta_weight).item()
        assert rank == 1, f"delta not rank-1: {rank}"
        print(f"  blocks.{i}.{suffix:<10} max|err| {_err(ad(x), direct):.3e}")
    assert worst < 1e-5, worst
    print(f"  worst across writers/merge/closed-form W': {worst:.3e}")

    # 2. peft interop: the same factors load into a real peft LoraModel and
    #    reproduce the projection end-to-end; merge_and_unload is exact.
    print("== peft round-trip (factors loaded into peft.LoraModel) ==")
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=1, lora_alpha=1, lora_dropout=0.0, bias="none",
                     target_modules=["o_proj", "down_proj"],
                     init_lora_weights=False)
    peft_model = get_peft_model(TinyDenseModel().eval(), cfg)
    with torch.no_grad():
        peft_model.load_state_dict(model.state_dict(), strict=False)
        n_injected = 0
        for name, mod in peft_model.named_modules():
            if not hasattr(mod, "lora_A"):
                continue
            parts = name.split(".")
            i = int(parts[parts.index("blocks") + 1])
            suffix = name.rsplit(".", 1)[-1]
            ad = adapters[(i, suffix)]
            mod.lora_A["default"].weight.copy_(ad.lora_A)
            mod.lora_B["default"].weight.copy_(ad.lora_B)
            n_injected += 1
    assert n_injected == len(adapters), n_injected
    assert cfg.lora_alpha / cfg.r == 1.0 and not cfg.use_rslora  # GOTCHA 1

    # Reference: the same model with every writer merged in closed form.
    ref = TinyDenseModel().eval()
    ref.load_state_dict(model.state_dict())
    with torch.no_grad():
        for (i, suffix), ad in adapters.items():
            ref.blocks[i][suffix].weight.copy_(
                ad.merged_linear().weight.to(ref.blocks[i][suffix].weight.dtype))

    h0 = torch.randn(4, model.d_model)
    e_peft = _err(peft_model(h0), ref(h0))
    merged_model = peft_model.merge_and_unload()
    e_merge = _err(merged_model(h0), ref(h0))
    print(f"  peft adapted vs merged reference:   max|err| {e_peft:.3e}")
    print(f"  peft merge_and_unload vs reference: max|err| {e_merge:.3e}")
    assert e_peft < 1e-5 and e_merge < 1e-5

    # 3. Semantic scope: the adapter projects each writer's NEW contribution;
    #    a d-component of the INCOMING stream passes through, where the
    #    runtime hook on the accumulated stream would scale it by (1-alpha).
    print("== caveat: incoming d-component passes through ==")
    ad = adapters[(0, "o_proj")]
    x = torch.randn(4, model.d_model)
    h_in = d.expand(4, -1).clone()                    # incoming stream IS d
    contribution = ad(x)                              # writer's new part
    baked_total = h_in + contribution
    runtime_total = (h_in + ad.base(x))
    runtime_total = runtime_total - alpha * (runtime_total @ d).unsqueeze(-1) * d
    passed = float((baked_total @ d).detach().mean())   # dot with d, averaged
    runtime = float((runtime_total @ d).detach().mean())
    expect_pass = 1.0 + (1 - alpha) * float((ad.base(x) @ d).detach().mean())
    print(f"  baked:  <h_out, d> = {passed:.6f}  (incoming component intact)")
    print(f"  runtime hook:      = {runtime:.6f}  (accumulated stream projected)")
    assert abs(passed - expect_pass) < 1e-5
    assert abs(passed - runtime) > 1e-2, "bake and runtime should differ here"

    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
