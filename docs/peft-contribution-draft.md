# Draft: upstream contribution to huggingface/peft

**Status: DRAFT ONLY — do not submit.** No issue opened, no PR filed. When/if
this goes upstream it goes as a docs issue (text below) proposing a short
"computed adapters" note; a docs PR only if maintainers ask for one.

The intent is ecosystem hygiene, not promotion: PEFT adapters are assumed to be
trained artifacts, and the library's defaults (scaling, init semantics) encode
that assumption. Computed adapters are legitimate users of the format whose
constraints are the inverse of trained ones; a short doc note prevents a class
of silent misuse for anyone who encounters one.

A runnable reference implementation of the adapter described below — a minimal
PEFT-style `ProjectionAdapterLayer` plus the numeric verification quoted in the
issue text — lives in-repo at
[`tools/captain-vector/examples/projection_adapter.py`](../tools/captain-vector/examples/projection_adapter.py)
(torch + peft; exits 0 when every assertion passes). It is part of the
proposal, not part of the submission.

---

## Proposed issue text

**Title:** Docs: a note on computed (non-trained) LoRA adapters — scaling and base-revision provenance

**Body:**

Most adapters in the wild are trained: `lora_A`/`lora_B` come out of gradient
descent, `lora_alpha / r` tunes the strength at serve time, and the base
revision matters only for reproducibility. A smaller class of adapters is
**computed**: the factors are derived algebraically and written directly into
the LoRA factorization, with no training run. peft loads and merges these
correctly today — the format has no notion of provenance — but two of the
ecosystem's default assumptions silently invert for computed adapters, and
both failure modes produce plausible-looking outputs rather than errors. I'd
like to propose a short docs note (snippet below, ready to paste) covering
them.

### The motivating example: a projection adapter

A widely used inference-time control is the projective steering edit

```
h ← h − α(h·d̂)d̂
```

applied to the residual stream by a runtime hook, where `d̂` is a unit
direction derived from activation statistics (e.g. difference-of-means) and
`α` a strength. The same edit can be exported ahead of time as a weight
delta. Where the residual stream is written by a linear map `h = Wx`
(`o_proj`, `down_proj`, …), substituting and collecting terms gives:

```
h − α(h·d̂)d̂  =  Wx − α·d̂(d̂ᵀWx)  =  (W − α·d̂d̂ᵀW)x

⟹  W′ = (I − α·d̂d̂ᵀ)W,   ΔW = −α·d̂d̂ᵀW = −α·d̂(Wᵀd̂)ᵀ
```

`ΔW` is an outer product — exactly rank-1, exactly LoRA-shaped, not an
approximation:

```
ΔW = BA    with   lora_B = d̂           (out×1)   unit norm
                  lora_A = −α·d̂ᵀW      (1×in)    strength baked in
                  r = 1, lora_alpha = 1          (peft scaling = 1.0)
```

This is a **projection adapter**: the projective edit carried as a per-writer
rank-1 weight delta. peft loads, applies, and `merge_and_unload()`s it with
zero special handling. On a synthetic tiny dense model (torch 2.14 / peft
0.20, float32), adapted writer outputs match the closed form
`h − α(h·d̂)d̂` to a max abs error of 2.4e-07 per writer and 3.0e-07
end-to-end through peft; `merge_and_unload()` matches the merged reference
bit-for-bit (0.0). Harness:
`tools/captain-vector/examples/projection_adapter.py` in the weightless repo.

### Scope of the equivalence — read this before using the form

The adapter is exact **per writer, on dense models**, and that is all it
claims:

- **It projects each writer's *new* contribution, not the accumulated
  residual stream.** After a baked layer, `h_out = h_in + Σᵢ(Wᵢ + ΔWᵢ)xᵢ`:
  every writer's contribution is projected, but a `d̂`-component already
  inside `h_in` — from the embeddings or from any writer not baked — passes
  through untouched. The runtime hook projects `h_out` *regardless of
  origin*. The two coincide only when the incoming stream carries no
  `d̂`-component; otherwise the baked adapter is close in practice (writers
  re-inject the direction every layer) but not bit-identical to runtime
  steering. The harness above demonstrates the gap numerically: with `h_in`
  carrying a unit `d̂`-component, the baked model preserves it while the
  runtime hook scales it by `(1 − α)`.
- **Dense models only, practically.** The closed form needs each writer's
  full on-disk `W`. On MoE models the residual writers are per-expert —
  hundreds of matrices per layer, each with a different `W`, each needing
  its own rank-1 computed from that `W` — so the adapter explodes in size
  and build time. Runtime steering is the only practical form there.
- **Writer hook points with on-disk weights only.** If the edit point is not
  `h = Wx`, or the base is quantized so the on-disk `W` is not the effective
  `W`, the closed form no longer applies.

### How this differs from the two things it looks like

- **Trained LoRA** is a *learned* additive delta, `y = Wx + (lora_alpha/r)·BAx`:
  both factors come out of gradient descent (B typically zero-initialized),
  `lora_alpha / r` is a serve-time strength knob, and loading against a
  nearby base revision is merely suboptimal. A projection adapter is the
  inverse on every axis: the factors are computed from one specific `W`, the
  strength is baked into `lora_A` so the scaling *must* be 1.0, and a wrong
  base revision is not suboptimal but silently wrong.
- **llama.cpp control vectors (CVC)** are activation *addition*,
  `h ← h + αv` applied to the stream — arithmetically a per-layer bias
  `b = αv`, not a weight delta. The LoRA additive path has no bias term, so
  a CVC cannot be expressed as a LoRA adapter at all. The projection edit
  can, precisely because it is multiplicative in `h` — the `(h·d̂)` inner
  product is what turns it into a rank-1 weight update. (The two are also
  different controls: addition injects a direction, projection removes one.)

### Two gotchas that follow from `lora_A` carrying the base weights

1. **Do not rescale.** The strength `α` is baked into `lora_A`. The config
   must ship `r = 1, lora_alpha = 1` so peft's scaling (`lora_alpha / r`) is
   1.0, and `use_rslora` must stay off. Any re-scaling multiplies `α` twice.
   Nothing warns.
2. **Pinned base revision.** Because `lora_A = −α·d̂ᵀW` is computed from one
   specific checkpoint, the adapter is only valid for that exact base
   revision. Against a different revision the applied edit is off by
   `d̂ᵀ(W_loaded − W_pinned)` — silently, since shapes still match. This is
   the inverse of a trained adapter, where a nearby base is merely
   suboptimal.

### Proposed doc snippet (ready to paste, e.g. under "Conceptual guides" or the LoRA docs as a short note)

> #### Computed (non-trained) adapters
>
> LoRA adapters are usually trained, but the format also carries *computed*
> adapters: low-rank updates derived algebraically rather than by gradient
> descent. Example: a projection adapter for activation steering folds the
> edit `h ← h − α(h·d̂)d̂` into the residual writers `h = Wx`, where it is
> exactly `ΔW = −α·d̂(d̂ᵀW) = BA` with `B = d̂`, `A = −α·d̂ᵀW` (an outer
> product, rank 1). peft loads and merges such adapters like any LoRA, with
> three caveats:
>
> - **Scaling is already baked in.** Such adapters ship `r = 1`,
>   `lora_alpha = 1` (scaling 1.0). Do not change `lora_alpha`, enable
>   `use_rslora`, or apply an adapter scale — the strength lives in `lora_A`
>   and any scaling double-applies it.
> - **Check the base revision.** Because `lora_A` is computed from the base
>   weights `W`, the adapter is valid only for the checkpoint revision it was
>   computed against (recorded in its `adapter_config.json` / metadata).
>   Loading it onto another revision fails *silently*: shapes match, the edit
>   is wrong. Pass the pinned `revision=` when loading the base.
> - **It projects each writer's output, not the accumulated residual
>   stream.** A `d̂`-component already in the incoming stream (embeddings,
>   untargeted writers) passes through, unlike a runtime hook applied to the
>   stream itself. On MoE models the per-expert writers make the baked form
>   impractical; it is a dense-model construction.
>
> Conversely, if an adapter's config carries `lora_alpha = 1`, `r = 1` and a
> pinned `revision`, treat it as computed and honor these constraints.

No code changes are requested — this is purely a documentation/provenance
note. Happy to open the docs PR if there's interest.

---

## Background for us (not part of the submission)

- Producer: `captain-vector bake` (`tools/captain-vector/captain_vector.py`),
  which emits exactly this form from a GLP control vector plus the pinned base
  checkpoint, with the no-rescale warning in the safetensors metadata.
- Reference implementation / verification harness (this repo, torch + peft):
  `tools/captain-vector/examples/projection_adapter.py` — minimal PEFT-style
  `ProjectionAdapterLayer`, numeric equivalence on a tiny dense model, peft
  round-trip via `get_peft_model` / `merge_and_unload()`, and a numerical
  demonstration of the writer-vs-stream scope caveat. Consumer-side walkthrough
  for real baked adapters: `tools/captain-vector/examples/peft_merge.py`.
  Both are folded into the unittest suite via `tests/test_projection_adapter.py`
  (skips where torch/peft are absent).
- Reference adapters (gated):
  `msuiche/Qwen3.8-27B-hedging-GLP-63-L1-63-a0.5`,
  `msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49`.
- Verification: synthetic tiny dense base + real `bake` output through peft
  0.20 / torch 2.14 — forward, merge, and steering-equivalence all exact to
  float32 noise (max abs err ≤ 3.0e-07 end-to-end; merge bit-exact at 0.0).
- Deliberately out of scope for the upstream note: the GLP GGUF format,
  runtime-steering hotfixes, and any mention of specific published adapters.
  The note stands on the algebra alone.
