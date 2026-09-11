# Draft: upstream contribution to huggingface/peft

**Status: DRAFT ONLY — do not submit.** No issue opened, no PR filed. When/if
this goes upstream it goes as a docs issue (text below) proposing a short
"computed adapters" note; a docs PR only if maintainers ask for one.

The intent is ecosystem hygiene, not promotion: PEFT adapters are assumed to be
trained artifacts, and the library's defaults (scaling, init semantics) encode
that assumption. Computed adapters are legitimate users of the format whose
constraints are the inverse of trained ones; a short doc note prevents a class
of silent misuse for anyone who encounters one.

---

## Proposed issue text

**Title:** Docs: a note on computed (non-trained) LoRA adapters — scaling and base-revision provenance

**Body:**

Most adapters in the wild are trained: `lora_A`/`lora_B` come out of gradient
descent, `lora_alpha` tunes the strength after the fact, and the base revision
matters only for reproducibility. A smaller class of adapters is **computed**:
the weights are derived algebraically and written directly into the LoRA
factorization, with no training run. peft loads and merges these correctly
today — the format has no notion of provenance — but two of the ecosystem's
default assumptions silently invert for computed adapters, and both failure
modes produce plausible-looking outputs rather than errors. I'd like to
propose a short docs note (snippet below, ready to paste) covering them.

The motivating example is activation steering. A widely used inference-time
control is the projective edit `h ← h − α(h·d̂)d̂`, where `d̂` is a unit
direction derived from activation statistics (e.g. difference-of-means) and
`α` a strength. When the residual stream is written by a linear map `h = Wx`
(`o_proj`, `down_proj`, …), the edit is *exactly* a rank-1 weight change:

```
h − α(h·d̂)d̂ = (W − α·d̂d̂ᵀW)x   ⟹   ΔW = −α·d̂(d̂ᵀW)
```

and `ΔW = BA` with `lora_B = d̂ (out×1)`, `lora_A = −α·d̂ᵀW (1×in)`, `r = 1`,
`lora_alpha = 1` is an exact LoRA representation of it — not an approximation.
Measured round-trip error is at float32 noise level (~1e-9 relative). peft
loads, applies, and `merge_and_unload()`s such an adapter with zero special
handling; verified against the closed form on a synthetic checkpoint.

Two gotchas follow from `lora_A` *carrying* the base weights:

1. **Do not rescale.** The strength `α` is baked into `lora_A`. The config
   must ship `r = 1, lora_alpha = 1` so peft's scaling (`lora_alpha / r`) is
   1.0, and `use_rslora` must stay off. Any re-scaling multiplies `α` twice.
   Nothing warns.
2. **Pinned base revision.** Because `lora_A = −α·d̂ᵀW` is computed from one
   specific checkpoint, the adapter is only valid for that exact base
   revision. Against a different revision the applied edit is off by
   `d̂ᵀ(W_loaded − W_pinned)` — silently, since shapes still match. This is the
   inverse of a trained adapter, where a nearby base is merely suboptimal.

And two cases where the equivalence does **not** hold, so a computed adapter
is the wrong form:

- **MoE models**: the residual writers are per-expert, so the edit needs one
  rank-1 per expert per layer (hundreds of matrices, each needing its own `W`)
  — practical only as runtime steering, not a baked adapter.
- **Non-writer hooks / nonlinear parametrizations** (e.g. editing at a point
  that isn't `h = Wx`, or quantized bases where the on-disk `W` is not the
  effective `W`): the closed form no longer applies.

### Proposed doc snippet (ready to paste, e.g. under "Conceptual guides" or the LoRA docs as a short note)

> #### Computed (non-trained) adapters
>
> LoRA adapters are usually trained, but the format also carries *computed*
> adapters: low-rank updates derived algebraically rather than by gradient
> descent (example: activation-steering directions folded into residual
> writers, where `h ← h − α(h·d̂)d̂` at `h = Wx` is exactly
> `ΔW = −α·d̂(d̂ᵀW) = BA` with `B = d̂`, `A = −α·d̂ᵀW`). peft loads and merges
> them like any LoRA, with two caveats:
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
>
> Conversely, if an adapter's config carries `lora_alpha = 1`, `r = 1` and a
> pinned `revision`, treat it as computed and honor both constraints.

No code changes are requested — this is purely a documentation/provenance
note. Happy to open the docs PR if there's interest.

---

## Background for us (not part of the submission)

- Producer: `captain-vector bake` (`tools/captain-vector/captain_vector.py`),
  which emits exactly this form from a GLP control vector plus the pinned base
  checkpoint, with the no-rescale warning in the safetensors metadata.
- Reference adapters (gated):
  `msuiche/Qwen3.8-27B-hedging-GLP-63-L1-63-a0.5`,
  `msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49`.
- Verification: synthetic tiny-Llama base + real `bake` output through peft
  0.20 / transformers 5.17 — forward, merge, and steering-equivalence all
  exact to float32 noise (see repo history for the harness).
- Deliberately out of scope for the upstream note: the GLP GGUF format,
  runtime-steering hotfixes, and any mention of specific published adapters.
  The note stands on the algebra alone.
