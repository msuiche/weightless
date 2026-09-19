# FAQ — GLP vectors and weightless

Collected from real questions (HF threads, DMs, issues). Short answers
here; the deep versions live in `spec/GLP.md`, the README's "Streams and
writers" section, `docs/llama-cpp-compat.md`, and `TROUBLESHOOTING.md`.

## What GLP is (and is not)

### How is GLP different from LoRA?

Different object. LoRA is a *trained* additive weight delta
(`W ← W + BA`). GLP is a *computed* projection applied to the residual
stream at inference (`h ← h − α(h·d̂)d̂`). Nothing is trained; the
directions come from a few hundred contrast prompts. A rank-1 LoRA file
can *carry* our projection per writer matrix — that is what
`captain-vector bake` exports — but bake is an interop/troubleshooting
option with a scope caveat (below), not the serving path.

### How is GLP different from llama.cpp control vectors (CVC)?

CVC *adds* a vector to the stream (`h ← h + αv`) — a constant shift,
arithmetically a bias. GLP *projects out* the component along a
direction, keyed on the model's own state: harmless prompt, small
component, almost nothing touched. This is why loading a projective
direction into an additive consumer is silently wrong — right tensor
names, right shapes, wrong operation, no error raised. The `glp.mode`
metadata key exists to make that fatal instead of silent; stock
llama.cpp ignores it, so do not serve GLP files through stock CVC. See
`docs/llama-cpp-compat.md`.

### Projection vs addition — why does it matter?

A constant shift fights the model blind: the refusal component varies
per token and per prompt, and a fixed subtracted vector over-cancels on
harmless prompts and under-cancels on loaded ones. Projection removes
exactly the component present. Arditi et al.'s own appendix measured the
consequence: activation addition bypasses refusal about as well but
inflates CE loss on harmless data; directional ablation is the surgical
one. Addition is also not expressible as a weight change at all (it is a
bias); projection is (`I − α·d̂d̂ᵀ`), which is why GLP can be baked.

### Does it work on MoE models? On looped/recursive models?

The runtime hook is architecture-blind — it acts on the accumulated
residual stream after all writers (attention, MoE experts, shared
layers) have merged, so dense and MoE are the same operation. Bake is
the opposite: exact on dense, impractical on MoE (hundreds of expert
writers per layer, one rank-1 each). Looped models (MoR, e.g.
Nanbeige4.2) work but refusal is *re-decided every loop pass*, not
carried — our measured median cross-pass direction cosine is 0.248 — so
single-pass coverage under-doses; steer all passes (or use rank-k with
per-pass directions, spec_version 2).

## Maintaining a vector

### I retrain my model with RL every week. Do I need to re-derive the vector?

Usually no. Split the answer:

- **The hook never moves.** GLP attaches by module path ("residual
  stream at decoder layer N"), not by anything address- or weight-like.
  Weight *values* can change every week; the architecture does not, so
  the apply point is stable.
- **The vector goes stale only when the steered behavior is retrained.**
  The directions are derived from activations, and they are robust: GLP-49
  survives int4 re-quantization (~12% weight perturbation, 3.6%
  intervention error) — far more displacement than a typical weekly RL
  run. Capability-focused rollouts (tool use, coding, math) leave the
  refusal direction alone for months in practice. Alignment-touching
  rollouts invalidate it (V4 → V4.1 wrote a new refusal skeleton; that
  direction needed re-derivation).
- **Cheap maintenance pattern**: after each training run, run the old
  vector against a small refusal suite. If the rate holds, ship as-is.
  If it drifted, re-derive — one captain-vector command, single-digit
  GPU-hours on a 27B, trivial next to the RL run itself. If you *bake*
  instead of hooking, re-bake after each run (the export is pinned to a
  weight revision); the hook does not care.

### What is the vector's effective range? (re-quants, merges, other models)

Valid for the exact base model and its re-quantizations (measured up to
int4). Weight *merges* partially work: error enters only through how far
the merge rotated the model's own direction off the stored one, and a
merge that keeps the base dominant perturbs less than int4 does — but a
rank-1 vector cannot cancel a rotated component fully, so expect partial
effect (this is exactly what external testers observed on Qwen+Gemma
glimmer merges). Not portable to a different model or size; the method
transfers, the file does not.

## Behavior and dosage

### The model still adds disclaimers / "as an AI" hedging after GLP. Why?

The main refusal direction is a category gate; hedging is a separate,
second direction. Soft rejections are seed-dependent because the model
sits near the decision boundary after the dominant component is removed
— sampling noise flips the hedge. We ship hedging vectors for exactly
this (derived from the model's own natural branch points, not
grammar-pinned capture — pinned capture yields a register axis, a
measured negative result). Rank-k files can carry both directions in one
artifact.

### What does α do, and what happens if I crank it?

α scales how much of the component is removed; α=1 is full projection of
the derived direction. Over-steering overshoots: at α=2 on the Qwen
27B, 37.5% of *harmless* prompts start getting refused (the model
refuses to help with sourdough). Calibrate per model — this is why
Inkling ships α=0.25 and the Qwen vectors ship α=1. `glp.alpha_default`
travels in the file; the CLI/env override it.

### Does steering damage termination (EOS) on long answers?

At high baked doses, yes — externally measured (credit: Rob E Lee's
OBLITERATUS re-measurement of the Qwen3.8-27B abliteration field,
including our GLP-49): free-running clean-EOS rate on fulfilled harmful
answers drops to ~26% for our vector vs 64% base; benign prompts
terminate fine. The damage is trajectory-localized and dose-dependent.
Mitigations: runtime α knob (lower dose), measure clean-stop rate
(`finish_reason == stop`) in your eval loop, and watch
position-decayed steering work in our backlog. His SFT arm keeps 91%
because termination is trained, not projected — if you need guaranteed
EOS integrity, that is the current ceiling.

## Measurement

### My refusal test gave a different verdict on a re-run. Is the vector flaky?

Probably not — refusal near the decision boundary is non-deterministic.
We watched a frontier model flip between refusal and compliance on
byte-identical requests across runs (and across endpoints: same weights,
different behavior per channel — log endpoint + model id, the probe
measures model+endpoint, not model alone). Single-draw verdicts are
noise on boundary prompts. Sample n>1 per prompt and report rates with
confidence intervals (`tools/sample_suite.py` in refusal-research does
exactly this, with an `unstable_prompts` flip detector).

### Does GLP hurt capability?

On benign suites, no at calibrated dose (32/32 benign holds across our
shipped vectors). The quant-cliff work showed something adjacent and
useful: refusal does not shift with bitrate, so a steering vector and a
fidelity vector are separable problems. Capability claims live in
`BENCHMARK.md`; the honest caveat is that cheap probes are blind to
long-form damage (knowledge-40 sees what capability12 cannot) — judge on
held-out knowledge tasks, not vibe checks.

## Serving

### How do I apply a GLP file?

- **vLLM**: the `weightless-steer/` plugin (`vllm.general_plugins`;
  set `WEIGHTLESS_STEER_PATH`, boot log confirms). Currently ships the
  nemotron_h arch adapter; more lanes are one module each.
- **HF transformers**: `glp.py` — `apply_glp(model, "repo/or/file.glp")`,
  context manager, works on any decoder model via
  `register_forward_hook` on the residual stream.
- **PEFT/LoRA pipeline**: `captain-vector bake` → rank-1 adapter
  (dense models; troubleshooting/interop path).
- **llama.cpp**: not supported for projection (see CVC answer above).

### What is bake's scope caveat, precisely?

Bake projects each writer matrix's *new contribution*
(`W' = (I − αd̂d̂ᵀ)W`); the incoming stream's component along d̂ passes
through untouched. The runtime hook projects the *accumulated* stream.
Measured demonstration: with a unit d̂-component arriving, bake leaves
⟨h,d̂⟩ = 0.947 where the hook leaves 0.447 at α=0.5. Exact per writer on
dense models; impractical on MoE; never the serving path.
