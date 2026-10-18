<p align="center">
  <img src="logo.webp" alt="Captain Vector — the captain who steers your vectors" width="320">
</p>

# Captain Vector

Derives a projective control vector from a model plus a pair of prompt sets
(difference-of-means and friends, with a held-out-vs-null validation gate) and
writes it as a GGUF.

This is the **canonical producer** of the `glp.*` GGUF files that the serving
hotfixes in `weightless/patches/` consume. They gate on the `glp.*` metadata
keys, so the writer format here must not drift. `validate_gguf` is stdlib-only
on purpose — it runs where the file is served, no torch, no gguf package:

```sh
python3 ../../weightless.py validate some.gguf    # exit 1 on FAIL
```

Two more stdlib-only commands produce derived views of a shipped file — the
GGUF stays canonical; `inspect` and `export` never create a second source of
truth:

```sh
python3 ../../weightless.py inspect some.gguf              # metadata, per-layer
                                                           # norm, adjacent cosine,
                                                           # top dims [--json] [--topk N]
python3 ../../weightless.py export some.gguf --out v.safetensors
                                        # direction tensors as .safetensors,
                                        # glp.*/general.* provenance in __metadata__
```

`audit` answers the supply-chain question about a file you did not derive —
could this vector hide a backdoor the way a poisoned checkpoint can? — from
the file alone, in the same stdlib-only lane:

```sh
python3 ../../weightless.py audit some.gguf [--json] [--expect-sha256 HEX]
```

A backdoored checkpoint hides its delta inside gigabytes of edited weights;
a GLP file leaves the weights stock (hash-verifiable against the vendor) and
*is* the complete delta. The audit proves the file stays inside that
contract: every byte is accounted for (direction tensors and declared
metadata — trailing or hidden bytes fail), every tensor is a finite fp32
`direction.N` of unit norm, the mechanism is the unconditional,
input-independent `h ← h − α(h·d̂)d̂` (so there is no conditional a trigger
phrase could fire), the effective α stays in the published band, and the
content hash recomputes from the tensor bytes. It ends with what it cannot
prove — semantic intent, and the base model's own integrity — because
claiming otherwise would be the lie. Exit 1 on any FAIL; WARNs list things
worth a human's eyes (unknown metadata keys, URLs stashed in inert fields,
out-of-band alphas, sign-flipped or spiky directions).

`bake` is the one command that produces something other than a view of the
GGUF. Treat it as a **troubleshooting and interop option, not a serving
path**: reach for it to validate that a direction lands at all (attach the
adapter, diff generations), to probe a merge or re-quant for direction
survival, or to feed an external stack that only takes adapters. The GGUF
stays canonical — the adapter is a *derived artifact*, regenerable from the
GGUF plus the base checkpoint at any time:

```sh
python3 ../../weightless.py bake some.gguf --base Qwen/Qwen3.8-27B --out adapter/
```

For llama.cpp specifically — why `--control-vector` is the wrong door and the
bake → `convert_lora_to_gguf.py` → `--lora` chain is the right one — see
[`../../docs/llama-cpp-compat.md`](../../docs/llama-cpp-compat.md).

It folds the vector into a rank-1 PEFT/LoRA adapter (`adapter_model.safetensors`
fp32 + `adapter_config.json` + `bake-report.json`), for serving stacks that
take adapters rather than control vectors. The math: `h ← h − α(h·d̂)d̂` at a
residual writer `h = Wx` is a weight edit `ΔW = −α·d̂(d̂ᵀW)`, which is exactly
LoRA with `lora_B = d̂`, `lora_A = −α·d̂ᵀW`, `r = lora_alpha = 1` (peft scaling
1.0 — α lives in `lora_A`, so **do not scale the adapter**). α defaults to the
GGUF's `glp.alpha_default`; `--alpha` overrides. Modules auto-detect per layer
from the base's shard index (the residual-writing set: `self_attn.o_proj` /
`linear_attn.out_proj` / `mlp.down_proj`); `--modules suf1,suf2` overrides.

One caveat to know before you read anything into baked results: the adapter
projects each residual *writer's* output, not the accumulated residual stream
that the runtime hooks steer. By linearity, projecting every writer of a
layer at α equals projecting that layer's *new contributions* at α — but a
d̂-component already sitting in the incoming stream (from the embeddings, or
from a writer you chose not to bake) passes through untouched, where runtime
projection at the layer removes it regardless of origin. Close in practice
when writers re-inject the direction every layer; not bit-identical to
runtime steering. bake validates the per-writer identity numerically on the
first and last baked matrix (`bake-report.json` → `roundtrip`).

When *not* to use it: `bake` works for **dense models only**. On MoE models
the residual writers are per-expert — hundreds of matrices per layer — so a
bake explodes into thousands of rank-1s against quantized weights, and it is
not the weightless serving path anyway: the hotfixes in `patches/` steer the
GGUF directions at runtime, which keeps α tunable and stackable. Think of the
two formats as complements: LoRA covers dense checkpoints and the merge
ecosystem; the GLP GGUF covers everything LoRA can't practically reach — MoE,
quantized bases, runtime-tunable and multi-vector steering — and the format
is built to extend further (rank-k, per-expert) if those cases ever need a
baked form.

Two hard requirements, both consequences of `lora_A` carrying `W`:

- **The base must BE the pinned revision.** The GGUF pins
  `general.base_model.0.version`; bake resolves `--base` to that exact
  revision — an HF repo id must have the pinned snapshot in the local HF
  cache (fetch it first with `hf download <repo> --revision <sha>`), a local
  `snapshots/<sha>` directory is checked by name, a plain directory only warns
  (unverifiable). Baked against the wrong checkpoint the adapter is garbage
  and nothing downstream flags it, so a mismatch exits 1. `--revision <sha>`
  is the escape hatch, with a loud warning.
- **It needs `torch` + `safetensors`** (unlike `validate`/`inspect`/`export`,
  which stay stdlib-only for serving machines). Shards are streamed lazily via
  `safetensors.safe_open`; a 27B-scale bake runs on CPU.

Derivation itself needs `torch`, `transformers`, `safetensors`, `gguf`. The
full parameter reference and design rationale live in
`refusal-research/derivation/captain-vector/README.md`.

## Applying a GLP vector to a transformers model

`apply_transformers.py` hooks the residual stream of any HF decoder model and
applies `h ← h − α(h·d̂)d̂` at exactly the layers the GGUF lists — experiment-time
use, for measuring what a vector does on a checkpoint you have loaded anyway.
The production serving path remains the `patches/` hotfixes, which keep α
tunable and vectors stackable.

```python
from apply_transformers import glp_steered
with glp_steered(model, "v.gguf"):       # α defaults to glp.alpha_default
    model.generate(**enc)                # hooks come off on exit
```

The GGUF is parsed by the same stdlib reader `validate` uses, so the two can
never disagree about the format. Decoder layers are located by attribute path
(`model.model.layers` and friends — llama/qwen/gemma-style, plus the hybrid
variants); anything else fails with a message listing what was found. It
refuses, loudly: non-`project` mode (including legacy additive llama.cpp
vectors), a `hook_point` other than `residual_stream_post_layer`, a direction
width that isn't the model's hidden size, and layer ids past the model's
depth. Needs `torch` only — transformers itself is not imported. A runnable
end-to-end generate-with/without demo is `examples/apply_hf.py`.

## Calibration: winsorize, interpolate, KL damage, the knee

Four calibration additions, borrowed from p-e-w/heretic (AGPL-3.0) —
reimplemented from the published technique; no heretic code is used in this
repository:

- `--winsorize Q` (0 < Q < 1) clamps every captured activation row to the
  ±Q-quantile of its own component magnitudes, before estimation. Where
  `--mask` zeroes the top dims of the direction outright, winsorization clips
  each prompt's outliers and keeps every dimension in place. The
  massive-activation screen that recommends `--mask 0.005` on a dominated
  checkpoint also names `--winsorize 0.95` as the softer try — and flags a
  `--winsorize` on a checkpoint with no massive activations, where clipping
  only shaves signal. Idea credit: heretic's winsorization quantile.
- `--interpolate-dropped` fills layers the null gate dropped with
  `normalize(lerp(dirs[lo], dirs[hi], (L-lo)/(hi-lo)))` between the nearest
  kept neighbours on both sides, so a gate pass does not punch holes in the
  shipped span. Edge gaps with a kept neighbour on one side only stay
  dropped; the filled layers are recorded in the GGUF's description
  metadata. Idea credit: heretic's interpolation between adjacent layer
  directions.
- `kldiv.py` measures the quality cost of a shipped vector: mean per-token
  KL(stock || steered) over greedy continuations of harmless prompts. The
  stock model writes the reference continuations; both arms then
  teacher-force prompt+continuation through the same `_attach` hook path
  `glp.py` uses, and KL is averaged over the continuation positions. The
  number to publish next to a delivery tally. Idea credit: heretic's
  co-minimization of refusals and KL divergence from the original model.
- `knee_search.py` replaces the hand-run alpha ladder with a multi-objective
  Optuna TPE search: refusal rate on a harmful set and KL on harmless
  prompts, both minimized, over alpha (0.25–3.0 by default) and optionally a
  shipped-layer span (`--search-span LO-HI`). Refusals are classified on
  short generations (default 100 tokens — a refusal appears early) with a
  classifier written fresh for this file. The output is the trial table, the
  Pareto front, and a recommended knee: among trials within a small delivery
  tolerance of the best refusal rate, the lowest KL wins, ties to the lower
  alpha. optuna is optional — only the sampling loop needs it; the decision
  logic is pure and unit-tested. Idea credit: heretic's Optuna parameter
  search.

The derivation methodology gates — null calibration, prompt-driven capture,
ship gates — are specified in `refusal-research/METHODOLOGY.md` (§18 and the
sections it builds on) and are required practice, not suggestions.

- `captain_vector.py` — the library and its CLI (single file on purpose)
- `apply_transformers.py` — experiment-time helper: applies a GLP GGUF to any
  HF decoder model via forward hooks (torch only, transformers not imported)
- `calibrate_null.py` — measures what held/null ratio pure noise reaches
- `kldiv.py` — quality cost of a shipped vector: mean per-token
  KL(stock || steered) on harmless prompts (idea credit: heretic)
- `knee_search.py` — Optuna alpha/span search co-minimizing refusal rate and
  KL, recommends the knee; optuna optional (idea credit: heretic)
- `publish_hf.py` — ship helper: creates the HF repo (public, `gated="auto"`,
  RUA checkbox via the card's `extra_gated_*` frontmatter), uploads card +
  GGUF, then re-downloads the file and asserts sha256 byte-identity (stdlib +
  `huggingface_hub`; the 2026-10-11 Gemma-3 family ship used it)
- `test_captain_vector.py` — self-test script; also runs in `../../tests/`
- `examples/` — a form-matched, benign prompt-pair template;
  `examples/peft_merge.py` loads a baked adapter with peft (attach, with/without
  generate, `merge_and_unload`) and enforces the no-rescale / pinned-revision
  gotchas; `examples/apply_hf.py` generates with/without GLP steering on a
  live transformers model
