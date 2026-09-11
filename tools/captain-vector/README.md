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

`bake` is the one command that produces something other than a view of the
GGUF — and even then the GGUF stays canonical: the adapter is a *derived
artifact*, regenerable from the GGUF plus the base checkpoint at any time:

```sh
python3 ../../weightless.py bake some.gguf --base Qwen/Qwen3.8-27B --out adapter/
```

It folds the vector into a rank-1 PEFT/LoRA adapter (`adapter_model.safetensors`
fp32 + `adapter_config.json` + `bake-report.json`), for serving stacks that
take adapters rather than control vectors. The math: `h ← h − α(h·d̂)d̂` at a
residual writer `h = Wx` is a weight edit `ΔW = −α·d̂(d̂ᵀW)`, which is exactly
LoRA with `lora_B = d̂`, `lora_A = −α·d̂ᵀW`, `r = lora_alpha = 1` (peft scaling
1.0 — α lives in `lora_A`, so **do not scale the adapter**). α defaults to the
GGUF's `glp.alpha_default`; `--alpha` overrides. Modules auto-detect per layer
from the base's shard index (the residual-writing set: `self_attn.o_proj` /
`linear_attn.out_proj` / `mlp.down_proj`); `--modules suf1,suf2` overrides.

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

The derivation methodology gates — null calibration, prompt-driven capture,
ship gates — are specified in `refusal-research/METHODOLOGY.md` (§18 and the
sections it builds on) and are required practice, not suggestions.

- `captain_vector.py` — the library and its CLI (single file on purpose)
- `calibrate_null.py` — measures what held/null ratio pure noise reaches
- `test_captain_vector.py` — self-test script; also runs in `../../tests/`
- `examples/` — a form-matched, benign prompt-pair template;
  `examples/peft_merge.py` loads a baked adapter with peft (attach, with/without
  generate, `merge_and_unload`) and enforces the no-rescale / pinned-revision
  gotchas
