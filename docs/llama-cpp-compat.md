# llama.cpp compatibility: GLP control vectors

Status: verified against upstream master `b0dcb8192b` (2026-09-11, shallow
clone at `/tmp/llama.cpp`); runtime smoke on Homebrew build b8920
(`15fa3c493`), arm64 Mac. The GLP format itself is specified in
[`../spec/GLP.md`](../spec/GLP.md); this doc is only about what happens when a
GLP file meets stock llama.cpp.

## 1. What llama.cpp's control-vector path actually does

The apply op is addition, full stop:

```cpp
// src/llama-adapter.cpp:22-29
ggml_tensor * llama_adapter_cvec::apply_to(ggml_context * ctx, ggml_tensor * cur, int il) const {
    ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
        cur = ggml_add(ctx, cur, layer_dir);
    }
    return cur;
}
```

```
h  <-  h + scale * d               ADD — no dot product, no norm, no mode
```

Everything else about the path, from source:

- **Hook site.** `build_cvec` runs at the end of each layer, immediately
  after the attention and FFN writes are folded into the residual — e.g.
  `src/models/llama.cpp:220-224` (`cur = ggml_add(ctx0, cur, ffn_inp);` then
  `cur = build_cvec(cur, il);`, then `inpL = cur`). The *site* is the same
  tensor GLP calls `residual_stream_post_layer`. The *op* is not. ~126 arch
  files call `build_cvec` via `llm_graph_context` (`src/llama-graph.cpp:1508-1512`).
- **Scale.** `--control-vector` is scale 1.0 (`common/arg.cpp:2976-2983`).
  `--control-vector-scaled FNAME:SCALE` folds SCALE into the data at load:
  `dst[j] += src[j] * load_info.strength` (`common/common.cpp:2072`). There is
  no per-layer-scaled variant in current master; the only related flags are
  `--control-vector`, `--control-vector-scaled`, and
  `--control-vector-layer-range START END` (`common/arg.cpp:2976-3004`).
- **Layer range.** Default is all layers: start clamped to 1, end to
  `n_layer` when unset (`common/common.cpp:1463-1464`), inclusive. Layer 0
  can never carry a tensor (`src/llama-adapter.cpp:65`).
- **Layer mapping.** `direction.N` applies at layer `N` — the loader stores
  slot `N` at offset `(N-1)*n_embd` (`common/common.cpp:2070`) and the
  adapter fills `tensors[il]` from offset `(il-1)*n_embd`
  (`src/llama-adapter.cpp:127`); the two off-by-ones cancel. Measured, not
  just read: see `spec/GLP.md` §"The layer mapping, stated flatly".
- **Merging.** Multiple `--control-vector*` files are summed elementwise
  (`common/common.cpp:2107-2110`). Meaningful for additive vectors only.
- **Metadata.** The loader reads tensor names, dtype (must be F32), and shape
  (must be 1-D) — `common/common.cpp:2022-2064`. It never reads `glp.mode`,
  `glp.hook_point`, `glp.alpha_default`, or the base-model pin. The only
  semantic guard is an `n_embd` match against the model
  (`src/llama-adapter.cpp:110-113`).

## 2. project vs add

| | GLP `mode=project` | llama.cpp control vector |
|---|---|---|
| op | `h ← h − α(h·d̂)d̂` | `h ← h + scale·d` |
| coefficient | `h·d̂` — depends on the activation, per token | constant — same shift for every token |
| what it does to the d̂-component | removes it (α=1: exactly; α>1: over-removes) | adds a fixed multiple of d̂ regardless of what was there |
| strength regime | α ~1–4, dimensionless | scale relative to residual magnitudes; unrelated numbers |
| strength storage | separate `glp.alpha_default`, never folded into data | folded into the data at load |
| merging two vectors | invalid (two projections ≠ projection along the sum) | elementwise sum, supported |
| failure mode when misapplied | — | no error; coherent-but-wrong output |

These are different operations, not one operation with a different knob.
Projection is input-dependent: the amount removed along d̂ is `h·d̂`, whatever
each token's activation happens to carry. Addition is input-independent:
every token at every steered layer gets the same constant vector. A
projective vector applied additively does not "steer more weakly" — with a
positive scale it pushes *along* the measured direction (for a refusal
direction: toward refusal, the exact inverse of the intent), and with a
negative scale it subtracts a constant that has no relationship to how much
d̂-component any given token actually has.

Our vectors were derived **and validated** in projective mode — α scans,
coverage sweeps, and the refusal suites all measured `h − α(h·d̂)d̂`. Additive
application of the same tensors is an unvalidated semantic. Nobody has run
the suites against it, and the α that ships in the file means nothing under
addition.

## 3. Loading a GLP file with `--control-vector`: silently wrong

Empirical check, because "reads fine, applies wrong" is the claim that
matters. Setup: `stories260K.gguf` (llama arch, `n_embd` 64, 5 layers, the
only small generative GGUF in the local caches) plus a synthetic GLP-format
file — `glp.mode=project`, `glp.hook_point=residual_stream_post_layer`,
`direction.1..4` random unit vectors. Homebrew llama.cpp b8920, CPU,
`--seed 42`, prompt "Once upon a time":

- baseline: `...there was a little boy named Timmy. Timmy loved to play with
  his toys...`
- `--control-vector synth.gguf` (scale 1.0): different, still coherent text
  (`...there was a clear boy named Tom...`). **No warning anywhere** — stderr
  shows the model-load chatter and nothing about `glp.mode`. The file has the
  right tensor names, dtype, and shapes, so it loads and applies additively.
- `--control-vector-scaled synth.gguf:-4.0` — a magnitude in the projective
  α regime — collapses output to `...in appeate,,,,,, he was a other,,,,,,,
  that,,,, ililing moding keee,,,,,`. As an additive constant, per-layer
  shifts of that size are destructive; as a projection coefficient, α=4 is
  merely "remove the component four times over".

This is a toy model with random directions — it demonstrates the mechanics
(silent acceptance, additive apply, wrong α regime), not refusal behavior.
The one thing stock llama.cpp *does* catch is an `n_embd` mismatch; the first
run of this smoke built the synthetic vector at width 256 and the run failed
loudly with `apply: control vector n_embd does not match model`. Shape is
checked; semantics are not.

Per the GLP reader-conformance rules (`spec/GLP.md` §Reader conformance), a
reader that cannot project must fail on `mode=project`. Stock llama.cpp does
not read the key, so the failure is on us to prevent: **do not hand GLP files
to stock llama.cpp's `--control-vector`.**

Additive use of our vectors is not defensible as shipped. It would become
defensible only as a separate artifact: directions derived or at least
re-validated for additive application, at an additive-appropriate scale,
shipped with `glp.mode=add` so every conforming reader treats them as what
they are.

## 4. The supported route: bake → LoRA → `--lora`

llama.cpp has no projective control-vector apply, but it has full LoRA
support, and a projection at a residual writer is exactly a rank-1 LoRA:
`h ← h − α(h·d̂)d̂` at `h = Wx` is `ΔW = −α·d̂(d̂ᵀW)`, i.e.
`lora_B = d̂`, `lora_A = −α·d̂ᵀW`. That is what `captain-vector bake` emits
(dense models only — see `tools/captain-vector/README.md`).

```sh
# 1. GLP GGUF + pinned base revision -> PEFT adapter (needs torch+safetensors;
#    the pinned base snapshot must be in the local HF cache)
python3 weightless.py bake Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf \
    --base Qwen/Qwen3.8-27B --out adapter/

# 2. PEFT adapter -> llama.cpp LoRA GGUF (from the llama.cpp tree; needs
#    transformers, torch, gguf-py). Reads adapter_config.json; the base config
#    comes from base_model_name_or_path or --base <dir with config.json> —
#    config only, weights are not needed here.
python3 convert_lora_to_gguf.py adapter/ --outfile glp-49-lora.gguf

# 3. serve — default adapter scale is 1.0 (common/arg.cpp:2955)
llama-server -m qwen3.8-27b-q8_0.gguf --lora glp-49-lora.gguf
```

Why the chain type-checks, from source:

- bake writes peft-standard `base_model.model.<stem>.lora_A/lora_B.weight`
  names with `r=1, lora_alpha=1` (`captain_vector.py:1130-1135, 1165-1174`).
  `convert_lora_to_gguf.py` strips exactly that prefix/suffix pair
  (`convert_lora_to_gguf.py:269-276`), emits `<name>.lora_a` / `.lora_b`
  tensors, and writes `adapter.lora.alpha` from `lora_alpha` (= 1.0)
  (`convert_lora_to_gguf.py:426-428, 525-528`).
- At runtime llama.cpp computes `res = Wx + scale·B(Ax)` with
  `scale = adapter_scale·alpha/rank` (`src/llama-graph.cpp:1514-1543`,
  `src/llama-adapter.h:53-57`). With `alpha = rank = 1` and the default
  adapter scale, scale = 1.0 — the baked α (which lives inside `lora_A`)
  applies exactly. **Do not use `--lora-scaled`**; scaling the adapter scales
  α on top of the baked value.
- Load-time checks: `general.type=adapter`, `adapter.type=lora`, arch match,
  and every LoRA tensor name/shape matched against the base model
  (`src/llama-adapter.cpp:202-216, 330-367`). A wrong-checkpoint adapter with
  the same dims is *not* flagged — which is why bake's pinned-revision check
  fails closed upstream of this.

One honest caveat on semantics: the baked adapter projects each residual
*writer's* output (`self_attn.o_proj`, `mlp.down_proj`) — Arditi et al.'s
weight-orthogonalization form — while runtime GLP steering projects the
*accumulated post-layer residual*. The post-layer residual is a sum with no
single `W` behind it, so it is not expressible as a rank-1 LoRA
(`spec/GLP.md` §"Naming"). bake validates the per-writer identity
`B(Ax) == h − α(h·d̂)d̂` numerically on the first and last baked matrix
(`bake-report.json` → `roundtrip`); the writer-vs-stream difference is real
but is the documented, validated baked form — and it is the only GLP-derived
artifact stock llama.cpp can serve.

The runtime-projection path in llama.cpp exists only in the fork
(`github.com/msuiche/llama.cpp`, currently private): projective branch in
`apply_to()`, `glp.*` metadata gating, and pinning tests — the change list is
in `spec/GLP.md` §Implementations.

## 5. Verification status

| claim | how verified |
|---|---|
| llama.cpp cvec apply is addition only | source, `src/llama-adapter.cpp:22-29` @ `b0dcb8192b` |
| hook site = post-layer residual | source, `src/models/llama.cpp:220-224` + `llama-graph.cpp:1508-1512` |
| scale/layer-range/merge/metadata behavior | source, `common/common.cpp:1462-1477, 2003-2120`, `common/arg.cpp:2976-3004` |
| GLP file loads silently, applies additively, wrong α regime | empirical, stories260K + synthetic GLP file, brew b8920 (§3) |
| bake → convert_lora_to_gguf → `--lora` chain | source-level: bake output format vs converter expectations vs runtime math; **not yet run end-to-end** (needs a base checkpoint + torch env) |
