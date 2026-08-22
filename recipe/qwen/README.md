# Qwen TP=1 lane — Qwen3.8-27B-NVFP4 + steering on a single DGX Spark

The serving recipe is the
[drowzeys single-Spark recipe](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark)
(vLLM 0.27 GB10 build, MTP-3, fp8 KV — 31.7 tok/s single-stream, c=8 @ 256K).
This directory holds **our state on top**: the steering hotfix wiring and the
canonical env. If the node is rebuilt: clone the drowzeys recipe (or just
pull the pinned image and model), copy `.env.qwen.example` to `.env.qwen`,
fill in the `<...>` placeholders, and run `serve-qwen38.sh`.

| file | what it is |
|---|---|
| `serve-qwen38.sh` | drowzeys Profile A (throughput) with steering; `STEER_MODE=gguf\|lora` selects the mechanism |
| `.env.qwen.example` | full config with site values as `<...>` placeholders (the real `.env.qwen` is gitignored) |
| `../../patches/hotfix-qwen38-steering-projective.py` | the GGUF-mode hook; patches the container's `qwen3_next.py` **and** `qwen3_5.py` at boot |

## Steering modes (both hardware-validated 2026-08-22, NVFP4)

Measured on one DGX Spark against this exact stack, refusal32 suite,
heuristic classifier (full JSONs on the node; suites/results stay outside
this repo):

| arm | refusal32 delivery | mechanism |
|---|---:|---|
| stock | 4/32 (12.5%) | — |
| **GGUF cvec** | **24/32 (75.0%)** | hotfix, `layers=49 [10..58]` confirmed in the boot log |
| **LoRA** | **24/32 (75.0%)** | stock vLLM `--enable-lora`, coexists with MTP-3 |

Both match their bf16 eval measurements (62.5% CI [45.3, 77.1] and 71.9%
respectively) — the bf16→NVFP4 transfer holds for both formats.

- **`STEER_MODE=gguf`** (default): fail-closed hotfix, spec-enforced loader,
  steers the full residual stream. The boot log must read
  `Qwen refusal steering active: hook=post_layer alpha=1.000 ... layers=49`.
- **`STEER_MODE=lora`**: no patch, no anchor fragility on image bumps; the
  steered model is served as the `qwen-abliterated` module next to the stock
  base model (so one boot gives you both arms). vLLM needs the **peft
  directory layout**, not the bare safetensors:

  ```sh
  mkdir -p "$MODELS/lora/qwen-abliterated"
  cp Qwen3.8-27B-refusal-abliterated-lora-r1-down_proj-L1-63-a1.safetensors \
     "$MODELS/lora/qwen-abliterated/adapter_model.safetensors"
  # adapter_config.json: peft_type=LORA, r=1, lora_alpha=1, bias=none,
  # task_type=CAUSAL_LM, target_modules=["mlp.down_proj"]
  ```

  Do not scale the adapter above 1.0 — α=1 removes the component, above
  that it reflects (α=2 measured refusing 37.5% of harmless prompts).

## Steering vector

The shipping artifact is
**`Qwen3.8-27B-refusal-cvec-per_layer-L10-58-a1.gguf`** — per-layer diff of
means, 49 directions over layers 10–58, n_embd 5120, spec-conformant per
[`../../spec/CONTROL-VECTOR.md`](../../spec/CONTROL-VECTOR.md). Published at
[`msuiche/Qwen3.8-27B-abliterated-cvec`](https://huggingface.co/msuiche/Qwen3.8-27B-abliterated-cvec)
(gated — fetch with an HF token). Put it in `$MODELS/cvec/` so it lands under
the `/models` mount:

```sh
huggingface-cli download msuiche/Qwen3.8-27B-abliterated-cvec \
  --include "*.gguf" --local-dir "$MODELS/cvec"
```

**Alpha is 1.0 here.** The evals found α=4 drives 37.5% over-refusal on
harmless prompts on this model ("alpha=4 destroys this model", evals §5.0.5),
so the shipping vector is a re-export whose `dspark.alpha_default` is the
measured 1.0 — the direction itself is unchanged (same content hash). Do not
import the DSV4 lane's 4.0 — α is checkpoint-specific.

## Differences from the DSV4 lane worth knowing

- **The hook steers `hidden_states + residual`.** vLLM's Qwen3-Next stack
  keeps the residual stream decomposed (the add is fused into the next
  layernorm), and the derivation measured the *full* post-layer stream, so
  the apply reconstructs it per layer. Steering `hidden_states` alone would
  remove the component from only part of the stream.
- **The patch targets `qwen3_next.py` AND `qwen3_5.py`** — `Qwen3_5Model`
  inherits its forward from `Qwen3NextModel` but overrides `__init__` and
  skips the parent's, so the steering buffers must be registered in both
  (the first hardware boot proved it: missing the qwen3_5 half crashes at
  torch.compile). Inert unless `QWEN_STEER_PATH` is set.
- **`QWEN_STEERING_MODEL_PY` may need setting.** The default assumes a
  dist-packages install; if the image has a source install the default
  silently patches a file nobody imports. Discover the real path (command in
  `.env.qwen.example`) and confirm the boot log reads
  `Qwen refusal steering active: hook=post_layer alpha=1.000 ... layers=49`.

## Validate

```sh
# structural, no GPU/torch: anchors + per-layer-loop regression guard
python3 ../../scripts/test-qwen-steering-structure.py

# the exact injected loader against the real vector (needs torch)
QWEN_STEER_PATH=$MODELS/cvec/Qwen3.8-27B-refusal-cvec-per_layer-L10-58-a1.gguf \
  python3 ../../patches/hotfix-qwen38-steering-projective.py --check
```

Boot-time refusal probe against the served endpoint, and the full eval
suite, live in `../../../refusal-research/qwen/` (outside this repo).

## Notes inherited from the drowzeys recipe (measured there)

- **Pinned image is mandatory** — stock vLLM 0.27 has no sm_121a NVFP4
  kernels. The mirror `ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813`
  falls back to `eugr/spark-vllm-b12x:nightly-20260813`.
- **Tool calling needs `--tool-call-parser qwen3_xml`**, not `hermes`.
- **Warm the large-prefill path** before the first client prompt (the serve
  script does); a cold 20K-token first prompt can stall or garble.
- **MTP depth ≤ 3** — the model has one MTP layer; ≥4 crashes.
- Long-context (1M, YaRN, c≈2) is upstream's Profile B
  (`deploy/serve_longctx.sh`); not vendored — the steering wiring would be
  identical.
