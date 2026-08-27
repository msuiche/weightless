# GLM-5.3-Flash TP=4 lane — FP8 on 4x DGX Spark + GLP-44 steering

GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`, ~306 GB FP8, 320B total / 18B
active, 45 layers, KDA + NoPE sparse MLA, mHC hyperconnections) on the day-0
image `vllm/vllm-openai:glm53-flash`, tensor-parallel across **four** Sparks
(head + 3 workers over RoCE), with the GLP-44 projective refusal vector
applied by a fail-closed boot hotfix.

**This lane needs 4 nodes.** 306 GB of FP8 weights do not fit 2x Spark
(~256 GB unified); at TP=4 that is ~77 GB of weights per 128 GB node, the
rest is KV + activation headroom. `machines.txt` listing two nodes means two
more must be racked and cabled before this lane can boot.

If the nodes are rebuilt: pull the image and model on all four, copy
`.env.glm53.example` to `.env.glm53`, fill in the `<...>` placeholders, and
run `start-glm53-flash-dspark.sh` on the head.

| file | what it is |
|---|---|
| `start-glm53-flash-dspark.sh` | head+3-worker boot: syncs env + hotfix to every worker, starts headless ranks 1–3, then the API rank 0 on the head |
| `.env.glm53.example` | full config with site values as `<...>` placeholders (the real `.env.glm53` is gitignored) |
| `../../patches/hotfix-glm53-steering-projective.py` | the steering hook; patches the container's `vllm/models/glm5next/nvidia/model.py` at boot |
| `../../patches/reference/glm5next.py` | the structure test's reference copy of that file — see the drift caveat below |

## Serve flags (from the official vLLM recipe)

[recipes.vllm.ai/zai-org/GLM-5.3-Flash](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash):
`--tensor-parallel-size 4 --kv-cache-dtype fp8` (fp8 KV is Blackwell-OK;
Hopper must not use it), `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'`,
`--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`.
The image must contain FlashInfer ≥ 0.6.17 (NoPE sparse MLA). Thinking is
always on; `reasoning_effort` low/high/max via chat template kwargs (default
max).

## Traps (each one documented before it costs a run)

- **α=2.0 is calibrated — and α≥2.5 GARBLES this model.** The cliff is
  abrupt (measured). Do not raise it; do not import another lane's alpha.
- **Reference is the PR source, not the image.** The vendored
  `patches/reference/glm5next.py` is vllm-project/vllm#53906 @ `142062f1`;
  the `glm53-flash` image predates that revision by a day. On first deploy,
  diff the image's file and re-vendor if it drifted:
  ```sh
  docker run --rm --entrypoint cat "$GLM53_IMAGE" \
    /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py \
    | diff - patches/reference/glm5next.py
  ```
  The hotfix's anchor check is fail-closed either way — drifted source aborts
  the boot when steering is requested instead of serving unsteered. Confirm
  the boot log reads `weightless GLP steering active: hook=post_layer
  alpha=2.000 ... layers=44`.
- **NCCL GID indexes drift** across reboots; this lane pins
  `NCCL_IB_GID_INDEX=3` on all four nodes (no sysfs auto-resolver). Re-verify
  after reboot.
- **Stream-order caveat:** the hotfix flattens the mHC stream HC-outer
  ([T, n, hidden] → n*hidden, stream k in columns [k*hidden:(k+1)*hidden]) —
  the convention the Qwen3.8-Flash-Next capture validated. It has not been
  re-measured for GLM-5.3; verify on first boot with a refusal32 A/B arm.

## Steering vector

The shipping artifact is
**`GLM-5.3-Flash-abliterated-GLP-44-L1-44-a2.gguf`** — per-layer
difference-of-means over the mHC stream (16384 = 4×4096), layers 1–44,
α=2.0, spec-conformant per [`../../spec/GLP.md`](../../spec/GLP.md).
Published at
[`msuiche/GLM-5.3-Flash-abliterated-GLP-44`](https://huggingface.co/msuiche/GLM-5.3-Flash-abliterated-GLP-44)
(gated — fetch with an HF token), at the **root of the HF cache on ALL FOUR
nodes** so it lands at `/cache/huggingface/` in every container:

```sh
huggingface-cli download msuiche/GLM-5.3-Flash-abliterated-GLP-44 \
  --include "*.gguf" --local-dir ~/.cache/huggingface   # on ALL FOUR nodes
```

The start script preflights its presence on every node when
`WEIGHTLESS_STEER_PATH` is set and refuses to boot otherwise (fail-closed).

## How the steering works on mHC

mHC (multi-hyperconnection) defers each layer's `hc_post` and fuses it into
the next layer's pre. The full post-layer stream an HF hook sees is the
parameter-free `layer.hc_post(hidden_states, residual, post, comb)`
materialization ([T, n, hidden], n=4 streams) — `MHCFusedPostPreOp` is
documented upstream as exactly `MHCPostOp` + `MHCPreOp`, so materializing and
letting the next layer take its standalone `hc_pre` is the same math, one
kernel less fused. The hotfix materializes per layer, projects the GLP
direction out of the flattened stream, and continues materialized. Stock
runs the LAST layer's `hc_post`+`hc_contract` inside the decoder, which would
escape steering — the hotfix defers it (the loop apply contracts after
steering, so layer 44 is covered). The apply is unconditional with a dense
zero-padded stack: the traced graph is identical for every layer set.
Buffers register on `Glm5NextModel` — the class whose forward contains the
apply (both serving wrappers delegate to it; no skip-parent trap).

## Validate

```sh
# structural, no GPU/torch: anchors + per-layer-loop + serving-class guards
python3 ../../scripts/test-glm53-steering-structure.py

# the exact injected loader against the real vector (needs torch)
WEIGHTLESS_STEER_PATH=$HF_CACHE/GLM-5.3-Flash-abliterated-GLP-44-L1-44-a2.gguf \
  python3 ../../patches/hotfix-glm53-steering-projective.py --check
```
