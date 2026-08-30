# GLM-5.3 743B TP=4 lane — Int4-Int8Mix on 4x DGX Spark + GLP-77 steering

GLM-5.3 (743B total / ~40B active, `glm_moe_dsa`: MLA + DeepSeek-sparse
attention over a deepseek_v2 backbone, **no** hyperconnection widening) across
**four** Sparks (head + 3 workers over RoCE), with the GLP-77 projective
refusal vector. The serving stack is tonyd2wild's **hardware-validated**
(2026-08-29) Int4-Int8Mix TP4 recipe:
[GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark),
which derives its image + kernel overlays from
[GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s).
Cookbook: [GLM-5.3-DGX-Spark-Cookbook](https://github.com/tonyd2wild/GLM-5.3-DGX-Spark-Cookbook).
We reference, we do not fork.

**This lane needs 4 nodes** and is **wired + structure-tested, NOT
hardware-validated** (no boot has run our wiring; the anchor test happens at
staging — see below).

| file | what it is |
|---|---|
| `start-glm53xl-dspark.sh` | head+3-worker boot: preflight (overlay/image/weights/vector on all 4 nodes), stages the steering patch per node, workers first then the head |
| `.env.glm53xl.example` | full config with site values as `<...>` placeholders (the real `.env.glm53xl` is gitignored) |
| `../../patches/hotfix-glm53xl-steering-projective.py` | the steering hook; patches the overlay's `deepseek_v2.py` copy at staging time |
| `../../patches/reference/deepseek_v2_glm53xl.py` | byte-identical copy of the overlay `deepseek_v2.py` the stack serves — the structure test's reference |

## What you need before this lane can boot

1. **4 DGX Spark nodes** on the RoCE fabric.
2. **The image** `vllm-node-tf5-glm52-b12x:probe-modded` — a LOCAL build per
   tonyd2wild's GLM-5.2 QuantTrio repo (stock vLLM kernels fault on sm121;
   the 10 overlay files are required), present with the same image ID on all
   four nodes (the start script checks).
3. **The kernel overlay dir** (`KERNELS_DIR`, e.g. `~/glm-triton`) with the 10
   sm12x overlay files on every node, from the same repo.
4. **The weights**: [`2wild4tv/GLM-5.3-Int4-Int8Mix`](https://huggingface.co/2wild4tv/GLM-5.3-Int4-Int8Mix)
   (377.4 GiB) — local NVMe on the head, NFS-shared to the workers at the
   same relative path (nodes can't each hold 378 GB).
5. **The NCCL re-pin lib** at `$HF_CACHE/hub/nccl-2.30.4/libnccl.so.2`
   (LD_PRELOAD'd; their fabric dies on the image's NCCL).
6. **The vector**: [`msuiche/GLM-5.3-abliterated-GLP-77`](https://huggingface.co/msuiche/GLM-5.3-abliterated-GLP-77)
   (gated) at the root of the HF cache on ALL FOUR nodes:
   ```sh
   huggingface-cli download msuiche/GLM-5.3-abliterated-GLP-77 \
     --include "*.gguf" --local-dir ~/.cache/huggingface   # on ALL FOUR nodes
   ```

## Boot ritual (from their deploy report — each rule cost a boot)

- `cache_flusher.sh` (their repo's `launch/cache_flusher.sh`) on every node
  during boot + `sync; echo 3 > /proc/sys/vm/drop_caches` — the script
  attempts the drop non-interactively and warns when no flusher is running.
  GB10's host page cache eats CUDA-visible memory 1:1; gmu 0.91 only boots
  with the cache held down.
- `vm.swappiness=10` on every node (at 0–1 the box wedges with swap
  untouched).
- Tear down ALL ranks before relaunching any (the script refuses otherwise).
- Identical image ID on all four nodes (the script checks).
- Pin `--kv-cache-memory-bytes` — sizing KV off "currently free" memory makes
  boots page-cache-dependent. Note the pin makes vLLM skip memory profiling
  entirely.
- Per-node SM clocks under load: one quiet rank at a third clock gates the
  whole TP group (their head node hit this post-crash; a reboot fixed it).

## Steering: how it differs from the glm53 (Flash) lane

- **Different arch**: `glm_moe_dsa` on the `deepseek_v2` path, NOT mHC. The
  stream is the decomposed `(hidden_states, residual)` pair; the apply steers
  `hidden_states + residual` and writes back into `hidden_states` (same
  convention as the Qwen3.8 lane). GLP-77 directions are plain 6144-wide.
- **Staged, not entrypoint-patched**: the overlay `deepseek_v2.py` is
  bind-mounted read-only in the container, so the start script copies it,
  runs the hotfix on the copy inside a throwaway container (`--entrypoint
  python3`, the image has torch), and mounts the patched copy. Anchors fail
  closed at STAGING, before any GPU time — the first real run is the anchor
  test, and it cannot silently serve unsteered.
- Anchors are validated against the overlay file (vendored in
  `patches/reference/`) AND match vLLM v0.28.0's `deepseek_v2.py` (what the
  GLP-77 capture/eval ran on, 8×H100).
- **α=1.0 is calibrated — do not raise it**: on this model higher α makes
  refusal WORSE at full length (measured; the opposite of a strength dial
  here).

## GLP-77 measured effect (NVFP4, 8×H100, greedy, max 1400 new tokens)

From the derivation lane (refusal-research `experiments/20260829-glm53-flagship`,
vLLM 0.28.0, repo scorer / answer-audited split at `</think>`):

| alpha | refusal32 | cyber32 | benign32 | capability12 |
|-------|-----------|---------|----------|--------------|
| 0.0 (stock) | 1/32 / 1/32 | 18/32 | 31/32 | 12/12 |
| **1.0 (shipped)** | **12/32 / 6/32 audited** | **32/32** | 32/32 | 12/12 |

Null control (77 random unit vectors, same protocol): 1/32 — identical to
stock. Refusal on the 753B flagship is much stickier than on Flash; the cyber
cluster saturates (18/32 → 32/32) with benign and capability untouched. α>1
over-projects on this model.

## Validate

```sh
# structural, no GPU/torch: anchors + per-layer-loop + serving-class guards
python3 ../../scripts/test-glm53xl-steering-structure.py
```
