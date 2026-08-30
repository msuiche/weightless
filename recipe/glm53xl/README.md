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
staging — see below). As wired it serves MTP k=4 + fp8_ds_mla; upstream's
DFlash2 / NVFP4 KV lanes (finished 2026-08-29, after we vendored) are
documented below as **opt-in, not wired**.

| file | what it is |
|---|---|
| `start-glm53xl-dspark.sh` | head+3-worker boot: preflight (overlay/image/weights/vector on all 4 nodes), stages the steering patch per node, workers first then the head |
| `.env.glm53xl.example` | full config with site values as `<...>` placeholders (the real `.env.glm53xl` is gitignored) |
| `../../patches/hotfix-glm53xl-steering-projective.py` | the steering hook; patches the overlay's `deepseek_v2.py` copy at staging time |
| `../../patches/reference/deepseek_v2_glm53xl.py` | byte-identical copy of the overlay `deepseek_v2.py` the stack serves — the structure test's reference |
| `../../patches/vendor/patch_chat_template_thinking.py` | vendored upstream patch: adds the missing `enable_thinking` knob to the stock chat template (see below) |

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
6. **The vector**: [`msuiche/GLM-5.3-abliterated-cyber-GLP-77`](https://huggingface.co/msuiche/GLM-5.3-abliterated-cyber-GLP-77)
   (gated) at the root of the HF cache on ALL FOUR nodes:
   ```sh
   huggingface-cli download msuiche/GLM-5.3-abliterated-cyber-GLP-77 \
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
  whole TP group (their head node hit this post-crash — a hard-crashed node
  can come back in a degraded power state, 721 MHz under load with throttle
  reasons 0x0 and `-lgc` silently ignored; only a clean reboot fixed it).
  Check clocks under load before trusting any throughput number.
- Bare + cudagraph FULL is NOT automatically the safe mode:
  `FlashMLASparseBackend` only advertises `UNIFORM_BATCH`, so FULL silently
  escalates to `FULL_AND_PIECEWISE` — two graph sets, and that extra
  allocation is what took a node down upstream. MTP produces uniform batches
  and keeps it to one set. Watch for the `setting
  cudagraph_mode=FULL_AND_PIECEWISE` log line.

## Opt-in lanes from upstream (2026-08-29) — docs only, NOT wired here

Upstream finished these after we vendored. The start script still boots the
MTP k=4 + fp8_ds_mla lane; nothing below is wired or hardware-validated by
us. Reference launchers (their repo):
[`launch/launch-glm53-nvfp4.sh`](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/main/launch/launch-glm53-nvfp4.sh),
[`launch/launch-glm53-nvfp4-dflash2.sh`](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/main/launch/launch-glm53-nvfp4-dflash2.sh).

### DFlash2 speculative decoding — 1.98× on structured output

Drafter: [`incoai/GLM-5.3-DFlash2`](https://huggingface.co/incoai/GLM-5.3-DFlash2)
(**CC BY-NC-ND — never vendor it; users download it to all 4 nodes**).
Measured upstream 2026-08-29 (end-to-end tok/s, TP4, 80K context):

| lane | KV dtype | e2e tok/s | accept | correct |
|---|---|---|---|---|
| **DFlash2 (k=7)** | **fp8** | **53.32** | 95.6% | 100/100 |
| DFlash2 (k=7) | fp8_ds_mla | 51.03 | 95.6% | 100/100 |
| MTP-4 | fp8_ds_mla | 26.91 | 97.0% | 100/100 |

53.32 tok/s on structured output (count-to-100) = **1.98× over MTP-4**.
Free-form prose is a wash (~23% acceptance vs MTP's ~39%). Upstream
suspected the stock Eagle3 aux taps `(6,20,34,48,62,76)` were untuned for
the 743B — our serve-time tap sweep on GLM-5.3-Flash (2026-08-30) refutes
the tap-tuning hypothesis: the taps ship in the drafter's own config
(training-matched), every non-stock set measured worse, and tapping the
final target layer collapses acceptance to single digits. Low prose
acceptance is a drafter-capacity limit — it degrades speed silently, never
correctness.

The non-obvious requirements (each cost boots upstream):

- Method string is `"dflash"`, not `"dflash2"` (v1/v2 dispatch is by the
  drafter's `architectures` field); `num_speculative_tokens` is
  `block_size − 1 = 7`.
- Needs vLLM ≥ v0.28.0 upstream (PR #52816). Pin the **v0.28.0 tag, not
  main** — broken window 2026-08-22 → 08-25 (issue #53428).
- KV dtype trap: use `--kv-cache-dtype fp8` (NOT `fp8_e4m3`, NOT plain
  `fp8_ds_mla`) plus `--kv-cache-dtype-skip-layers sliding_window` — the
  drafter's layers are non-MLA sliding-window and every backend refuses
  `fp8_ds_mla` for them.
- **DCP is impossible with DFlash2**: the drafter's sliding-window layers
  assert `decode_context_parallel_size == 1`. Pick DFlash2 (speed) or DCP
  (pool).
- Exact page fit is structurally impossible on the 743B (MLA page
  576 = 64×9 B/token), so the drafter gets standalone tensors costing ~8%
  of the KV pool — 200K no longer fits; upstream serves **80K** in the
  fp8+DFlash2 lane.
- Two patches are required on the deepseek_v2 path:
  [`patches/patch_swa_under_mla.sh`](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/main/patches/patch_swa_under_mla.sh)
  (per-layer SWA-under-MLA assert) and
  [`dflash2-port/patch_base_kv_dsa.py`](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/main/dflash2-port/patch_base_kv_dsa.py)
  (KV-group patch). **Never set `page_size_padded` on the drafter group** —
  13.59 GB demanded from a 377 MB tensor.
- Build write-up: [`dflash2-port/README.md`](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/main/dflash2-port/README.md).
  Port DFlash2 into the proven image, not the other way round — the Flash
  DFlash2 image's kernels emit digit soup on the 743B.
- **GLP steering and acceptance (ours)**: measured on GLM-5.3-Flash
  (2026-08-30, DFlash2, GLP-44 α=2.0): structured 81.4% → 87.1% steered,
  prose 26.1% → 24.4% — no meaningful cost, because the aux taps capture
  pre-steering features, so the drafter barely couples to the vector.
  Unmeasured on the 743B; the same mechanism should hold, and the untuned
  stock aux taps (~23% prose, above) dominate any steering effect anyway.
  Verification is lossless regardless.

### NVFP4 KV cache — 317,278-token pool at 300K context

`nvfp4_ds_mla` is a 400 B/token record vs fp8_ds_mla's 656. Measured
upstream 2026-08-29 (image `vllm-node-tf5-glm52-b12x:nvfp4-v1`):

| lane | count100 | C6 agg | KV pool | ctx |
|---|---|---|---|---|
| **NVFP4 + MTP-5** | 25.37 | **53.31** | **317,278** | **300K** |
| fp8 + DFlash2 (k=7) | **53.32** | 49.88 | 179,479 | 80K |
| fp8 + MTP-4 | 26.91 | 51.93 | 200,064 | 200K |

+77% KV pool and 3.75× context for ~6% single-stream cost; best C6
aggregate measured. The traps:

- **The enabling switch is the kernel overlay DIRECTORY**, not the image:
  `KERNELS_DIR=/var/tmp/glm-triton-nvfp4`, NOT `~/glm-triton`. Only 2 of the
  10 overlay files differ; `flashmla_sparse.py`'s
  `supported_kv_cache_dtypes` must list `nvfp4_ds_mla` or backend selection
  fails.
- **Pin `cudagraph_capture_sizes` `[6,12,18,24,30,36]`.** With
  `{"cudagraph_mode":"FULL"}` and no sizes, the first concurrent CHAT
  request kills the engine (`KeyError: 'chatcmpl-<id>'`) while raw
  `/v1/completions` looks fine — it is not a chat-template bug.

### Best of both: NVFP4 KV + DFlash2

Image `vllm-glm52-b12x:nvfp4-dflash2-p2` (both bases are the same vLLM
commit, so the DFlash2 port applies unchanged): **51.03 tok/s** structured,
**293,447-token pool at 270K context**, byte-identical output vs the MTP
lane. Leave ~8% pool headroom for the drafter group — 300K fails
(11.06 GiB needed vs 10.2 GiB available), hence 270K.

### Thinking toggle (`enable_thinking`)

GLM-5.3's stock chat template has **no `enable_thinking` variable** (unlike
Flash's updated template) — passing it per request or via
`--default-chat-template-kwargs` silently does nothing; the template ends
with an unconditional open-`<think>` tag. The vendored
`../../patches/vendor/patch_chat_template_thinking.py` (upstream
[`patches/patch_chat_template_thinking.py`](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/main/patches/patch_chat_template_thinking.py))
adds the variable, emitting a pre-closed `<think></think>` when off. Run it
against the weights copy's `chat_template.jinja`, then serve with
`--default-chat-template-kwargs '{"enable_thinking": false}'`
(single-quote the JSON or argparse rejects it). Note `--reasoning-parser
glm45` is a different thing: it routes reasoning into `reasoning_content`
so it stops polluting `content`, but the model still spends tokens
thinking. Use both.

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
