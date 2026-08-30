# GLM-5.3-Flash TP=4 lane — NVFP4 on 4x DGX Spark + GLP-44 steering

GLM-5.3-Flash (320B total / 18B active, 45 layers, KDA + NoPE sparse MLA,
mHC hyperconnections) across **four** Sparks (head + 3 workers over RoCE),
with the GLP-44 projective refusal vector applied by a fail-closed boot
hotfix.

**This lane needs 4 nodes.** If your `machines.txt` lists two, rack and cable
two more first. A 2-node TP2 config exists (262K context, ~97 GiB weights per
rank, KV-starved — needs local weights on both ranks plus an aggressive
cache-flush ritual; see tonyd2wild's TP2 repo below) but stays out of scope
for this wizard lane: `nodes` stays 4.

The serving stack is **hardware-validated** — but not by the stock vendor
image. Everything below the flags comes from tonyd2wild's day-0 GB10
deployments:
[GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
(TP4 flagship, this lane's config) and
[GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark)
(TP2 deep-dive; its `docs/DEPLOY-REPORT.md` has every failure, root cause,
and receipt). We reference, we do not fork.

## Model and image — both are non-obvious

- **Model: the NVFP4 quant
  [`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
  (compressed-tensors W4A4), not the 306 GB FP8 original.** This is
  upstream's current default: corruption-free, ~2x faster load, ungated, and
  drop-in (same arch/flags). At TP4 that is ~50 GiB of weights per rank,
  and the GB10 KV-allocation ceiling dissolves: the shipped config pins
  **24 GiB KV/rank = 3,774,873 fp8 tokens = 3.6 concurrent full-1M-context
  requests** at `--max-model-len 1048576` (model-native 1M, no rope
  scaling). On TP2 the same model is ~97 GiB/rank and the driver grants only
  ~4.5–5.5 GiB of KV afterward — that is the 262K TP2 ceiling.
  - Caveats: W4A4 scores a few points lower on hard reasoning than the
    ModelOpt quant, and vision needs `chat_template_mm.jinja` in the weights
    dir (the start script resolves it from the HF snapshot).
  - Legacy alternative:
    [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
    (ModelOpt) — **intermittently emits corrupted token IDs** (vLLM issue
    [#54150](https://github.com/vllm-project/vllm/issues/54150); upstream's
    probe saw 4/9/8 U+FFFD over 3 runs vs 0/0/0 for RedHatAI, and corruption
    inside a tool-call block desyncs the parsers). Prefer RedHatAI.
- **Image: the sm121-v8 patch stack, never stock.** The day-0
  `vllm/vllm-openai:glm53-flash-arm64-cu130` (vLLM 0.1.dev20051) works on
  B200 but dies **five separate ways** on GB10: the NoPE-MLA SM12x sparse
  backend gap (the stock capability-12 path hardcodes DeepSeek's
  `pe_dim=64`; GLM is NoPE), a FlashInfer 0.6.17 FA2 MLA NaN on SM121 (fix:
  0.6.18 nightly), two dependency downgrades that nightly sneaks in
  (NCCL → re-pin 2.30.7; cutlass-dsl → re-pin 4.6.2), a PDL race surface
  (gated off on SM12x), and an uninitialized indexer top-k buffer
  (`torch.empty` → init −1 + clamp). v8 adds the fp8-KV shared-memory tile
  fix (a Hopper 228 KB smem assumption vs GB10's ~101 KB).
  - Prebuilt: **`radixark/vllm-glm53-flash:sm121-v8`** (what their TP4
    launcher ships) — the `.env` default.
  - Local build: their `docker/Dockerfile.glm53-sm121*` **v1→v8 in order**,
    each `FROM` the previous, on the day-0 arm64-cu130 base.
  - **v9/InstantTensor is UNSTABLE multi-node** (a rank dies silently ~1 min
    post-load in every v9 boot) — do not use it; v8 is the stable ceiling.
  - The start script refuses any image tag without `sm121` in it, and checks
    the image ID is identical on all four nodes.

## Serve flags (theirs, hardware-validated)

`--block-size 2304` (DeepGEMM's arch-12 fp8 paged-MQA accepts only 64-entry
pool pages; 2304 is a multiple of kpool·64 and of the MLA 128 alignment),
`--gpu-memory-utilization 0.85` with **`--kv-cache-memory` pinned**
(25769803776 = 24 GiB/rank — the MTP draft head OOMs riding the gmu edge;
the number is stress-gated behind 3× concurrent 20K prefills — 38 GiB boots
and answers short prompts, then the first long prefill NVRM-OOMs the
engine), `--kv-cache-dtype fp8_e4m3`, MTP `num_speculative_tokens=4` (their
acceptance data suggests 3 as a micro-tune — position 4 nearly free-rides),
`--moe-backend marlin`, `--enforce-eager`, `--max-num-seqs 6`,
`--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`,
thinking off by default via `--default-chat-template-kwargs`, and
`--chat-template …/chat_template_mm.jinja` (the checkpoint ships a text-only
template; vision requests 500 without the mm variant — the start script
resolves it from the HF snapshot). Full 1M prefills take minutes of wall
clock; cap `--max-model-len` lower (e.g. 300000) for a snappier multi-user
endpoint.

Measured on their 4-node stack: TTFT 0.204 s median; decode **35.7 tok/s**
generic greedy median — and that is a floor: MTP acceptance is
content-regime dependent, so structured/agentic output (what agents actually
generate) warms to **53–64 tok/s**, freeform prose sits ~37.

## DFlash2 speculative decoding (opt-in)

tonyd2wild's newer TP2 stack swaps MTP for a DFlash2 drafter — a serving
option, not a separate lane (the GLP-44 steering is orthogonal to the
spec-decode method):

- **Image:** `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (the
  start script's `sm121` image gate accepts the tag).
- **Drafter:**
  [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  (2.2 GB, license **CC BY-NC-ND** — non-commercial, no-derivatives). We do
  **not** vendor it: download it yourself to **both** nodes, e.g.
  `/var/tmp/models/GLM-5.3-Flash-DFlash2`.
- **Measured upstream: 46.9 tok/s vs 21.8 tok/s MTP-4 at 262K context TP2**,
  zero KV cost. Reference launcher: his repo's
  `launch-glm53-vllm-tp2-dflash2.sh`.
- **Speculative config:**
  `--speculative-config '{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":7}'`
  with `-v /var/tmp/models/GLM-5.3-Flash-DFlash2:/models/dflash2-draft:ro`.
- **Steering and draft acceptance — measured, not assumed** (2026-08-30,
  4×H100, RedHatAI NVFP4, day-0 image + this drafter, GLP-44 α=2.0):
  structured **81.4% → 87.1% steered** (+5.7 pts), prose 26.1% → 24.4%
  (−1.6 pts, −4.3% tok/s) — **no meaningful acceptance cost**. The
  aux-capture taps land before the steering block in the decoder loop, so
  the drafter conditions on pre-steering features in both arms; only greedy
  token-choice divergence couples through. Verification is lossless
  regardless.
- **Aux taps are training-matched — do not retune at serve time.** Our
  sweep (2026-08-30, same stack): the stock `target_layer_ids`
  `[5,14,24,33,42]` ship in the drafter's own `config.json`, and every
  non-stock set measured worse — uniformly, down to single digits. Two hard
  rules found: **never tap the final target layer** (44) — acceptance
  collapses to ~6%; and the tap count is fixed by the checkpoint's `fc`
  width (5 taps). Prose acceptance (~26%) is a drafter-capacity limit, not
  a tap-tuning opportunity.

This lane keeps MTP-4 as shipped; DFlash2 is a TP2 opt-in you wire by hand
from his launcher (still bind-mount the kpool fix — both images need it).

## Boot ritual (each rule cost them a boot)

The start script bakes in what it can; the rest is on the operator:

1. `sync; echo 3 > /proc/sys/vm/drop_caches` on **every node** before boot
   (the script attempts it non-interactively and warns if it can't).
2. Their **`cache_flusher.sh` sidecar on every node during boot** — GB10's
   driver fails allocations against page-cache-full memory; get it from the
   TP4 repo (`cache_flusher.sh`, mechanism in `docs/GB10-KV-MEMORY-LADDER.md`).
   The script warns when no flusher is running on a worker.
3. Workers first, ~20 s apart, **head last** (the script does this).
4. **Tear down ALL ranks before relaunching any** — a fresh rank that
   rendezvouses with a dying one hangs. The script refuses to start if the
   container exists on any node.
5. Identical image ID on all four nodes (the script checks).
6. A node that has been through many boot cycles accumulates allocator
   degradation — reboot it when a proven config starts dying.
7. Capture `docker logs` before `docker rm -f`.

## The patch layers (order and failure semantics)

1. **Image-build time (theirs):** the sm121 v1→v8 Dockerfile/string-patch
   stack — fixes GB10 kernel/backend bugs. Baked into the image.
2. **Container start (ours):**
   `../../patches/hotfix-glm53-steering-projective.py` runs in the entrypoint
   before `vllm serve` on every rank, and the `&&` makes it fail-closed: a
   boot asked for steering that cannot apply it never serves unsteered.
3. **Container start (theirs, vendored):**
   `../../patches/vendor/sparse_attn_indexer_kpool_sm121.py` is bind-mounted
   over the in-image
   `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` on every node —
   the SM121 indexer top-k fix. Both published images (sm121-v8 and the
   dflash2 one) hard-kill on decode past ~24K context without it (forensics:
   `docs/SM121-CRASH-FORENSICS-2026-08-27.md` in tonyd2wild's DFlash2 repo).
   The start script syncs it to all four nodes and fails closed if the
   vendored file is missing.

Anchor caveat: our hotfix's anchors target the PR-head `glm5next` source
(vllm-project/vllm#53906 @ `142062f1`, vendored in
`../../patches/reference/glm5next.py`); the image is a `0.1.dev20051`
snapshot that predates it, and the sm121 stack patches other vLLM files at
build time (`sparse_attn_indexer_kpool.py`, `glm5next/nvidia/ops/kpool_compress.py`
— not known to touch `model.py`). **The first real boot is the anchor test**;
if the anchors fail, the boot refuses (fail-closed) — then diff the image's
file against the vendored reference, update the anchors, re-vendor:

```sh
docker run --rm --entrypoint cat "$GLM53_IMAGE" \
  /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py \
  | diff - ../../patches/reference/glm5next.py
```

Confirm steering from the boot log: `weightless GLP steering active:
hook=post_layer alpha=2.000 ... layers=44` (layers=1 means the
per-layer-loop regression; layers=0 means unsteered).

## Steering vector

The shipping artifact is
**`GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf`** — per-layer
difference-of-means over the mHC stream (16384 = 4×4096), layers 1–44,
α=2.0, spec-conformant per [`../../spec/GLP.md`](../../spec/GLP.md).
Published at
[`msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44`](https://huggingface.co/msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44)
(gated — fetch with an HF token), at the **root of the HF cache on ALL FOUR
nodes**:

```sh
huggingface-cli download msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44 \
  --include "*.gguf" --local-dir ~/.cache/huggingface   # on ALL FOUR nodes
```

**α=2.0 is calibrated — and α≥2.5 GARBLES this model.** The cliff is abrupt
(measured). Do not raise it; do not import another lane's alpha.

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
WEIGHTLESS_STEER_PATH=$HF_CACHE/GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf \
  python3 ../../patches/hotfix-glm53-steering-projective.py --check
```

## EXL/3 / B12X variant (x86 SM120) — brandonmusic's build

[`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
is the same GLM-5.3-Flash base as a uniform-K4 EXL3 quant, served by a
**custom vLLM/B12X fork** (`verdictai/glm53-flash-exl3-k4:*-v84`, TP2/EP2/DCP2
on 2x RTX PRO 6000, NVFP4 MLA KV, DFlash2 or built-in MTP3) — not TabbyAPI,
not stock vLLM. Because it is vLLM under the hood, the GLP-44 hotfix applies
with the same mechanics as this lane — but with a **variant patch**:
`../../patches/hotfix-glm53-exl3-steering-projective.py`. The fork's
`vllm/models/glm5next/nvidia/model.py` (at `/opt/infernal-invocation/vllm`
on PYTHONPATH) added a DFlash aux-hidden-state branch, so the stock single
decoder loop is two loops there; the variant patches both (in the aux loop,
steering lands after the aux capture, so DFlash features are pre-steering
and the stream continues steered). Anchors are verified against the model
file extracted from the published image (vendored:
`../../patches/reference/glm5next_b12x_exl3.py`, from OCI layer
`sha256:7f03081e…`). His
launcher env (`--moe-backend b12x`, `--attention-backend B12X_MLA_SPARSE`,
DCP flags) stays as shipped; steering is orthogonal to all of it.

**Status: runtime-validated 2026-08-30 on real SM120** (Vast.ai 2x RTX PRO
6000 Max-Q, verdictai v84 language-only image, eager + gmu 0.95). Anchors
applied exactly once against the image's model file (md5-identical to the
vendored reference); steered boot logged on both TP workers. Measured with
GLP-44 α=2.0, greedy, 1400 tokens (n=32/cell): refusal32 1/32 → 15/32,
cyber32 14/32 → 31/32, benign32 32/32 → 31/32, zero garbled — direction
transfer to uniform-K4 EXL3 confirmed, somewhat attenuated vs the NVFP4
reference on the contrast suite. Two serving notes from that run: the
image's NGC torch 2.13.0 has no CPU LAPACK, so the hotfix's basis QR runs
on the target device (fixed in the patch); and brandonmusic's published
`gmu 0.986 + CUDA graphs` has no stable memory point at 8 concurrent seqs
(MTP capture OOM at 0.986, empty KV pool at 0.95) — the run used
`--enforce-eager` + gmu 0.95, KV pool ~192K tokens, ~2.5 min boot.
The image remains x86 SM120-only (sm_120a cubins, no PTX) — it will not
boot on H100/B200 or the aarch64 GB10 Sparks.

## Credits

- Model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash);
  quant: [RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
  (compressed-tensors W4A4, the corruption-free default; the earlier
  [LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
  ModelOpt quant's sm_121 notes fed directly into the deployment).
- The GB10 deployment and the sm121 patch stack: **[@tonyd2wild](https://github.com/tonyd2wild),
  deployed and debugged by Knox (Claude)** — the day-0 failure receipts are
  in their `docs/DEPLOY-REPORT.md`.
- **barrydeen** — the gmu 0.85 reference config and quantization-coverage
  table from their independent DGX Spark recipe.
- vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) authors
  for the day-0 image; FlashInfer for the 0.6.18 SM90-NoPE MLA path.
