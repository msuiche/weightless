# Muse-Glimmer-30B NVFP4 — single DGX Spark

Stock `nvidia/Muse-Glimmer-30B-NVFP4` serving recipe: one Spark, TP=1,
no patches, no overlays. This is the toolkit's simplest lane — vLLM 0.28.0
serves the `muse_glimmer` architecture, its ATEM tool-call parser, its
reasoning parser, and the DFlash block-diffusion drafter all natively
(registry: `MuseGlimmerForConditionalGeneration` +
`MuseGlimmerAssistantModel` → `qwen3_dflash`), so there is no hotfix and no
steering (no GLP vector exists for this arch yet).

The model: Meta Superintelligence Lab's Muse-Glimmer-30B (Apache 2.0),
quantized by NVIDIA Model Optimizer to a mixed W4A16-NVFP4 / FP8 / BF16
checkpoint (~24.7 GB on disk). Dense 52-layer causal transformer (hidden
6656, GQA 32/2, 3:1 sliding:full attention, sliding window 2048) plus a
~1.8B ViT-G/14 perception encoder for interleaved image input. Context
131,072. Knowledge cutoff 2026-01-04.

## Serve command (what the launcher runs)

```sh
vllm serve nvidia/Muse-Glimmer-30B-NVFP4 \
  --revision f45fad5689e9a4d937f7e872fbec20c4e8a74154 \
  --served-model-name muse-glimmer-nvfp4 \
  --host 0.0.0.0 --port 8084 \
  --tensor-parallel-size 1 --max-model-len 131072 \
  --max-num-seqs 8 --gpu-memory-utilization 0.92 \
  --kv-cache-dtype auto --enable-prefix-caching --enable-chunked-prefill \
  --mamba-cache-mode align \
  --reasoning-parser muse_glimmer --tool-call-parser muse_glimmer \
  --enable-auto-tool-choice \
  --speculative-config '{"method":"dflash","model":"meta-models/Muse-Glimmer-30B-assistant","revision":"e8192f3a8f617f74be2ce220360c89ef4789f39f","num_speculative_tokens":16}'
```

This is NVIDIA's card-tested vLLM 0.28.0 invocation
([model card](https://huggingface.co/nvidia/Muse-Glimmer-30B-NVFP4), checked
2026-09-09) plus the DFlash drafter. `--mamba-cache-mode align` comes from
the card and is a no-op on this dense attention arch. The card also passes
`--trust-remote-code`; the checkpoint ships no remote code and the arch is
native in 0.28.0, so the launcher omits it.

## Sampling and the reasoning knob

Matched-quality sampling per the model card: `temperature=1.0`,
`top_p=0.95`, `top_k=64`, thinking enabled. Reasoning strength is a
system-prompt knob — `Reasoning strength: low|medium|high|xhigh` — and the
bundled chat template already defaults it to `high`, so a request with no
system prompt runs at high. Set `SPECULATIVE_MODE=none` in the env to serve
the target without the drafter.

## Spec decode (DFlash)

The drafter (`meta-models/Muse-Glimmer-30B-assistant`, 5 sliding-window
layers, block size 16, predicts from target hidden layers {1, 13, 25, 37,
49}) is a first-class vLLM 0.28.0 citizen: the speculative config
auto-detects `method: dflash` from its `MuseGlimmerAssistantModel`
architecture. The wizard prefetches it next to the target weights; the
launcher pins its revision. Meta's card reports ~3.1x decode on an RTX 5090
with the quantized drafter — measure on the Spark before believing a number
here.

## Configure and launch

Use a free Spark — the lane is TP=1 and touches one node only.

```sh
cd recipe/museglimmer
cp .env.museglimmer.example .env.museglimmer
# Edit HF_CACHE (and the port if 8084 is taken).
bash serve-museglimmer.sh --dry-run
bash serve-museglimmer.sh
docker logs -f museglimmer
```

Docker needs NVIDIA GPU support. The wizard's asset step prefetches target +
drafter with the offline-first pattern (`HF_HUB_OFFLINE=1 hf download` fast
path, online fallback) into `HF_CACHE`; the launcher sets
`HF_HUB_CACHE=/root/.cache/huggingface` in the container so the boot reuses
that prefetch instead of re-downloading under `hub/`. Both revisions are
pinned to reviewed commit SHAs. Existing named containers are never replaced
by the launcher.

## Verify before adding clients

```sh
curl --fail http://localhost:8084/health
cd ../..
export WEIGHTLESS_BASE_URL=http://localhost:8084/v1
export WEIGHTLESS_MODEL=muse-glimmer-nvfp4
bash tests/smoke/01-endpoint.sh
bash tests/smoke/02-chat.sh
bash tests/smoke/03-tool-call.sh
```

These check model discovery, actual text generation, and a structured tool
call (the muse_glimmer ATEM parser). After configuring omp for this
endpoint, run `tests/smoke/04-omp-headless.sh` with `WEIGHTLESS_OMP_MODEL`
set to its provider/model ID. The endpoint is multimodal (image input via
the perception encoder); the omp template entry is text-only until an agent
client image path is validated.

Structural lane checks without Docker or a GPU:

```sh
python3 tests/structure/test-museglimmer-structure.py
```

## Modal compatibility result — 2026-09-09

On one NVIDIA H100 80GB with stock `vllm/vllm-openai:v0.28.0` (full record:
`refusal-research/experiments/20260909-muse-glimmer-bringup/RESULT.md`):

- Stock boot ready in 270 s (465 s with the drafter), `max_model_len`
  131072 confirmed over `/v1/models`.
- Exact-answer generations clean (`stop`), an ATEM `get_weather` tool call +
  tool-result continuation round-tripped, and a needle was retrieved from a
  104,088-token prompt.
- **DFlash spec decode works**: 288.8 vs 92.5 tok/s (3.1x, matching the
  card's RTX-5090 claim); 1954/7872 drafted tokens accepted over 492
  16-token blocks.
- The card lists only Blackwell as supported hardware; the mixed
  NVFP4/FP8/BF16 checkpoint nonetheless boots and serves correctly on
  Hopper. Note for the eval suites: the model emitted no `to=self`
  reasoning spans at any reasoning-strength setting in this configuration —
  answers arrive without a visible trace.

**DGX boot validation is pending** — like the Nemotron lane before it, this
recipe is validated on Modal H100 first; GB10 (sm121) behaviour is
unverified until a free Spark boots it. The image is multi-arch and already
proven on the rig by the Inkling/Nemotron lanes.
