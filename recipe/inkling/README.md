# Inkling-Small TP=2 lane — NVFP4 on 2x DGX Spark + GLP-41 steering

Inkling-Small (`thinkingmachines/Inkling-Small-NVFP4`, 159 GiB) on stock
vLLM v0.28.0 (inkling is day-0 since that release), tensor-parallel across
**both** Sparks (head + worker over RoCE), with the GLP-41 projective refusal
vector applied via a bind-mounted pre-patched `model.py` (the same integration
pattern as the qwen38fn lane).

**STRUCTURE-VALIDATED ONLY (2026-09-02)** — written from the measured qwen38fn
lane and the Modal validation (4×H100, vLLM 0.28.0: refusal32 0/32 → 30/32 at
α=0.25, benign 30/32) but not yet booted on the Sparks. First boot: watch it.

| file | what it is |
|---|---|
| `start-inkling-dspark.sh` | head+worker boot with the wedge-proofing from the qwen38fn saga (preflight free-memory gate + zombie check, drop_caches on both nodes, `--restart no`, capped logs) |
| `.env.inkling.example` | full config with site values as `<...>` placeholders |
| `../../patches/hotfix-inkling-steering-projective.py` | the steering hook for `vllm/models/inkling/nvidia/model.py` — handles Inkling's deferred residual add (`pending` flush via the file's own `_sconv_add_norm` idiom) |

## Traps

- **α=0.25 is calibrated, not a default.** α=1.0 and α=0.5 garble EVERYTHING
  on this model (including benign) — the most dose-sensitive model in the
  program. Do not raise it.
- **The steered `model.py` must be pre-patched and staged on both nodes**
  (`files/inkling-model-steered.py`): extract the image's file once, apply the
  hotfix offline (`WEIGHTLESS_STEERING_MODEL_PY=<copy> python3
  ../../patches/hotfix-inkling-steering-projective.py`), copy to both nodes.
- **drop_caches needs passwordless sudo** on both nodes
  (`/etc/sudoers.d/drop-caches` — see the qwen38fn README trap; the script
  uses `sudo -n` and will fail loudly without it).
- **NVFP4 is 159 GiB → ~85 GiB/rank at TP=2** — fits 2 Sparks with KV at
  util 0.835 (the measured GB10 envelope from the qwen38fn lane).

## STATUS 2026-09-03: DGX lane BLOCKED — engine-level, not config

Fifteen controlled boot attempts on 2× DGX Spark (spark-4687 + spark-5bc3,
vllm/vllm-openai:v0.28.0, TP=2 over RoCE) all die ~28s after weight-load
reaches the final layer, on BOTH nodes, with no Python error — the engine
tears down and `docker logs` shows only the wrapper. NVRM `NV_ERR_NO_MEMORY`
appears in dmesg at the same moment but is a red herring (see probe below).

Eliminated (one variable per boot): GPU_MEMORY_UTILIZATION 0.835/0.89/0.90
(the 0.90 startup-gate fail was arithmetic: 0.90×121.69=109.52 > 109.32
CUDA-free), MAX_MODEL_LEN 262K/128K/32K, MAX_NUM_SEQS 8/4,
MAX_NUM_BATCHED_TOKENS 8192/2048, page-cache/drop_caches, stale containers,
full reboot of both nodes (117 GiB free each, clean), NCCL HCA typo
(`==rocep1s0f0` → `=rocep1s0f0`), NCCL_IB_DISABLE=1 (socket transport),
--enforce-eager (not graph capture), no MTP spec-config, no
VLLM_USE_V2_MODEL_RUNNER, and — critically — **stock unpatched model.py**
(not our hotfix). A driver probe (`torch.empty` 5 GiB chunks) allocates
115 GiB on the worker with no error, so it is not raw capacity.

Not eliminated / current suspicion: inkling day-0 support in v0.28.0 has
never run cross-node TP on aarch64/GB10 (the Modal lane that produced all
our validation numbers is single-node TP=4 on x86 H100). Something in the
post-load phase (profiling forward or first cross-node collective of the
inkling custom ops — short_conv state, rel_attention) kills the worker proc
before it can report ready.

**What works:** the Modal lane — 4×H100 single node, stock vLLM 0.28.0,
`WEIGHTLESS_STEER_PATH=<GLP-41 gguf> WEIGHTLESS_STEER_ALPHA=0.25`,
`--tokenizer-mode inkling --trust-remote-code`. refusal32 0/32 → 30/32,
benign 30/32 (2026-09-02).

**ROOT CAUSE CONFIRMED 2026-09-03** (live probes, full trail in
[`GB10-ANALYSIS.md`](GB10-ANALYSIS.md) §8, in this repo):
GB10 (sm_121) falls through `_use_sheared_bias()` (`major in (10,11)`), so
inkling's rel-attention is sent to the Hopper cute path, which asserts
`Paged KV not supported on SM 12.0 in this PR`. Forcing the intended
tml-fa4 sheared path (gate patched `major >= 10`) gets past weight-load and
KV sizing, then dies at `assert tile_n == 128` in tml_fa4's rel_bias
metadata — inkling is diff-headdim (128/64, rel_extent 1024) and the
SplitKV heuristic shrinks tile_n to 64, which rel_bias rejects. **Neither
FA4 backend in v0.28.0 can run Inkling on GB10; the fix is upstream**
(cute SM12 paged-KV support — marked "in this PR" — or tml_fa4 rel_bias
learning tile_n=64). The gate patch + probes live in
`~/dspark-inkling/files/` on the head for the day the tml_fa4 side lands.

**Retest path for the DGX lane:** next vLLM release (≥0.28.1), then:
`bash start-inkling-dspark.sh` (unchanged config; the boot chain, env pins
and staging in this directory are all correct and were re-verified during
the eliminations). If it still dies at ~28s post-load, the bug is upstream —
bisect vLLM's inkling support on 2×GB10, don't touch this recipe.
