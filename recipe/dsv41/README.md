# DSV4.1-Flash lane — NOT DEPLOYABLE

DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, sha
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`, MIT). New architecture
(`deepseek_v41`): 40 layers, hidden 5120, 384 routed experts + 1 shared,
top-6/token, MLA (q_lora 1280, o_lora 1024, o_groups 8) with KV-compression
layers, hc_mult=4 hyper-connections (sinkhorn), engram layers [1, 14],
built-in DSpark drafter (num_nextn 3), vision tower, 1M ctx. 510.3 GB
checkpoint: FP8 [32,32] ue8m0 with MXFP4 experts, 48 shards.

## Why this lane refuses to deploy

1. **vLLM support is a PR branch, not a release.** Serving requires
   `vllm-project/vllm` PR #56201 (branch `dsv41-feat`). The GLP vector was
   derived and validated against a source build of that branch
   (sha recorded in the experiment RESULT.md). Until it merges, every
   "release" path (day-0 image, pip vllm) cannot load the model.
2. **Nothing fits 2x128 GB.** The stock checkpoint is 510 GB. The rig is
   2x DGX Spark (256 GB total). No NVFP4/other community quant of
   V4.1-Flash exists yet, and the GB10 kernel path for this arch
   (mhc tilelang kernels, sparse-MLA indexer, engram) has never been
   brought up on sm121.

Unblock conditions: **(1) #56201 merged**, **(2) a 2x128GB-fitting quant
published and validated on GB10**. Until both hold: do NOT point this lane
at the rig. The daily driver stays DSV4-0731 (recipe/anemll).

## What exists today

- **GLP-39 vector** (`glp.mode=project`, `glp.hook_point=
  residual_stream_post_layer`, L1-39, alpha per the published card):
  `msuiche/DeepSeek-V4.1-Flash-abliterated-cyber-GLP-39-L1-39-a0.5` (published 2026-09-10, gated=auto). The hook is the
  post-layer hyper-connection fold reduced to the single stream (mean over
  the 4 hc copies, the model's own aux-hidden-state reduction), pre-engram;
  derived AND applied at that site. This is NOT the DSV4-0731
  `ffn_out_pre_residual` hook — the DSV4 hotfix must not load this file,
  and a conforming reader refuses the mismatch.
- **Validated serving shape (research, not a deployment):** Modal, single
  node 4x H200 TP=4, source-built `dsv41-feat` wheel, offline LLM driver.
  Capture + eval lane: `refusal-research/experiments/20260910-dsv41-flash-glp/`
  (staging/patch_dsv41.py ports the steering hotfix to the PR's
  `deepseek_v4_1/nvidia/model.py`; the in-layer fold site steers
  direction[i-1], the post-loop fold steers direction[39]).
- **`start-deepseek-v41-flash-dspark.sh`**: fail-closed placeholder. It
  prints this blocker and exits 1. It does not compose, ssh, or touch any
  host — replace it with a real recipe only after both unblock conditions
  hold and the recipe has been validated on the rig.

## Env file

`.env.dsv41.example` mirrors the DSV4 lane's env surface so the wizard's
render path stays shaped correctly, but every deploy knob is annotated
blocked. The wizard entry (`setup.py` LANES, `[BLOCKED] DSV4.1-Flash`)
bounces with the blocker before any deploy step.
