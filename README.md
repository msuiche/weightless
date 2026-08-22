# dspark-deploy

Serving DeepSeek V4 Flash 0731 on 2x DGX Spark (GB10, SM121, TP=2 over RoCE)
with projective refusal steering — the self-contained recipe: serving config,
boot hotfixes, the steering patch, and the control-vector spec.

The live stack is the Anemll image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`,
vLLM 0.25.2) driven by the MiaAI 2x recipe, with our state on top vendored in
`recipe/anemll/`. The retired v027 stack's patch is kept for reference and as
the fallback path.

## What this is: lean abliteration steering as a patch

The model weights are never redistributed — what this repo ships is the
*intervention*, tested and self-contained:

- **The steering file.** A spec-conformant control-vector GGUF
  ([`spec/CONTROL-VECTOR.md`](spec/CONTROL-VECTOR.md)) with per-layer
  directions, derived by us and published under
  [`msuiche/`](https://huggingface.co/msuiche) (see
  [Steering artifacts](#steering-artifacts-ours)). No model weights inside.
- **The patch.** A fail-closed boot hotfix per lane
  (`patches/hotfix-*.py`) that loads the extended GGUF and installs the
  projective hook — no image build, no forked runtime. Structural guard
  tests in `scripts/`, hardware-validated numbers in each lane's README.
- **Quant-friendly.** The hook steers the residual stream at runtime, so it
  works on quantized checkpoints (NVFP4 verified on both lanes,
  bf16→NVFP4 transfer measured) as long as the base architecture is intact.
  The one exception is REAP-pruned builds: deleting experts changes the
  shape of the base model, the vector no longer matches the circuit, and
  the lane is rejected on those grounds (see [Lanes](#lanes)).
- **A no-patch option for Qwen.** The same intervention folded into a
  rank-1 LoRA we reproduced — it loads in stock vLLM/peft with no hotfix at
  all and matches the GGUF arm on hardware (both 24/32 refusal32).

Internals write-up:
[Abliteration without redistributing the model](https://www.msuiche.com/posts/autoresearch-abliteration-without-redistributing-the-model/).

## Lanes

Two serving setups are relevant to us. This repo is the home of the first;
the second is vendored under `recipe/qwen/`.

| lane | hardware | model | steering | status |
|---|---|---|---|---|
| **DSV4 TP=2** | both Sparks over dual-rail RoCE | DeepSeek-V4-Flash-0731, NVFP4 (166.9 GB) | projective cvec, live on 29 layers | **live** — this repo |
| **Qwen TP=1** | one Spark | Qwen3.8-27B-NVFP4 (~13.5 GB) | per-layer cvec, L10–58 at α=1.0 ([shipping artifact](https://huggingface.co/msuiche/Qwen3.8-27B-abliterated-cvec)) | **hardware-validated** — `recipe/qwen/`; stock 4/32, GGUF 24/32, LoRA 24/32 on refusal32 (2026-08-22) |

**Single-Spark DSV4 (EXL3 3.0bpw + REAP-K216): evaluated and rejected.**
The full NVFP4 checkpoint (166.9 GB) cannot fit one Spark, so single-node
DSV4 exists only as the pruned/quantized artifact (99.5 GiB — REAP deletes
40 of 256 routed experts per MoE layer, ~15% of the circuit that produces
the residual stream, on top of the 3.0bpw quant). Benchmarks pass, but the
degradation concentrates in rare behaviour by construction, which is not a
trade we want in our recipes. It would also have meant re-deriving the
control vector on the pruned circuit and porting the hook to sparkinfer.
TP=1 DSV4 is therefore not "this config with TP flipped" — it is a smaller,
approximated model, and we do not serve it. The steering *contract* in
`spec/CONTROL-VECTOR.md` remains lane-independent.

## Layout

| | |
|---|---|
| `recipe/anemll/` | **live**: canonical copies of our compose / start script / `.env.dspark.example` for the MiaAI 2x clone, plus rebuild notes (the real `.env.dspark` is gitignored) |
| `recipe/qwen/` | **Qwen TP=1 lane**: serve script + `.env.qwen.example` for the drowzeys single-Spark recipe; `STEER_MODE=gguf\|lora` — fail-closed hotfix (default) or no-patch LoRA, both hardware-validated 2026-08-22 |
| `patches/hotfix-dsv4-steering-projective.py` | **live**: steering as a fail-closed boot hotfix for the 0.25.2 image (embedded GGUF reader; no image build) |
| `patches/hotfix-qwen38-steering-projective.py` | the same steering for the Qwen lane: patches `qwen3_next.py` + `qwen3_5.py` in the eugr/drowzeys vLLM 0.27 image (steers `hidden_states + residual` — vLLM's decomposed convention) |
| `patches/0001-dspark-projective-steering.patch` | the same hook as a git patch against vLLM v0.27.0 (fallback stack) |
| `patches/0002-dspark-steering-test.patch` | vLLM-side test that extracts and exercises the real GGUF loader (pairs with 0001) |
| `recipe/` (top level) | retired v027 stack: `Dockerfile.gguf-dep`, `Dockerfile.steering-overlay`, `docker-compose.v027.yml` |
| `scripts/` | structural guard tests for the steering patches (DSV4 + Qwen lanes) |
| `install.py` | interactive setup wizard: prereqs, endpoint probe, omp provider install, then the test suite |
| `tests/` | endpoint smoke tests for the deployed stack (endpoint/chat/tool-call/headless omp agent loop); `tests/README.md` |
| `spec/CONTROL-VECTOR.md` | the projective control-vector GGUF format: the `dspark.mode` contract, layer-id mapping, and why an additive reader must refuse the file |
| `BENCHMARK.md` | every serving measurement, with shapes stated (Runs 001–007) |
| `.env.v027.working.example` | env template for the fallback v027 stack |

Real `.env` files are gitignored — only `*.example` templates are tracked
(`.env.dspark.example` under `recipe/anemll/` is the live one). The live
values live on the cluster and in `../DSPARK-HANDOFF.md`. **History was
rewritten twice to purge site-specific material — 2026-08-21 (committed env
files) and 2026-08-22 (internal hostnames/subnets, `.omc` tooling state) —
old clones will not fast-forward; re-clone.**

Maintenance home for the patch is the `dspark-steering-v027` branch of
[`msuiche/vllm`](https://github.com/msuiche/vllm) (public), so upstream bumps get
a real 3-way merge instead of a hand-resolved `.patch` conflict. Regenerate with:

```sh
git -C ../vllm diff v0.27.0..dspark-steering-v027 \
  -- vllm/models/deepseek_v4/nvidia/model.py \
  > patches/0001-dspark-projective-steering.patch
```

## Applying it (fallback v027 stack): no image build required

The patch is one file, so the fast path is to run the **stock** v027 image and
bind-mount the patched `model.py` over the original. Verified on this setup: a
bind-mounted file resolves at its package path and Python reads the mounted
content, so no overlay build is needed to iterate.

```sh
# discover the path in the image rather than assuming it
VLLM_PKG=$(docker run --rm --entrypoint python3 "$BASE_IMAGE" \
  -c 'import importlib.util,pathlib;print(pathlib.Path(importlib.util.find_spec("vllm").origin).parent)')

docker run ... \
  -v /opt/dspark/model.py:$VLLM_PKG/models/deepseek_v4/nvidia/model.py:ro \
  ...
```

Resolve `VLLM_PKG` from the image rather than hardcoding it. In this base it is
**`/opt/vllm-src/vllm`**, a source install, not the
`/usr/local/lib/python3.12/dist-packages/vllm` that the image's `PATH` and
`LD_LIBRARY_PATH` imply. Bind-mounting onto a path that does not exist in the
container silently creates it, which yields an inert patch and a server that
answers normally, so the `layers=29` boot line is the confirmation that steering
is actually live.

Verified in the image: `vllm` resolves to `/opt/vllm-src/vllm`, it self-reports
`0.27.1.dev0+g4bdc8a788` (a post-v0.27.0 dev build), its
`models/deepseek_v4/nvidia/model.py` is 1548 lines matching upstream v0.27.0 so
the patch applies unchanged, and the DSV4 sparse-MLA decode dispatch carries
`(32,256)@pbs64` and `(32,128)@pbs256`.

Use `recipe/Dockerfile.steering-overlay` once the configuration is settled, to
get one reproducible artifact instead of a mount that has to match on both
nodes.

## Base image (fallback v027 stack)

`ghcr.io/bjk110/vllm-spark:v027-ngc2607-dsv4-0731-dspark-k7-256k-production`

vLLM `v0.27.0` (`4bdc8a78`) on NGC PyTorch 26.07, CUDA 13.3.1, torch 2.13.0a0,
`VLLM_USE_DEEP_GEMM_E8M0=1`. Built 2026-08-14. The registry is rate-limited for
anonymous pulls: `docker login ghcr.io` first, or layers retry indefinitely.

This base supersedes six fixes we had queued as hand-ports. All six merged
upstream before v0.27.0 was cut, so they come for free:

| | PR | upstream's measured win |
|---|---|---|
| MTP dead buffer | #50312 | 448 MiB + a per-step copy_ |
| workspace reuse | #50298 | 1.88x on that kernel |
| c128 empty-launch skip | #48957 | ~2x on that kernel |
| adaptive topk width | #50004 | 1.0% E2E |
| grammar across reasoning | #44993 | correctness |
| short-context indexer skip | #49486 | 3.4% E2E TTFT |

Those are upstream's numbers on upstream's hardware, and the 2x figures are
per-kernel: a 2x kernel is not a 2x server.

## Steering

Off unless `DSPARK_STEER_PATH` is set. On the **live** Anemll stack the env
vars are passed through by `recipe/anemll/docker-compose.dspark.yml` and the
GGUF reader is embedded in the hotfix, so nothing here needs building — this
section's build guidance concerns the fallback v027 stack.

**The v027 base image has no `gguf` module**, so a `.gguf` steer path fails and the
server logs `DSpark steering load failed (No module named 'gguf'); serving
unsteered` — honest, but unsteered. Build `recipe/Dockerfile.gguf-dep` (one pip
layer on top of v027, tagged `vllm-dspark-steering:v027-gguf`) and point
`DSPARK_VLLM_IMAGE` at it. Do **not** tag it after the upstream image: the old
launcher uses `DSPARK_VLLM_IMAGE` as its *build output* name, so reusing the
upstream tag overwrites the pulled image with a local build.

Loading a `.pt` instead works and skips the dependency, but it goes through
`torch.load` and bypasses every check in the reader — the `dspark.mode=project`
enforcement, the layer-id cross-check, the `direction.0` rejection. Serve the
GGUF.

```sh
# the general/broad direction, recovered from Keys' published weights by SVD.
# The cyber-derived alternative is ...-cyber-abliterated-cvec-L10-38-a4.gguf;
# both are 29 directions over layers 10-38 at n_embd 4096.
DSPARK_STEER_PATH=/cache/huggingface/DeepSeek-V4-Flash-0731-general-abliterated-cvec-L10-38-a4-keysdir.gguf
DSPARK_STEER_ALPHA=4.0
DSPARK_STEER_LAYERS=$(seq -s, 10 38)
```

Confirm from the boot log that the layer list resolved, and that the count is
what you expect:

```
DSpark GGUF control vector: 29 directions, n_embd=4096, layers [10, 11, ... 38]
DSpark refusal steering active: hook=post_layer alpha=4.000 rank=... layers=29 [10, ...]
```

**Check that `layers=` reads 29 and not 1.** A previous revision dedented the
per-layer assignment out of its loop and kept only the final iteration, steering
one layer while the loader still reported all 29. Coverage is the variable that
dominates this intervention: 6 layers leaves 18.0% refusal, 16 leaves 3.8%, 29
leaves 0.0%.

`alpha` is not a strength dial. At 1 the component is removed; past 1 it is
reflected, which installs the behaviour rather than removing it. To run weaker,
subset the layers and leave alpha alone. The 4.0 above is calibrated for this
checkpoint's residual stream, where it saturates rather than inverting; do not
carry it to another model.

## Steering artifacts (ours)

Both lanes' vectors are published under `msuiche/` on Hugging Face (gated —
fetch with an HF token), spec-conformant per
[`spec/CONTROL-VECTOR.md`](spec/CONTROL-VECTOR.md) and verified against their
pinned checkpoint revisions.

- [`msuiche/DeepSeek-V4-Flash-0731-cyber-abliterated-cvec`](https://huggingface.co/msuiche/DeepSeek-V4-Flash-0731-cyber-abliterated-cvec)
  — **DSV4 lane, GGUF.** Cyber-contrast vector: 29 per-layer directions over
  L10–38, n_embd 4096, α=4.0. The live config currently serves the
  general-contrast variant from the same repo family — a *third-party*
  direction we reformatted, re-measured and repackaged (attribution in the
  file metadata); swapping is one `DSPARK_STEER_PATH` line. Wiring:
  `recipe/anemll/README.md`.
- [`msuiche/Qwen3.8-27B-abliterated-cvec`](https://huggingface.co/msuiche/Qwen3.8-27B-abliterated-cvec)
  — **Qwen lane, GGUF + LoRA.** The GGUF is canonical: per-layer diff of
  means, 49 directions over L10–58, n_embd 5120, α=1.0 (α=4 measurably
  over-refuses on this model). The rank-1 LoRA (`mlp.down_proj`, L1–63,
  α=1.0 baked) is the same intervention folded into weights — it loads in
  stock vLLM/peft/llama.cpp with no hotfix, matches the GGUF on NVFP4
  hardware (both 24/32 refusal32, 2026-08-22), but is still unscored on the
  cyber holdout and is valid only for checkpoint revision `1d4bf0f2ff60`
  (`lora_A` embeds `W`). Wiring: `recipe/qwen/README.md`.

## Roadmap

- **k=7/greedy draft A/B** (upstream issue #84): now one env line
  (`DRAFT_SAMPLE_METHOD=greedy MTP_NUM_TOKENS=7`) after the 2026-08-21
  upstream merge.
- **Qwen LoRA cyber-holdout score** (run L): both Qwen steering arms match
  on refusal32; the LoRA arm still needs its cyber-holdout number before it
  can replace GGUF as the default. Lives in the eval pipeline, not this repo.

## References & credits

- [Abliteration without redistributing the model](https://www.msuiche.com/posts/autoresearch-abliteration-without-redistributing-the-model/) — our write-up of the steering internals (the method this repo packages)
- [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — the model (MIT)
- [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) — the original 2x DSpark NVFP4-KV recipe; its issue #18 is the spec-decode corruption we root-caused before migrating
- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) — the 2-node recipe we run (cloned at `~/dspark-miaai`, our state vendored in `recipe/anemll/`)
- [MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) — single-Spark EXL3/REAP sibling (lane rejected — REAP pruning degrades the tail); source of the b12x backport survey and the KV-pool boot-variance explanation
- [Anemll's GX10 image](https://github.com/anemll) (`ghcr.io/anemll/dspark-vllm-gx10`) — the vLLM 0.25.2 runtime
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) — the sparkinfer/b12x kernel stack inside the image
- [0xSero/deepseek-v4-flash-0731-spark](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) — the single-Spark EXL3/REAP build (evaluated, rejected — see Lanes)
- [Loke-60000/deepseek-v4-flash-0731-spark-vision](https://huggingface.co/Loke-60000/deepseek-v4-flash-0731-spark-vision) — community Spark vision serving fix (surveyed, not adopted)
- [drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark) — the Qwen TP=1 lane's serving recipe

## Status

**2026-08-21: the serving stack is now the MiaAI Anemll recipe** (Anemll
`0.1.1`, vLLM 0.25.2), not the v027+steering image below. Stage-C was retired
after its DSpark draft was root-caused as the long-context corruption source.
**Steering is ported**: `patches/hotfix-dsv4-steering-projective.py` applies
the same projective hook to the 0.25.2 `model.py` as a MiaAI-style fail-closed
boot hotfix (the image ships no `gguf` package, so the spec-conformant GGUF
reader is embedded). Live on both TP ranks with the identical vector, alpha
and layer set as the v027 config; validation in [`BENCHMARK.md`](BENCHMARK.md)
Run 007. The v027 stack remains parked as fallback. Cluster ops runbook:
`../DSPARK-HANDOFF.md`.

<details><summary>Superseded status (v027 stack, 2026-08-19)</summary>

**Serving.** Continuously up on 2x DGX Spark since 2026-08-17, TP=2, steering
active on 29 layers. Measurements and their full configuration are in
[`BENCHMARK.md`](BENCHMARK.md); the working values are in `.env.v027.working.example`.

Settled configuration:

| | |
|---|---|
| image | `vllm-dspark-steering:v027-gguf` (v027 + `gguf==0.19.0`) |
| context | **1,032,192** — see BENCHMARK.md Run 005; 1,048,576 collapses prefill to 110 tok/s, 1,032,192 does not |
| KV dtype | `fp8_ds_mla` (`nvfp4_ds_mla` does not exist upstream in v0.27) |
| speculative decoding | **on**, `method=dspark`, `num_speculative_tokens=5` |
| `max-num-seqs` | 6 — 32 dies during warmup, only ~14 GiB is left for KV |

Headline numbers, all with their shape stated in `BENCHMARK.md`: prefill flat at
1,774–2,341 tok/s from 2k to 262k input; decode flat at 29–37 ms/tok over the same
range; 77.3 tok/s on the peak-finder prompt at 99.7 % draft acceptance, which is
parity with the retired stack's 78.4 tok/s and needs no Patch 4.

Known gaps: no needle is perfect at ~0.5M (5 of 6 exact to 514,035 tokens, one
returned 9 of 10 characters), 1M context remains impractical, and there is no
throughput baseline on the retired image so the upgrade is not attributable.

</details>
