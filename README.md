# dspark-deploy

Serving DeepSeek V4 Flash 0731 on 2x DGX Spark (GB10, SM121, TP=2 over RoCE)
with projective refusal steering — the self-contained recipe: serving config,
boot hotfixes, the steering patch, and the control-vector spec.

The live stack is the Anemll image (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`,
vLLM 0.25.2) driven by the MiaAI 2x recipe, with our state on top vendored in
`recipe/anemll/`. The retired v027 stack's patch is kept for reference and as
the fallback path.

## Layout

| | |
|---|---|
| `recipe/anemll/` | **live**: canonical copies of our compose / start script / `.env.dspark` for the MiaAI 2x clone, plus rebuild notes |
| `patches/hotfix-dsv4-steering-projective.py` | **live**: steering as a fail-closed boot hotfix for the 0.25.2 image (embedded GGUF reader; no image build) |
| `patches/0001-dspark-projective-steering.patch` | the same hook as a git patch against vLLM v0.27.0 (fallback stack) |
| `recipe/` (top level) | retired v027 stack: `Dockerfile.gguf-dep`, `Dockerfile.steering-overlay`, `docker-compose.v027.yml` |
| `scripts/` | build-guard tests |
| `spec/CONTROL-VECTOR.md` | the projective control-vector GGUF format: the `dspark.mode` contract, layer-id mapping, and why an additive reader must refuse the file |
| `BENCHMARK.md` | every serving measurement, with shapes stated (Runs 001–007) |

Maintenance home for the patch is the `dspark-steering-v027` branch of
[`msuiche/vllm`](https://github.com/msuiche/vllm) (public), so upstream bumps get
a real 3-way merge instead of a hand-resolved `.patch` conflict. Regenerate with:

```sh
git -C ../vllm diff v0.27.0..dspark-steering-v027 \
  -- vllm/models/deepseek_v4/nvidia/model.py \
  > patches/0001-dspark-projective-steering.patch
```

## Applying it: no image build required

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

## Base image

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

Off unless `DSPARK_STEER_PATH` is set.

**The base image has no `gguf` module**, so a `.gguf` steer path fails and the
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

## Steering vector

The cyber control vector is published at
[`msuiche/DeepSeek-V4-Flash-0731-cyber-abliterated-cvec`](https://huggingface.co/msuiche/DeepSeek-V4-Flash-0731-cyber-abliterated-cvec)
(gated; 478 KB GGUF, spec-conformant per [`spec/CONTROL-VECTOR.md`](spec/CONTROL-VECTOR.md)).
Fetch it into the HF cache on both nodes and point `DSPARK_STEER_PATH` at it —
see `recipe/anemll/README.md` for the exact wiring. The live config currently
serves a general-contrast variant from the same derivation; both are verified
against the pinned checkpoint revision.

## Roadmap

- **Serve Flash + Flash-Vision.** `DeepSeek-V4-Flash-Vision-Exp` launched on
  the DeepSeek API on 2026-08-21 as an experimental, API-only release — no
  open weights yet. It is the same base family we already serve, so when
  weights drop the plan is a second serving profile in this repo (vision
  weights + tower on the same 2-node stack), not a new stack. The community
  adapter grafts circulating on HF were evaluated and deliberately skipped.
- **k=7/greedy draft A/B** (upstream issue #84): now one env line
  (`DRAFT_SAMPLE_METHOD=greedy MTP_NUM_TOKENS=7`) after the 2026-08-21
  upstream merge.

## References & credits

- [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — the model (MIT)
- [tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) — the original 2x DSpark NVFP4-KV recipe; its issue #18 is the spec-decode corruption we root-caused before migrating
- [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) — the 2-node recipe we run (cloned at `~/dspark-miaai`, our state vendored in `recipe/anemll/`)
- [MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) — single-Spark EXL3 sibling; source of the b12x backport survey and the KV-pool boot-variance explanation
- [Anemll's GX10 image](https://github.com/anemll) (`ghcr.io/anemll/dspark-vllm-gx10`) — the vLLM 0.25.2 runtime
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) — the sparkinfer/b12x kernel stack inside the image
- [0xSero/deepseek-v4-flash-0731-spark](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) — the single-Spark EXL3/REAP build (reference)
- [Loke-60000/deepseek-v4-flash-0731-spark-vision](https://huggingface.co/Loke-60000/deepseek-v4-flash-0731-spark-vision) — community Spark vision serving fix (surveyed, not adopted)
- [msuiche/DeepSeek-V4-Flash-0731-cyber-abliterated-cvec](https://huggingface.co/msuiche/DeepSeek-V4-Flash-0731-cyber-abliterated-cvec) — our steering vector

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
[`BENCHMARK.md`](BENCHMARK.md); the working values are in `.env.v027.working`.

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
