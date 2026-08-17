# dspark-deploy

Serving DeepSeek V4 Flash 0731 on 2x DGX Spark (GB10, SM121) with projective
refusal steering.

Replaces the previous `DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`
fork, which vendored 18 vLLM files. DeepSeek V4 Flash is native in vLLM as of
v0.27.0, so 12 of those files now exist upstream and the vendored copy of the
model implementation was 1,527 lines behind it. What is genuinely ours is the
steering hook, and that is a 271-line patch to one file.

## Layout

| | |
|---|---|
| `patches/` | the steering patch against vLLM v0.27.0, applied after docker |
| `recipe/` | overlay Dockerfile that applies it to a v027 base image |
| `scripts/` | build-guard tests |

Maintenance home for the patch is the `dspark-steering-v027` branch of the
private `msuiche/vllm` fork, so upstream bumps get a real 3-way merge instead of
a hand-resolved `.patch` conflict. Regenerate with:

```sh
git -C ../vllm diff v0.27.0..dspark-steering-v027 \
  -- vllm/models/deepseek_v4/nvidia/model.py \
  > patches/0001-dspark-projective-steering.patch
```

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

```sh
DSPARK_STEER_PATH=/cache/huggingface/DeepSeek-V4-Flash-0731-cyber-abliterated-cvec-L10-38-a4.gguf
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

## Status

Untested end to end. The patch applies cleanly to a pristine v0.27.0 tree and
compiles, and `tests/dspark_steering_test.py` in the vllm fork passes 15 checks
against the real 29-layer control vector, including that all 29 layers reach the
stack and that alpha=2 reflects rather than removes. Nothing has been served on
v027 yet, and there is no throughput baseline on the outgoing image to compare
against, so the performance question is still open.
