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
| `recipe/` | `Dockerfile.gguf-dep` (**required**, adds `gguf`), `Dockerfile.steering-overlay` (optional, bakes the patch), and the working `docker-compose.v027.yml` |
| `scripts/` | build-guard tests |
| `spec/CONTROL-VECTOR.md` | the projective control-vector GGUF format: the `dspark.mode` contract, layer-id mapping, and why an additive reader must refuse the file |

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

## Status

**Serving.** Continuously up on 2x DGX Spark since 2026-08-17, TP=2, steering
active on 29 layers. Measurements and their full configuration are in
[`BENCHMARK.md`](BENCHMARK.md); the working values are in `.env.v027.working`.

Settled configuration:

| | |
|---|---|
| image | `vllm-dspark-steering:v027-gguf` (v027 + `gguf==0.19.0`) |
| context | **524,288** — 65,536 is needlessly small, 1,048,576 collapses prefill to 110 tok/s |
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
