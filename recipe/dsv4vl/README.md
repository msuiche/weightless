# recipe/dsv4vl — DSV4-Vision-Exp VL lane (weightless lane 12)

Vision-capable serving of DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark
(GB10, SM121, TP=2 over RoCE). This is the VL counterpart of lane 11:
lane 11 serves the vision-**stripped** text tree on the Anemll 0.25.2
image; this lane serves the **unstripped** checkpoint (hub snapshot
`6821d6ad`, FP8) with the vision tower live, on the day-0 VL image.

## Why the files live here

Wizard lanes are self-contained: `DEPLOY_MAP` syncs a lane's
env/compose/launcher/patches to the rig, and the serve flow owns park →
assets → boot → tests. The VL bring-up lived in dspark-fork
(`recipe/upstream-vision/`, phases 1-4 + K1-K4, see
`benchmarks/20260920-upstream-vision-*.md` there) because that repo owns
the upstream-image investigation. What shipped here is the productionized
subset: the K4-validated compose shape + launcher + the steering hotfix,
rewired to the wizard contract (env placeholders, fail-closed validation,
steering-active log gate). dspark-fork stays the test-lane; this dir is
the lane of record.

## The image

`vllm-upstream-vision:k2-topk512` — `vllm-upstream-vision:phase1` (day-0
`vllm/vllm-openai:deepseekv4-flash-vision` + the phase-1 L-layers, incl.
the b12x package) with flashinfer-python 0.6.18 replaced by the
dual-prefill topk 192/256/512 backport
([msuiche/flashinfer@sm121-dsv4-prefill-topk512](https://github.com/msuiche/flashinfer/tree/sm121-dsv4-prefill-topk512)).
Without the backport, VL prefill dies at warmup
(`Unsupported sparse-MLA prefill configuration: ... topk=512`) because the
vision wrapper widens every SWA index row to 128 + 384. The lane declares
`local_image=True`: the wizard verifies presence with
`docker image inspect` on both nodes instead of pulling.

Rebuild (on the head, CPU-only):

```sh
git clone --branch sm121-dsv4-prefill-topk512 https://github.com/msuiche/flashinfer ~/flashinfer-sm121/src
# build the wheel + precompile the sparse_mla_sm120 module for 12.1a
# inside the day-0 image, then:
docker build -f recipe/upstream-vision/Dockerfile.k2-flashinfer-topk512 \
  -t vllm-upstream-vision:k2-topk512 ~/flashinfer-sm121
docker save vllm-upstream-vision:k2-topk512 | ssh <worker> docker load
```

The exact, fail-closed procedure (AOT swap — required, because
`JitSpecNvcc.try_load()` returns the stock flashinfer-jit-cache artifact
unconditionally) is in dspark-fork `recipe/upstream-vision/Dockerfile.k2-flashinfer-topk512`
+ `verify-k2-flashinfer-topk512.py`; the build aborts unless the patched
dispatch, the TK=512 kernel symbol, and the post-swap sha256 all verify.

## Steering

`patches/hotfix-dsv4vl-steering-projective.py` — bind-mounted by the
compose and executed in the entrypoint before `exec vllm serve`
(`|| exit 1`). Same injected code as the Anemll-lane hotfix (the
structure test byte-compares them); the four anchors were re-verified
against the day-0 tree at vLLM `5ab628dd1`, where the model loop carries
the pending FFN write between layers (fold deferred to the next layer's
`mhc_fused_post_pre_tilelang`) — the same deferred-fold convention as the
0.25.2 tree, so the hook stays `ffn_out_pre_residual` and the served
vector is the `-ffn` relabel of the GLP-29 Vision-Exp cvec (the original
residual-site file is refused by the loader). α=1.0, L10–38.

The VL wrapper's construction path is covered:
`DeepseekV4ForConditionalGeneration` builds its language model via
`init_vllm_registered_model(..., architectures=["DeepseekV4ForCausalLM"])`,
whose `.model` is the same `DeepseekV4Model` the hotfix patches.

## Nospec, stated plainly

No `--speculative-config` on this lane. The Vision-Exp checkpoint declares
`num_nextn_predict_layers=3` (vLLM requires k % 3 == 0) and
`vl_model.py`'s weights mapper drops `mtp.` weights for the vision
variant — even a divisible k has no draft head to load. MTP_NUM_TOKENS=6
satisfies the divisibility but not the weights; revisit if upstream VL
spec support lands.

## Files

- `.env.dsv4vl.example` — site template (placeholders; the real
  `.env.dsv4vl` is gitignored)
- `docker-compose.dsv4vl.yml` — one service, host network, nospec VL
  serve + hotfix entrypoint
- `start-dsv4-vl-dspark.sh` — worker-first launch, per-node RoCE GID
  resolution (never pinned), fail-closed env validation, steering-active
  log gate on both ranks, API + chat smoke
- `../../patches/hotfix-dsv4vl-steering-projective.py` — the steering
  hotfix (deployed under `patches/` on both nodes)

## Validation record

- 2026-09-20 (K4 window): the day-0 VL boot this lane productizes passed
  warmup through the backported TK=512 dual prefill and
  `vision_smoke_probe` went green (image tokens 315∈[314,317] and
  111∈[110,113] on the controlled grids, image+text mix, text-only
  regression, 2-way concurrency).
- 2026-09-21 (L6 window): steering A/B + vision smoke under steering —
  see CHANGELOG.
