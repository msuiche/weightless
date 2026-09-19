# Reference model files for the structural hotfix tests

`qwen3_8_flash_next.py` is a **byte-identical** copy of the day-0 image's
model file, pulled from `vllm/vllm-openai:qwen38-flash-next`
(vllm 0.1.dev20073+g8e685d198) at

```
/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/model.py
```

(md5 `b55a94c09438bde77d625159454a4b2e`; pulled 2026-08-26 for
`refusal-research/experiments/20260826-flash-next-vllm-capture/img_src/`).

`glm5next.py` is a **byte-identical** copy of the GLM-5.3-Flash model file
from the day-0 support PR (vllm-project/vllm#53906), fetched 2026-08-27:

```
https://raw.githubusercontent.com/ZJY0516/vllm/142062f13d16bed254b5d97cc3d371fbd4f7790a/vllm/models/glm5next/nvidia/model.py
```

(md5 `a357ebb22402cdcf946d5c33950d04f0`). CAVEAT: this is the PR branch head,
not a file pulled out of the `vllm/vllm-openai:glm53-flash` image (published
2026-08-26, one day before this PR revision) — the first real deploy should
diff the image's `vllm/models/glm5next/nvidia/model.py` against this copy and
re-vendor if the image differs. The hotfix's fail-closed anchor check refuses
to patch a drifted file either way.

`glm5next_v11_dflash2.py` is a byte-identical copy of
`vllm/models/glm5next/nvidia/model.py` pulled out of the
`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` image on 2026-09-07
(md5 `b6c8eb2d6a3e28339cda4e14deac874e`). v11 restructured the forward loop
for DFlash2 EAGLE-3 aux hidden-state capture, so the hotfix carries a second
forward-apply anchor variant (the steering apply runs after the aux capture:
the block nulls residual/post/comb and widens hidden_states, which the aux
branches cannot consume, and the drafter was trained on unsteered aux
states). The structure test applies to both references:

```
python3 tests/structure/test-glm53-steering-structure.py            # v8 (glm5next.py)
python3 tests/structure/test-glm53-steering-structure.py patches/reference/glm5next_v11_dflash2.py
```

The local `../vllm` checkout predates the arch — day-0 support for
`qwen3_8_flash_next` is image-only — so the structure test for
`../hotfix-qwen38fn-steering-projective.py` applies the hotfix to a scratch
copy of THIS file. Keep it byte-identical to the image: on an image bump,
re-pull the file from the new container and diff — drifted anchors are
exactly what the hotfix's fail-closed anchor check exists to catch. The same
applies to `glm5next.py` and `../hotfix-glm53-steering-projective.py`.

`deepseek_v4_nvidia_model.py` is a **byte-identical** copy of the day-0
image's model file, pulled from the `dsv4-0731` Modal volume's `src/` dump
(the `fetch_model_src` step of
`refusal-research/experiments/20260905-dsv4-residual-glp`, which copied it
out of `vllm/vllm-openai:deepseekv4-flash-vision` — the Vision-Exp day-0
tag; the text stack is the same file) at

```
/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py
```

(md5 `f897a3354a9ac7508c20be7c5d5f7d63`; pulled 2026-09-05, vendored into
this repo 2026-09-19). This is the reference for
`../hotfix-dsv4-steering-projective.py` (its four anchors all match) and for
the plugin adapter pin in
`vllm-plugin/tests/test_archs/test_dsv4.py`. On an image bump, re-pull and
diff — drifted anchors are exactly what the fail-closed checks exist to
catch.

`qwen3_8_flash_next_ple_layer.py` is a byte-identical copy of the same
image's `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` (md5
`eb23dad30fbb00590704288bcc5010a2`, same pull) — the reference for
`../patch-qwen38fn-ple-fp8-nvfp4.py`, which teaches the day-0 image to load
the RadixArk NVFP4 checkpoint's FP8-serialized PLE N-gram table (found during
Modal B200 validation 2026-08-29: without it, NVFP4 serving dies on the
unknown parameter `ngram_embedding.weight_scale`).

`deepseek_v2_glm53xl.py` is a byte-identical copy of the GB10 kernel-overlay
`deepseek_v2.py` that actually serves the GLM-5.3 743B lane — vLLM
0.23.1rc1-era `deepseek_v2.py` plus the `GlmMoeDsaForCausalLM` class
(tonyd2wild's stack bind-mounts it over the image's file at
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py`):

```
https://raw.githubusercontent.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark--36tok-s/main/kernels/deepseek_v2.py
```

(md5 `7fc0271cb6587dcd69fd32a0ec660b32`; fetched 2026-08-30 from that repo's
main). This is the reference for `../hotfix-glm53xl-steering-projective.py`.
NOTE: our GLP-77 capture/steering validation ran against vLLM v0.28.0's
`deepseek_v2.py` (refusal-research
experiments/20260829-glm53-flagship/upstream_src/deepseek_v2_v0280.py) — the
anchor strings happen to be identical in both, but the lane hotfix anchors on
THIS file because it is what tonyd2wild's image will execute. On an overlay
bump, re-fetch and diff.

`glm5next_b12x_exl3.py` is a byte-identical copy of the GLM-5.3-Flash model
file from brandonmusic's EXL3/B12X fork image
(`verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-*`), extracted
2026-08-30 from OCI layer
`sha256:7f03081ec4e66729470668e9b4ff5825e57ea07f7bcae650db72763445400cdb`
(the fork lives at `/opt/infernal-invocation/vllm` on PYTHONPATH; the file is
`vllm/models/glm5next/nvidia/model.py` in it):

(md5 `fc6efc65cddc2f75ea6c6e6d8c9afc31`). This is the reference for
`../hotfix-glm53-exl3-steering-projective.py`. The fork adds a DFlash
aux-hidden-state branch — TWO decoder loops, nested one level deeper than the
day-0 file — which is why the EXL3 variant has two forward anchors.

`qwen3_next_v0280.py` is a **byte-identical** copy of vLLM v0.28.0's
`vllm/model_executor/models/qwen3_next.py` (md5
`67beddd57e889a8841e8f47894b21d01`), fetched 2026-09-19 from

```
https://raw.githubusercontent.com/vllm-project/vllm/v0.28.0/vllm/model_executor/models/qwen3_next.py
```

This is the structural reference for the qwen38 plugin adapter
(`vllm-plugin/weightless_steer/archs/qwen38.py`): `Qwen3_5Model` inherits
`Qwen3NextModel.forward` unchanged, so the adapter's copied layer loop pins
against THIS file. The Modal preflight re-checks the same anchor against the
image's own copy before any GPU spend; on an image bump, re-pull and diff.

`hy_v4_nvidia_model.py` is a byte-identical copy of the day-0 image's model
file, pulled from `vllm/vllm-openai:hy4-preview` at

```
/usr/local/lib/python3.12/dist-packages/vllm/models/hy_v4/nvidia/model.py
```

(md5 `5dae595ddd3ba1f8f09a43682fe77ebf`; extracted 2026-09-02 by the
`fetch_model_src` step of
`refusal-research/experiments/20260902-hy4-preview-glp/staging/modal_app.py`,
kept there under `staging/srcdl/src/hy_v4/nvidia/model.py` and used as the
fail-closed anchor reference for that lane's patcher). This is the reference
for `../hotfix-hy4-steering-projective.py` and for the plugin adapter
`vllm-plugin/weightless_steer/archs/hy4.py`.
`ouro_v0260.py` is a **byte-identical** copy of
`vllm/model_executor/models/ouro.py` at the vLLM v0.26.0 tag:

```
https://raw.githubusercontent.com/vllm-project/vllm/v0.26.0/vllm/model_executor/models/ouro.py
```

(md5 `7aac1c6ff186b59f13857be82000b5f2`; fetched 2026-09-19). Ouro was
removed from upstream main in #49786 (~2h after the v0.26.0 image was
built), so v0.26.0 is both the last carrying the arch and the pin target —
this is the reference for `../hotfix-ouro-steering-projective.py` and for
the plugin adapter `vllm-plugin/weightless_steer/archs/ouro.py` (whose
copied forward loop test_ouro.py pins against this file).
