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

The local `../vllm` checkout predates the arch — day-0 support for
`qwen3_8_flash_next` is image-only — so the structure test for
`../hotfix-qwen38fn-steering-projective.py` applies the hotfix to a scratch
copy of THIS file. Keep it byte-identical to the image: on an image bump,
re-pull the file from the new container and diff — drifted anchors are
exactly what the hotfix's fail-closed anchor check exists to catch. The same
applies to `glm5next.py` and `../hotfix-glm53-steering-projective.py`.

`qwen3_8_flash_next_ple_layer.py` is a byte-identical copy of the same
image's `vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` (md5
`eb23dad30fbb00590704288bcc5010a2`, same pull) — the reference for
`../patch-qwen38fn-ple-fp8-nvfp4.py`, which teaches the day-0 image to load
the RadixArk NVFP4 checkpoint's FP8-serialized PLE N-gram table (found during
Modal B200 validation 2026-08-29: without it, NVFP4 serving dies on the
unknown parameter `ngram_embedding.weight_scale`).
