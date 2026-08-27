# Reference model files for the structural hotfix tests

`qwen3_8_flash_next.py` is a **byte-identical** copy of the day-0 image's
model file, pulled from `vllm/vllm-openai:qwen38-flash-next`
(vllm 0.1.dev20073+g8e685d198) at

```
/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/model.py
```

(md5 `b55a94c09438bde77d625159454a4b2e`; pulled 2026-08-26 for
`refusal-research/experiments/20260826-flash-next-vllm-capture/img_src/`).

The local `../vllm` checkout predates the arch — day-0 support for
`qwen3_8_flash_next` is image-only — so the structure test for
`../hotfix-qwen38fn-steering-projective.py` applies the hotfix to a scratch
copy of THIS file. Keep it byte-identical to the image: on an image bump,
re-pull the file from the new container and diff — drifted anchors are
exactly what the hotfix's fail-closed anchor check exists to catch.
