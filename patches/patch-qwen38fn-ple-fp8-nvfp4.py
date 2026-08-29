#!/usr/bin/env python3
"""Fail-closed runtime patch: teach the day-0 image's PLE layer to load the
FP8-serialized PLE n-gram table shipped inside the RadixArk NVFP4 checkpoint
(NVFP4-serve lane, experiments/20260827-flash-next-nvfp4-serve).

The bug: `Qwen3_8FlashNextNGramEmbedding` picks its quant method via
`_get_ple_embedding_quant_method`, which only returns the FP8 embedding
method when the GLOBAL quant config is an `Fp8Config`. The RadixArk
checkpoint's global config is modelopt_fp4, but its PLE table is NOT nvfp4 --
`*.ple.*` sits in hf_quant_config.json's exclude_modules and the table is
stored as 128 FP8_E4M3 shards ([2500012, 160] each) plus one global BF16
scale in model-plefp8-*.safetensors. With the stock image:

  * the embedding is built UNQUANTIZED (bf16 VocabParallelEmbedding),
  * the PLE CPU-offload worker then dies in load_weights with
    "There is no module or parameter named 'ngram_embedding.weight_scale'"
    (run3, 2026-08-29),
  * and even without offload the same ValueError fires on the GPU side --
    the day-0 image cannot serve this checkpoint's PLE table at all.

The fix: when the quant config is present but not an Fp8Config (i.e. the
modelopt_nvfp4 case), build the PLE embedding with the existing FP8 embedding
method -- fp8 weight + global bf16 scale -- which is exactly how this
checkpoint serializes the table. The FP8 serving path around it
(`_offload_weight_scale` retention on the GPU worker, fp8
`get_offload_output_dtype`, `_dequantize_embeddings` applying the scale) is
already in the image and needs no changes.

Anchors against the image's
`vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py` (vendored copy:
../20260826-flash-next-vllm-capture/img_src/). Exit codes: 0 patched (or
already patched), 1 anchor mismatch (image drifted -- do NOT proceed).
"""

import importlib
import os
import py_compile
import sys

MARKER = "nvfp4-ple-fp8"

ANCHOR = '''    if not isinstance(quant_config, Fp8Config):
        return None
'''

REPLACEMENT = '''    if not isinstance(quant_config, Fp8Config):
        # [nvfp4-ple-fp8] The RadixArk NVFP4 checkpoint runs under a
        # modelopt_fp4 global quant config, but its PLE n-gram table is
        # FP8_E4M3 shards plus one global BF16 scale (model-plefp8-*.safetensors;
        # "*.ple.*" is in hf_quant_config.json exclude_modules, so the table
        # is not nvfp4). The Fp8Config test never fires for it, the embedding
        # is built unquantized, and loading dies on the unknown parameter
        # 'ngram_embedding.weight_scale'. Quantized checkpoints of this arch
        # all serialize PLE the same FP8 way, so: quant config present (and
        # not the Fp8Config case handled below) -> FP8 PLE embedding method.
        if quant_config is not None:
            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()
        return None
'''


def _find_ple_file() -> str:
    # Test/override hook first (same convention as the steering hotfix's
    # WEIGHTLESS_STEERING_MODEL_PY).
    override = os.environ.get("WEIGHTLESS_PLE_LAYER_PY")
    if override:
        return override
    # Same module-layout duality as the steering hotfix: the day-0 image
    # packages the arch as qwen3_8_flash_next; the PR branch renamed it.
    for mod in ("vllm.models.qwen3_8_flash_next.nvidia.ple_layer",
                "vllm.models.qwen4_exp.nvidia.ple_layer"):
        try:
            m = importlib.import_module(mod)
            return m.__file__
        except ModuleNotFoundError:
            continue
    raise SystemExit("PATCH FAILED: no qwen3_8_flash_next/qwen4_exp ple_layer "
                     "module found in this vllm install")


def main() -> None:
    path = _find_ple_file()
    src = open(path).read()
    if MARKER in src:
        print(f"[{MARKER}] already applied to {path}")
        return
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"PATCH FAILED: anchor found {n} times (expected 1) "
                         f"in {path}; image drifted, do not proceed")
    out = src.replace(ANCHOR, REPLACEMENT)
    open(path, "w").write(out)
    py_compile.compile(path, doraise=True)
    print(f"[{MARKER}] applied to {path} (1 anchor)")


if __name__ == "__main__":
    main()
