# Inkling-Small TP=2 lane — NVFP4 on 2x DGX Spark + GLP-41 steering

Inkling-Small (`thinkingmachines/Inkling-Small-NVFP4`, 159 GiB) on stock
vLLM v0.28.0 (inkling is day-0 since that release), tensor-parallel across
**both** Sparks (head + worker over RoCE), with the GLP-41 projective refusal
vector applied via a bind-mounted pre-patched `model.py` (the same integration
pattern as the qwen38fn lane).

**STRUCTURE-VALIDATED ONLY (2026-09-02)** — written from the measured qwen38fn
lane and the Modal validation (4×H100, vLLM 0.28.0: refusal32 0/32 → 30/32 at
α=0.25, benign 30/32) but not yet booted on the Sparks. First boot: watch it.

| file | what it is |
|---|---|
| `start-inkling-dspark.sh` | head+worker boot with the wedge-proofing from the qwen38fn saga (preflight free-memory gate + zombie check, drop_caches on both nodes, `--restart no`, capped logs) |
| `.env.inkling.example` | full config with site values as `<...>` placeholders |
| `../../patches/hotfix-inkling-steering-projective.py` | the steering hook for `vllm/models/inkling/nvidia/model.py` — handles Inkling's deferred residual add (`pending` flush via the file's own `_sconv_add_norm` idiom) |

## Traps

- **α=0.25 is calibrated, not a default.** α=1.0 and α=0.5 garble EVERYTHING
  on this model (including benign) — the most dose-sensitive model in the
  program. Do not raise it.
- **The steered `model.py` must be pre-patched and staged on both nodes**
  (`files/inkling-model-steered.py`): extract the image's file once, apply the
  hotfix offline (`WEIGHTLESS_STEERING_MODEL_PY=<copy> python3
  ../../patches/hotfix-inkling-steering-projective.py`), copy to both nodes.
- **drop_caches needs passwordless sudo** on both nodes
  (`/etc/sudoers.d/drop-caches` — see the qwen38fn README trap; the script
  uses `sudo -n` and will fail loudly without it).
- **NVFP4 is 159 GiB → ~85 GiB/rank at TP=2** — fits 2 Sparks with KV at
  util 0.835 (the measured GB10 envelope from the qwen38fn lane).
