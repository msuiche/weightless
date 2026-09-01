# Qwen3.8-Flash-Next TP=2 lane — NVFP4 on 2x DGX Spark + GLP-47 steering

Qwen3.8-Flash-Next (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, ~135 GB NVFP4) on the
day-0 image `vllm/vllm-openai:qwen38-flash-next`, tensor-parallel across
**both** Sparks (head + worker over RoCE, like the DSV4 lane), with the GLP-47
projective refusal vector applied by a fail-closed boot hotfix. If the nodes
are rebuilt: pull the image and model on both, copy `.env.qwen38fn.example` to
`.env.qwen38fn`, fill in the `<...>` placeholders, and run
`start-qwen38-flash-next-dspark.sh` on the head.

| file | what it is |
|---|---|
| `start-qwen38-flash-next-dspark.sh` | head+worker boot: syncs env + hotfix to the worker, starts the headless rank 1 there, then the API rank 0 on the head |
| `.env.qwen38fn.example` | full config with site values as `<...>` placeholders (the real `.env.qwen38fn` is gitignored) |
| `../../patches/hotfix-qwen38fn-steering-projective.py` | the steering hook; patches the container's `vllm/models/qwen3_8_flash_next/nvidia/model.py` at boot |
| `../../patches/patch-qwen38fn-ple-fp8-nvfp4.py` | required for NVFP4 at all: the day-0 image can't load the checkpoint's FP8-serialized PLE N-gram table without it (found on Modal B200, 2026-08-29); chained fail-closed before the hotfix |
| `../../patches/reference/qwen3_8_flash_next.py` | byte-identical copy of that image file — the structure test's reference (the `../vllm` checkout predates the arch) |

## Traps (each one cost a run somewhere)

- **`--no-enable-prefix-caching` is mandatory.** With this arch, prefix
  caching forces `mamba_cache_mode="align"`, which splits every prefill at a
  block boundary and defers the tail to a second forward — it corrupted the
  capture lane's run 3 and would corrupt steering the same way. The start
  script hardwires the flag.
- **`VLLM_PLE_CPU_OFFLOAD=1` does not work at nnodes=2 on the arm64 image**
  (hard `ValueError: Unsupported settings: nnodes=2`; the x86 build allows it
  — do not copy a B200 config onto the Sparks). Leave it 0: the PLE table then
  lives in HBM (~25.5 GB/rank), so the safe envelope on 128 GB unified memory
  is util ≤ 0.88 with ≤ 128K ctx, or util ≤ 0.80 at 262K. The start script
  fails closed on both, reproduces vLLM's free-memory startup gate before the
  10-minute load, and refuses to boot over leftover GPU processes (a
  crash-loop's CUDA memory survives `docker rm -f`; stacked loads drove the
  pair to 0% free and earlyoom cannot kill a CUDA-stuck process — that wedged
  both nodes on 2026-09-01). Lane containers run `--restart no` with capped
  logs for the same reason.
- **α=1.0 is calibrated, not a default to tune.** GLP-47 was measured
  peaking at α=1.0 (higher over-projects). Do not import the DSV4 lane's 4.0.
- **The hotfix targets a package path, not `model_executor/models/`.** In the
  day-0 image the arch lives at
  `/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/model.py`.
  If a newer image moves it, set `WEIGHTLESS_STEERING_MODEL_PY` (discover
  command in `.env.qwen38fn.example`) — and confirm the boot log reads
  `weightless GLP steering active: hook=post_layer alpha=1.000 ... layers=47`.
- **NCCL GID indexes drift** across reboots; this lane pins
  `NCCL_IB_GID_INDEX=3` (no sysfs auto-resolver — that's anemll-lane
  machinery). Re-verify after reboot.

## Steering vector

The shipping artifact is
**`Qwen3.8-Flash-Next-abliterated-cyber-GLP-47-L1-47-a1.gguf`** — per-layer
difference-of-means over the widened hyper-connection stream (10240 =
4×2560), layers 1–47, α=1.0, spec-conformant per
[`../../spec/GLP.md`](../../spec/GLP.md). Derived 2026-08-26 (8×H100 BF16
reference) and reproduced on the vLLM lane at cos 0.9931; refusal32 3.1% →
81.2% at α=1.0 with benign/capability clean. Published at
[`msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47`](https://huggingface.co/msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47)
(gated — fetch with an HF token), at the **root of the HF cache on BOTH
nodes** so it lands at `/cache/huggingface/` in both containers:

```sh
huggingface-cli download msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47 \
  --include "*.gguf" --local-dir ~/.cache/huggingface   # on BOTH nodes
```

The start script preflights its presence on both nodes when
`WEIGHTLESS_STEER_PATH` is set and refuses to boot otherwise (fail-closed).

## How the steering differs from the Qwen3.8 lane

Not the decomposed `(hidden_states, residual)` convention: this arch uses
delayed-combine hyper-connections. Decoder layers return
`(hidden_states, block_output, injection)` with the MLP output still pending;
the hotfix materializes the post-layer stream per layer with the
parameter-free `mlp_hyper_connection.combine()`, projects the GLP direction
out of it, and continues materialized (next layer takes `mix()` — same math,
one kernel less fused). The apply is unconditional with a dense zero-padded
stack, so the compiled graph is identical for every layer set. No
`__init__`-override trap here (contrast the Qwen3.8 lane's `Qwen3_5Model`):
`Qwen3_8FlashNextModel.__init__` is the serving class's init.

## Validate

```sh
# structural, no GPU/torch: anchors + per-layer-loop + serving-class guards
python3 ../../scripts/test-qwen38fn-steering-structure.py

# the exact injected loader against the real vector (needs torch)
WEIGHTLESS_STEER_PATH=$HF_CACHE/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47-L1-47-a1.gguf \
  python3 ../../patches/hotfix-qwen38fn-steering-projective.py --check
```

Anchors were validated against the image's own file in
`refusal-research/experiments/20260826-flash-next-vllm-capture` (that
experiment's patch also carries a capture probe; this lane ships the steering
half only).
