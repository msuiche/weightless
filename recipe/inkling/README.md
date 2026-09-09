# Inkling-Small TP=2 lane — NVFP4 on 2x DGX Spark

Inkling-Small (`thinkingmachines/Inkling-Small-NVFP4`, 159 GiB) on stock
vLLM v0.28.0 (inkling is day-0 since that release), tensor-parallel across
**both** Sparks (head + worker over RoCE), with SM121 attention and GB10 load
reclaim patches. The SM121 launcher supports GLP-41 at α=0.1 using a combined
steering + load-reclaim model patch. An empty `WEIGHTLESS_STEER_PATH` selects
the unsteered model patch.

**WORKING (2026-09-04)** — real Inkling-Small-NVFP4 weights booted TP=2 on
2×GB10 with CUDA graphs enabled. The 4-prompt API smoke passed. Use the SM121
start script and both hotfixes below; the generic/steered lane is separate.

**Unsteered agent validation (2026-09-05):** at ctx 65536 / util 0.78 with both native
tool-parser flags, all four endpoint smoke tests passed through the router
(model discovery, chat, structured tool call, Mac omp file creation). Hermes
on the DGX head also created and read back a scratch file. Large first prompts
can take minutes, especially with competing requests; subsequent Hermes model
calls in that test took 6.9 and 4.5 seconds. Larger context remains unvalidated.
The later normal Mac omp probe completed in 2.0 seconds with the prompt cache
warm. A fresh Hermes CLI session created and read a file in 35.7 seconds after
automatic title generation was disabled. These measurements are not cold-start
guarantees: the user's preceding gateway request took 126 seconds with competing
requests and title retries.

**GLP-41 SM121 validation (2026-09-05):** both ranks loaded the same 41-layer
vector with `alpha=0.100`. Exact `pong` generation passed in 7.2 seconds,
structured `get_weather` passed in 4.3 seconds, and Mac omp created its scratch
file in 74.1 seconds, including cold prompt processing. Head Hermes created
and read back its scratch file in 90.7 seconds (session
`20260905_103152_f722e2`, exit 0). The preceding 0.25
run failed the structured tool test and exposed reasoning in its answer, so
the SM121 default is now 0.1. These checks establish serving/tool compatibility;
they do not reproduce the earlier H100 calibration at this lower strength.

| file | what it is |
|---|---|
| `start-inkling-dspark.sh` | head+worker boot with the wedge-proofing from the qwen38fn saga (preflight free-memory gate + zombie check, drop_caches on both nodes, `--restart no`, capped logs) |
| `start-inkling-sm121.sh` | real-weight GB10 boot: compatibility patches, optional GLP-41 steering, native tool parser |
| `files/inkling-model-gb10-steered.py` | combined model patch retaining GB10 load reclaim and applying GLP to the materialized post-layer residual |
| `.env.inkling.example` | full config with site values as `<...>` placeholders |
| `../../patches/hotfix-inkling-steering-projective.py` | the steering hook for `vllm/models/inkling/nvidia/model.py` — handles Inkling's deferred residual add (`pending` flush via the file's own `_sconv_add_norm` idiom) |
| `../../patches/hotfix-inkling-gb10-load-reclaim.py` | per-tensor source-page and CUDA-cache reclaim that removes the unified-memory load spike |
| `../../patches/hotfix-inkling-sm121-relattn.py` | numerics-validated SM121 attention fallback; v2 supports CUDA graph capture |

## Traps

- **Use α=0.1 for SM121 agent serving.** The earlier H100 calibration used
  0.25, but the live SM121 test at 0.25 leaked reasoning and returned prose
  instead of tool calls. At 0.1, exact-answer chat and native tools passed.
  The direction file retains `a0.25` in its name; the runtime alpha overrides
  that calibration. Values 0.5 and 1.0 also garbled earlier validation output.
- **The steered `model.py` must be pre-patched and staged on both nodes**
  (`files/inkling-model-gb10-steered.py`). The generic
  `inkling-model-steered.py` lacks the GB10 load-reclaim fix and is rejected by
  the SM121 launcher. It checks patch/vector checksums across both ranks and
  mounts the vector read-only at `/opt/weightless/steering.gguf`.
- **drop_caches needs passwordless sudo** on both nodes
  (`/etc/sudoers.d/drop-caches` — see the qwen38fn README trap; the script
  uses `sudo -n` and will fail loudly without it).
- **NVFP4 is 159 GiB → 78.3 GiB/rank at TP=2.** Steady state fits, but stock
  loading does not. Keep lazy safetensors and the load-reclaim hotfix enabled.
  The original short-prompt profile was ctx 8192 and util 0.82. The example
  now matches the agent-client profile: ctx 65536, util 0.78, 2 sequences,
  1024 batched tokens. Validate long prompts before increasing context.
- **Native tools require both parser flags:** `--enable-auto-tool-choice
  --tool-call-parser inkling`, alongside `--reasoning-parser inkling`.
  Omitting them causes Hermes's `tool_choice: auto` requests to fail with
  HTTP 400. The native Inkling parser handles `content_invoke_tool_json`;
  earlier claims that the model was chat-only were incorrect.
- **Test generation after readiness.** `/health` can pass while a first
  request is compiling kernels or processing a long prompt. Use
  `WEIGHTLESS_BASE_URL=http://HEAD:8000/v1 WEIGHTLESS_MODEL=inkling-small-nvfp4
  bash tests/smoke/run.sh` from the repository root. Port 8000 also tests the
  router's streaming path.

## Enabling GLP in the SM121 lane

Set `WEIGHTLESS_STEER_PATH` to the GLP-41 GGUF's host path on both nodes and
`WEIGHTLESS_STEER_ALPHA=0.1` in `.env.inkling`. The launcher selects
`files/inkling-model-gb10-steered.py` automatically. Leave `MODEL_PATCHED_PY`
unset unless overriding it with another file carrying both required patches.
The wizard stages the combined patch on both nodes before boot.

To regenerate the combined file from the GB10-patched source, from this directory:

```sh
cp files/inkling-model-gb10.py files/inkling-model-gb10-steered.py
WEIGHTLESS_STEERING_MODEL_PY=files/inkling-model-gb10-steered.py \
  python3 ../../patches/hotfix-inkling-steering-projective.py
```

After staging identical files and the vector on both nodes, validate them
without restarting the current service:

```sh
bash start-inkling-sm121.sh --check-steering
```

On boot, both ranks must log `weightless GLP steering active` with
`alpha=0.100` and `layers=41`. Then run the endpoint suite and a real client
tool loop. A configured path alone is not proof that steering is active.

## Historical status 2026-09-03: DGX lane was blocked

Fifteen controlled boot attempts on 2× DGX Spark (spark-4687 + spark-5bc3,
vllm/vllm-openai:v0.28.0, TP=2 over RoCE) all die ~28s after weight-load
reaches the final layer, on BOTH nodes, with no Python error — the engine
tears down and `docker logs` shows only the wrapper. NVRM `NV_ERR_NO_MEMORY`
appears in dmesg at the same moment but is a red herring (see probe below).

Eliminated (one variable per boot): GPU_MEMORY_UTILIZATION 0.835/0.89/0.90
(the 0.90 startup-gate fail was arithmetic: 0.90×121.69=109.52 > 109.32
CUDA-free), MAX_MODEL_LEN 262K/128K/32K, MAX_NUM_SEQS 8/4,
MAX_NUM_BATCHED_TOKENS 8192/2048, page-cache/drop_caches, stale containers,
full reboot of both nodes (117 GiB free each, clean), NCCL HCA typo
(`==rocep1s0f0` → `=rocep1s0f0`), NCCL_IB_DISABLE=1 (socket transport),
--enforce-eager (not graph capture), no MTP spec-config, no
VLLM_USE_V2_MODEL_RUNNER, and — critically — **stock unpatched model.py**
(not our hotfix). A driver probe (`torch.empty` 5 GiB chunks) allocates
115 GiB on the worker with no error, so it is not raw capacity.

Not eliminated / current suspicion: inkling day-0 support in v0.28.0 has
never run cross-node TP on aarch64/GB10 (the Modal lane that produced all
our validation numbers is single-node TP=4 on x86 H100). Something in the
post-load phase (profiling forward or first cross-node collective of the
inkling custom ops — short_conv state, rel_attention) kills the worker proc
before it can report ready.

**What works:** the Modal lane — 4×H100 single node, stock vLLM 0.28.0,
`WEIGHTLESS_STEER_PATH=<GLP-41 gguf> WEIGHTLESS_STEER_ALPHA=0.25`,
`--tokenizer-mode inkling --trust-remote-code`. refusal32 0/32 → 30/32,
benign 30/32 (2026-09-02).

**ROOT CAUSE CONFIRMED 2026-09-03** (live probes, full trail in
[`GB10-ANALYSIS.md`](GB10-ANALYSIS.md) §8, in this repo):
GB10 (sm_121) falls through `_use_sheared_bias()` (`major in (10,11)`), so
inkling's rel-attention is sent to the Hopper cute path, which asserts
`Paged KV not supported on SM 12.0 in this PR`. Forcing the intended
tml-fa4 sheared path (gate patched `major >= 10`) gets past weight-load and
KV sizing, then dies at `assert tile_n == 128` in tml_fa4's rel_bias
metadata — inkling is diff-headdim (128/64, rel_extent 1024) and the
SplitKV heuristic shrinks tile_n to 64, which rel_bias rejects. **Neither
FA4 backend in v0.28.0 can run Inkling on GB10; the fix is upstream**
(cute SM12 paged-KV support — marked "in this PR" — or tml_fa4 rel_bias
learning tile_n=64). The gate patch + probes live in
`~/dspark-inkling/files/` on the head for the day the tml_fa4 side lands.

The SM121 fallback and load-reclaim hotfixes now bypass both historical
blockers. See the current working status at the top of this file.

## Watchdog: optional, and dangerous if miscalibrated (2026-09-05)

`../../scripts/memory-watchdog-gpu.sh` (v3) guards the overnight unattended
case: a real GB10 wedge is a D-state process you cannot kill, so the watchdog
kills the container early when MemAvailable collapses. Calibrated wrong it is
worse than the disease — on 2026-09-05 it killed FOUR healthy Inkling runs:
a 2 s dip during KV alloc (thresh 8 GiB, need 2), a 15 s low stretch during
the 64k-ctx boot ("active drain"), a first-request Triton JIT stall (logs
stale >60 s while compiling kernels), and a serving dip to 5 GB while a
160 GB HF download ran alongside (page cache eats the same unified memory).
Every one was a false positive. Rules if you enable it:

- Never during the day's interactive work. Arm it for unattended nights only.
- util 0.78 (not 0.82) so healthy dips stay clear of the threshold.
- thresh 5120 MB, need 3, fresh_need 30, stale 180 s is the least-bad set
  found; expect to tune again per lane.
- Downloads and serving share one memory pool on GB10: pause big HF pulls or
  accept that the watchdog will see "drains" that are just page cache.
