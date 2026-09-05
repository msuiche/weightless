# GLM-5.3-Flash TP=2 lane — NVFP4 on 2x DGX Spark + optional GLP-44

Default agentic profile for **two** GB10/SM121 Sparks: RedHatAI NVFP4,
131,072-token context, fp8 KV, CUDA graphs, port **8080**, model ID
**`glm53-flash`**. GLP-44 steering uses the existing projective hotfix at
alpha **2.0** on both ranks. Clear `WEIGHTLESS_STEER_PATH` to disable it.

**Structure-validated; NOT yet booted by us.** Hardware evidence belongs to
[tonyd2wild's TP2 deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark),
read at commit `050081dc41ce6edd4d3f15fa19dc3410ba4210e3` (2026-09-05).
Its [DEPLOY-REPORT](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark/blob/050081dc41ce6edd4d3f15fa19dc3410ba4210e3/docs/DEPLOY-REPORT.md)
records the working TP2 stack and day-0 failures. Later README corrections
supersede its original ModelOpt checkpoint and pinned-KV advice. Our exact
128K, no-speculation, graph-enabled profile, with or without GLP-44, still
needs GPU boot, long-prefill, tool-call, and output-quality validation.

## Model, image, and memory

Use **[`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)**,
compressed-tensors **W4A4**. The launcher rejects other model IDs and checks
the cached config's quantization method. LibertAIDAI/ModelOpt builds have
intermittent corrupted token IDs ([vLLM #54150](https://github.com/vllm-project/vllm/issues/54150));
the reference's Hangul probe counted 4/9/8 replacement characters versus
0/0/0 with RedHatAI. Corruption inside tool calls can desynchronize parsing.
The older report's weight-only quant and throughput figures are not W4A4
quality measurements.

The actual base TP2 launcher pins
**`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8`**. Published manifest digest:
`sha256:d77d375c742fc54f436dec5108b440f58f021bc6600052bf0e8fe5840357e78f`.
The script accepts that tag with an optional `@sha256:...` suffix and
requires identical image IDs on both nodes. This is the custom SM121 patch
stack on `vllm/vllm-openai:glm53-flash-arm64-cu130`, not a stock release.
The separate `sm121-v11-dflash2` image modifies model capture for its drafter
and is outside this lane. v9/InstantTensor failed multi-node in the reference.

**TP2 is KV-starved:** budget roughly **97 GiB weights/rank** and only
**4.5–5.5 GiB KV headroom**, dependent on rank, loader, and profile; this is
not a guaranteed allocation. The reference memory ladder actually survived
at 4.14 GiB with MTP and failed at 5.5 GiB. Worker headroom can be several
GiB below the head, and the minimum rank limits the pool. `MemAvailable`
includes reclaimable page cache that GB10's GPU allocator may not reclaim.

Keep **`GPU_MEMORY_UTILIZATION=0.85`** and let vLLM's profiler size KV after
activation peaks; do not copy the TP4 24 GiB pin or the early TP2 launcher's
`--kv-cache-memory`. The reference's later long-prompt experiments withdrew
the pinned-budget recommendation: allocation and a short answer can pass,
then a long prefill OOMs. Lower 0.78–0.80 settings starved KV in the original
boots. This script rejects utilization above 0.85 and explicit KV pins.
See the reference's [current KV ceiling corrections](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark/blob/050081dc41ce6edd4d3f15fa19dc3410ba4210e3/README.md#kv-pool-ceiling-on-tp2-2026-08-28).

The default reserves more room for agent requests: `MAX_MODEL_LEN=131072`,
`MAX_NUM_SEQS=2`, `MAX_NUM_BATCHED_TOKENS=2048`, **no MTP or DFlash2**.
MTP adds roughly 5 GB per rank and caused unified-memory OOMs in the report.
Retained reference flags: Marlin, `--block-size 2304`, `fp8_e4m3` KV,
multi-node `mp`, and `glm47` tool / `glm45` reasoning parsers. Thinking stays
on for agentic use so the parser separates reasoning from answer content.

**CUDA graphs stay ON:** explicit `FULL_AND_PIECEWISE`, no `--enforce-eager`.
The original report deferred graphs; the later
[2026-09-02 TP2 comparison](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark/blob/050081dc41ce6edd4d3f15fa19dc3410ba4210e3/docs/TP2-SPEC-DEPTH-AND-KV-2026-09-02.md)
ran graphs successfully with DFlash2 and found roughly flat average speed.
It kept eager as a performance choice, not a GB10 correctness requirement.
That comparison does not validate graph capture with our GLP hotfix.

To reach the **262K serving ceiling**, set `MAX_MODEL_LEN=262144` in the env,
retain profiler-sized KV, then validate long prompts and concurrent requests
on both ranks before making it the default. This lane caps at 262144;
that is an operational limit, not the model's native context limit. Inspect
per-rank KV bytes and real preemptions; reported token pools at different
context lengths are not directly comparable.

## GB10 traps carried into this launcher

- Stock SM12x sparse MLA requires DeepSeek's `pe_dim=64`; GLM uses NoPE
  (`pe_dim=0`). v8 includes the SM90 NoPE backend extension with FA2 on GB10.
- FlashInfer 0.6.17 produced NaNs for 64–256-row batches. The reference pins
  `0.6.18.dev20260819`, restores NCCL **2.30.7** and cutlass-dsl **4.6.2**
  after nightly dependency downgrades, disables PDL on SM12x, initializes
  top-k entries to -1, and clamps pool expansion. v8 also fixes fp8 MLA's
  Hopper-sized shared-memory tile for GB10's smaller limit.
- Block **2304** aligns both MLA and kpool's required 64-entry pages;
  the automatic 2176 choice dies in DeepGEMM warmup.
- Published v8 still needs the decode top-k fix past roughly 24K context.
  The script stages and mounts the existing
  [`patches/vendor/sparse_attn_indexer_kpool_sm121.py`](../../patches/vendor/sparse_attn_indexer_kpool_sm121.py)
  on both nodes. Missing shared patches abort the boot.
- Keep weights on local NVMe on each node. Keep swap available with
  `vm.swappiness=0` (recheck after reboot): default swappiness caused UVM
  livelock during load; disabling swap entirely killed Marlin repack.
  The launcher checks this, GPU processes and available memory, and requires
  `sync`/`drop_caches` through non-interactive sudo on both nodes. The
  reference's `cache_flusher.sh` remains useful during shard loading.
- Any running vLLM container, recognized serving lane, or GPU process on
  either node blocks launch. Existing same-name stopped containers also
  block it. Save logs, stop **both** old ranks, and remove their containers
  explicitly. This script never removes a container or crash-loops it;
  logs rotate at **50 MB × 2**. After partial startup failure, stop the
  surviving rank before retrying. Run only one launch operation at a time.

## Steering anchors and validation

The v8 build chain does not patch `glm5next/nvidia/model.py`; its v2 debug
branch is not an ancestor of v8. We extracted that file directly from the
published v8 image's vLLM install layer (without running the image):

```text
image config: sha256:08e3703a018ecac5150c1c756d92711e10af808a5c2bb9377088ff9db43967f2
vLLM layer:   sha256:2c55b4653d4b2c7d4497169b14edc16f44b3fc3058a9ab9cd302e365783e7cbb
model.py:    sha256:ca6320e867b41a90b7c007d9d91021ac8bc1379292a98e8acb5ff9d01b24c2ea
```

`python3 scripts/test-glm53-steering-structure.py /tmp/glm53tp2-image-model.py`
passed all checks: all five unique anchors, patched Python AST, per-layer
buffers, post-layer mHC projection, last-layer deferral, idempotence, and
failure with missing anchors when steering is requested. This is source
compatibility evidence, not GPU or output-quality validation.

Every steered launch runs the existing hotfix against the **installed file
in the selected image on each node**, in disposable CPU-only containers,
before either serving rank starts. It checks anchors and validates the GGUF.
Both serving containers then run `python3 ...hotfix.py && exec vllm serve ...`;
any patch/vector failure aborts instead of serving unsteered. Runtime vector
width/layer errors also fail closed in the shared hotfix.

Vector: `msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44`, file
`GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf`, at the HF cache root
on **both** nodes. Container env: `WEIGHTLESS_STEER_PATH=/cache/huggingface/<file>`
and `WEIGHTLESS_STEER_ALPHA=2.0`. Empty `WEIGHTLESS_STEER_LAYERS` uses all
44 directions. Alpha >=2.5 garbles this model; do not raise it. Confirm
`weightless GLP steering active ... layers=44` in both logs and test a
known-refusal prompt after the first hardware boot.

## Bring-up

On both nodes, pull the pinned image and download the RedHatAI snapshot
into the configured HF hub cache (the layout must contain `models--...`,
`refs/main`, and `snapshots/...`):

```bash
docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8
HF_HUB_CACHE=/home/USER/.cache/huggingface hf download RedHatAI/GLM-5.3-Flash-NVFP4
# Optional; the GLP repo is gated and needs your accepted HF token:
hf download msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44 \
  --include '*.gguf' --local-dir /home/USER/.cache/huggingface
```

On the head, use this checkout or stage the script/env into `~/dspark-glm53tp2`
with the existing hotfix and vendored kpool file in its `patches/` directory.
Copy `.env.glm53tp2.example` to `.env.glm53tp2`, replace the placeholders,
verify both nodes' fabric NIC/GID settings and passwordless SSH/sudo, then:

```bash
bash start-glm53-flash-tp2.sh
```

The head stages the shared patches on the worker, checks both nodes,
starts rank 1, waits 25 seconds, then starts rank 0. Readiness polls
**`/health`**, not `/v1/models` (which can return 200 with a dead engine).
Allow roughly 15 minutes and inspect both logs if it fails.

The `setup.py` lane entry supports env generation, the existing steering
structure test, and deploy-command construction. **Wizard SSH deployment
has an existing integration gap:** `remote_preflight()` uses separate
index-keyed `DEPLOY_MAP` / `CONTAINER_GREP` tables that lack this new lane
(and Inkling). Selecting that path raises `KeyError`; skip wizard SSH
deployment and launch from the head as above. This change is restricted
to a `LANES` entry; those tables are outside the permitted edit scope.

After `/health` passes, test real generation and tool calling:

```bash
curl -fsS http://HEAD:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash","messages":[{"role":"user","content":"2+2=?"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
curl -fsS http://HEAD:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash","messages":[{"role":"user","content":"Use lookup to look up Paris."}],"tools":[{"type":"function","function":{"name":"lookup","description":"Look up a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"auto","max_tokens":512,"temperature":0}'
```

Check coherent content and structured `tool_calls`, then exercise long
prefills beyond 24K, the configured context limit, and two concurrent
requests while watching both ranks for OOMs/preemption. No benchmark numbers
are claimed for this lane. The default is text/tools; vision requires the
reference's multimodal chat template and separate validation.
