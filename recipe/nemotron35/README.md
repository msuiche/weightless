# Nemotron 3.5 Lightning — single DGX Spark

Stock `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` serving recipe,
with optional GLP projective steering via a fail-closed boot hotfix.
**Stock generation and a tool-call/result loop passed on one Modal H100;
DGX boot validation is still pending.**

NVIDIA documents one-Spark deployment with vLLM 0.27.1, Marlin W4A16 compute
for the NVFP4 weights, FP8 KV cache, and a separate DSpark draft model.
The model has 30B total / 3B active parameters and a hybrid Mamba/attention/MoE
architecture. NVIDIA reports a 1M context window; this recipe starts at 65,536
tokens until memory, long prompts, and tool calls have been tested locally.
Source: [NVIDIA model card and Spark recipe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4#1x-dgx-spark-gb10),
checked 2026-09-05.

## Modal compatibility result — 2026-09-05

On one NVIDIA H100 80GB, the pinned target and vLLM 0.27.1 passed two exact-answer
prompts (`NEMOTRON_OK`, `42`, both clean `stop`), a structured
`get_weather({"city":"Paris"})` call, and a final response correctly using the
synthetic tool result (21°C, sunny). Readiness took 280.2 seconds; the whole
smoke took 326.6 seconds. Tool generation took 6.89 seconds and the tool-result
continuation 2.30 seconds.

This tested **8,192 context, eager execution, four maximum sequences, no draft**,
Marlin, FP8 KV, FlashInfer Mamba, and FP16 Mamba SSM cache. The DGX launcher uses
65K context and DSpark by default; those differences and GB10 hardware still
need validation. These timings are smoke-test latency, not throughput benchmarks.
The model's default thinking mode remained enabled. No steering was applied.
Complete outputs and server logs are retained in the local experiment directory
`refusal-research/experiments/20260905-nemotron35-stock-smoke/out/`.

## GLP steering — validated on Modal, 2026-09-05

Setting `WEIGHTLESS_GLP` in `.env.nemotron35` to a host path holding a
`glp.*` GGUF control vector enables projective steering
(`h ← h − α(h·d̂)d̂`) at the post-layer residual stream
(`h = hidden_states + residual` under nemotron_h's fused add+norm
convention; the mixer output folds in at the next norm). The boot hotfix
`patches/hotfix-nemotron35-steering-projective.py` patches `nemotron_h.py`
inside the container BEFORE `vllm serve` and fails closed: a missing/invalid
vector, non-`project` mode, wrong `hook_point`, layer-list mismatch, anchor
drift, or a failed patch all stop the boot instead of serving unsteered.
Anchors are verified against both vLLM v0.27.1 and the v0.28.0 image on the
DGX Spark (`patches/reference/nemotron_h_v0280.py`).

Validated vector:
`msuiche/Nemotron-3.5-Lightning-30B-A3B-abliterated-cyber-GLP-51-L1-51-a1.0`
(gated; 51 unit directions, layers 1–51, layer 0 intentionally absent,
width 2688, `alpha_default` 1.0). Measured on vLLM v0.27.1, one H100,
greedy, alpha = 1.0:

| suite          | stock  | steered |
|----------------|--------|---------|
| refusal32      | 0/32   | 32/32   |
| cyber32        | 1/32   | 31/32   |
| benign32       | 32/32  | 32/32 (unchanged) |
| propaganda32   | 28/32  | 32/32   |

Termination is intact (clean EOS stops). `WEIGHTLESS_GLP_ALPHA` overrides
alpha (default 1.0, the vector's `alpha_default`);
`WEIGHTLESS_GLP_LAYERS` optionally restricts to a comma list of layer ids.

**MTP caveat:** the checkpoint carries one MTP nextn layer which is NOT
steered. Speculative decoding would let unsteered MTP draft tokens bypass
the edit, so the launcher refuses to combine `WEIGHTLESS_GLP` with
`SPECULATIVE_MODE=dspark` — set `SPECULATIVE_MODE=none`.

**GB10 image note:** `vllm/vllm-openai:v0.27.1` is multi-arch upstream but
is not pulled on the rig; the locally available `vllm/vllm-openai:v0.28.0`
(the live Inkling lane's image) contains the same nemotron_h anchors and is
what the DGX deployment should use.

```sh
# .env.nemotron35 additions for the steered lane:
SPECULATIVE_MODE=none
NEMOTRON_IMAGE=vllm/vllm-openai:v0.28.0
WEIGHTLESS_GLP=/home/msuiche/.cache/huggingface/hub/models--msuiche--Nemotron-3.5-Lightning-30B-A3B-abliterated-cyber-GLP-51-L1-51-a1.0/snapshots/<rev>/Nemotron-3.5-Lightning-30B-A3B-abliterated-GLP-51-L1-51-a1.0.gguf
```

## Configure and launch

Use an available Spark. The running two-node Inkling deployment occupies both
GPUs; this recipe does not stop it, change the router, or modify client defaults.

```sh
cd recipe/nemotron35
cp .env.nemotron35.example .env.nemotron35
# Edit the image, cache, port, and context for this host.
bash serve-nemotron35.sh --dry-run
bash serve-nemotron35.sh
docker logs -f nemotron35
```

Docker needs NVIDIA GPU support. The image downloads the public target and draft
into `HF_CACHE` on first boot. Set `MODEL_REVISION` to a reviewed Hugging Face
commit for reproducible target weights; the example pins the revision checked
on 2026-09-05. DSpark's
draft follows its repository default. Set `SPECULATIVE_MODE=none` to evaluate
the target alone. Existing named containers are never removed by the launcher.

Tool calls use NVIDIA's `qwen3_coder` parser and `--enable-auto-tool-choice`;
reasoning uses `nemotron_v3`. The endpoint is `http://<spark-host>:8083/v1` and
its model ID is `nemotron35-lightning-nvfp4`.

## Verify before adding clients

```sh
curl --fail http://localhost:8083/health
cd ../..
export WEIGHTLESS_BASE_URL=http://localhost:8083/v1
export WEIGHTLESS_MODEL=nemotron35-lightning-nvfp4
bash tests/smoke/01-endpoint.sh
bash tests/smoke/02-chat.sh
bash tests/smoke/03-tool-call.sh
```

These check model discovery, actual text generation, and a structured tool call.
After configuring omp for this endpoint, run `tests/smoke/04-omp-headless.sh` with
`WEIGHTLESS_OMP_MODEL` set to its provider/model ID. Test a fresh Hermes session
with file creation/readback too. Read the live `/v1/models` limit before setting
client context. Increasing `MAX_MODEL_LEN` requires a fresh launch and renewed
long-prompt tests; an upstream 1M report is not a local validation result.

Local launcher regression checks, without Docker or a GPU:

```sh
python3 recipe/nemotron35/test_launcher.py     # launcher incl. GLP gating
python3 tests/test_nemotron35_hotfix.py        # hotfix anchors + GGUF gates
```
