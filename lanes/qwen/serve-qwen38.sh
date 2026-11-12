#!/usr/bin/env bash
# Qwen TP=1 lane: Qwen3.8-27B-NVFP4 on a single DGX Spark (Profile A:
# c=8 @ 256K, MTP-3), with refusal steering.
#
# Two steering modes (STEER_MODE in .env.qwen):
#
#   gguf  (default) — projective control vector via the boot hotfix: the
#          hotfix patches qwen3_next.py + qwen3_5.py inside the container
#          BEFORE vllm serve starts, and the `&&` makes that fail-closed —
#          a boot asked for steering that cannot apply it never serves
#          unsteered.
#   lora  — rank-1 LoRA on mlp.down_proj, loaded by stock vLLM
#          (--enable-lora). No patch, no anchor fragility. Served as the
#          "qwen-abliterated" module next to the stock base model. Needs the
#          peft layout (adapter_config.json + adapter_model.safetensors) —
#          see README.
#
# Both measured on NVFP4 hardware (2026-08-22): stock 4/32, gguf 24/32,
# lora 24/32 refusal32 delivery.
#
# Usage: cp .env.qwen.example .env.qwen, fill it in, then bash serve-qwen38.sh
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .env.qwen

STEER_MODE="${STEER_MODE:-gguf}"

DOCKER_EXTRA=()
ENTRY='exec vllm serve /models/Qwen3.8-27B-NVFP4'

if [ "$STEER_MODE" = "gguf" ]; then
  HOTFIX="$(cd ../../patches && pwd)/hotfix-qwen38-steering-projective.py"
  DOCKER_EXTRA+=(-v "$HOTFIX":/patches/hotfix-qwen38-steering-projective.py:ro)
  if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
    DOCKER_EXTRA+=(-e "WEIGHTLESS_STEER_PATH=$WEIGHTLESS_STEER_PATH"
                   -e "WEIGHTLESS_STEER_ALPHA=${WEIGHTLESS_STEER_ALPHA:-1.0}")
    [ -n "${WEIGHTLESS_STEER_LAYERS:-}" ] && DOCKER_EXTRA+=(-e "WEIGHTLESS_STEER_LAYERS=$WEIGHTLESS_STEER_LAYERS")
  fi
  [ -n "${WEIGHTLESS_STEERING_MODEL_PY:-}" ] && DOCKER_EXTRA+=(-e "WEIGHTLESS_STEERING_MODEL_PY=$WEIGHTLESS_STEERING_MODEL_PY")
  ENTRY='python3 /patches/hotfix-qwen38-steering-projective.py && '"$ENTRY"
elif [ "$STEER_MODE" = "lora" ]; then
  ENTRY="$ENTRY"' --enable-lora --lora_modules qwen-abliterated='"${QWEN_LORA_DIR:-/models/lora/qwen-abliterated}"
else
  echo "STEER_MODE must be gguf or lora, got '$STEER_MODE'" >&2
  exit 1
fi

docker rm -f qwen38 2>/dev/null || true
docker run -d --restart unless-stopped --name qwen38 --gpus all --ipc=host --network host \
  -v "$MODELS":/models \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  "${DOCKER_EXTRA[@]}" \
  --entrypoint bash \
  "$QWEN_IMAGE" \
  -c "$ENTRY"' \
        --served-model-name '"$SERVED_MODEL_NAME"' \
        --host 0.0.0.0 --port '"$VLLM_PORT"' \
        --max-model-len '"$MAX_MODEL_LEN"' \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization '"$GPU_MEMORY_UTILIZATION"' \
        --enable-flashinfer-autotune \
        --enable-auto-tool-choice --tool-call-parser qwen3_xml \
        --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":'"$MTP_NUM_TOKENS"'}'"'"

echo "qwen38 up: http://localhost:${VLLM_PORT}/v1 (model id ${SERVED_MODEL_NAME})"
if [ "$STEER_MODE" = "gguf" ]; then
  echo "Confirm steering in the boot log: 'weightless GLP steering active ... layers=49'"
  echo "(layers=1 means the per-layer-loop regression; layers=0 means unsteered.)"
else
  echo "Steered model id: qwen-abliterated (base ${SERVED_MODEL_NAME} serves stock)"
fi

# --- first-load warmup (drowzeys "first stuck prompt" gotcha) ----------------
# Compile the LARGE-prefill path before any client connects. A cold serve that
# takes a big first prompt can stall or return a garbled first reply; a tiny
# "hello" does NOT cover this — it must be a large prompt.
echo "Warming the large-prefill path (first-load fix)..."
python3 - <<PY
import json, time, urllib.request
base = "http://localhost:${VLLM_PORT}"
for _ in range(180):
    try: urllib.request.urlopen(base + "/v1/models", timeout=3); break
    except Exception: time.sleep(5)
prompt = ("Unified memory bandwidth bounds decode throughput on edge accelerators today. " * 2200) + "\nReply with: OK"
body = json.dumps({"model": "${SERVED_MODEL_NAME}", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
try:
    urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=300).read()
    print("  warmup ok (~26K-token prefill compiled) — first client prompt will be fast")
except Exception as e:
    print("  WARN warmup request failed (serve may still be compiling):", e)
PY
