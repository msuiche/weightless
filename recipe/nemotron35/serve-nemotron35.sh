#!/usr/bin/env bash
# Stock Nemotron 3.5 Lightning on one DGX Spark, following NVIDIA's recipe.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
CONFIG="${NEMOTRON_ENV_FILE:-$HERE/.env.nemotron35}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Copy $HERE/.env.nemotron35.example to $CONFIG and configure it first." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CONFIG"
: "${NEMOTRON_IMAGE:?}" "${MODEL_ID:?}" "${HF_CACHE:?}" "${CONTAINER_NAME:?}"
: "${VLLM_PORT:?}" "${SERVED_MODEL_NAME:?}" "${MAX_MODEL_LEN:?}"
: "${GPU_MEMORY_UTILIZATION:?}" "${MAX_NUM_SEQS:?}"

args=(run -d --restart unless-stopped --name "$CONTAINER_NAME"
  --gpus all --ipc=host --network host
  --mount "type=bind,src=$HF_CACHE,dst=/root/.cache/huggingface"
  --entrypoint vllm "$NEMOTRON_IMAGE" serve "$MODEL_ID"
  --revision "${MODEL_REVISION:-main}"
  --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port "$VLLM_PORT"
  --tensor-parallel-size 1 --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --moe-backend marlin --kv-cache-dtype fp8 --enable-prefix-caching
  --mamba-backend flashinfer --mamba-cache-mode align
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder
  --enable-auto-tool-choice)
case "${SPECULATIVE_MODE:-dspark}" in
  dspark)
    : "${DRAFT_MODEL_ID:?}"
    args+=(--speculative_config.model "$DRAFT_MODEL_ID"
      --speculative_config.num_speculative_tokens "${NUM_SPECULATIVE_TOKENS:-3}")
    ;;
  none) ;;
  *) echo "SPECULATIVE_MODE must be dspark or none" >&2; exit 2 ;;
esac
if [[ ${1:-} == --dry-run ]]; then
  printf '%q ' docker "${args[@]}"
  printf '\n'
  exit 0
fi
# Never replace a serving container implicitly.
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container $CONTAINER_NAME already exists. Stop/remove it explicitly or choose another name." >&2
  exit 1
fi
mkdir -p "$HF_CACHE"
docker "${args[@]}"
echo "Container started; generation is not yet verified."
echo "Follow docker logs -f $CONTAINER_NAME, then run the README's endpoint tests."
