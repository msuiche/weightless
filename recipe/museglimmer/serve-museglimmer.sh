#!/usr/bin/env bash
# Muse-Glimmer-30B NVFP4 on one DGX Spark, following NVIDIA's model card.
# Stock vLLM 0.28.0 serves the muse_glimmer arch natively (multimodal
# wrapper, ATEM tool parser, reasoning parser) and auto-detects the DFlash
# block-diffusion drafter from its MuseGlimmerAssistantModel architecture.
# No steering: no GLP vector exists for this arch yet.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -gt 1 || ( $# -eq 1 && $1 != --dry-run ) ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi
CONFIG="${MUSE_ENV_FILE:-$HERE/.env.museglimmer}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Copy $HERE/.env.museglimmer.example to $CONFIG and configure it first." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CONFIG"
: "${MUSE_IMAGE:?}" "${MODEL_ID:?}" "${HF_CACHE:?}" "${CONTAINER_NAME:?}"
: "${VLLM_PORT:?}" "${SERVED_MODEL_NAME:?}" "${MAX_MODEL_LEN:?}"
: "${GPU_MEMORY_UTILIZATION:?}" "${MAX_NUM_SEQS:?}"

args=(run -d --restart unless-stopped --name "$CONTAINER_NAME"
  --gpus all --ipc=host --network host
  --mount "type=bind,src=$HF_CACHE,dst=/root/.cache/huggingface"
  # The wizard prefetches with hf download --cache-dir $HF_CACHE, which
  # lays snapshots at the cache ROOT (models--org--repo/), not under hub/.
  # Point HF_HUB_CACHE at the mount so the container reuses that prefetch
  # instead of re-downloading 25 GB into hub/ on first boot.
  -e HF_HUB_CACHE=/root/.cache/huggingface)
# Card-tested serve flags (nvidia/Muse-Glimmer-30B-NVFP4, vLLM 0.28.0).
# --kv-cache-dtype auto: the quant recipe leaves the KV cache unquantized.
# The card also passes --mamba-cache-mode align; harmless here (dense
# attention arch, no mamba state) and kept to match the tested command.
serve_args=(serve "$MODEL_ID"
  --revision "${MODEL_REVISION:-main}"
  --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port "$VLLM_PORT"
  --tensor-parallel-size 1 --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype auto --enable-prefix-caching --enable-chunked-prefill
  --mamba-cache-mode align
  --reasoning-parser muse_glimmer --tool-call-parser muse_glimmer
  --enable-auto-tool-choice)
case "${SPECULATIVE_MODE:-dflash}" in
  dflash)
    : "${DRAFTER_MODEL_ID:?}"
    serve_args+=(--speculative-config
      "{\"method\":\"dflash\",\"model\":\"$DRAFTER_MODEL_ID\",\"revision\":\"${DRAFTER_REVISION:-main}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS:-16}}")
    ;;
  none) ;;
  *) echo "SPECULATIVE_MODE must be dflash or none" >&2; exit 2 ;;
esac
args+=(--entrypoint vllm "$MUSE_IMAGE" "${serve_args[@]}")
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
