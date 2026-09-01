#!/usr/bin/env bash
# Inkling-Small NVFP4 TP=2 across 2x DGX Spark (head + worker over RoCE) on
# stock vLLM v0.28.0 (day-0 inkling support), GLP-41 projective steering at
# the CALIBRATED alpha=0.25 (DO NOT raise — alpha>=0.5 garbles this model).
#
# STRUCTURE-VALIDATED ONLY (2026-09-02): written from the measured qwen38fn
# lane + the Inkling Modal validation (4xH100, vLLM 0.28.0) but NOT yet booted
# on the Sparks. The wedge-proofing is inherited from the qwen38fn saga
# (preflight, drop_caches, --restart no, log caps).
#
# Usage: cp .env.inkling.example .env.inkling, fill in the <...> values, then
#        bash start-inkling-dspark.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.inkling}"
CONTAINER="${CONTAINER_NAME:-inkling}"

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE — copy .env.inkling.example and fill it." >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${MASTER_ADDR:?}"; : "${WORKER_HOST:?}"; : "${HF_CACHE:?}"
: "${INKLING_IMAGE:?}"; : "${MODEL:?}"
: "${NCCL_IB_HCA:?}"; : "${NCCL_SOCKET_IFNAME:?}"

MASTER_PORT="${MASTER_PORT:-25020}"
VLLM_PORT="${VLLM_PORT:-8082}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.835}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-inkling-small-nvfp4}"
STEERED_MODEL_PY="${STEERED_MODEL_PY:-$SCRIPT_DIR/files/inkling-model-steered.py}"

# --- hardware preflight (from the qwen38fn wedge saga, 2026-09-01) -----------
need_free_gib() {
  if [ -z "$1" ]; then
    awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
      "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo"
  fi
}
TOTAL_GIB=121
NEED_GIB=$(awk -v u="$GPU_MEMORY_UTILIZATION" -v t="$TOTAL_GIB" 'BEGIN{print int(u*t)}')
for h in "" "$WORKER_HOST"; do
  free_gib=$(need_free_gib "$h") || { echo "preflight: cannot read memory on '${h:-head}'" >&2; exit 1; }
  [ "$free_gib" -ge "$NEED_GIB" ] || {
    echo "preflight: ${h:-head} has ${free_gib} GiB free < ${NEED_GIB} GiB requested" >&2; exit 1; }
  if [ -z "$h" ]; then
    zombies=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  else
    zombies=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
      "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l")
  fi
  [ "${zombies:-99}" -eq 0 ] || {
    echo "preflight: ${zombies} GPU process(es) alive on ${h:-head} — kill/reboot first." >&2; exit 1; }
done

# drop page caches on both nodes (mandatory on GB10 unified memory)
sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
ssh -o BatchMode=yes "$WORKER_HOST" "sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null"

# --- steering preflight -------------------------------------------------------
[ -f "$STEERED_MODEL_PY" ] || {
  echo "Missing $STEERED_MODEL_PY — build it once: extract the image's
  vllm/models/inkling/nvidia/model.py, then apply the hotfix offline:
  WEIGHTLESS_STEERING_MODEL_PY=<copy> python3 ../../patches/hotfix-inkling-steering-projective.py" >&2
  exit 1
}
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  fname="$(basename "$WEIGHTLESS_STEER_PATH")"
  [ -f "$HF_CACHE/$fname" ] || { echo "Missing $HF_CACHE/$fname on the head." >&2; exit 1; }
  ssh -o BatchMode=yes "$WORKER_HOST" "test -f '$HF_CACHE/$fname'" || {
    echo "Missing $HF_CACHE/$fname on the worker." >&2; exit 1; }
  STEER_ENV="-e WEIGHTLESS_STEER_PATH=/cache/huggingface/$fname -e WEIGHTLESS_STEER_ALPHA=${WEIGHTLESS_STEER_ALPHA:-0.25}"
else
  STEER_ENV=""
fi

# --- refuse to clobber a running stack ---------------------------------------
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" && {
  echo "$CONTAINER already running on the head — docker rm -f it (and on the worker) for a cold start." >&2; exit 3; }
ssh -o BatchMode=yes "$WORKER_HOST" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'" && {
  echo "$CONTAINER already running on the worker — docker rm -f it there." >&2; exit 3; }

echo "Inkling-Small TP=2: head $MASTER_ADDR (api :$VLLM_PORT), worker $WORKER_HOST, ctx $MAX_MODEL_LEN, util $GPU_MEMORY_UTILIZATION"
[ -n "$STEER_ENV" ] && echo "  steering: $fname alpha=${WEIGHTLESS_STEER_ALPHA:-0.25}"

# --- launch: worker (rank 1) first, then head (rank 0) ------------------------
build_cmd() {  # $1 rank, $2 api|headless, $3 host IP
  local rank="$1" mode="$2" hostip="$3"
  local ep="--headless"
  [ "$mode" = "api" ] && ep="--host 0.0.0.0 --port $VLLM_PORT"
  cat <<EOF
docker run -d --restart no --name $CONTAINER \
  --log-opt max-size=50m --log-opt max-file=2 \
  --gpus all --ipc=host --network host --shm-size 64gb \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add SYS_NICE \
  --device /dev/infiniband \
  -v $HF_CACHE:/cache/huggingface \
  -v $STEERED_MODEL_PY:/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/model.py:ro \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_HOST_IP=$hostip \
  -e NCCL_IB_HCA=$NCCL_IB_HCA -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
  -e NCCL_IB_AUTO_DETECT=0 \
  -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME -e TP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  -e GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME -e NCCL_DEBUG=WARN \
  $STEER_ENV \
  --entrypoint bash $INKLING_IMAGE \
  -c 'exec vllm serve $MODEL \
        --served-model-name $SERVED_MODEL_NAME \
        --tensor-parallel-size 2 --nnodes 2 --node-rank $rank \
        --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
        --distributed-executor-backend mp \
        --tokenizer-mode inkling --reasoning-parser inkling \
        --trust-remote-code \
        --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":3}'"'"' \
        --max-num-seqs ${MAX_NUM_SEQS:-8} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-8192} \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        $ep'
EOF
}

echo "Starting worker rank 1 on $WORKER_HOST..."
ssh -o BatchMode=yes "$WORKER_HOST" "$(build_cmd 1 headless "$WORKER_HOST")"
echo "Starting head rank 0 (API :$VLLM_PORT)..."
eval "$(build_cmd 0 api "$MASTER_ADDR")"
echo "Waiting for the API (load takes ~10 min; watch: docker logs -f $CONTAINER)"
for _ in $(seq 1 120); do
  curl -sf -m 5 "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1 && {
    echo "API UP on :$VLLM_PORT"; exit 0; }
  sleep 15
done
echo "Timed out waiting for the API — check docker logs $CONTAINER" >&2
exit 1
