#!/usr/bin/env bash
# Inkling-Small NVFP4 TP=2 across 2x DGX Spark (head + worker over RoCE) on
# stock vLLM v0.28.0 + the SM121 rel-attention fallback hotfix
# (weightless/patches/hotfix-inkling-sm121-relattn.py, applied offline to
# files/fa4_rel_attention-sm121.py and bind-mounted over the container's
# vllm/models/inkling/nvidia/ops/fa4_rel_attention.py).
#
# BRING-UP LANE: steering OFF (stock model.py, no WEIGHTLESS_STEER_* env),
# no MTP speculative config,. Decode routes to the ROCm-lane
# Triton split-KV kernel; prefill/extend to the torch SDPA fallback.
#
# Usage: bash start-inkling-sm121.sh            # real weights
#        LOAD_FORMAT=dummy bash start-inkling-sm121.sh   # dummy weights
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.inkling}"
CONTAINER="${CONTAINER_NAME:-inkling-sm121}"

[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
# CLI/env overrides win over the sourced file (the file assigns unconditionally).
[ -n "${GPU_MEMORY_UTILIZATION_OVERRIDE:-}" ] && GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION_OVERRIDE"

: "${MASTER_ADDR:?}"; : "${WORKER_HOST:?}"; : "${HF_CACHE:?}"
: "${INKLING_IMAGE:?}"; : "${MODEL:?}"
: "${NCCL_IB_HCA:?}"; : "${NCCL_SOCKET_IFNAME:?}"

MASTER_PORT="${MASTER_PORT:-25020}"
VLLM_PORT="${VLLM_PORT:-8082}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-inkling-small-nvfp4}"
FA4_PATCHED_PY="${FA4_PATCHED_PY:-$SCRIPT_DIR/files/fa4_rel_attention-sm121.py}"
LOAD_FORMAT="${LOAD_FORMAT:-auto}"

# --- hardware preflight (from the qwen38fn wedge saga) -----------------------
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

# --- patch staging preflight ---------------------------------------------------
[ -f "$FA4_PATCHED_PY" ] || { echo "Missing $FA4_PATCHED_PY" >&2; exit 1; }
grep -q "sm121-relattn-hotfix" "$FA4_PATCHED_PY" || {
  echo "$FA4_PATCHED_PY lacks the sm121 hotfix marker — refusing to boot stock." >&2; exit 1; }
ssh -o BatchMode=yes "$WORKER_HOST" "grep -q sm121-relattn-hotfix '$FA4_PATCHED_PY'" || {
  echo "Missing/unpatched $FA4_PATCHED_PY on the worker." >&2; exit 1; }

# --- refuse to clobber a running stack ---------------------------------------
docker ps --format '{{.Names}}' | grep -qx "$CONTAINER" && {
  echo "$CONTAINER already running on the head — docker rm -f it first." >&2; exit 3; }
ssh -o BatchMode=yes "$WORKER_HOST" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'" && {
  echo "$CONTAINER already running on the worker — docker rm -f it there." >&2; exit 3; }

echo "Inkling-Small TP=2 SM121-fallback: head $MASTER_ADDR (api :$VLLM_PORT), worker $WORKER_HOST, ctx $MAX_MODEL_LEN, util $GPU_MEMORY_UTILIZATION, load-format $LOAD_FORMAT"

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
  -v $FA4_PATCHED_PY:/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/ops/fa4_rel_attention.py:ro \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e INKLING_REL_ATTN_BACKEND=${INKLING_REL_ATTN_BACKEND:-triton} \
  \
  -e VLLM_HOST_IP=$hostip \
  -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} -e NCCL_IB_HCA=$NCCL_IB_HCA -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
  -e NCCL_IB_AUTO_DETECT=0 \
  -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME -e TP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  -e GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME -e NCCL_DEBUG=WARN -e LAMPORT_RS_SCONV=${LAMPORT_RS_SCONV:-1} \
  -e VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO} \
  -e TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0} \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800} \
  --entrypoint bash $INKLING_IMAGE \
  -c 'exec vllm serve $MODEL \
        --served-model-name $SERVED_MODEL_NAME \
        --tensor-parallel-size 2 --nnodes 2 --node-rank $rank \
        --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
        --distributed-executor-backend mp \
        --tokenizer-mode inkling --reasoning-parser inkling \
        --trust-remote-code --load-format $LOAD_FORMAT \
        ${AUTOTUNE_FLAG:---no-enable-flashinfer-autotune} \
        --max-num-seqs ${MAX_NUM_SEQS:-4} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-2048} \
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
