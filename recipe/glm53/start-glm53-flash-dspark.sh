#!/usr/bin/env bash
# GLM-5.3-Flash FP8 lane: TP=4 across 4x DGX Spark (head + 3 workers over
# RoCE) on the day-0 image vllm/vllm-openai:glm53-flash, with GLP-44
# projective refusal steering.
#
# THIS LANE NEEDS 4 NODES: ~306 GB of FP8 weights do not fit 2x Spark
# (~77 GB/node at TP=4). Flow mirrors the other multi-node lanes: run this on
# the HEAD node; it syncs env + steering hotfix to every worker, boots the
# worker ranks first (headless), then the head rank (OpenAI API on :8080).
# The hotfix runs inside ALL FOUR containers before vllm serve and the `&&`
# makes that fail-closed — a boot asked for steering that cannot apply it
# never serves unsteered.
#
# Serve flags follow the official vLLM recipe (recipes.vllm.ai
# /zai-org/GLM-5.3-Flash): fp8 KV, MTP k=5, glm47 tool parser, glm45
# reasoning parser. See README.md for the image/source drift caveat.
#
# Usage: cp .env.glm53.example .env.glm53, fill in the <...> values, then
#        bash start-glm53-flash-dspark.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.glm53}"
CONTAINER="${CONTAINER_NAME:-glm53}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-160}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy .env.glm53.example to .env.glm53 and edit node-specific values." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${WORKER2_HOST:?WORKER2_HOST must be set in $ENV_FILE}"
: "${WORKER3_HOST:?WORKER3_HOST must be set in $ENV_FILE}"
: "${HF_CACHE:?HF_CACHE must be set in $ENV_FILE}"
: "${GLM53_IMAGE:?GLM53_IMAGE must be set in $ENV_FILE}"
: "${MODEL:?MODEL must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"

MASTER_PORT="${MASTER_PORT:-25020}"
WORKER_DIR="${WORKER_DIR:-$HOME/dspark-glm53}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm53-flash}"
VLLM_PORT="${VLLM_PORT:-8080}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-5}"
VLLM_HOST_IP="${VLLM_HOST_IP:-$MASTER_ADDR}"
# rank 1..3, in order
WORKERS=("$WORKER_HOST" "$WORKER2_HOST" "$WORKER3_HOST")
# On the node the deploy layout puts the hotfix in ./patches next to this
# script; in a bare repo checkout it lives at ../../patches.
HOTFIX="${WEIGHTLESS_STEERING_HOTFIX:-$SCRIPT_DIR/patches/hotfix-glm53-steering-projective.py}"
if [ ! -f "$HOTFIX" ]; then
  HOTFIX="$(cd "$SCRIPT_DIR/../../patches" && pwd)/hotfix-glm53-steering-projective.py"
fi
[ -f "$HOTFIX" ] || { echo "Missing steering hotfix (looked in $SCRIPT_DIR/patches and the repo)." >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}
need_cmd docker
need_cmd ssh
need_cmd scp
need_cmd curl

# --- steering preflight -----------------------------------------------------
# The vector must exist at the root of the HF cache on ALL FOUR nodes (each
# rank reads WEIGHTLESS_STEER_PATH inside its own container).
STEER_ENV=""
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  fname="$(basename "$WEIGHTLESS_STEER_PATH")"
  [ -f "$HF_CACHE/$fname" ] || {
    echo "Steering requested but $HF_CACHE/$fname is missing on the head." >&2
    echo "  hf download msuiche/GLM-5.3-Flash-abliterated-GLP-44 --include '*.gguf' --local-dir $HF_CACHE" >&2
    exit 1
  }
  for w in "${WORKERS[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "test -f '$WORKER_HF_CACHE/$fname'" || {
      echo "Steering requested but $WORKER_HF_CACHE/$fname is missing on worker $w." >&2
      echo "  run the same hf download there (the wizard can sync it)." >&2
      exit 1
    }
  done
  STEER_ENV="-e WEIGHTLESS_STEER_PATH=$WEIGHTLESS_STEER_PATH -e WEIGHTLESS_STEER_ALPHA=${WEIGHTLESS_STEER_ALPHA:-2.0}"
  [ -n "${WEIGHTLESS_STEER_LAYERS:-}" ] && STEER_ENV="$STEER_ENV -e WEIGHTLESS_STEER_LAYERS=$WEIGHTLESS_STEER_LAYERS"
  [ -n "${WEIGHTLESS_STEERING_MODEL_PY:-}" ] && STEER_ENV="$STEER_ENV -e WEIGHTLESS_STEERING_MODEL_PY=$WEIGHTLESS_STEERING_MODEL_PY"
fi

# --- refuse to clobber a running stack --------------------------------------
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "$CONTAINER already running on the head. docker rm -f $CONTAINER (and on the workers) for a cold start." >&2
  exit 3
fi
for w in "${WORKERS[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "true" >/dev/null || {
    echo "Cannot reach worker with passwordless SSH: $w" >&2
    exit 1
  }
  if ssh "$w" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'"; then
    echo "$CONTAINER already running on worker $w. docker rm -f $CONTAINER there for a cold start." >&2
    exit 3
  fi
done
docker image inspect "$GLM53_IMAGE" >/dev/null || {
  echo "Missing local Docker image $GLM53_IMAGE — docker pull it first." >&2
  exit 1
}
for w in "${WORKERS[@]}"; do
  ssh "$w" "docker image inspect '$GLM53_IMAGE' >/dev/null" || {
    echo "Missing worker Docker image $GLM53_IMAGE on $w — docker pull it there first." >&2
    exit 1
  }
done

# --- docker run command (per node) ------------------------------------------
# $1 rank, $2 api|headless, $3 host HF cache, $4 host hotfix path, $5 host IP.
# Env values are operator-controlled and must not contain spaces.
build_cmd() {
  local rank="$1" mode="$2" hf="$3" hotfix="$4" hostip="$5"
  local ep_args="--headless"
  if [ "$mode" = "api" ]; then
    ep_args="--host 0.0.0.0 --port $VLLM_PORT"
  fi
  cat <<EOF
docker run -d --restart unless-stopped --name $CONTAINER \
  --gpus all --ipc=host --network host --shm-size 64gb \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband \
  -v $hf:/cache/huggingface \
  -v $hotfix:/patches/hotfix-glm53-steering-projective.py:ro \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
  -e VLLM_HOST_IP=$hostip \
  -e NCCL_NET=${NCCL_NET:-IB} \
  -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} \
  -e NCCL_IB_HCA=$NCCL_IB_HCA \
  -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  -e TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME} \
  -e GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
  -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1} \
  -e NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0} \
  -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
  $STEER_ENV \
  --entrypoint bash \
  $GLM53_IMAGE \
  -c 'python3 /patches/hotfix-glm53-steering-projective.py && exec vllm serve $MODEL \
        --served-model-name $SERVED_MODEL_NAME \
        --tensor-parallel-size 4 \
        --nnodes 4 --node-rank $rank \
        --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8 \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":$MTP_NUM_TOKENS}'"'"' \
        --tool-call-parser glm47 \
        --reasoning-parser glm45 \
        --enable-auto-tool-choice \
        $ep_args'
EOF
}

echo "Resolved GLM-5.3-Flash profile:"
echo "  image: $GLM53_IMAGE"
echo "  model: $MODEL (~306 GB FP8, TP=4 = ~77 GB/node)"
echo "  served model: $SERVED_MODEL_NAME"
echo "  head: $MASTER_ADDR (api :$VLLM_PORT)  workers: ${WORKERS[*]}"
echo "  max model len: $MAX_MODEL_LEN, gpu util: $GPU_MEMORY_UTILIZATION, mtp k: $MTP_NUM_TOKENS"
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  echo "  steering: $WEIGHTLESS_STEER_PATH (α=${WEIGHTLESS_STEER_ALPHA:-2.0}, layers=${WEIGHTLESS_STEER_LAYERS:-all in file})"
  echo "  WARNING: α>=2.5 garbles this model — 2.0 is the calibrated peak, do not raise it"
else
  echo "  steering: off (WEIGHTLESS_STEER_PATH empty)"
fi

# --- workers first ------------------------------------------------------------
rank=1
for w in "${WORKERS[@]}"; do
  echo "Syncing env + hotfix to ${w}:${WORKER_DIR}"
  ssh "$w" "mkdir -p '$WORKER_DIR/patches'"
  scp "$ENV_FILE" "$w:$WORKER_DIR/.env.glm53" >/dev/null
  scp "$HOTFIX" "$w:$WORKER_DIR/patches/hotfix-glm53-steering-projective.py" >/dev/null

  echo "Starting worker rank $rank (headless) on $w..."
  ssh "$w" "docker rm -f $CONTAINER >/dev/null 2>&1 || true"
  ssh "$w" "$(build_cmd "$rank" headless "$WORKER_HF_CACHE" "$WORKER_DIR/patches/hotfix-glm53-steering-projective.py" "$w")"
  rank=$((rank + 1))
done

echo "Starting head rank 0 (API :$VLLM_PORT)..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
bash -c "$(build_cmd 0 api "$HF_CACHE" "$HOTFIX" "$VLLM_HOST_IP")"

echo "Waiting for the API (306 GB of FP8 weights stream from all four caches; this takes a while)..."
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "GLM-5.3-Flash is running: http://$MASTER_ADDR:$VLLM_PORT/v1 (model id $SERVED_MODEL_NAME)"
    if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
      echo "Confirm steering in the boot log: 'weightless GLP steering active ... layers=44'"
      echo "(layers=1 means the per-layer-loop regression; layers=0 means unsteered.)"
    fi
    exit 0
  fi
  sleep "$WAIT_SECONDS"
done

echo "Timed out waiting for the API. Recent head logs:" >&2
docker logs --tail 120 "$CONTAINER" >&2 || true
for w in "${WORKERS[@]}"; do
  echo "Recent worker logs ($w):" >&2
  ssh "$w" "docker logs --tail 120 '$CONTAINER'" >&2 || true
done
exit 1
