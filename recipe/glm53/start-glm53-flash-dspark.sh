#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 lane: TP=4 across 4x DGX Spark (head + 3 workers over
# RoCE), with GLP-44 projective refusal steering.
#
# THIS LANE NEEDS 4 NODES. The serve config is the hardware-validated
# tonyd2wild TP4 flagship (see README.md for the full citation list): NVFP4
# quant (~50 GiB/rank), the sm121-v8 patch stack on the day-0 image, fp8 KV,
# MTP k=4, block 2304, 24 GiB/rank pinned KV.
#
# Boot ritual (each rule cost them a boot — README.md has the receipts):
#   * the STOCK vendor image dies five different ways on GB10 — this script
#     refuses any image tag without "sm121" in it;
#   * the image ID must be IDENTICAL on all four nodes (checked below);
#   * cache-flush ritual on every node before boot (drop_caches; their
#     cache_flusher.sh sidecar is recommended for the weight read);
#   * workers first (~20 s apart), head LAST;
#   * never relaunch a rank while others are up — this script refuses to
#     start if $CONTAINER exists on ANY node (tear down all, then boot).
#
# The GLP-44 steering hotfix runs inside ALL FOUR containers before
# vllm serve and the `&&` makes that fail-closed — a boot asked for steering
# that cannot apply it never serves unsteered.
#
# Usage: cp .env.glm53.example .env.glm53, fill in the <...> values, then
#        bash start-glm53-flash-dspark.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.glm53}"
CONTAINER="${CONTAINER_NAME:-glm53}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-160}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"
RANK_STAGGER_SECONDS="${RANK_STAGGER_SECONDS:-20}"

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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-25769803776}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-4}"
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

# --- image gate: never boot the stock day-0 image ---------------------------
# It dies five different ways on GB10 (README.md); only the sm121 patch stack
# works. v9/InstantTensor is unstable multi-node — v8 is the ceiling.
case "$GLM53_IMAGE" in
  *sm121*) ;;
  *)
    echo "Refusing to boot '$GLM53_IMAGE': the stock glm53-flash image fails on GB10." >&2
    echo "Use the sm121-v8 patch stack (radixark/vllm-glm53-flash:sm121-v8 or a local" >&2
    echo "v1->v8 build — see README.md)." >&2
    exit 1
    ;;
esac

# --- preflight: stack down everywhere, image identical on all nodes ---------
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "$CONTAINER already running on the head. Tear down ALL ranks before relaunching any" >&2
  echo "(docker rm -f $CONTAINER on every node)." >&2
  exit 3
fi
HEAD_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$GLM53_IMAGE" 2>/dev/null)" || {
  echo "Missing local Docker image $GLM53_IMAGE — docker pull it first." >&2
  exit 1
}
for w in "${WORKERS[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "true" >/dev/null || {
    echo "Cannot reach worker with passwordless SSH: $w" >&2
    exit 1
  }
  if ssh "$w" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'"; then
    echo "$CONTAINER already running on worker $w. Tear down ALL ranks before relaunching any." >&2
    exit 3
  fi
  wid="$(ssh "$w" "docker image inspect --format '{{.Id}}' '$GLM53_IMAGE' 2>/dev/null")" || {
    echo "Missing worker Docker image $GLM53_IMAGE on $w — docker pull it there first." >&2
    exit 1
  }
  if [ "$wid" != "$HEAD_IMAGE_ID" ]; then
    echo "Image ID mismatch: head has $HEAD_IMAGE_ID, $w has $wid." >&2
    echo "Mystery boots have been traced to exactly this — sync the image to all four nodes." >&2
    exit 1
  fi
done
echo "image gate: $GLM53_IMAGE (${HEAD_IMAGE_ID}) identical on all 4 nodes"

# --- cache-flush ritual (GB10 unified memory) --------------------------------
# Best-effort and non-interactive (sudo -n fails cleanly without passwordless
# sudo — the warning tells you to run it by hand). Their cache_flusher.sh
# sidecar (see README.md) is additionally recommended on every node during
# the weight read.
flush_caches() {
  local target="$1" # empty = local
  if [ -z "$target" ]; then
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  else
    ssh -o BatchMode=yes "$target" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null
  fi
}
echo "cache-flush ritual (drop_caches on all 4 nodes; start cache_flusher.sh if you have it):"
flush_caches "" || echo "  WARN: head drop_caches failed — run 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches' by hand" >&2
for w in "${WORKERS[@]}"; do
  flush_caches "$w" || echo "  WARN: drop_caches failed on $w — run it there by hand" >&2
  pgrep_out="$(ssh -o BatchMode=yes "$w" "pgrep -fc cache_flusher 2>/dev/null || true")"
  [ "${pgrep_out:-0}" -ge 1 ] 2>/dev/null || \
    echo "  WARN: no cache_flusher running on $w (recommended during the weight read — see README.md)" >&2
done

# --- model + steering preflight ----------------------------------------------
MODEL_CACHE_DIR="models--${MODEL//\//--}"
[ -d "$HF_CACHE/$MODEL_CACHE_DIR" ] || {
  echo "Model snapshot $MODEL_CACHE_DIR not found in $HF_CACHE on the head." >&2
  echo "  hf download $MODEL --local-dir $HF_CACHE  (this lane serves the NVFP4 quant)" >&2
  exit 1
}
for w in "${WORKERS[@]}"; do
  ssh -o BatchMode=yes "$w" "test -d '$WORKER_HF_CACHE/$MODEL_CACHE_DIR'" || {
    echo "Model snapshot $MODEL_CACHE_DIR not found in $WORKER_HF_CACHE on $w." >&2
    exit 1
  }
done

# The vector must exist at the root of the HF cache on ALL FOUR nodes (each
# rank reads WEIGHTLESS_STEER_PATH inside its own container).
STEER_ENV=""
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  fname="$(basename "$WEIGHTLESS_STEER_PATH")"
  [ -f "$HF_CACHE/$fname" ] || {
    echo "Steering requested but $HF_CACHE/$fname is missing on the head." >&2
    echo "  hf download msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44 --include '*.gguf' --local-dir $HF_CACHE" >&2
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

# --- docker run command (per node) ------------------------------------------
# $1 rank, $2 api|headless, $3 host HF cache, $4 host hotfix path, $5 host IP.
# Env values are operator-controlled and must not contain spaces.
build_cmd() {
  local rank="$1" mode="$2" hf="$3" hotfix="$4" hostip="$5"
  local ep_args="--headless"
  if [ "$mode" = "api" ]; then
    ep_args="--host 0.0.0.0 --port $VLLM_PORT"
  fi
  local kv_args=""
  [ -n "$KV_CACHE_MEMORY" ] && kv_args="--kv-cache-memory $KV_CACHE_MEMORY"
  cat <<EOF
docker run -d --restart unless-stopped --name $CONTAINER \
  --gpus all --ipc=host --network host --shm-size 64gb \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK \
  --device /dev/infiniband \
  -v $hf:/cache/huggingface \
  -v $hotfix:/patches/hotfix-glm53-steering-projective.py:ro \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
  -e VLLM_HOST_IP=$hostip \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  $STEER_ENV \
  --entrypoint bash \
  $GLM53_IMAGE \
  -c 'CT=\$(ls /cache/huggingface/$MODEL_CACHE_DIR/snapshots/*/chat_template_mm.jinja 2>/dev/null | head -1); \
      CT_ARG=; [ -n "\$CT" ] && CT_ARG="--chat-template \$CT"; \
      python3 /patches/hotfix-glm53-steering-projective.py && exec vllm serve $MODEL \
        --served-model-name $SERVED_MODEL_NAME \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --nnodes 4 --node-rank $rank \
        --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8_e4m3 \
        $kv_args \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-seqs $MAX_NUM_SEQS \
        --block-size $BLOCK_SIZE \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --moe-backend marlin \
        --enforce-eager \
        --speculative-config '"'"'{"method":"mtp","num_speculative_tokens":$MTP_NUM_TOKENS}'"'"' \
        --tool-call-parser glm47 \
        --enable-auto-tool-choice \
        --reasoning-parser glm45 \
        \$CT_ARG \
        --default-chat-template-kwargs '"'"'{"enable_thinking": false}'"'"' \
        $ep_args'
EOF
}

echo "Resolved GLM-5.3-Flash profile:"
echo "  image: $GLM53_IMAGE"
echo "  model: $MODEL (NVFP4, ~50 GiB/rank at TP=4)"
echo "  served model: $SERVED_MODEL_NAME"
echo "  head: $MASTER_ADDR (api :$VLLM_PORT)  workers: ${WORKERS[*]}"
echo "  max model len: $MAX_MODEL_LEN, kv pinned: ${KV_CACHE_MEMORY:-vllm-suggested} bytes/rank, gmu: $GPU_MEMORY_UTILIZATION"
echo "  block size: $BLOCK_SIZE, mtp k: $MTP_NUM_TOKENS, kv: fp8_e4m3, eager, thinking off by default"
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  echo "  steering: $WEIGHTLESS_STEER_PATH (α=${WEIGHTLESS_STEER_ALPHA:-2.0}, layers=${WEIGHTLESS_STEER_LAYERS:-all in file})"
  echo "  WARNING: α>=2.5 garbles this model — 2.0 is the calibrated peak, do not raise it"
else
  echo "  steering: off (WEIGHTLESS_STEER_PATH empty)"
fi

# --- workers first, ~20 s apart; head LAST -----------------------------------
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
  [ "$rank" -le 3 ] && sleep "$RANK_STAGGER_SECONDS"
done

echo "Starting head rank 0 (API :$VLLM_PORT)..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
bash -c "$(build_cmd 0 api "$HF_CACHE" "$HOTFIX" "$VLLM_HOST_IP")"

echo "Waiting for the API (~12 min boot is normal: quarter weights per rank)..."
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
