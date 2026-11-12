#!/usr/bin/env bash
# Qwen3.8-Flash-Next NVFP4 lane: TP=2 across 2x DGX Spark (head + worker over
# RoCE) on the day-0 image vllm/vllm-openai:qwen38-flash-next, with GLP-47
# projective refusal steering.
#
# Flow mirrors the anemll lane: run this on the HEAD node; it syncs env +
# steering hotfix to the worker, boots the worker rank first (headless), then
# the head rank (OpenAI API on :8079). The hotfix runs inside BOTH containers
# before vllm serve and the `&&` makes that fail-closed — a boot asked for
# steering that cannot apply it never serves unsteered.
#
# Traps this script is written around (see README.md):
#   * --no-enable-prefix-caching is MANDATORY: prefix caching forces
#     mamba_cache_mode="align" on this arch and splits every prefill at a
#     block boundary, corrupting the steered stream (capture-lane run 3).
#   * VLLM_PLE_CPU_OFFLOAD=1 would keep the 51B N-gram table in host RAM, but
#     the arm64 day-0 image rejects it at nnodes=2 — the start script fails
#     closed if it is set. With it OFF, the PLE table lives in HBM (~25.5
#     GB/rank), so keep util <= 0.88 (64K ctx) or <= 0.80 (262K ctx).
#
# Usage: cp .env.qwen38fn.example .env.qwen38fn, fill in the <...> values,
#        then bash start-qwen38-flash-next-dspark.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.qwen38fn}"
CONTAINER="${CONTAINER_NAME:-qwen38fn}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-120}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy .env.qwen38fn.example to .env.qwen38fn and edit node-specific values." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${HF_CACHE:?HF_CACHE must be set in $ENV_FILE}"
: "${QWEN38FN_IMAGE:?QWEN38FN_IMAGE must be set in $ENV_FILE}"
: "${MODEL:?MODEL must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"

MASTER_PORT="${MASTER_PORT:-25010}"
WORKER_DIR="${WORKER_DIR:-$HOME/dspark-qwen38fn}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen38-flash-next-nvfp4}"
VLLM_PORT="${VLLM_PORT:-8079}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_HOST_IP="${VLLM_HOST_IP:-$MASTER_ADDR}"
WORKER_VLLM_HOST_IP="${WORKER_VLLM_HOST_IP:-$WORKER_HOST}"
# On the node the deploy layout puts the hotfix in ./patches next to this
# script; in a bare repo checkout it lives at ../../patches.
HOTFIX="${WEIGHTLESS_STEERING_HOTFIX:-$SCRIPT_DIR/patches/hotfix-qwen38fn-steering-projective.py}"
if [ ! -f "$HOTFIX" ]; then
  HOTFIX="$(cd "$SCRIPT_DIR/../../patches" && pwd)/hotfix-qwen38fn-steering-projective.py"
fi
[ -f "$HOTFIX" ] || { echo "Missing steering hotfix (looked in $SCRIPT_DIR/patches and the repo)." >&2; exit 1; }
# The day-0 image cannot load the RadixArk NVFP4 checkpoint's FP8-serialized
# PLE N-gram table without this patch (unknown param 'ngram_embedding.weight_
# scale'; found during Modal B200 validation, 2026-08-29). Same fail-closed
# chain as the hotfix.
PLE_PATCH="${PLE_FP8_PATCH:-$SCRIPT_DIR/patches/patch-qwen38fn-ple-fp8-nvfp4.py}"
if [ ! -f "$PLE_PATCH" ]; then
  PLE_PATCH="$(cd "$SCRIPT_DIR/../../patches" && pwd)/patch-qwen38fn-ple-fp8-nvfp4.py"
fi
[ -f "$PLE_PATCH" ] || { echo "Missing PLE FP8 patch (looked in $SCRIPT_DIR/patches and the repo)." >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}
need_cmd docker
need_cmd ssh
need_cmd scp
need_cmd curl

# --- hardware preflight (learned 2026-09-01, the hard way) -------------------
# (a) The arm64 day-0 image rejects VLLM_PLE_CPU_OFFLOAD=1 at nnodes=2
#     ("Unsupported settings: nnodes=2"). It works on the x86 build — that
#     asymmetry cost a crash-loop day. Fail BEFORE the 10-minute weight load.
if [ "${VLLM_PLE_CPU_OFFLOAD:-1}" = "1" ]; then
  echo "VLLM_PLE_CPU_OFFLOAD=1 is unsupported at nnodes=2 on this image." >&2
  echo "Set VLLM_PLE_CPU_OFFLOAD=0 in $ENV_FILE — and see (b): with the PLE" >&2
  echo "table in HBM you ALSO need util<=0.88 (64K ctx) or <=0.80 (262K ctx)." >&2
  exit 1
fi
# (b) Reproduce vLLM's own startup gate early: free memory must exceed
#     util x total on BOTH nodes. On GB10 unified memory, a fail here means
#     weights (~67.5 GB/rank) + PLE (~25.5 GB/rank when offloaded=0) + KV do
#     not fit the chosen util — lower GPU_MEMORY_UTILIZATION or MAX_MODEL_LEN.
need_free_gib() {  # $1 host (empty=local)
  local free_gib
  if [ -z "$1" ]; then
    free_gib=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  else
    free_gib=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" \
      "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo") || return 1
  fi
  echo "$free_gib"
}
TOTAL_GIB=121  # GB10 visible-to-CUDA memory
NEED_GIB=$(awk -v u="$GPU_MEMORY_UTILIZATION" -v t="$TOTAL_GIB" 'BEGIN{print int(u*t)}')
for h in "" "$WORKER_HOST"; do
  free_gib=$(need_free_gib "$h") || { echo "preflight: cannot read memory on '${h:-head}'" >&2; exit 1; }
  if [ "$free_gib" -lt "$NEED_GIB" ]; then
    echo "preflight: ${h:-head} has ${free_gib} GiB free < ${NEED_GIB} GiB requested" >&2
    echo "  (util $GPU_MEMORY_UTILIZATION x $TOTAL_GIB GiB). Lower util or free memory." >&2
    exit 1
  fi
done
# (c) Refuse to boot over zombie GPU processes. A crash-looped container's
#     CUDA memory survives docker rm -f; the next load stacks on top, the
#     unified pool hits 0%, and earlyoom cannot kill a CUDA-stuck process —
#     that is the exact sequence that wedged both nodes on 2026-09-01.
for h in "" "$WORKER_HOST"; do
  if [ -z "$h" ]; then
    zombies=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  else
    zombies=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
      "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l") || zombies=99
  fi
  if [ "${zombies:-99}" -gt 0 ]; then
    echo "preflight: ${zombies} GPU compute process(es) still alive on ${h:-head}." >&2
    echo "  kill them (or reboot) before booting — stacked loads wedge GB10." >&2
    exit 1
  fi
done

# --- steering preflight -----------------------------------------------------
# The vector must exist at the root of the HF cache on BOTH nodes (each rank
# reads WEIGHTLESS_STEER_PATH inside its own container).
STEER_ENV=""
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  fname="$(basename "$WEIGHTLESS_STEER_PATH")"
  [ -f "$HF_CACHE/$fname" ] || {
    echo "Steering requested but $HF_CACHE/$fname is missing on the head." >&2
    echo "  hf download msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47 --include '*.gguf' --local-dir $HF_CACHE" >&2
    exit 1
  }
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "test -f '$WORKER_HF_CACHE/$fname'" || {
    echo "Steering requested but $WORKER_HF_CACHE/$fname is missing on the worker ($WORKER_HOST)." >&2
    echo "  run the same hf download there (the wizard can sync it)." >&2
    exit 1
  }
  STEER_ENV="-e WEIGHTLESS_STEER_PATH=$WEIGHTLESS_STEER_PATH -e WEIGHTLESS_STEER_ALPHA=${WEIGHTLESS_STEER_ALPHA:-1.0}"
  [ -n "${WEIGHTLESS_STEER_LAYERS:-}" ] && STEER_ENV="$STEER_ENV -e WEIGHTLESS_STEER_LAYERS=$WEIGHTLESS_STEER_LAYERS"
  [ -n "${WEIGHTLESS_STEERING_MODEL_PY:-}" ] && STEER_ENV="$STEER_ENV -e WEIGHTLESS_STEERING_MODEL_PY=$WEIGHTLESS_STEERING_MODEL_PY"
fi

# --- refuse to clobber a running stack --------------------------------------
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "$CONTAINER already running on the head. docker rm -f $CONTAINER (and on the worker) for a cold start." >&2
  exit 3
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" "true" >/dev/null || {
  echo "Cannot reach worker with passwordless SSH: $WORKER_HOST" >&2
  exit 1
}
if ssh "$WORKER_HOST" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'"; then
  echo "$CONTAINER already running on the worker. docker rm -f $CONTAINER there for a cold start." >&2
  exit 3
fi
docker image inspect "$QWEN38FN_IMAGE" >/dev/null || {
  echo "Missing local Docker image $QWEN38FN_IMAGE — docker pull it first." >&2
  exit 1
}
ssh "$WORKER_HOST" "docker image inspect '$QWEN38FN_IMAGE' >/dev/null" || {
  echo "Missing worker Docker image $QWEN38FN_IMAGE — docker pull it there first." >&2
  exit 1
}

# --- docker run command (per node) ------------------------------------------
# $1 rank, $2 api|headless, $3 host HF cache, $4 host hotfix path, $5 host IP,
# $6 host PLE-patch path.
# Env values are operator-controlled and must not contain spaces.
build_cmd() {
  local rank="$1" mode="$2" hf="$3" hotfix="$4" hostip="$5" plepatch="$6"
  local ep_args="--headless"
  if [ "$mode" = "api" ]; then
    ep_args="--host 0.0.0.0 --port $VLLM_PORT"
  fi
  cat <<EOF
docker run -d --restart no --name $CONTAINER \
  --log-opt max-size=50m --log-opt max-file=2 \
  --gpus all --ipc=host --network host --shm-size 64gb \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --device /dev/infiniband \
  -v $hf:/cache/huggingface \
  -v $hotfix:/patches/hotfix-qwen38fn-steering-projective.py:ro \
  -v $plepatch:/patches/patch-qwen38fn-ple-fp8-nvfp4.py:ro \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
  -e VLLM_PLE_CPU_OFFLOAD=${VLLM_PLE_CPU_OFFLOAD:-0} \
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
  $QWEN38FN_IMAGE \
  -c 'python3 /patches/patch-qwen38fn-ple-fp8-nvfp4.py && python3 /patches/hotfix-qwen38fn-steering-projective.py && exec vllm serve $MODEL \
        --served-model-name $SERVED_MODEL_NAME \
        --tensor-parallel-size 2 \
        --nnodes 2 --node-rank $rank \
        --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
        --distributed-executor-backend mp \
        --load-format safetensors --safetensors-load-strategy lazy \
        --max-num-seqs ${MAX_NUM_SEQS:-8} \
        --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS:-8192} \
        --no-enable-prefix-caching \
        --max-model-len $MAX_MODEL_LEN \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        $ep_args'
EOF
}

echo "Resolved Qwen3.8-Flash-Next profile:"
echo "  image: $QWEN38FN_IMAGE"
echo "  model: $MODEL (~135 GB NVFP4, TP=2)"
echo "  served model: $SERVED_MODEL_NAME"
echo "  head: $MASTER_ADDR (api :$VLLM_PORT)  worker: $WORKER_HOST"
echo "  max model len: $MAX_MODEL_LEN, gpu util: $GPU_MEMORY_UTILIZATION"
echo "  prefix caching: OFF (mandatory — mamba align corrupts steering)"
echo "  VLLM_PLE_CPU_OFFLOAD: ${VLLM_PLE_CPU_OFFLOAD:-1} (0 = PLE table in HBM; 1 is unsupported at nnodes=2 on the arm64 image)"
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  echo "  steering: $WEIGHTLESS_STEER_PATH (α=${WEIGHTLESS_STEER_ALPHA:-1.0}, layers=${WEIGHTLESS_STEER_LAYERS:-all in file})"
else
  echo "  steering: off (WEIGHTLESS_STEER_PATH empty)"
fi

# --- worker first ------------------------------------------------------------
echo "Syncing env + hotfix to ${WORKER_HOST}:${WORKER_DIR}"
ssh "$WORKER_HOST" "mkdir -p '$WORKER_DIR/patches'"
scp "$ENV_FILE" "$WORKER_HOST:$WORKER_DIR/.env.qwen38fn" >/dev/null
scp "$HOTFIX" "$WORKER_HOST:$WORKER_DIR/patches/hotfix-qwen38fn-steering-projective.py" >/dev/null
scp "$PLE_PATCH" "$WORKER_HOST:$WORKER_DIR/patches/patch-qwen38fn-ple-fp8-nvfp4.py" >/dev/null

# Drop page caches on both nodes before launch — mandatory on GB10 unified
# memory (the model rsync just wrote ~135 GB of page cache; CUDA allocations
# squeeze against it and the boot OOMs/wedges). MiaAI-Lab's measured recipe.
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null
ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_HOST" \
  "sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null"

echo "Starting worker rank 1 (headless) on $WORKER_HOST..."
ssh "$WORKER_HOST" "docker rm -f $CONTAINER >/dev/null 2>&1 || true"
ssh "$WORKER_HOST" "$(build_cmd 1 headless "$WORKER_HF_CACHE" "$WORKER_DIR/patches/hotfix-qwen38fn-steering-projective.py" "$WORKER_VLLM_HOST_IP" "$WORKER_DIR/patches/patch-qwen38fn-ple-fp8-nvfp4.py")"

echo "Starting head rank 0 (API :$VLLM_PORT)..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
bash -c "$(build_cmd 0 api "$HF_CACHE" "$HOTFIX" "$VLLM_HOST_IP" "$PLE_PATCH")"

echo "Waiting for the API (model load takes minutes; NVFP4 weights stream from both caches)..."
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "Qwen3.8-Flash-Next is running: http://$MASTER_ADDR:$VLLM_PORT/v1 (model id $SERVED_MODEL_NAME)"
    if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
      echo "Confirm steering in the boot log: 'weightless GLP steering active ... layers=47'"
      echo "(layers=1 means the per-layer-loop regression; layers=0 means unsteered.)"
    fi
    exit 0
  fi
  sleep "$WAIT_SECONDS"
done

echo "Timed out waiting for the API. Recent head logs:" >&2
docker logs --tail 120 "$CONTAINER" >&2 || true
echo "Recent worker logs:" >&2
ssh "$WORKER_HOST" "docker logs --tail 120 '$CONTAINER'" >&2 || true
exit 1
