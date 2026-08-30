#!/usr/bin/env bash
# GLM-5.3 743B lane: TP=4 across 4x DGX Spark (head + 3 workers over RoCE) on
# tonyd2wild's Int4-Int8Mix stack, with GLP-77 projective refusal steering.
#
# THIS LANE NEEDS 4 NODES. The serve config is tonyd2wild's hardware-validated
# TP4 recipe (see README.md): Int4-Int8Mix weights (~95.5 GiB/rank), the
# glm-triton sm12x kernel overlay, fp8_ds_mla KV with the budget pinned, MTP
# k=4, CUDA graphs FULL.
#
# Boot ritual (each rule cost someone a boot — README.md has the receipts):
#   * the image is a LOCAL build (vllm-node-tf5-glm52-b12x:probe-modded) and
#     must have the SAME ID on all four nodes (checked below);
#   * the 10-file kernel overlay must be present on every node, and
#     deepseek_v2.py <-> sparse_attn_indexer.py must be a matched pair;
#   * cache-flush ritual on every node before boot (GB10 page cache eats
#     CUDA-visible memory 1:1 — gmu 0.91 only boots with the cache held down);
#   * workers first (~20 s apart), head LAST;
#   * never relaunch a rank while others are up — this script refuses to
#     start if $CONTAINER exists on ANY node (tear down all, then boot).
#
# Steering is staged, not entrypoint-patched: their stack bind-mounts the
# overlay deepseek_v2.py read-only, so this script copies it per node, runs
# the hotfix on the copy INSIDE a throwaway container (the image has torch;
# the node host may not), and mounts the PATCHED copy. The anchor check fails
# closed before any GPU time is spent; the runtime loader re-raises on any
# vector failure, so a steered boot never serves unsteered.
#
# Usage: cp .env.glm53xl.example .env.glm53xl, fill in the <...> values, then
#        bash start-glm53xl-dspark.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.glm53xl}"
CONTAINER="${CONTAINER_NAME:-glm5xl}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-200}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"
RANK_STAGGER_SECONDS="${RANK_STAGGER_SECONDS:-20}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE. Copy .env.glm53xl.example to .env.glm53xl and edit node-specific values." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${WORKER2_HOST:?WORKER2_HOST must be set in $ENV_FILE}"
: "${WORKER3_HOST:?WORKER3_HOST must be set in $ENV_FILE}"
: "${HF_CACHE:?HF_CACHE must be set in $ENV_FILE}"
: "${GLM53XL_IMAGE:?GLM53XL_IMAGE must be set in $ENV_FILE}"
: "${KERNELS_DIR:?KERNELS_DIR must be set in $ENV_FILE}"
: "${WEIGHTS_HEAD:?WEIGHTS_HEAD must be set in $ENV_FILE}"
: "${WEIGHTS_WORKERS:?WEIGHTS_WORKERS must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"

MASTER_PORT="${MASTER_PORT:-25030}"
WORKER_DIR="${WORKER_DIR:-$HOME/dspark-glm53xl}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.3}"
VLLM_PORT="${VLLM_PORT:-8081}"
SPEC_MODE="${SPEC_MODE:-mtp}"
MTP_NUM_TOKENS="${MTP_NUM_TOKENS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-200000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.91}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-10950000000}"
VLLM_HOST_IP="${VLLM_HOST_IP:-$MASTER_ADDR}"
LD_PRELOAD_LIB="${NCCL_LD_PRELOAD:-/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2}"
# rank 1..3, in order
WORKERS=("$WORKER_HOST" "$WORKER2_HOST" "$WORKER3_HOST")
HOTFIX="${WEIGHTLESS_STEERING_HOTFIX:-$SCRIPT_DIR/patches/hotfix-glm53xl-steering-projective.py}"
if [ ! -f "$HOTFIX" ]; then
  HOTFIX="$(cd "$SCRIPT_DIR/../../patches" && pwd)/hotfix-glm53xl-steering-projective.py"
fi
[ -f "$HOTFIX" ] || { echo "Missing steering hotfix (looked in $SCRIPT_DIR/patches and the repo)." >&2; exit 1; }

# in-image overlay targets (from tonyd2wild's launcher)
MLA="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla"
OPS="/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/deepseek_v4_ops"
LAYERS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers"
MODELS="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"

KERNEL_FILES=(sparse_mla_kernels.py sparse_mla_env.py sm12x_sparse_mla_attn.py
  patch_flashmla_ops.py flashmla_sparse.py sm12x_deep_gemm_fallbacks.py
  sm12x_mqa.py b12x_sparse_helpers.py sparse_attn_indexer.py deepseek_v2.py)

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }
}
need_cmd docker
need_cmd ssh
need_cmd scp
need_cmd curl

# Run a command on a node (empty arg = local head).
on_node() {
  local target="$1"; shift
  if [ -z "$target" ]; then bash -c "$*"; else ssh -o BatchMode=yes "$target" "$*"; fi
}

# --- kernel overlay preflight (their repo's issue #5 checks) ----------------
overlay_check() {
  local target="$1" kdir="$2"
  on_node "$target" "
    set -e
    for f in ${KERNEL_FILES[*]}; do
      [ -f '$kdir/'\$f ] || { echo \"kernel overlay missing: $kdir/\$f\" >&2; exit 4; }
    done
    if grep -q fused_indexer_q_rope_quant '$kdir/deepseek_v2.py' 2>/dev/null \
       && ! grep -Eq 'def[[:space:]]+fused_indexer_q_rope_quant' '$kdir/sparse_attn_indexer.py' 2>/dev/null; then
      echo 'kernel mismatch (issue #5): version-skewed overlays' >&2; exit 5
    fi
    grep -q GlmMoeDsaForCausalLM '$kdir/deepseek_v2.py' || {
      echo 'overlay deepseek_v2.py does not define GlmMoeDsaForCausalLM' >&2; exit 6; }
  " >/dev/null
}
echo "preflight: kernel overlay on all 4 nodes"
overlay_check "" "$KERNELS_DIR"
for w in "${WORKERS[@]}"; do overlay_check "$w" "$KERNELS_DIR"; done

# --- preflight: stack down, image identical, weights + NCCL lib visible ------
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "$CONTAINER already running on the head. Tear down ALL ranks before relaunching any." >&2
  exit 3
fi
HEAD_IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$GLM53XL_IMAGE" 2>/dev/null)" || {
  echo "Missing local Docker image $GLM53XL_IMAGE — build per the GLM-5.2 QuantTrio repo first." >&2
  exit 1
}
[ -f "$WEIGHTS_HEAD/config.json" ] || { echo "weights not visible at $WEIGHTS_HEAD" >&2; exit 1; }
[ -f "$HF_CACHE/hub/nccl-2.30.4/libnccl.so.2" ] || \
  echo "WARN: $HF_CACHE/hub/nccl-2.30.4/libnccl.so.2 missing on the head (NCCL re-pin; see README)" >&2
for w in "${WORKERS[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$w" "true" >/dev/null || {
    echo "Cannot reach worker with passwordless SSH: $w" >&2; exit 1; }
  if ssh "$w" "docker ps --format '{{.Names}}' | grep -qx '$CONTAINER'"; then
    echo "$CONTAINER already running on worker $w. Tear down ALL ranks before relaunching any." >&2
    exit 3
  fi
  wid="$(ssh "$w" "docker image inspect --format '{{.Id}}' '$GLM53XL_IMAGE' 2>/dev/null")" || {
    echo "Missing worker Docker image $GLM53XL_IMAGE on $w." >&2; exit 1; }
  if [ "$wid" != "$HEAD_IMAGE_ID" ]; then
    echo "Image ID mismatch: head has $HEAD_IMAGE_ID, $w has $wid — sync the image first." >&2
    exit 1
  fi
  ssh "$w" "test -f '$WEIGHTS_WORKERS/config.json'" || {
    echo "weights not visible at $WEIGHTS_WORKERS on $w (NFS mount down?)" >&2; exit 1; }
done
echo "preflight: image ${HEAD_IMAGE_ID} identical on all 4 nodes"

# --- cache-flush ritual (GB10 unified memory) --------------------------------
flush_caches() {
  local target="$1"
  if [ -z "$target" ]; then
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null
  else
    ssh -o BatchMode=yes "$target" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" 2>/dev/null
  fi
}
echo "cache-flush ritual (drop_caches on all 4 nodes; run cache_flusher.sh during the boot):"
flush_caches "" || echo "  WARN: head drop_caches failed — run it by hand" >&2
for w in "${WORKERS[@]}"; do
  flush_caches "$w" || echo "  WARN: drop_caches failed on $w — run it there by hand" >&2
  pgrep_out="$(ssh -o BatchMode=yes "$w" "pgrep -fc cache_flusher 2>/dev/null || true")"
  [ "${pgrep_out:-0}" -ge 1 ] 2>/dev/null || \
    echo "  WARN: no cache_flusher running on $w (recommended during the weight read — README.md)" >&2
done

# --- steering preflight ------------------------------------------------------
STEER_ENV=""
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  fname="$(basename "$WEIGHTLESS_STEER_PATH")"
  [ -f "$HF_CACHE/$fname" ] || {
    echo "Steering requested but $HF_CACHE/$fname is missing on the head." >&2
    echo "  hf download msuiche/GLM-5.3-abliterated-cyber-GLP-77 --include '*.gguf' --local-dir $HF_CACHE" >&2
    exit 1
  }
  for w in "${WORKERS[@]}"; do
    ssh -o BatchMode=yes "$w" "test -f '$WORKER_HF_CACHE/$fname'" || {
      echo "Steering requested but $WORKER_HF_CACHE/$fname is missing on worker $w." >&2
      exit 1
    }
  done
  STEER_ENV="-e WEIGHTLESS_STEER_PATH=$WEIGHTLESS_STEER_PATH -e WEIGHTLESS_STEER_ALPHA=${WEIGHTLESS_STEER_ALPHA:-1.0}"
  [ -n "${WEIGHTLESS_STEER_LAYERS:-}" ] && STEER_ENV="$STEER_ENV -e WEIGHTLESS_STEER_LAYERS=$WEIGHTLESS_STEER_LAYERS"
  [ -n "${WEIGHTLESS_STEERING_MODEL_PY:-}" ] && STEER_ENV="$STEER_ENV -e WEIGHTLESS_STEERING_MODEL_PY=$WEIGHTLESS_STEERING_MODEL_PY"
fi

# --- steering staging: patch a COPY of the overlay deepseek_v2.py ------------
# Runs the hotfix inside a throwaway container (the image has python3 + torch;
# the node host may not). Fail-closed: bad anchors or a bad vector abort here,
# before any GPU time. $1 = node (empty = head), $2 = node's HF cache.
stage_steering() {
  local target="$1" hf="$2"
  on_node "$target" "
    set -e
    mkdir -p '$WORKER_DIR/patches'
    cp '$KERNELS_DIR/deepseek_v2.py' '$WORKER_DIR/patches/deepseek_v2.py'
  "
  if [ -n "$target" ]; then
    scp "$HOTFIX" "$target:$WORKER_DIR/patches/hotfix-glm53xl-steering-projective.py" >/dev/null
  else
    cp "$HOTFIX" "$WORKER_DIR/patches/hotfix-glm53xl-steering-projective.py"
  fi
  on_node "$target" "
    docker run --rm \
      -v '$WORKER_DIR/patches:/patches' \
      -v '$hf:/cache/huggingface' \
      -e WEIGHTLESS_STEERING_MODEL_PY=/patches/deepseek_v2.py \
      $STEER_ENV \
      --entrypoint python3 \
      '$GLM53XL_IMAGE' /patches/hotfix-glm53xl-steering-projective.py
  "
}

# --- docker run command (per node) ------------------------------------------
# $1 rank, $2 api|headless, $3 host HF cache, $4 host weights dir, $5 host IP.
build_cmd() {
  local rank="$1" mode="$2" hf="$3" weights="$4" hostip="$5"
  local ep_args="--headless"
  if [ "$mode" = "api" ]; then
    ep_args="--host 0.0.0.0 --port $VLLM_PORT"
  fi
  local spec_args=""
  case "$SPEC_MODE" in
    none) ;;
    mtp) spec_args="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_NUM_TOKENS,\"draft_tensor_parallel_size\":1,\"attention_backend\":\"FLASHMLA_SPARSE\"}'" ;;
    *) echo "SPEC_MODE must be none|mtp (got $SPEC_MODE)" >&2; exit 2 ;;
  esac
  cat <<EOF
docker run -d --restart no --name $CONTAINER \
  --gpus all --ipc=host --network host --shm-size 10gb \
  --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  --device /dev/infiniband \
  -v $hf:/cache/huggingface \
  -v $weights:/models/glm-5.3:ro \
  -v $KERNELS_DIR/sparse_mla_kernels.py:$MLA/sparse_mla_kernels.py:ro \
  -v $KERNELS_DIR/sparse_mla_env.py:$MLA/sparse_mla_env.py:ro \
  -v $KERNELS_DIR/sm12x_sparse_mla_attn.py:$MLA/sm12x_sparse_mla_attn.py:ro \
  -v $KERNELS_DIR/patch_flashmla_ops.py:$MLA/patch_flashmla_ops.py:ro \
  -v $KERNELS_DIR/flashmla_sparse.py:$MLA/flashmla_sparse.py:ro \
  -v $KERNELS_DIR/sm12x_deep_gemm_fallbacks.py:$OPS/sm12x_deep_gemm_fallbacks.py:ro \
  -v $KERNELS_DIR/sm12x_mqa.py:$OPS/sm12x_mqa.py:ro \
  -v $KERNELS_DIR/b12x_sparse_helpers.py:$OPS/b12x_sparse_helpers.py:ro \
  -v $KERNELS_DIR/sparse_attn_indexer.py:$LAYERS/sparse_attn_indexer.py:ro \
  -v $WORKER_DIR/patches/deepseek_v2.py:$MODELS/deepseek_v2.py:ro \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e LD_PRELOAD=$LD_PRELOAD_LIB \
  -e HF_HOME=/cache/huggingface \
  -e TRITON_CACHE_DIR=/cache/huggingface/.tritoncache \
  -e HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e GLM52_BIND_HOST_TRITON=1 \
  -e GLM52_MQA_LOGITS_TRITON=1 \
  -e GLM52_PAGED_MQA_TRITON=1 \
  -e GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192 \
  -e GLM52_B12X_MLA=1 \
  -e VLLM_DISABLE_FLASHINFER_AUTOTUNE=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
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
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
  $STEER_ENV \
  $GLM53XL_IMAGE \
  vllm serve /models/glm-5.3 \
    --served-model-name $SERVED_MODEL_NAME \
    --trust-remote-code \
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
    --enable-prefix-caching \
    --async-scheduling \
    $spec_args \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --max-model-len $MAX_MODEL_LEN --max-num-seqs $MAX_NUM_SEQS --max-num-batched-tokens 8192 \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --kv-cache-memory-bytes $KV_CACHE_MEMORY_BYTES \
    --kv-cache-dtype fp8_ds_mla \
    --distributed-executor-backend mp \
    --compilation-config '{"cudagraph_mode":"FULL"}' \
    --nnodes 4 --node-rank $rank \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    $ep_args
EOF
}

echo "Resolved GLM-5.3 743B profile:"
echo "  image: $GLM53XL_IMAGE"
echo "  weights: head=$WEIGHTS_HEAD workers=$WEIGHTS_WORKERS (Int4-Int8Mix, ~95.5 GiB/rank)"
echo "  served model: $SERVED_MODEL_NAME"
echo "  head: $MASTER_ADDR (api :$VLLM_PORT)  workers: ${WORKERS[*]}"
echo "  max model len: $MAX_MODEL_LEN, kv pinned: $KV_CACHE_MEMORY_BYTES bytes/rank, gmu: $GPU_MEMORY_UTILIZATION"
echo "  kv: fp8_ds_mla, spec: $SPEC_MODE (k=$MTP_NUM_TOKENS), cudagraph FULL"
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  echo "  steering: $WEIGHTLESS_STEER_PATH (α=${WEIGHTLESS_STEER_ALPHA:-1.0}, layers=${WEIGHTLESS_STEER_LAYERS:-all in file})"
  echo "  NOTE: α=1.0 is calibrated — higher alpha makes refusal WORSE at full length on this model"
else
  echo "  steering: off (WEIGHTLESS_STEER_PATH empty)"
fi

echo "Staging steering (patch overlay copy) on all 4 nodes..."
stage_steering "" "$HF_CACHE"
for w in "${WORKERS[@]}"; do stage_steering "$w" "$WORKER_HF_CACHE"; done

# --- workers first, ~20 s apart; head LAST -----------------------------------
rank=1
for w in "${WORKERS[@]}"; do
  echo "Syncing env to ${w}:${WORKER_DIR}"
  ssh "$w" "mkdir -p '$WORKER_DIR'"
  scp "$ENV_FILE" "$w:$WORKER_DIR/.env.glm53xl" >/dev/null

  echo "Starting worker rank $rank (headless) on $w..."
  ssh "$w" "docker rm -f $CONTAINER >/dev/null 2>&1 || true"
  ssh "$w" "$(build_cmd "$rank" headless "$WORKER_HF_CACHE" "$WEIGHTS_WORKERS" "$w")"
  rank=$((rank + 1))
  [ "$rank" -le 3 ] && sleep "$RANK_STAGGER_SECONDS"
done

echo "Starting head rank 0 (API :$VLLM_PORT)..."
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
bash -c "$(build_cmd 0 api "$HF_CACHE" "$WEIGHTS_HEAD" "$VLLM_HOST_IP")"

echo "Waiting for the API (378 GB of weights stream over NFS/local NVMe; this takes a while)..."
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "GLM-5.3 is running: http://$MASTER_ADDR:$VLLM_PORT/v1 (model id $SERVED_MODEL_NAME)"
    if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
      echo "Confirm steering in the boot log: 'weightless GLP steering active ... layers=77'"
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
