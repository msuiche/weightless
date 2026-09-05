#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 TP=2 across 2x DGX Spark (GB10, sm_121).
# Reference: tonyd2wild's TP2 deployment report + later KV/graph corrections.
# sm121-v8, fp8 KV, block 2304, profiler-sized memory, CUDA graphs on.
# Default agentic profile: 128K, no MTP/drafter; optional GLP-44 at alpha 2.
#
# Run on the head. Refuse any live serving lane or leftover GPU process on
# either node; never remove containers. Check the actual image's steering
# anchors on BOTH nodes before starting either rank, then patch again in
# each serving container with an && guard. Worker first, head last.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.glm53tp2}"
[ -f "$ENV_FILE" ] || { echo "Missing $ENV_FILE — copy .env.glm53tp2.example and edit it." >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${MASTER_ADDR:?}"; : "${WORKER_HOST:?}"; : "${HF_CACHE:?}"
: "${GLM53_IMAGE:?}"; : "${MODEL:?}"
: "${NCCL_IB_HCA:?}"; : "${NCCL_SOCKET_IFNAME:?}"
CONTAINER="${CONTAINER_NAME:-vllm-glm53tp2}"
MASTER_PORT="${MASTER_PORT:-29521}"
VLLM_PORT="${VLLM_PORT:-8080}"
WORKER_DIR="${WORKER_DIR:-$HOME/dspark-glm53tp2}"
WORKER_HF_CACHE="${WORKER_HF_CACHE:-$HF_CACHE}"
VLLM_HOST_IP="${VLLM_HOST_IP:-$MASTER_ADDR}"
WORKER_VLLM_HOST_IP="${WORKER_VLLM_HOST_IP:-$WORKER_HOST}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm53-flash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION_OVERRIDE:-${GPU_MEMORY_UTILIZATION:-0.85}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-240}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"
RANK_STAGGER_SECONDS="${RANK_STAGGER_SECONDS:-25}"
MODEL_CACHE_DIR="models--${MODEL//\//--}"
MODEL_PY=/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py
KPOOL_PY=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py

die() { echo "$*" >&2; exit 1; }
for cmd in docker ssh scp curl awk; do
  command -v "$cmd" >/dev/null 2>&1 || die "Missing required command: $cmd"
done
case "$GLM53_IMAGE" in
  ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8|ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8@sha256:*) ;;
  *) die "Use the reference ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8 image (optionally digest-pinned)." ;;
esac
[ "$MODEL" = RedHatAI/GLM-5.3-Flash-NVFP4 ] || die "This lane requires RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-tensors W4A4)."
[[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] && [ "$MAX_MODEL_LEN" -ge 1 ] && [ "$MAX_MODEL_LEN" -le 262144 ] || die "MAX_MODEL_LEN must be 1..262144."
awk -v u="$GPU_MEMORY_UTILIZATION" 'BEGIN {exit !(u ~ /^0\.[0-9]+$/ && u > 0 && u <= 0.85)}' || die "GPU_MEMORY_UTILIZATION must be >0 and <=0.85; higher TP2 budgets need hardware qualification."
[ -z "${KV_CACHE_MEMORY:-}" ] || die "Do not pin KV_CACHE_MEMORY on TP2; the profiler must reserve activation headroom."

# Shell-quote each argument, including the complete remote command. This
# keeps paths/env values intact through SSH's extra shell, without eval.
shell_join() { printf '%q ' "$@"; }
run_node() {
  local target="$1"; shift
  if [ -z "$target" ]; then
    "$@"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" "$(shell_join bash -c "$(shell_join "$@")")"
  fi
}

# Resolve existing shared patches in either a checkout or wizard deployment.
HOTFIX="${WEIGHTLESS_STEERING_HOTFIX:-$SCRIPT_DIR/patches/hotfix-glm53-steering-projective.py}"
[ -f "$HOTFIX" ] || HOTFIX="$SCRIPT_DIR/../../patches/hotfix-glm53-steering-projective.py"
KPOOL="${WEIGHTLESS_KPOOL_FIX:-$SCRIPT_DIR/patches/sparse_attn_indexer_kpool_sm121.py}"
[ -f "$KPOOL" ] || KPOOL="$SCRIPT_DIR/../../patches/vendor/sparse_attn_indexer_kpool_sm121.py"
[ -f "$KPOOL" ] || die "Missing the shared SM121 kpool top-k fix: $KPOOL"
[ -f "$HOTFIX" ] || die "Missing the shared GLP-44 hotfix: $HOTFIX"
HOTFIX="$(cd "$(dirname "$HOTFIX")" && pwd)/$(basename "$HOTFIX")"
KPOOL="$(cd "$(dirname "$KPOOL")" && pwd)/$(basename "$KPOOL")"

# --- no-clobber + hardware preflight (both nodes before any staging) --------
check_idle() {
  local target="$1" running gpu_pids existing
  running=$(run_node "$target" docker ps --no-trunc --format '{{.Names}} {{.Image}} {{.Command}}') || die "Cannot inspect Docker on ${target:-head}."
  if printf '%s\n' "$running" | awk -v name="$CONTAINER" '
    $1 == name || tolower($0) ~ /vllm|sglang|inkling|glm53|glm5xl|qwen|deepseek/ {found=1}
    END {exit !found}'; then
    die "A serving container is running on ${target:-head}. Save logs and stop ALL ranks of the active lane before relaunching."
  fi
  existing=$(run_node "$target" docker ps -a --format '{{.Names}}') || die "Cannot inspect stopped containers on ${target:-head}."
  if printf '%s\n' "$existing" | awk -v name="$CONTAINER" '$0 == name {found=1} END {exit !found}'; then
    die "$CONTAINER already exists on ${target:-head}. Save its logs and remove it explicitly first."
  fi
  gpu_pids=$(run_node "$target" nvidia-smi --query-compute-apps=pid --format=csv,noheader) || die "GPU preflight failed on ${target:-head}."
  [ -z "$gpu_pids" ] || die "GPU processes still alive on ${target:-head}: $gpu_pids"
}
HEAD_IMAGE_ID=
for h in "" "$WORKER_HOST"; do
  check_idle "$h"
  iid=$(run_node "$h" docker image inspect --format '{{.Id}}' "$GLM53_IMAGE") || die "Pull $GLM53_IMAGE on ${h:-head} first."
  [ -n "$iid" ] || die "Empty image ID on ${h:-head}."
  [ -n "$HEAD_IMAGE_ID" ] || HEAD_IMAGE_ID="$iid"
  [ "$iid" = "$HEAD_IMAGE_ID" ] || die "Image ID mismatch on $h; sync the image to BOTH nodes."
  run_node "$h" bash -c '[ "$(cat /proc/sys/vm/swappiness)" = 0 ] && [ "$(awk '\''/SwapTotal/{print $2}'\'' /proc/meminfo)" -gt 0 ]' || die "${h:-head}: keep swap available and set vm.swappiness=0 (reference UVM-livelock/repack rule)."
  # Mandatory, non-interactive cache flush; a failure must stop this boot.
  run_node "$h" bash -c 'sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null' || die "drop_caches failed on ${h:-head}; configure passwordless sudo for the flush."
  mem=$(run_node "$h" awk '/MemTotal/{t=$2} /MemAvailable/{a=$2} END{print t, a}' /proc/meminfo) || die "Cannot read memory on ${h:-head}."
  printf '%s\n' "$mem" | awk -v u="$GPU_MEMORY_UTILIZATION" '{exit !(NF == 2 && $1 > 0 && $2 >= u*$1)}' || die "Insufficient MemAvailable on ${h:-head} for utilization $GPU_MEMORY_UTILIZATION ($mem KiB total/available)."
done

# --- stage shared patches and check the actual image before either rank ----
run_node "$WORKER_HOST" mkdir -p "$WORKER_DIR/patches"
scp -o BatchMode=yes "$HOTFIX" "$KPOOL" "$WORKER_HOST:$WORKER_DIR/patches/" >/dev/null
STEER_ENV=()
PATCH_ENTRY='exec "$@"'
if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
  # The wizard and TP4 lane place the vector at the cache root on each node.
  [ "$WEIGHTLESS_STEER_PATH" = "/cache/huggingface/$(basename "$WEIGHTLESS_STEER_PATH")" ] || die "WEIGHTLESS_STEER_PATH must name a file at /cache/huggingface/<GLP-44 gguf>."
  STEER_ENV=(-e "WEIGHTLESS_STEER_PATH=$WEIGHTLESS_STEER_PATH"
    -e "WEIGHTLESS_STEER_ALPHA=${WEIGHTLESS_STEER_ALPHA:-2.0}"
    -e "WEIGHTLESS_STEER_LAYERS=${WEIGHTLESS_STEER_LAYERS:-}")
  PATCH_ENTRY='python3 /patches/hotfix-glm53-steering-projective.py && exec "$@"'
fi
for h in "" "$WORKER_HOST"; do
  hf="$HF_CACHE"; hotfix="$HOTFIX"
  if [ -n "$h" ]; then
    hf="$WORKER_HF_CACHE"; hotfix="$WORKER_DIR/patches/$(basename "$HOTFIX")"
  fi
  # refs/main identifies the snapshot vLLM will actually load, not an
  # arbitrary old snapshot. The checkpoint must be cached before boot.
  run_node "$h" bash -c '
    root="$1/$2"; revision=$(cat "$root/refs/main") || exit 1
    config="$root/snapshots/$revision/config.json"
    test -f "$config" && awk '\''/"quant_method"[[:space:]]*:[[:space:]]*"compressed-tensors"/ {ok=1} END{exit !ok}'\'' "$config"
  ' bash "$hf" "$MODEL_CACHE_DIR" || die "Missing RedHatAI compressed-tensors main snapshot in $hf on ${h:-head}."
  if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
    run_node "$h" test -f "$hf/$(basename "$WEIGHTLESS_STEER_PATH")" || die "Missing steering vector on ${h:-head}."
    # CPU-only disposable check: exact image file + all five anchors + GGUF
    # validation. The hotfix returns nonzero on drift with steering armed.
    run_node "$h" docker run --rm --network none \
      --log-opt max-size=50m --log-opt max-file=2 \
      -v "$hf:/cache/huggingface:ro" -v "$hotfix:/patches/hotfix-glm53-steering-projective.py:ro" \
      "${STEER_ENV[@]}" -e "WEIGHTLESS_STEERING_MODEL_PY=$MODEL_PY" \
      --entrypoint python3 "$GLM53_IMAGE" /patches/hotfix-glm53-steering-projective.py || die "Steering preflight failed on ${h:-head}; refusing to start either rank."
  fi
done

# --- launch: worker first, then head; no automatic removal or restart -------
build_cmd() { # rank, host IP, HF cache, hotfix, kpool
  local rank="$1" hostip="$2" hf="$3" hotfix="$4" kpool="$5"
  CMD=(docker run -d --restart no --name "$CONTAINER"
    --log-opt max-size=50m --log-opt max-file=2
    --gpus all --ipc=host --network host --shm-size 32g
    --ulimit memlock=-1:-1 --ulimit stack=67108864 --cap-add IPC_LOCK
    --device /dev/infiniband
    -v "$hf:/cache/huggingface"
    -v "$hotfix:/patches/hotfix-glm53-steering-projective.py:ro"
    -v "$kpool:$KPOOL_PY:ro"
    -e HF_HOME=/cache/huggingface -e HF_HUB_CACHE=/cache/huggingface
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e "VLLM_HOST_IP=$hostip" -e VLLM_ENGINE_READY_TIMEOUT_S=3600
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a
    -e FLASHINFER_DISABLE_VERSION_CHECK=1
    -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=$NCCL_IB_HCA"
    -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}"
    -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET
    -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    -e "TP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME" -e "MN_IF_NAME=$NCCL_SOCKET_IFNAME"
    -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0
    -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    ${STEER_ENV[@]+"${STEER_ENV[@]}"} --entrypoint bash "$GLM53_IMAGE" -c "$PATCH_ENTRY" bash
    vllm serve "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --trust-remote-code
    --tensor-parallel-size 2 --nnodes 2 --node-rank "$rank"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --distributed-executor-backend mp
    --kv-cache-dtype fp8_e4m3 --block-size 2304 --moe-backend marlin
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
    --default-chat-template-kwargs '{"enable_thinking":true}')
  if [ "$rank" -eq 0 ]; then CMD+=(--host 0.0.0.0 --port "$VLLM_PORT"); else CMD+=(--headless); fi
}
show_logs() {
  docker logs --tail 80 "$CONTAINER" >&2 || true
  run_node "$WORKER_HOST" docker logs --tail 80 "$CONTAINER" >&2 || true
}
is_running() {
  [ "$(run_node "$1" docker inspect --format '{{.State.Running}}' "$CONTAINER")" = true ]
}
echo "GLM-5.3-Flash TP=2: $GLM53_IMAGE ($HEAD_IMAGE_ID), ctx $MAX_MODEL_LEN, util $GPU_MEMORY_UTILIZATION, fp8 KV, CUDA graphs, API :$VLLM_PORT"
echo "Steering: ${WEIGHTLESS_STEER_PATH:-off}; alpha ${WEIGHTLESS_STEER_ALPHA:-2.0}"
# Repeat the idle check after staging so another lane started meanwhile
# blocks launch too. Docker's name reservation protects same-name races.
check_idle ""
check_idle "$WORKER_HOST"
build_cmd 1 "$WORKER_VLLM_HOST_IP" "$WORKER_HF_CACHE" "$WORKER_DIR/patches/$(basename "$HOTFIX")" "$WORKER_DIR/patches/$(basename "$KPOOL")"
run_node "$WORKER_HOST" "${CMD[@]}"
sleep "$RANK_STAGGER_SECONDS"
is_running "$WORKER_HOST" || { show_logs; die "Worker exited before head launch; inspect logs before teardown."; }
build_cmd 0 "$VLLM_HOST_IP" "$HF_CACHE" "$HOTFIX" "$KPOOL"
"${CMD[@]}" || { show_logs; die "Head launch failed; stop the worker explicitly before retrying."; }

echo "Waiting for /health (~15 min boot; docker logs -f $CONTAINER)..."
for ((i=0; i<WAIT_ATTEMPTS; i++)); do
  if ! is_running "" || ! is_running "$WORKER_HOST"; then
    show_logs; die "A rank exited; no automatic restart. Save logs and tear down both ranks."
  fi
  if curl -fsS --max-time 5 "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then
    echo "Engine healthy: http://$MASTER_ADDR:$VLLM_PORT/v1 (model $SERVED_MODEL_NAME). Run the README smoke tests."
    exit 0
  fi
  sleep "$WAIT_SECONDS"
done
show_logs
die "Timed out waiting for /health; preserve logs before stopping both ranks."
