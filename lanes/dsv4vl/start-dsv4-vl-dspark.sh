#!/usr/bin/env bash
# start-dsv4-vl-dspark.sh — two-node boot of the DSV4-Vision-Exp VL lane
# (weightless lane 12). Worker-first launch with per-node RoCE GID
# resolution, lifted from the K4-validated upstream-vision test launcher
# (dspark-fork scripts/start-upstream-vision-test.sh) and hardened to the
# weightless lane contract: fail-closed env validation, hotfix sync to the
# worker, steering-active log gate, API + chat smoke.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dsv4vl}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dsv4vl.yml}"
API_URL="${API_URL:-http://127.0.0.1:8888/v1/models}"
CHAT_URL="${CHAT_URL:-http://127.0.0.1:8888/v1/chat/completions}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-120}"
WAIT_SECONDS="${WAIT_SECONDS:-15}"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE (copy .env.dsv4vl.example and fill the placeholders)." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

: "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE}"
: "${MASTER_ADDR:?MASTER_ADDR must be set in $ENV_FILE}"
: "${NCCL_IB_HCA:?NCCL_IB_HCA must be set in $ENV_FILE}"
: "${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME must be set in $ENV_FILE}"
: "${VLLM_HOST_IP:?VLLM_HOST_IP must be set in $ENV_FILE}"
: "${WORKER_VLLM_HOST_IP:?WORKER_VLLM_HOST_IP must be set in $ENV_FILE}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set in $ENV_FILE}"

# Defense in depth (wizard validate_lane_env mirrors these): serving
# containers are host-networked with no avahi inside — a .local name is a
# silent 600s TCP-store timeout or a post-NCCL shm_broadcast hang
# (2026-09-09, found via py-spy).
for kv in "MASTER_ADDR=$MASTER_ADDR" "VLLM_HOST_IP=$VLLM_HOST_IP" \
          "WORKER_VLLM_HOST_IP=$WORKER_VLLM_HOST_IP"; do
  case "$kv" in
    *.local)
      echo "FATAL: $kv — serving containers cannot resolve .local names. Use the literal fabric IP." >&2
      exit 1
      ;;
  esac
done

NCCL_IB_GID_AUTO="${NCCL_IB_GID_AUTO:-1}"
WORKER_NCCL_IB_HCA="${WORKER_NCCL_IB_HCA:-$NCCL_IB_HCA}"
WORKER_NCCL_SOCKET_IFNAME="${WORKER_NCCL_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME}"

on_node() {
  if [ -z "$1" ]; then bash -c "$2"; else ssh "$1" "bash -s" <<<"$2"; fi
}

iface_ipv4() {
  on_node "$1" "ip -4 -o addr show dev '$2' 2>/dev/null | awk '{print \$4}' | head -1 | cut -d/ -f1"
}

# RoCEv2 IPv4 GID index for every HCA in a comma-separated list (the index
# is per node and moves with fabric addressing — resolving it from sysfs on
# each boot is why this lane never pins NCCL_IB_GID_INDEX).
resolve_gid_index() {
  local target="$1" hcas="$2" snippet
  snippet=$(cat <<EOF
seen=''
for hca in \$(printf '%s' '$hcas' | tr ',' ' '); do
  netdev=\$(ls /sys/class/infiniband/\$hca/device/net 2>/dev/null | head -1)
  if [ -z "\$netdev" ]; then echo "no netdev for HCA \$hca" >&2; exit 1; fi
  hcaip=\$(ip -4 -o addr show dev "\$netdev" 2>/dev/null | awk '{print \$4}' | head -1 | cut -d/ -f1)
  if [ -z "\$hcaip" ]; then echo "\$hca (\$netdev) has no IPv4 address" >&2; exit 1; fi
  oldifs=\$IFS; IFS=.; set -- \$hcaip; IFS=\$oldifs
  hex=\$(printf '%02x%02x:%02x%02x' "\$1" "\$2" "\$3" "\$4")
  idx=''
  for g in /sys/class/infiniband/\$hca/ports/1/gids/*; do
    [ -e "\$g" ] || continue
    i=\${g##*/}
    t=\$(cat /sys/class/infiniband/\$hca/ports/1/gid_attrs/types/\$i 2>/dev/null || true)
    [ "\$t" = 'RoCE v2' ] || continue
    case \$(cat "\$g" 2>/dev/null) in *ffff:\$hex) idx=\$i; break ;; esac
  done
  if [ -z "\$idx" ]; then echo "no RoCEv2 GID on \$hca matching \$hcaip" >&2; exit 1; fi
  if [ -z "\$seen" ]; then seen=\$idx
  elif [ "\$seen" != "\$idx" ]; then
    echo "GID index differs across HCAs (\$seen vs \$idx); pin NCCL_IB_GID_INDEX" >&2
    exit 1
  fi
done
printf '%s' "\$seen"
EOF
)
  on_node "$target" "$snippet"
}

if [ "$NCCL_IB_GID_AUTO" = "1" ]; then
  HEAD_FABRIC_IP="$(iface_ipv4 "" "$NCCL_SOCKET_IFNAME")"
  WORKER_FABRIC_IP="$(iface_ipv4 "$WORKER_HOST" "$WORKER_NCCL_SOCKET_IFNAME")"
  [ -n "$HEAD_FABRIC_IP" ] || { echo "FATAL: no IPv4 on head $NCCL_SOCKET_IFNAME." >&2; exit 1; }
  [ -n "$WORKER_FABRIC_IP" ] || { echo "FATAL: no IPv4 on worker $WORKER_NCCL_SOCKET_IFNAME." >&2; exit 1; }
  NCCL_IB_GID_INDEX="$(resolve_gid_index "" "$NCCL_IB_HCA")" || {
    echo "FATAL: head GID resolve failed for $NCCL_IB_HCA." >&2
    exit 1
  }
  WORKER_NCCL_IB_GID_INDEX="$(resolve_gid_index "$WORKER_HOST" "$WORKER_NCCL_IB_HCA")" || {
    echo "FATAL: worker GID resolve failed for $WORKER_NCCL_IB_HCA." >&2
    exit 1
  }
  echo "RoCEv2 GID index: head=$NCCL_IB_GID_INDEX ($HEAD_FABRIC_IP) worker=$WORKER_NCCL_IB_GID_INDEX ($WORKER_FABRIC_IP)"
else
  : "${NCCL_IB_GID_INDEX:?NCCL_IB_GID_AUTO=0 requires NCCL_IB_GID_INDEX in $ENV_FILE}"
  WORKER_NCCL_IB_GID_INDEX="${WORKER_NCCL_IB_GID_INDEX:-$NCCL_IB_GID_INDEX}"
  echo "RoCEv2 GID index (pinned): head=$NCCL_IB_GID_INDEX worker=$WORKER_NCCL_IB_GID_INDEX"
fi
export NCCL_IB_GID_INDEX

WORKER_DIR="${WORKER_DIR:?WORKER_DIR must be set in $ENV_FILE}"
REMOTE_WORKER_DIR="$(printf '%q' "$WORKER_DIR")"
REMOTE_COMPOSE="cd $REMOTE_WORKER_DIR && env -u MASTER_ADDR -u MASTER_PORT -u NODE_RANK -u HEADLESS COMPOSE_DISABLE_ENV_FILE=1"

echo "Syncing lane files to ${WORKER_HOST}:${WORKER_DIR}"
ssh "$WORKER_HOST" "mkdir -p $REMOTE_WORKER_DIR/patches"
scp "$COMPOSE_FILE" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/docker-compose.dsv4vl.yml"
scp "$ENV_FILE" "${WORKER_HOST}:${REMOTE_WORKER_DIR}/.env.dsv4vl"
scp "$SCRIPT_DIR/patches/hotfix-dsv4vl-steering-projective.py" \
  "${WORKER_HOST}:${REMOTE_WORKER_DIR}/patches/hotfix-dsv4vl-steering-projective.py"

echo "Starting VL worker on ${WORKER_HOST}..."
ssh "$WORKER_HOST" "$REMOTE_COMPOSE NODE_RANK=1 HEADLESS=1 VLLM_HOST_IP='$WORKER_VLLM_HOST_IP' NCCL_IB_HCA='$WORKER_NCCL_IB_HCA' NCCL_SOCKET_IFNAME='$WORKER_NCCL_SOCKET_IFNAME' NCCL_IB_GID_INDEX='$WORKER_NCCL_IB_GID_INDEX' HF_CACHE='${WORKER_HF_CACHE:-${HF_CACHE:-}}' JIT_CACHE_DIR='${WORKER_JIT_CACHE_DIR:-${JIT_CACHE_DIR:-}}' docker compose --env-file .env.dsv4vl -f docker-compose.dsv4vl.yml up -d"

echo "Starting VL head..."
COMPOSE_DISABLE_ENV_FILE=1 NODE_RANK=0 HEADLESS= \
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d

echo "Waiting for VL API at $API_URL ..."
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "$API_URL" >/dev/null; then
    echo "VL serve is running: $API_URL"
    curl -fsS --max-time 5 "$API_URL"
    echo
    if [ -n "${WEIGHTLESS_STEER_PATH:-}" ]; then
      # Steering was requested: the patched model must have logged the
      # active line on BOTH ranks (the fail-closed hotfix would otherwise
      # have kept the container down, so a missing line means the patch
      # never ran — e.g. a stale image without the mount).
      for node in "" "$WORKER_HOST"; do
        label="${node:-head}"
        if ! on_node "$node" "docker logs deepseek-v4-flash-vision-vl-dspark-1 2>&1 | grep -q 'weightless GLP steering active'"; then
          echo "FATAL: WEIGHTLESS_STEER_PATH is set but 'weightless GLP steering active' is absent on $label." >&2
          exit 1
        fi
        echo "Steering active line present on $label."
      done
    fi
    echo "Running minimal OpenAI-compatible chat request..."
    curl -fsS --max-time 180 "$CHAT_URL" \
      -H "Content-Type: application/json" \
      -d '{"model":"'"$SERVED_MODEL_NAME"'","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":64,"temperature":0.0}'
    echo
    echo "Minimal chat request succeeded."
    exit 0
  fi
  sleep "$WAIT_SECONDS"
done

echo "Timed out waiting for VL API. Recent head logs:" >&2
docker logs --tail=150 deepseek-v4-flash-vision-vl-dspark-1 >&2 || true
echo "Recent worker logs:" >&2
ssh "$WORKER_HOST" "docker logs --tail=150 deepseek-v4-flash-vision-vl-dspark-1" >&2 || true
exit 1
