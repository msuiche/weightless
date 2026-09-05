#!/usr/bin/env bash
# memory-watchdog-gpu v3 — GB10 unified-memory guard for the vLLM window.
# v2 false-positive (2026-09-05): killed a HEALTHY Inkling boot 2 s into a
# transient MemAvailable dip (8182MB) during KV alloc / graph capture.
# v3 distinguishes wedge (low memory AND no log progress) from a healthy
# transient (low memory but logs moving):
#   avail <= THRESH and container log stale > STALE_SECS  -> kill fast (NEED=2)
#   avail <= THRESH but log fresh                          -> allow FRESH_NEED s,
#                                                             then kill anyway
#   (draining with progress for 15 s straight is not a transient)
set -u
THRESH_MB="${GPU_WATCHDOG_THRESH_MB:-8192}"
NEED="${GPU_WATCHDOG_NEED:-2}"
FRESH_NEED="${GPU_WATCHDOG_FRESH_NEED:-15}"
STALE_SECS="${GPU_WATCHDOG_STALE_SECS:-60}"
CONTAINER="${CONTAINER_NAME:-inkling-sm121}"
LOG="${GPU_WATCHDOG_LOG:-$HOME/memory-watchdog-gpu.log}"
hits=0
echo "$(date -u +%FT%TZ) watchdog-v3 start thresh=${THRESH_MB}MB need=${NEED} fresh_need=${FRESH_NEED} stale=${STALE_SECS}s container=${CONTAINER}" >> "$LOG"
log_fresh() {
  # /var/lib/docker is root-only on the rig; ask the daemon for recent lines
  # instead of stating the json.log file.
  [ -n "$(docker logs --since "${STALE_SECS}s" --tail 1 "$CONTAINER" 2>/dev/null)" ]
}
kill_stack() {
  local why="$1"
  echo "$(date -u +%FT%TZ) $why — killing $CONTAINER" >> "$LOG"
  docker kill "$CONTAINER" >> "$LOG" 2>&1
  pkill -9 -f 'VLLM::Worker' 2>/dev/null
  pkill -9 -f 'vllm serve' 2>/dev/null
  sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null 2>&1
  echo "$(date -u +%FT%TZ) post-kill MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)MB" >> "$LOG"
}
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -le "$THRESH_MB" ]; then
    hits=$((hits+1))
    if log_fresh; then
      [ "$hits" -ge "$FRESH_NEED" ] && { kill_stack "MemAvailable ${avail}MB <= ${THRESH_MB}MB x${hits} (logs fresh — active drain)"; hits=0; sleep 30; }
    else
      [ "$hits" -ge "$NEED" ] && { kill_stack "MemAvailable ${avail}MB <= ${THRESH_MB}MB x${hits} (logs stale >${STALE_SECS}s — wedge)"; hits=0; sleep 30; }
    fi
  else
    hits=0
  fi
  sleep 1
done
