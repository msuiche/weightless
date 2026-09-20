#!/bin/bash
# memory-watchdog — keeps GB10 unified-memory boxes from wedging under
# page-cache + GPU-load pressure. vLLM/docker are NEVER killed; discretionary
# download/sync processes are the sacrificial layer. Log: ~/memory-watchdog.log
# MEMORY_WATCHDOG_ALLOW_RE (default: match nothing) exempts args patterns
# from the discretionary kill — use for a bounded, supervised download.
set -u
LOG=/home/msuiche/memory-watchdog.log
LOW_MB=8192        # stage 1: drop caches below this
CRIT_MB=6144       # stage 2: kill discretionary consumers below this
PROTECT_RE='(vllm|VLLM|docker|containerd|systemd|sshd|Xorg|gnome|kwin|pipewire|NetworkManager|earlyoom)'
KILL_RE='(hf |huggingface|hf_transfer|rsync|aria2c|wget|curl .*download|snapshot_download)'
ALLOW_RE="${MEMORY_WATCHDOG_ALLOW_RE:-^$}"

avail() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }

while sleep 2; do
  A=$(avail)
  if [ "$A" -lt "$LOW_MB" ]; then
    echo "$(date '+%F %T') LOW ${A}MB -> drop_caches" >> "$LOG"
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
    sleep 3
    A=$(avail)
    if [ "$A" -lt "$CRIT_MB" ]; then
      # top discretionary consumer by RSS, excluding protected and allowed
      read -r PID RSS CMD < <(ps -eo pid=,rss=,args= --sort=-rss \
        | awk -v prot="$PROTECT_RE" -v killre="$KILL_RE" -v allow="$ALLOW_RE" \
          '$0 ~ killre && $0 !~ prot && $0 !~ allow {print $1, $2, $3; exit}')
      if [ -n "${PID:-}" ]; then
        echo "$(date '+%F %T') CRIT ${A}MB -> SIGTERM pid=$PID rss=${RSS}KB cmd=$CMD" >> "$LOG"
        kill -TERM "$PID" 2>/dev/null
        sleep 5
        if kill -0 "$PID" 2>/dev/null; then
          echo "$(date '+%F %T') still alive -> SIGKILL pid=$PID" >> "$LOG"
          kill -KILL "$PID" 2>/dev/null
        fi
      else
        echo "$(date '+%F %T') CRIT ${A}MB but no discretionary target; holding" >> "$LOG"
        sleep 10
      fi
    fi
  fi
done
