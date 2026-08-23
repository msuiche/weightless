#!/usr/bin/env bash
# install.sh — install (or merge) the dspark provider into ~/.omp/agent/models.yml
# and register it in omp's modelRoles (config.yml).
# Env: WEIGHTLESS_MODEL (default deepseek-v4-flash-dspark),
#      WEIGHTLESS_OMP_ALL_ROLES=1 routes every text role (smol, slow, plan,
#      task, commit, tiny, advisor, designer) — not just default. vision is
#      never touched: the endpoint serves text-only models.
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)/models.yml"
dst="$HOME/.omp/agent/models.yml"
mkdir -p "$(dirname "$dst")"

if [ -f "$dst" ]; then
  if grep -q "^  dspark:" "$dst"; then
    echo "dspark provider already present in $dst"
  else
    cp "$dst" "$dst.bak.$(date +%s)"
    tail -n +2 "$src" >> "$dst"  # drop our 'providers:' header, append the block
    echo "merged dspark provider into $dst (backup: $dst.bak.*)"
  fi
else
  cp "$src" "$dst"
  echo "installed $dst"
fi

# register the endpoint in omp's modelRoles (merge into config.yml,
# preserving other keys and sibling roles)
cfg="$HOME/.omp/agent/config.yml"
ref="dspark/${WEIGHTLESS_MODEL:-deepseek-v4-flash-dspark}"
roles="default"
if [ "${WEIGHTLESS_OMP_ALL_ROLES:-0}" = "1" ]; then
  roles="default smol slow plan task commit tiny advisor designer"
fi

[ -f "$cfg" ] && cp "$cfg" "$cfg.bak.$(date +%s)"
touch "$cfg"
for role in $roles; do
  tmp="$cfg.tmp.$$"
  awk -v ref="$ref" -v role="$role" '
    /^modelRoles:/ { inmr=1; seenblock=1; print; next }
    inmr && /^[^ \t]/ { if (!donedef) { print "  " role ": " ref; donedef=1 }; inmr=0 }
    inmr && $1 == role":" { if (!donedef) { print "  " role ": " ref; donedef=1 } next }
    { print }
    END {
      if (seenblock && inmr && !donedef) print "  " role ": " ref
      if (!seenblock) print "modelRoles:\n  " role ": " ref
    }
  ' "$cfg" > "$tmp" && mv "$tmp" "$cfg"
done
echo "omp roles [$roles] -> '$ref' ($cfg)"
