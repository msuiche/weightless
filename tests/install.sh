#!/usr/bin/env bash
# install.sh — install (or merge) the dspark provider into ~/.omp/agent/models.yml
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

# register the endpoint as omp's default model (merge into config.yml,
# preserving other keys and sibling roles)
cfg="$HOME/.omp/agent/config.yml"
ref="dspark/${WEIGHTLESS_MODEL:-deepseek-v4-flash-dspark}"
if [ -f "$cfg" ] && grep -A5 "^modelRoles:" "$cfg" | grep -q "default: $ref"; then
  echo "omp default model already '$ref'"
else
  [ -f "$cfg" ] && cp "$cfg" "$cfg.bak.$(date +%s)"
  touch "$cfg"
  tmp="$cfg.tmp.$$"
  awk -v ref="$ref" '
    /^modelRoles:/ { inmr=1; seenblock=1; print; next }
    inmr && /^[^ \t]/ { if (!donedef) { print "  default: " ref; donedef=1 }; inmr=0 }
    inmr && /^[ \t]+default:/ { if (!donedef) { print "  default: " ref; donedef=1 } next }
    { print }
    END {
      if (seenblock && inmr && !donedef) print "  default: " ref
      if (!seenblock) print "modelRoles:\n  default: " ref
    }
  ' "$cfg" > "$tmp" && mv "$tmp" "$cfg"
  echo "omp default model -> '$ref' ($cfg)"
fi
