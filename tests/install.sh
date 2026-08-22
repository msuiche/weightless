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
