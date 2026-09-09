#!/usr/bin/env bash
# install.sh — install (or merge) the weightless provider (all local DGX
# lanes, per-model baseUrl per lane port) into ~/.omp/agent/models.yml
# and register a lane in omp's modelRoles (config.yml).
# Env: WEIGHTLESS_MODEL (default deepseek-v4-flash-dspark),
#      WEIGHTLESS_OMP_ALL_ROLES=1 routes every text role (smol, slow, plan,
#      task, commit, tiny, advisor, designer) — not just default. vision is
#      never touched: the endpoint serves text-only models.
set -euo pipefail

src="$(cd "$(dirname "$0")" && pwd)/models.yml"
dst="$HOME/.omp/agent/models.yml"
mkdir -p "$(dirname "$dst")"

if [ -f "$dst" ]; then
  merged=""
  for prov in weightless; do
    if grep -q "^  ${prov}:" "$dst"; then
      echo "${prov} provider already present in $dst"
    else
      [ -n "$merged" ] || { cp "$dst" "$dst.bak.$(date +%s)"; merged=1; }
      awk -v p="${prov}" '
        $0 == "  " p ":" { inb=1; print; next }
        inb && /^  [^ ]/ { inb=0 }
        inb { print }
      ' "$src" >> "$dst"
      echo "merged ${prov} provider into $dst"
    fi
  done
else
  cp "$src" "$dst"
  echo "installed $dst"
fi

# register the endpoint in omp's modelRoles (merge into config.yml,
# preserving other keys and sibling roles)
cfg="$HOME/.omp/agent/config.yml"
ref="weightless/${WEIGHTLESS_MODEL:-deepseek-v4-flash-dspark}"
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
