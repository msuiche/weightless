#!/usr/bin/env bash
# 01-endpoint: the deployed endpoint answers /v1/models and lists our model.
set -u
BASE="${WEIGHTLESS_BASE_URL:-http://localhost:8888/v1}"
MODEL="${WEIGHTLESS_MODEL:-deepseek-v4-flash-dspark}"

resp=$(curl -sf -m 15 "$BASE/models") || { echo "FAIL: $BASE/models unreachable"; exit 1; }
if ! printf '%s\n' "$resp" | python3 -c '
import json, sys
data = json.load(sys.stdin)
sys.exit(0 if any(m.get("id") == sys.argv[1] for m in data.get("data", [])) else 1)
' "$MODEL"; then
  echo "FAIL: model $MODEL not listed at $BASE"
  echo "$resp" | head -c 400
  exit 1
fi
echo "PASS: $MODEL listed at $BASE"
