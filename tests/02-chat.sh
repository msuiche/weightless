#!/usr/bin/env bash
# 02-chat: a basic chat completion returns non-empty content.
# (Generous max_tokens: a thinking model can burn tokens on reasoning first.)
set -u
BASE="${WEIGHTLESS_BASE_URL:-http://localhost:8888/v1}"
MODEL="${WEIGHTLESS_MODEL:-deepseek-v4-flash-dspark}"

resp=$(curl -sf -m 180 -H 'Content-Type: application/json' -d '{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
  "max_tokens": 1024,
  "temperature": 0
}' "$BASE/chat/completions") || { echo "FAIL: chat/completions request failed"; exit 1; }

content=$(echo "$resp" | python3 -c '
import json, sys
d = json.load(sys.stdin)
m = d["choices"][0]["message"]
print((m.get("content") or "").strip())
' 2>/dev/null) || { echo "FAIL: unparseable response"; echo "$resp" | head -c 800; exit 1; }

if [ -z "$content" ]; then
  echo "FAIL: empty content"
  echo "$resp" | head -c 800
  exit 1
fi
echo "PASS: chat completion returned: ${content:0:80}"
