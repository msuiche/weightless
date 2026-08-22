#!/usr/bin/env bash
# 03-tool-call: the model emits a well-formed tool call — the capability omp's
# whole agent loop depends on.
set -u
BASE="${WEIGHTLESS_BASE_URL:-http://localhost:8888/v1}"
MODEL="${WEIGHTLESS_MODEL:-deepseek-v4-flash-dspark}"

payload='{
  "model": "'"$MODEL"'",
  "messages": [{"role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather in a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"]
      }
    }
  }],
  "max_tokens": 1024,
  "temperature": 0
}'

# Known server-side flake (documented in recipe/anemll/README.md): the image
# occasionally emits raw control characters inside JSON strings on this path.
# One retry absorbs the flake; a second failure is a real regression and fails
# with the byte offset + context.
attempt=1
while :; do
  resp=$(curl -sf -m 180 -H 'Content-Type: application/json' -d "$payload" "$BASE/chat/completions") \
    || { echo "FAIL: tool-call request failed"; exit 1; }
  if ! echo "$resp" | python3 -c 'import json,sys; json.loads(sys.stdin.read())' 2>/dev/null; then
    if [ "$attempt" -eq 1 ]; then
      echo "note: invalid JSON from server (serializer flake) — retrying once"
      attempt=2
      continue
    fi
    echo "$resp" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    json.loads(raw)
except json.JSONDecodeError as e:
    print(f"invalid JSON persisted across retry, byte {e.pos}: {raw[max(0, e.pos-40):e.pos+20]!r}")
'
    echo "FAIL: server emitted invalid JSON twice"
    exit 1
  fi
  break
done

echo "$resp" | python3 -c '
import json, sys
d = json.loads(sys.stdin.read())
m = d["choices"][0]["message"]
calls = m.get("tool_calls") or []
assert calls, "no tool_calls in response: %s" % json.dumps(m)[:400]
fn = calls[0]["function"]
assert fn["name"] == "get_weather", "unexpected tool: %s" % fn["name"]
args = json.loads(fn["arguments"])
assert "city" in args, "bad arguments: %s" % fn["arguments"]
print("PASS: tool call get_weather(%s)" % json.dumps(args))
' || { echo "FAIL: tool-call assertion failed"; echo "$resp" | head -c 800; exit 1; }
