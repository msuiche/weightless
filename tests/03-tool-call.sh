#!/usr/bin/env bash
# 03-tool-call: the model emits a well-formed tool call — the capability omp's
# whole agent loop depends on.
set -u
BASE="${WEIGHTLESS_BASE_URL:-http://localhost:8888/v1}"
MODEL="${WEIGHTLESS_MODEL:-deepseek-v4-flash-dspark}"

resp=$(curl -sf -m 180 -H 'Content-Type: application/json' -d '{
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
}' "$BASE/chat/completions") || { echo "FAIL: tool-call request failed"; exit 1; }

echo "$resp" | python3 -c '
import json, sys
d = json.load(sys.stdin)
m = d["choices"][0]["message"]
calls = m.get("tool_calls") or []
assert calls, "no tool_calls in response: %s" % json.dumps(m)[:400]
fn = calls[0]["function"]
assert fn["name"] == "get_weather", "unexpected tool: %s" % fn["name"]
args = json.loads(fn["arguments"])
assert "city" in args, "bad arguments: %s" % fn["arguments"]
print("PASS: tool call get_weather(%s)" % json.dumps(args))
' || { echo "FAIL: tool-call assertion failed"; echo "$resp" | head -c 800; exit 1; }
