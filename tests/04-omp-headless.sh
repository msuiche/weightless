#!/usr/bin/env bash
# 04-omp-headless: full agent loop — omp (non-interactive) drives our endpoint
# to create a file in a scratch dir. Exercises tool schemas, streaming, and
# the edit/write path end to end.
set -u
SELECTOR="${DSPARK_OMP_MODEL:-dspark/deepseek-v4-flash-dspark}"

if ! command -v omp >/dev/null 2>&1; then
  echo "SKIP: omp not installed (curl -fsSL https://omp.sh/install | sh)"
  exit 2
fi
if ! grep -q "^  dspark:" "$HOME/.omp/agent/models.yml" 2>/dev/null; then
  echo "FAIL: dspark provider missing from ~/.omp/agent/models.yml — run tests/install.sh first"
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

( cd "$tmp" && omp -p --no-session --no-title --no-lsp --no-extensions --no-skills --no-rules \
    --model "$SELECTOR" --auto-approve --approval-mode yolo --max-time=240 \
    "Create a file named omp_probe.txt whose entire content is the single line: omp-ok. Then stop." )

if [ ! -f "$tmp/omp_probe.txt" ]; then
  echo "FAIL: agent did not create omp_probe.txt"
  exit 1
fi
if ! grep -qx "omp-ok" "$tmp/omp_probe.txt"; then
  echo "FAIL: unexpected content in omp_probe.txt:"
  cat "$tmp/omp_probe.txt"
  exit 1
fi
echo "PASS: omp agent loop created omp_probe.txt via $SELECTOR"
