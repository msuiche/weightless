#!/usr/bin/env bash
# run.sh — run all numbered tests, report a summary.
set -u
cd "$(dirname "$0")"

pass=0; fail=0; skip=0
for t in 0*.sh; do
  echo "== $t"
  bash "$t"
  rc=$?
  case $rc in
    0) pass=$((pass+1));;
    2) skip=$((skip+1));;
    *) fail=$((fail+1));;
  esac
done
echo "== pass=$pass fail=$fail skip=$skip"
[ "$fail" -eq 0 ]
