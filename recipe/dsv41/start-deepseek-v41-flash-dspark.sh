#!/usr/bin/env bash
# NOT DEPLOYABLE — fail-closed placeholder for the DSV4.1-Flash lane.
#
# DeepSeek-V4.1-Flash cannot be deployed on the rig today (2026-09-10):
#   1. vLLM support is vllm-project/vllm PR #56201 (branch dsv41-feat),
#      unmerged. No release image loads deepseek_v41.
#   2. The 510 GB FP8+MXFP4 checkpoint fits nothing on 2x DGX Spark
#      (256 GB total). No 2x128GB-fitting quant exists.
#
# This script exists so an automated flow (or a tired human) cannot
# half-deploy the lane: it prints the blocker and exits nonzero BEFORE
# touching docker, ssh, or any host. Replace it with a real recipe only
# after BOTH unblock conditions hold and the recipe is validated on the
# rig. The daily driver stays DSV4-0731 (recipe/anemll).
set -euo pipefail

cat >&2 <<'EOF'
[start-dsv41] NOT DEPLOYABLE — refusing to bring up the DSV4.1-Flash lane.
[start-dsv41]   blocker 1: vLLM #56201 (dsv41-feat) is unmerged.
[start-dsv41]   blocker 2: no 2x128GB-fitting quant of the 510 GB checkpoint.
[start-dsv41] The GLP-39 vector + validated Modal serving shape live in
[start-dsv41]   refusal-research/experiments/20260910-dsv41-flash-glp/.
[start-dsv41] See weightless/recipe/dsv41/README.md. Daily driver: DSV4-0731.
EOF
exit 1
