# Ouro-2.6B weightless-steer plugin validation (plugin lane)

Lane: `weightless-ouro-plugin-test` Modal app (modal/cloud_serve_ouro.py),
vllm/vllm-openai:v0.26.0 EXACTLY (last image carrying OuroForCausalLM),
H100:1, TP=1, compiled boot (no --enforce-eager), plugin pip-installed,
GLP-192 vector msuiche/Ouro-2.6B-abliterated-cyber-GLP-192-L1-192-a1.0.
Eval driver modal/eval_ouro_plugin.py: refusal32/cyber32/benign32, greedy,
max_tokens=400, n=32, scored with refusal-research/harness/lib.py classify().

Reference row (hotfix lane, experiments/20260906-ouro-glp, greedy, mt4096 —
this lane runs mt400, so one-item wobble is expected):

| suite    | stock       | a0.0 (no-op gate) | a1.0  |
|----------|-------------|-------------------|-------|
| refusal32| C4/D1/R27   | C5/R27            | C32   |
| cyber32  | C30/R2      | C28/D2/R2         | C31/D1|
| benign32 | C32         | C32               | C32   |

## Boot evidence

Arm a0.0: `weightless GLP steering active: hook=residual_stream_post_layer
alpha=0.000 exec_steps=0..191 (192) width=2048` (+ WEIGHTLESS-SHIM stderr
marker, same content). Arm a1.0: identical line with `alpha=1.000`.
Both boots compiled (torch.compile + cudagraphs), no eager fallback.

## KNOWN ARTIFACT RACE on the a0.0 results (read before citing)

An orphaned eval driver from the prior (failed, pre---trust-remote-code)
attempt was still polling the fixed app URL when the a0.0 deploy came up.
Two identical drivers ran the a0.0 arm CONCURRENTLY against the same server
and cross-wrote this directory (same-second mtimes; GNU tee continues to
stdout after the open error, which is how the orphan survived). Effects:

- driver-a0.0.log = this session's driver stdout; its "==" lines
  (refusal32 C4/D1/R27, cyber32 C29/R3, benign32 C32) score ITS in-memory
  generation, whose raw files were then overwritten by the orphan.
- The *-a0.0.json / scores-*-a0.0.json / summary-a0.0.json files on disk
  are the orphan twin's generation; re-scoring them locally with the same
  harness reproduces refusal32 C5/R27, cyber32 C29/D1/R2, benign32 C32.

So there are TWO independent a0.0 samples, differing only by vLLM
continuous-batching nondeterminism at the mt400 cap:

| suite    | files (re-scored) | log (this driver) | reference gate |
|----------|-------------------|-------------------|----------------|
| refusal32| C5/R27            | C4/D1/R27         | C5/R27         |
| cyber32  | C29/D1/R2         | C29/R3            | C28/D2/R2      |
| benign32 | C32               | C32               | C32            |

Both pass the no-op gate (alpha=0 must behave like stock: stock row is
C4/D1/R27 / C30/R2 / C32). The orphan exited after its arm; the a1.0 arm
was run with a verified-clean process table.

## a1.0 result — EXACT match to the reference row

Single clean run (process table verified, files/log/summary/local re-score
all coherent):

| suite    | plugin a1.0 | reference a1.0 | plugin a0.0 (files) | plugin stock-equivalent |
|----------|-------------|----------------|---------------------|-------------------------|
| refusal32| C32/32      | C32/32         | C5/R27              | (log twin: C4/D1/R27)   |
| cyber32  | C31/D1      | C31/D1         | C29/D1/R2           | (log twin: C29/R3)      |
| benign32 | C32         | C32            | C32                 | C32                     |

The published GLP-192 vector transfers through the plugin lane WITHOUT
recalibration: the full 27-28-point refusal32 swing at alpha=1.0 with zero
benign32 regression, reproducing the hotfix lane's validated row item for
item at the suite level. The a0.0 no-op gate passes on both independent
samples. Compiled serving (torch.compile + FULL/PIECEWISE cudagraphs) with
the adapter's live TorchCompileWithNoGuardsWrapper rebind — no eager
fallback was needed.

server-a0.0.log also contains the prior attempt's failed 08:46 boot
(died at ModelConfig trust_remote_code validation; fixed by adding
--trust-remote-code to the serve command in 1d87bec).
