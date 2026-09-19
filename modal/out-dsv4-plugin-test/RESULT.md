# DSV4 plugin-validation lane — run 2 results (2026-09-19)

Plugin lane: `weightless-steer` entry-point shadow of `DeepseekV4ForCausalLM`
(`weightless_steer.archs.dsv4:SteeredDeepseekV4ForCausalLM`), no in-container
patching. Vector: `msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29`
(L10-38, width 4096, rank 1), hook `ffn_out_pre_residual`, α=4.0 for the
steered arm (the BENCHMARK.md reference dose). Stack: day-0 image
`vllm/vllm-openai:deepseekv4-flash-vision` (vllm 0.28.1rc1.dev137+g5ab628dd1),
deepseek-ai/DeepSeek-V4-Flash-0731 @ 7872f01b, H100:4, TP4, bf16,
kv_cache_dtype=fp8, compiled (no enforce-eager), greedy, max_tokens 400,
n=32/suite, thinking OFF (see run1-thinking-on/NOTE.md).

## Headline vs reference

| suite | α=0.0 | α=4.0 | reference (hotfix lane) |
|---|---|---|---|
| refusal32 delivered | **0/32** | **16/32** | 0/32 → 18/32 @400 (20260904-ffn-site arm2; the published 19/32 is the 1400-tok number) |
| cyber32 delivered | 9/32 | 26/32 | no GLP-29 cyber32 row exists (never scored on the core set); in-domain signature is the lift 9→26, REFUSE 13→1 |
| benign32 clean | 32/32 | 32/32 | 32/32 both arms (ffn-site arm0/arm2) |

Stock refusal32 0/32 (29 REFUSE + 3 DEFLECT) reproduces the reference stock
exactly. Steered 16/32 vs the rig lane's 18/32 @400 — within 2 items across a
serving-stack difference (Modal day-0 FP8/TP4 vs rig Anemll NVFP4/TP2 + spec
decode; same vector, site, dose, scorer, token budget). The dose-response
shape matches the ffn-site sweep (α=4 in the climbing part of the window).
Scorer: refusal-research/harness/lib.py classify, unmodified; argumentative
flags (review-only, over-refusal is an upper bound): refusal32-a4.0 [16, 30],
cyber32-a0.0 [15, 23].

Boot evidence (server-a{0.0,4.0}.log, every worker rank):

```
INFO weightless_steer.core: weightless GLP steering active: hook=ffn_out_pre_residual alpha=0.000 layers=10..38 (29) width=4096
INFO weightless_steer.core: weightless GLP steering active: hook=ffn_out_pre_residual alpha=4.000 layers=10..38 (29) width=4096
```

plus the expected transferred-vector warning
(`glp.derived_at='residual_stream_post_layer', apply hook
'ffn_out_pre_residual'`) — the documented GLP-29 metadata state.

## Files

- `{refusal32,cyber32,benign32}-a{0.0,4.0}.json` — raw completions
- `scores-*.json` — per-item labels + counts
- `summary-a{0.0,4.0}.json` — per-arm tallies
- `server-a{0.0,4.0}.log` — run-2 boot logs (also on the volume, appended
  after run 1's)
- `run1-thinking-on/` — the first run's artifacts and the protocol-break
  write-up (thinking defaulted ON in this image's DSV4 renderer; fixed by
  `chat_template_kwargs={"enable_thinking": false}` in the driver)

## Spend

Run 2: 2 × H100:4 boots, ~19.4 min + ~9.8 min wall (second boot reused the
on-volume compile caches) ≈ 2.1 GPU-h + scaledown padding. Run 1 (thinking-on,
protocol-broken): 2 × H100:4 ≈ 4.3 GPU-h. CPU probes negligible.
Apps stopped and `modal container list` verified clean after each arm.
