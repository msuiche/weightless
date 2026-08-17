# Benchmarks

A throughput number without its configuration is noise. Every row here carries
the shape it was measured at, and single-stream is reported separately from
aggregate because they are different quantities: at concurrency 6 this stack
delivers ~1.9x the aggregate tokens of concurrency 1 while each individual stream
runs 2.9x slower.

Append new runs; do not overwrite. Warm and cold are recorded separately, because
the difference is large enough to invert conclusions.

---

## Run 001 — v027 bring-up, steering on, no speculative decoding

**2026-08-17.** First measurement on vLLM v0.27.x. This is a deliberately
conservative bring-up configuration, not a tuned one.

### Hardware

| | |
|---|---|
| nodes | **2x NVIDIA DGX Spark** (GB10, SM121a), 121 GiB unified memory each |
| interconnect | 2x 200 Gb/s RoCE (ConnectX); NCCL over `enp1s0f0np0`, `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0` |
| host driver | 580.173.02, container runs CUDA 13.3 via forward compatibility (610.43.02) |

### Model and runtime

| | |
|---|---|
| model | `deepseek-ai/DeepSeek-V4-Flash-0731` (official release) |
| served as | `deepseek-v4-flash-dspark` |
| quantisation | `deepseek_v4_fp8` (weights), dtype bfloat16 |
| runtime | vLLM `0.27.1.dev0+g4bdc8a788.d20260813.cu133` |
| image | `vllm-dspark-steering:v027-gguf` (bjk110 v027 + `gguf==0.19.0`) |
| parallelism | TP=2, PP=1, `--nnodes 2`, executor `mp` |
| KV cache dtype | **`fp8_ds_mla`** (not NVFP4: `nvfp4_ds_mla` does not exist upstream in v0.27) |
| block size | 256 |
| max model len | **65,536** (not 1M) |
| max num seqs | **6** |
| max batched tokens | 8,192 |
| gpu mem utilisation | 0.80 |
| prefix caching | on |
| chunked prefill | on |
| async scheduling | on |
| **speculative decoding** | **OFF** |
| **torch.compile** | **OFF** — `VLLM_USE_BREAKABLE_CUDAGRAPH=1` auto-enables and forces `-cc.mode=none` |
| cudagraph capture | sizes `[1,2,4,8]`, max 8 |

Resulting capacity: model load **74.11 GiB / 172.2 s**, GPU KV cache **182,410
tokens**, maximum concurrency **2.78x** for 65,536-token requests.

### Steering

| | |
|---|---|
| state | **active** |
| artifact | `DeepSeek-V4-Flash-0731-general-abliterated-cvec-L10-38-a4-keysdir.gguf` |
| provenance | general/broad direction, recovered from Keys' published weights by SVD (third party, not ours) |
| loader | our GGUF reader extension; `mode=project` enforced, `spec_version=1`, layer ids cross-checked |
| hook point | `post_layer` |
| alpha | 4.0 |
| layers | 29 directions, layers 10–38, `n_embd=4096`, rank 1 per layer |

### Method

`vllm bench serve`, run inside the serving container against the live endpoint.

```
--backend openai-chat --endpoint /v1/chat/completions
--dataset-name random --random-input-len 1024 --random-output-len 128
--ignore-eos --seed 0
--tokenizer deepseek-ai/DeepSeek-V4-Flash-0731 --trust-remote-code
```

`--tokenizer` is required: passing the served name to `--model` alone makes the
harness try to resolve `deepseek-v4-flash-dspark` as a HuggingFace repo id.

### Results

1024 input / 128 output tokens, greedy, `--ignore-eos`.

| run | conc. | warm | output tok/s | peak | total tok/s | TTFT median | TTFT p99 | TPOT median | per-stream tok/s |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| single stream | 1 | yes | **25.88** | — | 233.71 | 93 ms | — | 38.2 ms | **26.2** |
| aggregate | 6 | yes | **49.12** | 65.00 | 443.60 | 243 ms | 6884 ms | 109.1 ms | 9.2 |
| aggregate | 6 | **no** | 39.09 | 60.00 | 353.03 | 2797 ms | 15307 ms | 108.3 ms | 9.2 |

Cold start dominates TTFT and nothing else: between the cold and warm
concurrency-6 rows, median TTFT falls 2797 → 243 ms while TPOT is unchanged at
~109 ms. The first benchmark iteration alone took 29.3 s against a 3.27 s
steady-state average. **Always warm before measuring**, and never quote a cold
TTFT.

Concurrency scaling is poor: 1 → 6 gives 1.9x aggregate throughput while
per-stream drops 2.9x (26.2 → 9.2 tok/s). Consistent with `max-num-seqs 6` and a
cudagraph capture ceiling of 8.

---

## Reference: the previous stack, for context

From the outgoing repo's README, measured on **2x DGX Spark, TP=2, k=5, nvfp4 KV,
1M context** — i.e. a different KV dtype, a different context length, and
**speculative decoding on**. Not directly comparable to Run 001, which has spec
decode off.

| | accept | tok/step | steps/s | mean tok/s | peak tok/s |
|---|---|---|---|---|---|
| 0731, stock draft loader | 25.7% | 2.28 | 14.4 | 32.7 | 42.0 |
| 0731, with Patch 4 | 60.2% | 4.01 | 13.8 | **55.4** | **66.1** |

The decomposition matters more than the totals. `steps/s` is ~14 in both rows, so
Patch 4's entire gain is draft acceptance (`tok/step` 2.28 → 4.01), not engine
speed. Patch 4 added two rows to the DSpark draft loader's stacked-parameter
mapping so `.shared_experts.w1/.w3` resolve to `gate_up_proj`; without them the
draft's always-on shared expert loaded uninitialised, and the unmapped weights
were dropped through a `logger.debug` path that is silent at INFO.

Placing Run 001 against that, per stream and warm:

```
previous stack, Patch 4   55.4 tok/s = 13.8 steps/s x 4.01 tok/step
Run 001                   25.9 tok/s = 25.9 steps/s x 1.00 tok/step
```

So this stack's raw step rate is roughly **1.9x** the previous one, and the
deficit is entirely the missing speculative-decode multiplier. That reframes the
gap: it is not that v027 is slow, it is that we are running it with the draft
head switched off.

**Patch 4 does not apply to v0.27.0.** Upstream contains no
`_STACKED_PARAM_NAME_MAPPING` and no `map_dspark_stacked_param_name`; the draft
loader was rewritten. Whether the same class of bug survived that rewrite is an
empirical question, and acceptance rate answers it directly: ~60% means the
loader is fine, ~25% means it is not. Measure, do not port.

---

## Open levers, in expected-payoff order

1. **Enable speculative decoding.** `--speculative-config` with method `dspark`,
   `num_speculative_tokens=5`. Report acceptance and `tok/step` alongside tok/s,
   since that is what diagnoses Patch 4's bug class on the new loader.
2. **Re-enable torch.compile.** `VLLM_USE_BREAKABLE_CUDAGRAPH=1` is auto-enabled
   and disables it. Find out whether that is protecting against a real GB10
   cudagraph failure or is merely conservative.
3. **Raise `max-num-seqs` and the cudagraph capture sizes.** 6 and 8 are the
   binding constraints on the concurrency scaling measured above.
4. **NVFP4 KV.** `fp8_ds_mla` roughly doubles bytes per token against the
   previous stack's `nvfp4_ds_mla`, which is what limits KV to 182,410 tokens and
   concurrency to 2.78x at 65k. This is the binding constraint for a 1M target.
5. **Raise context to 1M** once the above are settled.

Each should be measured at the Run 001 shape (1024/128, concurrency 1 and 6,
warm) so the rows stay comparable.
