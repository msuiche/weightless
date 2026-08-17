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

## Run 002 — speculative decoding ON, same stack

**2026-08-17.** Identical to Run 001 except `--speculative-config
{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}`.
Steering still active on the same general direction at alpha 4.0, layers 10-38.

Cost of enabling it: model load 74.11 -> **79.54 GiB** and 172 -> **223 s**, and GPU
KV cache 182,410 -> **133,197 tokens**, because the draft stages need weights and
KV of their own. vLLM also warns that `max_num_scheduled_tokens` drops to 8,168.

### The finding that reframes Run 001

**Acceptance depends enormously on how predictable the output is, so the choice
of benchmark dataset dominates the result.** Same server, same config, three
prompt shapes:

| prompt shape | out tok/s | acceptance | MAL (tok/step) |
|---|---:|---:|---:|
| `random` tokens, 1024/128 (Run 001 shape) | 25.7 | **31.0 %** | 2.55 |
| prose: "explain TCP congestion control" | **43.1** | 46.6 % | 3.33 |
| peak-finder: "Count from 1 to 300, separated by commas." | **77.3** | **99.7 %** | **5.99** |

A drafter cannot predict random tokens, so `--dataset-name random` is close to a
worst case for any speculative stack and understates it by ~3x here. Run 001's
numbers are not wrong, but they measure the shape they were taken at and must not
be read as this stack's throughput.

### Against the previous stack

The outgoing README's peak-finder figure was **78.4 tok/s at 98.9 % acceptance,
5.95 accepted tokens per step out of 6**, with Patch 4 applied, at 1M context on
nvfp4 KV.

| | out tok/s | acceptance | MAL |
|---|---:|---:|---:|
| previous stack, Patch 4, 1M ctx, nvfp4 KV | 78.4 | 98.9 % | 5.95 |
| **this stack, no Patch 4, 65k ctx, fp8_ds_mla KV** | **77.3** | **99.7 %** | **5.99** |

Parity on the same prompt, within 1.4 %.

### Patch 4 is not needed on v0.27.0

Settled by measurement rather than by porting. Patch 4 existed because the DSpark
draft loader dropped `.shared_experts.w1/.w3`, leaving the draft's always-on
shared expert uninitialised and collapsing acceptance to 25.7 % at 2.28 tok/step.
Upstream v0.27.0 rewrote that loader: there is no
`_STACKED_PARAM_NAME_MAPPING` and no `map_dspark_stacked_param_name` to patch.

At 99.7 % acceptance and 5.99 of a maximum 6 accepted tokens per step, the draft
is accepting essentially everything it proposes, which is not possible with an
uninitialised always-on expert. The rewrite fixed it. `DSpark draft model loaded:
96 params`.

Also worth noting: steering was active throughout, so a rank-1 projection on 29
layers does not measurably damage draft acceptance.

### Where spec decode does and does not help

At the Run 001 random-token shape, single stream: TPOT improved 38.2 -> 32.1 ms
(16 %) but TTFT rose 93 -> 726 ms, so aggregate throughput was flat at ~25.7
tok/s. The drafting overhead is paid on every step whether or not the draft is
accepted, and at 31 % acceptance it roughly cancels. On predictable output the
same machinery yields 3x. Enable it, but do not expect it to help uniformly.

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

1. ~~Enable speculative decoding.~~ **Done, Run 002.** 99.7 % acceptance on
   predictable output, 77.3 tok/s, parity with the previous stack. Patch 4 not
   needed.
2. ~~Re-enable torch.compile.~~ **Dead end, settled by reading v0.27.0.**
   `vllm/config/vllm.py:1211-1234` auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH`
   for an explicit list of architectures including `DeepseekV4ForCausalLM` and
   `DeepSeekV4MTPModel`, with the comment: *"For model classes don't carry
   @support_torch_compile — the breakable cudagraph is the supported PIECEWISE
   path."* There are **zero** occurrences of `support_torch_compile` anywhere
   under `vllm/models/deepseek_v4/`, against e.g. `qwen3.py` which has it. So
   torch.compile is not accidentally disabled, it is unsupported for this
   architecture. Setting the env var to 0 opts out of the *supported* cudagraph
   path rather than enabling compilation, and should be expected to hurt.
3. **Raise `max-num-seqs` and the cudagraph capture sizes.** 6 and 8 are the
   binding constraints on the concurrency scaling measured above.
4. **NVFP4 KV.** `fp8_ds_mla` roughly doubles bytes per token against the
   previous stack's `nvfp4_ds_mla`, which is what limits KV to 182,410 tokens and
   concurrency to 2.78x at 65k. This is the binding constraint for a 1M target.
5. **Raise context to 1M** once the above are settled.

Each should be measured at the Run 001 shape (1024/128, concurrency 1 and 6,
warm) so the rows stay comparable.
