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

### Reproducibility, and cold start on the peak prompt

Re-measured after an unrelated restart, same config:

| prompt shape | first measurement | after restart |
|---|---:|---:|
| peak-finder | 77.3 tok/s, 99.7 %, MAL 5.99 | 65.8 tok/s, 97.4 %, MAL 5.87 |
| prose | 43.1 tok/s, 46.6 %, MAL 3.33 | 43.2 tok/s, 46.6 %, MAL 3.33 |

The prose row reproduces to within 0.1 tok/s and identical MAL, so the harness is
stable. The peak row is the one that moves, and it moves with warmth: the outgoing
README documents the same effect, quoting 78.4 tok/s warm against 56.8 tok/s
immediately after a cold start. Quote the peak figure only with its warm-up state
attached.

### Where spec decode does and does not help

At the Run 001 random-token shape, single stream: TPOT improved 38.2 -> 32.1 ms
(16 %) but TTFT rose 93 -> 726 ms, so aggregate throughput was flat at ~25.7
tok/s. The drafting overhead is paid on every step whether or not the draft is
accepted, and at 31 % acceptance it roughly cancels. On predictable output the
same machinery yields 3x. Enable it, but do not expect it to help uniformly.

---

## Run 003 — 1M declared context, and a correction

**2026-08-17.** Same as Run 002 with `--max-model-len 1048576`, restoring the
context length the previous stack ran at.

I predicted this would be a declaration rather than a capability, reasoning that
14.53 GiB of KV held 138,742 tokens at 65k, so one 1,048,576-token request would
need ~7.6x more than the whole cache. **That was wrong.**

| declared `max-model-len` | KV memory | GPU KV cache | max concurrency at full length |
|---:|---:|---:|---:|
| 65,536 | 14.53 GiB | 138,742 tokens | 2.12x |
| **1,048,576** | 13.54 GiB | **1,375,854 tokens** | **1.31x** |

Roughly the same memory yields **10x more KV tokens**. The error was assuming KV
cost per token is a constant. On DeepSeek V4 Flash it is not: sparse MLA with
C128 compression means the compression regime is chosen from `max_model_len`, so
declaring a longer context makes each cached token cheaper. The cache holds 1.31
full-length requests, so 1M is real here, not nominal.

`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` is baked into both the old and new images, so
vLLM would have started at 1M regardless of whether the KV could back it. That
flag is why the declared number cannot be trusted on its own and the
`Maximum concurrency` line has to be read.

**This invalidates the justification for the NVFP4 KV lever.** It was listed as
the binding constraint for reaching 1M; 1M is reached without it. NVFP4 KV is
still worth having for concurrency at a given context, but it is not what stood
between us and long context.

### The cost of declaring 1M: prefill collapses

Chasing the slow prompt gave the other half of the trade-off, and it is the more
important half.

| declared `max-model-len` | KV per 13.5 GiB | prefill |
|---:|---:|---:|
| 65,536 | 138,742 tokens | **2,495 tok/s** |
| 1,048,576 | 1,375,854 tokens | **110 tok/s** |

Measured directly: a 58,012-token prompt took **526.4 s**, i.e. 110 tok/s, about
**23x slower** than the same server at 65k. It was not a hang and not JIT, which
was my first guess: `num_requests_running` was 1 the whole time and the request
completed correctly. Confirming symptom, a trivial "Say OK" request submitted
alongside took **36.9 s**, because it queued behind the prefill.

The likely mechanism is the mirror of the KV win. Sparse MLA sizes its indexer
and compressor work from `max_model_len`, so declaring 1M makes every prefill
chunk pay 1M-sized cost regardless of the live prompt length, while
simultaneously choosing a compression regime that makes each cached token
cheaper. One number improves 10x and the other degrades 23x.

At 110 tok/s a genuine 1,048,576-token prompt needs **~2.6 hours** to prefill. So
1M is real in KV capacity and impractical in latency, and "we support 1M context"
needs both numbers attached or it is misleading.

#### The control, and the actual answer: 256k is free

The first comparison was confounded — 4k-token prefill at 65k declared against
58k-token prefill at 1M declared, so prompt length and declared context both
moved. Re-run with prompt length held constant at 58,008 tokens, and swept:

| declared `max-model-len` | GPU KV cache | conc. at full length | prefill, 58,008-tok prompt | trivial request |
|---:|---:|---:|---:|---:|
| 65,536 | 138,742 | 2.12x | 2,089 tok/s | 0.20 s |
| **262,144** | **492,549** | **1.88x** | **2,098 tok/s** | **0.20 s** |
| 1,048,576 | 1,375,854 | 1.31x | **110 tok/s** | 36.9 s |

The confound was worth checking and the effect survived it: at fixed prompt
length, 1M is **19x** slower to prefill than 65k. Prompt length itself barely
matters, 4k to 58k moving prefill only 2,495 -> 2,089 tok/s.

But the degradation is **not gradual, it is a cliff between 256k and 1M**. At
262,144 the context is 4x larger than 65k with 3.5x the KV tokens and *no
measurable prefill cost at all* (2,098 against 2,089 tok/s, within noise), and
decode is unaffected (peak-finder 77.5 tok/s at 99.7 % acceptance).

That is consistent with the base image being named
`v027-ngc2607-dsv4-0731-dspark-k7-**256k**-production`: 256k looks like the
configuration it was built and validated for, and 1M is outside it.

**Settled on `max-model-len 262144`.** 65,536 was needlessly small; 1,048,576 is
capacity-real and latency-impractical. Anything needing true 1M prompts should
expect ~2.6 h of prefill and be scheduled as batch work, not served
interactively.

---

## Prefill vs decode — the two numbers, kept apart

Same server, same config as Run 002 (v027, TP=2 on 2x DGX Spark, spec decode on,
`fp8_ds_mla` KV, 65k ctx, steering on, warm, concurrency 1).

| phase | how it is measured | result |
|---|---|---:|
| **prefill** | `prompt_tokens / median TTFT`, input 1024, output 1 | **1,522 tok/s** |
| **prefill** | `prompt_tokens / median TTFT`, input 4096, output 1 | **2,495 tok/s** |
| **decode** | output tok/s, random tokens (worst case for drafting) | 25.7 tok/s |
| **decode** | output tok/s, prose | 43.2 tok/s |
| **decode** | output tok/s, peak-finder (predictable) | 77.3 tok/s |

Prefill runs ~60x decode because it is compute-bound and processes the whole
prompt in parallel, while decode is memory-bandwidth-bound and strictly serial.
Prefill also improves with length (1,522 -> 2,495 tok/s from 1k to 4k) as the
GPU fills up. **The headline "77 tok/s" is decode.**

Do not quote `Total token throughput` from `vllm bench serve` as a speed: it is
`(input + output) / duration`, which blends the two phases and flatters the
result (443.6 tok/s for the same run that decodes at 49.1).

### How not to measure prefill

A first attempt gave 83, 896, 2,476, 24,800 and 1,612 tok/s across runs of the
same script — non-monotonic and unusable. Two causes, both mine:

- `--enable-prefix-caching` is on and the filler text was one sentence repeated,
  so later requests shared a long prefix with earlier ones. The 24,800 tok/s row
  is a cache hit, not a prefill.
- the first request of the process was cold, giving 83 tok/s.

Measuring via `--dataset-name random` fixes both: tokens are unique per request
so nothing is shared, and TTFT p99 lands within 5 % of the median (748 vs 673 ms
at 1k, 1697 vs 1642 ms at 4k), which is what a trustworthy measurement looks
like.

Note these TTFTs are far above Run 001's 93 ms at the same 1024 input, because
Run 001 had speculative decoding **off**. Turning it on raises TTFT (the draft
runs during prefill too) and lowers TPOT; see Run 002.

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
3. **Raise `max-num-seqs`** — attempted and **reverted**. `max-num-seqs=32` at
   `gpu-memory-utilization 0.80` with speculative decoding on **fails during
   warmup**: the worker dies in `compile_or_warm_up_model` with exit code None
   (killed, not an exception), and the engine core then aborts. The visible
   `UnicodeDecodeError` in `torch/library.py:_del_library` is torch's atexit
   handler failing while unwinding, not the cause.

   The headroom explains it: with the draft loaded, model weights take 79.54 GiB
   and vLLM reports **`Available KV cache memory: 13.43 GiB`**, so ~93 of the
   ~97 GiB that 0.80 of 121 GiB allows is already committed before the larger
   capture buffers 32 sequences require. Restored to `max-num-seqs=6`, which
   serves.

   Next attempt should move `gpu-memory-utilization` **down** to free capture
   headroom, or step `max-num-seqs` to 12 or 16, rather than jumping to 32.
   Each attempt costs a full reload (~4 min to `Application startup complete`).
4. **NVFP4 KV.** Still worth having: `fp8_ds_mla` roughly doubles bytes per token
   against the previous stack's `nvfp4_ds_mla`, so it buys concurrency at a fixed
   context. It is **not** the blocker for long context, contrary to what this list
   said before Run 003.
5. ~~Raise context to 1M.~~ **Done and reverted, Run 003.** KV holds 1.31
   full-length requests, but prefill drops to 110 tok/s (23x slower), so a full
   1M prompt would take ~2.6 hours. But the loss is a cliff between 256k and 1M:
   **262,144 costs nothing measurable** and is now the setting. Running at 262,144.

Each should be measured at the Run 001 shape (1024/128, concurrency 1 and 6,
warm) so the rows stay comparable.
