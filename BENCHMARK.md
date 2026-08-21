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
| 262,144 | 492,549 | 1.88x | 2,098 tok/s | 0.20 s |
| **524,288** | **883,902** | **1.69x** | **2,089 tok/s** | **0.16 s** |
| 1,048,576 | 1,375,854 | 1.31x | **110 tok/s** | 36.9 s |

The confound was worth checking and the effect survived it: at fixed prompt
length, 1M is **19x** slower to prefill than 65k. Prompt length itself barely
matters: at 262,144 declared, prefill is flat across an order of magnitude of
prompt size — **2,495 tok/s at 4k, 2,098 at 58k, 2,087 at 101k**. Whatever the 1M
setting does, it is not a function of how long the prompt actually is.

But the degradation is **not gradual, it is a cliff between 256k and 1M**. At
262,144 the context is 4x larger than 65k with 3.5x the KV tokens and *no
measurable prefill cost at all* (2,098 against 2,089 tok/s, within noise), and
decode is unaffected (peak-finder 77.5 tok/s at 99.7 % acceptance).

**512K is free as well**, which narrows the cliff considerably: 524,288 matches
262,144 on every measure — prefill 2,089 against 2,098 tok/s, trivial request
0.16 against 0.20 s, decode 77.9 tok/s at 99.7 % acceptance — while holding
883,902 KV tokens. So the collapse is specific to 1,048,576 and not a gradient
across the range.

That it is a threshold rather than a slope is itself informative. If the cost
tracked the compressor's topk width, which scales with `max_model_len`, 512K
should sit halfway to 1M's penalty. It does not, it sits at zero. vLLM v0.27.0
does contain PR #50004 (adaptive topk width, which loops the live context rather
than the maximum), so the most plausible reading is that the adaptive path covers
up to some bound and 1,048,576 falls outside it. Unconfirmed; worth a look if 1M
is ever needed.

The base image is named
`v027-ngc2607-dsv4-0731-dspark-k7-**256k**-production`, so 256k is presumably
what it was validated at. 512K measuring identically suggests the validated
figure is conservative rather than a hard boundary.

**Settled on `max-model-len 524288`.** 65,536 was needlessly small, 262,144 was an
unnecessary compromise, and 1,048,576 is capacity-real and latency-impractical. Anything needing true 1M prompts should
expect ~2.6 h of prefill and be scheduled as batch work, not served
interactively.

#### Verify the window functionally, not from metadata

`/v1/models` reporting a number is weak evidence, and `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
is baked into these images, so a declared value need not be backed by KV. The
check that actually settles it is to send a prompt larger than the value you
doubt:

```
at 262,144 declared:  101,211 prompt tokens in  48.5 s (2,087 tok/s), replied "OK"
at 524,288 declared:  307,609 prompt tokens in 152.8 s (2,013 tok/s), replied "OK"
```

Each overflows the next setting down, so neither window is nominal. Note also
that prefill is flat across the whole range tested — 2,495 tok/s at 4k, 2,098 at
58k, 2,087 at 101k, 2,013 at 307k — so on this architecture prefill throughput is
very nearly independent of prompt length.

**Clients cache this value.** A long-running client kept reporting a 65K window
for ~47 minutes after the server moved to 262,144, because it had read
`/v1/models` once at its own startup — the server had genuinely been at 65,536 for
hours beforehand. The endpoint was truthful and the reader was stale. Same shape
as reading `docker compose logs` after a relaunch and getting the previous
container's output, which happened twice while producing this file: after
changing serving config, refresh the reader before doubting the source, and
prefer a functional probe to any reported field.

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
   **262,144 and 524,288 both cost nothing measurable**; 524,288 is the setting.

Each should be measured at the Run 001 shape (1024/128, concurrency 1 and 6,
warm) so the rows stay comparable.

---

## Run 004 — decode at depth, and a needle test

**2026-08-19.** `max-model-len 524288`, spec decode on, steering on, TP=2, warm.
Prompted by a comparison against `Entrpi/ds4`, which publishes deep-context
numbers and needle validation that we had no equivalent of.

### Decode does not degrade with depth; prefill is the only thing that scales

`vllm bench serve`, random tokens (unique per request, so prefix caching cannot
serve them), 128 output tokens, concurrency 1. TPOT excludes the first token, so
it isolates decode from prefill.

| input | median TTFT | implied prefill | median TPOT | decode |
|---:|---:|---:|---:|---:|
| 2,048 | 1.13 s | 1,812 tok/s | 29.24 ms | 34.2 tok/s |
| 32,768 | 14.04 s | 2,341 tok/s | 31.47 ms | 31.8 tok/s |
| 131,072 | 64.90 s | 2,019 tok/s | 36.55 ms | 27.4 tok/s |
| 262,144 | 147.78 s | 1,774 tok/s | **32.52 ms** | **30.8 tok/s** |

Decode is flat within noise across two orders of magnitude of context: 29–37
ms/tok, with 262k faster than 131k. Prefill is likewise flat per token
(1,774–2,341 tok/s), so TTFT grows linearly with prompt length and nothing else
does.

**This is the clearest illustration of why `Output token throughput` must not be
quoted as a speed.** Over those same four rows it reads 24.95, 7.08, 1.98, 0.84
tok/s — a 30x "collapse" — while actual decode moves by 12%. It is
`output_tokens / total_duration`, and at 262k input the duration is 99% prefill.

### Needle-in-haystack

Never tested before. A unique 10-hex-character code is planted at a known depth,
the filler carries a per-run id in every sentence so no run shares a prefix with
another, and the model is asked to return only the code.

| prompt tokens | needle depth | wall | result |
|---:|---:|---:|---|
| 161,436 | 5 % | 81.4 s | exact |
| 161,436 | 50 % | 81.6 s | exact |
| 161,438 | 95 % | 81.6 s | exact |
| 488,237 | 5 % | 345.5 s | **9 of 10 characters** |
| 514,036 | 50 % | 374.6 s | exact |
| 514,035 | 95 % | 373.8 s | exact |

**5 of 6 exact, up to 514,035 tokens.** The exception returned `98F231EFF`
against `98F231EFF1` — the correct code missing its final character.

That is a real recall error and not output truncation, which was checked: re-run
at `max_tokens` 64 and 128 both produced 6 output tokens with
`finish_reason=stop`, so the model chose to stop after nine characters. It sits
at the deepest tested prompt with the needle earliest in it.

So deep context is substantially trustworthy but not perfect at ~0.5M tokens, and
a claim of "every needle found" depends on how strictly the match is scored — a
one-character error passes a fuzzy check and fails an exact one.

### Against Entrpi/ds4 (one Spark, Q2 weights)

Their published figures against ours, noting these are different quantisations on
different node counts and not a like-for-like contest.

| | ds4, 1 Spark, Q2 | ours, 2 Sparks, fp8 |
|---|---|---|
| prefill, shallow | 1,008 tok/s @14k | **2,341 tok/s @32k** |
| prefill, deep | 776 tok/s @515k, 633 @975k | **1,774 tok/s @262k** |
| decode, shallow | 22.66 tok/s @2k | **34.2 tok/s @2k** |
| decode, deep | 45.7 ms/tok @240k; 146–177 ms/tok @248–519k | **32.52 ms/tok @262k** |
| spec decode | ~2.0 tok/step @85.7 % | **5.99 tok/step @99.7 %** (3.00 on prose) |
| max active context | **3,019,176 tok** (1 GiB floor) | 883,902 KV tok |
| nodes | **1** | 2 |

We are 2–3x faster on prefill and 1.4–5x faster on decode at depth. They hold
~3.4x more active context on half the hardware.

Their capacity advantage is mechanical, not mysterious: Q2 weights (~81 GB, whole
model on one node, no TP duplication), compressed KV (FP8 codes with an FP4 e2m1
indexer), and demand-mapped banks so idle context costs nearly nothing. Three
design choices we do not have.

The cost is precision they have not quantified — their docs state the 2-bit
quantisations "behave well" and offer no perplexity or accuracy comparison against
higher precision. For measuring what a rank-1 intervention does to capability
that is disqualifying: our own numbers put 3.57 % error into the projection at
int4 against 0.61 % at int8, and 2-bit weight noise would swamp the effect being
measured. They also have no steering, control-vector, LoRA or hook path at all.

**Worth borrowing regardless:** a memory floor with graceful admission refusal
instead of an OOM kill (ours SIGKILLed a worker at `max-num-seqs 32`), and
demand-mapped KV so idle context is nearly free.

**And a correction to an idea this comparison prompted.** Seeing 47 GB of KV on
their box against our 14.22 GiB, I assumed `--gpu-memory-utilization 0.80` was
leaving 24 GiB per node unused and was about to recommend raising it. Measured
first: the box reports 111 GiB used of 121 GiB with 9 GiB available while
serving. On unified memory that fraction does not correspond to idle capacity,
and there is no 24 GiB to reclaim.

---

## Run 005 — the 1M collapse localised, and worked around

**2026-08-19.** The Run 003 collapse turned out to be a narrow, avoidable bug
rather than a property of long context. **Running at 1,032,192.**

### My stated mechanism was wrong

Run 004 blamed the compressor's adaptive topk width and cited PR #50004. Reading
`vllm/models/deepseek_v4/sparse_mla.py:264` disproves that:

```python
active_topk_width = min(
    max(next_power_of_2(cm.max_seq_len // compress_ratio), _C128A_TOPK_ALIGNMENT),
    self.c128a_max_compressed,
)
```

`cm.max_seq_len` is the **live** batch length, so a 58,008-token prompt yields
`next_pow2(453) = 512` at every declared context. The adaptive path works and is
not the cause.

### Bisection

Same 58,008-token prompt at each setting. `c128a_max_compressed`, the buffer row
width, is `align(ctx / 128, 128)`.

| declared context | c128a width | prefill | trivial request |
|---:|---:|---:|---:|
| 65,536 | 512 | 2,089 tok/s | 0.20 s |
| 262,144 | 2048 | 2,098 tok/s | 0.20 s |
| 524,288 | 4096 | 2,089 tok/s | 0.16 s |
| 786,432 | 6144 | 2,076 tok/s | 0.21 s |
| **1,032,192** | **8064** | **2,090 tok/s** | **0.19 s** |
| 1,048,576 | **8192** | **110 tok/s** | 36.9 s |

**A 1.6 % increase in declared context (16,384 tokens) changes prefill by 19x.**
Everything up to width 8064 is at full speed; width 8192 collapses.

8192 is not an arbitrary number. It is the kernel's hardcoded default width, per
the comment at `sparse_mla.py:178-182`: *"Otherwise the kernel's default 8192
iterates past row width and spills writes into adjacent rows."* The pathological
case is precisely where the computed width equals that default. Cause not proven
beyond the coincidence, but the boundary is exact and reproducible.

### The workaround

Declare **1,032,192** instead of 1,048,576: 98.4 % of the context, full prefill
speed, and KV of 1,367,456 tokens against 1,375,854 (0.6 % less).

Verified functionally, not from metadata — an **815,037-token** prompt, which
exceeds the 786,432 setting below it:

```
815,037 prompt tokens in 737.2 s = 1,106 tok/s prefill
needle at 50 % depth: FOUND (58FC9C942D)
```

### Correction: prefill is not flat at extreme length

Run 004 concluded prefill is "very nearly independent of prompt length" on
evidence up to 307k. At 815k it is **1,106 tok/s against 2,090 at 58k**, roughly
half. So prefill degrades gradually with prompt length after all — the claim was
right over the range measured and wrong as stated generally.

That degradation is a different phenomenon from the width-8192 collapse: gradual
and ~1.9x over a 14x longer prompt, against abrupt and 19x from a 1.6 % config
change.

For scale, `Entrpi/ds4` reports 633 tok/s ingesting 975k tokens on one Spark at
2-bit. At 815k we measure 1,106 tok/s, about 1.75x faster at comparable depth.

### Worth reporting upstream

A declared context of 1,048,576 — the model's own native maximum, and the obvious
value to configure — lands exactly on the kernel's default width and costs 19x
prefill. Anyone serving DeepSeek V4 Flash at its documented context on vLLM
v0.27.x hits this, and the symptom looks like "long context is slow" rather than
like a bug.


---

## Run 006 — MiaAI Anemll recipe, migration validation

**2026-08-21.** The serving stack changed underneath this document. The Stage-C
overlay (vLLM 0.21.1) was retired after its DSpark draft path was isolated as
the source of deterministic long-context output corruption (spec-gate A/B:
identical probes clean with spec off, looping with spec on; onset position a
deterministic function of prompt length — the same family as tonyd2wild issue
#18). The replacement is the maintained
[MiaAI-Lab recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark):
Anemll `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2.dev0) + 12 boot
hotfixes (#21/#22/#26v2/#27/#43/#55 etc.), recipe clone `~/dspark-miaai` on
both nodes.

| | |
|---|---|
| runtime | vLLM `0.25.2.dev0`, Anemll `0.1.1` image (ID `3430d6614a8e`) |
| model revision | `7872f01b` (on-disk snapshot; recipe-tested pin is `9e165c30`) |
| max model len | **1,048,576** — KV pool 1,857,271 tokens, 1.77x at 1M |
| KV cache dtype | `nvfp4_ds_mla` (recipe default + issue-22 dispatch hotfix) |
| speculative decoding | **on**, `dspark` k=5 probabilistic |
| max num seqs / batched tokens | 6 / 8,192, prefill threshold 1,024 |
| gpu mem utilisation | 0.80 |
| steering | **OFF** — the 271-line patch targets v0.27's `model.py` and does not apply to 0.25.2; port pending |

### Results (client wall over WiFi, TTFT included — not `vllm bench serve`)

| shape | out tok/s | acceptance | note |
|---|---:|---:|---|
| peak-finder (count 1–300) | **79.1** | ~100 % | fr=stop |
| prose (TCP congestion control) | 31.9 | 46 % | fr=length at 512 |

Deep context (needle at 40 % depth, temp 0, generation checked for repetition
onset):

| prompt tokens | prefill tok/s | needle | repeat onset |
|---:|---:|---|---|
| 28,542 | 2,243 | HIT | none |
| 114,041 | 1,611 | HIT | none |
| 233,383 | 1,453 | HIT | none |

The tonyd2wild #18 deterministic loop reproducer (Spanish factory arithmetic,
temp 0) — which verbatim-looped to budget exhaustion on Stage-C every time —
completes in **152 tokens with the correct answer (636)**.

### Reading these numbers

- Peak-finder at 79.1 tok/s matches Run 002's 77.3 on v027 and the retired
  stack's 78.4 with Patch 4 — spec decode is at parity or better, with none of
  the Stage-C corruption. probe6 (repeat-onset sweep 0/4k/16k/64k chars) is
  clean at every depth.
- Prose at 31.9 vs Run 002's 43.1: this measurement includes TTFT and WiFi
  latency; Run 002's was taken in-container. Same shape, same ~46 % acceptance
  — treat as equal.
- Prefill at 233k (1,453 tok/s) is inside Run 004/005's gradual-degradation
  envelope, not the width-8192 collapse. The 1M-collapse localisation was on
  v0.27.x; whether 0.25.2 shares it was not tested — a full 1M-token prefill
  was not exercised (233k was the gate, matching Run 004's functional depth).
- nvfp4_ds_mla is back in use (it does not exist upstream in v0.27, which
  forced fp8 there); on 0.25.2 it is the recipe default and the issue-22
  hotfix covers its long-context dispatch regression.

## Run 007 — steering ported to the Anemll stack (0.25.2 hotfix)

Same MiaAI Anemll 0.1.1 stack as Run 006, now with projective steering
re-enabled via `patches/hotfix-dsv4-steering-projective.py` (boot hotfix, 4
anchors on the 0.25.2 `model.py`; the v027 patch does not apply there and the
image ships no `gguf` package, so the spec-conformant GGUF reader is embedded
in the hotfix). Identical vector, alpha and layer set as the v027 config:
`DeepSeek-V4-Flash-0731-general-abliterated-cvec-L10-38-a4-keysdir.gguf`,
alpha 4.0, layers 10–38 (rank 1, 29 directions, n_embd 4096).

Artifact provenance re-verified before wiring: the GGUF parses clean against
the spec (mode=project, hook residual_stream_post_layer, declared layers =
tensor layers, base rev 7872f01b = the pinned checkpoint), and the two .pt
files on disk resolve unambiguously — `steer-keysdir-29.pt` ≡ general-keysdir
GGUF and `steer-29.pt` ≡ cyber GGUF (cos 1.0000 per layer both ways; the two
families sit at cos ~0.20 against each other).

Validation (2026-08-21, spec decode ON, steering ON, both TP ranks logging
`DSpark refusal steering active: hook=post_layer alpha=4.000 ... layers=29`):

| check | result |
|---|---|
| boot hotfix | applied at 4/4 anchors both nodes; `--check` validates vector before model load; fail-closed when `DSPARK_STEER_PATH` set but patch/vector can't apply |
| refusal A/B probe | 8× refusal32 + 4× blueteam32, temp 0: **12/12 bypass, 0 refuse, 0 garbled** (stock refuses the refusal32 class; blue-team answers stayed coherent) |
| recipe smoke | 6/6 |
| garble sweep (0/4k/16k/64k chars, temp 0) | **ALL CLEAN**, incl. the 64k case that looped on Stage-C |
| spec acceptance | 2040/5470 = **37.3 %**, per-position 783/535/357/225/140 — healthy decay, no draft degradation from steering |

Cost: none measured, consistent with the v027 finding (projection is one
GEMV + one outer product per steered layer against full attention+MoE
matmuls; v027 measured 42–44 tok/s steered vs unsteered, inside noise).

Ops notes:

- Steering is config in `.env.dspark` (`DSPARK_STEER_PATH/_ALPHA/_LAYERS`),
  wired through compose env passthrough; off = empty `DSPARK_STEER_PATH`.
  The cyber variant is a one-line swap.
- The start script syncs hotfixes and `.env.dspark` to the worker but **not**
  `docker-compose.dspark.yml` — sync it manually when it changes.
- v027 stack stays parked as the fallback; its steering image is no longer
  needed for steering since this port, only for v027-specific experiments.
