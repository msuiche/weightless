# Steering artifacts (ours)

Naming convention: **GLP-n** is a GLP vector touching **n layers** — GLP-29
below is the DSV4 vector, GLP-49 the Qwen one, GLP-47 the Flash-Next one,
GLP-44 the GLM-Flash one, GLP-77 the GLM 743B one. Every vector below is
published under `msuiche/` on Hugging Face (gated — fetch with an HF token),
spec-conformant per [`docs/spec/GLP.md`](spec/GLP.md) and
verified against its pinned checkpoint revision.

- [`msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29`](https://huggingface.co/msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29)
  — **DSV4 lane, GLP-29 (GGUF).** Cyber-contrast vector: 29 per-layer
  directions over
  L10–38, n_embd 4096, α=4.0. The live config currently serves the
  general-contrast variant from the same repo family — a *third-party*
  direction we reformatted, re-measured and repackaged (attribution in the
  file metadata); swapping is one `WEIGHTLESS_STEER_PATH` line. Wiring:
  `recipe/anemll/README.md`.
- [`msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49`](https://huggingface.co/msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49)
  — **Qwen lane, GLP-49 (GGUF + LoRA).** The GGUF is canonical: per-layer
  diff of
  means, 49 directions over L10–58, n_embd 5120, α=1.0 (α=4 measurably
  over-refuses on this model). The rank-1 LoRA (`mlp.down_proj`, L1–63,
  α=1.0 baked) is the same intervention folded into weights — it loads in
  stock vLLM/peft/llama.cpp with no hotfix, matches the GGUF on NVFP4
  hardware (both 24/32 refusal32, 2026-08-22), but is still unscored on the
  cyber holdout and is valid only for checkpoint revision `1d4bf0f2ff60`
  (`lora_A` embeds `W`). Wiring: `recipe/qwen/README.md`.
- [`msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47`](https://huggingface.co/msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47)
  — **Qwen3.8-Flash-Next (`qwen4_exp`), GLP-47, day-0.** Per-layer
  difference-of-means over the widened hyper-connection stream (10240 =
  4×2560), layers 1–47, α=1.0. Derived 2026-08-26 on Modal (8×H100, BF16
  reference) from AdvBench-vs-Alpaca. refusal32 3.1% → 81.2% at α=1.0 with
  benign32 32/32 and capability12 12/12 (higher α over-projects — do not
  raise it; the shipped default is the measured peak). Direction verified
  engine-independent: a vLLM-lane capture reproduces it at cos 0.9931.
  Serving lane: `recipe/qwen38fn/` (TP=2 on 2x Spark, day-0 image; the
  hotfix steers the materialized hyper-connection stream).
- [`msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44`](https://huggingface.co/msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44)
  — **GLM-5.3-Flash (`glm5_next`), GLP-44, day-0.** Per-layer dom over the
  Sinkhorn hyper-connection stream (16384 = 4×4096), layers 1–44, α=2.0.
  Derived from the FP8-native canonical checkpoint (snapshot `3f1971b7`).
  refusal32 3.1% → 65.6% at α=2.0, benign/capability clean; **α=2.5+
  garbles everything** — the cliff is abrupt, stay at or below 2.0.
  Serving lane: `recipe/glm53/` (TP=4 on **4x Spark**, NVFP4 quant on the
  sm121-v8 patched day-0 image; the hotfix steers the materialized mHC
  stream).
- [`msuiche/GLM-5.3-abliterated-cyber-GLP-77`](https://huggingface.co/msuiche/GLM-5.3-abliterated-cyber-GLP-77)
  — **GLM-5.3 743B (`glm_moe_dsa`), GLP-77.** Per-layer dom over the plain
  residual stream (6144 — no hyperconnection widening on this arch), layers
  1–77, α=1.0. Derived 2026-08-30 on the NVFP4 deployment (8×H100, vLLM
  0.28.0) with the vLLM capture lane; every shipped layer clears the null
  gate. cyber32 18/32 → **32/32** at α=1.0 with benign/capability clean and a
  passing null arm; refusal32 moves 1/32 → 12/32 (37.5%; 6/32 answer-audited)
  — the 753B flagship's refusal is much stickier than Flash's, and **α>1
  makes it WORSE at full length** (do not raise it). Serving lane:
  `recipe/glm53xl/` (TP=4 on 4x Spark, tonyd2wild's Int4-Int8Mix stack; the
  hotfix steers `hidden_states + residual` on the `deepseek_v2` path).
- [`msuiche/DeepSeek-V4.1-Flash-abliterated-cyber-GLP-39-L1-39-a0.5`](https://huggingface.co/msuiche/DeepSeek-V4.1-Flash-abliterated-cyber-GLP-39-L1-39-a0.5)
  — **DeepSeek-V4.1-Flash (`deepseek_v41`), GLP-39.** Per-layer dom over the
  post-layer hyper-connection fold reduced to the single stream (the mean over
  the 4 hc copies, pre-engram — derive and apply at the same site), layers
  1–39, α=0.5, mask 0.005 (the massive-activation screen flagged 571–3104×).
  Derived 2026-09-10 on Modal (4×H200, vLLM `dsv41-feat` capture lane):
  refusal32 4→24, cyber32 5→31, propaganda32 32/32, one benign classifier
  false positive. Research vector only — the arch loads in no vLLM release
  and no 2×128 GB quant exists, so `recipe/dsv41/` fails closed.
- [`msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75`](https://huggingface.co/msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-39-L1-39-a0.75)
  — **DeepSeek-V4.1-Flash-1017, GLP-39.** Same hook over the same arch (the
  1017 config is byte-identical to the 0909 release), layers 1–39, α=0.75 —
  the knee moved: this checkpoint refuses harder at stock and code-mixes
  earlier. Derived 2026-10-17 on the same Modal lane: refusal32 1→23,
  cyber32 12→29, propaganda32 32/32, benign32 clean, zero garbles. Research
  vector only, same blockers as the 0909 vector above.
- [`msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-27-L11-37-a3.5`](https://huggingface.co/msuiche/DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-27-L11-37-a3.5)
  — **DeepSeek-V4.1-Flash-1017 (`deepseek_v41`), GLP-27, spec_version 3.**
  The first GLP v3 artifact: 544 KB of weight-SVD directions from the FFN
  writers, applied at the post-fold, pre-Engram `hc_fold_pre_engram` site over
  the mean of the four hyper-connection copies, with the attention `o_proj`
  writer armed on the same stack. Layers 11–37, α=3.5, pinned to base revision
  `1bae0e10`. Model-card validation at the shipped dose: weaponize32 4/32
  refused, refusal32 6/32, benign32-holdout 9/32 (stock 8/32). Built-in serving
  is available in [christopherowen/spark-ds41f](https://github.com/christopherowen/spark-ds41f).
- [`msuiche/gemma-3-1b-it-abliterated-GLP-20-L4-25-a0.25`](https://huggingface.co/msuiche/gemma-3-1b-it-abliterated-GLP-20-L4-25-a0.25)
  — **Gemma-3 1B (`gemma3_text`), GLP-20.** Per-layer dom over the post-layer
  residual stream (1152), layers 4, 5, 8–25 zero-based — the null gate
  dropped 6 and 7, so the span genuinely carries gaps — α=0.25. First
  Google-family vector. Derived 2026-10-11 on Modal (A10G, transformers
  capture lane) from the byte-identical `unsloth/gemma-3-1b-it` mirror — the
  `google/` repo is license-gated for our token, so the mirror revision is
  the pin in `glp.base_revision`. Stock already complies 29/32 on refusal32;
  the vector closes the last 3 with benign-holdout 32/32 at every rung, a
  byte-exact α=0 no-op and zero garbles. No serving lane — apply with
  `glp.py` or `tools/captain-vector/apply_transformers.py`.
- [`msuiche/gemma-3-4b-it-abliterated-GLP-28-L6-33-a0.25`](https://huggingface.co/msuiche/gemma-3-4b-it-abliterated-GLP-28-L6-33-a0.25)
  — **Gemma-3 4B (`gemma3` multimodal), GLP-28.** Same recipe over the
  post-layer residual stream (2560), layers 6–33 zero-based (contiguous),
  α=0.25. First multimodal-arch derivation: the checkpoint is
  `Gemma3ForConditionalGeneration` and the steered text stack resolves at
  `model.language_model.layers`. Stock refusal32 30/32 → 32/32 at α=0.25,
  benign-holdout 32/32 every arm, α=0 no-op byte-exact, zero garbles. Same
  unsloth-mirror base pin; apply with `glp.py` / `apply_transformers.py`.
- [`msuiche/gemma-3-12b-it-abliterated-GLP-43-L5-47-a0.25`](https://huggingface.co/msuiche/gemma-3-12b-it-abliterated-GLP-43-L5-47-a0.25)
  — **Gemma-3 12B (`gemma3` multimodal), GLP-43.** Post-layer residual stream
  (3840), layers 5–47 zero-based (contiguous), α=0.25. Stock refusal32 30/32
  → 32/32 at α=0.25, benign-holdout 32/32 every arm, α=0 no-op byte-exact,
  zero garbles. Same unsloth-mirror base pin; apply with `glp.py` /
  `apply_transformers.py`.
- [`msuiche/gemma-3-27b-it-abliterated-GLP-51-L11-61-a0.25`](https://huggingface.co/msuiche/gemma-3-27b-it-abliterated-GLP-51-L11-61-a0.25)
  — **Gemma-3 27B (`gemma3` multimodal), GLP-51.** Post-layer residual stream
  (5376), layers 11–61 zero-based (contiguous), α=0.25. Stock refusal32
  29/32 → 31/32 at the shipped α=0.25 (32/32 from α=0.5 up — the knee rule
  ships the bottom rung), benign-holdout 32/32 every arm, α=0 no-op
  byte-exact, zero garbles. Same unsloth-mirror base pin; apply with `glp.py`
  / `apply_transformers.py`.
- [`msuiche/gemma-4-E2B-it-abliterated-GLP-31-L4-34-a0.5`](https://huggingface.co/msuiche/gemma-4-E2B-it-abliterated-GLP-31-L4-34-a0.5)
  — **Gemma-4 E2B (`gemma4` multimodal), GLP-31.** Post-layer residual
  stream (1536), layers 4–34 zero-based (contiguous), α=0.5. Derived
  2026-10-11 on Modal (A10G, transformers 5.19.0 capture lane) from
  canonical `google/gemma-4-E2B-it` — unlike gemma-3, the gemma-4 google/
  repos serve weights to our token, so no mirror; the google/ revision sha
  is the pin in `glp.base_revision`. Stock refuses hard — 4/32 comply on
  refusal32 — and the ladder is monotone (28 → 31 → 32), so the vector does
  real unlocking: 31/32 at the shipped α=0.5 with benign-holdout 32/32 at
  every rung, a byte-exact α=0 no-op and zero garbles. No serving lane —
  apply with `glp.py` or `tools/captain-vector/apply_transformers.py`.
- [`msuiche/gemma-4-E4B-it-abliterated-GLP-33-L5-41-a1`](https://huggingface.co/msuiche/gemma-4-E4B-it-abliterated-GLP-33-L5-41-a1)
  — **Gemma-4 E4B (`gemma4` multimodal), GLP-33.** Post-layer residual
  stream (2560), layers 5, 10–41 zero-based — the null gate dropped 6–9, so
  the span genuinely carries a gap — α=1.0. The family's hardest refuser:
  stock refuses 31/32 on refusal32 (the lone "comply" is a scorer artifact)
  and the ladder is the steepest (15 → 25 → 32), so the knee rule climbs
  past 0.5 and ships full α — 32/32 steered with benign-holdout 32/32 at
  every rung, α=0 no-op byte-exact, zero garbles. Same canonical google/
  base pin; apply with `glp.py` / `apply_transformers.py`.
- [`msuiche/gemma-4-12B-it-abliterated-GLP-33-L15-47-a0.5`](https://huggingface.co/msuiche/gemma-4-12B-it-abliterated-GLP-33-L15-47-a0.5)
  — **Gemma-4 12B (`gemma4_unified`), GLP-33.** Post-layer residual stream
  (3840), layers 15–47 zero-based (contiguous), α=0.5. The unified arch:
  vision/audio embedders into the shared text stack, no towers (the four
  siblings are plain `gemma4` with real towers + PLE; both nest the steered
  stack at `model.language_model.layers`). Stock already complies 32/32 on
  refusal32, so this ladder is the no-regression story — and it is
  non-monotone: α=0.25 dips to 29/32, α=0.5 and 1.0 hold 32/32 with
  benign-holdout 32/32 every arm, α=0 no-op byte-exact, zero garbles. Same
  canonical google/ base pin; apply with `glp.py` / `apply_transformers.py`.
- [`msuiche/gemma-4-26B-A4B-it-abliterated-GLP-27-L3-29-a0.5`](https://huggingface.co/msuiche/gemma-4-26B-A4B-it-abliterated-GLP-27-L3-29-a0.5)
  — **Gemma-4 26B-A4B (`gemma4` multimodal, MoE), GLP-27.** Post-layer
  residual stream (2816), layers 3–29 zero-based (contiguous), α=0.5. The
  first MoE derivation in the gemma line: 128 experts top-8 summed with a
  dense MLP per hybrid layer (~26B total / ~4B active) — the hook sits on
  the decoder-layer output downstream of router+experts, and direction
  quality matches the dense 12B (span min cos 0.575 vs 0.559). Stock
  complies 32/32; the ladder dips to 25/32 at α=0.25 and recovers to 31/32
  at the shipped α=0.5 (α=1.0 flips two benign-holdout prompts and fails
  that gate), benign-holdout 32/32 at the shipped rung, α=0 no-op
  byte-exact, zero garbles. Same canonical google/ base pin; apply with
  `glp.py` / `apply_transformers.py`.
- [`msuiche/gemma-4-31B-it-abliterated-GLP-54-L6-59-a0.5`](https://huggingface.co/msuiche/gemma-4-31B-it-abliterated-GLP-54-L6-59-a0.5)
  — **Gemma-4 31B (`gemma4` multimodal), GLP-54.** Post-layer residual
  stream (5376), layers 6–59 zero-based (contiguous), α=0.5. The dense
  flagship (60L, vision/audio towers + PLE). Stock complies 32/32 on
  refusal32; the shallowest non-monotone dip of the family — α=0.25 gives
  30/32, α=0.5 holds 32/32, α=1.0 re-engages one refusal — benign-holdout
  32/32 at every rung, α=0 no-op byte-exact, zero garbles. Same canonical
  google/ base pin; apply with `glp.py` / `apply_transformers.py`.
