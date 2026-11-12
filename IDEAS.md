# IDEAS

The public backlog. Each entry is a self-contained piece of work with the
why, the where, and the acceptance test — pick one up and open a PR. If an
entry is unclear, open an issue referencing its number before building.

Hardware legend: **[offline]** = CPU-only, no GPU, no cloud account needed.
**[Modal]** = needs rented GPUs (we can run the validation for you on
report). **[rig]** = needs the 2× DGX Spark pair (we run it; describe the
change and we validate).

---

## Good first PRs

### 1. captain-vector `probe` verb — single-token α-ladder readout [offline+endpoint]

Dose calibration today costs full generations (up to 192 tokens × prompts
× arms × α values) per ladder. A `probe` verb would ask a served endpoint
for a single verdict token under `guided_choice` + `logprobs` and print
the per-α first-token compliance curve — one decode step per prompt per α.
The full-length eval stays the ship gate (first-token compliance is a
proxy; refusals pivot mid-generation — see refusal-research/METHODOLOGY's
"structural vs behavioural" trap); `probe` is the screen that picks the
α worth gating.

- Where: `tools/captain-vector/` (new verb alongside `inspect`/`export`/
  `bake`); hit any OpenAI-compatible vLLM endpoint.
- Done when: for one published vector (e.g. GLP-41 Inkling, α=0.25), the
  probe curve's knee matches the α chosen by the full-length ladder.

---

## Bigger builds

### 2. Serve a wizard lane on the plugin instead of the hotfix [rig]

All ten arch adapters in `plugins/vllm/weightless_steer/archs/` are
GPU-validated (numbers in `BENCHMARK.md`), but the live lanes still boot
the string-patched hotfixes (`patches/hotfix-*.py`). Migrate one wizard
lane end-to-end: pip-install `weightless-steer` into the lane's image
(or PYTHONPATH + a `sitecustomize.py` shim for registration), keep the
fail-closed contract (a boot asked for steering never serves unsteered),
and reproduce the lane's hotfix reference numbers.

- Where: `plugins/vllm/`, the lane's `lanes/`.
- Done when: the lane serves steered with zero hotfix files and its
  numbers land in the hotfix row's envelope on the lane's own suite.

### 3. Per-request controls (milestone 2) [rig]

The plugin's `control_plane.py` turns one step's scheduled requests into
per-request α rows and layer gates — but per-request serving is refused
at construction today (`WEIGHTLESS_ENABLE_MILESTONE_2` raises; no runner
binding exists). The build: validate request metadata against the bounded
policy (`weightless_runtime/`), bind the control plane to the engine's
batch metadata, and serve two concurrent requests at different α with
measurably different steering.

- Where: `plugins/vllm/weightless_steer/control_plane.py`, `core.py`,
  `weightless_runtime/`.
- Done when: a two-request probe (same prompt, α=0 vs α=calibrated)
  returns stock vs steered output on one engine, on Modal, with the
  fail-closed gates intact.

### 6. New plugin archs [offline+Modal]

One module per architecture — the pattern is established
(`archs/glm5next.py` is the reference): subclass the upstream model,
copy its forward verbatim, insert the projection at the hotfix-calibrated
site, register in `SHADOWED_ARCHS`, mirror the offline tests, verify on
Modal. Candidates: **Kimi Linear 3B** (KDA linear attention — new site
research), **GLM-5.3-Flash EXL3** (the B12X fork's split decoder loop —
see `patches/hotfix-glm53-exl3-steering-projective.py` for the two-loop
trap), or the next day-0 arch vLLM ships.

- Where: `plugins/vllm/weightless_steer/archs/`,
  `docs/vllm-plugin-design.md`, `patches/hotfix-*.py` for sites.
- Done when: offline suite passes plus a Modal boot shows
  `weightless GLP steering active` with the arch's published vector.

### 7. GLP vectors for uncovered models [Modal]

The derivation pipeline is fully public: `tools/captain-vector/` +
refusal-research/METHODOLOGY.md (contrast choice, massive-activation
screen, null gate, per-model α calibration, behavioural verify). Derive a
projective control vector for an open model we haven't covered, run the
gates, and we'll review for hosting under `msuiche/` on Hugging Face.

- Done when: the METHODOLOGY checklist passes end-to-end — null gate,
  dose calibration on the model (never carried), held-out behavioural
  verify, GGUF spec-conformant per `docs/spec/GLP.md`.

---

## Research

### 8. MoE LoRA fold [offline research]

`bake` folds a GLP vector into a closed-form rank-1 LoRA for **dense**
models (the Qwen3.8-27B lane matches the GGUF arm on hardware). MoE
models currently need the GGUF/GLP extension at runtime. Can the
per-writer closed-form fold be extended to expert writers without the
circuit mismatch that REAP-pruned builds already trip? Negative results
with measurements are a valid outcome here.

- Where: `tools/captain-vector/` (`bake`), README "Streams and writers".
- Done when: a MoE folded arm either matches its GGUF arm on a held-out
  suite, or the writeup shows exactly which writer breaks and why.

### 9. DSV4.1-Flash wizard lane [blocked upstream — watch vllm#56201]

Native serving needs vllm-project/vllm PR #56201 (`dsv41-feat`) merged.
Everything else exists: the EXL3 2.9bpw quant fits 2× DGX Spark (served
on the rig via the external MiaAI stack), the GLP-39 vector is published
and Modal-validated. When the PR lands: a `deepseek_v41` plugin adapter
(new arch — engram layers, see `lanes/dsv41/`), then the lane.

- Done when: a wizard lane boots DSV4.1-Flash steered on stock-ish vLLM.

### 10. Upstream the steering hook [advocacy]

The design doc's recommended upstream ask — a `SupportsSteering`
interface — would shrink every per-arch subclass to an interface
implementation with the steering core untouched. The survey of viable
extension points is written (`docs/vllm-plugin-design.md`); what's
needed is the upstream conversation and possibly a reference PR against
vLLM.

- Done when: vLLM has a merged extension point that makes our
  ModelRegistry shadowing unnecessary.

### 11. Bit-exact α=0 for the mHC (glm5next) plugin lane [offline research]

The glm5next adapter de-fuses upstream's inter-layer `MHCFusedPostPreOp`
(materialize `hc_post`, standalone `hc_pre` next layer), so α=0 serving is
behaviorally equivalent but NOT byte-identical to stock: measured
2026-10-06 on stock vLLM 0.31.0, 0/12 greedy 400-token probes
byte-identical, first divergence at tokens ~5–63, suite-level equivalence
holds (numbers in `plugins/vllm/README.md`). A fusion-preserving steering
site — or an upstream split/fused kernel-equivalence guarantee — would make
α=0 bit-exact. Parked; acceptance standard today is suite-level
equivalence.
