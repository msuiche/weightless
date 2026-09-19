# Changelog

Date-based sections — the repo has no versioned releases yet; captain-vector
carries its own version numbers. Newest first. Steering-effectiveness numbers
live in `BENCHMARK.md`; this file tracks what shipped.

## 2026-09-19

### Added
- `FAQ.md` — collected answers from real user questions: RL-retraining
  vector maintenance, effective range (requants/merges), GLP vs LoRA vs
  llama.cpp CVC, projection vs addition, MoE/looped models, hedging
  directions, α calibration, termination integrity, refusal
  non-determinism, serving paths, bake scope caveat. Second pass: vector
  vs checkpoint redistribution, behavior-agnostic derivation, subspace
  vs single direction, add mode, closed models, filename convention,
  derive-your-own recipe + gates, GLP×GCD composition, serving overhead,
  upstream support outlook (23 entries total).
- Lane 11 (`setup.py serve 11`): **DSV4-Vision-Exp TP=2 serving — 2x DGX
  Spark, Anemll recipe**. Same image and DeepseekV4ForCausalLM arch as lane
  0, but its own env (`recipe/anemll/.env.dsv4vx`, tracked
  `.env.dsv4vx.example`), compose project (`deepseek-v4-flash-visionexp`)
  and remote dir (`dspark-visionexp`), so the 0731 lane's on-rig files
  survive swaps. New start-script variant
  `start-deepseek-v4-flash-visionexp-dspark.sh` (env/project/dir names only).
  setup.py: `DEPLOY_MAP`/`CONTAINER_GREP` (`visionexp` — longest match beats
  lane 0's `deepseek`) and the `DSPARK_REVISION`/hub-expose gates now cover
  lane 11.
- `.env.dsv4vx` values that differ from 0731 and why: FP8 original pinned at
  snapshot `6821d6ad`, served name `deepseek-v4-flash-vision-exp-dspark`,
  `GPU_MEMORY_UTILIZATION_TEXT=0.90` (202 GB FP8 ≈ 101 GB of weights per
  node — a 0.80 budget does not fit them), `MAX_MODEL_LEN=262144` (1M does
  not fit at this footprint), `MTP_NUM_TOKENS=6` (this checkpoint's nextn
  count is 3 so k must divide by 3, and dspark_block_size=5 so k<5 truncates
  draft blocks; 6 is the smallest value satisfying both — 0731's 5 fails
  SpeculativeConfig validation here), `ENABLE_VLLM_GB10_PATCH=0` with the
  NVFP4 flip documented as one env block.
- Steering: the fresh per-layer GLP-29 Vision-Exp vector at **α=1.0** (its
  own alpha_default; α=4 garbles — dose cliff per BENCHMARK.md). The vector
  was captured at `residual_stream_post_layer` and this lane's hotfix only
  implements `ffn_out_pre_residual`, so the served file is a hook-point
  relabel (`relabel_gguf.py`: hook_point→ffn_out_pre_residual,
  derived_at stays residual — the 2026-09-04 transferred-vector pattern;
  tensor bytes and glp.content_sha256 untouched), uploaded to the gated
  vector repo as `...-L10-38-a1.0-ffn.gguf` next to the original.
- **vllm-plugin fleet expansion: 8 new arch adapters GPU-validated on
  Modal** (`dsv4`, `qwen38`, `qwen38fn`, `glm53xl`, `kimi_k3`, `ouro`,
  `inkling`, `hy4` — 10 adapters total with `nemotron_h`/`glm5next`), each
  a lazy `ModelRegistry` shadow in `weightless_steer/archs/` with its own
  Modal lane (`modal/cloud_serve_<arch>.py` + eval/smoke driver) and raw
  artifacts in `modal/out-<arch>-plugin-test/`. Full-eval outcomes (all vs
  the hotfix reference rows, numbers in BENCHMARK.md): **qwen38** refusal32
  1→19/32, cyber32 4→20/32, benign32 31→27/32 (3 refusals, the documented
  cost) — in reference envelope; **dsv4** refusal32 0→16/32 @400tok (ref
  0→18/32 @400tok), cyber32 9→26/32 (first GLP-29 cyber32 measurement),
  benign32 32/32; **qwen38fn** refusal32 0→26/32 (ref 1→26), cyber32
  7→31/32 (ref 5→32), benign32 32/32 — matches calibration; **ouro** α=1.0
  exact-matches the hotfix reference (refusal32 32/32, cyber32 31/32 +1
  deflect, benign32 32/32); **inkling** refusal32 0→31/32 (ref 0→30),
  cyber32 1→28/32, benign32 31→30/32 — matches. Boot+smoke lanes:
  **hy4** (α=2.000, layers 1..77, width 6144; smoke 3/3 COMPLY), **glm53xl**
  (α=1.000, layers 1..77, width 6144; smoke 3/3 COMPLY), **kimi-k3**
  (2×H200:8 PP2×TP8 lockstep driver; α=1.000, layers 1..92, width 7168 on
  all 16 ranks; refusal-adjacent smoke prompt complied substantively,
  benign controls normal).

### Known issues
- The hotfix lane's `modal/k3_serve_driver.py` still carries the
  ungated-proxy EADDRINUSE bug the plugin lane fixed: torchrun spawns 8
  driver processes per node sharing the port space, so the ungated
  `start_tcp_proxy(8000, ...)` bind races and 7 of 8 local ranks crash.
  The plugin lane's `modal/k3_plugin_serve_driver.py` gates the proxy on
  `LOCAL_RANK==0` (comment at the fix site documents the mechanism);
  backporting to the hotfix driver is deliberately not done in this lane.
- DSV4 day-0 image: thinking defaults ON and `--chat-template` is ignored
  (the renderer's `apply_chat_template` override never consults the jinja).
  Drivers must pass `chat_template_kwargs={"enable_thinking": false}` —
  this broke the first dsv4 plugin eval run before it was caught
  (`modal/out-dsv4-plugin-test/run1-thinking-on/NOTE.md`; caveat also in
  `vllm-plugin/README.md`).

### Blocked
- The FP8 original checkpoint carries the vision tower (259 `vision.*`
  tensors + `aligner` + image-token embeddings) and the Anemll 0.1.1 image's
  `DeepseekV4ForCausalLM` refuses them at weight load ("no module or
  parameter named 'aligner'") — engine dies before profiling. Steering had
  already engaged on both ranks (`hook=ffn_out_pre_residual alpha=1.000,
  layers=29`). No weight surgery improvised (the strip_vision.py call is a
  conversation); the lane waits on the NVFP4 flip
  (`msuiche/DeepSeek-V4-Flash-Vision-Exp-NVFP4`, download running) or that
  decision. 0731 relaunched after the attempt.

## 2026-09-18

### Added
- `modal/cloud_serve_glm53.py` + `modal/eval_glm53_plugin.py`: the
  GLM-5.3-Flash plugin-validation lane on Modal (4×H100, day-0 x86 image,
  RedHatAI NVFP4, TP4). First GPU boot of the vllm-plugin glm5next adapter:
  steering engages in compiled mode (CUDA graphs on — the compile-rebind
  fix exercised for real); cyber32 3→31/32 matches the hotfix reference
  exactly, benign32 32/32 both arms, refusal32 2→9/32 (hotfix reference
  1→21/32 — stiffer stock baseline on this stack, not a swap artifact;
  numbers and raw outputs in BENCHMARK.md + `modal/out-glm53-plugin-test/`).
- `modal/sitecustomize.py`: registers the plugin at every interpreter start
  and pre-parses the vector fail-closed. Root cause of a day of "the plugin
  never runs" debugging: it ran all along — this vLLM build's dictConfig
  attaches a handler only to the `vllm` logger, so `weightless_steer.*`
  INFO lines never rendered. The shim prints its markers to stderr instead.

## 2026-09-17

### Added
- `weightless.py serve <lane>`: non-interactive lane switching. Resolves a
  lane by index or name substring, parks whatever is serving — wizard lanes
  plus known external stacks (`EXTERNAL_STACKS`: the MiaAI DSV4.1-Flash EXL3
  stack, which the wizard's container greps cannot see and which is itself a
  `serve` target, so DSV4 ↔ DSV4.1 switches are symmetric) — checks the
  lane's port is free, syncs the recipe, boots, and waits for the endpoint.
  Saved env values only; the wizard owns env edits. Harden steps stay
  wizard-only (sudo needs a tty). Flags: `--skip-assets`, `--skip-wait`.

### Fixed
- DSV4 peer gate: probe the peer's sshd over bash `/dev/tcp` instead of
  `ping` — the gx10 image ships no iputils, so the gate read "peer
  unreachable" forever and vLLM never started (`docker-compose.dsv4.yml`).

### Changed
- README: the delivery split is now explicit — `patches/hotfix-*.py` is the
  current production mechanism; `vllm-plugin/weightless_steer/archs/` (one
  adapter per architecture, registry shadowing) is where new archs land.

## 2026-09-11

### Added
- captain-vector 0.3: `bake` command — GLP GGUF + pinned base checkpoint →
  rank-1 PEFT/LoRA adapter. Fail-closed revision pin, per-writer roundtrip
  validation in `bake-report.json`. Dense models only.
- captain-vector 0.2: `inspect` + `export` commands (stdlib-only).
- captain-vector moves to `tools/captain-vector/`; tools-vs-scripts
  convention documented; `validate` CLI verb in `weightless.py`.
- `apply_transformers.py`: project GLP vectors on any HF decoder model via
  `register_forward_hook` — experiment lane, torch-only, no transformers
  import; hook-site context documented in source.
- `examples/peft_merge.py` + PEFT upstream contribution draft.
- `docs/vllm-plugin-design.md`: extension-point survey (registry shadowing
  is the viable pure-plugin path) + recommended `SupportsSteering` upstream
  ask. Design only, no code yet.
- `docs/llama-cpp-compat.md`: source-verified — stock llama.cpp cvec apply is
  addition-only; GLP files load silently with wrong semantics; supported
  route is bake → convert_lora_to_gguf → `--lora`.
- README: "Streams and writers — and why GLP is not a LoRA" section.
- Node hardening: thermal caps (GPU 300–2100MHz, CPU 2.4GHz lock) with loud
  failure; kdump arming for vmcore post-mortems.

### Changed
- bake repositioned as a troubleshooting/interop option, not the serving
  path; writer-vs-stream caveat documented in README, docstrings, CLI help,
  and both published HF model cards (GLP-49, GLP-63).

## 2026-09-11 (evening batch)

### Added
- `weightless-steer/`: GLP steering as an installable vLLM
  `vllm.general_plugins` package — shadows `NemotronHForCausalLM` via
  `ModelRegistry.register_model` when `WEIGHTLESS_STEER_PATH` is set
  (implements `docs/vllm-plugin-design.md` path 1). Shared GGUF container
  reader (all spec gates enforced at load), graph-safe steering core,
  nemotron_h arch adapter, 49 CPU-only tests. Same env vars and
  fail-closed semantics as the hotfix fleet; unset env = stock registry.
- `glp.py`: first-class GLP steering API for HF transformers —
  `apply_glp(model, path_or_hf_repo, alpha=None)` applies a GLP GGUF vector
  via residual-stream forward hooks (tuple re-wrapping, context manager +
  idempotent detach). Loads through weightless-steer's `container.py`, the
  one canonical reader. Alpha precedence: argument > `WEIGHTLESS_STEER_ALPHA`
  > `glp.alpha_default` > 1.0. Verified against the published GLP-49
  artifact; projection exact to ~2e-09 on the tiny-model smoke test.
- GLP format rank-k: subspace projection. `direction.<N>.<j>` tensors,
  `glp.dir_scales` / `glp.layer_scales` alpha model
  (`α_{L,j} = alpha_default · dir_scales[j] · layer_scales[L]`),
  `glp.spec_version 2` gate, Gram–Schmidt at write time (loaders validate,
  never re-orthogonalize). Rank-1 artifacts byte-compatible — same
  `content_sha256` (`spec/GLP.md`, *Rank-k* section).
- captain-vector 0.4.0: rank-k across write/validate/inspect/export;
  `bake` generalized from rank-1 to rank-k LoRA (B = stacked basis, r=k).
- `tools/captain-vector/examples/projection_adapter.py`: runnable
  PEFT-style `ProjectionAdapterLayer` reference implementation with numeric
  verification (≤3.0e-07 end-to-end vs closed form, `merge_and_unload`
  bit-exact; incoming-stream pass-through caveat demonstrated numerically).

### Changed
- `apply_transformers.py` / `glp.py`: hook applies
  `h − Σⱼ αⱼ(h·d̂ⱼ)d̂ⱼ` (rank-k); GGUF and safetensors lanes share one
  validation/hook implementation (`_attach`).
- `weightless-steer` container reader parses and gates rank-k; serving
  `SteeringCore` refuses rank-k explicitly (serving stays rank-1).
- `docs/peft-contribution-draft.md` tightened: writer-vs-stream scope
  section, differentiation from trained LoRA and llama.cpp CVC, exactness
  claim scoped to dense per-writer, in-repo verification numbers.
- `docs/vllm-plugin-design.md`: path 1 marked implemented.

## 2026-09-10

### Added
- DSV4.1 lane notes: GLP-39 vector exists; deployment marked NOT DEPLOYABLE.
- Muse-Glimmer-30B NVFP4 wizard lane (first single-node TP=1) + Modal
  bring-up.
- Diagnose chain: crash-loop/fabric/peer checks, mDNS→IP fallback, ssh
  identity support; DSV4 peer gate.
- Automatic node hardening on deploy + `TROUBLESHOOTING.md`; 2026-09-10
  wedge incident write-up with the self-healing layers now armed.

## 2026-09-09

### Added
- GLP × GCD stack diagram (svg + png).
- Wizard: single-word top menu (Serve/Deploy/Watch/Configure/Endpoint) with
  lane submenus; swap flow with drop-caches step; catches the day's two
  boot-killers before deploy.
- Asset-prep cached-complete fast path.

### Changed
- `tests/` split into `smoke/` (shell) and `structure/` (python), salvaged
  from PR #1.
- anemll example env: literal-IP rule for `MASTER_ADDR`/`VLLM_HOST_IP`;
  GLP-29 cyber α 4.0 → 6.0 at the FFN site.
- dash default lane → DSV4 (:8888) after the GLM→DSV4 rig swap.

## 2026-09-08

### Added
- `weightless.py` front-door CLI (wizard / dash / test dispatch).
- `scripts/dash.py`: live lane metrics — prefill/decode tok/s, queue, KV
  pressure, prefix-cache hit rate, TTFT, spec-decode acceptance; ANSI color,
  `--once` scriptable snapshot; wizard "Watch a lane" entry.

### Fixed
- Wizard crash on cloud lanes.

## 2026-09-07

### Changed
- glm53tp2: `MAX_NUM_BATCHED_TOKENS` 2048 → 4096 (prefill 2×; 8192 starves
  KV below the 131K floor).
- DFlash2 lane: v11 forward anchor, drafter cache-dir mount, port 8081.

### Added
- `patches/hotfix-ouro-steering-projective.py`: Ouro-2.6B looped-model
  steering (vLLM v0.26.0) — first loop-transformer lane.

## 2026-09-06

### Added
- glm53tp2: opt-in DFlash2 spec-decode lane (`SPECULATIVE=dflash2`, v11
  image, drafter staging+mount, enforce-eager).
- Kimi K3 2.9T as a second Modal cloud lane (2×H200:8, PP=2×TP=8) —
  GLP-92, refusal32 1/32 → 31/32, zero collateral.

## 2026-09-05

### Added
- GLM-5.3 743B as a Modal cloud lane — the endpoint is the deliverable.
- Nemotron-3.5-Lightning lane (GLP-51, single-node) with stock Spark recipe
  and projective steering at `residual_stream_post_layer`.
- Inkling SM121 lane: GLP steering with tested agent defaults; agent tools
  verified against the live endpoint.
- Wizard: asset preparation + confirm-gated lane switching; agent-clients
  step (omp `weightless` provider + hermes); glm53tp2 lane; dspark-router
  front door (:8000) with SSE strip for Inkling.

### Changed
- omp: all DGX lanes consolidated under one `weightless` provider with
  per-model baseUrl.
- Inkling watchdog shipped with calibration warnings after the
  false-positive saga (daytime off, nights only).

## 2026-09-04

### Added
- Inkling-Small TP=2 lane (2× DGX Spark, vLLM v0.28.0): real-weights
  serving with the load-reclaim + sm121 rel-attention hotfixes.
- Hotfixes warn on transferred vectors (`glp.derived_at` ≠ hook point).
- GLP spec: ds4 pre-residual hooks + ds4 reader registered (PR #970),
  `glp.derived_at` for transferred vectors.

### Fixed
- **DSV4 hook-site correction**: the true site is `ffn_out_pre_residual`,
  not a post-layer residual — site-label parity across patch, hotfix, and
  tests; the 9× figure rescoped honestly (it measured the attention write).

## 2026-09-03

### Added
- Vision-Exp NVFP4 published: lossless MXFP4→NVFP4 transcode (byte-exact vs
  NVIDIA's recipe), boot-validated on stock vLLM 0.28.0.

### Changed
- Inkling DGX lane marked BLOCKED — root cause confirmed: neither FA4
  backend runs Inkling on GB10; 15-boot elimination table + retest path;
  Modal lane is the working path. **(Resolved next day — 2026-09-04: the
  sm121 rel-attention + load-reclaim hotfixes boot real weights TP=2 on
  2×GB10 with CUDA graphs. The lane works on the Sparks; do not re-mark
  it blocked from this entry alone.)**
- hotfix-qwen38 tolerates v0.28.0 forward-anchor drift
  (`_maybe_add_hidden_state`).
- Hy4 propaganda32 framing: the per-country map is the signal, not the
  tally.

## 2026-09-02

### Added
- **GLP-77 Hy4-preview**: 770B steered at α=2.0 — largest model with a
  published refusal vector (refusal32 1/32 → 24/32, benign 32/32).
- **GLP-41 Inkling-Small** at the calibrated α=0.25: 0/32 stock lockdown →
  30/32 — the most dose-sensitive model in the program (α=1.0 garbles).
- Inkling TP=2 lane skeleton (structure-validated).

## 2026-09-01

### Added
- GLP-29 Vision-Exp published (gated).
- qwen38fn hardware-validated steered on 2× DGX Spark: refusal32 22/32,
  262K native context.

### Changed
- qwen38fn: native 262144 ctx restored — the 32–64K debugging values were
  protecting against a page-cache/PLE problem, not KV pressure; MiaAI-Lab's
  measured 2-Spark profile folded in (drop_caches, GMU 0.835, EP, lazy
  load).

## 2026-08-31

### Added
- DFlash2 acceptance findings published: GLP-44 steering costs the drafter
  no acceptance (taps sit pre-injection); tap set is training-matched — do
  not retune.

## 2026-08-30

### Added
- **GLM-5.3 743B lane** (TP=4, 4× Spark, tonyd2wild Int4-Int8Mix) with
  GLP-77 steering.
- GLM-5.3-Flash EXL3/B12X (brandonmusic, SM120) variant with two-loop
  steering hotfix — runtime-validated on 2× RTX PRO 6000.
- BENCHMARK.md steering-effectiveness results published.

### Changed
- DFlash2 aux-tap sweep: NULL result — taps are training-matched.
- Vector repos renamed to the `abliterated-cyber` convention.
- qwen38fn: PLE FP8 patch required for NVFP4 serving (Modal B200 finding).

## 2026-08-29

### Changed
- Hotfixes fail closed on runtime load failure; qwen38 patch made atomic
  across its two files; glm53 α default 2.0.

## 2026-08-27

### Added
- **GLM-5.3-Flash lane** (TP=4, 4× Spark) with GLP-44 — tonyd2wild's
  hardware-validated GB10 deployment folded in.
- **Qwen3.8-Flash-Next lane** (TP=2, 2× Spark, day-0 image) with GLP-47.

## 2026-08-23

### Added
- omp roles: offer routing every text role to the endpoint, not just the
  default.

## 2026-08-22 — the rebrand + the wizard

### Added
- **Rebrand: dspark-deploy → weightless**; the format named GLP (GGUF Layer
  Projection); `glp.*` becomes the canonical metadata namespace (legacy
  `dspark.*` alias later dropped); steering env vars unified to
  `WEIGHTLESS_STEER_*`; GLP-n naming convention (vector named by layer
  coverage).
- HF artifacts published under the GLP branding (GLP-29/GLP-49 filenames;
  canonical Qwen L10-58 vector).
- The setup wizard (install.py → setup.py): full chain — lane pick → env →
  steering validation → confirm-gated ssh deploy → omp provider + endpoint
  tests — curses TUI with CLI fallback, stdlib-only; mDNS host discovery;
  deploy preflight (container status + md5 comparison); endpoint smoke
  suite inside the UI; brand palette + animated feather logo; DEMO=1
  screenshot mode.
- omp harness suite for the deployed endpoint (endpoint / chat / tool-call
  / headless agent loop); BENCHMARK Run 008 (clock-lock A/B, prefill/decode
  interference).

### Changed
- Qwen lane hardware-validated: `STEER_MODE=lora` added, 2×2 results
  recorded (GGUF and LoRA both 24/32 refusal32).

### Fixed
- Qwen hotfix patches `qwen3_5.py` too — `Qwen3_5Model.__init__` skips its
  parent (the skip-parent trap).
- tests/03 retries the invalid-JSON serializer flake once with byte-context
  diagnosis.

## 2026-08-21

### Added
- Qwen TP=1 serving profile with projective steering; three-lane serving
  matrix (DSV4 TP=2, DSV4 TP=1 EXL3, Qwen TP=1).
- Live Anemll recipe state vendored into `recipe/anemll`; projective
  steering ported to the 0.25.2 stack as a boot hotfix.
- Refusal probe tool (`scripts/probe-refusal.py`).

### Changed
- Single-Spark DSV4 lane **rejected**: REAP pruning degrades the tail;
  serving env files moved out of the repo (sanitized examples shipped).

## 2026-08-19

### Fixed
- 1M prefill collapse localized to c128a width 8192 and worked around;
  512K context adopted (also free, narrowing the cliff to 1M alone).

## 2026-08-17 — first serving

### Added
- DeepSeek V4 Flash serving on vLLM v0.27.0 with steering applied
  post-docker (bind-mounted patched file over overlay builds).
- `BENCHMARK.md` with the first v027 measurements; spec-decode parity run;
  prefill/decode separated with the working method; 256K context settled
  (4× the window at no measurable cost); the 1M trade-off recorded (prefill
  collapses 23×).
- Control-vector format spec moved here from the retiring fork.
- `torch.compile` recorded as unsupported for DSV4, not merely off.
