# Upstreaming GLP steering into vLLM: extension-point survey and recommended path

Status: design doc for what shipped. Path (1) is the fleet: ten arch
adapters in `vllm-plugin/`, every one GPU-validated on Modal
(2026-09-18/19), 223 offline tests. No upstream PR has been opened or is
proposed here as a first step.

SGLang counterpart: `sglang-plugin/`, with the same loader and gates and no
SGLang source change; design in `docs/sglang-plugin-design.md`.

Today we serve steered models through `patches/hotfix-*-steering-projective.py`:
fail-closed boot scripts that rewrite vLLM model files inside the container
with anchor-matched string replacement. Every lane re-implements the same
loader and the same CUDA-graph discipline; every image bump risks a broken
anchor. This doc surveys what vLLM's supported extension points can actually
do, and picks a path off the monkey-patch fleet.

Reference checkout: local `../vllm` is **v0.27.0 + 4 weightless commits**
(`e32b56e2`, branch `dspark-steering-v027`; the four commits are the retired
DSV4 overlay now kept as `patches/0001-*.patch`). Upstream main is at
**v0.29.0** (2026-09-08): Model Runner V2 is the default for all models,
MRV1 is deprecated with removal targeted at v0.32
([releases](https://github.com/vllm-project/vllm/releases)). The plugin
machinery below is byte-identical between the local v0.27 tree and current
main (`vllm/plugins/__init__.py` diff: none).

## 1. What the existing hotfixes actually patch

Read in full: `hotfix-nemotron35-steering-projective.py` (representative
single-vector lane), `hotfix-dsv4-steering-projective.py`,
`hotfix-glm53-steering-projective.py`, and the dual-vector probe patch
`../refusal-research/experiments/20260910-qwen38-27b-hedging-dir/staging/patch_qwen38_hedge.py`.

**Generic (verbatim or near-verbatim across every lane):**

- The GGUF v3 container reader and the full spec gate set (`glp.mode`
  enforcement, `glp.hook_point` match, `glp.layer_ids_zero_based`
  cross-check, `direction.0` rejection). Carried as one `GGUF_SRC` constant
  and copied between files — the header of `patch_qwen38_hedge.py` says
  "VERBATIM from hotfix-qwen38 ... the container contract is shared across
  lanes, so the code that enforces it must be too."
- Env wiring: `WEIGHTLESS_STEER_PATH / _ALPHA / _LAYERS / _HOOK`.
- CUDA-graph discipline: dense zero-padded `_steer_stack` indexed by global
  layer id, alpha as a registered tensor buffer, unconditional per-layer
  apply (no Python `if` inside the traced region), non-persistent buffers so
  `load_weights` never sees them. All four were measured failure modes.
- Fail-closed boot semantics.

**Model-specific (the part that cannot be shared by construction today):**

- The hook-site *expression*. Three different ones exist:
  - nemotron_h / qwen3_next-style fused add+norm: post-layer stream is
    `h = hidden_states + residual`, write back `hidden_states <- h' - residual`.
  - DSV4 (Anemll tree): the model-loop `hidden_states` is the pre-fold FFN
    writer output (`ffn_out_pre_residual`) — the hook-site label was wrong
    for months before a shape probe corrected it.
  - GLM-5.3 mHC: the stream is widened `[T, n, hidden]` with `hc_post`
    deferred and fused into the next layer; the apply materializes
    `MHCPostOp` by hand and has a special case for the last mHC layer.
- Anchor text: class names, loop shapes, constructor signatures. This is
  what breaks on image bumps and what the `--status`/`--check` machinery
  exists to detect.
- `patch_qwen38_hedge.py` additionally patches
  `vllm/v1/worker/gpu_model_runner.py` (per-step probe gate + flush). Note
  this is the **only** runner-level patch; the steering applies all live in
  `model_executor/models/*`, which is shared by MRV1 and MRV2. The runner
  patch is for activation *capture* gating, and it sits on the deprecated
  MRV1 clock.

So: the steering core (load, gate, apply math, compile safety) is already a
library; what forks per model is a ~10-line hook-site adapter plus the
registration of buffers.

## 2. vLLM's extension points, and what each can touch

Surveyed against the local v0.27 tree; line references are from it.

**`vllm.general_plugins` entry points** (`vllm/plugins/__init__.py:18,77`).
Arbitrary callables executed in process0, the engine-core process, and every
worker process (`vllm/engine/arg_utils.py:795`, `vllm/v1/engine/core.py:117`,
`vllm/v1/worker/worker_base.py:247`, and the model-inspection subprocess
`vllm/model_executor/models/registry.py:1495`). vLLM itself uses this group
to register its LoRA resolvers (`pyproject.toml:46`). A plugin here runs
*before* model-class resolution in every process that resolves one. This is
the load-bearing fact: it can call `ModelRegistry.register_model` and it
lands everywhere that matters.

**`ModelRegistry.register_model`** (`vllm/model_executor/models/registry.py:1057`).
Public, documented, intended for out-of-tree models. Registering an arch name
that already exists is legal and simply overwrites (debug-logged,
registry.py:1077). So a plugin can shadow a built-in arch with a subclass —
no upstream change required.

**`worker_extension_cls`** (`vllm/config/parallel.py:265`, wired in
`vllm/v1/worker/worker_base.py:261-284`). Dynamically mixed into the worker
class; new methods become reachable via `collective_rpc`. This is how
vLLM-Lens and IBM's vLLM-Hook install forward hooks into a running engine
([IBM/vLLM-Hook](https://github.com/IBM/vLLM-Hook),
[preprint](https://arxiv.org/html/2603.06588v1),
[vLLM-Lens writeup](https://www.lesswrong.com/posts/3bs27nZQuEcKhXf7q/vllm-lens-fast-interpretability-tooling-that-scales-to)).
Genuine extension point, but hooks installed *after* CUDA-graph capture are
not in the captured graph; install ordering relative to
`GPUModelRunner.load_model` (`vllm/v1/worker/gpu_model_runner.py:5303`) and
capture has to be managed by hand, and module discovery by class name is
fragile across arch conventions (the three hook-site expressions above are
why).

**Logits processors** (`vllm/v1/sample/logits_processor/interface.py`).
Insufficient by construction: the interface is `apply(logits) -> logits`.
Hidden states never cross it. Steering needs the residual stream, not the
final distribution.

**IO processors, endpoint plugins, stat loggers, platform plugins**
(`vllm/plugins/`). All process-0 or frontend-scoped; none run inside the
worker forward path. Dead ends for steering.

**LoRA — the precedent for a first-class path.** LoRA is served per-request
because every layer class opts in via `SupportsLoRA`
(`vllm/model_executor/models/interfaces.py:564`) and the model files carry
LoRA-aware modules; the runner wraps the loaded model when
`lora_config` is set (`gpu_model_runner.py:5327`). LoRA works through weight
space, so it never needed a hidden-state hook. Steering has no equivalent:
there is no `SupportsSteering`, no per-layer edit site in the model loop.

**Hidden-states extraction (upstream, read-only).** RFC
[vllm-project/vllm#33118](https://github.com/vllm-project/vllm/issues/33118)
and the [2026-03-30 blog](https://vllm.ai/blog/2026-03-30-extract-hidden-states)
landed a supported path for getting hidden states *out* (aux-hidden-states
mixins, `extract_hidden_states` spec-decode method, KV-connector transport —
see `EagleModelMixin._maybe_add_hidden_state`, interfaces.py:1405, which even
computes `hidden_states + residual`, i.e. upstream has already blessed that
exact expression as "the stream"). It does not write back. It proves
upstream will accept per-layer stream access in principle, and it is the
natural skeleton for a mutating counterpart.

**Control vectors as a feature request:** upstream issue
[vllm-project/vllm#3451](https://github.com/vllm-project/vllm/issues/3451)
(open since 2024-03, stale-labeled, no linked PR). Nobody has built it.

**EasySteer** ([arXiv 2509.25175](https://arxiv.org/abs/2509.25175),
[ZJU-REAL/EasySteer](https://github.com/ZJU-REAL/EasySteer), EMNLP 2026
demo): the most mature steering framework on vLLM. Worth noting how it ships:
**a fork overlay** — `rsync -a vllm-steer/vllm/ "$VLLM_DIR"/` over an
official wheel, currently tracking v0.29.0, with a `vllm.steer_vectors`
module inside the fork (`SteeringSpec`/`ApplySpec`, GGUF vectors — the same
container convention). Even the group with a paper and a maintained project
did not manage pure-plugin steering; they patch the tree. That is evidence,
not a coincidence.

## 3. The honest assessment

### (a) What can be a plugin today, zero upstream changes

A `vllm.general_plugins` package that shadows each steered arch via
`ModelRegistry.register_model("NemotronHForCausalLM", "weightless_steer.archs.nemotron_h:SteeredNemotronHForCausalLM")`,
where each `Steered*` class subclasses the upstream model class, registers
the `_steer_stack`/`_steer_alpha` buffers, and overrides the inner model's
`forward` to apply the projection at the layer loop.

- Kills the sed-patch machinery: no anchors, no in-container file edits,
  `--check` becomes a normal import-time validation.
- Keeps every piece of the discipline (dense stack, tensor alpha,
  unconditional apply, fail-closed) — it all ports as-is.
- Still per-arch: the subclass overrides a method whose signature and return
  convention (`(hidden_states, residual)` tuples, mHC widened streams) is an
  upstream internal that can change in any release. A break now fails loudly
  at import (`AttributeError`/signature mismatch) instead of silently
  serving unsteered or garbling — strictly better failure mode than anchors.
- CUDA graphs and torch.compile: the apply lives inside the overridden
  forward, so it is traced and captured exactly as the patched version is
  today. No new compile-safety work.
- The probe/flush lane (`patch_qwen38_hedge.py`'s runner patch) does **not**
  fit this route — per-step capture gating is runner logic. Keep it as a
  research-only patch, or rebuild capture on upstream's
  hidden-states-extraction machinery when it matures.

This works on v0.27 through current main. It is what "plugin" can mean
today.

### (b) The smallest upstream ask

One mutating counterpart to the extraction path, in the style of
`SupportsLoRA`:

- A `SupportsSteering` protocol in `vllm/model_executor/models/interfaces.py`
  plus a documented per-layer edit point: a method on the model loop,
  `hidden_states, residual = self._steer_layer(layer_idx, hidden_states, residual)`,
  default no-op, called where `EagleModelMixin._maybe_add_hidden_state` is
  called from today (the extraction sites are exactly the stream expressions
  steering needs to edit).
- A `SteeringConfig` on `ModelConfig` (path, mode, hook point) so buffers
  and gates initialize at model build, before compile and capture.
- Nothing per-request, nothing in the scheduler, no runner changes. v1.

That is a ~100-line upstream diff plus docs, and it converts every
steering framework (ours, EasySteer, vLLM-Hook) from fork to plugin. The
natural home for the conversation is issue #3451, citing RFC #33118 as the
read-only half that already landed. Realistic timeline: months, acceptance
far from guaranteed — steering is safety-adjacent and the MRV2 migration is
consuming review bandwidth. Do not block on it.

### (c) Recommended path

Three viable paths:

1. **Pure plugin via registry shadowing ((a)).** Effort: ~1 week for the
   steering core + one arch adapter, then ~0.5 day per additional lane;
   each adapter is testable offline against the vendored reference files in
   `patches/reference/`. Risk: moderate — per-arch upstream coupling remains,
   but failures move from silent-at-serve to loud-at-import. No upstream
   dependency.
2. **Upstream hook + plugin ((b)).** Effort: small code, large process.
   Risk: timeline and acceptance. Right long-term answer, wrong next step.
3. **EasySteer-style fork overlay.** Effort: high and recurring (rebase per
   release, per-arch integration on their controller model). Risk: high
   maintenance, and it replaces our format/gates with theirs. Only sensible
   if we adopt EasySteer wholesale, which we have no reason to: our GLP
   container, fail-closed gates, and measured alphas are the product.

**Recommendation: do (1) now, pursue (2) in parallel at comment-on-#3451
level of investment.** The plugin package subsumes today's hotfixes lane by
lane; when/if upstream lands a hook, the per-arch subclasses shrink to
interface implementations and the steering core is untouched.

Skeleton (as implemented — nemotron_h was the first lane; the fleet now
covers all ten serving archs):

```
vllm-plugin/
├── pyproject.toml                     # entry_points: [vllm.general_plugins]
│                                      #   weightless_steer = weightless_steer.plugin:register
├── weightless_steer/
│   ├── plugin.py                      # register(): reads WEIGHTLESS_STEER_*,
│   │                                  #   shadows archs via ModelRegistry.register_model
│   ├── container.py                   # read_gguf_cvec(), load_control_vector()
│   │                                  #   — GGUF_SRC as a real module, one copy, tested
│   ├── core.py                        # class SteeringCore:
│   │                                  #   from_env(config) -> Self | None   (fail-closed)
│   │                                  #   register_buffers(module)          (dense stack,
│   │                                  #     tensor alpha, non-persistent)
│   │                                  #   apply(layer_idx, h) -> h          (projective,
│   │                                  #     unconditional, graph-safe)
│   └── archs/
│       ├── base.py                    # class SteeredModelMixin: __init__ wires
│       │                              #   SteeringCore; forward-loop adapter contract
│       ├── nemotron_h.py              # SteeredNemotronHModel(NemotronHModel, Mixin):
│       │                              #   h = hidden + residual; write back
│       ├── glm5_next.py               # mHC adapter: materialize MHCPostOp,
│       │                              #   flatten [T, n*hidden], last-layer case
│       └── qwen3_next.py              # shared by qwen3_next/qwen3_5
│                                      #   (the qwen3_5 skip-parent trap handled here)
└── tests/
    ├── test_container.py              # spec gates, bad-file corpus
    ├── test_core.py                   # apply math, zero-row no-op, PP indexing
    └── test_archs/                    # one structure test per adapter, diffed
                                       #   against patches/reference/ as today
```

## References

- Local checkout: `../vllm` @ `e32b56e2` (v0.27.0+4, branch
  `dspark-steering-v027`). Upstream main: v0.29.0,
  [releases](https://github.com/vllm-project/vllm/releases).
- Plugin machinery: `vllm/plugins/__init__.py`, `pyproject.toml:46`,
  `vllm/model_executor/models/registry.py:1057`,
  `vllm/v1/worker/worker_base.py:247,261`,
  `vllm/config/parallel.py:265`.
- Upstream issues/RFCs:
  [control vectors #3451](https://github.com/vllm-project/vllm/issues/3451),
  [hidden-states extraction RFC #33118](https://github.com/vllm-project/vllm/issues/33118),
  [extraction blog](https://vllm.ai/blog/2026-03-30-extract-hidden-states).
- Precedents: [EasySteer](https://arxiv.org/abs/2509.25175) /
  [repo](https://github.com/ZJU-REAL/EasySteer) (fork overlay on v0.29.0),
  [IBM vLLM-Hook](https://github.com/IBM/vLLM-Hook) /
  [preprint](https://arxiv.org/html/2603.06588v1) (worker-extension hooks),
  [vLLM-Lens](https://www.lesswrong.com/posts/3bs27nZQuEcKhXf7q/vllm-lens-fast-interpretability-tooling-that-scales-to)
  (worker-extension steering per request).
- GLP container spec: `spec/GLP.md`. Hotfix fleet: `patches/README.md`.
