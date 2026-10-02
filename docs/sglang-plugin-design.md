# Upstreaming GLP steering into SGLang: extension-point survey and recommended path

Status: design doc for what ships in `sglang-plugin/`. Nine architectures have
a row; Qwen3.8-27B (one RTX 5090) and GLM-5.3-Flash (3x RTX PRO 6000,
pipeline-parallel) are GPU-validated, the other seven are structure-tested on
CPU. No SGLang change is needed, and none is proposed as a first step.

This is the SGLang counterpart of `docs/vllm-plugin-design.md`. The GLP file,
the loader and the gates are the vLLM plugin's (`weightless_steer.container`,
`weightless_steer.core`), so a file behaves the same on both runtimes. Only
the wiring into the engine is new.

Reference checkout: SGLang upstream main at `2f5c9ac43d76` (2026-09-25).
File:line references are to that tree, under `python/sglang/srt/`.

## 1. What the plugin has to do

A GLP file holds one unit direction `d` per steered layer. The edit is a
projection on the residual stream after the layer
(`glp.hook_point = residual_stream_post_layer`):

```
h' = h - alpha (h . d) d
```

`h` is the residual stream, the running sum every layer adds its output to.
`alpha` is the strength: 0 changes nothing, 1 removes the component along `d`,
2 reflects it.

SGLang decoder layers mostly do not return `h`. A Qwen3.5 layer returns
`(hidden, residual)` and the next layer's fused add+RMSNorm adds them
(`models/qwen3_5.py:1833`), so the stream after layer `i` is
`h = hidden + residual`. Other models carry a widened stream (mHC,
hyper-connections, iHC) or leave the FFN write pending until the next layer
(DeepSeek-V4). The plugin steers the same stream at the same site as the vLLM
plugin, per model (section 4).

The discipline is the vLLM plugin's: a dense direction stack indexed by global
layer id and a tensor alpha in GPU buffers, an unconditional apply with no
host sync, and fail-closed checks before anything serves.

## 2. SGLang's extension points, and what each can touch

**General plugins** (entry point group `sglang.srt.plugins`). SGLang calls
`load_plugins()` at the start of every scheduler process, one per GPU rank,
before the model exists (`managers/scheduler.py:6020`). A plugin's
`register()` can put BEFORE, AFTER, AROUND or REPLACE hooks on any SGLang
function through `HookRegistry` (`plugins/hook_registry.py:373-379`). This is
the supported way to extend SGLang without forking it. One catch: SGLang
catches and only logs an exception raised by `register()`
(`plugins/__init__.py:83-84`), so a plugin cannot fail the boot there; it has
to fail inside the code it hooks.

**`--forward-hooks`** (a JSON list of torch forward hooks). SGLang registers
them after CUDA-graph capture, on purpose
(`model_executor/model_runner_components/cuda_graph_setup.py:430-437`). A CUDA
graph is a recording of GPU kernels replayed without running Python, so a hook
added after the recording never runs on a replay. With graphs on (the default)
the steering would apply to some steps and not others, with no error. Usable
only with `--disable-cuda-graph`, as a slow reference.

**Model registry** (`ModelRegistry.register(..., overwrite=True)`;
`models/registry.py:32-45,154-157`). A plugin can replace a model class by
name, as the vLLM plugin does, but the Qwen3.5 multimodal wrapper builds its
language model directly (`models/qwen3_5.py:2290-2296`), and a replacement
class would copy the upstream layer loop, which changes often.

**Weight loaders, platform plugins, `--model-impl transformers`, logits
processors**: the first runs on weight updates only, the second abstracts
devices, the third loses SGLang's quantized and linear-attention kernels, the
last only sees the final logits. None is a steering seam.

## 3. The honest assessment

### (a) What can be a plugin today, zero upstream changes

A general plugin whose `register()` puts one AFTER hook on
`ModelRunner.load_model` (`weightless_sglang/plugin.py`). `load_model` returns
after the weights load and before the CUDA graphs are captured
(`model_executor/model_runner.py:654` and `:1100`). The hook loads the GLP
file, registers the buffers and installs a site on each steered decoder layer
(`install.py`, `install_steering`), so the sites' kernels are recorded into
every captured prefill and decode graph.

- No SGLang source change and no copied model code: the site sits on the
  layer's output (or on one call inside the layer), so upstream can change the
  loop as long as the layer's output keeps its shape. If it does not, the site
  refuses to run.
- Every check runs inside `load_model`, where an exception stops the boot, so
  a server asked to steer never serves unsteered.
- Still per architecture: each model's output tuple is an upstream internal,
  declared as data (section 4) and pinned by tests that parse the SGLang model
  files.

### (b) The smallest upstream ask

Nothing is required. Two small SGLang additions would help, and neither is
needed by the plugin:

- An opt-in key for `--forward-hooks`, such as `"capture": true`, that
  registers a graph-safe hook in `load_model` instead of after capture. Any
  steering hook could then be installed from a server flag. About 15 lines in
  SGLang.
- Named server flags (`--weightless-steer-path`, `-alpha`, `-layers`) that
  call the same `install_steering` from `load_model`. The setting would show
  in the server arguments, and a missing plugin package would fail the boot
  instead of serving stock.
  `install_steering(runner, source="server_args", path=..., alpha=..., layers=...)`
  already takes explicit values for such a caller.

### (c) Recommended path

Use (a). It is what ships and what was validated. Raise (b) with SGLang
maintainers only if they want a named feature; per-request alpha (per-token
alpha rows filled from the batch before each graph replay, like the vLLM
"milestone 2" work) comes after that.

Layout, mirroring `vllm-plugin/`:

```
sglang-plugin/
├── pyproject.toml            # entry point [sglang.srt.plugins]
│                             #   weightless_steer = weightless_sglang.plugin:register
├── weightless_sglang/
│   ├── plugin.py             # register(): AFTER hook on ModelRunner.load_model
│   ├── install.py            # install_steering(): loader and gates, buffers, sites,
│   │                         #   kernel choice, the log line and the manifest
│   ├── core.py               # the edit, the site, the tuple codec, the slot check,
│   │                         #   the fired check, the forward wrapper, the diagnostics
│   ├── fused.py              # the same edit as one Triton kernel (the default on CUDA)
│   └── archs/
│       ├── base.py           # ArchRow; the special-handler registries
│       ├── __init__.py       # ARCH (served class -> row) and REFUSED (class -> reason)
│       └── qwen38.py, glm5next.py, glm53xl.py, dsv4.py, hy4.py, qwen38fn.py,
│           nemotron_h.py, kimi_k3.py, nanbeige.py   # one module per architecture
└── tests/                    # CPU; the CUDA cases run when a GPU is visible
    ├── test_core.py, test_plugin.py
    └── test_archs/           # one file per architecture, test_refused.py, test_real_sglang.py
```

## 4. Architecture rows

What differs between architectures is data: one `ArchRow` per served model
class names the backbone path, the exact decoder-layer class names (never
`isinstance`), the hook points, the stream width, the layer's output tuple
(its length, the edited slot, the residual slot or none), the install kind,
the execution ids, whether TP > 1 is allowed and the accepted `model_hint`
values. `install_steering` is generic.

| row | install kind | site | why this kind |
|---|---|---|---|
| Qwen3.8-27B | hook | `(hidden, residual)`: `hidden + residual` | the loop calls `layer(...)` and the stream exists between layers |
| GLM-5.3 | hook | `(hidden, residual, topk)`: `hidden + residual`, `topk` passed through | as Qwen3.5 |
| Hy4-preview | hook, per stream | `(streams [T, 4, 6144], topk)` | the layer merges its iHC streams itself; one direction per stream |
| Qwen3.8-Flash-Next | hook | `(stream [T, 4 x 2560], None)` | the layer combines its hyper-connection streams itself |
| Kimi-K3 | hook | `(hidden, residual or None, flag)` | with attention residuals the MLP folds the prefix sum, so `hidden` is the whole stream |
| Nanbeige4.2-3B | hook, looped | `(hidden, residual)` at each of 22 x 2 execution steps | one hook per physical layer dispatches on the loop index the loop passes (a Python int, fixed per captured graph) |
| Nemotron-3.5-Lightning | forward wrap | `(hidden, residual)` | the loop calls `layer.forward(...)`, which bypasses forward hooks, so the instance's `forward` is wrapped |
| GLM-5.3-Flash | special | the output of `layer_communicator.mhc.mlp_combine`, `[T, 4 x 4096]` | the last layer contracts its stream inside the layer, so the site is the call that materialises it |
| DeepSeek-V4-Flash | special | `output[0]` of the fused-mode 4-tuple, the FFN write (`ffn_out_pre_residual`) | no post-layer stream exists between layers in the fused mode; the file is measured at the FFN write |

Refused by name, with the reason in the boot error: Ouro (no SGLang model),
Inkling (each layer's short convolution is deferred to the next layer, so the
post-layer stream never exists), DeepSeek-V4.1 (its loop prepares the next
layer's input from the unsteered stream), and two names that are a different
model or no SGLang class.

At each steered layer's first forward the site checks that the output slots it
reads hold what the row declares (the type, a float of 16 bits or more, the
row's width; None only where the row allows it) and fails the boot otherwise.
After every forward that runs Python (the warmup, each capture, each eager
forward) a check on the backbone raises if a steered site did not run, so a
model loop that stops reaching a site fails the server instead of serving
unsteered.

## 5. The edit per layer, and the kernel

For a site that returns `(x, r)` (`x` = hidden, `r` = residual, or none):

```
c  = (f32(x) + f32(r)) . d          # the stream's component along d, in float32
x' = round(f32(x) - (alpha * c) d)  # rounded once, to the model dtype
return (x', r)                       # the residual is returned untouched
```

In exact arithmetic `x' + r = h - alpha (h . d) d`. This form gives back `x`
bit for bit at alpha 0 and never rounds `x + r` to bf16 first, which would
lose the low bits of `x` next to very large residual values.

The dot product is a fixed-point sum: each product is scaled by a power of
two, cut to a 64-bit integer, and the integers are added, so the result does
not depend on the order of the additions, the batch size or how the GPU splits
the sum. The torch path runs these steps as torch operations, in slices that
keep its temporaries under 256 MB. The fused kernel (`fused.py`) runs them in
one Triton program per token row, reading `x` and `r` once and writing `x'`
once, and rounds like torch. The two give the same bits, so
`WEIGHTLESS_STEER_KERNEL=auto` (the default) uses the fused kernel after a
start-up self-check against the torch path on every rank, and falls back to
torch with the reason in the log line.

## 6. CUDA graphs, TP, PP

**CUDA graphs.** Prefill, decode and speculative verify go through the same
layer loop. The site has no host sync, no Python branch on tensor values and
no data-dependent allocation, and reads the stack and alpha from GPU buffers,
so its kernels are recorded once and stay valid. On GLM-5.3-Flash, graph
decode equals eager decode bit for bit on a steered boot pair.

**Tensor parallel.** SGLang may hand the next layer a partial sum whose
all-reduce is deferred. The site completes an `UnreducedOutput` with SGLang's
own `reduce_output` and refuses the older partial-sum marker and a deferred
MoE handoff. Rows whose outputs may be partial sums at the site refuse TP > 1.
TP > 1 is unit-tested, not yet run on GPUs.

**Pipeline parallel.** The stack is indexed by global layer id; each rank
installs only the steered layers in its own `[start_layer, end_layer)`.
GPU-validated on GLM-5.3-Flash with three ranks.

**Speculative decoding.** The MTP/NEXTN draft runs on its own `ModelRunner`
and stays stock, as in the vLLM plugin.

## 7. Validation

On GPUs (numbers in `BENCHMARK.md`, "SGLang plugin"): Qwen3.8-27B on one
RTX 5090, where alpha 0 is bit-identical to stock, the fused kernel is
bit-identical to the torch path and costs 1.3 % on decode against the torch
path's 9.7 %; and GLM-5.3-Flash on 3x RTX PRO 6000 with PP 3, where alpha 0
equals stock on 64 of 64 prompts, cyber32 delivery goes 8 → 31/32 with
benign32 at 32/32, and decode stays within 5 % of stock.

The CPU suite (`sglang-plugin/tests/`) checks the edit against the vLLM and HF
references, the recorded bits of the first release and of the GLM handler,
every row on fakes and on SGLang's own layers and loops, and every published
GLP file on its row and on every other row.

Not done yet: rank-k files, `add` mode, per-request alpha, and re-installing
the sites after a weight reload from disk (which replaces the model object).

## References

- SGLang upstream main `2f5c9ac43d76` (2026-09-25):
  `managers/scheduler.py:6020` (`load_plugins()` per rank),
  `plugins/__init__.py:83-84` (errors only logged),
  `plugins/hook_registry.py:373-379` (AFTER hook),
  `model_executor/model_runner.py:654,1100` (`load_model`, graph capture),
  `model_executor/model_runner_components/cuda_graph_setup.py:430-437`
  (`--forward-hooks` after capture), `models/qwen3_5.py:1833,2290-2296` (layer
  loop, multimodal wrapper), `layers/communicator.py` (`UnreducedOutput`,
  `reduce_output`), `models/registry.py:32-45,154-157`.
- SGLang plugin docs: `docs/docs/hardware-platforms/plugin.mdx` in the SGLang
  repository.
- This repository: `sglang-plugin/README.md`,
  `vllm-plugin/weightless_steer/core.py` and `archs/base.py` (the shared
  loader and the vLLM site), `docs/vllm-plugin-design.md`, `spec/GLP.md`.
