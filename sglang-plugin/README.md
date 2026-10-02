# weightless-sglang: GLP steering as an SGLang plugin

Applies `h ← h − α(h·d̂)d̂` to the residual stream `h` (the running hidden
state that every decoder layer adds its output to) right after every
steered decoder layer, with per-layer unit directions `d̂` from a GLP GGUF
control vector (`../spec/GLP.md`); alpha (`α`) is the strength of the edit.
It is the SGLang counterpart of `../vllm-plugin/`: the same files, the same
`WEIGHTLESS_STEER_*` variables, the same loader and gates (it imports
`weightless_steer.container` and `weightless_steer.core`); only the wiring
into the engine differs, and SGLang itself is not modified.

How it hooks in: SGLang loads general plugins from the `sglang.srt.plugins`
entry point in every rank process (one process per GPU). `register()` adds
one AFTER hook on `ModelRunner.load_model`. When the weights are loaded,
that hook reads the GLP file, stores the directions and alpha as float32
GPU buffers and installs a steering site on each steered decoder layer,
before SGLang captures its CUDA graphs, so the edit is part of every
captured prefill and decode graph. The weights are never modified. Design:
`../docs/sglang-plugin-design.md`.

## Install

Into the environment that runs SGLang. The vLLM plugin package
(`weightless-steer`, in `vllm-plugin/`) is needed for its loader; vLLM
itself is not. `weightless-steer` is not on PyPI, so it must be installed
from this repository first or in the same `pip install` as the SGLang
plugin, as below.

Either install SGLang from PyPI:

```bash
pip install sglang
```

Or install SGLang from a git clone:

```bash
git clone https://github.com/sgl-project/sglang.git && pip install -e sglang/python
```

Then install the two plugin packages from this repository:

```bash
git clone https://github.com/msuiche/weightless.git && cd weightless
pip install ./vllm-plugin ./sglang-plugin
```

## Serve

```bash
export WEIGHTLESS_STEER_PATH=<vector.gguf>
python -m sglang.launch_server --model-path <model> --host <host> --port <port>
```

A model that needs several GPUs keeps its usual flags: `--pp-size` for
pipeline parallel (PP, each GPU holds a slice of the layers) or `--tp-size`
for tensor parallel (TP, each layer is split across the GPUs) on the rows
that accept TP. When steering is on, every rank logs a line like this one;
check it:

```
weightless GLP steering active (sglang): source=env tp_rank=0 layers=10..58 (49; 12 full-attn, 37 linear) alpha=1.000 hook=residual_stream_post_layer file_sha256=... diag=0 pp_rank=0 row=Qwen3_5ForConditionalGeneration install=hook site=decoder-layer output width=5120 kernel=triton (auto: fused Triton kernel, start-up self-check passed (width 5120, with residual))
```

With `WEIGHTLESS_STEER_PATH` unset the plugin registers nothing and SGLang
runs stock. With it set but the package not installed, SGLang never reads
the variable and also runs stock, so no log line means no steering.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `WEIGHTLESS_STEER_PATH` | unset | the GLP `.gguf` file. Unset: nothing is registered, SGLang runs stock |
| `WEIGHTLESS_STEER_ALPHA` | the file's `glp.alpha_default` | the strength: 0 changes nothing (bit-identical to stock), 1 removes the component along `d̂`, 2 reflects it |
| `WEIGHTLESS_STEER_LAYERS` | every layer in the file | comma list of layer ids to steer (global ids, as in the file); on a looped model, execution steps counted from 0 |
| `WEIGHTLESS_STEER_HOOK` | unset | if set, must equal the file's `glp.hook_point`; anything else fails the boot |
| `WEIGHTLESS_STEER_KERNEL` | `auto` | `auto`, `triton` or `torch` (see Kernel) |
| `WEIGHTLESS_STEER_MANIFEST_DIR` | unset | writes `weightless-manifest-<pid>.json` per rank: row, site, width, layers, alpha, file sha256, kernel and why, and the sites that ran in each CUDA-graph capture |
| `WEIGHTLESS_STEER_DIAG` | unset | `1`: eager-only per-layer statistics of the edit (use with `--disable-cuda-graph`) |
| `WEIGHTLESS_STEER_DIAG_DIR` | the working directory | where the statistics go |

`SGLANG_PLUGINS` is SGLang's own optional allowlist; if you set it, include
`weightless_steer`.

## Kernel

The edit needs, for every token row, the dot product of the stream with the
direction, then one scaled subtraction. The torch path does this with
ordinary torch operations: it converts the rows to float32, computes the
dot product and the result, and writes them back in the model's dtype. That
takes several passes over temporary copies of the rows (done in slices, so
the temporaries stay under 256 MB). The fused kernel
(`weightless_sglang/fused.py`, written in Triton) does the same steps in
one GPU program per token row: it reads the row once, keeps it in registers
and writes the result once.

Both paths add the products as integers after a fixed scaling, so the sum
does not depend on the order of the additions, and both round the same way:
they give the same bits, on any batch size. That is why the faster one can
be the default. With `auto`, every rank runs a start-up self-check before
the CUDA graphs are captured: the fused kernel against the torch path on
random rows at the row's width and residual form. The fused kernel is used
only when the bits match; otherwise (no Triton, no CUDA device, a mismatch
or a compile error) the rank falls back to the torch path and the log line
says why. `triton` forces the fused kernel and fails the boot if it cannot
run; `torch` forces the torch path.

Measured on Qwen3.8-27B NVFP4, one RTX 5090, CUDA graphs on, against stock:
the fused kernel costs 1.3 % on decode and 0.3 % on prefill with GLP-49
(1.7 % and 0.6 % with GLP-63); the torch path costs 9.7 % and 11.3 %
(12.1 % and 14.2 %). Details: `../BENCHMARK.md`.

## Architectures

One row per served model class in `weightless_sglang/archs/` (one module
per architecture, named as in the vLLM plugin where it has the same one). A
class that is not listed fails the boot with
`has no SGLang steering adapter`. Some models carry several parallel copies
of the residual stream instead of one, called hyper-connection streams (mHC
in GLM-5.3-Flash and DeepSeek-V4, iHC in Hy4-preview); the stream the edit
sees there is several times the hidden size wide. The install kinds:
**hook** (a torch forward hook on the layer), **forward wrap** (a wrapper
on the layer instance's `forward`, for a loop that calls
`layer.forward(...)`), **special** (a wrapper on the one call inside the
layer that produces the stream), **looped** (one hook per physical layer
that dispatches on the loop index).

| architecture | served classes | steered stream and site | status |
|---|---|---|---|
| Qwen3.8-27B | `Qwen3_5ForConditionalGeneration`, `Qwen3_5ForCausalLM` | `hidden + residual` after each layer; hook | GPU-validated, one RTX 5090: alpha 0 bit-identical to stock on 64/64 greedy sequences, fused kernel bit-identical to the torch path, fused cost 1.3 % decode |
| GLM-5.3-Flash | `Glm5NextForConditionalGeneration` | the 16384-wide mHC stream (4 x 4096) that `mhc.mlp_combine` returns, before the last layer's contract; special | GPU-validated, 3x RTX PRO 6000, PP 3 (one pipeline stage per GPU): alpha 0 equals stock on 64/64 prompts, cyber32 8 → 31/32, benign32 32/32, decode within 5 % of stock; TP > 1 refused |
| GLM-5.3 | `GlmMoeDsaForCausalLM` | `hidden + residual`, `topk_indices` passed through; hook | structure-tested |
| DeepSeek-V4-Flash | `DeepseekV4ForCausalLM` (model_type `deepseek_v4`) | the FFN write before the mHC fold (`ffn_out_pre_residual`), SGLang's fused mHC mode only; special | structure-tested |
| Hy4-preview | `HYV4ForCausalLM` | each of the 4 iHC streams, one direction per stream; hook | structure-tested |
| Qwen3.8-Flash-Next | `Qwen4ExpForConditionalGeneration` | the combined hyper-connection stream (4 x 2560); hook | structure-tested |
| Nemotron-3.5-Lightning | `NemotronHForCausalLM`, `NemotronHPuzzleForCausalLM` | `hidden + residual`; forward wrap | structure-tested; TP > 1 refused |
| Kimi-K3 | `KimiK3ForConditionalGeneration`, `KimiK3LinearForCausalLM` | the post-layer stream (whole in `hidden` with attention residuals); hook | structure-tested; TP > 1 refused |
| Nanbeige4.2-3B | `NanbeigeForCausalLM` | `hidden + residual` at each of 44 execution steps (22 layers x 2 loops); looped | structure-tested |
| Ouro | `OuroForCausalLM` | | refused: SGLang has no Ouro model |
| Inkling | `InklingForConditionalGeneration`, `InklingForCausalLM` | | refused: the post-layer stream never exists between layers |
| DeepSeek-V4.1 | `DeepseekV41ForCausalLM`, or V4.1 served as `DeepseekV4ForCausalLM` | | refused: its loop prepares the next layer's input from the unsteered stream |
| other names | `KimiLinearForCausalLM`, `DeepseekV4ForConditionalGeneration` | | refused: a different model under that name, or no such SGLang class |

Structure-tested means: fakes shaped like the SGLang classes, SGLang's own
decoder layers and model loop run on CPU, a parse of the SGLang model files
that pins what the row relies on, the published file installed at the
checkpoint's depth and width, and SGLang building the model from its public
config on the meta device. It has not run on a GPU. TP > 1 is unit-tested
but not yet run on GPUs. On GLM-5.3-Flash the per-layer edit check passes
on 40 of 44 layers against a bound of 1e-2 of the pre-edit component along
the direction; layers 21, 24, 29 and 33 measure 1.5e-2 to 2.8e-2, which
fails that strict bound but is 13 to 35 times below the bf16 rounding bound
of the 16384-wide stream (bf16 is the 16-bit float format the model's
activations are stored in, so its rounding sets the smallest error that can
be measured).

Use each file's own `glp.alpha_default` (the plugin's default) unless you
have measured another value; the model cards give the safe range.

## Behaviour contract

- **Fail-closed.** With `WEIGHTLESS_STEER_PATH` set, a missing file, a
  non-`project` file, a wrong `glp.hook_point`, width, `model_hint` or
  `glp.structure`, layers out of range or filtered to nothing, alpha
  multipliers, a rank-k file, or a nonfinite alpha stops the boot. All
  checks run inside `load_model` (SGLang only logs errors raised by
  `register()`), so a server asked to steer never serves unsteered.
- **Exact names.** The served class and every steered decoder layer must
  match a row by exact class name (never `isinstance`); anything else fails
  the boot. So do TP > 1 on a row that refuses it and two-batch overlap
  (`--enable-two-batch-overlap`, which splits each batch in two to overlap
  compute with communication, and runs the layers through a path the sites
  are not on).
- **The slot check.** At each steered layer's first forward the site checks
  the slots of the layer's output that the row reads: a bool, None, or an
  integer or 8-bit float tensor where the row expects a stream, or a stream
  of another width, stops the boot. It checks the type, the dtype (a float
  of 16 bits or more) and the last dimension (3-D on a per-stream row), and
  on Kimi-K3 accepts None in the residual slot. It cannot tell apart two
  slots of the same dtype and width, nor count the streams of a per-stream
  row; the structure tests pin those. On a looped row the layer number in
  its message is the execution step.
- **The fired check.** After every forward that runs Python (the warmup,
  each CUDA-graph capture, every eager forward) each rank checks that every
  steered site ran, and raises otherwise.
- The MTP/NEXTN draft model (the small extra model that speculative
  decoding uses to guess the next tokens) stays stock. Alpha 0 is
  bit-identical to stock. Reloading weights from disk replaces the model
  and drops the sites: restart the server instead.

## Tests

No GPU needed (CUDA-only cases skip without one):

```bash
cd sglang-plugin && python -m unittest discover -s tests
cd sglang-plugin && python -m pytest tests
```

Optional, to run more: `WEIGHTLESS_TEST_GLP_DIR` (the folder of the
published GLP files), `WEIGHTLESS_TEST_CONFIG_DIR` (a folder of
`<org>__<repo>/config.json`), `WEIGHTLESS_TEST_GLM_CONFIG`,
`WEIGHTLESS_TEST_QWEN_CONFIG` and `WEIGHTLESS_TEST_DSV41_CONFIG` (one
checkpoint's config.json each), and `WEIGHTLESS_TEST_SGLANG_TREES` (more
SGLang package folders for the structure tests, os.pathsep separated). The
tests that run SGLang's own code need SGLang in the environment. Every skip
names what is missing. The suite has 365 tests; on SGLang main with every
optional variable set, only the 6 CUDA-only cases skip on a CPU host, and
with none set 87 skip.
