# weightless-sglang: GLP steering as an SGLang plugin

Applies `h ← h − α(h·d̂)d̂` to the residual stream `h` (the running hidden
state that every decoder layer adds its output to) right after every
steered decoder layer, with per-layer unit directions `d̂` from a GLP GGUF
control vector (`../docs/spec/GLP.md`); alpha (`α`) is the strength of the edit.
It is the SGLang counterpart of `../plugins/vllm/`: the same files, the same
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
(`weightless-steer`, in `plugins/vllm/`) is needed for its loader; vLLM
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
weightless GLP steering active (sglang): source=env tp_rank=0 layers=10..58 (49; 12 full-attn, 37 linear) alpha=1.000 hook=residual_stream_post_layer file_sha256=... diag=0 pp_rank=0 row=Qwen3_5ForConditionalGeneration install=hook site=decoder-layer output (residual stream) width=5120 layout=stage-boundary kernel=triton (auto: fused Triton kernel, start-up self-check passed (width 5120, with residual))
```

`layout=` names the SGLang layer layout the sites were installed on (see
SGLang layouts below). Each rank reads its `tp_rank`, `tp_size`, `pp_rank`
and `pp_size` from SGLang's ModelRunner when it has them, else from SGLang's
`get_parallel()`; when neither answers, the boot stops. The manifest records
which (`rank_source`).

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
| `WEIGHTLESS_STEER_HC_COPIES` | `mean` | DeepSeek-V4.1 only: how the edit lands on the 4 hyper-connection copies. `mean`: every copy gets the same subtraction, alpha times the mean's component along the direction (the mean is projected, the copy differences stay). `each`: every copy is projected on its own (its mean is projected too). See Architectures |

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

DeepSeek-V4.1's `mean` edit has its own fused kernel (`_glp_hc_mean_kernel`:
one program per token, its 4 copies of 5120, the same steps as the torch path
`archs/dsv41.hc_mean_delta`) with its own start-up self-check
(`self_check_hc_mean`); the `each` edit is the row kernel above on the copies
as rows.

Measured on Qwen3.8-27B NVFP4, one RTX 5090, CUDA graphs on, against stock:
the fused kernel costs 1.3 % on decode and 0.3 % on prefill with GLP-49
(1.7 % and 0.6 % with GLP-63); the torch path costs 9.7 % and 11.3 %
(12.1 % and 14.2 %). Details: `../BENCHMARK.md`.

## SGLang layouts

Between 2026-09-24 and 2026-09-30 SGLang rebuilt its decoder layers on
"stage boundaries" (`sglang.srt.layers.layer_boundary`; upstream #41553 for
Qwen3.5, #41554 and #41750 for GLM-5.3-Flash's mHC, #41813 for the ranks).
A layer built that way holds `attn_boundary` and `ffn_boundary`, returns the
FFN's output alone and keeps the residual in `forward_batch.residual_stream`;
mHC moved from the layer communicator into the FFN boundary's residual
update. The plugin serves both layouts and picks one per rank from what the
layers hold:

- Qwen3.8-27B: on the older layout the hook edits `hidden` of the
  `(hidden, residual)` pair the layer returns; on stage boundaries it reads
  the layer's output and the residual from the forward batch's residual
  stream and writes the edited output in place (the stream must still own
  the tensor the layer returned). Same stream, same edit, same bits.
- GLM-5.3-Flash: on the older layout the site is
  `layer_communicator.mhc.mlp_combine`; on stage boundaries it is the
  `MHCState.apply_post` call inside the FFN boundary's `_FfnUpdate.update`
  (the feed-forward post-combine), exactly once per layer and before the
  last layer's `hc_contract`; the attention update's `apply_post` stays
  unsteered. Same stream, same edit, same bits.
- DeepSeek-V4.1 (SGLang builds it on stage boundaries only, as
  `DeepseekV4CudaDecoderLayer`): the site is the FFN boundary's `HcUpdate.post`,
  the call that folds the layer's FFN write into the 4 hyper-connection
  copies; see Architectures.
- Every other row is still validated on the older layout only. On a stage-
  boundary layer it fails the boot ("needs a port to the layer-boundary
  API") before anything is hooked; a layer layout the plugin does not know
  fails the boot too.

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
| Qwen3.8-27B | `Qwen3_5ForConditionalGeneration`, `Qwen3_5ForCausalLM` | `hidden + residual` after each layer; hook (both SGLang layouts) | older layout (SGLang 2f5c9ac4): GPU-validated, one RTX 5090: alpha 0 bit-identical to stock on 64/64 greedy sequences, fused kernel bit-identical to the torch path, fused cost 1.3 % decode. Stage boundaries (SGLang 15aa0b5a): CPU-validated on SGLang's real classes, plus one steered boot on the RTX 5090 (GLP-63, fused kernel self-check passed, every CUDA-graph capture fired 63 of 63); no alpha-0 identity run there yet |
| GLM-5.3-Flash | `Glm5NextForConditionalGeneration` | the 16384-wide mHC stream (4 x 4096) after the feed-forward post-combine (`mhc.mlp_combine`, or `apply_post` in the FFN boundary's update), before the last layer's contract; special (both SGLang layouts) | older layout (SGLang b62df575): GPU-validated, 3x RTX PRO 6000, PP 3 (one pipeline stage per GPU): alpha 0 equals stock on 64/64 prompts, cyber32 8 → 31/32, benign32 32/32, decode within 5 % of stock. Stage boundaries (SGLang 15aa0b5a): CPU-validated on SGLang's real classes only; no GPU run yet. TP > 1 refused |
| GLM-5.3 | `GlmMoeDsaForCausalLM` | `hidden + residual`, `topk_indices` passed through; hook | structure-tested |
| DeepSeek-V4-Flash | `DeepseekV4ForCausalLM` (model_type `deepseek_v4`) | the FFN write before the mHC fold (`ffn_out_pre_residual`), SGLang's fused mHC mode only; special | structure-tested |
| DeepSeek-V4.1-Flash | `DeepseekV4ForCausalLM` with model_type `deepseek_v41` (row `DeepseekV4ForCausalLM:deepseek_v41`; SGLang rewrites the `DeepseekV41ForCausalLM` name to the V4 class) | the 4 hyper-connection copies (4 x 5120) after each layer's FFN post (`ffn_boundary.update.post`), before the next layer reads Engram (`residual_stream_post_layer`); the next input the post precomputed is dropped and rebuilt from the edited copies; special. Refuses SGLang's decoder tail graph (set `SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS=0`; under pipeline parallelism the engine keeps bounded replay off, so there is no tail graph and nothing to set) and a file for the FFN write | structure-tested (CPU, on SGLang's real HcUpdate / HcState / materialize / FinalRead, and on its real model loop and stage boundaries over three pipeline stages where the SGLang carries the V4.1 pipeline wiring, `tests/test_archs/test_dsv41_stages.py`); not run on a GPU yet. How the edit of the mean maps onto the copies (`WEIGHTLESS_STEER_HC_COPIES`) is not documented by the file; `mean` is the default |
| Hy4-preview | `HYV4ForCausalLM` | each of the 4 iHC streams, one direction per stream; hook | structure-tested |
| Qwen3.8-Flash-Next | `Qwen4ExpForConditionalGeneration` | the combined hyper-connection stream (4 x 2560); hook | structure-tested |
| Nemotron-3.5-Lightning | `NemotronHForCausalLM`, `NemotronHPuzzleForCausalLM` | `hidden + residual`; forward wrap | structure-tested; TP > 1 refused |
| Kimi-K3 | `KimiK3ForConditionalGeneration`, `KimiK3LinearForCausalLM` | the post-layer stream (whole in `hidden` with attention residuals); hook | structure-tested; TP > 1 refused |
| Nanbeige4.2-3B | `NanbeigeForCausalLM` | `hidden + residual` at each of 44 execution steps (22 layers x 2 loops); looped | structure-tested |
| Ouro | `OuroForCausalLM` | | refused: SGLang has no Ouro model |
| Inkling | `InklingForConditionalGeneration`, `InklingForCausalLM` | | refused: the post-layer stream never exists between layers |
| other names | `KimiLinearForCausalLM`, `DeepseekV4ForConditionalGeneration`, `DeepseekV41ForCausalLM` | | refused: a different model under that name, or no such SGLang class |

GLP spec_version 3 files (`hc_fold_pre_engram`, the GLP-27 vector) are served by the
DeepSeek-V4.1 row: the same fold and the same edit of the mean (`mean` only), plus
the companion writer the spec makes part of that site. The attention write the
attention boundary's `HcUpdate.post` folds in (o_proj / wo_b) is projected with the
same stack and alpha, the layer at position k with stack row k + 1 (the vLLM
plugin's `archs/dsv41.py` mapping): for GLP-27, folds on layers 11..37 and writers on
layers 10..36, with their own fired check, boot-line field and manifest entry
(`companion`). Attention TP above 1 is refused for such a file. Every other row
refuses the hook. CPU-validated (`tests/test_archs/test_dsv41_fold.py`, and SGLang's
real loop over three pipeline stages in `test_dsv41_stages.py`); not run on a GPU.

On SGLang trees with stage boundaries, the GLM-5.3, Hy4-preview,
Qwen3.8-Flash-Next, Nemotron-3.5-Lightning and Nanbeige rows are refused
(their layers are built that way and the rows are not ported yet). The
DeepSeek-V4 and Kimi-K3 models keep the older layout on 15aa0b5a, and their
CPU tests pass there (DeepSeek-V4's with SGLang's real decoder layers and
model loop); neither has run on a GPU. A non-special row marked for stage
boundaries must be a one-stream `(hidden, residual)` hook row, the only form
the residual-stream site reads; any other is refused at install.

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
  steered site ran exactly once, and raises otherwise: a site that did not
  run skips the steering, and a site that ran twice edits the edited stream
  (at alpha 2 the second reflection undoes the first).
- The MTP/NEXTN draft model (the small extra model that speculative
  decoding uses to guess the next tokens) stays stock. Alpha 0 is
  bit-identical to stock. Reloading weights from disk replaces the model
  and drops the sites: restart the server instead.
- Bit-identity holds within a boot. Across boots the engine can flip the
  bits on its own: SGLang autotunes some kernels at start-up without a
  disk cache, and two configs of one kernel can round differently (seen on
  GLM-5.3-Flash, issue #5). For bit-comparable validation across boots,
  pin the pick: `TRITON_CACHE_AUTOTUNING=1` and one `TRITON_CACHE_DIR`
  shared by every rank and boot.

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
tests that run SGLang's own code need SGLang in the environment; the
GLM-5.3-Flash and Qwen3.8-27B ones build the real model classes from the two
config files (`tests/test_archs/sglang_tiny.py`) and run on either SGLang
layout. Every skip names what is missing. The suite has 481 tests (407
before the DeepSeek-V4.1 row; its own are `tests/test_archs/test_dsv41.py`, `test_dsv41_stages.py`
and, for spec_version 3 files, `test_dsv41_fold.py`). On
a CPU host, with `WEIGHTLESS_TEST_GLP_DIR`, the GLM-5.3-Flash and Qwen3.8-27B
configs and both trees in `WEIGHTLESS_TEST_SGLANG_TREES` (no
`WEIGHTLESS_TEST_CONFIG_DIR`), the plugin pip-installed: 36 skip on SGLang
b62df575 (the older layout), 64 on 15aa0b5a (stage boundaries: the rows not
ported to it skip their older-layout tests). The DeepSeek-V4.1 tests also run
SGLang's own V4.1 mHC boundary code when SGLang is importable, and the fused
hc-mean kernel under Triton's CPU interpreter (`TRITON_INTERPRET=1`).

The structure tests pin what each row relies on in SGLang's model files, and
upstream moves fast. A newer SGLang is not supported until the rows are
re-pinned against it and the suite is green again; serving one anyway fails
closed at boot rather than steering the wrong tensor. Upstream's reference
checkout is `2f5c9ac43d76` (2026-09-25, `../docs/sglang-plugin-design.md`).
