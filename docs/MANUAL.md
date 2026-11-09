# The GLP Operator's Manual

Serve a model with a GLP steering vector in production. Three engine
recipes — vLLM, SGLang, Tensorfold — each ending in a verifiable boot.
Everything here is fail-closed by design: if the vector and the engine
disagree about anything, the boot refuses rather than serve silently
mis-steered. Format contract: [`spec/GLP.md`](spec/GLP.md). Vector catalog
with calibrated alphas: [`VECTORS.md`](VECTORS.md).

## The 60-second version

A GLP file is a few hundred KB of per-layer directions that delete one
component from a model's residual stream at inference:

```
h  <-  h - alpha * (h . d) d
```

No weight edits, no re-training, off means stock. To serve with one you need
three things: the **base checkpoint** (exact revision the vector is pinned
to), the **vector** (a `.gguf` from the [catalog](VECTORS.md)), and an
**engine that implements the vector's hook** (below). The vector is validated
at load — mode, hook point, spec version, base revision, content hash, layer
range — and every failure is a boot error with a message, never a fallback.

## Picking the vector (and the hook you must have)

Every published vector names its apply site in `glp.hook_point`. Your engine
must implement *that exact site* — a reader that applies elsewhere degrades
silently, which is why every runtime below refuses a hook it does not have:

| hook_point | the site | engines that serve it |
|---|---|---|
| `residual_stream_post_layer` | the accumulated stream after a layer's writes | vLLM plugin, SGLang plugin (10 archs) |
| `ffn_out_pre_residual` | the FFN/MoE write alone (DeepSeek-V4-0731) | vLLM plugin, SGLang plugin (dsv4 row) |
| `hc_fold_pre_engram` (spec 3) | the hc fold + attention `o_proj` companion, DeepSeek-V4.1 | vLLM plugin (dsv41 row), SGLang plugin (dsv41 row), spark-ds41f, Tensorfold (Dual-Spark PR), gr8 |

`glp.alpha_default` is the measured operating point for that checkpoint —
start there, do not port alpha across models or hooks. `WEIGHTLESS_STEER_ALPHA`
overrides at serve time; alpha 0 is a bit-exact no-op, which is also how you
prove the hook itself changes nothing.

Gated vectors (the `msuiche/` cyber series) need an HF token with the
agreement accepted once on the repo page; the file then caches like any
other.

## Recipe 1 — vLLM

```sh
pip install ./vllm-plugin                       # from this repo
hf download msuiche/GLM-5.3-abliterated-cyber-GLP-77   # example vector
export WEIGHTLESS_STEER_PATH=/path/to/GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf
python -m vllm.entrypoints.openai.api_server --model <base-checkpoint> \
    --tensor-parallel-size <n> --port 8000
```

Every rank logs one line — **no line means stock serving**:

```
weightless GLP steering active: hook=residual_stream_post_layer alpha=1.000 layers=1..77 (77) width=6144
```

Verify the layer count and alpha match the file, then A/B a probe request
with `WEIGHTLESS_STEER_ALPHA=0` (stock arm) against the file's default. For
DeepSeek-V4.1 spec-3 vectors (`hc_fold_pre_engram`), the plugin's dsv41 row
fires at the fold and arms the companion `o_proj` writer; the boot line
reflects both. Env vars: `WEIGHTLESS_STEER_PATH` (the file),
`WEIGHTLESS_STEER_ALPHA` (default: the file's), `WEIGHTLESS_STEER_LAYERS`
(comma list to restrict coverage). Deep dive: `docs/vllm-plugin-design.md`.

## Recipe 2 — SGLang

```sh
pip install ./vllm-plugin ./sglang-plugin     # the loader/gates are shared
export WEIGHTLESS_STEER_PATH=/path/to/vector.gguf
python -m sglang.launch_server --model-path <base-checkpoint> --port 30000
```

Each rank logs `weightless GLP steering active (sglang): ...`; again, no
line means stock. SGLang loads the plugin via its `sglang.srt.plugins`
entry point in every rank process — SGLang itself is unmodified. Multi-GPU:
`--pp-size` (pipeline: each rank wraps only its own layers) or `--tp-size`.
DeepSeek-V4.1 spec-3 vectors serve on the dsv41 row: fold edit on the mean
of the four hc copies, companion writer per layer (`row k+1` mapping), with
`WEIGHTLESS_STEER_HC_COPIES=mean` enforced for spec-3 files and refused
attention TP > 1. Extra knobs: `WEIGHTLESS_STEER_KERNEL` (`auto` = fused
Triton with a start-up bit-exact self-check, `torch` fallback),
`WEIGHTLESS_STEER_MANIFEST_DIR` (per-rank manifest: row, site, layers,
alpha, file sha, kernel choice, and which sites ran inside each CUDA-graph
capture). Details: `sglang-plugin/README.md`.

## Recipe 3 — Tensorfold (Dual-Spark, DeepSeek-V4.1)

The reference integration is
[x00byte/DeepSeek-V4.1-Flash-1017-Dual-Spark#1](https://github.com/x00byte/DeepSeek-V4.1-Flash-1017-Dual-Spark/pull/1):
a spec-3 reader, the fold projection, the companion writer applied after
rank reduction (overlapped prefill and split emission included), and a boot
fingerprint that refuses when the two nodes' vector copies differ.

```sh
# config/cluster.env, on both nodes at the same path
GLP_FILE=/srv/vectors/glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf
#GLP_ALPHA=            # default: the file's alpha_default (3.5 for GLP-27)
NO_DRAFTS=1           # required while steering: the DSpark drafter has no
                      # calibrated directions and would draft unsteered tokens
python3 scripts/cluster.py check    # validates the vector offline (tools/glpcheck.py)
python3 scripts/cluster.py up       # both nodes; the launcher mounts it read-only
```

`cluster.py check` runs the engine's exact loader offline, so a bad file
fails before a boot is attempted; the startup fingerprint of vector +
computed alpha is part of both two-rank settings exchange and
persisted-session identity, so a node that swaps files cannot join silently.

## Verify you are actually steered

1. **Boot line.** Every runtime prints a steering-active line per rank
   (`weightless GLP steering active...`); its absence means stock, silently
   — the plugin registers nothing when `WEIGHTLESS_STEER_PATH` is unset.
2. **Alpha 0 arm.** Reboot with `WEIGHTLESS_STEER_ALPHA=0`: outputs must be
   bit-identical to the unsteered boot. This proves the hook itself is not
   what changes anything.
3. **Probe request.** A prompt the vector targets (a refusal-class prompt
   for the cyber series): stock arm refuses or hedges, steered arm answers.
   Dose response is normal — the same prompt can sharpen as alpha rises
   toward the file's default.

## Tuning alpha

`alpha_default` is the measured knee for the pinned checkpoint, not a
suggestion to exceed. The failure mode past it is capability damage, and it
announces itself: exact-token corruption first (repeated digits, mangled
identifiers), then register shifts, then garbling. If you see any of those,
lower alpha — do not raise it to "push through." The per-model ladders with
the numbers are in `BENCHMARK.md`; the spec explains why alpha does not
port across checkpoints, hooks, or quantization schemes.

## Troubleshooting

| symptom | meaning |
|---|---|
| boot dies with `glp.hook_point ... does not match` | the vector targets a site this engine/row does not implement — wrong engine or wrong vector, not a bug |
| boot dies on `spec_version` | the file needs a newer contract than this build implements (spec 3 fold files in a pre-v3 reader) |
| boot dies on `content_sha256` | the file is corrupt or truncated; re-download and re-check the sha |
| boot dies on layer range / width | the vector is pinned to a different checkpoint revision than the one being served |
| no `steering active` line | `WEIGHTLESS_STEER_PATH` unset, plugin not installed in the serving env, or the env var died in your launcher — serving stock |
| refusals unchanged | same as above, or alpha too low for this checkpoint |
| garbled / corrupted identifiers | alpha above the knee — lower it |

## Implementing GLP in your own engine

The porting pattern every integration above follows: find the site's tensor
→ wrap it with the projection reading from GPU buffers (no host sync, no
data-dependent allocation — CUDA-graph safe) → validate the file fail-closed
at load → make off byte-identical to stock → print a fired line at boot.
The reader-conformance contract is [`spec/GLP.md`](spec/GLP.md) §"Reader
conformance"; the canonical loader is
`vllm-plugin/weightless_steer/container.py`; worked examples live in
`vllm-plugin/weightless_steer/archs/`, `sglang-plugin/weightless_sglang/archs/`,
and the Tensorfold PR linked in Recipe 3.
