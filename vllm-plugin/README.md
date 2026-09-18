# weightless-steer — GLP steering as a vLLM plugin

Applies `h ← h − α(h·d̂)d̂` on the post-layer residual stream of every
steered decoder layer, with per-layer unit directions `d̂` from a GLP GGUF
control vector (`../spec/GLP.md`). This is the plugin successor to the
`../patches/hotfix-*-steering-projective.py` fleet: same container gates,
same CUDA-graph discipline (dense zero-padded stack indexed by global
layer id, tensor alpha buffer, unconditional apply, fail-closed), but no
in-container file rewrites — vLLM's `vllm.general_plugins` entry point
shadows the model class in the `ModelRegistry` with a steered subclass.
Design: `../docs/vllm-plugin-design.md`, path (1).

## Install

Into the same environment that runs vLLM:

```bash
pip install /path/to/weightless/vllm-plugin
```

## Serve

```bash
export WEIGHTLESS_STEER_PATH=/path/to/vector.gguf
vllm serve <model>            # arch must be one the plugin shadows
```

Env vars (unchanged from the hotfixes):

| var | meaning |
|---|---|
| `WEIGHTLESS_STEER_PATH` | GLP `.gguf` control vector. **Unset: the plugin registers nothing and the served model is byte-for-byte stock.** |
| `WEIGHTLESS_STEER_ALPHA` | float; overrides the file's `glp.alpha_default` (which is the default when unset) |
| `WEIGHTLESS_STEER_LAYERS` | optional comma list restricting steered layer ids |
| `WEIGHTLESS_STEER_HOOK` | if set, must equal the adapter's hook (`residual_stream_post_layer`) — anything else fails closed |

Per-request controls (off unless opted into):

| var | meaning |
|---|---|
| `WEIGHTLESS_ENABLE_MILESTONE_2` | **the gate — not honoured by this plugin's serving path yet.** Unset (the only supported setting): no per-request buffers are registered, the apply is the scalar one, and a request carrying control xargs is refused. Set: model construction raises, because nothing here parses a request's controls or installs its rows — see below |
| `WEIGHTLESS_CONTROL_MIN_ALPHA` / `WEIGHTLESS_CONTROL_MAX_ALPHA` | the window a client-supplied alpha must fall in; default `[0.0, 4.0]`. Bounds **overrides only** — the server's own `WEIGHTLESS_STEER_ALPHA` is never checked against it |
| `WEIGHTLESS_CONTROL_MAX_LAYERS` / `_MAX_LAYER_ID` | cap on a request's layer mask (default 128 / 1023) |
| `WEIGHTLESS_CONTROL_MAX_SEGMENTS` / `_MAX_POSITION` | cap on a request's alpha schedule (default 8 / 1048576) |

## Per-request controls

**Status: the primitives ship, the serving path does not.** The buffers,
the apply, the row installer and the request policy are all here and
tested, but no code in this plugin reads a request's controls or calls
`set_weightless_control_rows` during serving — binding that to an engine's
batch metadata is the piece that stays engine-shaped, and it is unwritten.
So `WEIGHTLESS_ENABLE_MILESTONE_2` **raises at model construction** rather
than booting a server that advertises per-request steering and then serves
every request at the server default. The rest of this section describes
what the primitives do, and what enabling the gate will mean once a runner
binding exists.

With the gate set, a request may narrow what steering does to it — a
different alpha, a different alpha for prefill than for decode, a token
schedule that ramps alpha across the generation, or a mask restricting
which layers are steered:

```json
{"vllm_xargs": {"weightless": {"version": 1, "controls": {
  "prefill_alpha": 1.0,
  "decode_alpha": 2.0,
  "layers": "10-38",
  "schedule": [{"start": 0, "end": 16, "start_alpha": 0.5, "end_alpha": 2.0}]
}}}}
```

The legacy flat form (`vllm_xargs.weightless_alpha`, `weightless_layers`)
still parses and is checked for agreement with the structured form.

Two properties this lane is built around:

- **Defaults reproduce the scalar lane exactly.** The alpha rows are
  pre-filled with the server alpha and the layer gate with ones, so a step
  where nothing writes a control plan — no plumbing attached, a warmup
  forward, padded rows past the batch — steers exactly as the server is
  configured to. The failure direction is "steered as configured", never
  "silently unsteered".
- **A mask can only narrow, never invent.** Layers outside the model's
  depth, or outside the set the loaded vector actually carries, are refused
  at request validation rather than silently steering nothing.

`weightless_steer/control_plane.py` turns one step's scheduled requests
into the three row tensors and installs them; it imports no vLLM, so it is
tested on CPU (`tests/test_control_plane.py`). Binding it to a specific
engine's batch metadata is the one piece that stays engine-shaped.

Request policy — what a client may ask for, and the caps above — lives in
the top-level `weightless_runtime/` package, which imports neither vLLM nor
torch and is shared with the hotfix fleet. It ships inside this wheel.

Supported archs today: `NemotronHForCausalLM` (nemotron_h / Nemotron-H
3.5) and `Glm5NextForCausalLM` (glm5next / GLM-5.3-Flash — mHC widened
stream; the site is the materialized post-layer `hc_post` stream flattened
to `mhc_num_residual_streams × hidden`, and the adapter defers the last
layer's in-decoder contract so the final layer is steered too — see the
adapter docstring). glm5next is GPU-validated (2026-09-18, Modal 4×H100,
RedHatAI NVFP4, compiled mode: cyber32 exactly the hotfix reference at
31/32, benign32 clean, refusal32 2→9/32 vs the hotfix's 1→21/32 — the
stack's stock baseline is stiffer; numbers in `../BENCHMARK.md`). Each
additional lane is one module under `weightless_steer/archs/`.

## Behaviour contract

- **Fail-closed.** `WEIGHTLESS_STEER_PATH` set + a vector that is missing,
  non-`project`, wrong `glp.hook_point`, wrong width, out of layer range,
  or filtered to nothing: model construction raises and the engine boot
  dies. A boot asked for steering never serves unsteered. (Validation
  deliberately lives in the model `__init__`, not the plugin entry point —
  vLLM swallows plugin-load exceptions.)
- **Unsupported arch + `WEIGHTLESS_STEER_PATH` set serves stock.** Only
  shadowed archs are steered; confirm from the boot log line
  `weightless GLP steering active: hook=... alpha=... layers=...`.
  Caveat: vLLM's default dictConfig attaches a handler only to the `vllm`
  logger, so these INFO lines may not render even when steering IS active
  — a missing line is not proof of absence (that misreading cost a day of
  debugging a working plugin, 2026-09-17). Confirm by behaviour (an α=0 vs
  α=2 contrast), or force root INFO via `VLLM_LOGGING_CONFIG_PATH` +
  `VLLM_CONFIGURE_LOGGING=1`; `../modal/sitecustomize.py` does both and
  prints its markers to stderr.
- **Speculative decoding:** the steering applies to the trunk layers only.
  On checkpoints with MTP/nextn draft layers, serve without speculative
  decoding (same caveat as the nemotron hotfix).
- torch.compile / CUDA graphs: the apply lives inside the overridden
  forward, so it is traced and captured exactly as the hotfix-patched
  version was. Upstream's `@support_torch_compile` constructor captures
  its *bound* forward before the adapter swaps `model.__class__`, so the
  adapter re-runs `TorchCompileWithNoGuardsWrapper.__init__` after the
  swap; without that the compiled callable would run the stock forward and
  compiled serving would be silently unsteered.
- **Alpha multipliers are refused, not ignored.** A spec-version-2 vector
  carrying `glp.dir_scales` or `glp.layer_scales` fails the load: this
  lane applies one scalar alpha to every steered layer, so serving such a
  file would apply it at the wrong strength with nothing in the output to
  say so.
- **Nonfinite or zero values are refused.** A NaN/inf alpha (from the file
  or `WEIGHTLESS_STEER_ALPHA`), an alpha that is finite in f32 but
  overflows the model dtype, and a direction that is nonfinite or
  all-zero are all rejected before any buffer is registered — a zero
  direction steers nothing, a NaN one NaNs the whole stream.

## Tests

No GPU, no vllm install needed (torch + numpy only):

```bash
cd vllm-plugin && python -m unittest discover -s tests
```

The request-policy tests need neither (stdlib only):

```bash
python -m unittest tests.test_controls          # from the repo root
```
