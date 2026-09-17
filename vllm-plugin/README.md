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
| `WEIGHTLESS_ENABLE_MILESTONE_2` | **the gate.** Unset: no per-request buffers are registered, the apply is the scalar one, and a request carrying control xargs is refused. Set: the model registers per-token alpha rows and a per-request layer gate |
| `WEIGHTLESS_CONTROL_MIN_ALPHA` / `WEIGHTLESS_CONTROL_MAX_ALPHA` | the window a client-supplied alpha must fall in; default `[0.0, 4.0]`. Bounds **overrides only** — the server's own `WEIGHTLESS_STEER_ALPHA` is never checked against it |
| `WEIGHTLESS_CONTROL_MAX_LAYERS` / `_MAX_LAYER_ID` | cap on a request's layer mask (default 128 / 1023) |
| `WEIGHTLESS_CONTROL_MAX_SEGMENTS` / `_MAX_POSITION` | cap on a request's alpha schedule (default 8 / 1048576) |

## Per-request controls

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
3.5). Each additional lane is one module under `weightless_steer/archs/`.

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
- **Speculative decoding:** the steering applies to the trunk layers only.
  On checkpoints with MTP/nextn draft layers, serve without speculative
  decoding (same caveat as the nemotron hotfix).
- torch.compile / CUDA graphs: the apply lives inside the overridden
  forward, so it is traced and captured exactly as the hotfix-patched
  version was.

## Tests

No GPU, no vllm install needed (torch + numpy only):

```bash
cd vllm-plugin && python -m unittest discover -s tests
```

The request-policy tests need neither (stdlib only):

```bash
python -m unittest tests.test_controls          # from the repo root
```
