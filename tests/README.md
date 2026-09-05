# tests/ — plug the deployed model into the omp harness

Smoke tests that the cluster endpoint
(`http://localhost:8888/v1`, model `deepseek-v4-flash-dspark`) is fit
to drive [omp](https://omp.sh/) — the batteries-included Pi fork we prefer
over stock Pi (hash-anchored edits, faster tool harness, LSP/DAP wired in).
The last test runs a real headless omp agent loop against the endpoint, so a
pass means the served model handles omp's tool schemas, streaming, and edit
path — not just that it answers chat.

The wizard tests a completed answer before changing client defaults and reads
the selected server's `max_model_len` for both omp and Hermes. If that metadata
is unavailable, it uses the selected model's window from `tests/models.yml`.
Hermes is no longer capped to 65,536 tokens for every lane. The router lists
only ready engines; an empty model list means no lane is ready.
Both clients keep the exact verified endpoint, including custom ports and
proxy paths. Inkling checks prefer the router only when it advertises the
model; a fresh deployment without a router uses its configured engine port.

Inkling requires `--enable-auto-tool-choice --tool-call-parser inkling` on
both serving ranks. The native tokenizer/parser supports tools; the earlier
claim that Inkling was chat-only was incorrect. Test it through port 8000 to
cover the router's streaming path. Existing Hermes gateways need restarting
after configuration changes, and saved `/model` session overrides take
precedence over `model.default`.
On this rig, Hermes's automatic title generation repeatedly hit its 30-second
timeout and retried against the busy local model. It can be disabled without
affecting chat or tools in `~/.hermes/config.yaml`:

```yaml
auxiliary:
  title_generation:
    enabled: false
```

Restart an existing gateway after changing this setting. Long initial prompts
still require prefill time; disabling titles does not remove that cost.

## Setup

Interactive wizard — probes the endpoint, lists the models it actually
serves, installs the omp provider, registers it in omp's modelRoles
(`~/.omp/agent/config.yml` — default role only, or every text role:
smol, slow, plan, task, commit, tiny, advisor, designer; vision is left
untouched), offers to run the suite (the root `setup.py` does this plus
env generation, steering validation, and ssh deploy for the serving
lanes):

```sh
python3 setup.py   # repo root
```

Non-interactive equivalent:

```sh
curl -fsSL https://omp.sh/install | sh   # needs bun >= 1.3.14 (`bun upgrade`)
sh tests/install.sh                       # merges the dspark provider into ~/.omp/agent/models.yml
                                          # and sets it as omp's default model
                                          # (WEIGHTLESS_OMP_ALL_ROLES=1 for every text role)
```

## Run

```sh
sh tests/run.sh
```

| test | what it proves |
|---|---|
| `01-endpoint.sh` | `/v1/models` answers and lists our model id |
| `02-chat.sh` | a chat completion returns non-empty content |
| `03-tool-call.sh` | the model emits a well-formed tool call (omp's foundation) |
| `04-omp-headless.sh` | `omp -p` drives our endpoint to create a file in a scratch dir |

Exit code 2 from a test means SKIP (e.g. omp not installed); `run.sh` only
fails the suite on real failures.

## Configuration

- `WEIGHTLESS_BASE_URL` — default `http://localhost:8888/v1`
- `WEIGHTLESS_MODEL` — default `deepseek-v4-flash-dspark`
- `WEIGHTLESS_OMP_MODEL` — omp selector for test 04, default `weightless/$WEIGHTLESS_MODEL`

Offline setup and router regression tests:

```sh
python3 -m unittest discover -s tests -p 'test_setup.py'
```

`tests/models.yml` is the omp provider definition. The `compat` block mirrors
the official DeepSeek guidance for omp (system role, `max_tokens`, no
`tool_choice`, reasoning-content round-trip) — those three fields are what
keep thinking-mode tool conversations from 400ing. If the endpoint's chat
template changes, that block is the first place to look.
