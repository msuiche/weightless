# tests/ — plug the deployed model into the omp harness

Smoke tests that the cluster endpoint
(`http://localhost:8888/v1`, model `deepseek-v4-flash-dspark`) is fit
to drive [omp](https://omp.sh/) — the batteries-included Pi fork we prefer
over stock Pi (hash-anchored edits, faster tool harness, LSP/DAP wired in).
The last test runs a real headless omp agent loop against the endpoint, so a
pass means the served model handles omp's tool schemas, streaming, and edit
path — not just that it answers chat.

## Setup

Interactive wizard — probes the endpoint, lists the models it actually
serves, installs the omp provider, offers to run the suite (the root
`setup.py` does this plus env generation, steering validation, and
ssh deploy for the serving lanes):

```sh
python3 setup.py   # repo root
```

Non-interactive equivalent:

```sh
curl -fsSL https://omp.sh/install | sh   # needs bun >= 1.3.14 (`bun upgrade`)
sh tests/install.sh                       # merges the dspark provider into ~/.omp/agent/models.yml
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
- `WEIGHTLESS_OMP_MODEL` — omp selector for test 04, default `dspark/deepseek-v4-flash-dspark`

`tests/models.yml` is the omp provider definition. The `compat` block mirrors
the official DeepSeek guidance for omp (system role, `max_tokens`, no
`tool_choice`, reasoning-content round-trip) — those three fields are what
keep thinking-mode tool conversations from 400ing. If the endpoint's chat
template changes, that block is the first place to look.
