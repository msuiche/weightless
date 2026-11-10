# weightless.json — the lane convention (v1)

One file a serving recipe carries so any GLP-aware launcher can deploy it
with a vector. The recipe owns its launch machinery; the lane file owns the
contract between that machinery and GLP. This is how weightless supports
engines it does not host: the recipe repo keeps its lane current in the same
commits that change its launch flow, and the launcher never goes stale.

Design rules, inherited from the GLP spec:

- **Fail closed.** A lane that cannot be verified before launch is a launch
  that does not happen. Vector sha mismatch, missing check command, a check
  that fails — all hard errors, never fallbacks. Hook mismatches are the
  engine loader's refusal, not the lane file's: the vector declares its
  `glp.hook_point` and the reader refuses what it does not implement.
- **The recipe's own commands run.** The launcher never reimplements a
  recipe's build, check, or boot logic; it renders GLP inputs and invokes
  what the lane file declares.
- **Off is stock.** A lane launched without a vector must behave exactly as
  the recipe behaves on its own.

## The file

`weightless.json` at the recipe repo root:

```json
{
  "schema": "weightless/lane/1",
  "name": "spark-ds41f",
  "description": "DeepSeek-V4.1-Flash on 4x DGX Spark (TileLang, ring), GLP spec-3 fold steering",
  "homepage": "https://github.com/christopherowen/spark-ds41f",
  "pin": {"ref": "main", "recommended_revision": "<40-hex>"},
  "glp": {
    "placement": "config",
    "config": {
      "path": "config/cluster.json",
      "key": "glp",
      "fields": {"repo": "repo", "revision": "revision", "file": "file", "sha256": "sha256"}
    }
  },
  "commands": {
    "check": ["bin/spark", "doctor"],
    "up": ["bin/spark", "up"],
    "down": ["bin/spark", "down"],
    "status": ["bin/spark", "status"]
  },
  "verify": {
    "boot_line": "glp: armed",
    "note": "bin/spark doctor validates the vector revision and sha256 on every node before launch"
  }
}
```

## Fields

| field | meaning |
|---|---|
| `schema` | exactly `weightless/lane/1`. Unrecognised value: refuse. |
| `name` | lane name, used in output and lock files. |
| `description` | one line. |
| `pin.ref` | git ref the lane is developed against. |
| `pin.recommended_revision` | optional 40-hex the lane author last verified. The launcher records what it used. |
| `glp` | the GLP steering section. The file is sectioned on purpose — future sections (model, auth, …) extend it without a schema bump. |
| `glp.placement` | how the vector reaches the engine: `config` (the recipe reads it from a config file the launcher renders), `env` (the launcher sets environment variables), or `self` (the recipe fetches it given repo/revision/file/sha256 from its own config, which the launcher verifies and may set). |
| `glp.config` | for `config`/`self`: the file and key, and the field names the recipe expects (`repo`, `revision`, `file`, `sha256` — all four required). |
| `glp.env` | for `env`: variable names for the resolved local file path and alpha, e.g. `{"path": "GLP_FILE", "alpha": "GLP_ALPHA"}`. |
| `commands.check` | preflight the recipe already owns (config validation, offline vector check). Runs before `up`; non-zero exit aborts. |
| `commands.up` / `down` / `status` | the recipe's own verbs. Only `up` is required. |
| `verify.boot_line` | a string or regex that must appear in the boot logs of a steered launch. Its absence means stock — and the launcher says so instead of claiming steering. |
| `verify.probe` | optional: a prompt whose output should differ stock vs steered, for the launcher's post-boot A/B. |

## Launcher semantics (`weightless lane`)

`lane up <repo> [--ref R] --vector <repo-id[:file] | path> [--alpha A]`:

1. Clone the recipe at the pin (or `--ref`) into the lane workspace; record
   the exact revision in `lane.lock.json`.
2. Read `weightless.json`; refuse unknown schema.
3. Resolve the vector: download from HF (gated flows need the token as
   always) and verify sha256 against the lane's or the catalog's record.
   Hook enforcement belongs to the vector's own metadata and the engine's
   loader (`glp.hook_point` in the file; a reader refuses a hook it does not
   implement), so the lane file deliberately does not duplicate it.
4. Place the vector: render `config` fields / export `env` vars / verify
   `self` fields, per `placement`.
5. Run `commands.check`; non-zero aborts.
6. Run `commands.up`; watch for `verify.boot_line`; report it or say plainly
   that the boot is stock.
7. With `verify.probe`, A/B the probe prompt (alpha 0 vs file alpha) and
   print both.

`lane list` shows discovered lanes: a small registry in this repo
(`lanes/KNOWN.md`) of lane files the maintainers have seen work, plus
whatever the user added locally. The registry is a link list with freshness
dates, not a fork of anyone's recipe.

## Where the file lives: weightless first, recipe-repo override

The convention has two locations:

1. **This repo** (`lanes/<name>/weightless.json`) — the primary path, and
   the one most lanes will ever use. Support for a third-party recipe lands
   here the day it is written, without waiting on the recipe's maintainers.
   Each file carries `last_verified` and the upstream ref it was verified
   against; the lane's own `check` command catches drift at launch time,
   which is the honest answer to staleness.
2. **The recipe repo** (`weightless.json` at its root) — an optional
   override that wins whenever it exists, on the theory that the author
   knows their own launch flow best. Recipe authors adopt it to own their
   lane; nothing is required of them for the lane to work.

## What a recipe author needs to do

1. Add `weightless.json` at the repo root (fields above; only `schema`,
   `name`, one placement, and `commands.up` are required).
2. Make the recipe read the vector from the declared placement (it almost
   always already does).
3. Keep the file in the same commits that change the launch flow. A lane
   file that lies is worse than none.

Worked first example: the spark-ds41f lane
(`christopherowen/spark-ds41f`, GLP spec-3 fold site).
