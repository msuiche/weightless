#!/usr/bin/env python3
"""weightless — one front door for the toolkit.

    python3 weightless.py                 interactive setup wizard (default)
    python3 weightless.py setup           same wizard: lane → env → deploy → clients
    python3 weightless.py dash [url]      live lane metrics (scripts/dash.py)
    python3 weightless.py test            endpoint smoke suite (tests/smoke/run.sh)
    python3 weightless.py validate f.gguf GLP spec check on a control-vector GGUF
    python3 weightless.py inspect f.gguf  metadata + per-layer stats of a GLP GGUF
    python3 weightless.py export f.gguf --out v.safetensors
                                          direction tensors as .safetensors
    python3 weightless.py bake f.gguf --base <model> --out <dir>
                                          fold the vector into a PEFT LoRA adapter

Stdlib only. Each subcommand execs the real script with the remaining
arguments, so the scripts keep working standalone and there is exactly one
implementation of each feature.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

COMMANDS = {
    "setup": ([PY, os.path.join(HERE, "setup.py")],
              "interactive setup wizard: lane pick → env → deploy → omp/hermes + tests"),
    "dash": ([PY, os.path.join(HERE, "scripts", "dash.py")],
             "live metrics for a serving lane (prefill/decode, queue, KV, spec decode)"),
    "test": (["sh", os.path.join(HERE, "tests", "smoke", "run.sh")],
             "endpoint smoke suite against the configured base URL"),
    "validate": ([PY, os.path.join(HERE, "tools", "captain-vector", "captain_vector.py"), "--validate"],
                 "GLP spec check on a control-vector GGUF (exit 1 on FAIL)"),
    "inspect": ([PY, os.path.join(HERE, "tools", "captain-vector", "captain_vector.py"), "inspect"],
                "metadata and per-layer stats of a control-vector GGUF [--json] [--topk N]"),
    "export": ([PY, os.path.join(HERE, "tools", "captain-vector", "captain_vector.py"), "export"],
               "export direction tensors as .safetensors (--out required)"),
    "bake": ([PY, os.path.join(HERE, "tools", "captain-vector", "captain_vector.py"), "bake"],
             "bake a GGUF vector into a rank-1 PEFT LoRA adapter (--base/--out required)"),
}


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        os.execvp(PY, COMMANDS["setup"][0])
    cmd, rest = argv[0], argv[1:]
    if cmd in ("-h", "--help", "help"):
        print(__doc__.strip() + "\n\ncommands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:6s} {desc}")
        return 0
    if cmd in COMMANDS:
        target = COMMANDS[cmd][0]
        os.execvp(target[0], target + rest)
    print(f"unknown command: {cmd} (try --help)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
