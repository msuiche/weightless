#!/usr/bin/env python3
"""weightless — one front door for the toolkit.

    python3 weightless.py                 interactive setup wizard (default)
    python3 weightless.py setup           same wizard: lane → env → deploy → clients
    python3 weightless.py serve <lane>    switch the rig to a lane, non-interactive
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
import difflib
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

COMMANDS = {
    "setup": ([PY, os.path.join(HERE, "setup.py")],
              "interactive setup wizard: lane pick → env → deploy → omp/hermes + tests"),
    "serve": ([PY, os.path.join(HERE, "setup.py"), "serve"],
              "switch the rig to a lane, non-interactive: serve <name|#> [--skip-assets] [--skip-wait]"),
    "dash": ([PY, os.path.join(HERE, "scripts", "dash.py")],
             "live metrics for a serving lane (prefill/decode, queue, KV, spec decode)"),
    "test": (["bash", os.path.join(HERE, "tests", "smoke", "run.sh")],
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


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["setup"] if sys.stdin.isatty() else ["--help"]
    cmd, rest = argv[0], argv[1:]
    if cmd == "help" and rest:
        cmd, rest = rest[0], rest[1:] + ["--help"]
    if cmd in ("-h", "--help", "help"):
        print(__doc__.strip() + "\n\ncommands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:6s} {desc}")
        return 0
    if cmd in COMMANDS:
        if cmd == "validate" and any(arg in ("-h", "--help") for arg in rest):
            parser = argparse.ArgumentParser(prog="weightless validate",
                                             description=COMMANDS[cmd][1])
            parser.add_argument("file", metavar="FILE.gguf", help="control-vector GGUF to validate")
            parser.print_help()
            return 0
        if cmd == "test":
            parser = argparse.ArgumentParser(
                prog="weightless test", description="Run endpoint smoke tests.",
                epilog="Configure the endpoint with WEIGHTLESS_BASE_URL and WEIGHTLESS_MODEL.")
            parser.parse_args(rest)
        target = COMMANDS[cmd][0]
        try:
            os.execvp(target[0], target + rest)
        except OSError as exc:
            print(f"weightless: cannot start {cmd}: {exc}", file=sys.stderr)
            return 127 if isinstance(exc, FileNotFoundError) else 126
    suggestion = difflib.get_close_matches(cmd, COMMANDS, n=1)
    hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    print(f"weightless: unknown command {cmd!r}.{hint} Try --help.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
