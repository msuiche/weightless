#!/usr/bin/env python3
"""Every compose command block must render to valid bash.

2026-09-17: a #-comment inside docker-compose.dsv4.yml's folded command
scalar (`>`) swallowed the rest of the folded mega-line — YAML folds
same-indent lines into one, so the comment ate the peer gate and an `if`
opener, and the container crash-looped on a stray `fi`. `docker compose
config` renders the block; bash -n is the gate. Runs offline (no daemon).
"""

import glob
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    try:
        import yaml
    except ImportError:
        print("  [SKIP] pyyaml not installed")
        return 0
    failures = 0
    checked = 0
    for path in sorted(glob.glob(os.path.join(REPO, "recipe", "**", "*.yml"),
                                 recursive=True)):
        try:
            doc = yaml.safe_load(open(path))
        except Exception:
            continue
        for name, svc in (doc or {}).get("services", {}).items():
            cmd = (svc or {}).get("command")
            if not cmd:
                continue
            script = cmd[-1] if isinstance(cmd, list) else cmd
            script = script.replace("$$", "$")  # compose escape
            r = subprocess.run(["bash", "-n"], input=script.encode(),
                               capture_output=True)
            checked += 1
            rel = os.path.relpath(path, REPO)
            if r.returncode == 0:
                print(f"  [PASS] {rel} [{name}] renders to valid bash")
            else:
                failures += 1
                detail = r.stderr.decode().splitlines()
                print(f"  [FAIL] {rel} [{name}]: {detail[0] if detail else '?'}")
    if not checked:
        print("  [SKIP] no compose command blocks found")
    print()
    print("compose command blocks: "
          + ("all checks passed" if not failures else f"{failures} failure(s)"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
