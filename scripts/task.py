#!/usr/bin/env python3
"""weightless task — token and time usage of an omp agent session (stdlib only).

Reads omp's session journal (~/.omp/agent/sessions/<cwd-slug>/<id>.jsonl),
sums the per-call usage blocks, and answers "how much has this task burned":
elapsed wall time, model calls, fresh input / output / cache tokens, and the
live context size against the lane's window.

Usage:
    python3 scripts/task.py                  most recently active session
    python3 scripts/task.py --all            one line per session active today
    python3 scripts/task.py SESSION.jsonl    a specific session file

Cache reads are prefix-cache hits: tokens the server reused instead of
recomputing. They inflate neither fresh-input cost nor context.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

SESSIONS = os.path.expanduser("~/.omp/agent/sessions")
CTX_WINDOW = 1_048_576  # glm53-flash-tp4 lane; purely cosmetic for the % line


def human(n: float) -> str:
    a = abs(n)
    for scale, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= scale:
            return f"{n / scale:.1f}{suf}"
    return f"{n:.0f}"


def parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def scan(path: str) -> dict | None:
    """Aggregate one session journal. Per-call usage is reported per request,
    so session totals are the sum over assistant messages."""
    inp = out = cache_read = cache_write = calls = 0
    first = last = None
    ctx = 0
    try:
        lines = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return None
    with lines:
        for line in lines:
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # partial tail line while omp is writing
            ts = parse_ts(rec.get("timestamp", ""))
            if ts:
                first = first or ts
                last = ts
            msg = rec.get("message") or {}
            if msg.get("role") != "assistant":
                continue
            u = msg.get("usage")
            if not u:
                continue
            calls += 1
            inp += u.get("input", 0)
            out += u.get("output", 0)
            cache_read += u.get("cacheRead", 0)
            cache_write += u.get("cacheWrite", 0)
            ctx = u.get("totalTokens", ctx)
    if not calls:
        return None
    return {
        "path": path, "first": first, "last": last, "calls": calls,
        "input": inp, "output": out, "cache_read": cache_read,
        "cache_write": cache_write, "ctx": ctx,
        "mtime": os.path.getmtime(path),
    }


def fmt_row(s: dict) -> str:
    dur = (s["last"] - s["first"]) if s["first"] and s["last"] else None
    dur_s = str(dur).split(".")[0] if dur else "?"
    lines = [
        f"elapsed {dur_s}   {s['calls']} model calls",
        f"input    {human(s['input'])} fresh   output {human(s['output'])}",
        f"cache    {human(s['cache_read'])} read   {human(s['cache_write'])} written",
        f"context  {human(s['ctx'])} tok ({s['ctx'] / CTX_WINDOW * 100:.0f}% of {human(CTX_WINDOW)})",
    ]
    return "\n  ".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Token/time usage of an omp agent session.",
                                 allow_abbrev=False)
    ap.add_argument("session", nargs="?", help="session .jsonl (default: most recently active)")
    ap.add_argument("--all", action="store_true", help="one block per session active today")
    args = ap.parse_args(argv)

    if args.session:
        files = [args.session]
    else:
        files = glob.glob(os.path.join(SESSIONS, "*", "*.jsonl"))
        if not files:
            print(f"no omp sessions under {SESSIONS}", file=sys.stderr)
            return 1
        if args.all:
            today = datetime.now(timezone.utc).date()
            files = [f for f in files if datetime.fromtimestamp(
                os.path.getmtime(f), timezone.utc).date() == today]
            files.sort(key=os.path.getmtime)
        else:
            files = [max(files, key=os.path.getmtime)]

    shown = 0
    for path in files:
        s = scan(path)
        if not s:
            if args.session:
                print(f"no model usage recorded in {path}", file=sys.stderr)
                return 1
            continue
        shown += 1
        slug = os.path.basename(os.path.dirname(path))
        print(f"{slug}  ·  {os.path.basename(path)}")
        print(f"  {fmt_row(s)}")
        print()
    return 0 if shown else 1


if __name__ == "__main__":
    sys.exit(main())
