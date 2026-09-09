#!/usr/bin/env python3
"""weightless dash — live terminal view of a vLLM lane (stdlib only).

Polls the lane's Prometheus /metrics endpoint and renders the numbers that
answer "what is the server doing right now": prefill/decode throughput,
queue depth, KV pressure, prefix-cache health, TTFT, and speculative-decode
acceptance. Works against any vLLM endpoint, local or remote.

Usage:
    python3 scripts/dash.py [url]                 live view (2s refresh)
    python3 scripts/dash.py [url] --once          one snapshot, for scripts
    python3 scripts/dash.py [url] --interval 5    slower refresh
    python3 scripts/dash.py [url] --no-color      plain output (also: NO_COLOR=1)

Colors are on when stdout is a terminal: pink/cyan brand accents (the
setup.py palette), green → yellow → red as KV pressure climbs, yellow on
queued requests. Output piped to a file or pipe is always plain.

Examples:
    python3 scripts/dash.py                                   # glm53-flash on the rig
    python3 scripts/dash.py http://spark-4687.local:8888      # DSV4 lane
    python3 scripts/dash.py https://<modal-endpoint> --once   # cloud lane snapshot
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.request
from collections import deque

SPARK = "▁▂▃▄▅▆▇█"
HIST = 40


def palette(enabled: bool) -> dict:
    """ANSI codes, or empty strings when color is off. Pink/cyan are the
    weightless brand ramp endpoints (setup.py LOGO_RAMP)."""
    codes = {
        "r": "\033[0m", "b": "\033[1m", "d": "\033[2m",
        "pink": "\033[38;5;170m", "cyan": "\033[38;5;80m",
        "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
    }
    return codes if enabled else dict.fromkeys(codes, "")


def gauge(c: dict, pct: float) -> str:
    """green < 70, yellow < 90, red ≥ 90."""
    col = c["green"] if pct < 70 else c["yellow"] if pct < 90 else c["red"]
    return f"{col}{pct:.0f}%{c['r']}"


def fetch(url: str, timeout: float = 5.0) -> str:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as r:
        return r.read().decode()


def parse_metrics(text: str) -> dict:
    """Flatten the Prometheus text format to {name_or_name|labels: value}."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([0-9eE.+-]+)$", line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        key = name if not labels else f"{name}|{labels[1:-1]}"
        out[key] = val
    return out


def g(m: dict, name: str, default=0.0) -> float:
    """Label-agnostic lookup: exact key, or first key of the form name|labels."""
    if name in m:
        return m[name]
    for k, v in m.items():
        if k.startswith(name + "|"):
            return v
    return default


def gsum(m: dict, prefix: str) -> float:
    return sum(v for k, v in m.items() if k.startswith(prefix))


def spark(vals: deque, width: int = HIST) -> str:
    if not vals:
        return ""
    v = list(vals)[-width:]
    peak = max(v) or 1.0
    return "".join(SPARK[min(7, int(x / peak * 7.999))] for x in v)


def render(target: str, m: dict, prev: dict | None, dt: float,
           hist_pre: deque, hist_dec: deque, up_s: float, c: dict) -> str:
    def rate(name: str) -> float:
        if prev is None or dt <= 0:
            return 0.0
        return max(0.0, (g(m, name) - g(prev, name)) / dt)

    pre, dec = rate("vllm:prompt_tokens_total"), rate("vllm:generation_tokens_total")
    hist_pre.append(pre)
    hist_dec.append(dec)

    running = g(m, "vllm:num_requests_running")
    waiting = g(m, "vllm:num_requests_waiting")
    kv = g(m, "vllm:kv_cache_usage_perc") * 100
    hits_q = g(m, "vllm:prefix_cache_hits_total")
    queries = g(m, "vllm:prefix_cache_queries_total")
    hit_rate = (hits_q / queries * 100) if queries else 0.0
    done = {re.search(r'finished_reason="([^"]+)"', k).group(1): int(v)
            for k, v in m.items()
            if k.startswith("vllm:request_success_total|") and 'finished_reason="' in k}
    ttft_c, ttft_s = g(m, "vllm:time_to_first_token_seconds_count"), g(m, "vllm:time_to_first_token_seconds_sum")
    ttft_avg = (ttft_s / ttft_c) if ttft_c else 0.0
    ttft_win = max(0.0, ((ttft_s - g(prev, "vllm:time_to_first_token_seconds_sum")) /
                         max(1e-9, ttft_c - g(prev, "vllm:time_to_first_token_seconds_count")))) if prev else 0.0

    drafts = rate("vllm:spec_decode_num_draft_tokens_total")
    accepted = rate("vllm:spec_decode_num_accepted_tokens_total")
    d_tot, a_tot = g(m, "vllm:spec_decode_num_draft_tokens_total"), g(m, "vllm:spec_decode_num_accepted_tokens_total")
    acc_rate = (a_tot / d_tot * 100) if d_tot else 0.0
    per_pos = sorted(
        ((int(k.split('position="')[1].split('"')[0]), v) for k, v in m.items()
         if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total|")),
        key=lambda t: t[0])
    pos_pct = " ".join(f"{int(v / max(1, per_pos[0][1]) * 100)}" for _, v in per_pos) if per_pos else ""

    L = []
    L.append(f"{c['pink']}{c['b']}weightless{c['r']} {c['cyan']}{c['b']}dash{c['r']}"
             f" {c['d']}— {target}  ·  up {int(up_s // 60)}m  ·  {time.strftime('%H:%M:%S')}{c['r']}")
    L.append("")
    L.append(f"  {c['cyan']}prefill{c['r']} {c['b']}{pre:7.0f}{c['r']} tok/s  {c['cyan']}{spark(hist_pre)}{c['r']}")
    L.append(f"  {c['pink']}decode {c['r']} {c['b']}{dec:7.1f}{c['r']} tok/s  {c['pink']}{spark(hist_dec)}{c['r']}")
    L.append("")
    if waiting > 0:
        wait_txt = f"{c['yellow']}{int(waiting)}{c['r']}  {c['red']}← QUEUED (raise MAX_NUM_SEQS){c['r']}"
    else:
        wait_txt = f"{int(waiting)}"
    L.append(f"  {c['d']}requests{c['r']}   running {int(running)}  waiting {wait_txt}")
    L.append(f"  {c['d']}kv cache{c['r']}   {gauge(c, kv)} used    {c['d']}prefix hit{c['r']} {c['green']}{hit_rate:.0f}%{c['r']}")
    L.append(f"  {c['d']}ttft{c['r']}       {ttft_win:.1f}s recent   {c['d']}{ttft_avg:.1f}s lifetime (incl. queue wait){c['r']}")
    if d_tot:
        L.append(f"  {c['d']}spec dec{c['r']}   accept {c['green']}{acc_rate:.0f}%{c['r']}   draft {drafts:.1f} → accepted {accepted:.1f} tok/s   {c['d']}per-pos {pos_pct}{c['r']}")
    L.append("")
    L.append(f"  {c['d']}done {sum(done.values())}  ({', '.join(f'{k} {v}' for k, v in sorted(done.items()))})"
             f"   prompt {int(g(m, 'vllm:prompt_tokens_total')):,} tok   gen {int(g(m, 'vllm:generation_tokens_total')):,} tok{c['r']}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Live terminal view of a vLLM lane.")
    ap.add_argument("url", nargs="?", default="http://spark-4687.local:8888",
                    help="lane base URL (default: %(default)s — DSV4 on the rig)")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--interval", type=float, default=2.0, help="refresh seconds (default: %(default)s)")
    ap.add_argument("--no-color", action="store_true",
                    help="plain output (auto when piped or NO_COLOR is set)")
    args = ap.parse_args()

    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR") and not args.no_color
    c = palette(color)

    hist_pre, hist_dec = deque(maxlen=HIST), deque(maxlen=HIST)
    prev, prev_t, t0 = None, 0.0, time.time()
    while True:
        try:
            m = parse_metrics(fetch(args.url))
        except Exception as e:
            print(f"\r\033[K{c['red']}cannot reach {args.url}/metrics: {e}{c['r']}", file=sys.stderr)
            if args.once:
                return 1
            time.sleep(args.interval)
            continue
        now = time.time()
        out = render(args.url, m, prev, now - prev_t, hist_pre, hist_dec, now - t0, c)
        if args.once:
            print(out)
            return 0
        print("\033[H\033[J" + out, end="", flush=True)
        prev, prev_t = m, now
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return 0


if __name__ == "__main__":
    sys.exit(main())
