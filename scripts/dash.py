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
import math
import os
import re
import shutil
import sys
import time
import urllib.request
import urllib.parse
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
        try:
            val = float(m.group(3))
        except ValueError:
            continue  # regex-shaped but not a number ("1e", "1.2.3", ".")
        if not math.isfinite(val):
            continue  # overflow ("1e999") — "nan"/"inf" text can't match the regex
        name, labels = m.group(1), m.group(2) or ""
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


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def clip(line: str, width: int | None) -> str:
    """ANSI-aware truncate to a terminal width; no-op when width is falsy.
    Appends a reset when a colored line was cut so the terminal can't bleed."""
    if not width or width <= 0 or len(ANSI.sub("", line)) <= width:
        return line
    out, n, i = [], 0, 0
    for mch in ANSI.finditer(line):
        seg = line[i:mch.start()]
        take = seg[:width - n]
        out.append(take)
        n += len(take)
        out.append(mch.group(0))
        i = mch.end()
        if n >= width:
            break
    if n < width:
        out.append(line[i:i + (width - n)])
    codes = ANSI.findall("".join(out))
    if not codes or codes[-1] != "\033[0m":
        out.append("\033[0m")
    return "".join(out)


def human(n: float) -> str:
    """16_964_612 -> '17.0M', 3_023 -> '3.0k', 604 -> '604'."""
    a = abs(n)
    for scale, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= scale:
            return f"{n / scale:.1f}{suf}"
    return f"{n:.0f}"


def fmt_rate(n: float) -> str:
    return human(n) if n >= 1000 else f"{n:.1f}"


def fmt_dur(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m}m" if m < 60 else f"{m // 60}h{m % 60:02d}m"


PROMPT_TOK = "vllm:prompt_tokens_total"
GEN_TOK = "vllm:generation_tokens_total"
DRAFT_TOK = "vllm:spec_decode_num_draft_tokens_total"
ACC_TOK = "vllm:spec_decode_num_accepted_tokens_total"


class RateWindow:
    """Sliding-window rate over a monotonic counter. Single-interval deltas
    spike (a 2s window with one big prefill batch reads as 100k tok/s);
    dividing the counter delta over the whole window span tracks the real
    trend. A counter that goes backwards (server restart) resets the window."""

    def __init__(self, span: float = 30.0) -> None:
        self.span = span
        self.samples: deque = deque()

    def add(self, t: float, value: float) -> None:
        if self.samples and value < self.samples[-1][1]:
            self.samples.clear()
        self.samples.append((t, value))
        while len(self.samples) > 2 and self.samples[0][0] < t - self.span:
            self.samples.popleft()

    def rate(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        (t0, v0), (t1, v1) = self.samples[0], self.samples[-1]
        if t1 <= t0:
            return 0.0
        return max(0.0, (v1 - v0) / (t1 - t0))


def fetch_json(base: str, path: str, timeout: float) -> dict:
    import json
    with urllib.request.urlopen(base.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def lane_info(base: str, timeout: float) -> dict:
    """Context for the header: served model id, context length, engine
    version. Best-effort — a lane without /v1 or /version just shows less."""
    info = {}
    try:
        data = fetch_json(base, "/v1/models", timeout).get("data") or []
        if data:
            if data[0].get("id"):
                info["model"] = data[0]["id"]
            if data[0].get("max_model_len"):
                info["ctx"] = int(data[0]["max_model_len"])
    except Exception:
        pass
    try:
        ver = fetch_json(base, "/version", timeout).get("version")
        if ver:
            info["version"] = ver
    except Exception:
        pass
    return info


def render(target: str, m: dict, prev: dict | None, dt: float,
           hist_pre: deque, hist_dec: deque, up_s: float, c: dict,
           info: dict | None = None, width: int | None = None,
           wins: dict | None = None, avgs: tuple | None = None,
           session: tuple | None = None) -> str:
    def rate(name: str) -> float:
        if prev is None or dt <= 0:
            return 0.0
        return max(0.0, (g(m, name) - g(prev, name)) / dt)

    def stable(name: str) -> float:
        w = wins.get(name) if wins else None
        return w.rate() if w is not None else rate(name)

    pre, dec = stable(PROMPT_TOK), stable(GEN_TOK)
    hist_pre.append(pre)
    hist_dec.append(dec)

    running = g(m, "vllm:num_requests_running")
    waiting = g(m, "vllm:num_requests_waiting")
    kv = g(m, "vllm:kv_cache_usage_perc") * 100
    hits_q = g(m, "vllm:prefix_cache_hits_total")
    queries = g(m, "vllm:prefix_cache_queries_total")
    hit_rate = (hits_q / queries * 100) if queries else 0.0
    done = {}
    for k, v in m.items():
        if not k.startswith("vllm:request_success_total|"):
            continue
        reason = re.search(r'(?:^|[|,])finished_reason="([^"]+)"', k)
        if reason:
            done[reason.group(1)] = int(v)
    ttft_c, ttft_s = g(m, "vllm:time_to_first_token_seconds_count"), g(m, "vllm:time_to_first_token_seconds_sum")
    ttft_avg = (ttft_s / ttft_c) if ttft_c else 0.0
    ttft_win = max(0.0, ((ttft_s - g(prev, "vllm:time_to_first_token_seconds_sum")) /
                         max(1e-9, ttft_c - g(prev, "vllm:time_to_first_token_seconds_count")))) if prev else 0.0

    drafts, accepted = stable(DRAFT_TOK), stable(ACC_TOK)
    d_tot, a_tot = g(m, DRAFT_TOK), g(m, ACC_TOK)
    acc_rate = (a_tot / d_tot * 100) if d_tot else 0.0
    per_pos = []
    for k, v in m.items():
        if not k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total|"):
            continue
        pos = re.search(r'(?:^|[|,])position="(\d+)"', k)
        if pos:
            per_pos.append((int(pos.group(1)), v))
    per_pos.sort(key=lambda t: t[0])
    pos_pct = " ".join(f"{int(v / max(1, per_pos[0][1]) * 100)}" for _, v in per_pos) if per_pos else ""

    spark_w = max(10, width - 46) if width else HIST

    L = []
    L.append(f"{c['pink']}{c['b']}weightless{c['r']} {c['cyan']}{c['b']}dash{c['r']}"
             f" {c['d']}— {target} · up {fmt_dur(up_s)} · {time.strftime('%H:%M:%S')}{c['r']}")
    if info:
        bits = []
        if info.get("model"):
            bits.append(str(info["model"]))
        if info.get("ctx"):
            bits.append(f"ctx {human(info['ctx'])}")
        if info.get("version"):
            bits.append(f"vllm {info['version']}")
        if bits:
            L.append(f"  {c['d']}lane{c['r']}  " + f"{c['d']} · {c['r']}".join(bits))
    L.append("")
    avg_pre = f"  {c['d']}avg {fmt_rate(avgs[0])}{c['r']}" if avgs else ""
    avg_dec = f"  {c['d']}avg {fmt_rate(avgs[1])}{c['r']}" if avgs else ""
    L.append(f"  {c['cyan']}prefill{c['r']} {c['b']}{fmt_rate(pre):>7}{c['r']} tok/s{avg_pre}"
             f"  {c['cyan']}{spark(hist_pre, spark_w)}{c['r']}")
    L.append(f"  {c['pink']}decode {c['r']} {c['b']}{fmt_rate(dec):>7}{c['r']} tok/s{avg_dec}"
             f"  {c['pink']}{spark(hist_dec, spark_w)}{c['r']}")
    L.append("")
    if waiting > 0:
        wait_txt = f"{c['yellow']}{int(waiting)}{c['r']}  {c['red']}← QUEUED (raise MAX_NUM_SEQS){c['r']}"
    else:
        wait_txt = f"{int(waiting)}"
    per_req = f"  {c['d']}decode {fmt_rate(dec / running)} tok/s/req{c['r']}" if running > 0 else ""
    L.append(f"  {c['d']}requests{c['r']}   running {int(running)}  waiting {wait_txt}{per_req}")
    L.append(f"  {c['d']}kv cache{c['r']}   {gauge(c, kv)} used    {c['d']}prefix hit{c['r']} {c['green']}{hit_rate:.0f}%{c['r']}")
    L.append(f"  {c['d']}ttft{c['r']}       {ttft_win:.1f}s recent   {c['d']}{ttft_avg:.1f}s lifetime (incl. queue wait){c['r']}")
    if d_tot:
        L.append(f"  {c['d']}spec dec{c['r']}   accept {c['green']}{acc_rate:.0f}%{c['r']}   draft {drafts:.1f} → accepted {accepted:.1f} tok/s   {c['d']}per-pos {pos_pct}{c['r']}")
    L.append("")
    L.append(f"  {c['d']}done {sum(done.values())}  ({', '.join(f'{k} {v}' for k, v in sorted(done.items()))}){c['r']}")
    if session is not None:
        sess = (f"   {c['d']}this dash: +{human(max(0.0, g(m, PROMPT_TOK) - session[0]))} prompt"
                f"   +{human(max(0.0, g(m, GEN_TOK) - session[1]))} gen{c['r']}")
    else:
        sess = ""
    L.append(f"  {c['d']}tokens{c['r']}    prompt {human(g(m, PROMPT_TOK))}   gen {human(g(m, GEN_TOK))}{sess}")
    if width:
        L = [clip(line, width) for line in L]
    return "\n".join(L)


def positive_seconds(value):
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive, finite number of seconds")
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive, finite number of seconds")
    return seconds


def metrics_base(value):
    try:
        url = urllib.parse.urlsplit(value)
        if url.scheme not in ("http", "https") or not url.hostname or url.query or url.fragment:
            raise ValueError
        url.port  # Validate ports before making a request.
    except ValueError:
        raise argparse.ArgumentTypeError("use an http:// or https:// base URL without a query or fragment")
    path = url.path.rstrip("/")
    for suffix in ("/metrics", "/v1"):
        if path.endswith(suffix):
            path = path[:-len(suffix)]
            break
    return urllib.parse.urlunsplit((url.scheme, url.netloc, path, "", ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live terminal view of a vLLM lane.", allow_abbrev=False)
    ap.add_argument("url", nargs="?", type=metrics_base, default="http://spark-4687.local:8888",
                    help="lane base URL (default: %(default)s — the rig's serving lane)")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit")
    ap.add_argument("--interval", type=positive_seconds, default=2.0, help="refresh seconds (default: %(default)s)")
    ap.add_argument("--timeout", type=positive_seconds, default=5.0, help="request timeout seconds (default: %(default)s)")
    ap.add_argument("--no-color", action="store_true",
                    help="plain output (auto when piped or NO_COLOR is set)")
    args = ap.parse_args(argv)

    terminal = sys.stdout.isatty()
    color = terminal and "NO_COLOR" not in os.environ and not args.no_color
    c = palette(color)

    hist_pre, hist_dec = deque(maxlen=HIST), deque(maxlen=HIST)
    wins = {name: RateWindow() for name in (PROMPT_TOK, GEN_TOK, DRAFT_TOK, ACC_TOK)}
    first, info, info_t = None, {}, 0.0
    prev, prev_t, t0 = None, 0.0, time.monotonic()
    while True:
        try:
            m = parse_metrics(fetch(args.url, timeout=args.timeout))
            if not any(key.startswith("vllm:") for key in m):
                raise ValueError("endpoint returned no vLLM metrics")
        except Exception as e:
            print(f"{c['red']}cannot read {args.url}/metrics: {e}{c['r']}", file=sys.stderr)
            prev = None
            if args.once:
                return 1
            time.sleep(args.interval)
            continue
        now = time.monotonic()
        for name, w in wins.items():
            w.add(now, g(m, name))
        if now - info_t > 300:
            info, info_t = lane_info(args.url, args.timeout), now
        if first is None:
            first, first_t = m, now
        elapsed = now - first_t
        avgs = None
        if elapsed > 1.0:
            avgs = (max(0.0, (g(m, PROMPT_TOK) - g(first, PROMPT_TOK)) / elapsed),
                    max(0.0, (g(m, GEN_TOK) - g(first, GEN_TOK)) / elapsed))
        width = shutil.get_terminal_size().columns if terminal else None
        out = render(args.url, m, prev, now - prev_t, hist_pre, hist_dec, now - t0, c,
                     info=info, width=width, wins=wins, avgs=avgs,
                     session=(g(first, PROMPT_TOK), g(first, GEN_TOK)))
        if args.once:
            print(out)
            return 0
        print(("\033[H\033[J" if terminal else "") + out,
              end="" if terminal else "\n\n", flush=True)
        prev, prev_t = m, now
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        # Prevent a second flush error during interpreter shutdown.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
