#!/usr/bin/env python3
"""weightless dash — live terminal view of a vLLM lane (stdlib only).

Polls the lane's Prometheus /metrics endpoint and renders the numbers that
answer "what is the server doing right now": prefill/decode throughput,
engine step rate, queue depth, KV pressure, prefix-cache health, TTFT
percentiles, request lifecycle, preemptions and speculative-decode
acceptance. Works against any vLLM endpoint, local or remote.

Usage:
    python3 scripts/dash.py [url]                 live view (5s refresh)
    python3 scripts/dash.py [url] --once          one snapshot, for scripts
    python3 scripts/dash.py [url] --once --json   machine-readable snapshot
    python3 scripts/dash.py [url] --interval 2    faster refresh
    python3 scripts/dash.py [url] --window 60     longer rate window (default 30s)
    python3 scripts/dash.py [url] --no-color      plain output (also: NO_COLOR=1)
    python3 scripts/dash.py [url] --bench --out run1.json
                                                  active benchmark: warmup, 4 serial
                                                  cases + a concurrency wave, each
                                                  trial isolated by request counters
    python3 scripts/dash.py [url] --bench --save-baseline
                                                  bless the results as this lane's
                                                  baseline (shown as base N.NNx ratios)

Keys in the live view:  q quit · p pause · r reset session counters ·
w cycle the rate window (10 / 30 / 60 / 300s).

How the numbers are made
    Every counter is read twice: the headline is the delta over the rate
    window (steady, lags by up to a window), `now` is the delta since the
    previous poll (instant, jittery) and the sparkline plots `now`. The
    prefill headline is tokens the engine actually computed
    (prefix_cache_queries − prefix_cache_hits); `admitted` is what
    prompt_tokens_total counts, cache hits included. Series are summed
    across the `engine` label, so data-parallel lanes report the whole
    lane. `--once` takes two samples --interval apart so its rates are real.
    `ms/step` in the spec-decode row is decode-phase wall time per forward
    step (decode seconds delta ÷ drafts delta) — the number that decides how
    much speculative decoding actually buys. --bench replays fixed prompts
    with cache-busting nonces, waits for an idle lane, and asserts every
    trial is exactly isolated in the request counters — other traffic fails
    the run loudly instead of silently skewing it. Baselines live in
    ~/.cache/weightless/dash/baselines.json keyed by model|ctx|version.

Colors are on when stdout is a terminal: pink/cyan brand accents (the
setup.py palette), green → yellow → red as KV pressure climbs, yellow on
queued requests, red on preemptions. Output piped to a file or pipe is
always plain.

Examples:
    python3 scripts/dash.py                                   # glm53-flash on the rig
    python3 scripts/dash.py http://spark-4687.local:8888      # DSV4 lane
    python3 scripts/dash.py https://<modal-endpoint> --once   # cloud lane snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

SPARK = "▁▂▃▄▅▆▇█"
HIST = 60                 # sparkline samples kept (one per poll)
WINDOW = 30.0             # seconds behind every headline rate
LONG_WINDOW = 300.0       # per-request histograms only move on completion
WINDOWS = (10.0, 30.0, 60.0, 300.0)   # `w` key cycle
NA = "—"


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


# ---------------------------------------------------------------- metrics ---

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


def series(m: dict, name: str) -> list:
    """Every (labels, value) sample of a metric: the bare key plus every
    `name|labels` key. Does not match `name_suffix|...`."""
    out = []
    if name in m:
        out.append(("", m[name]))
    prefix = name + "|"
    for k, v in m.items():
        if k.startswith(prefix):
            out.append((k[len(prefix):], v))
    return out


def has(m: dict, name: str) -> bool:
    return bool(series(m, name))


def g(m: dict, name: str, default: float = 0.0) -> float:
    """Label-agnostic lookup summed across label sets — counters and
    request gauges are additive across the `engine` label vLLM puts on
    every series, so a data-parallel lane reports the whole lane."""
    s = series(m, name)
    return sum(v for _, v in s) if s else default


def gmean(m: dict, name: str, default: float = 0.0) -> float:
    """Label-agnostic mean across label sets (for ratios like KV usage)."""
    s = series(m, name)
    return sum(v for _, v in s) / len(s) if s else default


def label(labels: str, key: str) -> str | None:
    mch = re.search(r'(?:^|[|,])' + re.escape(key) + r'="([^"]*)"', labels)
    return mch.group(1) if mch else None


def pick(m: dict, *names: str) -> str:
    """First metric name present in the payload (vLLM renames: KV usage,
    ITL); the preferred name when none is."""
    for name in names:
        if has(m, name):
            return name
    return names[0]


def buckets(m: dict, name: str) -> dict:
    """Cumulative histogram buckets {le: count}, summed across label sets."""
    out: dict = {}
    for labels, v in series(m, name + "_bucket"):
        le = label(labels, "le")
        if le is None:
            continue
        try:
            edge = float(le)
        except ValueError:
            continue
        out[edge] = out.get(edge, 0.0) + v
    return out


PROMPT_TOK = "vllm:prompt_tokens_total"
GEN_TOK = "vllm:generation_tokens_total"
DRAFT_TOK = "vllm:spec_decode_num_draft_tokens_total"
ACC_TOK = "vllm:spec_decode_num_accepted_tokens_total"
DRAFTS = "vllm:spec_decode_num_drafts_total"
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
CACHE_Q = "vllm:prefix_cache_queries_total"
CACHE_H = "vllm:prefix_cache_hits_total"
PREEMPT = "vllm:num_preemptions_total"
ITER_SUM = "vllm:iteration_tokens_total_sum"
ITER_CNT = "vllm:iteration_tokens_total_count"
KV_NAMES = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
ITL_NAMES = ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds")
TTFT = "vllm:time_to_first_token_seconds"
E2E = "vllm:e2e_request_latency_seconds"
QUEUE_T = "vllm:request_queue_time_seconds"
PREF_T = "vllm:request_prefill_time_seconds"
DEC_T = "vllm:request_decode_time_seconds"
REQ_PROMPT = "vllm:request_prompt_tokens"
REQ_GEN = "vllm:request_generation_tokens"
SUCCESS = "vllm:request_success_total"
CACHE_CFG = "vllm:cache_config_info"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"

# Counters rated over the short window; per-request histograms (sum/count)
# over the long one, because they only record when a request completes.
SHORT = (PROMPT_TOK, GEN_TOK, DRAFT_TOK, ACC_TOK, DRAFTS, CACHE_Q, CACHE_H, PREEMPT,
         ITER_SUM, ITER_CNT) + tuple(n + "_sum" for n in ITL_NAMES)
LONG = tuple(n + s for n in (TTFT, E2E, QUEUE_T, PREF_T, DEC_T, REQ_PROMPT, REQ_GEN)
             for s in ("_sum", "_count"))


# ---------------------------------------------------------------- windows ---

class RateWindow:
    """Sliding-window rate over a monotonic counter. Single-interval deltas
    spike (a 2s window with one big prefill batch reads as 100k tok/s);
    dividing the counter delta over the whole window span tracks the real
    trend. A counter that goes backwards (server restart) resets the window."""

    def __init__(self, span: float = WINDOW) -> None:
        self.span = span
        self.samples: deque = deque()

    def add(self, t: float, value: float) -> None:
        if self.samples and value < self.samples[-1][1]:
            self.samples.clear()
        self.samples.append((t, value))
        self._prune(t)

    def _prune(self, t: float) -> None:
        while len(self.samples) > 2 and self.samples[0][0] < t - self.span:
            self.samples.popleft()

    def set_span(self, span: float) -> None:
        self.span = span
        if self.samples:
            self._prune(self.samples[-1][0])

    def delta(self) -> float:
        """Counter growth over the window (not per second)."""
        if len(self.samples) < 2:
            return 0.0
        return max(0.0, self.samples[-1][1] - self.samples[0][1])

    def rate(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        (t0, v0), (t1, v1) = self.samples[0], self.samples[-1]
        if t1 <= t0:
            return 0.0
        return max(0.0, (v1 - v0) / (t1 - t0))


class HistWindow:
    """Windowed quantiles from cumulative Prometheus histogram buckets —
    the delta of every bucket over the window, then the usual
    histogram_quantile linear interpolation inside the chosen bucket."""

    def __init__(self, span: float = LONG_WINDOW) -> None:
        self.span = span
        self.samples: deque = deque()

    def add(self, t: float, bkts: dict) -> None:
        total = bkts.get(math.inf, 0.0)
        if self.samples and total < self.samples[-1][1].get(math.inf, 0.0):
            self.samples.clear()
        self.samples.append((t, bkts))
        while len(self.samples) > 2 and self.samples[0][0] < t - self.span:
            self.samples.popleft()

    def quantile(self, q: float) -> float | None:
        if len(self.samples) < 2:
            return None
        b0, b1 = self.samples[0][1], self.samples[-1][1]
        total = b1.get(math.inf, 0.0) - b0.get(math.inf, 0.0)
        if total <= 0:
            return None
        target = q * total
        prev_le, prev_c = 0.0, 0.0
        for le in sorted(b1):
            c = b1[le] - b0.get(le, 0.0)
            if c >= target:
                if math.isinf(le):
                    return prev_le  # Prometheus returns the last finite edge
                frac = (target - prev_c) / (c - prev_c) if c > prev_c else 0.0
                return prev_le + (le - prev_le) * frac
            prev_le, prev_c = le, c
        return None


# ------------------------------------------------------------------ state ---

class Lane:
    """Everything the dash remembers between polls. `observe(m, now)` folds a
    metrics sample in; `snapshot()` is the derived numbers (None = unknown)
    that `render` and `--json` print."""

    def __init__(self, span: float = WINDOW, long_span: float = LONG_WINDOW) -> None:
        self.span = span
        self.wins = {name: RateWindow(span) for name in SHORT}
        self.wins.update({name: RateWindow(long_span) for name in LONG})
        self.ttft = HistWindow(long_span)
        self.hist_pre: deque = deque(maxlen=HIST)   # instant prefill tok/s per poll
        self.hist_dec: deque = deque(maxlen=HIST)
        self.run_hist: deque = deque()              # (t, running) for busy%
        self.m: dict = {}
        self.prev: dict | None = None
        self.prev_t = 0.0
        self.first: dict | None = None
        self.first_t = 0.0
        self.t0: float | None = None
        self.now = 0.0

    def set_span(self, span: float) -> None:
        self.span = span
        for name in SHORT:
            self.wins[name].set_span(span)

    def reset_session(self) -> None:
        self.first, self.first_t = self.m, self.now
        self.hist_pre.clear()
        self.hist_dec.clear()

    def observe(self, m: dict, now: float) -> None:
        if self.t0 is None:
            self.t0 = now
        self.m, self.now = m, now
        if self.first is None or (g(m, PROMPT_TOK) + g(m, GEN_TOK)
                                  < g(self.first, PROMPT_TOK) + g(self.first, GEN_TOK)):
            self.reset_session()  # first sample, or the server restarted
        for name, w in self.wins.items():
            w.add(now, g(m, name))
        self.ttft.add(now, buckets(m, TTFT))
        if self.prev is not None and now > self.prev_t:
            dt = now - self.prev_t
            self.hist_pre.append(max(0.0, (self._computed(m) - self._computed(self.prev)) / dt))
            self.hist_dec.append(max(0.0, (g(m, GEN_TOK) - g(self.prev, GEN_TOK)) / dt))
        self.run_hist.append((now, g(m, RUNNING)))
        while len(self.run_hist) > 2 and self.run_hist[0][0] < now - self.span:
            self.run_hist.popleft()
        self.prev, self.prev_t = m, now

    @staticmethod
    def _computed(m: dict) -> float:
        """Prefill tokens the engine actually ran: prompt tokens minus what
        the prefix cache served. Falls back to prompt_tokens_total on lanes
        without prefix-cache counters."""
        if has(m, CACHE_Q):
            return max(0.0, g(m, CACHE_Q) - g(m, CACHE_H))
        return g(m, PROMPT_TOK)

    def _phase(self, name: str) -> float | None:
        sr, cr = self.wins[name + "_sum"].rate(), self.wins[name + "_count"].rate()
        return sr / cr if cr > 0 else None

    def _busy(self) -> float | None:
        if len(self.run_hist) < 2:
            return None
        items = list(self.run_hist)
        tot = items[-1][0] - items[0][0]
        on = sum(tb - ta for (ta, ra), (tb, _) in zip(items, items[1:]) if ra > 0)
        return on / tot if tot > 0 else None

    def snapshot(self) -> dict:
        m, w = self.m, self.wins
        warm = len(w[GEN_TOK].samples) >= 2   # rates need two samples
        pre_total = w[PROMPT_TOK].rate()
        q_rate, h_rate = w[CACHE_Q].rate(), w[CACHE_H].rate()
        cache = has(m, CACHE_Q)
        pre = max(0.0, q_rate - h_rate) if cache else pre_total
        dec = w[GEN_TOK].rate()
        itl_rate = w[pick(m, *(n + "_sum" for n in ITL_NAMES))].rate()
        running, waiting = g(m, RUNNING), g(m, WAITING)
        kv_name = pick(m, *KV_NAMES)
        steps = w[ITER_CNT].rate()
        drafts_tok, acc_tok = w[DRAFT_TOK].rate(), w[ACC_TOK].rate()
        drafts = w[DRAFTS].rate()
        d_tot, a_tot, n_tot = g(m, DRAFT_TOK), g(m, ACC_TOK), g(m, DRAFTS)
        by_pos: dict = {}
        for labels, v in series(m, PER_POS):
            pos = label(labels, "position")
            if pos and pos.isdigit():
                by_pos[int(pos)] = by_pos.get(int(pos), 0.0) + v
        per_pos = sorted(by_pos.items())
        norm = n_tot if n_tot > 0 else (per_pos[0][1] if per_pos else 0.0)
        done = {}
        for labels, v in series(m, SUCCESS):
            reason = label(labels, "finished_reason")
            if reason:
                done[reason] = done.get(reason, 0) + int(v)
        kv_cap = None
        for labels, _ in series(m, CACHE_CFG):
            try:
                kv_cap = int(float(label(labels, "num_gpu_blocks") or 0)) * \
                    int(float(label(labels, "block_size") or 0)) or None
            except ValueError:
                kv_cap = None
            break
        elapsed = self.now - self.first_t if self.first else 0.0
        ttft_c, ttft_s = g(m, TTFT + "_count"), g(m, TTFT + "_sum")
        preempt_life = g(m, PREEMPT)
        return {
            "watching_s": self.now - (self.t0 or self.now),
            "window_s": self.span,
            "warm": warm,
            "prefill": pre,
            "prefill_admitted": pre_total if cache else None,
            "prefill_now": self.hist_pre[-1] if self.hist_pre else None,
            "prefill_peak": max(self.hist_pre) if self.hist_pre else None,
            "cached_pct": (h_rate / q_rate * 100) if q_rate > 0 else None,
            "cache_hit_life_pct": (g(m, CACHE_H) / g(m, CACHE_Q) * 100) if g(m, CACHE_Q) > 0 else None,
            "decode": dec,
            "decode_now": self.hist_dec[-1] if self.hist_dec else None,
            "decode_peak": max(self.hist_dec) if self.hist_dec else None,
            "stream_per_req": dec / itl_rate if itl_rate > 0 else None,
            "decode_per_req": dec / running if running > 0 else None,
            "avg_prefill": (max(0.0, self._computed(m) - self._computed(self.first)) / elapsed
                            if elapsed > 1.0 else None),
            "avg_decode": (max(0.0, g(m, GEN_TOK) - g(self.first, GEN_TOK)) / elapsed
                           if elapsed > 1.0 else None),
            "session_prompt": max(0.0, g(m, PROMPT_TOK) - g(self.first, PROMPT_TOK)) if self.first else 0.0,
            "session_gen": max(0.0, g(m, GEN_TOK) - g(self.first, GEN_TOK)) if self.first else 0.0,
            "running": int(running),
            "waiting": int(waiting),
            "kv_pct": gmean(m, kv_name) * 100 if has(m, kv_name) else None,
            "kv_capacity_tok": kv_cap,
            "busy": self._busy(),
            "steps_per_s": steps if has(m, ITER_CNT) else None,
            "tok_per_step": (w[ITER_SUM].rate() / steps) if steps > 0 else None,
            "preempt_total": int(preempt_life) if has(m, PREEMPT) else None,
            "preempt_rate": w[PREEMPT].rate() if has(m, PREEMPT) else None,
            "ttft_p50": self.ttft.quantile(0.5),
            "ttft_p90": self.ttft.quantile(0.9),
            "ttft_mean": self._phase(TTFT),
            "ttft_life": (ttft_s / ttft_c) if ttft_c > 0 else None,
            "queue_s": self._phase(QUEUE_T),
            "prefill_s": self._phase(PREF_T),
            "decode_s": self._phase(DEC_T),
            "e2e_s": self._phase(E2E),
            "req_prompt_tok": self._phase(REQ_PROMPT),
            "req_gen_tok": self._phase(REQ_GEN),
            "spec": d_tot > 0,
            "accept_pct": (acc_tok / drafts_tok * 100) if drafts_tok > 0
                          else (a_tot / d_tot * 100) if d_tot > 0 else None,
            "accept_len": (1 + acc_tok / drafts) if drafts > 0
                          else (1 + a_tot / n_tot) if n_tot > 0 else None,
            # Decode-phase wall time per forward step. drafts (short window)
            # and decode seconds (long window) both scale with concurrency,
            # so the ratio is a per-step time regardless of load.
            "ms_per_step": (1000 * w[DEC_T + "_sum"].rate() / drafts)
                           if drafts > 0 and w[DEC_T + "_sum"].rate() > 0 else None,
            "draft_rate": drafts_tok,
            "accepted_rate": acc_tok,
            "per_pos_pct": [v / norm * 100 for _, v in per_pos] if norm > 0 else [],
            "done": done,
            "prompt_total": g(m, PROMPT_TOK),
            "gen_total": g(m, GEN_TOK),
        }


# ----------------------------------------------------------------- render ---

def spark(vals, width: int = HIST) -> str:
    if not vals:
        return ""
    v = list(vals)[-width:]
    peak = max(v) or 1.0
    return "".join(SPARK[min(7, int(x / peak * 7.999))] for x in v)


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def visible(line: str) -> int:
    return len(ANSI.sub("", line))


def clip(line: str, width: int | None) -> str:
    """ANSI-aware truncate to a terminal width; no-op when width is falsy.
    Appends a reset when a colored line was cut so the terminal can't bleed."""
    if not width or width <= 0 or visible(line) <= width:
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
    """16_964_612 -> '17.0M', 3_023 -> '3.0k', 604 -> '604', 999_999 -> '1.0M'."""
    a = abs(n)
    for scale, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= scale * 0.99995:  # what would print as 1000.0k is 1.0M
            return f"{n / scale:.1f}{suf}"
    return f"{n:.0f}"


def fmt_rate(n: float | None) -> str:
    if n is None:
        return NA
    return human(n) if n >= 1000 else f"{n:.1f}"


def fmt_secs(s: float | None) -> str:
    if s is None:
        return NA
    return f"{s * 1000:.0f}ms" if s < 1 else f"{s:.1f}s"


def fmt_pct(p: float | None) -> str:
    return NA if p is None else f"{p:.0f}%"


def fmt_ctx(n: int) -> str:
    """Context lengths are powers of two: 131072 -> '128k', 1048576 -> '1M'."""
    if n >= 1 << 20 and n % (1 << 20) == 0:
        return f"{n >> 20}M"
    if n >= 1024 and n % 1024 == 0:
        return f"{n >> 10}k"
    return human(n)


def fmt_dur(seconds: float) -> str:
    m = int(seconds // 60)
    return f"{m}m" if m < 60 else f"{m // 60}h{m % 60:02d}m"


def fetch_json(base: str, path: str, timeout: float) -> dict:
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


def ratio_txt(c: dict, current: float | None, base: float | None) -> str:
    """`base 0.94×` suffix — green ≥ 0.97, yellow ≥ 0.90, red below."""
    if not base or current is None or current <= 0:
        return ""
    ratio = current / base
    col = c["green"] if ratio >= 0.97 else c["yellow"] if ratio >= 0.90 else c["red"]
    return f"  {c['d']}base {col}{ratio:.2f}×{c['r']}"


def render(target: str, s: dict, c: dict, info: dict | None = None,
           width: int | None = None, error: str | None = None,
           paused: bool = False, baseline: dict | None = None) -> str:
    d, r, b = c["d"], c["r"], c["b"]
    win = f"{s['window_s']:.0f}s"
    L = []
    state = f"  {c['yellow']}PAUSED{r}" if paused else ""
    L.append(f"{c['pink']}{b}weightless{r} {c['cyan']}{b}dash{r}"
             f" {d}— {target} · watching {fmt_dur(s['watching_s'])} · window {win}"
             f" · {time.strftime('%H:%M:%S')}{r}{state}")
    if info:
        bits = []
        if info.get("model"):
            bits.append(str(info["model"]))
        if info.get("ctx"):
            bits.append(f"ctx {fmt_ctx(info['ctx'])}")
        if info.get("version"):
            bits.append(f"vllm {info['version']}")
        if bits:
            L.append(f"  {d}lane{r}  " + f"{d} · {r}".join(bits))
    if error:
        L.append(f"  {c['red']}error{r}  {error}  {d}(showing last good sample){r}")
    L.append("")

    def row(name: str, col: str, rate: float, now, peak, avg, extra: str, hist) -> None:
        head = (f"  {col}{name:<7}{r} {b}{fmt_rate(rate):>7}{r} tok/s"
                f"  {d}now {fmt_rate(now):>6}  peak {fmt_rate(peak):>6}{r}")
        if avg is not None:
            head += f"  {d}avg {fmt_rate(avg)}{r}"
        head += extra
        sw = max(8, width - visible(head) - 3) if width else HIST
        L.append(f"{head}  {col}{spark(hist, sw)}{r}")

    pre_extra = ""
    if s["prefill_admitted"] is not None:
        pre_extra = (f"  {d}admitted {fmt_rate(s['prefill_admitted'])}"
                     f" · cached {fmt_pct(s['cached_pct'])}{r}")
    pre_extra += ratio_txt(c, s["prefill"], (baseline or {}).get("prefill_tps"))
    dec_extra = ""
    if s["stream_per_req"] is not None:
        dec_extra = f"  {d}stream {fmt_rate(s['stream_per_req'])}/req{r}"
    dec_extra += ratio_txt(c, s["decode"], (baseline or {}).get("decode_tps"))
    row("prefill", c["cyan"], s["prefill"], s["prefill_now"], s["prefill_peak"],
        s["avg_prefill"], pre_extra, s.get("_hist_pre", ()))
    row("decode", c["pink"], s["decode"], s["decode_now"], s["decode_peak"],
        s["avg_decode"], dec_extra, s.get("_hist_dec", ()))
    L.append("")

    waiting = s["waiting"]
    if waiting > 0:
        why = "KV full" if (s["kv_pct"] or 0) >= 90 else "raise MAX_NUM_SEQS"
        wait_txt = f"{c['yellow']}{waiting}{r}  {c['red']}← QUEUED ({why}){r}"
    else:
        wait_txt = f"{waiting}"
    per_req = (f"  {d}decode {fmt_rate(s['decode_per_req'])} tok/s/req{r}"
               if s["decode_per_req"] is not None else "")
    L.append(f"  {d}requests{r}   running {s['running']}  waiting {wait_txt}{per_req}")

    engine = []
    if s["steps_per_s"] is not None:
        engine.append(f"steps {s['steps_per_s']:.1f}/s")
        if s["tok_per_step"] is not None:
            engine.append(f"tok/step {s['tok_per_step']:.1f}")
    if s["busy"] is not None:
        engine.append(f"busy {s['busy'] * 100:.0f}%")
    if s["preempt_total"] is not None:
        col = c["red"] if (s["preempt_rate"] or 0) > 0 else d
        engine.append(f"{col}preempt {s['preempt_total']} (+{s['preempt_rate']:.2f}/s){r}{d}")
    if engine:
        L.append(f"  {d}engine{r}     {d}{'   '.join(engine)}{r}")

    kv = gauge(c, s["kv_pct"]) if s["kv_pct"] is not None else NA
    cap = f" of {human(s['kv_capacity_tok'])} tok" if s["kv_capacity_tok"] else ""
    hit = (f"{c['green']}{fmt_pct(s['cached_pct'])}{r}" if s["cached_pct"] is not None
           else NA)
    life = f"  {d}lifetime {fmt_pct(s['cache_hit_life_pct'])}{r}" if s["cache_hit_life_pct"] is not None else ""
    L.append(f"  {d}kv cache{r}   {kv} used{cap}    {d}prefix hit{r} {hit}{life}")
    L.append(f"  {d}ttft{r}       p50 {fmt_secs(s['ttft_p50'])}  p90 {fmt_secs(s['ttft_p90'])}"
             f"  mean {fmt_secs(s['ttft_mean'])}   {d}lifetime {fmt_secs(s['ttft_life'])}"
             f" (incl. queue wait){r}")
    if all(s[k] is not None for k in ("queue_s", "prefill_s", "decode_s")):
        e2e = f"   {d}e2e {fmt_secs(s['e2e_s'])}{r}" if s["e2e_s"] is not None else ""
        L.append(f"  {d}lifecycle{r}  queue {fmt_secs(s['queue_s'])} → prefill {fmt_secs(s['prefill_s'])}"
                 f" → decode {fmt_secs(s['decode_s'])}{e2e}   {d}per request{r}")
    if s["req_prompt_tok"] is not None or s["req_gen_tok"] is not None:
        L.append(f"  {d}request{r}    prompt {fmt_rate(s['req_prompt_tok'])} tok"
                 f"   gen {fmt_rate(s['req_gen_tok'])} tok   {d}mean, finished requests{r}")
    if s["spec"]:
        pos = " ".join(f"{p:.0f}" for p in s["per_pos_pct"])
        acc_len = f"   acc.len {s['accept_len']:.2f}" if s["accept_len"] is not None else ""
        step = f"   step {s['ms_per_step']:.1f}ms" if s["ms_per_step"] is not None else ""
        L.append(f"  {d}spec dec{r}   accept {c['green']}{fmt_pct(s['accept_pct'])}{r}{acc_len}{step}"
                 f"   draft {fmt_rate(s['draft_rate'])} → accepted {fmt_rate(s['accepted_rate'])} tok/s"
                 f"   {d}per-pos {pos}{r}")
    L.append("")
    done = s["done"]
    L.append(f"  {d}done {sum(done.values())}  ({', '.join(f'{k} {v}' for k, v in sorted(done.items()))}){r}")
    L.append(f"  {d}tokens{r}    prompt {human(s['prompt_total'])}   gen {human(s['gen_total'])}"
             f"   {d}this dash: +{human(s['session_prompt'])} prompt   +{human(s['session_gen'])} gen{r}")
    if not s["warm"]:
        L.append(f"  {d}rates need a second sample{r}")
    if width:
        L = [clip(line, width) for line in L]
    return "\n".join(L)


# --------------------------------------------------------------- baseline ---

def baseline_file() -> str:
    return (os.environ.get("WEIGHTLESS_DASH_BASELINES")
            or os.path.expanduser("~/.cache/weightless/dash/baselines.json"))


def baseline_key(info: dict) -> str | None:
    """A baseline only means something against the same lane identity —
    same served model, context length and engine build."""
    if not info.get("model"):
        return None
    return "|".join(str(info.get(k, "?")) for k in ("model", "ctx", "version"))


def load_baselines() -> dict:
    try:
        with open(baseline_file()) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_baseline(info: dict) -> dict | None:
    key = baseline_key(info)
    base = load_baselines().get(key) if key else None
    return dict(base, key=key) if isinstance(base, dict) else None


def save_baseline(info: dict, fields: dict) -> str:
    key = baseline_key(info)
    if not key:
        raise RuntimeError("lane identity unknown (no /v1/models id) — cannot key a baseline")
    kept = {k: v for k, v in fields.items() if v is not None}
    if not kept:
        raise RuntimeError("no live rates to bless — run against a loaded lane")
    path = baseline_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    allbases = load_baselines()
    allbases[key] = {"saved_at": time.time(), **kept}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(allbases, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------- bench ---
#
# Methodology (after coolbho3k's draft-sampling replay driver): never measure
# a request the lane was already handling. Every trial — warmup, serial case,
# concurrency wave — starts on an idle lane, snapshots the request counters,
# and afterwards asserts the counter delta contains exactly the trial's
# requests and tokens. Other traffic fails the run loudly instead of
# silently skewing it. Prompts carry a per-run nonce so a previous bench
# run's prefix cache cannot serve this one.

BENCH_SENTENCES = (
    "Throughput without latency is a number without a question.",
    "The cache holds keys and values for every token seen so far.",
    "A queue forms when the KV cache cannot admit another sequence.",
    "Prefill is parallel over positions; decode is one step per token.",
    "Speculative decoding spends draft tokens to save forward passes.",
    "A good benchmark isolates the trial from every other request.",
    "The tail of the latency distribution is where users actually live.",
    "Prefix caching turns repeated context into free context.",
)


def bench_passage() -> str:
    """~2.4k tokens of stable, non-degenerate prose for the prefill case."""
    return "\n".join(f"Note {i}: {BENCH_SENTENCES[i % len(BENCH_SENTENCES)]}"
                     for i in range(160))


def bench_cases() -> list:
    return [
        ("short", "Briefly introduce yourself in two sentences.", 64),
        ("medium", "Explain how a transformer KV cache works, with a small worked example.", 256),
        ("long", "Write a detailed design document for a token-bucket rate limiter: "
                 "goals, architecture, edge cases, and a phased rollout plan.", 1024),
        ("prefill", "Summarize the following notes in one sentence.\n\n" + bench_passage(), 32),
    ]


ISO_COUNTERS = (PROMPT_TOK, GEN_TOK, DRAFT_TOK, ACC_TOK, DRAFTS, PREEMPT, SUCCESS,
                REQ_GEN + "_count", REQ_GEN + "_sum", TTFT + "_sum",
                QUEUE_T + "_sum", PREF_T + "_sum", DEC_T + "_sum", E2E + "_sum")


def counter_snapshot(m: dict) -> dict:
    snap = {name: g(m, name) for name in ISO_COUNTERS}
    snap["finished"] = (g(m, REQ_GEN + "_count") if has(m, REQ_GEN + "_count")
                        else g(m, SUCCESS))
    return snap


def require_isolation_metrics(m: dict) -> None:
    if not has(m, REQ_GEN + "_count") and not has(m, SUCCESS):
        raise RuntimeError("lane exposes no finished-request counters "
                           "(vllm:request_generation_tokens) — bench cannot isolate trials")


def wait_idle(url: str, timeout: float, deadline_s: float) -> dict:
    deadline = time.monotonic() + deadline_s
    while True:
        m = sample(url, timeout)
        running, waiting = g(m, RUNNING), g(m, WAITING)
        if running == 0 and waiting == 0:
            return m
        if time.monotonic() >= deadline:
            raise RuntimeError(f"lane busy: {int(running)} running / {int(waiting)} waiting "
                               f"after {deadline_s:.0f}s — bench needs the lane to itself")
        time.sleep(1.0)


def await_counters(url: str, timeout: float, before: dict, n: int,
                   tries: int = 20, poll_s: float = 0.25) -> dict:
    """Per-request counters land shortly after the HTTP response — poll
    until this trial's requests are visible (usually one check, ≤ 5s)."""
    for _ in range(tries):
        snap = counter_snapshot(sample(url, timeout))
        if snap["finished"] - before["finished"] >= n:
            return snap
        time.sleep(poll_s)
    raise RuntimeError(f"metrics did not record {n} finished request(s) "
                       f"within {tries * poll_s:.0f}s")


def check_isolation(before: dict, after: dict, n: int, gen_tokens: float) -> dict:
    delta = {k: after[k] - before[k] for k in before}
    if any(v < 0 for v in delta.values()):
        raise RuntimeError("a counter went backwards mid-trial (engine restart?)")
    if delta["finished"] != n:
        raise RuntimeError(f"request counters do not isolate the trial: expected {n} "
                           f"finished, counted {delta['finished']:g} — other traffic on the lane?")
    if delta[GEN_TOK] != gen_tokens:
        raise RuntimeError(f"generation counter does not isolate the trial: expected "
                           f"{gen_tokens:g} tokens, counted {delta[GEN_TOK]:g}")
    return delta


def chat_completion(base: str, model: str, content: str, max_tokens: int,
                    temperature: float, timeout: float) -> tuple:
    payload = {"model": model,
               "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": temperature, "top_p": 1.0}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    usage = body.get("usage") or {}
    if "completion_tokens" not in usage:
        raise RuntimeError("chat response carried no usage block — cannot isolate the trial")
    return usage, time.monotonic() - t0


def trial_row(label: str, usage: dict, wall: float, delta: dict) -> dict:
    comp, prompt = usage["completion_tokens"], usage.get("prompt_tokens", 0)
    decode_s = delta[DEC_T + "_sum"] or None
    prefill_s = delta[PREF_T + "_sum"] or None
    drafts = delta[DRAFTS]
    return {
        "label": label,
        "prompt_tokens": prompt,
        "completion_tokens": comp,
        "wall_s": round(wall, 3),
        "ttft_s": delta[TTFT + "_sum"] or None,
        "queue_s": delta[QUEUE_T + "_sum"] or None,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "e2e_s": delta[E2E + "_sum"] or None,
        "decode_tps": comp / decode_s if decode_s else None,
        "prompt_tps": prompt / prefill_s if prefill_s else None,
        "accept_pct": delta[ACC_TOK] / delta[DRAFT_TOK] * 100 if delta[DRAFT_TOK] > 0 else None,
        "tok_per_step": comp / drafts if drafts > 0 else None,
        "ms_per_step": 1000 * decode_s / drafts if decode_s and drafts > 0 else None,
    }


def run_trial(args, model: str, label: str, content: str, max_tokens: int) -> dict:
    wait_idle(args.url, args.timeout, args.idle_timeout)
    before = counter_snapshot(sample(args.url, args.timeout))
    usage, wall = chat_completion(args.url, model, content, max_tokens,
                                  args.temperature, args.request_timeout)
    delta = check_isolation(before, await_counters(args.url, args.timeout, before, 1),
                            1, usage["completion_tokens"])
    return trial_row(label, usage, wall, delta)


def run_wave(args, model: str, nonce: str) -> dict:
    n = args.wave
    wait_idle(args.url, args.timeout, args.idle_timeout)
    before = counter_snapshot(sample(args.url, args.timeout))
    barrier = threading.Barrier(n + 1)

    def one(i: int):
        barrier.wait(timeout=15)
        return chat_completion(args.url, model,
                               f"[bench {nonce} #{i}] Explain how a transformer KV cache "
                               f"works, with a small worked example.",
                               256, args.temperature, args.request_timeout)

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(one, i) for i in range(n)]
        started = time.monotonic()
        barrier.wait(timeout=15)
        results = [f.result() for f in futures]
        elapsed = time.monotonic() - started
    tokens = sum(u["completion_tokens"] for u, _ in results)
    delta = check_isolation(before, await_counters(args.url, args.timeout, before, n),
                            n, tokens)
    return {
        "n": n,
        "elapsed_s": round(elapsed, 3),
        "completion_tokens": tokens,
        "aggregate_tps": tokens / elapsed if elapsed > 0 else None,
        "wall_s_per_req": [round(w, 3) for _, w in results],
        "accept_pct": delta[ACC_TOK] / delta[DRAFT_TOK] * 100 if delta[DRAFT_TOK] > 0 else None,
        "ms_per_step": (1000 * delta[DEC_T + "_sum"] / delta[DRAFTS]
                        if delta[DEC_T + "_sum"] and delta[DRAFTS] > 0 else None),
    }


def script_sha256() -> str | None:
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def bench(args, err_c: dict) -> int:
    info = lane_info(args.url, args.timeout)
    model = info.get("model")
    if not model:
        print(f"{err_c['red']}bench needs the served model id, but "
              f"{args.url}/v1/models did not answer{err_c['r']}", file=sys.stderr)
        return 2
    if args.out and os.path.exists(args.out):
        print(f"{err_c['red']}refusing to overwrite {args.out} — preserve previous "
              f"evidence; pick a new --out path{err_c['r']}", file=sys.stderr)
        return 2
    nonce = f"{int(time.time()):x}"
    report = {"status": "running", "target": args.url, "lane": info,
              "started_at": time.time(), "script_sha256": script_sha256(),
              "settings": {"temperature": args.temperature, "top_p": 1.0,
                           "wave": args.wave, "nonce": nonce},
              "cases": []}

    def save() -> None:
        if args.out:
            with open(args.out, "w") as f:
                json.dump(report, f, indent=2)
                f.write("\n")

    try:
        require_isolation_metrics(sample(args.url, args.timeout))
        # Exercise lazy kernel paths before collecting timings; excluded.
        report["warmup"] = run_trial(args, model, "excluded_warmup",
                                     f"[bench {nonce}] List everyday objects and explain "
                                     f"their uses in full sentences.", 64)
        save()
        for label, prompt, max_tokens in bench_cases():
            row = run_trial(args, model, label, f"[bench {nonce}] {prompt}", max_tokens)
            report["cases"].append(row)
            save()
            print(json.dumps({k: row[k] for k in
                              ("label", "decode_tps", "prompt_tps", "ttft_s", "accept_pct")}),
                  flush=True)
        report["wave"] = run_wave(args, model, nonce)
        serial = [r["decode_tps"] for r in report["cases"] if r["decode_tps"]]
        prefill_row = next((r for r in report["cases"]
                            if r["label"] == "prefill" and r["prompt_tps"]), None)
        steps = [r["ms_per_step"] for r in report["cases"] if r["ms_per_step"]]
        accepts = [r["accept_pct"] for r in report["cases"] if r["accept_pct"] is not None]
        report["summary"] = {
            "decode_tps": statistics.median(serial) if serial else None,
            "prefill_tps": prefill_row["prompt_tps"] if prefill_row else None,
            "wave_tps": report["wave"]["aggregate_tps"],
            "accept_pct": statistics.median(accepts) if accepts else None,
            "ms_per_step": statistics.median(steps) if steps else None,
        }
        base = load_baseline(info)
        if base:
            report["baseline"] = {k: v for k, v in base.items() if k != "key"}
            report["vs_baseline"] = {k: round(cur / base[k], 4)
                                     for k, cur in report["summary"].items()
                                     if cur and base.get(k)}
        if args.save_baseline:
            report["saved_baseline"] = save_baseline(
                info, {"source": "bench", **report["summary"]})
        report["status"] = "complete"
    except Exception as e:
        report.update(status="failed", error=repr(e))
        save()
        print(f"{err_c['red']}bench failed: {e}{err_c['r']}", file=sys.stderr)
        return 2
    save()
    summary = {**report["summary"]}
    if report.get("vs_baseline"):
        summary["vs_baseline"] = report["vs_baseline"]
    print(json.dumps({"status": "complete", "summary": summary, "out": args.out}))
    if not args.out:
        print(json.dumps(report, indent=2))
    return 0


# --------------------------------------------------------------- terminal ---

class Screen:
    """Alternate-screen, flicker-free redraw and single-key input for the
    live view. Everything degrades to plain prints when stdout is not a
    terminal, and the terminal is restored on any exit."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.keys = False
        self._saved = None

    def __enter__(self) -> "Screen":
        if not self.enabled:
            return self
        sys.stdout.write("\033[?1049h\033[?25l\033[H\033[2J")
        sys.stdout.flush()
        try:
            import termios
            import tty
            if sys.stdin.isatty():
                self._saved = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
                self.keys = True
        except Exception:
            self.keys = False
        return self

    def __exit__(self, *exc) -> None:
        if not self.enabled:
            return
        if self._saved is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
        sys.stdout.write("\033[?1049l\033[?25h")
        sys.stdout.flush()

    def draw(self, text: str) -> None:
        if self.enabled:
            body = "\n".join(line + "\033[K" for line in text.split("\n"))
            print("\033[H" + body + "\033[J", end="", flush=True)
        else:
            print(text, end="\n\n", flush=True)

    def wait(self, seconds: float) -> str | None:
        """Sleep, or return the first key pressed before the deadline."""
        if not self.keys:
            time.sleep(max(0.0, seconds))
            return None
        import select
        deadline = time.monotonic() + seconds
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            ready, _, _ = select.select([sys.stdin], [], [], left)
            if ready:
                return sys.stdin.read(1)


# ------------------------------------------------------------------- main ---

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


def sample(url: str, timeout: float) -> dict:
    m = parse_metrics(fetch(url, timeout=timeout))
    if not any(key.startswith("vllm:") for key in m):
        raise ValueError("endpoint returned no vLLM metrics")
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live terminal view of a vLLM lane.", allow_abbrev=False)
    ap.add_argument("url", nargs="?", type=metrics_base, default="http://spark-4687.local:8888",
                    help="lane base URL (default: %(default)s — the rig's serving lane)")
    ap.add_argument("--once", action="store_true",
                    help="print one snapshot and exit (two samples --interval apart, so rates are real)")
    ap.add_argument("--json", action="store_true", help="with --once: print the snapshot as JSON")
    ap.add_argument("--interval", type=positive_seconds, default=5.0, help="refresh seconds (default: %(default)s)")
    ap.add_argument("--window", type=positive_seconds, default=WINDOW,
                    help="seconds behind every headline rate (default: %(default)s)")
    ap.add_argument("--timeout", type=positive_seconds, default=5.0, help="request timeout seconds (default: %(default)s)")
    ap.add_argument("--no-color", action="store_true",
                    help="plain output (auto when piped or NO_COLOR is set)")
    ap.add_argument("--bench", action="store_true",
                    help="active benchmark: warmup + 4 serial cases + a concurrency wave, "
                         "every trial isolated by request counters (lane must be idle)")
    ap.add_argument("--wave", type=int, default=6, metavar="N",
                    help="concurrent requests in the bench wave (default: %(default)s)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="bench sampling temperature (default: %(default)s — reproducible)")
    ap.add_argument("--out", metavar="FILE",
                    help="bench report JSON (refuses to overwrite — preserve previous evidence)")
    ap.add_argument("--save-baseline", action="store_true",
                    help="bless current rates (with --once) or bench results (with --bench) "
                         "as this lane's baseline; later runs show base N.NNx ratios")
    ap.add_argument("--idle-timeout", type=positive_seconds, default=30.0,
                    help="seconds --bench waits for an idle lane before failing (default: %(default)s)")
    ap.add_argument("--request-timeout", type=positive_seconds, default=600.0,
                    help="per-request timeout for --bench (default: %(default)s)")
    args = ap.parse_args(argv)
    if args.json:
        args.once = True
    if args.wave < 1:
        ap.error("--wave must be at least 1")
    if args.out and not args.bench:
        ap.error("--out only makes sense with --bench")
    if args.save_baseline and not (args.once or args.bench):
        ap.error("--save-baseline needs --once (bless current rates) or --bench (bless results)")
    if args.temperature < 0:
        ap.error("--temperature must be non-negative")

    terminal = sys.stdout.isatty()
    color = terminal and "NO_COLOR" not in os.environ and not args.no_color
    c = palette(color)
    err_c = palette(color and sys.stderr.isatty())
    if args.bench:
        return bench(args, err_c)
    lane = Lane(span=args.window)

    def snapshot() -> dict:
        s = lane.snapshot()
        s["_hist_pre"], s["_hist_dec"] = lane.hist_pre, lane.hist_dec
        return s

    if args.once:
        try:
            lane.observe(sample(args.url, args.timeout), time.monotonic())
            time.sleep(args.interval)
            lane.observe(sample(args.url, args.timeout), time.monotonic())
        except Exception as e:
            print(f"{err_c['red']}cannot read {args.url}/metrics: {e}{err_c['r']}", file=sys.stderr)
            return 1
        info = lane_info(args.url, args.timeout)
        s = snapshot()
        baseline = load_baseline(info)
        if args.save_baseline:
            try:
                path = save_baseline(info, {
                    "source": "dash --once",
                    "prefill_tps": s["prefill"] or None,
                    "decode_tps": s["decode"] or None,
                    "stream_per_req": s["stream_per_req"],
                    "accept_pct": s["accept_pct"],
                    "ms_per_step": s["ms_per_step"]})
            except Exception as e:
                print(f"{err_c['red']}could not save baseline: {e}{err_c['r']}", file=sys.stderr)
                return 1
            print(f"baseline saved → {path}", file=sys.stderr)
        if args.json:
            out = {k: v for k, v in s.items() if not k.startswith("_")}
            out.update(target=args.url, info=info, time=time.time())
            if baseline:
                out["baseline"] = baseline
            print(json.dumps(out))
        else:
            width = shutil.get_terminal_size().columns if terminal else None
            print(render(args.url, s, c, info=info, width=width, baseline=baseline))
        return 0

    info, info_t, error, paused, baseline = {}, -math.inf, None, False, None
    with Screen(terminal) as screen:
        while True:
            now = time.monotonic()
            if not paused:
                try:
                    lane.observe(sample(args.url, args.timeout), time.monotonic())
                    error = None
                except Exception as e:
                    error = f"cannot read {args.url}/metrics: {e}"
                    if not terminal:
                        print(f"{err_c['red']}{error}{err_c['r']}", file=sys.stderr)
                if now - info_t > 300 and error is None:
                    info, info_t = lane_info(args.url, args.timeout), now
                    baseline = load_baseline(info) or baseline
            if lane.first is not None:
                width = shutil.get_terminal_size().columns if terminal else None
                screen.draw(render(args.url, snapshot(), c, info=info, width=width,
                                   error=error, paused=paused, baseline=baseline))
            elif terminal and error:
                screen.draw(f"{c['red']}{error}{c['r']}  {c['d']}retrying every"
                            f" {args.interval:g}s · q quits{c['r']}")
            key = screen.wait(args.interval - (time.monotonic() - now))
            if key in ("q", "Q", "\x03"):
                return 0
            if key in ("p", "P", " "):
                paused = not paused
            elif key in ("r", "R") and lane.first is not None:
                lane.reset_session()
            elif key in ("w", "W"):
                spans = [w for w in WINDOWS if w > lane.span] or [WINDOWS[0]]
                lane.set_span(spans[0])


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
