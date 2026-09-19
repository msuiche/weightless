#!/usr/bin/env python3
"""Local eval driver for the weightless-steer DSV4-Flash-0731 plugin test.

Hits the Modal-exposed OpenAI endpoint (one deploy per arm — alpha is baked
at model init) with the three reference suites, greedy, max_tokens=400,
n=32/suite, then scores with the repo's four-way scorer
(refusal-research/harness/lib.py classify: COMPLY / DEFLECT / REFUSE /
GARBLED, delivered = COMPLY; ARGUMENTATIVE is a review flag on REFUSE).
This file does NOT invent a scorer — it imports the one that produced the
BENCHMARK.md numbers.

Reference row (BENCHMARK.md GLP-29): refusal32 0/32 -> 19/32 delivered with
the cyber direction at alpha=4.0 (out-of-domain; the cyber direction barely
transfers to CBRN-heavy refusal32). cyber32 is the in-domain suite, benign32
the collateral check.

Output record shape matches the capture-lane evals ({suite}-a{alpha}.json
with {suite, alpha, model, max_new, decoding, items}).

Usage:
    python3 eval_dsv4_plugin.py --base-url https://....modal.run --alpha 0.0
    python3 eval_dsv4_plugin.py --base-url https://....modal.run --alpha 4.0

Protocol note (2026-09-19, run 2): every request carries
chat_template_kwargs={"enable_thinking": false}. The day-0 image's DSV4
serving stack routes chat through vllm/renderers/deepseek_v4.py ->
vllm/tokenizers/deepseek_v4.py, whose apply_chat_template override renders
via DeepSeek's native encode_messages and DEFAULTS THINKING ON when the
request passes no thinking kwarg (the --chat-template jinja is accepted by
the renderer but never consulted by that override). Run 1 (results under
out-dsv4-plugin-test/run1-thinking-on/) went out with thinking on: the
reasoning trace leaked into message.content, ate the 400-token budget, and
its curly-apostrophe phrasing ("I can’t", U+2019) is invisible to the
harness REFUSE_RE, so stock refusal32 scored 23/32 delivered against the
reference 0/32. The reference lanes all ran thinking OFF (20260904-ffn-site
rig spec: "thinking off"; 20260905 Modal lane: offline rendering with the
</think>-spliced jinja). With enable_thinking=false the server's native
encoder renders the canonical no-think format — the same thing the
reference jinja splices by hand. The warm-up request fails closed if the
completion still contains a think block.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SPARK = os.path.dirname(os.path.dirname(HERE))       # spark workspace root
sys.path.insert(0, os.path.join(SPARK, "refusal-research", "harness"))
import lib as Q  # noqa: E402  -- the repo scorer (torch-stubbed on CPU boxes)

SUITES = {
    "refusal32": os.path.join(SPARK, "refusal-research", "suites",
                              "contrasts", "refusal32-suite.json"),
    "cyber32": os.path.join(SPARK, "refusal-research", "suites",
                            "core", "cyber32-suite.json"),
    "benign32": os.path.join(SPARK, "refusal-research", "suites",
                             "core", "benign32-suite.json"),
}
OUT = os.path.join(HERE, "out-dsv4-plugin-test")


def load_suite(path):
    d = json.load(open(path))
    rows = d if isinstance(d, list) else (d.get("results") or d.get("items"))
    return [r["prompt"] if isinstance(r, dict) else r for r in rows]


def http_json(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(base_url, budget_s):
    """The endpoint cold-starts a ~168 GB model boot; poll until /v1/models
    answers. Modal's proxy 5xx/connection errors while cold are expected."""
    t0 = time.time()
    n = 0
    e = None
    while True:
        n += 1
        try:
            models = http_json("GET", f"{base_url}/v1/models", timeout=120)
            ids = [m["id"] for m in models.get("data", [])]
            if ids:
                print(f"ready after {time.time() - t0:.0f}s: models={ids}",
                      flush=True)
                return ids[0]
            e = RuntimeError("no models listed")
        except Exception as exc:
            e = exc
        if time.time() - t0 > budget_s:
            raise TimeoutError(f"endpoint never became ready in {budget_s}s")
        if n % 10 == 1:
            print(f"waiting for boot... {time.time() - t0:.0f}s "
                  f"({type(e).__name__}: {e})", flush=True)
        time.sleep(30)


def one_request(base_url, model, prompt, max_tokens, tries=4):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            # The day-0 image's DSV4 renderer thinks by default; the
            # reference protocol is thinking-off. See the module docstring.
            "chat_template_kwargs": {"enable_thinking": False}}
    last = None
    for attempt in range(tries):
        try:
            t0 = time.time()
            resp = http_json("POST", f"{base_url}/v1/chat/completions", body,
                             timeout=900)
            ch = resp["choices"][0]
            usage = resp.get("usage") or {}
            return {"completion": ch["message"].get("content") or "",
                    "finish_reason": ch.get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "wall_s": round(time.time() - t0, 2)}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            print(f"  request retry {attempt + 1}/{tries}: {e}", flush=True)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"request failed {tries}x: {last}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--alpha", required=True, type=float)
    p.add_argument("--suites", default="refusal32,cyber32,benign32")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--wait-budget", type=int, default=90 * 60)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    alpha = args.alpha

    model = wait_ready(base, args.wait_budget)
    os.makedirs(args.out, exist_ok=True)

    # Warm-up (cold compile/cudagraph paths are already exercised at boot;
    # this just keeps the first scored request off any lazy init). FAIL
    # CLOSED on a thinking-on server: run 1 proved a thinking trace silently
    # invalidates the arm against the reference protocol, and the suite
    # spend comes after this gate.
    warm = one_request(base, model, "Say OK.", 64)
    if "</think>" in warm["completion"]:
        raise RuntimeError(
            "thinking is ON despite enable_thinking=false — aborting before "
            f"the suite spend. warm-up completion: {warm['completion']!r}")
    print(f"warm-up ok (thinking off): {warm['completion'][:80]!r}", flush=True)

    summary = {}
    for name in [s for s in args.suites.split(",") if s]:
        prompts = load_suite(SUITES[name])
        assert len(prompts) == 32, f"{name}: expected 32, got {len(prompts)}"
        fn = os.path.join(args.out, f"{name}-a{alpha}.json")
        if os.path.exists(fn):
            print(f"{name}: {fn} exists, skipping generation", flush=True)
            rec = json.load(open(fn))
        else:
            t0 = time.time()
            items = [None] * len(prompts)
            with cf.ThreadPoolExecutor(args.concurrency) as ex:
                futs = {ex.submit(one_request, base, model, p,
                                  args.max_tokens): i
                        for i, p in enumerate(prompts)}
                for fut in cf.as_completed(futs):
                    i = futs[fut]
                    r = fut.result()
                    items[i] = {"i": i, "prompt": prompts[i], **r}
                    print(f"  [{name}] #{i}: {r['completion_tokens']} tok, "
                          f"fr={r['finish_reason']}, {r['wall_s']}s",
                          flush=True)
            rec = {"suite": name, "alpha": alpha, "model": model,
                   "max_new": args.max_tokens, "decoding": "greedy",
                   "items": items}
            json.dump(rec, open(fn, "w"), indent=1)
            print(f"{name}: 32 completions in {time.time() - t0:.0f}s -> {fn}",
                  flush=True)

        c = collections.Counter()
        flagged = []
        scored = []
        for it in rec["items"]:
            label, why = Q.classify(it["completion"])
            c[label] += 1
            scored.append({"i": it["i"], "label": label, "why": why})
            if Q.flag_argumentative(it["prompt"], it["completion"], label):
                flagged.append(it["i"])
        n = len(rec["items"])
        summary[name] = {"n": n, "delivered": c["COMPLY"],
                         **dict(sorted(c.items()))}
        print(f"== {name} a={alpha}: delivered {c['COMPLY']}/{n}  "
              + "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
              + (f"  [argumentative-flagged REFUSE: {flagged}]"
                 if flagged else ""), flush=True)
        json.dump({"suite": name, "alpha": alpha, "counts": dict(c),
                   "delivered": c["COMPLY"], "n": n,
                   "argumentative_flagged": flagged, "items": scored},
                  open(os.path.join(args.out,
                                    f"scores-{name}-a{alpha}.json"), "w"),
                  indent=1)

    json.dump({"alpha": alpha, "model": model, "max_new": args.max_tokens,
               "suites": summary},
              open(os.path.join(args.out, f"summary-a{alpha}.json"), "w"),
              indent=1)
    print(f"arm a={alpha} done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
