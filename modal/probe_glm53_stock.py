#!/usr/bin/env python3
"""Probe driver for the stock-vLLM-0.31 GLM-5.3-Flash plugin lane.

Three modes against a deployed cloud_serve_glm53_stock.py endpoint:

  parity  --base-url URL --tag TAG
      Run the fixed probe set (first 8 cyber32 + first 4 benign32 prompts,
      greedy, 400 tokens) and save completions to
      out-glm53-stock031/probe-<tag>.json. Run once per arm.

  compare --a stock --b a0
      Byte-compare two saved probe files. PASS = every completion
      byte-identical (the plugin at alpha=0 must reproduce stock exactly);
      on mismatch print the first differing character offset per prompt.

  lora    --base-url URL
      On the lora arm: send the probe set twice, once as the base model
      (glm53-flash) and once as the smoke LoRA (smoke-lora). Save both;
      report per-prompt divergence (outputs MUST differ with the adapter
      — it is random-init nonzero by construction) and, if
      probe-a0.json exists, byte-compare the base side against it (the
      lora arm serves alpha=0 steering, so its base side should match the
      a0 arm exactly).
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPARK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from eval_glm53_plugin import SUITES, load_suite, one_request, wait_ready \
    # noqa: E402

OUT = os.path.join(HERE, "out-glm53-stock031")

PROMPTS = (load_suite(SUITES["cyber32"])[:8]
           + load_suite(SUITES["benign32"])[:4])
PROBE_MODEL_BASE = "glm53-flash"
PROBE_MODEL_LORA = "smoke-lora"


def run_probes(base_url, model, tag):
    wait_ready(base_url, 90 * 60)
    one_request(base_url, model, "Say OK.", 8)
    print("warm-up ok", flush=True)
    items = [None] * len(PROMPTS)
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(one_request, base_url, model, p, 400): i
                for i, p in enumerate(PROMPTS)}
        for fut in cf.as_completed(futs):
            i = futs[fut]
            r = fut.result()
            items[i] = {"i": i, "prompt": PROMPTS[i], **r}
            print(f"  [{tag}] #{i}: {r['completion_tokens']} tok, "
                  f"fr={r['finish_reason']}, {r['wall_s']}s", flush=True)
    rec = {"tag": tag, "model": model, "max_new": 400, "decoding": "greedy",
           "items": items}
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"probe-{tag}.json")
    json.dump(rec, open(fn, "w"), indent=1)
    print(f"{len(items)} probes in {time.time() - t0:.0f}s -> {fn}",
          flush=True)
    return fn


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def compare(fa, fb):
    ra, rb = json.load(open(fa)), json.load(open(fb))
    ia = {it["i"]: it["completion"] for it in ra["items"]}
    ib = {it["i"]: it["completion"] for it in rb["items"]}
    assert set(ia) == set(ib), "probe sets differ"
    n_same = 0
    for i in sorted(ia):
        if ia[i] == ib[i]:
            n_same += 1
        else:
            d = first_diff(ia[i], ib[i])
            print(f"  #{i}: DIVERGES at char {d} "
                  f"(len {len(ia[i])} vs {len(ib[i])})")
    total = len(ia)
    verdict = "IDENTICAL" if n_same == total else "DIFFERENT"
    print(f"{os.path.basename(fa)} vs {os.path.basename(fb)}: "
          f"{n_same}/{total} byte-identical -> {verdict}", flush=True)
    return n_same == total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["parity", "compare", "lora"])
    p.add_argument("--base-url")
    p.add_argument("--tag")
    p.add_argument("--a", default="stock")
    p.add_argument("--b", default="a0")
    args = p.parse_args()

    if args.mode == "parity":
        assert args.base_url and args.tag
        run_probes(args.base_url.rstrip("/"), PROBE_MODEL_BASE, args.tag)
    elif args.mode == "compare":
        fa = os.path.join(OUT, f"probe-{args.a}.json")
        fb = os.path.join(OUT, f"probe-{args.b}.json")
        same = compare(fa, fb)
        sys.exit(0 if same else 1)
    else:
        assert args.base_url
        base = args.base_url.rstrip("/")
        run_probes(base, PROBE_MODEL_LORA, "lora")
        fl = os.path.join(OUT, "probe-lora.json")
        fa0 = os.path.join(OUT, "probe-a0.json")
        fl_base = os.path.join(OUT, "probe-lora-base.json")
        # base side of the lora arm, for the divergence + a0 cross-check
        run_probes(base, PROBE_MODEL_BASE, "lora-base")
        rl, rb = json.load(open(fl)), json.load(open(fl_base))
        la = {it["i"]: it["completion"] for it in rl["items"]}
        lb = {it["i"]: it["completion"] for it in rb["items"]}
        n_diff = sum(1 for i in la if la[i] != lb[i])
        for i in sorted(la):
            d = first_diff(la[i], lb[i])
            print(f"  #{i}: base-vs-lora "
                  + (f"diverge at char {d}" if d is not None
                     else "IDENTICAL (unexpected)"), flush=True)
        print(f"lora contrast: {n_diff}/{len(la)} prompts diverge "
              f"(expected {len(la)}/{len(la)}: the smoke LoRA is nonzero "
              f"by construction)", flush=True)
        if os.path.exists(fa0):
            print("cross-check vs a0 arm (same alpha=0 stack):", flush=True)
            compare(fa0, fl_base)
        if n_diff != len(la):
            sys.exit(1)


if __name__ == "__main__":
    main()
