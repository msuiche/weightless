#!/usr/bin/env python3
"""Local post-scorer for the stock-vLLM-0.31 GLM-5.3 lane.

Pulls the eval_arm result files from the glm53-flash volume
(out-glm53-stock031/) into ./out-glm53-stock031/from-volume/ and scores the
suite JSONs with the repo's four-way classifier
(refusal-research/harness/lib.py — the scorer behind the BENCHMARK.md
numbers), so stock-lane numbers are directly comparable to the fork-lane
plugin evals in ../modal/out-glm53-plugin-test/. Also byte-compares the
lora arm's base-side probes against the endpoint arms' probes and prints
the base-vs-smoke-lora divergence count.

Run from the weightless repo root:
    python3 modal/score_glm53_stock.py
"""
import collections
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))    # weightless/modal/
ROOT = os.path.dirname(HERE)                         # weightless/
SPARK = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(SPARK, "refusal-research", "harness"))
import lib as Q  # noqa: E402

OUT = os.path.join(HERE, "out-glm53-stock031")
os.makedirs(OUT, exist_ok=True)

# Pull every eval artifact the in-container runs wrote (per-file: modal
# volume get on a directory path downloads it as one file).
src = os.path.join(OUT, "from-volume")
os.makedirs(src, exist_ok=True)
ls = subprocess.run(["modal", "volume", "ls", "glm53-flash",
                     "out-glm53-stock031"], capture_output=True, text=True,
                    check=True, cwd="/tmp").stdout
names = [l.split("out-glm53-stock031/")[-1].strip() for l in
         ls.splitlines() if ".json" in l or ".log" in l]
for n in names:
    subprocess.run(["modal", "volume", "get", "glm53-flash",
                    f"out-glm53-stock031/{n}", os.path.join(src, n),
                    "--force"], check=True, cwd="/tmp", capture_output=True)
print("pulled:", ", ".join(sorted(names)))

for fn in sorted(os.listdir(src)):
    if not fn.endswith(".json") or fn.startswith(("probe-", "scores-",
                                                  "summary-")):
        continue
    rec = json.load(open(os.path.join(src, fn)))
    if "items" not in rec or not rec["items"]:
        continue
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
    print(f"== {fn} (model={rec['model']}): delivered {c['COMPLY']}/{n}  "
          + "  ".join(f"{k}={v}" for k, v in sorted(c.items()))
          + (f"  [argumentative-flagged REFUSE: {flagged}]" if flagged else ""))
    json.dump({"suite": rec["suite"], "alpha": rec["alpha"],
               "counts": dict(c), "delivered": c["COMPLY"], "n": n,
               "argumentative_flagged": flagged, "items": scored},
              open(os.path.join(OUT, f"scores-{fn}"), "w"), indent=1)

# Byte-compare the lora arm's base-side probes against the endpoint arms'
# probes (same alpha=0 stack; --enable-lora alone should be numerically
# inert — measured 2026-10-06: it is NOT byte-inert, see README).
for a, b in (("probe-a0.json", "probe-lora-base.json"),
             ("probe-stock.json", "probe-lora-base.json")):
    fa = os.path.join(OUT, a)
    fb = os.path.join(src, b)
    if not (os.path.exists(fa) and os.path.exists(fb)):
        continue
    ia = {it["i"]: it["completion"] for it in json.load(open(fa))["items"]}
    ib = {it["i"]: it["completion"] for it in json.load(open(fb))["items"]}
    same = sum(1 for i in ia if ia[i] == ib.get(i))
    print(f"byte-compare {a} vs {b}: {same}/{len(ia)} identical")

# LoRA divergence: base vs smoke-lora on the same probes (must ALL differ —
# the smoke adapter is random-init nonzero by construction).
fa = os.path.join(src, "probe-lora-base.json")
fb = os.path.join(src, "probe-lora-smoke.json")
if os.path.exists(fa) and os.path.exists(fb):
    ia = {it["i"]: it["completion"] for it in json.load(open(fa))["items"]}
    ib = {it["i"]: it["completion"] for it in json.load(open(fb))["items"]}
    diff = sum(1 for i in ia if ia[i] != ib.get(i))
    print(f"lora contrast base-vs-smoke-lora: {diff}/{len(ia)} diverge")
