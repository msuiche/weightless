#!/usr/bin/env python3
"""kldiv -- the quality cost of a GLP vector, in nats per token.

Mean per-token KL(stock || steered) over greedy continuations of harmless
prompts: the quality-retention number to publish next to a delivery tally. A
vector that refuses nothing but bends every harmless answer has failed too;
this is the measurement that catches it.

Flow: generate a short greedy reference continuation per prompt with the
STOCK model, then teacher-force prompt+continuation through the stock and the
steered model -- the same hook path glp.py and the serving hotfixes use,
apply_transformers._attach -- and average per-position KL over the
continuation positions and the prompts.

    python3 kldiv.py --model <path> --vector v.gguf --prompts harmless.json
    python3 kldiv.py ... --alpha 0.75 --max-new 128 --batch 4 --json

Teacher forcing scores the exact token ids the stock model generated, so no
re-tokenisation drift enters the comparison, and KL is measured only at the
positions that predict the continuation -- the prompt tokens are shared input,
and KL on them measures the prompt, not the steering.

Idea credit: p-e-w/heretic (AGPL-3.0) -- reimplemented from the published
technique; no heretic code is used in this repository. heretic co-minimizes
refusals and KL divergence from the original model; this script isolates the
KL term for runtime steering, where alpha is the knob.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import torch
except ImportError as e:
    raise ImportError(
        "kldiv needs torch -- it runs the model twice, steered and stock") from e
import captain_vector as cv
import apply_transformers as at


def kl_per_token(logits_p, logits_q):
    """KL(p || q) from raw logits, summed over the vocabulary. Pure torch.

    The last dim is the vocabulary; leading dims are preserved, so a
    (B, T, V) stack returns per-position (B, T) and a single position returns
    a scalar. p is the reference (stock) distribution, q the steered one:
    KL(p || q) grows when the steered model moves mass to where the stock
    model puts little, which is the damage under measurement. Nats per
    position. Non-negative, zero only at p == q, asymmetric by construction.
    """
    logp = torch.log_softmax(logits_p.float(), dim=-1)
    logq = torch.log_softmax(logits_q.float(), dim=-1)
    return (logp.exp() * (logp - logq)).sum(-1)


@torch.no_grad()
def reference_rows(tok, model, prompts, max_new, batch):
    """Greedy stock continuations, one row per prompt.

    Each row is {"ids", "mask", "cut"}: the full sequence (left-padded prompt
    + continuation, trimmed after the first EOS), the matching attention mask,
    and the index where the continuation starts. Token ids, never decoded
    text -- re-encoding a decoded continuation can shift a token, and the
    teacher-forced comparison must score exactly what the stock model wrote.
    """
    rows = []
    for s in range(0, len(prompts), batch):
        chunk = [cv.render(tok, p) for p in prompts[s:s + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        cut = enc["input_ids"].shape[1]
        for i in range(out.shape[0]):
            g = out[i, cut:]
            if tok.eos_token_id is not None:
                hit = (g == tok.eos_token_id).nonzero()
                if len(hit):
                    g = g[:hit[0] + 1]
            ids = torch.cat([out[i, :cut], g])
            mask = torch.cat([enc["attention_mask"][i],
                              torch.ones(len(g), dtype=torch.long,
                                         device=ids.device)])
            rows.append({"ids": ids, "mask": mask, "cut": cut})
    return rows


@torch.no_grad()
def continuation_logits(model, row):
    """Teacher-forced logits at the positions that predict the continuation.

    logits[t] predicts token t+1, so the continuation tokens (indices cut ..
    end-1) are scored by positions cut-1 .. end-2. Returned on CPU fp32: one
    continuation of vocab-sized logits is ~100 MB at V=150k and the steered
    pass needs the stock copy still around.
    """
    out = model(input_ids=row["ids"][None],
                attention_mask=row["mask"][None]).logits
    cut, end = row["cut"], row["ids"].shape[0]
    return out[0, cut - 1:end - 1].float().cpu()


def measure_kl(tok, model, prompts, vector, alpha, max_new=128, batch=4):
    """Mean per-token KL(stock || steered) and the per-prompt rows.

    The reference continuations come from the STOCK model: generating them
    steered would measure how far the steered model drifts from itself, which
    is zero by construction. Per row, the stock forward runs with hooks off
    and the steered forward under attach_glp_steering -- knee_search reuses
    this function as its second objective, so the number it optimizes is the
    number this script prints.
    """
    rows = reference_rows(tok, model, prompts, max_new, batch)
    per = []
    for r in rows:
        n = r["ids"].shape[0] - r["cut"]
        if n <= 0:
            per.append({"prompt": prompts[len(per)], "kl": None, "tokens": 0})
            continue
        lp = continuation_logits(model, r)
        st = at.attach_glp_steering(model, vector, alpha=alpha)
        try:
            lq = continuation_logits(model, r)
        finally:
            st.detach()
        per.append({"prompt": prompts[len(per)],
                    "kl": float(kl_per_token(lp, lq).mean()), "tokens": n})
    scored = [p for p in per if p["kl"] is not None]
    total = sum(p["tokens"] for p in scored)
    mean = (sum(p["kl"] * p["tokens"] for p in scored) / total) if total else None
    return mean, per


def load_model(a):
    """Tokenizer + model, loading the way captain_vector's CLI does.

    The class pick, left padding, dtype and device rules are captain_vector's;
    duplicated here as four lines rather than a shared helper, because the
    shared helper would have to grow a flag for every flag.
    """
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.model)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    if "ConditionalGeneration" in arch or "ImageText" in arch:
        from transformers import AutoModelForImageTextToText as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls
    dev = cv.pick_device(a.device)
    device_map = {"cpu": None, "all": "auto"}.get(dev, dev)
    model = Cls.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                device_map=device_map)
    if dev == "mps":
        model = model.to("mps")
    model.eval()
    print(f"  device {dev}")
    return tok, model


def main():
    p = argparse.ArgumentParser(
        prog="kldiv",
        description="Mean per-token KL(stock || steered) of a GLP vector on "
                    "harmless prompts -- the quality-retention number to "
                    "publish alongside a delivery tally.")
    p.add_argument("--model", required=True, help="local path or HF id")
    p.add_argument("--vector", required=True, metavar="FILE.gguf",
                   help="the GLP vector under test")
    p.add_argument("--prompts", required=True, metavar="FILE.json",
                   help="harmless prompts: JSON list, or {results:[{prompt}]}")
    p.add_argument("--alpha", type=float, default=None,
                   help="steering strength; default is the file's "
                        "glp.alpha_default")
    p.add_argument("--max-new", type=int, default=128, metavar="N",
                   help="reference continuation length in tokens (default 128)")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--json", action="store_true",
                   help="machine-readable output instead of text")
    a = p.parse_args()

    prompts = cv.load_prompts(a.prompts)
    tok, model = load_model(a)
    meta, _, _ = at.load_glp(a.vector)
    alpha_eff = float(a.alpha if a.alpha is not None
                      else meta.get("glp.alpha_default", 1.0))
    mean, per = measure_kl(tok, model, prompts, a.vector, a.alpha,
                           a.max_new, a.batch)

    if a.json:
        print(json.dumps({
            "model": a.model, "vector": a.vector, "alpha": alpha_eff,
            "max_new": a.max_new, "mean_kl": mean,
            "n_scored": sum(1 for r in per if r["kl"] is not None),
            "prompts": per}, indent=2))
        return
    print(f"\n  alpha {alpha_eff:g} (glp.alpha_default "
          f"{meta.get('glp.alpha_default', 1.0):g})"
          + (", --alpha override" if a.alpha is not None else ""))
    print(f"  {'#':>3} {'KL':>9} {'tok':>4}  prompt")
    for i, r in enumerate(per):
        kl = f"{r['kl']:9.4f}" if r["kl"] is not None else f"{'-':>9}"
        text = " ".join(r["prompt"].split())[:60]
        print(f"  {i:>3} {kl} {r['tokens']:>4}  {text}")
    if mean is None:
        print("\n  no prompt produced a continuation; nothing to score")
        return
    print(f"\n  mean per-token KL(stock || steered): {mean:.4f} nats over "
          f"{sum(r['tokens'] for r in per)} tokens, {len(per)} prompts")
    print("  numbers from the same checkpoint are comparable; across "
          "checkpoints they are not -- a KL is model-relative")


if __name__ == "__main__":
    main()
