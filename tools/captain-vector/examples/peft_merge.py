#!/usr/bin/env python3
"""Load a captain-vector baked adapter with peft, compare, merge, save.

`captain-vector bake` output is a rank-1 LoRA whose lora_A CARRIES the base
weights W (lora_A = -alpha * d^T W, lora_B = d). Two gotchas follow directly
from that, and both are silent if you get them wrong:

  GOTCHA 1 -- DO NOT RESCALE THE ADAPTER. alpha is already baked into
  lora_A. The config ships r=1, lora_alpha=1 so peft's scaling factor
  (lora_alpha / r) is exactly 1.0. Overriding lora_alpha at load time,
  enabling use_rslora, or applying any adapter scale multiplies alpha a
  second time. The assert below checks the loaded config instead of
  trusting it.

  GOTCHA 2 -- PINNED BASE REVISION ONLY. lora_A is computed from W of ONE
  specific checkpoint revision (adapter_config.json["revision"] and
  bake-report.json record it). Against any other revision the edit is off
  by d^T(W_loaded - W_pinned) and NOTHING in peft flags it -- the adapter
  loads fine and produces plausible garbage. Always pass --revision with
  the exact sha.

Reference adapters (gated; hf CLI must be authed):
  msuiche/Qwen3.8-27B-hedging-GLP-63-L1-63-a0.5
  msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49

Usage:
  python3 peft_merge.py \
      --base Qwen/Qwen3.8-27B --revision <pinned sha> \
      --adapter msuiche/Qwen3.8-27B-hedging-GLP-63-L1-63-a0.5 \
      --merged-out ./merged

Deps: torch, transformers, peft (adapter dir itself is plain safetensors).
"""
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_pinned_revision(adapter):
    """The revision the adapter was baked against, from its own metadata."""
    cfg_p = os.path.join(adapter, "adapter_config.json")
    if os.path.isdir(adapter) and os.path.exists(cfg_p):
        return json.load(open(cfg_p)).get("revision")
    return None  # HF repo: pass --revision explicitly, from the model card


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", required=True, help="base model repo id or path")
    p.add_argument("--revision", required=True,
                   help="EXACT base commit sha the adapter was baked against "
                        "(see bake-report.json / adapter_config.json)")
    p.add_argument("--adapter", required=True,
                   help="baked adapter directory or HF repo id")
    p.add_argument("--merged-out", default=None,
                   help="write the merged checkpoint (weights + tokenizer) here")
    p.add_argument("--prompt", default="The capital of France is",
                   help="prompt for the with/without comparison")
    p.add_argument("--max-new-tokens", type=int, default=32)
    a = p.parse_args()

    pinned = read_pinned_revision(a.adapter)
    if pinned and pinned != a.revision:
        raise SystemExit(
            f"adapter was baked against {pinned}, you passed {a.revision} -- "
            "lora_A carries W of the pinned checkpoint; refusing (GOTCHA 2).")

    tok = AutoTokenizer.from_pretrained(a.base, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, revision=a.revision, torch_dtype="auto", device_map="auto")
    model.eval()

    peft_model = PeftModel.from_pretrained(model, a.adapter)
    peft_model.eval()

    # GOTCHA 1 guard: peft scaling must be 1.0 -- alpha already lives in
    # lora_A. If this trips, someone edited the config; do not "fix" the
    # numbers, regenerate the adapter with bake instead.
    cfg = peft_model.peft_config["default"]
    scaling = cfg.lora_alpha / cfg.r
    assert scaling == 1.0 and not cfg.use_rslora, (
        f"adapter scaling is {scaling} (r={cfg.r}, lora_alpha={cfg.lora_alpha}"
        f", use_rslora={cfg.use_rslora}); baked adapters must load at 1.0")

    inputs = tok(a.prompt, return_tensors="pt").to(peft_model.device)

    @torch.no_grad()
    def gen(m):
        out = m.generate(**inputs, max_new_tokens=a.max_new_tokens,
                         do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    # Same weights, one forward with the delta and one without. For a
    # baked control vector the diff is the steering effect, by definition
    # of lora_A.
    with peft_model.disable_adapter():
        off = gen(peft_model)
    on = gen(peft_model)
    print(f"--- without adapter ---\n{off}\n--- with adapter ---\n{on}")

    # merge folds the delta into the base weights in place: W <- W + B@A
    # (scaling 1.0). The merged checkpoint no longer needs peft.
    if a.merged_out:
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(a.merged_out)
        tok.save_pretrained(a.merged_out)
        print(f"merged checkpoint written to {a.merged_out}")


if __name__ == "__main__":
    main()
