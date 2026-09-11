#!/usr/bin/env python3
"""End-to-end: steer a small HF model with a GLP control vector.

    python3 apply_hf.py --model Qwen/Qwen3-0.6B --glp v.gguf \
        --prompt "What is the capital of France?" [--alpha 2.0]

Generates once unsteered, once with the vector projected at its listed
layers, and prints both. Experiment-time only -- the serving path is the
patches/ hotfixes, which keep alpha tunable and vectors stackable.
"""
import argparse
import os
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    sys.exit("needs torch + transformers (pip install torch transformers)")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apply_transformers import glp_steered

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="HF repo id or local path")
p.add_argument("--glp", required=True, help="GLP control-vector GGUF")
p.add_argument("--prompt", default="What is the capital of France?")
p.add_argument("--alpha", type=float, default=None,
               help="steering strength; default is the GGUF's glp.alpha_default")
p.add_argument("--max-new", type=int, default=64)
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto").eval()
try:
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                   tokenize=False, add_generation_prompt=True)
except Exception:  # no chat template on this checkpoint
    text = a.prompt
enc = {k: v.to(model.device) for k, v in
       tok(text, return_tensors="pt").items()}


@torch.no_grad()
def gen():
    o = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


print(f"--- baseline ({a.model}) ---")
print(gen())
with glp_steered(model, a.glp, alpha=a.alpha) as st:
    print(f"--- steered ({os.path.basename(a.glp)}, alpha={st.alpha}, "
          f"layers {st.layers[0]}..{st.layers[-1]}) ---")
    print(gen())
