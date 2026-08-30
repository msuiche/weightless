# ---------------------------------------------------------------------------
# VENDORED — do not edit here; re-vendor from upstream instead.
#
# Source:   tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark,
#           patches/patch_chat_template_thinking.py (main branch)
# Author:   tonyd2wild (deployed and debugged by Knox / Claude)
# Fetched:  2026-08-30
#
# Adds an `enable_thinking` knob to GLM-5.3's stock chat template (the stock
# template has none, so chat_template_kwargs={"enable_thinking": false}
# silently does nothing). Idempotent; backs up the original template.
# ---------------------------------------------------------------------------
#!/usr/bin/env python3
"""
Make GLM-5.3's chat template honour enable_thinking.

The stock template ends with an UNCONDITIONAL open-thinking tag:

    {%- if add_generation_prompt -%}
        <|assistant|>{{- '<think>' -}}
    {%- endif -%}

There is no `enable_thinking` variable anywhere in it, which is why passing
chat_template_kwargs={"enable_thinking": false} silently does nothing -- there is
no knob to turn. This adds one. With enable_thinking=false we emit an already
closed <think></think> block, so the model has nothing to continue and goes
straight to the answer.

Takes effect at the NEXT engine restart; a running vLLM has already read the
template into memory.
"""
import shutil
import sys

T = sys.argv[1] if len(sys.argv) > 1 else \
    "/var/tmp/models/GLM-5.3-Int4-Int8Mix/chat_template.jinja"

OLD = "    <|assistant|>{{- '<think>' -}}"
NEW = ("    <|assistant|>"
       "{%- if enable_thinking is defined and not enable_thinking -%}"
       "{{- '<think></think>' -}}"
       "{%- else -%}"
       "{{- '<think>' -}}"
       "{%- endif -%}")

src = open(T).read()

if "enable_thinking" in src:
    print("already patched - no change")
    sys.exit(0)

if OLD not in src:
    print(f"ERROR: anchor line not found in {T}", file=sys.stderr)
    print("expected exactly:", repr(OLD), file=sys.stderr)
    sys.exit(1)

if src.count(OLD) != 1:
    print(f"ERROR: anchor appears {src.count(OLD)} times, expected 1",
          file=sys.stderr)
    sys.exit(1)

shutil.copy2(T, T + ".bak-prethinking")
open(T, "w").write(src.replace(OLD, NEW))

print(f"patched {T}  (backup: {T}.bak-prethinking)")
print()
print("tail of patched template:")
for line in open(T).read().splitlines()[-4:]:
    print("   ", line)
print()
print("next restart must also pass:")
print("""    --default-chat-template-kwargs '{"enable_thinking": false}'""")
