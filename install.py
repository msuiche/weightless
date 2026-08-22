#!/usr/bin/env python3
"""Interactive setup wizard for the dspark-deploy endpoint tests.

Walks through: prerequisites (bun/omp), endpoint URL + model selection
(probed live from /v1/models), omp provider install, and optionally runs
the suite. Stdlib only. The non-interactive path is still
`sh tests/install.sh && sh tests/run.sh` with DSPARK_* env overrides.
"""
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OMP_MODELS = os.path.expanduser("~/.omp/agent/models.yml")
DEFAULT_BASE = "http://localhost:8888/v1"
DEFAULT_MODEL = "deepseek-v4-flash-dspark"
PROVIDER = "dspark"

COMPAT_BLOCK = """\
        compat:
          # DeepSeek V4 request shaping: system role, max_tokens, no
          # tool_choice, reasoning-content round-trip. Without the last two,
          # thinking-mode tool conversations 400. Drop this block for
          # non-DeepSeek models.
          supportsDeveloperRole: false
          supportsReasoningEffort: true
          maxTokensField: max_tokens
          supportsToolChoice: false
          requiresReasoningContentForToolCalls: true
          requiresAssistantContentForToolCalls: true
"""


def ask(prompt, default):
    s = input(f"{prompt} [{default}]: ").strip()
    return s or default


def yn(prompt, default=True):
    hint = "Y/n" if default else "y/N"
    s = input(f"{prompt} [{hint}]: ").strip().lower()
    return (s.startswith("y") if s else default)


def probe_models(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=10) as r:
            data = json.load(r)
        return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        print(f"  warning: could not probe {base}/models ({e})")
        return None


def render_provider(base, model, deepseek_compat):
    compat = COMPAT_BLOCK if deepseek_compat else ""
    return f"""{PROVIDER}:
    baseUrl: {base}
    auth: none
    api: openai-completions
    models:
      - id: {model}
        name: {model} ({PROVIDER})
        reasoning: true
        input: [text]
        contextWindow: 1048576
        maxTokens: 32768
{compat}"""


def install_provider(base, model, deepseek_compat):
    block = render_provider(base, model, deepseek_compat)
    os.makedirs(os.path.dirname(OMP_MODELS), exist_ok=True)
    if os.path.exists(OMP_MODELS):
        with open(OMP_MODELS) as f:
            existing = f.read()
        if f"\n  {PROVIDER}:" in "\n" + existing or existing.startswith(f"{PROVIDER}:"):
            print(f"  provider '{PROVIDER}' already present in {OMP_MODELS} — leaving it")
            return
        backup = OMP_MODELS + ".bak"
        shutil.copy(OMP_MODELS, backup)
        with open(OMP_MODELS, "a") as f:
            if "providers:" not in existing:
                f.write("\nproviders:\n")
            f.write("\n  " + block.replace("\n", "\n  ").rstrip() + "\n")
        print(f"  merged provider '{PROVIDER}' into {OMP_MODELS} (backup: {backup})")
    else:
        with open(OMP_MODELS, "w") as f:
            f.write("providers:\n  " + block.replace("\n", "\n  ").rstrip() + "\n")
        print(f"  wrote {OMP_MODELS}")


def main():
    print("== dspark-deploy tests: interactive setup ==\n")

    print("1. Prerequisites")
    omp = shutil.which("omp")
    bun = shutil.which("bun")
    print(f"   bun: {bun or 'MISSING — curl -fsSL https://bun.com/install | bash'}")
    print(f"   omp: {omp or 'MISSING — curl -fsSL https://omp.sh/install | sh'}")
    if not omp:
        print("\nInstall omp first, then re-run this wizard.")
        sys.exit(1)

    print("\n2. Endpoint")
    base = ask("   Base URL of the OpenAI-compatible server", DEFAULT_BASE)
    ids = probe_models(base)
    if ids:
        print(f"   served models: {', '.join(ids)}")
        default_model = DEFAULT_MODEL if DEFAULT_MODEL in ids else ids[0]
    else:
        default_model = DEFAULT_MODEL
        if not yn("   Continue anyway (endpoint unreachable)?", False):
            sys.exit(1)
    model = ask("   Model id", default_model)
    if ids and model not in ids:
        print(f"   note: '{model}' is not in the served list — tests will fail until it is")

    deepseek_compat = yn("\n3. DeepSeek V4 compat block (tool_call/reasoning shaping)?", "deepseek" in model.lower())

    print("\n4. omp provider config")
    install_provider(base, model, deepseek_compat)

    if yn("\n5. Run the test suite now?"):
        env = dict(os.environ, DSPARK_BASE_URL=base, DSPARK_MODEL=model,
                   DSPARK_OMP_MODEL=f"{PROVIDER}/{model}")
        sys.exit(subprocess.call(["sh", os.path.join(HERE, "tests", "run.sh")], env=env))
    print(f"\nDone. Later: sh {os.path.join(HERE, 'tests', 'run.sh')}")


if __name__ == "__main__":
    main()
