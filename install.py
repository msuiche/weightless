#!/usr/bin/env python3
"""Full-chain setup for dspark-deploy.

Pick a lane, fill in the site values (hosts, user), generate the real
gitignored env file from the shipped example, validate the steering patch,
deploy to the node(s) over ssh (confirm-gated), then install the omp
provider and run the endpoint smoke tests. Stdlib only: a light curses TUI
on a terminal, plain prompts otherwise. Non-interactive alternative:
`sh tests/install.sh && sh tests/run.sh` with DSPARK_* env overrides.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

try:
    import curses
except ImportError:  # non-POSIX / minimal builds — CLI fallback only
    curses = None

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

# lane -> (example env, target env, steering env key, structure test, vector repo)
LANES = [
    dict(name="DSV4 TP=2 serving — 2x DGX Spark, Anemll recipe",
         example="recipe/anemll/.env.dspark.example",
         target="recipe/anemll/.env.dspark",
         steer_key="DSPARK_STEER_PATH",
         structure_test="scripts/test-dsv4-hotfix-structure.py",
         vector_repo="msuiche/DeepSeek-V4-Flash-0731-cyber-abliterated-cvec",
         steer_modes=None),
    dict(name="Qwen TP=1 serving — single DGX Spark",
         example="recipe/qwen/.env.qwen.example",
         target="recipe/qwen/.env.qwen",
         steer_key="QWEN_STEER_PATH",
         structure_test="scripts/test-qwen-steering-structure.py",
         vector_repo="msuiche/Qwen3.8-27B-abliterated-cvec",
         steer_modes=[
             ("gguf", "gguf — hotfix-patched vLLM, fail-closed (default, validated)"),
             ("lora", "lora — stock vLLM --enable-lora, no patch (validated)")]),
]
PLACEHOLDER_HINTS = {
    "head-ip": ("Head node IP or hostname", ""),
    "worker-ip": ("Worker node IP or hostname", ""),
    "user": ("Remote username on the node(s)", os.environ.get("USER", "")),
}


# ---------------------------------------------------------------- core logic

def placeholders(example):
    """Ordered unique <...> placeholders in non-comment lines."""
    out = []
    with open(example) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            for m in re.finditer(r"<([^>]+)>", line):
                if m.group(1) not in out:
                    out.append(m.group(1))
    return out


def render_env(example, values, steer_mode=None, steering=True, steer_key=None):
    with open(example) as f:
        text = f.read()
    for key, val in values.items():
        text = text.replace(f"<{key}>", val)
    if steer_mode:
        text = re.sub(r"^STEER_MODE=.*$", f"STEER_MODE={steer_mode}", text, flags=re.M)
    if not steering and steer_key:
        text = re.sub(rf"^{steer_key}=.*$", f"{steer_key}=", text, flags=re.M)
    return text


def deploy_commands(lane_idx, values, qwen_host=None):
    """(description, argv) pairs to push the lane to its node(s) and boot it.
    DSV4 targets the MiaAI clone dir on the head (its start script syncs the
    worker itself); Qwen targets a repo-shaped dir on the single node."""
    user = values.get("user", os.environ.get("USER", ""))
    if lane_idx == 0:
        head = f"{user}@{values.get('head-ip', '<head-ip>')}"
        remote = "dspark-miaai"
        r = os.path.join(HERE, "recipe", "anemll")
        return [
            (f"prepare {head}:{remote}/",
             ["ssh", head, f"mkdir -p {remote}/patches"]),
            ("sync env + compose + start script",
             ["scp", os.path.join(r, ".env.dspark"),
              os.path.join(r, "docker-compose.dspark.yml"),
              os.path.join(r, "start-deepseek-v4-flash-dspark.sh"),
              f"{head}:{remote}/"]),
            ("sync steering hotfix",
             ["scp", os.path.join(HERE, "patches", "hotfix-dsv4-steering-projective.py"),
              f"{head}:{remote}/patches/"]),
            ("boot the stack (start script syncs the worker itself)",
             ["ssh", head, f"cd {remote} && bash start-deepseek-v4-flash-dspark.sh"]),
        ]
    host = f"{user}@{qwen_host or '<node-host>'}"
    remote = "dspark-deploy"
    return [
        (f"prepare {host}:{remote}/",
         ["ssh", host, f"mkdir -p {remote}/recipe/qwen {remote}/patches"]),
        ("sync serve script + env",
         ["scp", os.path.join(HERE, "recipe", "qwen", "serve-qwen38.sh"),
          os.path.join(HERE, "recipe", "qwen", ".env.qwen"),
          f"{host}:{remote}/recipe/qwen/"]),
        ("sync steering hotfix",
         ["scp", os.path.join(HERE, "patches", "hotfix-qwen38-steering-projective.py"),
          f"{host}:{remote}/patches/"]),
        ("boot the container",
         ["ssh", host, f"bash {remote}/recipe/qwen/serve-qwen38.sh"]),
    ]


def probe_models(base):
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=10) as r:
            data = json.load(r)
        return [m["id"] for m in data.get("data", [])], None
    except Exception as e:
        return None, str(e)


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
    """Merge the provider into ~/.omp/agent/models.yml. Returns status line."""
    block = render_provider(base, model, deepseek_compat)
    os.makedirs(os.path.dirname(OMP_MODELS), exist_ok=True)
    if os.path.exists(OMP_MODELS):
        with open(OMP_MODELS) as f:
            existing = f.read()
        if f"\n  {PROVIDER}:" in "\n" + existing or existing.startswith(f"{PROVIDER}:"):
            return f"provider '{PROVIDER}' already present in {OMP_MODELS} — left as-is"
        backup = OMP_MODELS + ".bak"
        shutil.copy(OMP_MODELS, backup)
        with open(OMP_MODELS, "a") as f:
            if "providers:" not in existing:
                f.write("\nproviders:\n")
            f.write("\n  " + block.replace("\n", "\n  ").rstrip() + "\n")
        return f"merged provider '{PROVIDER}' into {OMP_MODELS} (backup: {backup})"
    with open(OMP_MODELS, "w") as f:
        f.write("providers:\n  " + block.replace("\n", "\n  ").rstrip() + "\n")
    return f"wrote {OMP_MODELS}"


def prereqs():
    return shutil.which("bun"), shutil.which("omp")


def run_suite(base, model):
    env = dict(os.environ, DSPARK_BASE_URL=base, DSPARK_MODEL=model,
               DSPARK_OMP_MODEL=f"{PROVIDER}/{model}")
    return subprocess.call(["sh", os.path.join(HERE, "tests", "run.sh")], env=env)


# ---------------------------------------------------------------- IO adapters

class CliIO:
    def info(self, msg):
        print(msg)

    def text(self, prompt, default=""):
        try:
            s = input(f"{prompt} [{default}]: ").strip() if default \
                else input(f"{prompt}: ").strip()
        except EOFError:
            s = ""
        return s or default

    def confirm(self, prompt, default=True):
        hint = "Y/n" if default else "y/N"
        try:
            s = input(f"{prompt} [{hint}]: ").strip().lower()
        except EOFError:
            return default
        return s.startswith("y") if s else default

    def menu(self, title, items):
        print(title)
        for i, it in enumerate(items):
            label = it[1] if isinstance(it, tuple) else it
            print(f"  {i + 1}) {label}")
        while True:
            try:
                s = input("> ").strip()
            except EOFError:
                return 0
            if s.isdigit() and 1 <= int(s) <= len(items):
                return int(s) - 1
            print("  enter a number from the list")


class TuiIO:
    def __init__(self, stdscr):
        self.s = stdscr
        self.row = 0

    def _next(self, n=1):
        self.row += n
        return self.row - n

    def _clear(self):
        self.s.clear()
        self.row = 0

    def info(self, msg):
        r = self._next()
        try:
            self.s.addstr(r, 0, str(msg)[:78])
        except curses.error:
            pass
        self.s.refresh()

    def text(self, prompt, default=""):
        r = self._next()
        buf, fresh = default, True
        while True:
            self.s.addstr(r, 0, " " * 79)
            shown = buf[-(76 - len(prompt)):]
            self.s.addstr(r, 0, prompt + shown)
            self.s.refresh()
            ch = self.s.get_wch()
            if ch in ("\n", "\r"):
                return buf.strip() or default
            if ch == "\x1b":
                return default
            if ch in ("\x7f", "\x08") or ch == curses.KEY_BACKSPACE:
                buf = "" if fresh else buf[:-1]
                fresh = False
            elif isinstance(ch, str) and ch.isprintable():
                buf = ("" if fresh else buf) + ch
                fresh = False

    def confirm(self, prompt, default=True):
        r = self._next()
        hint = "Y/n" if default else "y/N"
        self.s.addstr(r, 0, f"{prompt} [{hint}] ")
        self.s.refresh()
        ch = self.s.get_wch()
        return ch.lower().startswith("y") if isinstance(ch, str) and ch.strip() else default

    def menu(self, title, items):
        r = self._next(len(items) + 1)
        self.s.addstr(r, 0, title)
        sel = 0
        while True:
            for i, it in enumerate(items):
                label = it[1] if isinstance(it, tuple) else it
                attr = curses.A_REVERSE if i == sel else curses.A_NORMAL
                self.s.addstr(r + 1 + i, 2, ("› " if i == sel else "  ") + label[:72], attr)
            self.s.refresh()
            ch = self.s.getch()
            if ch in (curses.KEY_UP, ord("k")):
                sel = (sel - 1) % len(items)
            elif ch in (curses.KEY_DOWN, ord("j")):
                sel = (sel + 1) % len(items)
            elif ch in (10, 13):
                return sel


# ---------------------------------------------------------------- chains

def lane_chain(io, lane_idx):
    lane = LANES[lane_idx]
    example = os.path.join(HERE, lane["example"])
    target = os.path.join(HERE, lane["target"])
    io.info(lane["name"])
    io.info("─" * 60)

    # 1. site values
    values = {}
    for p in placeholders(example):
        prompt, default = PLACEHOLDER_HINTS.get(p, (p.replace("-", " "), ""))
        values[p] = io.text(f"{prompt} ({p}): ", default)

    # 2. steering (default on)
    steering = io.confirm("Enable refusal steering (control-vector patch)?", True)
    steer_mode = None
    if steering and lane["steer_modes"]:
        labels = [d for _, d in lane["steer_modes"]]
        sel = io.menu("Steering mode:", labels)
        steer_mode = lane["steer_modes"][sel][0]

    # 3. write env
    text = render_env(example, values, steer_mode=steer_mode,
                      steering=steering, steer_key=lane["steer_key"])
    if os.path.exists(target) and not io.confirm(f"{target} exists — overwrite?", False):
        return 1
    with open(target, "w") as f:
        f.write(text)
    io.info(f"wrote {target}")

    # 4. validate the steering patch + point at the vector
    if steering:
        rc = subprocess.call([sys.executable, os.path.join(HERE, lane["structure_test"])],
                             stdout=subprocess.DEVNULL)
        verdict = {0: "PASS", 2: "PASS (apply tier skipped — no reference model.py)"}.get(rc, "FAIL")
        io.info(f"steering patch structural test: {verdict}")
        io.info(f"vector (gated, needs HF token): "
                f"huggingface-cli download {lane['vector_repo']} --include '*.gguf'")

    # 5. deploy (confirm-gated remote actions)
    if io.confirm("Deploy to the node(s) over ssh now?", False):
        qwen_host = None
        if lane_idx == 1:
            qwen_host = io.text("Node IP or hostname: ", values.get("head-ip", ""))
        for desc, argv in deploy_commands(lane_idx, values, qwen_host):
            io.info(f"$ {' '.join(argv)}")
            if not io.confirm(f"run: {desc}?", True):
                io.info("skipped")
                continue
            rc = subprocess.call(argv)
            if rc != 0:
                io.info(f"FAILED ({rc}) — fix and re-run; aborting deploy")
                return rc
    else:
        io.info("deploy skipped — commands are in the lane README when you're ready")

    # 6. omp provider + endpoint tests
    return tests_chain(io)


def tests_chain(io):
    bun, omp = prereqs()
    if not omp:
        io.info("omp not found — install: curl -fsSL https://omp.sh/install | sh")
        return 1
    base = io.text("Base URL of the OpenAI-compatible server: ", DEFAULT_BASE)
    io.info(f"probing {base}/models ...")
    ids, err = probe_models(base)
    if ids is None:
        io.info(f"probe failed: {err}")
        if not io.confirm("Continue anyway?", False):
            return 1
        model = io.text("Model id: ", DEFAULT_MODEL)
    else:
        model = ids[io.menu("Model to test:", ids)] if len(ids) > 1 else ids[0]
        io.info(f"model: {model}")
    compat = io.confirm(f"DeepSeek V4 compat block for '{model}'?",
                        "deepseek" in model.lower())
    io.info(install_provider(base, model, compat))
    if io.confirm("Run the test suite now?", True):
        if isinstance(io, TuiIO):
            curses.endwin()
        return run_suite(base, model)
    io.info(f"done — later: sh {os.path.join(HERE, 'tests', 'run.sh')}")
    return 0


# ---------------------------------------------------------------- entry

def run(io):
    bun, omp = prereqs()
    io.info(f"bun: {bun or 'not found (only needed for omp/tests)'}")
    io.info(f"omp: {omp or 'not found (only needed for tests)'}")
    choice = io.menu("What to set up:", [
        "DSV4 TP=2 serving — full chain (env → steering → deploy → omp/tests)",
        "Qwen TP=1 serving — full chain (env → steering → deploy → omp/tests)",
        "Endpoint tests only — omp provider + smoke suite",
    ])
    if choice in (0, 1):
        return lane_chain(io, choice)
    return tests_chain(io)


def _tui_main(stdscr):
    io = TuiIO(stdscr)
    stdscr.addstr(0, 0, "dspark-deploy setup", curses.A_BOLD)
    io.row = 2
    rc = run(io)
    if not curses.isendwin():
        io.info("press any key to exit")
        stdscr.getch()
    return rc


def main():
    if sys.stdout.isatty() and sys.stdin.isatty() and curses is not None:
        try:
            return curses.wrapper(_tui_main)
        except Exception as e:
            print(f"(TUI failed: {e} — falling back to prompts)", file=sys.stderr)
    io = CliIO()
    io.info("== dspark-deploy setup ==")
    return run(io)


if __name__ == "__main__":
    sys.exit(main())
