#!/usr/bin/env python3
"""Full-chain setup for dspark-deploy.

Pick a lane, fill in the site values (hosts, user), generate the real
gitignored env file from the shipped example, validate the steering patch,
deploy to the node(s) over ssh (confirm-gated), then install the omp
provider and run the endpoint smoke tests. If the endpoint is down, the
diagnose chain walks the layers (DNS → TCP → HTTP) and can check/boot the
stack over ssh. Stdlib only: a light curses TUI with colors on a terminal,
ANSI-colored prompts otherwise. Non-interactive alternative:
`sh tests/install.sh && sh tests/run.sh` with DSPARK_* env overrides.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import urllib.parse
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
         steer_modes=None,
         port=8888),
    dict(name="Qwen TP=1 serving — single DGX Spark",
         example="recipe/qwen/.env.qwen.example",
         target="recipe/qwen/.env.qwen",
         steer_key="QWEN_STEER_PATH",
         structure_test="scripts/test-qwen-steering-structure.py",
         vector_repo="msuiche/Qwen3.8-27B-abliterated-cvec",
         steer_modes=[
             ("gguf", "gguf — hotfix-patched vLLM, fail-closed (default, validated)"),
             ("lora", "lora — stock vLLM --enable-lora, no patch (validated)")],
         port=8078),
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


def read_lane_env():
    """Best-effort site values from an existing (gitignored) lane env."""
    vals = {}
    for lane in LANES:
        path = os.path.join(HERE, lane["target"])
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                m = re.match(r"^(MASTER_ADDR|WORKER_HOST|MODELS)=(\S+)", line)
                if m:
                    vals.setdefault("host", m.group(2))
                if m and m.group(1) == "MASTER_ADDR":
                    vals.setdefault("head-ip", m.group(2))
    return vals


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


def boot_command(lane_idx, values, qwen_host=None):
    """Just the boot step of deploy_commands, for the diagnose flow."""
    return deploy_commands(lane_idx, values, qwen_host)[-1]


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
    C = {"head": "\033[1;36m", "ok": "\033[32m", "err": "\033[31m",
         "warn": "\033[33m", "dim": "\033[2m", "reset": "\033[0m"}

    def __init__(self):
        self.color = sys.stdout.isatty()

    def _c(self, kind, msg):
        if self.color:
            return f"{self.C[kind]}{msg}{self.C['reset']}"
        return msg

    def info(self, msg):
        print(msg)

    def header(self, msg):
        print(self._c("head", msg))

    def ok(self, msg):
        print(self._c("ok", msg))

    def warn(self, msg):
        print(self._c("warn", msg))

    def err(self, msg):
        print(self._c("err", msg))

    def _eof(self):
        # EOF on piped stdin must not silently take defaults — a menu that
        # defaults to 0 can loop forever (diagnose → re-prompt → EOF → ...).
        print("\n(stdin closed — aborting)", file=sys.stderr)
        raise SystemExit(1)

    def text(self, prompt, default=""):
        try:
            s = input(f"{prompt} [{default}]: ").strip() if default \
                else input(f"{prompt}: ").strip()
        except EOFError:
            self._eof()
        return s or default

    def confirm(self, prompt, default=True):
        hint = "Y/n" if default else "y/N"
        try:
            s = input(f"{prompt} [{hint}]: ").strip().lower()
        except EOFError:
            self._eof()
        return s.startswith("y") if s else default

    def menu(self, title, items):
        print(self._c("head", title))
        for i, it in enumerate(items):
            label = it[1] if isinstance(it, tuple) else it
            print(f"  {self._c('head', str(i + 1) + ')')} {label}")
        while True:
            try:
                s = input("> ").strip()
            except EOFError:
                self._eof()
            if s.isdigit() and 1 <= int(s) <= len(items):
                return int(s) - 1
            print("  enter a number from the list")


class TuiIO:
    def __init__(self, stdscr):
        self.s = stdscr
        self.row = 0
        self.color = False
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            for i, fg in enumerate((curses.COLOR_CYAN, curses.COLOR_GREEN,
                                    curses.COLOR_RED, curses.COLOR_YELLOW), 1):
                curses.init_pair(i, fg, -1)
            self.color = True

    def _attr(self, kind):
        if not self.color:
            return {"head": curses.A_BOLD}.get(kind, curses.A_NORMAL)
        return {"head": curses.color_pair(1) | curses.A_BOLD,
                "ok": curses.color_pair(2),
                "err": curses.color_pair(3),
                "warn": curses.color_pair(4),
                "dim": curses.A_DIM}.get(kind, curses.A_NORMAL)

    def _next(self, n=1):
        self.row += n
        return self.row - n

    def _put(self, row, col, msg, kind=None):
        try:
            self.s.addstr(row, col, str(msg)[:78], self._attr(kind) if kind else curses.A_NORMAL)
        except curses.error:
            pass
        self.s.refresh()

    def info(self, msg):
        self._put(self._next(), 0, msg)

    def header(self, msg):
        self._put(self._next(), 0, msg, "head")

    def ok(self, msg):
        self._put(self._next(), 0, msg, "ok")

    def warn(self, msg):
        self._put(self._next(), 0, msg, "warn")

    def err(self, msg):
        self._put(self._next(), 0, msg, "err")

    def text(self, prompt, default=""):
        r = self._next()
        buf, fresh = default, True
        while True:
            self.s.addstr(r, 0, " " * 79)
            self._put(r, 0, prompt, "head")
            shown = buf[-(76 - len(prompt)):]
            try:
                self.s.addstr(r, len(prompt), shown)
            except curses.error:
                pass
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
        self.s.addstr(r, 0, f"{prompt} ", self._attr("head"))
        self.s.addstr(f"[{hint}] ", self._attr("dim"))
        self.s.refresh()
        ch = self.s.get_wch()
        return ch.lower().startswith("y") if isinstance(ch, str) and ch.strip() else default

    def menu(self, title, items):
        r = self._next(len(items) + 1)
        self.s.addstr(r, 0, title, self._attr("head"))
        sel = 0
        while True:
            for i, it in enumerate(items):
                label = it[1] if isinstance(it, tuple) else it
                if i == sel:
                    attr = self._attr("head") | curses.A_REVERSE
                else:
                    attr = curses.A_NORMAL
                try:
                    self.s.addstr(r + 1 + i, 2, ("› " if i == sel else "  ") + label[:72], attr)
                except curses.error:
                    pass
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
    io.header(lane["name"])
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
    io.ok(f"wrote {target}")

    # 4. validate the steering patch + point at the vector
    if steering:
        rc = subprocess.call([sys.executable, os.path.join(HERE, lane["structure_test"])],
                             stdout=subprocess.DEVNULL)
        if rc == 0:
            io.ok("steering patch structural test: PASS")
        elif rc == 2:
            io.ok("steering patch structural test: PASS (apply tier skipped — no reference model)")
        else:
            io.err("steering patch structural test: FAIL")
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
                io.warn("skipped")
                continue
            rc = subprocess.call(argv)
            if rc != 0:
                io.err(f"FAILED ({rc}) — fix and re-run; aborting deploy")
                return rc
    else:
        io.info("deploy skipped — commands are in the lane README when you're ready")

    # 6. omp provider + endpoint tests
    return tests_chain(io)


def diagnose_chain(io, base=None):
    """Layered failure isolation for an endpoint that won't answer, with an
    optional remote check/boot over ssh."""
    base = base or io.text("Base URL to diagnose: ", DEFAULT_BASE)
    u = urllib.parse.urlparse(base if "://" in base else "http://" + base)
    host = u.hostname or base
    port = u.port or (443 if u.scheme == "https" else 80)
    io.header(f"diagnosing {host}:{port}")
    io.info("─" * 60)

    # layer 1: DNS
    try:
        ip = socket.gethostbyname(host)
        io.ok(f"DNS: {host} → {ip}")
    except socket.gaierror as e:
        io.err(f"DNS: cannot resolve {host} ({e})")
        io.info("fix the name first — /etc/hosts, mDNS (is the node on?), or use an IP")
        return 1

    # layer 2: TCP
    try:
        with socket.create_connection((host, port), timeout=5):
            io.ok(f"TCP: {host}:{port} accepts connections")
    except OSError as e:
        io.err(f"TCP: {host}:{port} unreachable ({e})")
        io.info("the node is up but nothing serves the port — the stack is down")
        remote_diagnose(io, host)
        return 1

    # layer 3: HTTP /models
    ids, err = probe_models(f"{u.scheme or 'http'}://{host}:{port}/v1")
    if ids is None:
        io.err(f"HTTP: /v1/models failed ({err})")
        io.info("something listens but it is not a healthy OpenAI server — check container logs")
        remote_diagnose(io, host)
        return 1
    io.ok(f"HTTP: /v1/models answers — serving: {', '.join(ids)}")
    return 0


def remote_diagnose(io, host):
    """ssh to the serving node: container status, GPU, offer to boot."""
    saved = read_lane_env()
    default_target = f"{os.environ.get('USER', '')}@{saved.get('host', host)}"
    if not io.confirm("Check the node over ssh (docker ps, GPU)?", True):
        return
    target = io.text("ssh target (user@host): ", default_target)
    probe = ("docker ps -a --format '{{.Names}} {{.Status}}' "
             "| grep -i -E 'deepseek|qwen|vllm' || echo '(no serving container)'; "
             "nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu "
             "--format=csv,noheader 2>/dev/null || echo '(no GPU?)'")
    io.info(f"$ ssh {target} '<container + GPU status>'")
    rc = subprocess.call(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
                          target, probe])
    if rc != 0:
        io.err(f"ssh failed ({rc}) — node off, or keys/password not set up")
        return
    if io.confirm("Boot the stack on that node?", False):
        lane_idx = io.menu("Which lane runs there?", [l["name"] for l in LANES])
        values = {"user": target.split("@")[0], "head-ip": host}
        desc, argv = boot_command(lane_idx, values, qwen_host=host)
        io.info(f"$ {' '.join(argv)}")
        if io.confirm(f"run: {desc}?", True):
            rc = subprocess.call(argv)
            if rc == 0:
                io.ok("boot issued — give it a few minutes to load the model")
            else:
                io.err(f"boot FAILED ({rc})")


def tests_chain(io):
    bun, omp = prereqs()
    if not omp:
        io.err("omp not found — install: curl -fsSL https://omp.sh/install | sh")
        return 1
    base = io.text("Base URL of the OpenAI-compatible server: ", DEFAULT_BASE)
    io.info(f"probing {base}/models ...")
    ids, err = probe_models(base)
    while ids is None:
        io.err(f"probe failed: {err}")
        choice = io.menu("endpoint not answering:", [
            "Diagnose the connection (DNS → TCP → HTTP → ssh)",
            "Enter a different URL",
            "Continue anyway (provider will fail until the server is up)",
        ])
        if choice == 0:
            diagnose_chain(io, base)
            io.info(f"re-probing {base}/models ...")
            ids, err = probe_models(base)
            if ids is not None:
                io.ok("endpoint is up now")
        elif choice == 1:
            base = io.text("Base URL: ", base)
            io.info(f"probing {base}/models ...")
            ids, err = probe_models(base)
        else:
            model = io.text("Model id: ", DEFAULT_MODEL)
            break
    if ids is not None:
        model = ids[io.menu("Model to test:", ids)] if len(ids) > 1 else ids[0]
        io.ok(f"model: {model}")
    compat = io.confirm(f"DeepSeek V4 compat block for '{model}'?",
                        "deepseek" in model.lower())
    io.ok(install_provider(base, model, compat))
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
    io.info(f"omp:  {omp or 'not found (only needed for tests)'}")
    io.info("")
    choice = io.menu("What to set up:", [
        "DSV4 TP=2 serving — full chain (env → steering → deploy → omp/tests)",
        "Qwen TP=1 serving — full chain (env → steering → deploy → omp/tests)",
        "Endpoint tests — omp provider + smoke suite",
        "Diagnose endpoint — layered checks + remote container status",
    ])
    if choice in (0, 1):
        return lane_chain(io, choice)
    if choice == 2:
        return tests_chain(io)
    return diagnose_chain(io)


def _tui_main(stdscr):
    io = TuiIO(stdscr)
    io.header("  dspark-deploy setup")
    io._put(1, 0, "  lean abliteration steering, served", "dim")
    io.info("  " + "─" * 56)
    io.row = 4
    rc = run(io)
    if not curses.isendwin():
        io.info("")
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
    io.header("== dspark-deploy setup ==")
    io.info("")
    return run(io)


if __name__ == "__main__":
    sys.exit(main())
