#!/usr/bin/env python3
"""Full-chain setup for weightless.

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

# Launch splash: a feather (weightless) with omp's pink → cyan gradient,
# applied per column (horizontal), like the omp π logo.
LOGO = [
    "      ▄▄▄    ",
    "    ▄██████  ",
    "   ▄███▀████ ",
    "  ▄███▀ ▄████",
    "  ███▀ ▄███▀ ",
    " ▄██▀ ▄███▀  ",
    " ██▀ ▄███▀   ",
    " ██ ▄███▀    ",
    " █▌▄███▀     ",
    " ████▀▀      ",
    " ▀█▀         ",
    "  ▀          ",
]

# 256-color ramp per logo column. Source of truth: omp's brand gradient
# (softened #c666d8 → #9b4dff → #5ad8e6 — the bright pink pulled toward violet), packages/collab-web tokens.css), interpolated
# in RGB space and quantized to the nearest xterm-256 cube colors.
LOGO_RAMP = [170, 134, 134, 135, 135, 135, 99, 99, 105, 105, 75, 74, 80]

# exact brand RGBs per column for truecolor terminals (CLI path)
_BRAND_STOPS = [(0.0, (198, 102, 216)), (0.5, (155, 77, 255)), (1.0, (90, 216, 230))]


def _lerp_rgb(pos):
    for (p0, c0), (p1, c1) in zip(_BRAND_STOPS, _BRAND_STOPS[1:]):
        if p0 <= pos <= p1:
            f = (pos - p0) / (p1 - p0)
            return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
    return _BRAND_STOPS[-1][1]


LOGO_RGB = [_lerp_rgb(i / (len(LOGO[0]) - 1)) for i in range(len(LOGO[0]))]
LOGO_ANSI = [f"\033[38;5;{c}m" for c in LOGO_RAMP]
LOGO_TRUECOLOR = [f"\033[38;2;{r};{g};{b}m" for r, g, b in LOGO_RGB]
ANSI_RESET = "\033[0m"

# lane -> (example env, target env, steering env key, structure test, vector repo)
LANES = [
    dict(name="DSV4 TP=2 serving — 2x DGX Spark, Anemll recipe",
         example="recipe/anemll/.env.dspark.example",
         target="recipe/anemll/.env.dspark",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-dsv4-hotfix-structure.py",
         vector_repo="msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29",
         steer_modes=None,
         port=8888),
    dict(name="Qwen TP=1 serving — single DGX Spark",
         example="recipe/qwen/.env.qwen.example",
         target="recipe/qwen/.env.qwen",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-qwen-steering-structure.py",
         vector_repo="msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49",
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

def omp_provider_base():
    """The dspark provider's baseUrl from ~/.omp/agent/models.yml, if set."""
    if not os.path.exists(OMP_MODELS):
        return None
    with open(OMP_MODELS) as f:
        m = re.search(rf"(?ms)^  {PROVIDER}:.*?baseUrl: (\S+)", f.read())
    return m.group(1) if m else None


def default_base():
    """Best default endpoint: the configured omp provider's, else localhost."""
    return omp_provider_base() or DEFAULT_BASE


def detect_state():
    """Summarize existing local setup: per-lane env presence + key values,
    and whether the omp provider is installed."""
    lines = []
    for lane in LANES:
        path = os.path.join(HERE, lane["target"])
        label = lane["name"].split(" — ")[0]
        if not os.path.exists(path):
            lines.append(f"{label}: not configured")
            continue
        with open(path) as f:
            env = dict(re.findall(r"(?m)^([A-Z_]+)=(\S+)", f.read()))
        bits = []
        if env.get("MASTER_ADDR"):
            bits.append(f"head {env['MASTER_ADDR']}")
        if env.get("WORKER_HOST"):
            bits.append(f"worker {env['WORKER_HOST']}")
        if env.get("MODELS"):
            bits.append(env["MODELS"])
        if env.get("VLLM_PORT"):
            bits.append(f"port {env['VLLM_PORT']}")
        if env.get("STEER_MODE"):
            bits.append(f"steer={env['STEER_MODE']}")
        steer = env.get(lane["steer_key"], "")
        steer_line = None
        if steer:
            layers = [t for t in env.get("WEIGHTLESS_STEER_LAYERS", "").split(",") if t.strip()]
            alpha = env.get("WEIGHTLESS_STEER_ALPHA") or env.get("WEIGHTLESS_STEER_ALPHA") or "?"
            n = f", {len(layers)} layers" if layers else ""
            steer_line = f"steering on (α={alpha}{n})"
        elif lane["steer_key"] in env:
            bits.append("steering off")
        lines.append(f"{label}: " + (", ".join(bits) if bits else "configured"))
        if steer_line:
            fname = os.path.basename(steer)
            if len(fname) > 44:
                fname = "…" + fname[-43:]
            lines.append(f"  ↳ {steer_line} — {fname}")
    if os.path.exists(OMP_MODELS):
        base = omp_provider_base()
        if base:
            lines.append(f"omp provider '{PROVIDER}': installed ({base})")
        else:
            lines.append(f"omp provider '{PROVIDER}': config exists, provider missing")
    else:
        lines.append(f"omp provider '{PROVIDER}': not installed")
    return lines


def box(io, title, lines):
    """Draw a unicode info box."""
    width = min(72, max(len(title) + 2, *(len(l) for l in lines)) + 2)
    io.info("┌─ " + title + " " + "─" * (width - len(title) - 4) + "┐")
    for line in lines:
        io.info("│ " + line[:width - 2].ljust(width - 2) + "│")
    io.info("└" + "─" * width + "┘")


_MDNS_CACHE = None


def mdns_hosts(timeout=2.5):
    """SSH-advertising hosts on the LAN via mDNS (dns-sd on macOS,
    avahi-browse on Linux). Returns sorted hostnames like 'node-a.local'.
    Result is cached — one browse per process."""
    global _MDNS_CACHE
    if _MDNS_CACHE is not None:
        return _MDNS_CACHE
    hosts = set()
    import time as _time
    if shutil.which("dns-sd"):
        cmd = ["dns-sd", "-B", "_ssh._tcp", "local."]
        kill = True

        def parse(line):
            parts = line.split()
            if len(parts) >= 7 and parts[1].startswith("Add"):
                name = " ".join(parts[6:])
                if name.endswith(" SSH"):
                    name = name[:-4]
                if name and " " not in name:
                    hosts.add(name if name.endswith(".local") else name + ".local")
    elif shutil.which("avahi-browse"):
        cmd = ["avahi-browse", "-tp", "_ssh._tcp"]
        kill = False

        def parse(line):
            parts = line.split(";")
            if parts and parts[0] == "=" and len(parts) > 6 and parts[6]:
                h = parts[6].rstrip(".")
                hosts.add(h if "." in h else h + ".local")
    else:
        _MDNS_CACHE = []
        return []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
        if kill:
            _time.sleep(timeout)
            proc.terminate()
        out, _ = proc.communicate(timeout=timeout + 5)
        for line in out.splitlines():
            parse(line)
    except Exception:
        pass
    _MDNS_CACHE = sorted(hosts)
    return _MDNS_CACHE


def pick_host(io, prompt, default=""):
    """Host prompt backed by mDNS discovery: LAN hosts, plus a keep-current
    entry when the default isn't discovered (e.g. a fabric IP that is correct
    for the env but not mDNS-advertised), plus manual entry."""
    hosts = mdns_hosts()
    items = list(hosts)
    keep = None
    if default and default not in hosts:
        keep = f"keep current: {default}"
        items.append(keep)
    items.append("enter manually")
    if not hosts and keep is None:
        return io.text(prompt + ": ", default)
    preselect = items.index(keep) if keep else (hosts.index(default) if default in hosts else 0)
    sel = io.menu(f"{prompt} (mDNS):", items, preselect=preselect)
    if items[sel] == "enter manually":
        return io.text(prompt + ": ", default)
    if keep and items[sel] == keep:
        return default
    return items[sel]


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
    """Best-effort site values from existing (gitignored) lane env files —
    used to prefill wizard prompts and the diagnose ssh target."""
    vals = {}
    for lane in LANES:
        path = os.path.join(HERE, lane["target"])
        if not os.path.exists(path):
            continue
        with open(path) as f:
            env = dict(re.findall(r"(?m)^([A-Z_]+)=(\S+)", f.read()))
        if env.get("MASTER_ADDR"):
            vals.setdefault("head-ip", env["MASTER_ADDR"])
            vals.setdefault("host", env["MASTER_ADDR"])
        if env.get("WORKER_HOST"):
            vals.setdefault("worker-ip", env["WORKER_HOST"])
        m = re.search(r"/home/([^/]+)/", env.get("HF_CACHE", "") or env.get("MODELS", ""))
        if m:
            vals.setdefault("user", m.group(1))
    return vals


def deploy_commands(lane_idx, values, ssh_host=None):
    """(description, argv) pairs to push the lane to its node(s) and boot it.
    DSV4 targets the MiaAI clone dir on the head (its start script syncs the
    worker itself); Qwen targets a repo-shaped dir on the single node.
    ssh_host is the LAN-reachable name/IP — never the fabric address."""
    user = values.get("user", os.environ.get("USER", ""))
    if lane_idx == 0:
        head = f"{user}@{ssh_host or values.get('head-ip', '<head-ip>')}"
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
    host = f"{user}@{ssh_host or '<node-host>'}"
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


# remote files each lane deploys, paired with their local sources
DEPLOY_MAP = {
    0: [("recipe/anemll/.env.dspark", "dspark-miaai/.env.dspark"),
        ("recipe/anemll/docker-compose.dspark.yml", "dspark-miaai/docker-compose.dspark.yml"),
        ("recipe/anemll/start-deepseek-v4-flash-dspark.sh", "dspark-miaai/start-deepseek-v4-flash-dspark.sh"),
        ("patches/hotfix-dsv4-steering-projective.py", "dspark-miaai/patches/hotfix-dsv4-steering-projective.py")],
    1: [("recipe/qwen/serve-qwen38.sh", "dspark-deploy/recipe/qwen/serve-qwen38.sh"),
        ("recipe/qwen/.env.qwen", "dspark-deploy/recipe/qwen/.env.qwen"),
        ("patches/hotfix-qwen38-steering-projective.py", "dspark-deploy/patches/hotfix-qwen38-steering-projective.py")],
}
CONTAINER_GREP = {0: "deepseek", 1: "qwen38"}


def remote_preflight(io, lane_idx, values, ssh_host):
    """Check the remote before touching it: is the container up, and do the
    deployed files match the local ones (md5)? Returns True when the remote
    is fully synced and running."""
    import hashlib
    user = values.get("user", os.environ.get("USER", ""))
    target = f"{user}@{ssh_host}"
    probe = (
        f"docker ps --format '{{{{.Names}}}} {{{{.Status}}}}' 2>/dev/null "
        f"| grep -i {CONTAINER_GREP[lane_idx]} || echo 'CONTAINER_DOWN'; "
        + " ".join(f"md5sum {r} 2>/dev/null || echo MISSING {r};"
                   for _, r in DEPLOY_MAP[lane_idx])
    )
    io.info(f"$ ssh {target} '<container status + file checksums>'")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                        target, probe], capture_output=True, text=True)
    if r.returncode != 0:
        io.err(f"ssh failed ({r.returncode}) — cannot preflight")
        return False
    running = "CONTAINER_DOWN" not in r.stdout
    remote_md5 = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "MISSING":
            remote_md5[parts[1]] = None
        elif len(parts) == 2 and len(parts[0]) == 32:
            remote_md5[parts[1]] = parts[0]
    all_match = True
    for local, remote in DEPLOY_MAP[lane_idx]:
        lp = os.path.join(HERE, local)
        lm = hashlib.md5(open(lp, "rb").read()).hexdigest() if os.path.exists(lp) else None
        rm = remote_md5.get(remote)
        if rm is None:
            io.warn(f"  {remote}: missing on remote")
            all_match = False
        elif lm != rm:
            io.warn(f"  {remote}: differs from local")
            all_match = False
        else:
            io.ok(f"  {remote}: in sync")
    if running:
        io.ok(f"  container: running")
    else:
        io.warn(f"  container: not running")
    return all_match and running


def boot_command(lane_idx, values, ssh_host=None):
    """Just the boot step of deploy_commands, for the diagnose flow."""
    return deploy_commands(lane_idx, values, ssh_host)[-1]


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


def run_suite(io, base, model):
    """Run tests/0*.sh one by one, inside the wizard UI: each test's verdict
    line lands as a colored ✓/~/✗ row instead of dropping out to a shell."""
    import glob
    env = dict(os.environ, WEIGHTLESS_BASE_URL=base, WEIGHTLESS_MODEL=model,
               WEIGHTLESS_OMP_MODEL=f"{PROVIDER}/{model}")
    io.info("")
    io.header("endpoint test suite")
    io.info("─" * 40)
    fails = 0
    for t in sorted(glob.glob(os.path.join(HERE, "tests", "0*.sh"))):
        name = os.path.basename(t)
        io.info(f"… {name} running")
        try:
            r = subprocess.run(["sh", t], env=env, capture_output=True,
                               text=True, timeout=600)
        except subprocess.TimeoutExpired:
            io.err(f"✗ {name} — timed out after 600s")
            fails += 1
            continue
        lines = (r.stdout + r.stderr).strip().splitlines()
        # the test's own verdict line, not tool chatter interleaved on stdout
        verdict = next((l for l in lines if l.startswith(("PASS:", "FAIL:", "SKIP:"))),
                       lines[-1] if lines else "(no output)")
        if r.returncode == 0:
            io.ok(f"✓ {name} — {verdict}")
        elif r.returncode == 2:
            io.warn(f"~ {name} — {verdict}")
        else:
            io.err(f"✗ {name} — {verdict}")
            for l in lines[:-1][-3:]:
                io.err(f"    {l}")
            fails += 1
    io.info("─" * 40)
    if fails:
        io.err(f"{fails} test(s) failed")
        return 1
    io.ok("all endpoint tests passed")
    return 0


# ---------------------------------------------------------------- IO adapters

class CliIO:
    C = {"head": "\033[1;38;5;205m", "ok": "\033[38;5;80m", "err": "\033[31m",
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

    def menu(self, title, items, idle=None, preselect=0):  # idle is TUI-only
        print(self._c("head", title))
        for i, it in enumerate(items):
            label = it[1] if isinstance(it, tuple) else it
            mark = "›" if i == preselect else " "
            print(f" {mark} {self._c('head', str(i + 1) + ')')} {label}")
        while True:
            try:
                s = input("> ").strip()
            except EOFError:
                self._eof()
            if not s:
                return preselect
            if s.isdigit() and 1 <= int(s) <= len(items):
                return int(s) - 1
            print("  enter a number from the list")


class TuiIO:
    def __init__(self, stdscr):
        self.s = stdscr
        self.s.scrollok(True)
        self.row = 0
        self.color = False
        self.grad = []
        self.logo_pos = None
        self._tick = 0
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            # semantic palette on the brand gradient: pink headers, cyan ok,
            # red err, yellow warn, white-on-violet selection
            brand = (205, 80, curses.COLOR_RED, curses.COLOR_YELLOW)
            for i, fg in enumerate(brand, 1):
                curses.init_pair(i, fg, -1)
            try:
                curses.init_pair(5, curses.COLOR_WHITE, 99)  # violet bg
            except curses.error:  # basic terminal: magenta bg
                curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_MAGENTA)
            self.color = True
            if curses.COLORS >= 256:
                seen = {}
                for i, fg in enumerate(LOGO_RAMP):
                    if fg not in seen:
                        curses.init_pair(20 + len(seen), fg, -1)
                        seen[fg] = len(seen)
                self.grad = [curses.color_pair(20 + seen[fg]) for fg in LOGO_RAMP]
            else:
                # basic terminals: magenta left half, cyan right half
                n = len(LOGO[0])
                basic = [curses.COLOR_MAGENTA] * (n // 2) + [curses.COLOR_CYAN] * (n - n // 2)
                for i, fg in enumerate(dict.fromkeys(basic)):
                    curses.init_pair(20 + i, fg, -1)
                idx = {curses.COLOR_MAGENTA: 0, curses.COLOR_CYAN: 1}
                self.grad = [curses.color_pair(20 + idx[fg]) for fg in basic]

    def _attr(self, kind):
        if not self.color:
            return {"head": curses.A_BOLD}.get(kind, curses.A_NORMAL)
        return {"head": curses.color_pair(1) | curses.A_BOLD,
                "ok": curses.color_pair(2),
                "err": curses.color_pair(3),
                "warn": curses.color_pair(4),
                "sel": curses.color_pair(5),  # white on violet; bg carries the highlight
                "dim": curses.A_DIM}.get(kind, curses.A_NORMAL)

    def _next(self, n=1):
        max_y = self.s.getmaxyx()[0]
        if self.row + n > max_y - 1:  # overflow: scroll up, pin to bottom
            k = self.row + n - (max_y - 1)
            try:
                self.s.scroll(k)
            except curses.error:
                pass
            self.row -= k
            self.logo_pos = None  # the logo scrolled off — stop animating
        self.row += n
        return self.row - n

    def _w(self, row, col, s, attr=curses.A_NORMAL):
        """Bounds-safe write: clamp to the window, never raise addwstr ERR."""
        max_y, max_x = self.s.getmaxyx()
        if row < 0 or row >= max_y or col < 0 or col >= max_x:
            return
        s = str(s)[: max_x - col]
        if not s:
            return
        try:
            self.s.addstr(row, col, s, attr)
        except curses.error:
            pass

    def _put(self, row, col, msg, kind=None):
        self._w(row, col, msg, self._attr(kind) if kind else curses.A_NORMAL)
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
            max_x = self.s.getmaxyx()[1]
            self._w(r, 0, " " * (max_x - 1))
            self._w(r, 0, prompt, self._attr("head"))
            room = max_x - len(prompt) - 2
            shown = buf[-room:] if room > 0 else ""
            self._w(r, len(prompt), shown)
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
        self._w(r, 0, f"{prompt} ", self._attr("head"))
        self._w(r, len(prompt) + 1, f"[{hint}] ", self._attr("dim"))
        self.s.refresh()
        ch = self.s.get_wch()
        return ch.lower().startswith("y") if isinstance(ch, str) and ch.strip() else default

    def animate_logo(self):
        """Rotate the feather's gradient one column to the left."""
        if not self.logo_pos or not self.grad:
            return
        row0, col0 = self.logo_pos
        self._tick += 1
        n = len(self.grad)
        for i, art in enumerate(LOGO):
            for j, ch in enumerate(art):
                if ch != " ":
                    self._w(row0 + i, col0 + j, ch, self.grad[(j + self._tick) % n])
        self.s.refresh()

    def menu(self, title, items, idle=None, preselect=0):
        r = self._next(len(items) + 1)
        self._w(r, 0, title, self._attr("head"))
        sel = preselect
        if idle:
            self.s.timeout(150)
        try:
            while True:
                for i, it in enumerate(items):
                    label = it[1] if isinstance(it, tuple) else it
                    attr = self._attr("sel") if i == sel else curses.A_NORMAL
                    self._w(r + 1 + i, 2, ("› " if i == sel else "  ") + label, attr)
                self.s.refresh()
                ch = self.s.getch()
                if ch == -1:  # timeout tick — animate, keep waiting
                    idle()
                    continue
                if ch in (curses.KEY_UP, ord("k")):
                    sel = (sel - 1) % len(items)
                elif ch in (curses.KEY_DOWN, ord("j")):
                    sel = (sel + 1) % len(items)
                elif ch in (10, 13):
                    return sel
        finally:
            if idle:
                self.s.timeout(-1)


# ---------------------------------------------------------------- chains

def lane_chain(io, lane_idx):
    lane = LANES[lane_idx]
    example = os.path.join(HERE, lane["example"])
    target = os.path.join(HERE, lane["target"])
    io.header(lane["name"])
    io.info("─" * 60)

    # 1. site values (prefilled from the existing env file when present)
    saved = read_lane_env()
    values = {}
    for p in placeholders(example):
        prompt, default = PLACEHOLDER_HINTS.get(p, (p.replace("-", " "), ""))
        default = saved.get(p, default)
        if p in ("head-ip", "worker-ip"):
            values[p] = pick_host(io, f"{prompt} ({p})", default)
        else:
            values[p] = io.text(f"{prompt} ({p}): ", default)

    # 2. steering (default on)
    steering = io.confirm("Enable refusal steering (GLP vector patch)?", True)
    steer_mode = None
    if steering and lane["steer_modes"]:
        labels = [d for _, d in lane["steer_modes"]]
        sel = io.menu("Steering mode:", labels)
        steer_mode = lane["steer_modes"][sel][0]

    # 3. write env
    text = render_env(example, values, steer_mode=steer_mode,
                      steering=steering, steer_key=lane["steer_key"])
    if os.path.exists(target):
        if io.confirm(f"{target} exists — overwrite?", False):
            with open(target, "w") as f:
                f.write(text)
            io.ok(f"wrote {target}")
        else:
            # keep the existing file and continue the chain with it
            io.info("keeping the existing env file")
    else:
        with open(target, "w") as f:
            f.write(text)
        io.ok(f"wrote {target}")

    # 4. validate the steering patch + point at the vector
    if steering:
        io.info("validating the steering patch:")
        r = subprocess.run([sys.executable, os.path.join(HERE, lane["structure_test"])],
                           capture_output=True, text=True)
        for line in (r.stdout + r.stderr).splitlines():
            if "[PASS]" in line:
                io.ok(line.strip())
            elif "[FAIL]" in line:
                io.err(line.strip())
            elif "[SKIP]" in line:
                io.warn(line.strip())
            elif line.strip():
                io.info("  " + line.strip())
        if r.returncode not in (0, 2):
            io.err("steering validation failed — do not deploy until this is green")
        io.info(f"vector (gated, needs HF token): "
                f"huggingface-cli download {lane['vector_repo']} --include '*.gguf'")

    # 5. deploy (confirm-gated remote actions, preflight first)
    if io.confirm("Deploy to the node(s) over ssh now?", False):
        # MASTER_ADDR/head-ip is the RoCE fabric address — not routable from
        # the LAN. ssh needs a reachable host: the omp provider's by default.
        omp_host = urllib.parse.urlparse(default_base()).hostname
        note = "Head node ssh host" if lane_idx == 0 else "Node ssh host"
        ssh_host = pick_host(io, note + " (fabric IPs are not routable)",
                             omp_host or values.get("head-ip", ""))
        if remote_preflight(io, lane_idx, values, ssh_host):
            io.ok("remote is in sync and running — nothing to deploy")
            if not io.confirm("Redeploy and restart anyway?", False):
                io.info("deploy skipped")
                return tests_chain(io)
        for desc, argv in deploy_commands(lane_idx, values, ssh_host):
            io.info(f"$ {' '.join(argv)}")
            if not io.confirm(f"run: {desc}?", True):
                io.warn("skipped")
                continue
            rc = subprocess.call(argv)
            if rc != 0:
                io.err(f"FAILED ({rc}) — fix and re-run; aborting deploy")
                return rc
    else:
        io.info("deploy skipped (declined) — nothing was checked remotely. Answer y to "
                "preflight first: checksums + container status before anything is touched.")

    # 6. omp provider + endpoint tests
    return tests_chain(io)


def diagnose_chain(io, base=None):
    """Layered failure isolation for an endpoint that won't answer, with an
    optional remote check/boot over ssh."""
    base = base or io.text("Base URL to diagnose: ", default_base())
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
    # The env's MASTER_ADDR is the RoCE fabric IP — no sshd there. Prefer the
    # host actually being diagnosed; when that's loopback (stack runs remote
    # but we're testing locally), fall back to the omp provider's host.
    loopback = host in ("localhost", "127.0.0.1", "::1")
    omp_host = urllib.parse.urlparse(default_base()).hostname
    ssh_host = host if not loopback else (omp_host or saved.get("host", host))
    default_target = f"{saved.get('user', os.environ.get('USER', ''))}@{ssh_host}"
    if not io.confirm("Check the node over ssh (docker ps, GPU)?", True):
        return
    ssh_host = pick_host(io, "ssh host", ssh_host)
    target = f"{saved.get('user', os.environ.get('USER', ''))}@{ssh_host}"
    io.info(f"ssh target: {target}")
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
        desc, argv = boot_command(lane_idx, values, ssh_host=host)
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
    base = io.text("Base URL of the OpenAI-compatible server: ", default_base())
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
        return run_suite(io, base, model)
    io.info(f"done — later: sh {os.path.join(HERE, 'tests', 'run.sh')}")
    return 0


# ---------------------------------------------------------------- entry

def splash_tui(io):
    """Gradient feather left, local-setup box right (stacked if narrow),
    box vertically centered against the logo. The box is a real bordered
    curses window so the frame renders correctly at any terminal width."""
    rows, cols = io.s.getmaxyx()
    lines = detect_state()
    content_w = max(len("local setup") + 2, *(len(l) for l in lines))
    logo_col = 4                      # left margin; small cols crop on terminals whose block glyphs overhang
    bcol = logo_col + len(LOGO[0]) + 4
    side_by_side = cols >= bcol + min(content_w + 2, 56) + 2
    bh = len(lines) + 2
    if side_by_side:
        brow = io.row + max(0, (len(LOGO) - bh) // 2)
    else:
        brow = io.row + len(LOGO) + 1
        bcol = 0
    bw = min(content_w + 2, cols - bcol)  # outer width incl. borders

    win = curses.newwin(bh, bw, brow, bcol)
    if io.color:
        win.attron(io._attr("dim"))
    win.box()
    win.addstr(0, 2, " local setup ")
    if io.color:
        win.attroff(io._attr("dim"))
    for i, l in enumerate(lines):
        room = bw - 4
        shown = l if len(l) <= room else l[:room - 1] + "…"
        try:
            win.addstr(1 + i, 2, shown)
        except curses.error:
            pass
    win.refresh()

    io.logo_pos = (io.row, logo_col)
    for i, art in enumerate(LOGO):
        for j, ch in enumerate(art):
            if ch == " ":
                continue
            attr = io.grad[j] if io.grad else curses.A_NORMAL
            try:
                io.s.addstr(io.row + i, logo_col + j, ch, attr)
            except curses.error:
                pass
    io.s.refresh()
    io.row = max(io.row + len(LOGO), brow + bh) + 1  # one blank line after


def splash_cli(io):
    for art in LOGO:
        row = "  " + art  # same left margin as the TUI (block-glyph overhang)
        if io.color:
            row = "  " + "".join(
                (LOGO_TRUECOLOR[j] + ch + ANSI_RESET) if ch != " " else ch
                for j, ch in enumerate(art))
        print(row)
    box(io, "local setup", detect_state())


def run(io):
    bun, omp = prereqs()
    io.info(f"bun: {bun or 'not found (only needed for omp/tests)'}")
    io.info(f"omp:  {omp or 'not found (only needed for tests)'}")
    io.info("")
    choice = io.menu("What to set up:", idle=getattr(io, "animate_logo", None), items=[
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
    io.header("  weightless setup")
    io._put(1, 0, "  lean abliteration steering, served", "dim")
    io.row = 3
    splash_tui(io)
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
    io.header("== weightless setup ==")
    io.info("")
    splash_cli(io)
    io.info("")
    return run(io)


if __name__ == "__main__":
    sys.exit(main())
