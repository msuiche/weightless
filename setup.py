#!/usr/bin/env python3
"""Full-chain setup for weightless.

Pick a lane, fill in the site values (hosts, user), generate the real
gitignored env file from the shipped example, validate the steering patch,
deploy to the node(s) over ssh (confirm-gated), then configure agent clients
and run the endpoint smoke tests. If the endpoint is down, the
diagnose chain walks the layers (DNS → TCP → HTTP) and can check/boot the
stack over ssh. Stdlib only: a light curses TUI with colors on a terminal,
ANSI-colored prompts otherwise. Non-interactive alternative:
`sh tests/install.sh && sh tests/run.sh` with DSPARK_* env overrides.
"""
import json
import os
import re
import shlex
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
OMP_CONFIG = os.path.expanduser("~/.omp/agent/config.yml")
HERMES_CONFIG = os.path.expanduser("~/.hermes/config.yaml")
HERMES_MARKER = "# weightless lane models (--model):"
DEFAULT_BASE = "http://localhost:8888/v1"
DEFAULT_MODEL = "deepseek-v4-flash-dspark"
PROVIDER = "weightless"

# DEMO=1 (or DEMO_MODE=1): the probe and test suite replay
# canned output against whatever base URL is configured. For screenshots.
DEMO = os.environ.get("DEMO") == "1" or os.environ.get("DEMO_MODE") == "1"

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
# in RGB space and quantized to the nearest xterm-256 cube colors; 69 bridges
# the 105→75 quantization step so the animated cycle has no hard blue seam.
LOGO_RAMP = [170, 134, 134, 135, 135, 135, 99, 99, 105, 69, 75, 74, 80]

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

# lane -> (example env, target env, steering env key, structure test, vector
# repo). Multi-node lanes (nodes: 2|4, head+workers over RoCE) add the remote
# deploy layout: remote_dir, recipe_files (basenames in the example's dir),
# start_script, hotfix (basename in patches/).
LANES = [
    dict(name="DSV4 TP=2 serving — 2x DGX Spark, Anemll recipe",
         example="recipe/anemll/.env.dsv4.example",
         target="recipe/anemll/.env.dsv4",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-dsv4-hotfix-structure.py",
         vector_repo="msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29",
         model_repo="deepseek-ai/DeepSeek-V4-Flash-0731",
         docker_image="ghcr.io/anemll/dspark-vllm-gx10:0.1.1",
         image_key="DSPARK_VLLM_IMAGE",
         steer_modes=None,
         nodes=2,
         remote_dir="dspark-miaai",
         recipe_files=[".env.dsv4", "docker-compose.dsv4.yml",
                       "start-deepseek-v4-flash-dspark.sh"],
         start_script="start-deepseek-v4-flash-dspark.sh",
         hotfix="hotfix-dsv4-steering-projective.py",
         port=8888),
    dict(name="Qwen TP=1 serving — single DGX Spark",
         example="recipe/qwen/.env.qwen.example",
         target="recipe/qwen/.env.qwen",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-qwen-steering-structure.py",
         vector_repo="msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49",
         model_repo="unsloth/Qwen3.8-27B-NVFP4",
         docker_image="ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813",
         image_key="QWEN_IMAGE",
         steer_modes=[
             ("gguf", "gguf — hotfix-patched vLLM, fail-closed (default, validated)"),
             ("lora", "lora — stock vLLM --enable-lora, no patch (validated)")],
         port=8078),
    dict(name="Qwen3.8-Flash-Next TP=2 serving — 2x DGX Spark, day-0 image",
         example="recipe/qwen38fn/.env.qwen38fn.example",
         target="recipe/qwen38fn/.env.qwen38fn",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-qwen38fn-steering-structure.py",
         vector_repo="msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47",
         model_repo="RadixArk/Qwen3.8-Flash-Next-NVFP4",
         docker_image="vllm/vllm-openai:qwen38-flash-next",
         image_key="QWEN38FN_IMAGE",
         steer_modes=None,
         nodes=2,
         remote_dir="dspark-qwen38fn",
         recipe_files=[".env.qwen38fn", "start-qwen38-flash-next-dspark.sh"],
         start_script="start-qwen38-flash-next-dspark.sh",
         hotfix="hotfix-qwen38fn-steering-projective.py",
         extra_patches=["patch-qwen38fn-ple-fp8-nvfp4.py"],
         port=8079),
    dict(name="GLM-5.3-Flash TP=4 serving — 4x DGX Spark, day-0 image",
         example="recipe/glm53/.env.glm53.example",
         target="recipe/glm53/.env.glm53",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-glm53-steering-structure.py",
         vector_repo="msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44",
         model_repo="RedHatAI/GLM-5.3-Flash-NVFP4",
         docker_image="radixark/vllm-glm53-flash:sm121-v8",
         image_key="GLM53_IMAGE",
         steer_modes=None,
         nodes=4,
         remote_dir="dspark-glm53",
         recipe_files=[".env.glm53", "start-glm53-flash-dspark.sh"],
         start_script="start-glm53-flash-dspark.sh",
         hotfix="hotfix-glm53-steering-projective.py",
         extra_patches=["vendor/sparse_attn_indexer_kpool_sm121.py"],
         port=8080),
    dict(name="GLM-5.3 743B — Modal cloud (8x H100, GLP-77 α=1.0)",
         # Cloud lane: deploys via modal/cloud_serve.py and delivers an
         # endpoint URL. The on-prem 4x DGX recipe remains in recipe/glm53xl/.
         cloud="modal",
         modal_app="modal/cloud_serve.py",
         cloud_weights="~465 GB",
         cloud_boot="30 min",
         example="recipe/glm53xl/.env.glm53xl.example",
         target="recipe/glm53xl/.env.glm53xl",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-glm53xl-steering-structure.py",
         vector_repo="msuiche/GLM-5.3-abliterated-cyber-GLP-77",
         model_repo="RadixArk/GLM-5.3-NVFP4",
         steer_modes=None,
         port=8000),
    dict(name="Inkling-Small TP=2 serving — 2x DGX Spark, day-0 vLLM v0.28.0",
         example="recipe/inkling/.env.inkling.example",
         target="recipe/inkling/.env.inkling",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-inkling-steering-structure.py",
         vector_repo="msuiche/Inkling-Small-abliterated-cyber-GLP-41",
         model_repo="thinkingmachines/Inkling-Small-NVFP4",
         docker_image="vllm/vllm-openai:v0.28.0",
         image_key="INKLING_IMAGE",
         steer_modes=None,
         nodes=2,
         remote_dir="dspark-inkling",
         recipe_files=[".env.inkling", "start-inkling-sm121.sh",
                       "files/fa4_rel_attention-sm121.py",
                       "files/inkling-model-gb10.py",
                       "files/inkling-model-gb10-steered.py"],
         start_script="start-inkling-sm121.sh",
         steering_supported=True,
         hotfix="hotfix-inkling-steering-projective.py",
         extra_patches=["hotfix-inkling-gb10-load-reclaim.py",
                        "hotfix-inkling-sm121-relattn.py"],
         port=8082),
    dict(name="GLM-5.3-Flash TP=2 serving — 2x DGX Spark, sm121-v8, 128K",
         example="recipe/glm53tp2/.env.glm53tp2.example",
         target="recipe/glm53tp2/.env.glm53tp2",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-glm53-steering-structure.py",
         vector_repo="msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44",
         model_repo="RedHatAI/GLM-5.3-Flash-NVFP4",
         docker_image="ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v8",
         image_key="GLM53_IMAGE",
         steer_modes=None,
         nodes=2,
         remote_dir="dspark-glm53tp2",
         recipe_files=[".env.glm53tp2", "start-glm53-flash-tp2.sh"],
         start_script="start-glm53-flash-tp2.sh",
         hotfix="hotfix-glm53-steering-projective.py",
         extra_patches=["vendor/sparse_attn_indexer_kpool_sm121.py"],
         port=8080),
    dict(name="Nemotron-3.5-Lightning TP=1 — single DGX Spark, NVFP4",
         example="recipe/nemotron35/.env.nemotron35.example",
         target="recipe/nemotron35/.env.nemotron35",
         steer_key="WEIGHTLESS_GLP",
         vector_repo="msuiche/Nemotron-3.5-Lightning-30B-A3B-abliterated-cyber-GLP-51-L1-51-a1.0",
         steer_modes=None,
         nodes=1,
         remote_dir="nemotron35-glp",
         recipe_files=[".env.nemotron35", "serve-nemotron35.sh"],
         start_script="serve-nemotron35.sh",
         hotfix="hotfix-nemotron35-steering-projective.py",
         model_repo="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
         docker_image="vllm/vllm-openai:v0.28.0",
         image_key="NEMOTRON_IMAGE",
         port=8083),
    dict(name="Kimi K3 2.9T — Modal cloud (2x H200:8 PP2xTP8, GLP-92 α=1.0)",
         # Cloud lane: deploys via modal/cloud_serve_k3.py and delivers an
         # endpoint URL. The weights are ~1.56 TB (MXFP4 experts, no smaller
         # checkpoint exists); first ensure_weights run is the long pole.
         cloud="modal",
         modal_app="modal/cloud_serve_k3.py",
         cloud_weights="~1.56 TB",
         cloud_boot="45 min",
         steer_key="WEIGHTLESS_STEER_PATH",
         structure_test="scripts/test-k3-steering-structure.py",
         vector_repo="msuiche/Kimi-K3-abliterated-cyber-GLP-92-L1-92-a1.0",
         model_repo="moonshotai/Kimi-K3",
         steer_modes=None,
         port=8000),
]
PLACEHOLDER_HINTS = {
    "head-ip": ("Head node IP or hostname", ""),
    "worker-ip": ("Worker node IP or hostname", ""),
    "worker2-ip": ("Worker 2 node IP or hostname", ""),
    "worker3-ip": ("Worker 3 node IP or hostname", ""),
    "user": ("Remote username on the node(s)", os.environ.get("USER", "")),
}


# ---------------------------------------------------------------- core logic

def omp_provider_base():
    """The weightless provider's baseUrl from ~/.omp/agent/models.yml, if set."""
    if not os.path.exists(OMP_MODELS):
        return None
    with open(OMP_MODELS) as f:
        text = f.read()
    providers = yaml_block(text, "providers")
    body = text[slice(*providers)] if providers else ""
    block = yaml_block(body, PROVIDER, 2)
    m = re.search(r"(?m)^ +baseUrl: +(\S+)", body[slice(*block)]) if block else None
    return m.group(1).strip("\"'") if m else None


def default_base():
    """Best default endpoint: the configured omp provider's, else localhost."""
    return omp_provider_base() or DEFAULT_BASE


def detect_state():
    """Summarize existing local setup: per-lane env presence + key values,
    and whether the agent clients are configured."""
    if DEMO:
        return [
            "DSV4 TP=2 serving: head node-a.local, worker node-b.local, port 8888",
            "  ↳ steering on (α=4.0, 29 layers) — …cyber-GLP-29-L10-38-a4.gguf",
            "Qwen TP=1 serving: not configured",
            f"omp provider '{PROVIDER}': installed (http://node-a.local:8888/v1)",
            f"hermes: configured (model {DEFAULT_MODEL})",
        ]
    lines = []
    for lane in LANES:
        if lane.get("cloud"):
            label = lane["name"].split(" — ")[0]
            lines.append(f"{label}: cloud lane (Modal) — pick it to deploy")
            continue
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
    try:
        with open(HERMES_CONFIG) as f:
            text = f.read()
    except OSError:
        text = ""
    block = yaml_block(text, "model")
    model = re.search(r"(?m)^ +default: +([^\n]+)", text[slice(*block)]) if block else None
    if HERMES_MARKER in text.splitlines():
        value = model.group(1).strip().strip("\"'") if model else "?"
        lines.append(f"hermes: configured (model {value})")
    else:
        lines.append("hermes: not installed")
    return lines


def box(io, title, lines, border="dim", title_kind="head"):
    """Draw an info box. TUI: a real bordered curses window (never misaligned).
    CLI: unicode frame with a colored border and title."""
    if isinstance(io, TuiIO):
        io.tui_box(title, lines, border, title_kind)
        return
    width = min(72, max(len(title) + 2, *(len(l) for l in lines)) + 2)
    b = lambda s: io._c(border, s) if getattr(io, "color", False) else s
    tt = io._c(title_kind, title) if getattr(io, "color", False) else title
    io.info(b("┌─ ") + tt + b(" " + "─" * (width - len(title) - 4) + "┐"))
    for line in lines:
        io.info(b("│ ") + line[:width - 2].ljust(width - 2) + b("│"))
    io.info(b("└" + "─" * width + "┘"))


_MDNS_CACHE = None


def mdns_hosts(timeout=2.5):
    """SSH-advertising hosts on the LAN via mDNS (dns-sd on macOS,
    avahi-browse on Linux). Returns sorted hostnames like 'node-a.local'.
    Result is cached — one browse per process."""
    global _MDNS_CACHE
    if DEMO:
        return ["node-a.local", "node-b.local"]
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


def validate_lane_env(lane, env_text):
    """Hardware-fit rules for a rendered lane env. Returns (errors, warnings).

    These are measured failures, not theory: each rule names the date it cost
    us a rig-day. The same checks run fail-closed in the lane's start script
    (defense in depth — the env file can be edited by hand after the wizard).
    """
    errors, warnings = [], []
    env = dict(re.findall(r"(?m)^([A-Z_]+)=(\S*)", env_text))
    nodes = lane.get("nodes", 1)
    if lane.get("start_script", "").startswith("start-qwen38"):
        ple = env.get("VLLM_PLE_CPU_OFFLOAD", "1")
        util = float(env.get("GPU_MEMORY_UTILIZATION", "0.90") or 0.90)
        ctx = int(env.get("MAX_MODEL_LEN", "262144") or 262144)
        if ple == "1" and nodes > 1:
            errors.append(
                "VLLM_PLE_CPU_OFFLOAD=1 is rejected by the arm64 day-0 image "
                "at nnodes=2 (2026-09-01: crash-looped a 2x Spark boot). "
                "Set it to 0 and keep util <= 0.88.")
        if ple == "0":
            budget_ok = util <= 0.88 if ctx <= 131072 else util <= 0.80
            if not budget_ok:
                errors.append(
                    f"PLE in HBM (~25.5 GB/rank) + util {util} + ctx {ctx} "
                    "exceeds the GB10 unified-memory envelope: util <= 0.88 "
                    "at <=131K ctx, or util <= 0.80 at 262K. A miss wedges the "
                    "node (0% free -> earlyoom cannot kill CUDA-stuck procs).")
    return errors, warnings


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
        for i in ("2", "3"):
            w = env.get(f"WORKER{i}_HOST")
            if w:
                vals.setdefault(f"worker{i}-ip", w)
        m = re.search(r"/home/([^/]+)/", env.get("HF_CACHE", "") or env.get("MODELS", ""))
        if m:
            vals.setdefault("user", m.group(1))
    return vals


def lane_env(lane_idx, values):
    """Read the deployed configuration, or render example defaults for a plan.
    Parse assignments without executing shell expressions from an env file."""
    lane = LANES[lane_idx]
    path = os.path.join(HERE, lane["target"])
    if os.path.exists(path):
        with open(path) as f:
            text = f.read()
    else:
        text = render_env(os.path.join(HERE, lane["example"]), values)
    env = {}
    for key, value in re.findall(r"(?m)^([A-Z0-9_]+)=(.*)$", text):
        words = shlex.split(value, comments=True)
        env[key] = " ".join(words)
    return env


def node_command(values, ssh_host, command, worker=None):
    """Reach workers through the head, using their fabric addresses."""
    user = values.get("user", os.environ.get("USER", ""))
    if worker:
        command = shlex.join(["ssh", "-o", "BatchMode=yes", f"{user}@{worker}", command])
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            f"{user}@{ssh_host or values.get('head-ip', '<head-ip>')}", command]


def lane_workers(lane_idx, env):
    return [env.get(f"WORKER{n if n > 1 else ''}_HOST", f"<worker{n if n > 1 else ''}-ip>")
            for n in range(1, LANES[lane_idx].get("nodes", 1))]


def hf_token():
    """Keep credentials out of env files, command plans and argv."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        try:
            with open(os.path.expanduser("~/.cache/huggingface/token")) as f:
                token = f.read().strip()
        except OSError:
            pass
    return token


def asset_commands(lane_idx, values, ssh_host=None):
    """Idempotent (description, argv) asset plan. Downloads read a token on stdin.
    HF tooling lives in a remote venv; the wizard itself remains stdlib-only."""
    if DEMO:
        return []
    lane = LANES[lane_idx]
    if lane.get("cloud"):
        return []  # cloud assets live on Modal volumes, not the rig
    env = lane_env(lane_idx, values)
    user = values.get("user", os.environ.get("USER", ""))
    cache = env.get("HF_CACHE", f"/home/{user}/.cache/huggingface")
    workers = lane_workers(lane_idx, env)
    worker_cache = env.get("WORKER_HF_CACHE", cache)
    repo = (env.get("MODEL") or env.get("WEIGHTLESS_MODEL")
            or env.get("MODEL_ID") or lane["model_repo"])
    model_dir = "models--" + repo.replace("/", "--")
    image = env.get(lane["image_key"], lane["docker_image"])
    q = shlex.quote
    steps = []

    def add(desc, command, worker=None):
        steps.append((desc, node_command(values, ssh_host, command, worker)))

    # Read before installing tooling so pip never consumes the credential.
    download = ('IFS= read -r HF_TOKEN; export HF_TOKEN; '
                'unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; '
                'if [ ! -x "$HOME/.cache/weightless-hf/bin/hf" ]; then '
                'python3 -m venv "$HOME/.cache/weightless-hf" && '
                '"$HOME/.cache/weightless-hf/bin/pip" install huggingface_hub || exit 1; fi; '
                '"$HOME/.cache/weightless-hf/bin/hf" download ')
    revision = (env.get("DSPARK_REVISION") if lane_idx == 0
                else env.get("MODEL_REVISION") if lane_idx == 7 else None)
    args = [repo, "--cache-dir", cache]
    if revision:
        args += ["--revision", revision]
    add(f"download model weights: {repo}", download + shlex.join(args))
    for worker in workers:
        dest = f"{user}@{worker}:{worker_cache}/"
        mkdir = shlex.join(["ssh", "-o", "BatchMode=yes", f"{user}@{worker}",
                            "mkdir -p " + q(worker_cache)])
        add(f"rsync model cache to worker {worker} (fabric)",
            mkdir + " && " + shlex.join(["rsync", "-a", "--partial", "--progress", "-s",
                                         "-e", "ssh -o BatchMode=yes",
                                         f"{cache}/{model_dir}", dest]))
    for worker in [None, *workers]:
        node_cache = worker_cache if worker else cache
        label = worker or "head"
        # HF_HOME-based launchers resolve hub/models--..., while TP2 and the
        # GLM template lookup use models--... directly at the cache root.
        if lane_idx in (0, 2, 3, 5):
            link = f"{node_cache}/hub/{model_dir}"
            source = f"{node_cache}/{model_dir}"
            add(f"expose HF hub cache on {label}",
                f"mkdir -p {q(node_cache + '/hub')} && "
                f"if [ ! -e {q(link)} ] && [ ! -L {q(link)} ]; then "
                f"ln -s {q('../' + model_dir)} {q(link)}; "
                f"elif [ ! {q(link)} -ef {q(source)} ]; then "
                f"rsync -a --progress --link-dest={q(source)} {q(source + '/')} {q(link + '/')}; fi", worker)
        if lane.get("local_image") and image == lane["docker_image"]:
            add(f"verify recipe's locally built image on {label} (see recipe/glm53xl/README.md)",
                "docker image inspect " + q(image), worker)
        else:
            add(f"pull Docker image on {label}: {image}", "docker pull " + q(image), worker)
        # Qwen and GLM XL mount a directory, so dereference snapshot symlinks
        # when materializing weights; cache-relative links break in /models.
        weights = (env["MODELS"] + "/Qwen3.8-27B-NVFP4" if lane_idx == 1 else
                   env["WEIGHTS_WORKERS" if worker else "WEIGHTS_HEAD"] if lane_idx == 4 else None)
        if weights:
            root = f"{node_cache}/{model_dir}"
            add(f"stage mounted model directory on {label}: {weights}",
                f"revision=$(cat {q(root + '/refs/main')}) && mkdir -p {q(weights)} && "
                f"rsync -aL --partial --progress {q(root + '/snapshots/')}\"$revision\"/ {q(weights + '/')}", worker)
        steer = env.get(lane["steer_key"], "")
        if steer:
            fname = os.path.basename(steer)
            add(f"download GLP vector on {label}: {lane['vector_repo']}",
                download + shlex.join([lane["vector_repo"], fname, "--local-dir", node_cache]), worker)
            if lane_idx == 1:
                dest = env["MODELS"] + "/cvec"
                add("stage Qwen vector in the /models mount",
                    f"mkdir -p {q(dest)} && cp {q(node_cache + '/' + fname)} {q(dest + '/' + fname)}", worker)
    if lane_idx == 1 and env.get("STEER_MODE") == "lora":
        adapter = "Qwen3.8-27B-refusal-abliterated-lora-r1-down_proj-L1-63-a1.safetensors"
        mount = env.get("QWEN_LORA_DIR", "/models/lora/qwen-abliterated")
        if not mount.startswith("/models/"):
            raise ValueError("QWEN_LORA_DIR must be inside the recipe's /models mount")
        dest = env["MODELS"] + mount[len("/models"):]
        config = json.dumps(dict(peft_type="LORA", r=1, lora_alpha=1, bias="none",
                                 task_type="CAUSAL_LM", target_modules=["mlp.down_proj"]))
        add("download Qwen LoRA adapter", download + shlex.join([lane["vector_repo"], adapter,
                                                               "--local-dir", cache]))
        add("package Qwen LoRA for the launcher",
            f"mkdir -p {q(dest)} && cp {q(cache + '/' + adapter)} {q(dest + '/adapter_model.safetensors')} && "
            f"printf '%s\\n' {q(config)} > {q(dest + '/adapter_config.json')}")
    return steps


def prepare_assets(io, lane_idx, values, ssh_host):
    io.header("prepare assets")
    if DEMO:
        io.info("demo: asset preparation skipped")
        return True
    token = hf_token()
    env = lane_env(lane_idx, values)
    if (env.get(LANES[lane_idx]["steer_key"]) or env.get("STEER_MODE") == "lora") and not token:
        io.err("GLP repositories are gated: accept access on Hugging Face, then set "
               "HF_TOKEN or run hf auth login (~/.cache/huggingface/token) and retry.")
        return False
    try:
        cmds = asset_commands(lane_idx, values, ssh_host)
    except ValueError as exc:
        io.err(str(exc))
        return False
    for desc, argv in cmds:
        io.info("$ " + shlex.join(argv))
        if not io.confirm(f"run: {desc}?", True):
            io.warn("asset preparation skipped — deploy aborted")
            return False
        io.info(desc)
        # Only downloads receive the token; SSH forwards stdin to the worker.
        rc = subprocess.run(argv, input=(token + "\n") if desc.startswith("download ") else None,
                            text=True).returncode
        if rc != 0:
            io.err(f"FAILED ({rc}) — fix and re-run; aborting deploy")
            return False
    return True


def deploy_commands(lane_idx, values, ssh_host=None):
    """(description, argv) pairs to push the lane to its node(s) and boot it.
    Multi-node lanes (nodes: 2|4) target a remote dir on the head (their
    start script syncs the worker(s) itself); Qwen targets a repo-shaped dir
    on the single node.
    ssh_host is the LAN-reachable name/IP — never the fabric address."""
    if DEMO:
        return []
    user = values.get("user", os.environ.get("USER", ""))
    if LANES[lane_idx].get("nodes", 1) > 1:
        lane = LANES[lane_idx]
        head = f"{user}@{ssh_host or values.get('head-ip', '<head-ip>')}"
        remote = lane["remote_dir"]
        r = os.path.join(HERE, os.path.dirname(lane["example"]))
        # recipe_files may carry subdirs (inkling's files/*.py): preserve them
        # remotely instead of flattening into remote/.
        sync_steps = [
            (f"prepare {head}:{remote}/",
             ["ssh", head, f"mkdir -p {remote}/patches {remote}/files"]),
        ]
        for sub in sorted({os.path.dirname(f) for f in lane["recipe_files"]}):
            group = [f for f in lane["recipe_files"] if os.path.dirname(f) == sub]
            dest = f"{head}:{remote}/{sub + '/' if sub else ''}"
            sync_steps.append(
                (f"sync env + recipe files{' (' + sub + '/)' if sub else ''}",
                 ["scp", *[os.path.join(r, f) for f in group], dest]))
        sync_steps += [
            ("sync steering hotfix + extra patches",
             ["scp", *[os.path.join(HERE, "patches", p)
                       for p in [lane["hotfix"], *lane.get("extra_patches", [])]],
              f"{head}:{remote}/patches/"]),
            ("boot the stack (start script syncs the worker itself)",
             ["ssh", head, f"cd {remote} && bash {lane['start_script']}"]),
        ]
        if lane_idx == 5:
            worker = values.get("worker-ip", "<worker-ip>")
            staged = " ".join(f"{remote}/{f}" for f in lane["recipe_files"] if f.startswith("files/"))
            sync_steps.insert(-1, ("sync Inkling patches to worker",
                ["ssh", head, f"ssh {user}@{worker} mkdir -p {remote}/files && "
                              f"scp {staged} {user}@{worker}:{remote}/files/"]))
        return sync_steps
    if lane_idx == 7:
        # Nemotron single-node: repo-shaped remote (recipe/ + patches/), the
        # serve script resolves the hotfix at ../../patches/.
        host = f"{user}@{ssh_host or values.get('head-ip', '<node-host>')}"
        remote = LANES[7]["remote_dir"]
        return [
            (f"prepare {host}:{remote}/ (repo-shaped: recipe + patches)",
             ["ssh", host, f"mkdir -p {remote}/recipe/nemotron35 {remote}/patches"]),
            ("sync serve script + env",
             ["scp", os.path.join(HERE, "recipe", "nemotron35", "serve-nemotron35.sh"),
              os.path.join(HERE, "recipe", "nemotron35", ".env.nemotron35"),
              f"{host}:{remote}/recipe/nemotron35/"]),
            ("sync steering hotfix",
             ["scp", os.path.join(HERE, "patches", "hotfix-nemotron35-steering-projective.py"),
              f"{host}:{remote}/patches/"]),
            ("boot the container",
             ["ssh", host, f"bash {remote}/recipe/nemotron35/serve-nemotron35.sh"]),
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


def vector_paths(lane_idx):
    """(local_path, remote_home_relative_path) for the lane's GLP vector,
    derived from the lane env. None when steering isn't configured."""
    env_path = os.path.join(HERE, LANES[lane_idx]["target"])
    if not os.path.exists(env_path):
        return None
    with open(env_path) as f:
        env = dict(re.findall(r"(?m)^([A-Z_]+)=(\S+)", f.read()))
    steer = env.get(LANES[lane_idx]["steer_key"], "")
    if not steer:
        return None
    fname = os.path.basename(steer)
    local = os.path.join(HERE, ".vectors", fname)  # staging dir (gitignored)
    if LANES[lane_idx].get("nodes", 1) > 1:
        # multi-node lanes read the vector from the HF cache root on all nodes
        hf = env.get("HF_CACHE", "~/.cache/huggingface")
        home_rel = hf.split("/", 3)[3] if hf.startswith("/home/") else ".cache/huggingface"
        return (local, f"{home_rel}/{fname}")
    models = env.get("MODELS", "")
    if not models:
        return None
    base = os.path.basename(models.rstrip("/"))  # remote dir mirrors local name
    return (local, f"{base}/cvec/{fname}")


# remote files each lane deploys, paired with their local sources
DEPLOY_MAP = {
    0: [("recipe/anemll/.env.dsv4", "dspark-miaai/.env.dsv4"),
        ("recipe/anemll/docker-compose.dsv4.yml", "dspark-miaai/docker-compose.dsv4.yml"),
        ("recipe/anemll/start-deepseek-v4-flash-dspark.sh", "dspark-miaai/start-deepseek-v4-flash-dspark.sh"),
        ("patches/hotfix-dsv4-steering-projective.py", "dspark-miaai/patches/hotfix-dsv4-steering-projective.py")],
    1: [("recipe/qwen/serve-qwen38.sh", "dspark-deploy/recipe/qwen/serve-qwen38.sh"),
        ("recipe/qwen/.env.qwen", "dspark-deploy/recipe/qwen/.env.qwen"),
        ("patches/hotfix-qwen38-steering-projective.py", "dspark-deploy/patches/hotfix-qwen38-steering-projective.py")],
    2: [("recipe/qwen38fn/.env.qwen38fn", "dspark-qwen38fn/.env.qwen38fn"),
        ("recipe/qwen38fn/start-qwen38-flash-next-dspark.sh", "dspark-qwen38fn/start-qwen38-flash-next-dspark.sh"),
        ("patches/hotfix-qwen38fn-steering-projective.py", "dspark-qwen38fn/patches/hotfix-qwen38fn-steering-projective.py"),
        ("patches/patch-qwen38fn-ple-fp8-nvfp4.py", "dspark-qwen38fn/patches/patch-qwen38fn-ple-fp8-nvfp4.py")],
    3: [("recipe/glm53/.env.glm53", "dspark-glm53/.env.glm53"),
        ("recipe/glm53/start-glm53-flash-dspark.sh", "dspark-glm53/start-glm53-flash-dspark.sh"),
        ("patches/hotfix-glm53-steering-projective.py", "dspark-glm53/patches/hotfix-glm53-steering-projective.py"),
        ("patches/vendor/sparse_attn_indexer_kpool_sm121.py", "dspark-glm53/patches/sparse_attn_indexer_kpool_sm121.py")],
    4: [("recipe/glm53xl/.env.glm53xl", "dspark-glm53xl/.env.glm53xl"),
        ("recipe/glm53xl/start-glm53xl-dspark.sh", "dspark-glm53xl/start-glm53xl-dspark.sh"),
        ("patches/hotfix-glm53xl-steering-projective.py", "dspark-glm53xl/patches/hotfix-glm53xl-steering-projective.py")],
    5: [("recipe/inkling/.env.inkling", "dspark-inkling/.env.inkling"),
        ("recipe/inkling/start-inkling-sm121.sh", "dspark-inkling/start-inkling-sm121.sh"),
        ("patches/hotfix-inkling-steering-projective.py", "dspark-inkling/hotfix-inkling-steering-projective.py"),
        ("patches/hotfix-inkling-gb10-load-reclaim.py", "dspark-inkling/hotfix-inkling-gb10-load-reclaim.py"),
        ("patches/hotfix-inkling-sm121-relattn.py", "dspark-inkling/hotfix-inkling-sm121-relattn.py"),
        ("recipe/inkling/files/fa4_rel_attention-sm121.py", "dspark-inkling/files/fa4_rel_attention-sm121.py"),
        ("recipe/inkling/files/inkling-model-gb10.py", "dspark-inkling/files/inkling-model-gb10.py"),
        ("recipe/inkling/files/inkling-model-gb10-steered.py", "dspark-inkling/files/inkling-model-gb10-steered.py")],
    6: [("recipe/glm53tp2/.env.glm53tp2", "dspark-glm53tp2/.env.glm53tp2"),
        ("recipe/glm53tp2/start-glm53-flash-tp2.sh", "dspark-glm53tp2/start-glm53-flash-tp2.sh"),
        ("patches/hotfix-glm53-steering-projective.py", "dspark-glm53tp2/patches/hotfix-glm53-steering-projective.py"),
        ("patches/vendor/sparse_attn_indexer_kpool_sm121.py", "dspark-glm53tp2/patches/sparse_attn_indexer_kpool_sm121.py")],
    7: [("recipe/nemotron35/.env.nemotron35", "nemotron35-glp/recipe/nemotron35/.env.nemotron35"),
        ("recipe/nemotron35/serve-nemotron35.sh", "nemotron35-glp/recipe/nemotron35/serve-nemotron35.sh"),
        ("patches/hotfix-nemotron35-steering-projective.py", "nemotron35-glp/patches/hotfix-nemotron35-steering-projective.py")],
}
CONTAINER_GREP = {0: "deepseek", 1: "qwen38", 2: "qwen38fn", 3: "glm53", 4: "glm5xl",
                  5: "inkling-sm121", 6: "glm53tp2", 7: "nemotron35"}


def current_lanes(output):
    """(lane index, exact container name) pairs from docker ps --format Names.
    Longest match wins: qwen38fn and glm53tp2 must not match shorter lanes."""
    running = []
    for name in output.splitlines():
        name = name.strip()
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", name):
            continue
        for idx in sorted(CONTAINER_GREP, key=lambda i: len(CONTAINER_GREP[i]), reverse=True):
            if CONTAINER_GREP[idx] in name.lower():
                running.append((idx, name))
                break
    return running


def detect_current_lanes(values, ssh_host, worker=None):
    if DEMO:
        return []
    argv = node_command(values, ssh_host, "docker ps --format '{{.Names}}'", worker)
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"cannot detect current lane on {worker or ssh_host} (exit {r.returncode})")
    return current_lanes(r.stdout)


def park_commands(lane_idx, running, values, ssh_host=None, worker=None):
    """Only explicitly observed containers belonging to other lanes are parked."""
    if DEMO:
        return []
    return [(f"park {name} on {worker or ssh_host}",
             node_command(values, ssh_host, shlex.join(["docker", "rm", "-f", name]), worker))
            for idx, name in running if idx != lane_idx]


def park_other_lanes(io, lane_idx, values, ssh_host):
    """Collect the old lane's ranks before one explicit, default-no confirmation.
    A failed probe, declined park or failed removal always blocks the boot."""
    if DEMO:
        return True
    if LANES[lane_idx].get("cloud"):
        return True  # cloud lanes deploy to Modal; nothing to park on the rig
    try:
        running = detect_current_lanes(values, ssh_host)
        old_lanes = {idx for idx, _ in running if idx != lane_idx}
        cmds = park_commands(lane_idx, running, values, ssh_host)
        # Also inspect the selected workers to catch a surviving old rank.
        workers = set(lane_workers(lane_idx, lane_env(lane_idx, values)))
        checked_lanes, checked_workers = set(), set()
        while True:
            for idx in sorted(old_lanes - checked_lanes):
                env = lane_env(idx, values)
                for n, worker in enumerate(lane_workers(idx, env), 1):
                    if not worker or "<" in worker:
                        worker = io.text(f"Old lane {CONTAINER_GREP[idx]} worker {n} fabric IP: ", "")
                    if not worker or "<" in worker:
                        io.err("Old lane worker address is missing — boot aborted")
                        return False
                    workers.add(worker)
                checked_lanes.add(idx)
            pending = workers - checked_workers
            if not pending:
                break
            for worker in sorted(pending):
                if not worker or "<" in worker:
                    io.err("Worker address is missing — boot aborted")
                    return False
                ranks = detect_current_lanes(values, ssh_host, worker)
                old_lanes.update(idx for idx, _ in ranks if idx != lane_idx)
                cmds += park_commands(lane_idx, ranks, values, ssh_host, worker)
                checked_workers.add(worker)
    except (OSError, RuntimeError) as exc:
        io.err(str(exc))
        return False
    if not cmds:
        return True
    for idx in sorted(old_lanes):
        io.warn("Current lane: " + LANES[idx]["name"])
        io.info("To relaunch later (after parking the new lane): $ " +
                shlex.join(boot_command(idx, values, ssh_host)[1]))
    for desc, argv in cmds:
        io.info(desc + ": $ " + shlex.join(argv))
    if not io.confirm("Park these containers? This stops serving and removes them before boot.", False):
        io.warn("parking declined — boot aborted")
        return False
    for desc, argv in cmds:
        io.info(desc)
        rc = subprocess.call(argv)
        if rc:
            io.err(f"parking FAILED ({rc}) — boot aborted")
            return False
    return True


def remote_preflight(io, lane_idx, values, ssh_host):
    """Check the remote before touching it: is the container up, and do the
    deployed files match the local ones (md5)? Returns True when the remote
    is fully synced and running."""
    if DEMO:
        return False
    import hashlib
    user = values.get("user", os.environ.get("USER", ""))
    target = f"{user}@{ssh_host}"
    probe = (
        "docker ps --format '{{.Names}}' || exit 1; echo CONTAINER_STATUS_END; "
        + " ".join(f"md5sum {r} 2>/dev/null || echo MISSING {r};"
                   for _, r in DEPLOY_MAP[lane_idx])
    )
    io.info(f"$ ssh {target} '<container status + file checksums>'")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                        target, probe], capture_output=True, text=True)
    if r.returncode != 0:
        io.err(f"ssh failed ({r.returncode}) — cannot preflight")
        return False
    running = any(idx == lane_idx for idx, _ in current_lanes(r.stdout.split("CONTAINER_STATUS_END")[0]))
    remote_md5 = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "MISSING":
            remote_md5[parts[1]] = None
        elif len(parts) == 2 and len(parts[0]) == 32:
            remote_md5[parts[1]] = parts[0]
    deploy_map = list(DEPLOY_MAP[lane_idx])
    vp = vector_paths(lane_idx)
    if vp and os.path.exists(vp[0]):
        deploy_map.append((vp[0], vp[1]))
    all_match = True
    for local, remote in deploy_map:
        lp = local if os.path.isabs(local) else os.path.join(HERE, local)
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
    if DEMO:
        return ("demo: boot skipped", ["true"])
    return deploy_commands(lane_idx, values, ssh_host)[-1]


def probe_models(base):
    if DEMO:
        return [DEFAULT_MODEL], None
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=10) as r:
            data = json.load(r)
        return [m["id"] for m in data.get("data", [])], None
    except Exception as e:
        return None, str(e)


def probe_generation(base, model):
    """Readiness includes a completed answer, not just a live HTTP process."""
    if DEMO:
        return "pong", None
    payload = {"model": model, "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
               "max_tokens": 1024, "temperature": 0}
    try:
        request = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                         json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.load(response)
        content = (data["choices"][0]["message"].get("content") or "").strip()
        if not content:
            if data["choices"][0].get("finish_reason") == "length":
                return None, "generation exhausted its output budget before answering; retry with a larger budget"
            return None, "generation returned no answer"
        return content, None
    except Exception as exc:
        return None, str(exc)


def served_context(base, model):
    """Use the running server's window rather than its model's theoretical limit."""
    if DEMO:
        return None
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=10) as response:
            data = json.load(response)
        for entry in data.get("data", []):
            value = entry.get("max_model_len")
            if entry.get("id") == model and isinstance(value, int) and value > 0:
                return value
    except Exception:
        pass
    return None


def yaml_block(text, key, indent=0):
    """Locate an ordinary YAML mapping entry and its indented body."""
    match = re.search(rf"(?m)^{' ' * indent}{re.escape(key)}:[^\n]*(?:\n|$)", text)
    if not match:
        return None
    end = pos = match.end()
    for line in text[pos:].splitlines(keepends=True):
        content = line.strip() and not line.lstrip().startswith("#")
        if content and len(line) - len(line.lstrip(" ")) <= indent:
            break
        pos += len(line)
        if content:
            end = pos
    return match.start(), end


def url_host(host):
    """Bracket IPv6 addresses when embedding a host in a URL."""
    host = host or "localhost"
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def render_provider(head_host="localhost"):
    """Load all lanes and their per-model settings from the canonical template."""
    with open(os.path.join(HERE, "tests", "models.yml")) as f:
        text = f.read()
    providers = yaml_block(text, "providers")
    body = text[slice(*providers)] if providers else ""
    block = yaml_block(body, PROVIDER, 2)
    if not block:
        raise ValueError(f"tests/models.yml has no '{PROVIDER}' provider")
    return re.sub(r"(?m)^( +baseUrl:[^\n]*)",
                  lambda m: m.group(0).replace("localhost", url_host(head_host)),
                  body[slice(*block)])


def model_profile(model):
    """Read context, output and tools from the same template omp installs."""
    text = render_provider()
    entries = re.split(r"(?m)^      - id: +", text)[1:]
    for entry in entries:
        name, _, body = entry.partition("\n")
        if name.strip() != model:
            continue
        def integer(key, default):
            match = re.search(rf"(?m)^        {key}: +(\d+)", body)
            return int(match.group(1)) if match else default
        return {"context_length": integer("contextWindow", 65536),
                "max_tokens": integer("maxTokens", 8192),
                "supports_tools": not re.search(r"(?m)^        supportsTools: +false\b", body)}
    return {"context_length": 65536, "max_tokens": 8192, "supports_tools": True}


def install_provider(head_host="localhost", model=None, context_length=None, base_url=None):
    """Replace or append the complete provider, preserving other configuration."""
    if DEMO:
        return f"demo: configured omp '{PROVIDER}' provider (all local lanes)"
    block = render_provider(head_host)
    if model:
        pattern = rf"(?ms)^      - id: +{re.escape(model)}\n.*?(?=^      - id:|\Z)"
        def update_model(match):
            entry = match.group(0)
            if context_length:
                limit = min(model_profile(model)["max_tokens"], max(1, context_length // 4))
                for key, value in (("contextWindow", context_length), ("maxTokens", limit)):
                    entry = re.sub(rf"(?m)^(        {key}: +)\d+",
                                   lambda m: m.group(1) + str(value), entry)
            if base_url:
                entry = re.sub(r"(?m)^        baseUrl:.*$", lambda m: "        baseUrl: " + json.dumps(base_url), entry)
            return entry
        block = re.sub(pattern, update_model, block, count=1)
    existing = ""
    if os.path.exists(OMP_MODELS):
        with open(OMP_MODELS) as f:
            existing = f.read()
        shutil.copy(OMP_MODELS, OMP_MODELS + ".bak")
    providers = yaml_block(existing, "providers")
    if providers:
        start, end = providers
        body = existing[start:end]
        old = yaml_block(body, PROVIDER, 2)
        if old:
            body = body[:old[0]] + block + body[old[1]:]
        else:
            body = body.rstrip("\n") + "\n" + block
        text = existing[:start] + body + existing[end:]
    else:
        text = existing + ("\n" if existing and not existing.endswith("\n") else "")
        text += "providers:\n" + block
    os.makedirs(os.path.dirname(OMP_MODELS), exist_ok=True)
    with open(OMP_MODELS, "w") as f:
        f.write(text)
    return f"configured provider '{PROVIDER}' in {OMP_MODELS} (all local lanes)"


def install_hermes(head_host, model, context_length=None, base_url=None):
    """Update Hermes model settings, keeping unrelated keys and mappings."""
    if DEMO:
        return f"demo: configured hermes (model {model})"
    ids = re.findall(r"(?m)^      - id: +(\S+)", render_provider())
    header = HERMES_MARKER + "\n" + "".join(f"#   {model_id}\n" for model_id in ids)
    text = ""
    if os.path.exists(HERMES_CONFIG):
        with open(HERMES_CONFIG) as f:
            text = f.read()
        shutil.copy(HERMES_CONFIG, HERMES_CONFIG + ".bak")
    # Refresh our comment too, so newly added lanes show up on subsequent runs.
    text = re.sub(rf"(?m)^{re.escape(HERMES_MARKER)}\n(?:#   [^\n]*\n)*", "", text)
    block = yaml_block(text, "model")
    body = text[slice(*block)] if block else "model:\n"
    # Preserve the user's indentation and any extra model options.
    indents = re.findall(r"(?m)^( +)[^ #\n][^\n]*:", body)
    indent = min(map(len, indents)) if indents else 2
    profile = model_profile(model)
    context_length = context_length or profile["context_length"]
    settings = {
        "default": json.dumps(model),
        "provider": "custom",
        "base_url": json.dumps(base_url or f"http://{url_host(head_host)}:8000/v1"),
        "context_length": str(context_length),
        "max_tokens": str(min(profile["max_tokens"], max(1, context_length // 4))),
    }
    # A scalar or empty model entry becomes a mapping.
    if not re.match(r"^model:[ \t]*(?:#[^\n]*)?(?:\n|$)", body):
        body = re.sub(r"^model:[^\n]*(?:\n|$)", "model:\n", body, count=1)
    if not body.endswith("\n"):
        body += "\n"
    for key, value in settings.items():
        entry = " " * indent + f"{key}: {value}\n"
        old = yaml_block(body, key, indent)
        if old:
            body = body[:old[0]] + entry + body[old[1]:]
        else:
            body = body.rstrip("\n") + "\n" + entry
    if block:
        text = text[:block[0]] + header + body + text[block[1]:]
    else:
        text += ("\n" if text and not text.endswith("\n") else "") + header + body
    os.makedirs(os.path.dirname(HERMES_CONFIG), exist_ok=True)
    with open(HERMES_CONFIG, "w") as f:
        f.write(text)
    return f"configured hermes (model {model}) in {HERMES_CONFIG}"


# omp roles we route to the endpoint when the user picks "all text roles".
# vision stays untouched: the endpoint serves text-only models.
OMP_EXTRA_ROLES = ["smol", "slow", "plan", "task", "commit", "tiny",
                   "advisor", "designer"]


def read_omp_model_roles():
    """omp's modelRoles map from ~/.omp/agent/config.yml ({} if unset)."""
    try:
        with open(OMP_CONFIG) as f:
            text = f.read()
    except OSError:
        return {}
    block = re.search(r"(?m)^modelRoles:\s*\n(?:[ \t]+\S.*\n?)*", text)
    if not block:
        return {}
    return {m.group(1): m.group(2)
            for m in re.finditer(r"(?m)^\s+(\w+):\s*(\S+)", block.group(0))}


def set_omp_model_roles(model, roles):
    """Set modelRoles.<role> for each role in ~/.omp/agent/config.yml,
    preserving the rest of the file (including other roles)."""
    ref = f"{PROVIDER}/{model}"
    if DEMO:
        return f"demo: omp roles {', '.join(roles)} → '{ref}'"
    try:
        with open(OMP_CONFIG) as f:
            text = f.read()
    except OSError:
        text = ""
    block = re.search(r"(?m)^modelRoles:\s*\n(?:[ \t]+\S.*\n?)*", text)
    body = block.group(0) if block else "modelRoles:\n"
    for role in roles:
        if re.search(rf"(?m)^\s+{role}:", body):
            body = re.sub(rf"(?m)^(\s+){role}:.*",
                          lambda m: f"{m.group(1)}{role}: {ref}", body)
        else:
            body += f"  {role}: {ref}\n"
    text = (text[:block.start()] + body + text[block.end():]) if block \
        else (text + ("" if not text or text.endswith("\n") else "\n") + body)
    if os.path.exists(OMP_CONFIG):
        shutil.copy(OMP_CONFIG, OMP_CONFIG + ".bak")
    os.makedirs(os.path.dirname(OMP_CONFIG), exist_ok=True)
    with open(OMP_CONFIG, "w") as f:
        f.write(text)
    return f"omp roles {', '.join(roles)} → '{ref}' ({OMP_CONFIG})"


def prereqs():
    return shutil.which("bun"), shutil.which("omp")


def run_suite(io, base, model):
    """Run tests/0*.sh one by one, inside the wizard UI: each test's verdict
    line lands as a colored ✓/~/✗ row instead of dropping out to a shell."""
    import glob
    tests = sorted(glob.glob(os.path.join(HERE, "tests", "0*.sh")))
    io.info("")
    io.header("endpoint test suite")
    io.info("─" * 40)
    if DEMO:
        canned = {
            "01-endpoint.sh": f"PASS: {model} listed at {base}",
            "02-chat.sh": "PASS: chat completion returned: pong",
            "03-tool-call.sh": 'PASS: tool call get_weather({"city": "Paris"})',
            "04-omp-headless.sh": f"PASS: omp agent loop created omp_probe.txt via {PROVIDER}/{model}",
        }
        for t in tests:
            name = os.path.basename(t)
            io.begin(f"… {name} running")
            io.end(f"✓ {name} — {canned.get(name, 'PASS: demo')}", "ok")
        io.info("─" * 40)
        io.ok("all endpoint tests passed")
        return 0
    env = dict(os.environ, WEIGHTLESS_BASE_URL=base, WEIGHTLESS_MODEL=model,
               WEIGHTLESS_OMP_MODEL=f"{PROVIDER}/{model}")
    fails = 0
    for t in tests:
        name = os.path.basename(t)
        io.begin(f"… {name} running")
        try:
            r = subprocess.run(["bash", t], env=env, capture_output=True,
                               text=True, timeout=600)
        except subprocess.TimeoutExpired:
            io.end(f"✗ {name} — timed out after 600s", "err")
            fails += 1
            continue
        lines = (r.stdout + r.stderr).strip().splitlines()
        # the test's own verdict line, not tool chatter interleaved on stdout
        verdict = next((l for l in lines if l.startswith(("PASS:", "FAIL:", "SKIP:"))),
                       lines[-1] if lines else "(no output)")
        if r.returncode == 0:
            io.end(f"✓ {name} — {verdict}", "ok")
        elif r.returncode == 2:
            io.end(f"~ {name} — {verdict}", "warn")
        else:
            io.end(f"✗ {name} — {verdict}", "err")
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
    C = {"head": "\033[1;38;5;45m", "ok": "\033[38;5;80m", "err": "\033[31m",
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

    def begin(self, msg):
        print(f"  {msg}", end="", flush=True)
        self._begin_len = len(msg) + 2

    def end(self, msg, kind="ok"):
        pad = " " * max(0, self._begin_len - len(msg) - 2)
        print(f"\r  {self._c(kind, msg)}{pad}")

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
        enter = "Enter = yes" if default else "Enter = no"
        try:
            s = input(f"{prompt} [{hint} — {enter}]: ").strip().lower()
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
            # semantic palette on the brand gradient: turquoise headers, teal ok,
            # red err, yellow warn, white-on-violet selection
            brand = (45, 80, curses.COLOR_RED, curses.COLOR_YELLOW)
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

    def repaint(self):
        """Authoritative full redraw from the stdscr buffer. Scroll-heavy
        sessions can desync the physical screen (a terminal that drops or
        mis-handles scroll-up leaves half-erased boxes behind); clearok forces
        doupdate to repaint every cell from the buffer, ending clean."""
        self.s.clearok(True)
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
        enter = "Enter = yes" if default else "Enter = no"
        self._w(r, 0, f"{prompt} ", self._attr("head"))
        self._w(r, len(prompt) + 1, f"[{hint} — {enter}] ", self._attr("dim"))
        self.s.refresh()
        ch = self.s.get_wch()
        return ch.lower().startswith("y") if isinstance(ch, str) and ch.strip() else default

    def begin(self, msg):
        self._begin_row = self._next()
        self._w(self._begin_row, 0, f"  {msg}")
        self.s.refresh()

    def end(self, msg, kind="ok"):
        max_x = self.s.getmaxyx()[1]
        self._w(self._begin_row, 0, " " * (max_x - 1))
        self._w(self._begin_row, 0, f"  {msg}", self._attr(kind))
        self.s.refresh()

    def _draw_box(self, row, col, width, title, lines, border, title_kind):
        """Frame + title + body drawn straight onto stdscr. A floating newwin
        gets erased by the next stdscr scroll (its rows repaint as blanks), so
        boxes live in the main buffer like everything else."""
        battr = self._attr(border)
        self._w(row, col, "╭" + "─" * (width - 2) + "╮", battr)
        for i in range(1, len(lines) + 1):
            self._w(row + i, col, "│", battr)
            self._w(row + i, col + width - 1, "│", battr)
        self._w(row + len(lines) + 1, col, "╰" + "─" * (width - 2) + "╯", battr)
        self._w(row, col + 2, f" {title} ", self._attr(title_kind))
        for i, l in enumerate(lines):
            self._w(row + 1 + i, col + 2, l[: width - 4])
        self.s.refresh()

    def tui_box(self, title, lines, border="dim", title_kind="head"):
        max_y, max_x = self.s.getmaxyx()
        width = min(max(len(title) + 2, *(len(l) for l in lines)) + 4, max_x - 4)
        row = self._next(len(lines) + 2)
        self._draw_box(row, 2, width, title, lines, border, title_kind)

    def animate_logo(self):
        """Slide the gradient continuously. The palette is the ramp followed by
        itself reversed (pink→…→cyan→…→pink), so rotation never jumps: column
        colors run 1 2 3 4 3 2 1 through the palette instead of 1 2 3 4 1."""
        if not self.logo_pos or not self.grad:
            return
        row0, col0 = self.logo_pos
        self._tick += 1
        if not hasattr(self, "_grad_cycle"):
            self._grad_cycle = self.grad + self.grad[-2:0:-1]
        pal = self._grad_cycle
        m = len(pal)
        for i, art in enumerate(LOGO):
            for j, ch in enumerate(art):
                if ch != " ":
                    self._w(row0 + i, col0 + j, ch, pal[(j + self._tick) % m])
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
                if ch == 27:  # lone Esc or the start of an arrow sequence
                    # the idle timeout splits escape sequences — reassemble
                    self.s.nodelay(True)
                    c2 = self.s.getch()
                    c3 = self.s.getch() if c2 != -1 else -1
                    self.s.nodelay(False)
                    if c2 == ord("[") and c3 in (ord("A"), ord("B")):
                        ch = curses.KEY_UP if c3 == ord("A") else curses.KEY_DOWN
                    else:
                        continue  # actual Esc — ignore in menus
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

def cloud_chain(io, lane_idx):
    """Modal cloud lane: deploy, then return the endpoint — that's the whole
    deliverable. No rig access, no parking, no client rewiring (the user can
    point omp/Hermes at the URL afterwards if they want)."""
    lane = LANES[lane_idx]
    app_path = os.path.join(HERE, lane["modal_app"])
    weights_note = lane.get("cloud_weights", "~465 GB")
    boot_note = lane.get("cloud_boot", "30 min")
    io.header(lane["name"])
    io.info("─" * 60)
    steps = [
        ("check Modal auth", ["modal", "profile", "current"]),
        (f"ensure weights on the volume (idempotent, {weights_note} first time)",
         ["modal", "run", "--detach", app_path + "::ensure_weights"]),
        ("verify the GLP directions on the volume (never re-derives)",
         ["modal", "run", app_path + "::ensure_dirs"]),
        ("deploy the serving app", ["modal", "deploy", app_path]),
    ]
    if DEMO:
        io.info("demo: cloud lane plan printed; nothing executed")
    for desc, argv in steps:
        io.info("$ " + shlex.join(argv))
        if DEMO:
            continue
        if not io.confirm(f"run: {desc}?", True):
            io.warn("aborted — re-run the wizard to continue")
            return
        if desc.startswith("ensure weights"):
            io.info("detached; first download takes a while on Modal "
                    "bandwidth — `modal app logs` to watch")
            continue
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            io.err(f"FAILED ({r.returncode}): {(r.stderr or r.stdout)[-400:]}")
            return
        if desc.startswith("check Modal"):
            io.ok(r.stdout.strip())
        if desc.startswith("deploy"):
            m = re.search(r"https://\S+\.modal\.(?:run|direct)",
                          r.stdout + r.stderr)
            if m:
                base = m.group(0).rstrip("/")
                io.ok(f"deployed: {base}")
                io.info("OpenAI-compatible endpoint: " + base + "/v1")
                io.warn(f"cold start loads {weights_note} — first request can "
                        f"take up to {boot_note}; the endpoint 503s while "
                        "booting")
            else:
                io.warn("deployed, but the endpoint URL was not in the "
                        "output — check `modal app list`")
    if DEMO:
        return 0


def lane_chain(io, lane_idx):
    lane = LANES[lane_idx]
    if lane.get("cloud"):
        return cloud_chain(io, lane_idx)
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
        if p == "head-ip" or re.fullmatch(r"worker\d*-ip", p):
            values[p] = pick_host(io, f"{prompt} ({p})", default)
        else:
            if p == "hf-cache":
                default = f"/home/{values.get('user', saved.get('user', ''))}/.cache/huggingface"
            values[p] = io.text(f"{prompt} ({p}): ", default)

    # Only expose steering when the selected launcher actually applies it.
    steering = False
    if lane.get("steering_supported", True):
        steering = io.confirm("Enable refusal steering (GLP vector patch)?", True)
    else:
        io.info("The selected launcher does not support steering.")
    steer_mode = None
    if steering and lane["steer_modes"]:
        labels = [d for _, d in lane["steer_modes"]]
        sel = io.menu("Steering mode:", labels)
        steer_mode = lane["steer_modes"][sel][0]

    # 3. write env
    text = render_env(example, values, steer_mode=steer_mode,
                      steering=steering, steer_key=lane["steer_key"])
    if DEMO:
        io.info("demo: env generation, steering validation and deployment skipped")
        prepare_assets(io, lane_idx, values, None)
        return tests_chain(io)
    env_errors, env_warnings = validate_lane_env(lane, text)
    for w in env_warnings:
        io.warn(w)
    for e in env_errors:
        io.warn("hardware fit: " + e)
    if env_errors and not io.confirm(
            "Env fails hardware-fit checks for this lane — write anyway?",
            False):
        io.warn("not written — fix the flagged values and re-run")
        return
    changed = any(saved.get(k) != v for k, v in values.items() if k in saved) \
              or any(k not in saved for k in values)
    if os.path.exists(target):
        with open(target) as f:
            existing = f.read()
        m = re.search(rf"(?m)^{re.escape(lane['steer_key'])}=(\S*)", existing)
        if bool(m and m.group(1)) != steering:
            changed = True
        m2 = re.search(r"(?m)^STEER_MODE=(\w+)", existing)
        if steer_mode and m2 and m2.group(1) != steer_mode:
            changed = True
    if os.path.exists(target):
        if not changed:
            io.info("env unchanged — nothing to write")
        elif io.confirm(f"{target} exists — apply your changes?", True):
            with open(target, "w") as f:
                f.write(text)
            io.ok(f"wrote {target}")
        else:
            io.warn("your edits were NOT saved — the file on disk still has "
                    "the old values (it will prefill them again next run)")
    else:
        with open(target, "w") as f:
            f.write(text)
        io.ok(f"wrote {target}")

    # 4. validate the steering patch (assets are prepared on the nodes below)
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
            return 1
    # 5. deploy (confirm-gated remote actions, preflight first)
    ssh_host = None
    if io.confirm("Deploy to the node(s) over ssh now?", True):
        # MASTER_ADDR/head-ip is the RoCE fabric address — not routable from
        # the LAN. ssh needs a reachable host: the omp provider's by default.
        omp_host = urllib.parse.urlparse(default_base()).hostname
        note = ("Head node ssh host" if LANES[lane_idx].get("nodes", 1) > 1
                else "Node ssh host")
        ssh_host = pick_host(io, note + " (fabric IPs are not routable)",
                             omp_host or values.get("head-ip", ""))
        if remote_preflight(io, lane_idx, values, ssh_host):
            io.ok("remote is in sync and running — nothing to deploy")
            if not io.confirm("Redeploy and restart anyway?", False):
                io.info("deploy skipped")
                return tests_chain(io, ssh_host, lane_idx)
        if not prepare_assets(io, lane_idx, values, ssh_host):
            return 1
        cmds = deploy_commands(lane_idx, values, ssh_host)
        for desc, argv in cmds:
            io.info(f"$ {' '.join(argv)}")
            if not io.confirm(f"run: {desc}?", True):
                io.warn("skipped")
                continue
            if (desc, argv) == cmds[-1] and not park_other_lanes(io, lane_idx, values, ssh_host):
                return 1
            rc = subprocess.call(argv)
            if rc != 0:
                io.err(f"FAILED ({rc}) — fix and re-run; aborting deploy")
                return rc
    else:
        io.info("deploy skipped (declined) — nothing was checked remotely. Answer y to "
                "preflight first: checksums + container status before anything is touched.")

    # 6. endpoint tests + agent clients
    return tests_chain(io, ssh_host or values.get("head-ip"), lane_idx)


def diagnose_chain(io, base=None):
    """Layered failure isolation for an endpoint that won't answer, with an
    optional remote check/boot over ssh."""
    if DEMO:
        io.info("demo: DNS, TCP and HTTP diagnosis skipped")
        return 0
    base = base or io.text("Base URL to diagnose: ", default_base())
    base = normalize_base(base)
    u = urllib.parse.urlparse(base)
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
    ids, err = probe_models(base)
    if ids is None:
        io.err(f"HTTP: /v1/models failed ({err})")
        io.info("something listens but it is not a healthy OpenAI server — check container logs")
        remote_diagnose(io, host)
        return 1
    io.ok(f"HTTP: /v1/models answers — serving: {', '.join(ids)}")
    if not ids:
        io.err("No model is ready. A running router alone is not a serving engine.")
        return 1
    model = ids[io.menu("Model to diagnose:", ids)] if len(ids) > 1 else ids[0]
    io.info("Checking generation (up to 180 seconds)...")
    content, error = probe_generation(base, model)
    if error:
        io.err(f"Generation failed: {error}. Inspect worker logs even if /health passes.")
        return 1
    io.ok(f"Generation: {content[:80]}")
    return 0


def remote_diagnose(io, host):
    """ssh to the serving node: container status, GPU, offer to boot."""
    if DEMO:
        io.info("demo: remote diagnosis and boot skipped")
        return
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
             "| grep -i -E 'deepseek|qwen|vllm|inkling|glm' || echo '(no serving container)'; "
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
        values = dict(saved, user=target.split("@")[0])
        desc, argv = boot_command(lane_idx, values, ssh_host=ssh_host)
        io.info(f"$ {' '.join(argv)}")
        if io.confirm(f"run: {desc}?", True):
            if not park_other_lanes(io, lane_idx, values, ssh_host):
                return 1
            rc = subprocess.call(argv)
            if rc == 0:
                io.ok("boot issued — give it a few minutes to load the model")
            else:
                io.err(f"boot FAILED ({rc})")


def normalize_base(base):
    base = base.strip().rstrip("/")
    if "://" not in base:
        base = "http://" + base
    if not urllib.parse.urlparse(base).path:
        base += "/v1"
    return base


def tests_chain(io, head_host=None, lane_idx=None):
    preferred_model = None
    if lane_idx is not None:
        lane = LANES[lane_idx]
        with open(os.path.join(HERE, lane["target"])) as f:
            env = dict(re.findall(r"(?m)^([A-Z_]+)=(\S+)", f.read()))
        head_host = head_host or env.get("MASTER_ADDR") or "localhost"
        preferred_model = env.get("SERVED_MODEL_NAME", "").strip("\"'")
        port = env.get('VLLM_PORT') or lane['port']
        endpoint = f"http://{url_host(head_host)}:{port}/v1"
        if preferred_model == "inkling-small-nvfp4":
            router_base = f"http://{url_host(head_host)}:8000/v1"
            router_ids, _ = probe_models(router_base)
            if preferred_model in (router_ids or []):
                endpoint = router_base
    else:
        endpoint = default_base()
    base = normalize_base(io.text("Base URL of the OpenAI-compatible server: ", endpoint))
    io.info(f"probing {base}/models ...")
    ids, err = probe_models(base)
    while ids is None:
        io.err(f"probe failed: {err}")
        choice = io.menu("endpoint not answering:", [
            "Diagnose the connection (DNS → TCP → HTTP → ssh)",
            "Enter a different URL",
            "Skip agent setup (endpoint must answer first)",
        ])
        if choice == 0:
            diagnose_chain(io, base)
            io.info(f"re-probing {base}/models ...")
            ids, err = probe_models(base)
            if ids is not None:
                io.ok("endpoint is up now")
        elif choice == 1:
            base = normalize_base(io.text("Base URL: ", base))
            io.info(f"probing {base}/models ...")
            ids, err = probe_models(base)
        else:
            return 1
    if not ids:
        io.err("endpoint returned no models — agent setup skipped")
        return 1
    if preferred_model and not DEMO:
        if preferred_model not in ids:
            io.err(f"endpoint is not serving {preferred_model} — agent setup skipped")
            return 1
        model = preferred_model
    else:
        model = ids[io.menu("Model to test:", ids)] if len(ids) > 1 else ids[0]
    io.ok(f"model: {model}")
    io.info("Testing a completed answer (first request may need kernel warmup)...")
    content, error = probe_generation(base, model)
    if error:
        io.err(f"generation failed: {error} — client defaults were not changed")
        return 1
    io.ok(f"generation: {content[:80]}")
    context_length = served_context(base, model) or model_profile(model)["context_length"]
    io.info(f"Serving context: {context_length:,} tokens")
    head_host = urllib.parse.urlparse(base).hostname or head_host or "localhost"
    configure_omp = io.confirm("Configure omp (weightless provider)?", True)
    configure_hermes = io.confirm("Configure hermes (~/.hermes/config.yaml)?", True)
    if configure_omp:
        io.ok(install_provider(head_host, model=model, context_length=context_length, base_url=base))
        omp_roles_chain(io, model)
    if configure_hermes:
        io.ok(install_hermes(head_host, model, context_length=context_length, base_url=base))
        io.info("Hermes configuration updated on this machine. Restart an existing gateway; saved /model session overrides take precedence.")
    if io.confirm("Run the test suite now?", True):
        return run_suite(io, base, model)
    io.info(f"done — later: sh {os.path.join(HERE, 'tests', 'run.sh')}")
    return 0


def omp_roles_chain(io, model):
    ref = f"{PROVIDER}/{model}"
    configured = read_omp_model_roles()
    missing = [r for r in ["default"] + OMP_EXTRA_ROLES
               if configured.get(r) != ref]
    if not missing:
        io.ok(f"omp already routes all text roles to '{ref}'")
    elif configured.get("default") == ref:
        if io.confirm(f"also route {'/'.join(missing)} "
                      f"(sub-agents, planning, commits) to '{ref}'?", True):
            io.ok(set_omp_model_roles(model, missing))
    else:
        scope = io.menu(f"Route '{ref}' in omp:", [
            "default role only — the main session model",
            "all text roles — default + smol/slow/plan/task/commit/tiny/"
            "advisor/designer (vision untouched)",
        ], preselect=1)
        roles = ["default"] if scope == 0 else missing
        io.ok(set_omp_model_roles(model, roles))


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

    io._draw_box(brow, bcol, bw, "local setup", lines, "dim", "head")

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
    lane_items = [
        l["name"].split(" — ")[0] + " — full chain (env → assets → deploy → clients/tests)"
        for l in LANES
    ]
    choice = io.menu("What to set up:", idle=getattr(io, "animate_logo", None), items=[
        *lane_items,
        "Agent clients — configure omp / hermes + endpoint smoke suite",
        "Diagnose endpoint — layered checks + remote container status",
    ])
    if choice < len(LANES):
        return lane_chain(io, choice)
    if choice == len(LANES):
        return tests_chain(io)
    return diagnose_chain(io)


def completion(io, rc):
    """End-of-run summary. A boxed 'done' on success, a pointer on failure."""
    io.info("")
    if rc == 0:
        box(io, "Setup complete", [
            "Selected setup steps completed.",
            "Check the test results above for skips and limitations.",
        ], border="ok", title_kind="head")
    else:
        io.warn("setup finished with errors — scroll up for the red rows")


def _tui_main(stdscr):
    io = TuiIO(stdscr)
    io.header("  weightless setup")
    io._put(1, 0, "  lean abliteration steering, served", "dim")
    io._put(2, 0, "  by Matt Suiche (@msuiche)", "dim")
    io.row = 4
    splash_tui(io)
    rc = run(io)
    if not curses.isendwin():
        completion(io, rc)
        io.info("")
        io.info("press q or Esc to exit")
        io.repaint()
        while True:
            ch = stdscr.getch()  # deliberate exit only — Enter must not dismiss
            if ch in (ord("q"), ord("Q"), 27):
                break
    return rc


def main():
    if sys.stdout.isatty() and sys.stdin.isatty() and curses is not None:
        try:
            return curses.wrapper(_tui_main)
        except Exception as e:
            print(f"(TUI failed: {e} — falling back to prompts)", file=sys.stderr)
    io = CliIO()
    io.header("== weightless setup ==")
    io.info("by Matt Suiche (@msuiche)")
    io.info("")
    splash_cli(io)
    io.info("")
    rc = run(io)
    completion(io, rc)
    return rc


if __name__ == "__main__":
    sys.exit(main())
