"""GLM-5.3-Flash + the weightless-steer vLLM plugin: first real-GPU boot test.

The plugin (vllm-plugin/, dist `weightless-steer`) shadows Glm5NextForCausalLM
in vLLM's ModelRegistry through the `vllm.general_plugins` entry point — no
in-container file patching, unlike the hotfix lane this test is compared
against (patches/hotfix-glm53-steering-projective.py, reference numbers in
BENCHMARK.md's GLP-44 section). This app boots the day-0 serving image with
the plugin pip-installed and the published GLP-44 vector, and exposes an
OpenAI-compatible endpoint for the local eval driver
(modal/eval_glm53_plugin.py).

Stack (all validated facts, see recipe/glm53/README.md and
refusal-research/experiments/20260830-glp44-dflash2-acceptance):
  - Image: vllm/vllm-openai:glm53-flash-x86_64-cu130 (vLLM 0.1.dev20051 + PR
    #53906; ships vllm/models/glm5next/nvidia/model.py, the module the
    adapter imports).
  - Model: RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-tensors W4A4, 10 shards +
    MTP shard, ~198 GB; NEVER the LibertAIDAI ModelOpt quant — corrupted
    token IDs).
  - Shape: H100:4 single container, TP4 (~50 GiB weights/rank). Fallback:
    H200:2 TP2 (GLM53_GPU=H200:2 GLM53_TP=2).
  - Vector: msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44 (gated, hf-token
    secret), GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf. alpha=2.0
    is calibrated; alpha >= 2.5 garbles this model.

Two arms, one deploy each (alpha is a model-init buffer; no runtime override):
    modal run cloud_serve_glm53.py::ensure_vector    # CPU, seconds
    modal run cloud_serve_glm53.py::ensure_weights   # CPU, ~198 GB once
    modal run cloud_serve_glm53.py::preflight        # CPU, entry point + gguf
    WEIGHTLESS_STEER_ALPHA=0.0 modal deploy cloud_serve_glm53.py
    python3 modal/eval_glm53_plugin.py --base-url <url> --alpha 0.0
    modal app stop weightless-glm53-plugin-test
    WEIGHTLESS_STEER_ALPHA=2.0 modal deploy cloud_serve_glm53.py
    python3 modal/eval_glm53_plugin.py --base-url <url> --alpha 2.0
    modal app stop weightless-glm53-plugin-test

First boot runs WITHOUT --enforce-eager on purpose: the adapter's
torch.compile rebind fix is part of what is being tested. If output is
garbage or the steering-active line is missing in compiled mode, redeploy
with GLM53_ENFORCE_EAGER=1 to isolate.

Workaround for this dev build, documented per the 2026-09-18 debug session:
the plugin's register() DOES run via vllm's load_general_plugins (CPU repro
of the api_server arg phase shadows the registry), but its INFO log lines
never reach the serve log — vllm's dictConfig only attaches the `vllm`
logger (propagate=False), third-party INFO falls through to a WARNING root,
and even a VLLM_LOGGING_CONFIG_PATH root-INFO override did not surface them
in the real boot. serve() therefore prepends /opt/weightless-shim to the
vllm process's PYTHONPATH: modal/sitecustomize.py auto-imports at EVERY
python startup, runs register() itself, pre-parses the vector through
SteeringCore.from_env (fail-closed), and prints WEIGHTLESS-SHIM markers
straight to stderr — boot evidence that no logging config can eat.

Cost discipline: max_containers=1, scaledown_window=180, stop the app when
done. Expected spend: 2 boots x (load + eval) x 4 H100.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]           # weightless/
SPARK = ROOT.parent                                  # spark workspace
MODEL_ID = "RedHatAI/GLM-5.3-Flash-NVFP4"
SERVED_MODEL = "glm53-flash"
VOLUME_NAME = "glm53-flash"
VECTOR_REPO = "msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44"
VECTOR_FILE = "GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf"
VECTOR_PATH = "/data/vector/" + VECTOR_FILE
IMAGE = "vllm/vllm-openai:glm53-flash-x86_64-cu130"
# tonyd2wild's mm chat template (vendored by the 20260830 4xH100 lane): the
# checkpoint's shipped chat_template.jinja has no enable_thinking branch and
# would always open <think>, eating the 400-token eval budget. The reference
# protocol renders with enable_thinking=False.
CHAT_TEMPLATE = (SPARK / "refusal-research" / "experiments"
                 / "20260830-glp44-dflash2-acceptance" / "chat_template_mm.jinja")
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
PY = "/usr/bin/python3.12"

# Arm/shape switches, read at deploy time (same convention as cloud_serve_k3).
GPU = os.environ.get("GLM53_GPU", "H100:4")
TP = os.environ.get("GLM53_TP", "4")
KV_DTYPE = os.environ.get("GLM53_KV_DTYPE", "fp8_e4m3")   # "" to drop the flag
ENFORCE_EAGER = os.environ.get("GLM53_ENFORCE_EAGER", "")  # "1" to isolate
GMU = os.environ.get("GLM53_GMU", "0.92")
# gmu 0.85 OOMs in profile_cudagraph_memory (44.96 GiB weights/rank + mm
# encoder cache + FULL_AND_PIECEWISE capture don't fit in the 67 GiB budget);
# 0.92 is the headroom fix that KEEPS cudagraph mode. --enforce-eager remains
# the isolation fallback (both reference lanes for this image ran eager).
ALPHA = os.environ.get("WEIGHTLESS_STEER_ALPHA", "0.0")

app = modal.App("weightless-glm53-plugin-test")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

download_image = (modal.Image.debian_slim(python_version="3.12")
                  .pip_install("huggingface_hub", "hf_transfer"))

# The plugin installs from a copied source tree (the user edits vllm-plugin/
# concurrently; whatever is on disk at deploy time is what boots). The
# pyproject packages.find looks in [".", ".."] for weightless_steer* AND
# weightless_runtime*, so the runtime package sits NEXT TO the plugin dir in
# the image. Install with the image's own python — the one vllm serve runs
# under — NOT modal's add_python shim, or the entry point lands in the wrong
# site-packages.
image = (modal.Image.from_registry(IMAGE, add_python="3.12",
                                   setup_dockerfile_commands=["ENTRYPOINT []"])
         .add_local_dir(ROOT / "vllm-plugin", "/opt/weightless-src/vllm-plugin",
                        copy=True)
         .add_local_dir(ROOT / "weightless_runtime",
                        "/opt/weightless-src/weightless_runtime", copy=True)
         .add_local_file(CHAT_TEMPLATE, "/work/chat_template_mm.jinja",
                         copy=True)
         .add_local_file(ROOT / "modal" / "vllm_logging_config.json",
                         "/work/vllm_logging_config.json", copy=True)
         .add_local_file(ROOT / "modal" / "sitecustomize.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         .run_commands(
             f"{PY} -m pip --version || {PY} -m ensurepip --upgrade",
             "cd /opt/weightless-src/vllm-plugin && "
             f"{PY} -m pip install --no-deps ."))


@app.function(image=download_image, volumes={"/data": vol},
              timeout=24 * 3600, secrets=[hf_secret], env=ENV)
def ensure_weights():
    """Resume/cache the ~198 GB NVFP4 snapshot on CPU; no GPU allocation.

    Fail-closed size gate (same discipline as cloud_serve_k3.ensure_weights):
    11 safetensors (10 model shards + model_mtp) + index, zero .incomplete
    blobs, >190 GB before any GPU spend. Repo is ungated; the secret rides
    along for rate limits only."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID)
    snap = Path(snapshot)
    shards = sorted(snap.glob("*.safetensors"))
    assert (snap / "model.safetensors.index.json").is_file(), "index missing"
    hub = (Path(os.environ["HF_HOME"]) / "hub"
           / "models--RedHatAI--GLM-5.3-Flash-NVFP4")
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    print(f"shards: {len(shards)}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e9:.2f} GB", flush=True)
    assert len(shards) == 11, f"expected 11 safetensors, found {len(shards)}"
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 190e9, f"size-gate failed: {total / 1e9:.1f} GB"
    vol.commit()
    print("weights ready:", snapshot, flush=True)


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              secrets=[hf_secret], env=ENV)
def ensure_vector():
    """Fetch the gated GLP-44 GGUF and stage it at the stable VECTOR_PATH.

    The HF-cache snapshot path embeds a revision hash; the server env needs a
    path known at deploy time, so the file is copied out of the cache (same
    pattern as cloud_serve_k3.ensure_dirs). Fail-closed: <2 MB means a
    truncated or LFS-pointer download (real file is ~2.9 MB)."""
    from huggingface_hub import hf_hub_download

    dst = Path(VECTOR_PATH)
    if not dst.is_file():
        fetched = hf_hub_download(repo_id=VECTOR_REPO, filename=VECTOR_FILE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(fetched).read_bytes())
        vol.commit()
    size = dst.stat().st_size
    assert size > 2e6, f"{VECTOR_PATH} looks truncated ({size} B)"
    print(f"vector ready: {VECTOR_PATH} ({size} B)", flush=True)


_PREFLIGHT_RESOLVE = r"""
import os, sys
os.environ["WEIGHTLESS_STEER_PATH"] = sys.argv[1]   # sentinel or real gguf
from vllm.model_executor.models import ModelRegistry
# Spy on register_model instead of depending on resolve_model_cls's
# signature, which moved between vLLM vintages. Bound-method capture works
# whether ModelRegistry is a class of classmethods or a singleton instance.
calls = []
orig = ModelRegistry.register_model
def spy(arch, target, *a, **k):
    calls.append((arch, target))
    return orig(arch, target, *a, **k)
ModelRegistry.register_model = spy
# load_plugins_by_group only LOADS entry points; load_general_plugins (what
# engine startup calls in every process) is the one that executes them.
from vllm.plugins import load_general_plugins
load_general_plugins()
want = ("Glm5NextForCausalLM",
        "weightless_steer.archs.glm5next:SteeredGlm5NextForCausalLM")
assert want in calls, f"shadow registration missing; register_model calls: {calls}"
print("SHADOW OK: register_model called with", want)
# The real compatibility gate: import the adapter against the image's vLLM.
import weightless_steer.archs.glm5next as g
print("ADAPTER IMPORT OK:", g.SteeredGlm5NextForCausalLM.__mro__[1].__module__)
"""

_PREFLIGHT_STOCK = r"""
from vllm.model_executor.models import ModelRegistry
calls = []
orig = ModelRegistry.register_model
def spy(arch, target, *a, **k):
    calls.append((arch, target))
    return orig(arch, target, *a, **k)
ModelRegistry.register_model = spy
from vllm.plugins import load_general_plugins
load_general_plugins()
shadows = [c for c in calls if "weightless_steer" in str(c[1])]
assert not shadows, f"env unset but register_model was called: {calls}"
print("STOCK OK: WEIGHTLESS_STEER_PATH unset -> no shadowing")
"""

_PREFLIGHT_CORE = f"""
import os
os.environ["WEIGHTLESS_STEER_PATH"] = {VECTOR_PATH!r}
os.environ["WEIGHTLESS_STEER_ALPHA"] = "2.0"
from weightless_steer.core import SteeringCore
core = SteeringCore.from_env(hook="residual_stream_post_layer",
                             num_layers=45, hidden_size=16384)
assert core is not None and len(core.dirs) == 44, core
assert min(core.dirs) == 1 and max(core.dirs) == 44, sorted(core.dirs)
assert core.alpha == 2.0, core.alpha
print("CORE OK: real GGUF parses, 44 dirs on layers 1..44, alpha 2.0")
"""


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version and the adapter's upstream import path;
    2. pip metadata: the `vllm.general_plugins` entry point is registered;
    3. vLLM's real plugin loader runs register() and the registry resolves
       Glm5NextForCausalLM to the STEERED class when WEIGHTLESS_STEER_PATH is
       set (this import-drags the adapter against the image's vLLM — the
       anchor/API-mismatch failure mode dies here, on CPU);
    4. with the env unset the registry stays stock (installed-but-
       unconfigured has zero effect);
    5. the real GLP-44 GGUF parses through SteeringCore.from_env at the
       model's geometry (45 layers, 16384-wide mHC stream).
    """
    import subprocess

    def sh(code, *args):
        r = subprocess.run([PY, "-c", code, *args],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr, sep="", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"preflight step failed (rc={r.returncode})")

    r = subprocess.run(
        [PY, "-c",
         "import vllm; print('vllm', vllm.__version__); "
         "import vllm.models.glm5next.nvidia.model as m; "
         "print('glm5next at', m.__file__)"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "vllm.models.glm5next.nvidia.model import failed"

    r = subprocess.run(
        [PY, "-c",
         "from importlib.metadata import entry_points\n"
         "eps = {e.name: e.value for e in entry_points("
         "group='vllm.general_plugins')}\n"
         "print('general_plugins:', eps)\n"
         "assert eps.get('weightless_steer') == "
         "'weightless_steer.plugin:register', eps\n"
         "print('ENTRY POINT OK')"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "weightless_steer entry point not registered"

    sh(_PREFLIGHT_STOCK)
    sh(_PREFLIGHT_RESOLVE, "/nonexistent-sentinel.gguf")
    sh(_PREFLIGHT_CORE)
    print("PREFLIGHT PASSED", flush=True)


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def diagnose():
    """CPU-only root-cause pass: why does the serve path never run the
    plugin's register() (zero weightless lines in the boot log, env verified
    set in /proc/<apiserver>/environ, pip install in dist-packages confirmed,
    spawn-child test registers fine)?

    1. pip/entry-point visibility under the serve interpreter (re-run).
    2. call-site context: arg_utils.py:825 & :2953, worker_base.py:271,
       core.py:119 — which enclosing defs, are they on the serve path.
    3. THE FAITHFUL REPRO of the API-server arg phase: construct
       AsyncEngineArgs exactly as from_cli_args does (DeviceConfig patched
       to survive no-GPU), root logging at DEBUG — the loader's
       "Available plugins"/"Failed to load plugin" and the plugin's own
       "shadowed" line all become visible. Ends with the registry entry.
    4. same, but through vllm.entrypoints.launchers.api_server.entry.main's
       own parser path (make_arg_parser + from_cli_args).
    """
    import subprocess

    VLLM = "/usr/local/lib/python3.12/dist-packages/vllm"

    def sh(cmd, **kw):
        print("+", " ".join(cmd) if isinstance(cmd, list) else cmd, flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
        print(r.stdout[-6000:], r.stderr[-6000:], sep="", flush=True)
        print(f"rc={r.returncode}", flush=True)
        return r

    sh([PY, "-m", "pip", "show", "weightless-steer"])
    sh([PY, "-c",
        "from importlib.metadata import entry_points\n"
        "eps = {e.name: e.value for e in entry_points("
        "group='vllm.general_plugins')}\n"
        "print('general_plugins:', eps)"])

    # 2. call-site context
    for f, lo, hi in (("engine/arg_utils.py", 778, 832),
                      ("engine/arg_utils.py", 2945, 2965),
                      ("v1/worker/worker_base.py", 262, 278),
                      ("v1/engine/core.py", 112, 124),
                      ("entrypoints/launchers/api_server/entry.py", 200, 260)):
        sh(["bash", "-c", f"echo --- {f}:{lo}-{hi}; sed -n {lo},{hi}p {VLLM}/{f}"])

    # 3. faithful pid-4 repro: run the api_server main() itself on CPU.
    # DeviceConfig is patched to survive no-GPU and uvloop is stubbed so
    # main() stops right before serving — every plugin-load call that pid 4
    # makes (add_cli_args, __post_init__) executes for real. DEBUG logging
    # shows the loader's own "Available plugins" + any swallowed exception.
    repro = f"""
import os, sys, types, logging
os.environ["WEIGHTLESS_STEER_PATH"] = {VECTOR_PATH!r}
os.environ["WEIGHTLESS_STEER_ALPHA"] = "0.0"
os.environ["HF_HOME"] = "/data/hf"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_CONFIG_PATH"] = "/work/vllm_logging_config.json"
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
from importlib.metadata import entry_points
print("DISCOVERED:", [e.name for e in
      entry_points(group="vllm.general_plugins")], flush=True)
import vllm.plugins as vp
print("plugins_loaded BEFORE:", vp.plugins_loaded, flush=True)
import vllm.config.device as d
d.DeviceConfig.__post_init__ = lambda self: None
sys.modules["uvloop"] = types.SimpleNamespace(
    run=lambda coro: print("WOULD RUN SERVER NOW", flush=True))
sys.argv = ["vllm", "serve", "/nonexistent-model",
            "--tensor-parallel-size", "4", "--max-model-len", "32768",
            "--moe-backend", "marlin"]
import runpy
try:
    runpy.run_module("vllm.entrypoints.openai.api_server",
                     run_name="__main__")
except SystemExit as e:
    print("SystemExit:", e, flush=True)
print("plugins_loaded AFTER:", vp.plugins_loaded, flush=True)
from vllm.model_executor.models import ModelRegistry as R
print("REGISTRY ENTRY:", R.models.get("Glm5NextForCausalLM"), flush=True)
"""
    with open("/tmp/repro.py", "w") as f:
        f.write(repro)
    sh([PY, "/tmp/repro.py"])

    # 4. logging bisect: WHY did the shadow register without its INFO line
    # printing, in the repro AND the real boot? Four probes, one process each.
    probes = {
        # A: plain root INFO, no vllm config — the v2 spawn child did this
        # and PRINTED. Baseline.
        "A_basicConfig": """
import os, logging, sys
os.environ["WEIGHTLESS_STEER_PATH"] = %r
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from vllm.plugins import load_general_plugins
load_general_plugins()
print("PROBE-A done", flush=True)
""",
        # B: vllm applies MY json config (env present at import) — the 22:10
        # boot's setup.
        "B_vllm_json_config": """
import os, logging, sys
os.environ["WEIGHTLESS_STEER_PATH"] = %r
os.environ["VLLM_LOGGING_CONFIG_PATH"] = "/work/vllm_logging_config.json"
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
from vllm.plugins import load_general_plugins
load_general_plugins()
print("PROBE-B done", flush=True)
""",
        # C: my json config, then a DIRECT logger.info on the plugin logger —
        # separates "logger broken" from "register never ran".
        "C_direct_log": """
import os, logging, sys
os.environ["VLLM_LOGGING_CONFIG_PATH"] = "/work/vllm_logging_config.json"
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
import vllm  # applies the config at import
lg = logging.getLogger("weightless_steer.plugin")
print("PROBE-C logger:", lg, "effective level:", lg.getEffectiveLevel(),
      "root handlers:", logging.getLogger().handlers, flush=True)
lg.info("DIRECT TEST LINE C")
print("PROBE-C done", flush=True)
""",
    }
    for name, code in probes.items():
        code = code % (VECTOR_PATH,) if "%r" in code else code
        with open(f"/tmp/probe_{name}.py", "w") as f:
            f.write(code)
        print(f"===== probe {name} =====", flush=True)
        sh([PY, f"/tmp/probe_{name}.py"])

    # 5. the image's own logging machinery: when is the config applied, and
    # does the guard requiring VLLM_CONFIGURE_LOGGING exist in this build?
    sh(["bash", "-c",
        f"grep -n 'VLLM_CONFIGURE_LOGGING\\|_configure_vllm_root_logger()\\|"
        f"dictConfig\\|VLLM_LOGGING_CONFIG_PATH' {VLLM}/logger.py | head -20"])
    # 6. the sitecustomize shim: markers must print on plain interpreter
    # startup when WEIGHTLESS_STEER_PATH is set, and stay silent in the
    # modal-runtime python (no vllm) and when the env is unset.
    shim_env = {"PATH": "/usr/bin:/usr/local/bin",
                "PYTHONPATH": "/opt/weightless-shim",
                "WEIGHTLESS_STEER_PATH": VECTOR_PATH,
                "WEIGHTLESS_STEER_ALPHA": "0.0"}
    print("===== probe shim =====", flush=True)
    sh([PY, "-c", "pass"], env=shim_env)
    sh(["/usr/local/bin/python3.12", "-c", "pass"], env=shim_env)
    shim_env2 = dict(shim_env)
    shim_env2.pop("WEIGHTLESS_STEER_PATH")
    sh([PY, "-c", "pass"], env=shim_env2)
    print("DIAGNOSE DONE", flush=True)


def _serve_cmd(model_path: str) -> list:
    # Invoke through the image's own python: vllm lives in
    # /usr/local/lib/python3.12/dist-packages, importable ONLY by
    # /usr/bin/python3.12 — Modal's add_python shim (/usr/local/bin/python3)
    # has no vllm, and this function body itself must stay stdlib-only.
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", SERVED_MODEL,
        "--tensor-parallel-size", TP,
        "--max-model-len", "32768",          # eval speed; full 1M not needed
        "--moe-backend", "marlin",           # the NVFP4 lane's backend
        "--gpu-memory-utilization", GMU,
        "--max-num-seqs", "8",
        "--chat-template", "/work/chat_template_mm.jinja",
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
        "--port", "8000",
    ]
    if KV_DTYPE:
        cmd += ["--kv-cache-dtype", KV_DTYPE]
    if ENFORCE_EAGER:
        cmd += ["--enforce-eager"]
    # GB10/SM121-only items deliberately absent: no kpool bind-mount, no
    # --block-size 2304, no --kv-cache-memory pin (all Spark-lane specifics).
    return cmd


def _model_snapshot_path() -> str:
    """Resolve the on-volume snapshot dir with stdlib only (huggingface_hub
    is not importable from Modal's runtime python on this image)."""
    import glob

    snaps = sorted(glob.glob(
        "/data/hf/hub/models--RedHatAI--GLM-5.3-Flash-NVFP4/snapshots/*"))
    assert len(snaps) == 1, f"expected exactly one snapshot, got {snaps}"
    assert os.path.isdir(snaps[0]) and \
        any(f.endswith(".safetensors") for f in os.listdir(snaps[0])), snaps
    return snaps[0]


@app.function(image=image, volumes={"/data": vol}, gpu=GPU,
              min_containers=0, max_containers=1, scaledown_window=180,
              startup_timeout=3600, timeout=6 * 3600,
              env=dict(ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       # compile caches on the volume: arm B's boot reuses
                       # arm A's (identical graph, one less compile)
                       TORCHINDUCTOR_CACHE_DIR="/data/cache/inductor",
                       VLLM_CACHE_ROOT="/data/cache/vllm",
                       TRITON_CACHE_DIR="/data/cache/triton",
                       WEIGHTLESS_STEER_PATH=VECTOR_PATH,
                       WEIGHTLESS_STEER_ALPHA=ALPHA,
                       # Force third-party INFO logs (the plugin's
                       # steering-active line) to print in EVERY process —
                       # vllm's default dictConfig only attaches a handler to
                       # the "vllm" logger (propagate=False); third-party
                       # loggers fall through to root, which stays WARNING —
                       # so the 19:03 boot was almost certainly steered with
                       # its plugin logs silently dropped. This config puts a
                       # root INFO handler in every process.
                       VLLM_CONFIGURE_LOGGING="1",
                       VLLM_LOGGING_CONFIG_PATH="/work/vllm_logging_config.json",
                       # The serve cmd is built INSIDE the container, where
                       # module-level os.environ.get() re-runs with the
                       # container's env — deploy-shell overrides only reach
                       # it through this dict (learned the hard way: a local
                       # GLM53_KV_DTYPE="" deploy left fp8_e4m3 in the cmd).
                       GLM53_KV_DTYPE=KV_DTYPE,
                       GLM53_GMU=GMU,
                       GLM53_ENFORCE_EAGER=ENFORCE_EAGER))
@modal.web_server(port=8000, startup_timeout=3600)
def serve():
    """vllm serve with the plugin's entry point active.

    Alpha is baked into the steering buffer at model init (the plugin refuses
    per-request controls), so one deploy = one arm. The day-0
    Glm5NextProcessor treats the model id as a literal local path, so serve
    the resolved snapshot dir, not the repo id (HF_HUB_OFFLINE=1: the path is
    on the volume, no network).

    vllm's stdout goes to a file ON THE VOLUME in APPEND mode (Modal
    auto-restarts crashed web tasks; truncate mode erased the previous
    attempt's dying words once already; Modal's app-log window is also just
    the last 100 entries). No mid-boot vol.commit(): two boots died mid
    weight-load with 30 s commits running while both commit-free boots were
    healthy — the commit is suspected of stalling the 9P mount under a
    198 GB read stream. The log is committed on the death path and from the
    SIGTERM handler (so `modal app stop` preserves it).

    The function MUST NOT block: Modal's web_server only starts routing
    traffic to the port once the function returns (a blocking monitor left
    two healthy boots unreachable — 2026-09-17/18). It watches the boot log
    for the plugin's steering-active line (fail-closed: the line missing by
    the deadline = unsteered = failed boot), then returns so routing can
    start; uvicorn binds :8000 a few minutes later."""
    import signal
    import subprocess
    import time

    model_path = _model_snapshot_path()
    cmd = _serve_cmd(model_path)
    log_dir = "/data/out-glm53-plugin-test"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/server-a{ALPHA}.log"
    logf = open(log_path, "a")
    logf.write(f"\n\n===== boot attempt {time.strftime('%H:%M:%S')} "
               f"alpha={ALPHA} =====\n")
    logf.flush()
    print("+", " ".join(cmd), flush=True)
    print(f"serve: arm alpha={ALPHA} gpu={GPU} tp={TP} "
          f"kv_dtype={KV_DTYPE or 'auto'} eager={bool(ENFORCE_EAGER)} "
          f"-> {log_path}", flush=True)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=dict(
                                os.environ, PYTHONUNBUFFERED="1",
                                # sitecustomize shim: every python process of
                                # the serve stack (api server, spawned
                                # EngineCore + workers) registers the shadow
                                # and prints stderr markers — the logging-
                                # module config mess on this dev build made
                                # the plugin's own INFO lines unreliable as
                                # boot evidence. See modal/sitecustomize.py.
                                PYTHONPATH="/opt/weightless-shim:"
                                + os.environ.get("PYTHONPATH", "")))

    def _sigterm(signum, frame):
        vol.commit()
        proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    deadline = time.time() + 30 * 60
    while time.time() < deadline:
        if proc.poll() is not None:
            txt = open(log_path, errors="replace").read()
            print("vllm exited rc=", proc.returncode, "\n--- log tail ---\n",
                  txt[-6000:], sep="", flush=True)
            vol.commit()
            raise RuntimeError(f"vllm serve died, rc={proc.returncode}; "
                               f"see {log_path} on the volume")
        time.sleep(20)
        try:
            with open(log_path, errors="replace") as f:
                txt = f.read()
        except OSError:
            continue
        hit = [l for l in txt.splitlines()
               if "weightless GLP steering active" in l
               or "WEIGHTLESS-SHIM: steering core loaded" in l]
        if hit:
            for line in hit:
                print("STEERING-LINE:", line, flush=True)
            print("serve: steering confirmed in boot log; returning so "
                  "Modal starts routing (port binds when uvicorn comes up)",
                  flush=True)
            return
    # deadline passed with no steering line: fail closed
    txt = open(log_path, errors="replace").read()
    print("STEERING-LINE MISSING after 30 min — refusing to serve "
          "unsteered. Log tail:\n", txt[-4000:], sep="", flush=True)
    proc.terminate()
    vol.commit()
    raise RuntimeError("steering-active line never appeared in boot log")


@app.local_entrypoint()
def main():
    print(__doc__)
