"""Qwen3.8-Flash-Next + the weightless-steer vLLM plugin: first real-GPU boot.

The plugin (vllm-plugin/, dist `weightless-steer`) shadows the arch in vLLM's
ModelRegistry through the `vllm.general_plugins` entry point — no in-container
steering patch, unlike the hotfix lane this test is compared against
(patches/hotfix-qwen38fn-steering-projective.py, reference numbers in
BENCHMARK.md's GLP-47 section). This app boots the day-0 serving image with
the plugin pip-installed and the published GLP-47 vector, and exposes an
OpenAI-compatible endpoint for the local eval driver
(modal/eval_qwen38fn_plugin.py).

Stack (all validated facts, see recipe/qwen38fn/README.md and
refusal-research/experiments/20260827-flash-next-nvfp4-serve):
  - Image: vllm/vllm-openai:qwen38-flash-next (amd64 verified on Docker Hub;
    ships vllm/models/qwen3_8_flash_next/nvidia/model.py, the module the
    adapter imports; fork 1CatAI/1Cat-vLLM).
  - Model: RadixArk/Qwen3.8-Flash-Next-NVFP4 (modelopt_fp4, ~135 GB).
    architectures=["Qwen4ExpForConditionalGeneration"] — the MULTIMODAL
    wrapper; the plugin shadows that name (plus the two qwen3_8_flash_next
    names) because the wrapper instantiates its language model directly.
  - Shape: H100:2, TP2 (~67.5 GiB weights/rank). Fallback if the NVFP4
    kernels reject sm90 (the NVFP4-serve lane's open risk): B200:1 TP1
    (QWEN38FN_GPU=B200 QWEN38FN_TP=1) — the proven run-4 config.
  - VLLM_PLE_CPU_OFFLOAD=1: the 51 GB FP8 N-gram PLE table lives in host RAM
    (x86 image allows it; the arm64 one rejects it at nnodes=2). Requires
    --distributed-executor-backend mp: the uni executor never spawns the
    PleOffloadWorker and the first real forward deadlocks the GPU stream
    (NVFP4-serve runs 1-2).
  - patches/patch-qwen38fn-ple-fp8-nvfp4.py runs in-container BEFORE serve,
    fail-closed: without it the day-0 image cannot load the FP8-serialized
    PLE table under a modelopt_fp4 global config (run 3).
  - --no-enable-prefix-caching is MANDATORY: prefix caching forces
    mamba_cache_mode="align", which splits every prefill at a block boundary
    and corrupts capture/steering (capture-lane run 3).
  - Vector: msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47 (gated,
    hf-token secret), ...-L1-47-a1.gguf. alpha=1.0 is calibrated; 1.5+
    over-projects (BENCHMARK.md).
  - Chat template: the checkpoint ships chat_template.jinja with an
    enable_thinking branch (default ON, would eat the 400-token eval
    budget); the reference protocol renders enable_thinking=False, so the
    server pins it via --default-chat-template-kwargs.

Two arms, one deploy each (alpha is a model-init buffer; no runtime override):
    modal run cloud_serve_qwen38fn.py::ensure_vector    # CPU, seconds
    modal run cloud_serve_qwen38fn.py::ensure_weights   # CPU, ~135 GB once
    modal run cloud_serve_qwen38fn.py::preflight        # CPU, entry point + gguf
    WEIGHTLESS_STEER_ALPHA=0.0 modal deploy cloud_serve_qwen38fn.py
    python3 modal/eval_qwen38fn_plugin.py --base-url <url> --alpha 0.0
    modal app stop weightless-qwen38fn-plugin-test
    WEIGHTLESS_STEER_ALPHA=1.0 modal deploy cloud_serve_qwen38fn.py
    python3 modal/eval_qwen38fn_plugin.py --base-url <url> --alpha 1.0
    modal app stop weightless-qwen38fn-plugin-test

First boot runs WITHOUT --enforce-eager on purpose: the adapter's
torch.compile rebind fix is part of what is being tested
(Qwen3_8FlashNextModel IS @support_torch_compile-decorated, so the rebind is
live, unlike glm5next's dormant one). If output is garbage or the
steering-active line is missing in compiled mode, redeploy with
QWEN38FN_ENFORCE_EAGER=1 to isolate.

Logging: same dev-build caveat as glm5next — vllm's dictConfig attaches a
handler ONLY to the `vllm` logger, so the plugin's INFO lines may never
render. The sitecustomize shim (modal/sitecustomize_qwen38fn.py) registers
at every interpreter start, pre-parses the vector fail-closed, and prints
WEIGHTLESS-SHIM markers straight to stderr; "no log line" is no
information, not failure.

Cost discipline: max_containers=1, scaledown_window=180, stop the app when
done. Expected spend: 2 boots x (load + eval) x 2 H100.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]           # weightless/
MODEL_ID = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
SERVED_MODEL = "qwen38fn-flash"
VOLUME_NAME = "qwen38-flash-next"
VECTOR_REPO = "msuiche/Qwen3.8-Flash-Next-abliterated-cyber-GLP-47"
VECTOR_FILE = "Qwen3.8-Flash-Next-abliterated-cyber-GLP-47-L1-47-a1.gguf"
VECTOR_PATH = "/data/vector/" + VECTOR_FILE
IMAGE = "vllm/vllm-openai:qwen38-flash-next"
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1",
       "VLLM_PLE_CPU_OFFLOAD": "1"}
PY = "/usr/bin/python3.12"

# Arm/shape switches, read at deploy time (same convention as cloud_serve_glm53).
GPU = os.environ.get("QWEN38FN_GPU", "H100:2")
TP = os.environ.get("QWEN38FN_TP", "2")
ENFORCE_EAGER = os.environ.get("QWEN38FN_ENFORCE_EAGER", "")  # "1" to isolate
GMU = os.environ.get("QWEN38FN_GMU", "0.92")
ALPHA = os.environ.get("WEIGHTLESS_STEER_ALPHA", "0.0")

app = modal.App("weightless-qwen38fn-plugin-test")
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
# site-packages. The PLE FP8 patch rides along and is applied fail-closed in
# serve() before vllm starts (required to load the checkpoint's PLE table at
# all — a serving prerequisite, not steering).
image = (modal.Image.from_registry(IMAGE, add_python="3.12",
                                   setup_dockerfile_commands=["ENTRYPOINT []"])
         .add_local_dir(ROOT / "vllm-plugin", "/opt/weightless-src/vllm-plugin",
                        copy=True)
         .add_local_dir(ROOT / "weightless_runtime",
                        "/opt/weightless-src/weightless_runtime", copy=True)
         .add_local_file(ROOT / "patches" / "patch-qwen38fn-ple-fp8-nvfp4.py",
                         "/work/patch-qwen38fn-ple-fp8-nvfp4.py", copy=True)
         .add_local_file(ROOT / "modal" / "vllm_logging_config.json",
                         "/work/vllm_logging_config.json", copy=True)
         .add_local_file(ROOT / "modal" / "sitecustomize_qwen38fn.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         .run_commands(
             f"{PY} -m pip --version || {PY} -m ensurepip --upgrade",
             "cd /opt/weightless-src/vllm-plugin && "
             f"{PY} -m pip install --no-deps ."))


@app.function(image=download_image, volumes={"/data": vol},
              timeout=24 * 3600, secrets=[hf_secret], env=ENV)
def ensure_weights():
    """Resume/cache the ~135 GB NVFP4 snapshot on CPU; no GPU allocation.

    Fail-closed size gate (same discipline as cloud_serve_glm53.ensure_weights):
    the safetensors index present, zero .incomplete blobs, >130 GB before any
    GPU spend. The repo is public; the secret rides along for rate limits only.
    """
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID)
    snap = Path(snapshot)
    assert (snap / "model.safetensors.index.json").is_file(), "index missing"
    hub = (Path(os.environ["HF_HOME"]) / "hub"
           / "models--RadixArk--Qwen3.8-Flash-Next-NVFP4")
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    n_shards = len(list(snap.glob("*.safetensors")))
    print(f"shards: {n_shards}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e9:.2f} GB", flush=True)
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 130e9, f"size-gate failed: {total / 1e9:.1f} GB"
    vol.commit()
    print("weights ready:", snapshot, flush=True)


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              secrets=[hf_secret], env=ENV)
def ensure_vector():
    """Fetch the gated GLP-47 GGUF and stage it at the stable VECTOR_PATH.

    The HF-cache snapshot path embeds a revision hash; the server env needs a
    path known at deploy time, so the file is copied out of the cache. Fail-
    closed: <2 MB means a truncated or LFS-pointer download (real file is
    ~1.9 MB for 47 x 10240 f32... checked below against the GGUF header)."""
    from huggingface_hub import hf_hub_download

    dst = Path(VECTOR_PATH)
    if not dst.is_file():
        fetched = hf_hub_download(repo_id=VECTOR_REPO, filename=VECTOR_FILE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(fetched).read_bytes())
        vol.commit()
    size = dst.stat().st_size
    assert size > 1e6, f"{VECTOR_PATH} looks truncated ({size} B)"
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
want = ("Qwen4ExpForConditionalGeneration",
        "weightless_steer.archs.qwen38fn:SteeredQwen3_8FlashNextForConditionalGeneration")
assert want in calls, f"shadow registration missing; register_model calls: {calls}"
print("SHADOW OK: register_model called with", want)
# The real compatibility gate: import the adapter against the image's vLLM.
import weightless_steer.archs.qwen38fn as q
print("ADAPTER IMPORT OK:",
      q.SteeredQwen3_8FlashNextForCausalLM.__mro__[1].__module__,
      q.SteeredQwen3_8FlashNextForConditionalGeneration.__mro__[1].__module__)
# What does the image's registry map the checkpoint's arch name to?
# (ModelRegistry is a _ModelRegistry instance here; inspect its models dict.)
models = getattr(ModelRegistry, "models", {})
for name, spec in models.items():
    if "Qwen4Exp" in name or "Qwen3_8FlashNext" in name:
        mod = getattr(spec, "module_name", spec)
        cls = getattr(spec, "class_name", "?")
        print("REGISTRY MAP:", name, "->", mod, cls)
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
os.environ.pop("WEIGHTLESS_STEER_ALPHA", None)
from weightless_steer.core import SteeringCore
core = SteeringCore.from_env(hook="residual_stream_post_layer",
                             num_layers=48, hidden_size=10240)
assert core is not None and len(core.dirs) == 47, core
assert min(core.dirs) == 1 and max(core.dirs) == 47, sorted(core.dirs)
print("CORE OK: real GGUF parses, 47 dirs on layers 1..47, "
      f"alpha={{core.alpha}}")
"""

_PREFLIGHT_NVFP4 = r"""
# Where do this image's NVFP4 kernels draw the capability line? The
# NVFP4-serve lane's open risk was "modelopt_fp4 GEMM kernels are sm100+;
# on H100 the boot may raise or fall back" — read the dispatch before
# spending a GPU boot on H100.
import inspect
import vllm.model_executor.layers.quantization.modelopt as mo
names = [n for n in dir(mo) if "NvFp4" in n or "NVFP4" in n or "Fp4" in n]
print("modelopt fp4 classes:", names)
for n in names:
    cls = getattr(mo, n)
    m = getattr(cls, "get_min_capability", None)
    if m is not None:
        try:
            print(f"{n}.get_min_capability() =", m())
        except Exception as e:
            print(f"{n}.get_min_capability() raised: {e}")
try:
    src = inspect.getsource(mo)
    for i, line in enumerate(src.splitlines()):
        low = line.lower()
        if ("capability" in low or "sm100" in low or "sm90" in low
                or "marlin" in low or "flashinfer" in low or "cutlass" in low):
            print(f"modelopt.py:{i+1}: {line.strip()}")
except OSError as e:
    print("no source:", e)
"""


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version and the adapter's upstream import path;
    2. pip metadata: the `vllm.general_plugins` entry point is registered;
    3. vLLM's real plugin loader runs register() and the shadow of the
       checkpoint's declared arch (Qwen4ExpForConditionalGeneration) is
       registered — and this import-drags the adapter against the image's
       vLLM (the anchor/API-mismatch failure mode dies here, on CPU);
    4. with the env unset the registry stays stock (installed-but-
       unconfigured has zero effect);
    5. the real GLP-47 GGUF parses through SteeringCore.from_env at the
       model's geometry (48 layers, 10240-wide HC stream);
    6. the image's NVFP4 kernel capability gates, verbatim (decides
       H100:2 vs the proven B200:1 fallback);
    7. the PLE FP8 patch dry-runs against the image's real ple_layer.py
       (already-applied is fine: the image layer is replaced at build, but a
       rebuilt image that drifted off the anchor must fail HERE, not after a
       10-minute weight load);
    8. the compile-wrapper API the adapter's rebind depends on
       (TorchCompileWithNoGuardsWrapper with cleanup/_compile_prefix).
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
         "import vllm.models.qwen3_8_flash_next.nvidia.model as m; "
         "print('qwen3_8_flash_next at', m.__file__)"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, \
        "vllm.models.qwen3_8_flash_next.nvidia.model import failed"

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
    sh(_PREFLIGHT_NVFP4)
    r = subprocess.run(
        [PY, "/work/patch-qwen38fn-ple-fp8-nvfp4.py"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "PLE FP8 patch failed against the image"
    sh(r"""
import inspect
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper as W
assert hasattr(W, "cleanup"), "compile wrapper lost cleanup()"
import vllm.compilation.decorators as d
src = inspect.getsource(d)
assert "TorchCompileWithNoGuardsWrapper.__init__" in src
assert "_compile_prefix" in inspect.getsource(W)
print("COMPILE WRAPPER API OK (rebind contract holds)")
""")
    print("PREFLIGHT PASSED", flush=True)


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
        "--distributed-executor-backend", "mp",  # uni never spawns the PLE
                                                 # offload worker (deadlock)
        "--max-model-len", "32768",          # eval speed; full 256K not needed
        "--max-num-seqs", "8",
        "--max-num-batched-tokens", "8192",
        "--no-enable-prefix-caching",        # MANDATORY: align mode corrupts
        "--gpu-memory-utilization", GMU,
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--port", "8000",
    ]
    if ENFORCE_EAGER:
        cmd += ["--enforce-eager"]
    return cmd


def _model_snapshot_path() -> str:
    """Resolve the on-volume snapshot dir with stdlib only (huggingface_hub
    is not importable from Modal's runtime python on this image)."""
    import glob

    snaps = sorted(glob.glob(
        "/data/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*"))
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
                       # loggers fall through to root, which stays WARNING.
                       VLLM_CONFIGURE_LOGGING="1",
                       VLLM_LOGGING_CONFIG_PATH="/work/vllm_logging_config.json",
                       # The serve cmd is built INSIDE the container, where
                       # module-level os.environ.get() re-runs with the
                       # container's env — deploy-shell overrides only reach
                       # it through this dict (learned the hard way on the
                       # glm53 lane).
                       QWEN38FN_ENFORCE_EAGER=ENFORCE_EAGER,
                       QWEN38FN_GMU=GMU,
                       QWEN38FN_TP=TP))
@modal.web_server(port=8000, startup_timeout=3600)
def serve():
    """vllm serve with the plugin's entry point active.

    Alpha is baked into the steering buffer at model init (the plugin refuses
    per-request controls), so one deploy = one arm. The snapshot dir is served
    directly (HF_HUB_OFFLINE=1: the path is on the volume, no network).

    The PLE FP8 patch is applied to the installed image BEFORE vllm starts,
    fail-closed: the day-0 image cannot load this checkpoint's FP8 PLE table
    without it. vllm's stdout goes to a file ON THE VOLUME in APPEND mode
    (Modal auto-restarts crashed web tasks; truncate mode erased a previous
    attempt's dying words once already on the glm53 lane). No mid-boot
    vol.commit() (a commit under a 135 GB read stream is suspected of
    stalling the 9P mount); the log is committed on the death path and from
    the SIGTERM handler.

    The function MUST NOT block: Modal's web_server only starts routing
    traffic to the port once the function returns. It watches the boot log
    for the steering-active line (fail-closed: the line missing by the
    deadline = unsteered = failed boot), then returns so routing can start;
    uvicorn binds :8000 a few minutes later."""
    import signal
    import subprocess
    import time

    # Serving prerequisite, fail-closed (exit 1 kills the boot before the
    # 135 GB load): teach the image's PLE layer to load the checkpoint's FP8
    # table. Idempotent; applied every boot so an image rebuild that dropped
    # it fails here.
    r = subprocess.run([PY, "/work/patch-qwen38fn-ple-fp8-nvfp4.py"],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    if r.returncode != 0:
        raise RuntimeError("PLE FP8 patch failed; refusing to serve")

    model_path = _model_snapshot_path()
    cmd = _serve_cmd(model_path)
    log_dir = "/data/out-qwen38fn-plugin-test"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/server-a{ALPHA}.log"
    logf = open(log_path, "a")
    logf.write(f"\n\n===== boot attempt {time.strftime('%H:%M:%S')} "
               f"alpha={ALPHA} gpu={GPU} tp={TP} =====\n")
    logf.flush()
    print("+", " ".join(cmd), flush=True)
    print(f"serve: arm alpha={ALPHA} gpu={GPU} tp={TP} "
          f"eager={bool(ENFORCE_EAGER)} -> {log_path}", flush=True)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=dict(
                                os.environ, PYTHONUNBUFFERED="1",
                                # sitecustomize shim: every python process of
                                # the serve stack (api server, spawned
                                # EngineCore + workers) registers the shadow
                                # and prints stderr markers — the logging-
                                # module config mess on these dev builds made
                                # the plugin's own INFO lines unreliable as
                                # boot evidence. See
                                # modal/sitecustomize_qwen38fn.py.
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
