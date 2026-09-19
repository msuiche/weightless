"""GLM-5.3 743B (glm_moe_dsa) + the weightless-steer vLLM plugin: GPU smoke.

The plugin (vllm-plugin/, dist `weightless-steer`) shadows
GlmMoeDsaForCausalLM in vLLM's ModelRegistry through the
`vllm.general_plugins` entry point — no in-container file patching, unlike
the hotfix lane this smoke is compared against
(patches/hotfix-glm53xl-steering-projective.py; reference numbers in the
recipe/glm53xl/README.md GLP-77 table). This app boots stock vLLM 0.28.0
with the plugin pip-installed and the published GLP-77 vector, and exposes
an OpenAI-compatible endpoint for the local smoke driver
(modal/smoke_glm53xl_plugin.py).

Stack (all validated facts, see recipe/glm53xl/README.md and
refusal-research/experiments/20260829-glm53-flagship):
  - Image: vllm/vllm-openai:v0.28.0 (first release with glm_moe_dsa; its
    vllm/model_executor/models/deepseek_v2.py is byte-identical to
    patches/reference/deepseek_v2_v0280.py — preflight md5-gates this).
  - Model: RadixArk/GLM-5.3-NVFP4 (ModelOpt NVFP4 W4A4, 47 shards, ~465 GB;
    ungated). Its config declares quantization_config.kv_cache_scheme (fp8
    KV) which NO sm90 sparse-MLA backend accepts — prep_snapshot strips it
    (the 20260829 trap).
  - Shape: H100:8 single container, TP8 (~58 GiB weights/rank), marlin MoE
    backend (NVFP4 on Hopper), gmu 0.92.
  - Vector: msuiche/GLM-5.3-abliterated-cyber-GLP-77 (gated, hf-token),
    GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf. alpha=1.0 is
    calibrated — higher makes refusal WORSE on this model. Do not raise it.
  - Chat template: GLM-5.3's stock template has no enable_thinking knob and
    always opens <think>; prep_snapshot applies the vendored
    patches/vendor/patch_chat_template_thinking.py to the volume snapshot
    and the server passes enable_thinking=false (the reference protocol).

One arm, one deploy (alpha is a model-init buffer; no runtime override):
    modal run cloud_serve_glm53xl.py::ensure_weights   # CPU, ~465 GB once
    modal run cloud_serve_glm53xl.py::prep_snapshot    # CPU: kv scheme + template
    modal run cloud_serve_glm53xl.py::ensure_vector    # CPU, seconds
    modal run cloud_serve_glm53xl.py::preflight        # CPU, entry point + gguf + md5
    modal deploy cloud_serve_glm53xl.py
    python3 modal/smoke_glm53xl_plugin.py --base-url <url>
    modal app stop weightless-glm53xl-plugin-test

The boot runs WITHOUT --enforce-eager on purpose: unlike glm5next,
DeepseekV2Model IS @support_torch_compile'd in stock 0.28.0, so the
adapter's compile-rebind fix is live and is part of what is being tested
(the reference hotfix lane was eager-only). If output is garbage or the
steering-active line is missing in compiled mode, redeploy with
GLM53XL_ENFORCE_EAGER=1 to isolate.

The sitecustomize shim (modal/sitecustomize_glm53xl.py) is the evidence
channel: vLLM's dictConfig attaches a handler only to the `vllm` logger, so
plugin INFO lines may never render even when steering works. The shim runs
register() at every interpreter start, pre-parses the vector fail-closed,
and prints WEIGHTLESS-SHIM markers straight to stderr.

Cost discipline: max_containers=1, scaledown_window=180, stop the app when
done. Expected spend: 1 boot x (load + smoke) x 8 H100.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]           # weightless/
MODEL_ID = "RadixArk/GLM-5.3-NVFP4"
SERVED_MODEL = "glm53xl"
VOLUME_NAME = "glm53xl-plugin-test"
VECTOR_REPO = "msuiche/GLM-5.3-abliterated-cyber-GLP-77"
VECTOR_FILE = "GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf"
VECTOR_PATH = "/data/vector/" + VECTOR_FILE
IMAGE = "vllm/vllm-openai:v0.28.0"
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
PY = "/usr/bin/python3.12"
# md5 of the image's vllm/model_executor/models/deepseek_v2.py, pinned to
# the vendored reference the adapter's copied forward is tested against
# (patches/reference/deepseek_v2_v0280.py).
DEEPSEEK_V2_MD5 = "61da370634b4dfbe7f0158aaffa55202"

# Shape switches, read at deploy time (same convention as cloud_serve_glm53).
GPU = os.environ.get("GLM53XL_GPU", "H100:8")
TP = os.environ.get("GLM53XL_TP", "8")
ENFORCE_EAGER = os.environ.get("GLM53XL_ENFORCE_EAGER", "")  # "1" to isolate
GMU = os.environ.get("GLM53XL_GMU", "0.92")
ALPHA = os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0")  # calibrated; do NOT raise

app = modal.App("weightless-glm53xl-plugin-test")
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
         .add_local_file(ROOT / "patches" / "vendor"
                         / "patch_chat_template_thinking.py",
                         "/work/patch_chat_template_thinking.py", copy=True)
         .add_local_file(ROOT / "modal" / "vllm_logging_config.json",
                         "/work/vllm_logging_config.json", copy=True)
         .add_local_file(ROOT / "modal" / "sitecustomize_glm53xl.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         .run_commands(
             f"{PY} -m pip --version || {PY} -m ensurepip --upgrade",
             "cd /opt/weightless-src/vllm-plugin && "
             f"{PY} -m pip install --no-deps ."))


@app.function(image=download_image, volumes={"/data": vol},
              timeout=24 * 3600, secrets=[hf_secret], env=ENV)
def ensure_weights():
    """Resume/cache the ~465 GB NVFP4 snapshot on CPU; no GPU allocation.

    Fail-closed size gate: 47 safetensors shards + index, zero .incomplete
    blobs, >460 GB before any GPU spend. Repo is ungated; the secret rides
    along for rate limits only."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID)
    snap = Path(snapshot)
    shards = sorted(snap.glob("*.safetensors"))
    assert (snap / "model.safetensors.index.json").is_file(), "index missing"
    hub = (Path(os.environ["HF_HOME"]) / "hub"
           / "models--RadixArk--GLM-5.3-NVFP4")
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    print(f"shards: {len(shards)}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e9:.2f} GB", flush=True)
    assert len(shards) == 47, f"expected 47 safetensors, found {len(shards)}"
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 460e9, f"size-gate failed: {total / 1e9:.1f} GB"
    vol.commit()
    print("weights ready:", snapshot, flush=True)


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              secrets=[hf_secret], env=ENV)
def ensure_vector():
    """Fetch the gated GLP-77 GGUF and stage it at the stable VECTOR_PATH.

    The HF-cache snapshot path embeds a revision hash; the server env needs a
    path known at deploy time, so the file is copied out of the cache.
    Fail-closed: <1 MB means a truncated or LFS-pointer download (real file
    is ~1.9 MB: 77 directions x 6144 f32)."""
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


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def prep_snapshot():
    """One-time snapshot fixes, both idempotent, both fail-closed:

    1. Strip quantization_config.kv_cache_scheme from the cached config.json
       — the RadixArk checkpoint declares fp8 KV, which vLLM honors over
       kv_cache_dtype="auto", and NO sm90 sparse-MLA backend accepts fp8 KV
       (the 20260829 flagship lane's first boot died on this).
    2. Patch the snapshot's chat_template.jinja with the vendored
       enable_thinking knob (stock GLM-5.3 always opens <think>; the
       reference protocol renders with enable_thinking=False).
    """
    import glob
    import json
    import subprocess

    paths = glob.glob(
        "/data/hf/hub/models--RadixArk--GLM-5.3-NVFP4/snapshots/*/config.json")
    assert len(paths) == 1, paths
    p = paths[0]
    d = json.load(open(p))
    qc = d.get("quantization_config") or {}
    if "kv_cache_scheme" in qc:
        qc.pop("kv_cache_scheme")
        os.remove(p)  # p is a symlink into the HF blob store; replace it
        with open(p, "w") as f:
            json.dump(d, f, indent=2)
        print("kv_cache_scheme removed from", p, flush=True)
    else:
        print("kv_cache_scheme already absent:", p, flush=True)

    tpl = str(Path(p).parent / "chat_template.jinja")
    r = subprocess.run([PY, "/work/patch_chat_template_thinking.py", tpl],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, f"chat template patch failed on {tpl}"
    vol.commit()
    print("prep_snapshot done", flush=True)


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
want = ("GlmMoeDsaForCausalLM",
        "weightless_steer.archs.glm53xl:SteeredGlmMoeDsaForCausalLM")
assert want in calls, f"shadow registration missing; register_model calls: {calls}"
print("SHADOW OK: register_model called with", want)
# The real compatibility gate: import the adapter against the image's vLLM.
import weightless_steer.archs.glm53xl as g
print("ADAPTER IMPORT OK:", g.SteeredGlmMoeDsaForCausalLM.__mro__[1].__module__)
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
from weightless_steer.core import SteeringCore
core = SteeringCore.from_env(hook="residual_stream_post_layer",
                             num_layers=78, hidden_size=6144)
assert core is not None and len(core.dirs) == 77, core
assert min(core.dirs) == 1 and max(core.dirs) == 77, sorted(core.dirs)
assert core.alpha == 1.0, core.alpha
print("CORE OK: real GGUF parses, 77 dirs on layers 1..77, alpha 1.0")
"""


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version, the deepseek_v2 module the adapter imports,
       and — the compat gate for the copied forward — the image file's md5
       against the vendored reference the structure tests pin;
    2. pip metadata: the `vllm.general_plugins` entry point is registered;
    3. vLLM's real plugin loader runs register() and the registry shadows
       GlmMoeDsaForCausalLM when WEIGHTLESS_STEER_PATH is set (this
       import-drags the adapter against the image's vLLM — an API mismatch
       dies here, on CPU);
    4. with the env unset the registry stays stock (installed-but-
       unconfigured has zero effect);
    5. the real GLP-77 GGUF parses through SteeringCore.from_env at the
       model's geometry (78 layers, plain 6144-wide stream).
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
         "import hashlib\n"
         "import vllm\n"
         "print('vllm', vllm.__version__)\n"
         "import vllm.model_executor.models.deepseek_v2 as m\n"
         "print('deepseek_v2 at', m.__file__)\n"
         "md5 = hashlib.md5(open(m.__file__, 'rb').read()).hexdigest()\n"
         "print('deepseek_v2 md5', md5)\n"
         f"assert md5 == {DEEPSEEK_V2_MD5!r}, (\n"
         "    'image deepseek_v2.py drifted from the vendored reference; '\n"
         "    'the adapter forward copy needs re-pinning')\n"
         "assert hasattr(m, 'GlmMoeDsaForCausalLM')\n"
         "assert hasattr(m, 'DeepseekV2Model')\n"
         "print('DEEPSEEK_V2 REFERENCE MATCH OK')"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "deepseek_v2 reference check failed"

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
        "--max-model-len", "16384",          # smoke; the 1M ctx is not needed
        "--moe-backend", "marlin",           # NVFP4 routed experts on sm90
        "--gpu-memory-utilization", GMU,
        "--max-num-seqs", "8",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--port", "8000",
    ]
    if ENFORCE_EAGER:
        cmd += ["--enforce-eager"]
    # GB10/SM121-only items deliberately absent: no kernel overlay, no
    # --kv-cache-memory pin, no NCCL re-pin (all Spark-lane specifics).
    return cmd


def _model_snapshot_path() -> str:
    """Resolve the on-volume snapshot dir with stdlib only (huggingface_hub
    is not importable from Modal's runtime python on this image)."""
    import glob

    snaps = sorted(glob.glob(
        "/data/hf/hub/models--RadixArk--GLM-5.3-NVFP4/snapshots/*"))
    assert len(snaps) == 1, f"expected exactly one snapshot, got {snaps}"
    assert os.path.isdir(snaps[0]) and \
        any(f.endswith(".safetensors") for f in os.listdir(snaps[0])), snaps
    return snaps[0]


@app.function(image=image, volumes={"/data": vol}, gpu=GPU,
              min_containers=0, max_containers=1, scaledown_window=180,
              startup_timeout=3600, timeout=6 * 3600,
              env=dict(ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       # compile caches on the volume: a reboot reuses them
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
                       # it through this dict.
                       GLM53XL_GMU=GMU,
                       GLM53XL_ENFORCE_EAGER=ENFORCE_EAGER))
@modal.web_server(port=8000, startup_timeout=3600)
def serve():
    """vllm serve with the plugin's entry point active.

    Alpha is baked into the steering buffer at model init (the plugin refuses
    per-request controls), so one deploy = one arm. Serve the resolved
    snapshot dir, not the repo id (HF_HUB_OFFLINE=1: the path is on the
    volume, no network).

    vllm's stdout goes to a file ON THE VOLUME in APPEND mode (Modal
    auto-restarts crashed web tasks; truncate mode erased the previous
    attempt's dying words once already; Modal's app-log window is also just
    the last 100 entries). No mid-boot vol.commit(): two glm5next boots died
    mid weight-load with 30 s commits running — the commit is suspected of
    stalling the 9P mount under a multi-hundred-GB read stream. The log is
    committed on the death path and from the SIGTERM handler (so
    `modal app stop` preserves it).

    The function MUST NOT block: Modal's web_server only starts routing
    traffic to the port once the function returns. It watches the boot log
    for the plugin's steering-active line (fail-closed: the line missing by
    the deadline = unsteered = failed boot), then returns so routing can
    start; uvicorn binds :8000 a few minutes later."""
    import signal
    import subprocess
    import time

    model_path = _model_snapshot_path()
    cmd = _serve_cmd(model_path)
    log_dir = "/data/out-glm53xl-plugin-test"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/server-a{ALPHA}.log"
    logf = open(log_path, "a")
    logf.write(f"\n\n===== boot attempt {time.strftime('%H:%M:%S')} "
               f"alpha={ALPHA} =====\n")
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
                                # module config mess made the plugin's own
                                # INFO lines unreliable as boot evidence on
                                # the glm5next lane. See
                                # modal/sitecustomize_glm53xl.py.
                                PYTHONPATH="/opt/weightless-shim:"
                                + os.environ.get("PYTHONPATH", "")))

    def _sigterm(signum, frame):
        vol.commit()
        proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    deadline = time.time() + 45 * 60
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
    print("STEERING-LINE MISSING after 45 min — refusing to serve "
          "unsteered. Log tail:\n", txt[-4000:], sep="", flush=True)
    proc.terminate()
    vol.commit()
    raise RuntimeError("steering-active line never appeared in boot log")


@app.local_entrypoint()
def main():
    print(__doc__)
