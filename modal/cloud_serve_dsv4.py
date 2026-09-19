"""DSV4-Flash-0731 + the weightless-steer vLLM plugin: real-GPU boot test.

The plugin (vllm-plugin/, dist `weightless-steer`) shadows
DeepseekV4ForCausalLM in vLLM's ModelRegistry through the
`vllm.general_plugins` entry point — no in-container file patching, unlike
the hotfix lane this test is compared against
(patches/hotfix-dsv4-steering-projective.py, reference numbers in
BENCHMARK.md's GLP-29 section). This app boots the day-0 serving image with
the plugin pip-installed and the published GLP-29 vector, and exposes an
OpenAI-compatible endpoint for the local eval driver
(modal/eval_dsv4_plugin.py).

Stack (all validated facts, see refusal-research/METHODOLOGY.md SS14 and
refusal-research/experiments/20260905-dsv4-residual-glp — the capture/eval
lane that ran this exact model+image on H100:4):
  - Image: vllm/vllm-openai:deepseekv4-flash-vision (the Vision-Exp day-0
    tag; the text stack is the same vllm/models/deepseek_v4/nvidia/model.py
    — the module the adapter imports; the vendored reference is a
    byte-identical pull from it).
  - Model: deepseek-ai/DeepSeek-V4-Flash-0731, revision 7872f01b (matches
    GLP-29's glp.base_revision), ~168 GB FP8, already on the volume.
  - Shape: H100:4 single container, TP4 (the proven shape for this
    model+image). Fallback: H200:2 (DSV4_GPU=H200:2 DSV4_TP=2).
  - kv_cache_dtype=fp8 explicitly: the fp8_ds_mla layout asserts on the
    default "auto" at engine init (METHODOLOGY SS14).
  - Vector: msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29 (gated,
    hf-token secret), DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-
    L10-38-a4.gguf. The file carries glp.alpha_default=6.0 (the rig's
    serving dose); the BENCHMARK.md reference row is alpha=4.0, so the
    steered arm sets WEIGHTLESS_STEER_ALPHA=4.0 explicitly.

Two arms, one deploy each (alpha is a model-init buffer; no runtime override):
    modal run cloud_serve_dsv4.py::ensure_vector    # CPU, seconds
    modal run cloud_serve_dsv4.py::ensure_weights   # CPU, cached on volume
    modal run cloud_serve_dsv4.py::preflight        # CPU, entry point + gguf
    WEIGHTLESS_STEER_ALPHA=0.0 modal deploy cloud_serve_dsv4.py
    python3 modal/eval_dsv4_plugin.py --base-url <url> --alpha 0.0
    modal app stop weightless-dsv4-plugin-test
    WEIGHTLESS_STEER_ALPHA=4.0 modal deploy cloud_serve_dsv4.py
    python3 modal/eval_dsv4_plugin.py --base-url <url> --alpha 4.0
    modal app stop weightless-dsv4-plugin-test

Chat rendering (run-1 finding, 2026-09-19): this image serves DSV4 chat
through vllm/renderers/deepseek_v4.py + vllm/tokenizers/deepseek_v4.py —
DeepSeek's NATIVE encode_messages, with thinking DEFAULTED ON per request.
The --chat-template jinja flag is accepted by the renderer but never
consulted by the tokenizer override, so run 1 served thinking-on and the
arm was invalid against the reference protocol (every reference lane runs
thinking off). The fix lives in the eval driver
(chat_template_kwargs={"enable_thinking": false} per request), NOT in a
server flag: the jinja (modal/dsv4_chat_template.jinja, the reference
lane's exact file) is kept for offline-lane parity only and is
deliberately NOT passed to the server.

First boot runs WITHOUT --enforce-eager on purpose: the adapter is meant to
serve in the stock compilation mode (the 20260905 lane ran eager because
its probe was eager-only, not because the model needs it). If output is
garbage or the steering-active line is missing in compiled mode, redeploy
with DSV4_ENFORCE_EAGER=1 to isolate.

Boot evidence: the plugin's own INFO lines may never render under vllm's
dictConfig (it attaches a handler only to the `vllm` logger; the 2026-09-18
glm5next debug session proved a steered boot can show zero plugin lines).
serve() therefore prepends /opt/weightless-shim to the vllm process's
PYTHONPATH: modal/sitecustomize_dsv4.py auto-imports at EVERY python
startup, runs register() itself, pre-parses the vector through
SteeringCore.from_env (fail-closed), and prints WEIGHTLESS-SHIM markers
straight to stderr.

Cost discipline: max_containers=1, scaledown_window=180, stop the app when
done. Expected spend: 2 boots x (load + eval) x 4 H100.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]           # weightless/
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b"  # matches GLP-29 glp.base_revision
SERVED_MODEL = "dsv4-flash"
VOLUME_NAME = "dsv4-0731"
VECTOR_REPO = "msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29"
VECTOR_FILE = "DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4.gguf"
VECTOR_PATH = "/data/vector/" + VECTOR_FILE
IMAGE = "vllm/vllm-openai:deepseekv4-flash-vision"
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
PY = "/usr/bin/python3.12"

# Arm/shape switches, read at deploy time (same convention as cloud_serve_glm53).
GPU = os.environ.get("DSV4_GPU", "H100:4")
TP = os.environ.get("DSV4_TP", "4")
KV_DTYPE = os.environ.get("DSV4_KV_DTYPE", "fp8")   # fp8_ds_mla requires it
ENFORCE_EAGER = os.environ.get("DSV4_ENFORCE_EAGER", "")  # "1" to isolate
GMU = os.environ.get("DSV4_GMU", "0.90")
ALPHA = os.environ.get("WEIGHTLESS_STEER_ALPHA", "0.0")

app = modal.App("weightless-dsv4-plugin-test")
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
         .add_local_file(ROOT / "modal" / "vllm_logging_config.json",
                         "/work/vllm_logging_config.json", copy=True)
         .add_local_file(ROOT / "modal" / "sitecustomize_dsv4.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         .run_commands(
             f"{PY} -m pip --version || {PY} -m ensurepip --upgrade",
             "cd /opt/weightless-src/vllm-plugin && "
             f"{PY} -m pip install --no-deps ."))


@app.function(image=download_image, volumes={"/data": vol},
              timeout=24 * 3600, secrets=[hf_secret], env=ENV)
def ensure_weights():
    """Resume/cache the ~168 GB FP8 snapshot on CPU; no GPU allocation.

    Already on the volume from the 20260905 lane (revision 7872f01b) — this
    is a verification pass. Fail-closed size gate: index present, zero
    .incomplete blobs, >150 GB before any GPU spend. Repo is public; the
    secret rides along for rate limits only."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID, revision=REVISION)
    snap = Path(snapshot)
    shards = sorted(snap.glob("*.safetensors"))
    assert (snap / "model.safetensors.index.json").is_file(), "index missing"
    hub = (Path(os.environ["HF_HOME"]) / "hub"
           / "models--deepseek-ai--DeepSeek-V4-Flash-0731")
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    print(f"shards: {len(shards)}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e9:.2f} GB", flush=True)
    assert shards, "no safetensors in the snapshot"
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 150e9, f"size-gate failed: {total / 1e9:.1f} GB"
    vol.commit()
    print("weights ready:", snapshot, flush=True)


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              secrets=[hf_secret], env=ENV)
def ensure_vector():
    """Fetch the gated GLP-29 GGUF and stage it at the stable VECTOR_PATH.

    The HF-cache snapshot path embeds a revision hash; the server env needs a
    path known at deploy time, so the file is copied out of the cache.
    Fail-closed: <200 KB means a truncated or LFS-pointer download (the real
    file is 29 directions x 4096 f32 ~= 475 KB + metadata)."""
    from huggingface_hub import hf_hub_download

    dst = Path(VECTOR_PATH)
    if not dst.is_file():
        fetched = hf_hub_download(repo_id=VECTOR_REPO, filename=VECTOR_FILE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(fetched).read_bytes())
        vol.commit()
    size = dst.stat().st_size
    assert size > 2e5, f"{VECTOR_PATH} looks truncated ({size} B)"
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
want = ("DeepseekV4ForCausalLM",
        "weightless_steer.archs.dsv4:SteeredDeepseekV4ForCausalLM")
assert want in calls, f"shadow registration missing; register_model calls: {calls}"
print("SHADOW OK: register_model called with", want)
# The real compatibility gate: import the adapter against the image's vLLM.
import weightless_steer.archs.dsv4 as m
print("ADAPTER IMPORT OK:", m.SteeredDeepseekV4ForCausalLM.__mro__[1].__module__)
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
os.environ["WEIGHTLESS_STEER_ALPHA"] = "4.0"
from weightless_steer.core import SteeringCore
core = SteeringCore.from_env(hook="ffn_out_pre_residual",
                             num_layers=43, hidden_size=4096)
assert core is not None and len(core.dirs) == 29, core
assert min(core.dirs) == 10 and max(core.dirs) == 38, sorted(core.dirs)
assert core.alpha == 4.0, core.alpha
print("CORE OK: real GLP-29 GGUF parses, 29 dirs on layers 10..38, alpha 4.0")
"""


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version and the adapter's upstream import path;
    2. pip metadata: the `vllm.general_plugins` entry point is registered;
    3. vLLM's real plugin loader runs register() and the registry resolves
       DeepseekV4ForCausalLM to the STEERED class when WEIGHTLESS_STEER_PATH
       is set (this import-drags the adapter against the image's vLLM — the
       anchor/API-mismatch failure mode dies here, on CPU);
    4. with the env unset the registry stays stock (installed-but-
       unconfigured has zero effect);
    5. the real GLP-29 GGUF parses through SteeringCore.from_env at the
       model's geometry (43 layers, 4096-wide pre-fold FFN write).
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
         "import vllm.models.deepseek_v4.nvidia.model as m; "
         "print('deepseek_v4 at', m.__file__)"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "vllm.models.deepseek_v4.nvidia.model import failed"

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
        "--dtype", "bfloat16",
        "--max-model-len", "32768",          # eval speed; full 1M not needed
        "--gpu-memory-utilization", GMU,
        "--max-num-seqs", "8",
        "--port", "8000",
    ]
    if KV_DTYPE:
        # fp8_ds_mla asserts on the default "auto" at engine init
        # (METHODOLOGY SS14) — pass fp8 explicitly.
        cmd += ["--kv-cache-dtype", KV_DTYPE]
    if ENFORCE_EAGER:
        cmd += ["--enforce-eager"]
    return cmd


def _model_snapshot_path() -> str:
    """Resolve the on-volume snapshot dir with stdlib only (huggingface_hub
    is not importable from Modal's runtime python on this image)."""
    import glob

    snaps = sorted(glob.glob(
        "/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/*"))
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
                       # This config puts a root INFO handler in every
                       # process; the stderr shim below is the primary
                       # evidence channel either way.
                       VLLM_CONFIGURE_LOGGING="1",
                       VLLM_LOGGING_CONFIG_PATH="/work/vllm_logging_config.json",
                       # The serve cmd is built INSIDE the container, where
                       # module-level os.environ.get() re-runs with the
                       # container's env — deploy-shell overrides only reach
                       # it through this dict (learned the hard way on the
                       # glm53 lane).
                       DSV4_KV_DTYPE=KV_DTYPE,
                       DSV4_GMU=GMU,
                       DSV4_ENFORCE_EAGER=ENFORCE_EAGER))
@modal.web_server(port=8000, startup_timeout=3600)
def serve():
    """vllm serve with the plugin's entry point active.

    Alpha is baked into the steering buffer at model init (the plugin refuses
    per-request controls), so one deploy = one arm. The model is served from
    the resolved on-volume snapshot dir (HF_HUB_OFFLINE=1: no network).

    vllm's stdout goes to a file ON THE VOLUME in APPEND mode (Modal
    auto-restarts crashed web tasks; truncate mode erased a previous
    attempt's dying words once already on the glm53 lane; Modal's app-log
    window is also just the last 100 entries). No mid-boot vol.commit():
    boots died mid weight-load with 30 s commits running while commit-free
    boots were healthy — the commit is suspected of stalling the 9P mount
    under a >150 GB read stream. The log is committed on the death path and
    from the SIGTERM handler (so `modal app stop` preserves it).

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
    log_dir = "/data/out-dsv4-plugin-test"
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
                                # module config mess on these day-0 builds
                                # makes the plugin's own INFO lines
                                # unreliable as boot evidence. See
                                # modal/sitecustomize_dsv4.py.
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
