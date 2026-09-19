"""Inkling-Small-NVFP4 + the weightless-steer vLLM plugin: GPU boot test.

The plugin (vllm-plugin/, dist `weightless-steer`) shadows BOTH inkling entry
classes in vLLM's ModelRegistry through the `vllm.general_plugins` entry
point — InklingForCausalLM and InklingForConditionalGeneration, the latter
being what thinkingmachines/Inkling-Small-NVFP4 actually resolves to (the
multimodal wrapper builds InklingModel directly via _build, NOT through the
registry, so a text-class-only shadow would serve this checkpoint silently
unsteered). No in-container file patching, unlike the hotfix lane this test
is compared against (patches/hotfix-inkling-steering-projective.py; reference
numbers in refusal-research/experiments/20260901-inkling-small-glp/RESULT.md:
refusal32 0/32 stock -> 30/32 at alpha 0.25, benign32 30/32).

Stack (all validated facts from the 2026-09-01/02 reference lane):
  - Image: vllm/vllm-openai:v0.28.0 (ships vllm/models/inkling/nvidia/
    model.py, the module the adapter imports; vendored byte-identical at
    patches/reference/inkling_v0280.py).
  - Model: thinkingmachines/Inkling-Small-NVFP4 (~266B MoE, 159 GiB NVFP4,
    10 shards), already cached on the `inkling-small` volume (mounted at
    /data; this lane writes nothing there — outputs go to the per-arch
    /out volume).
  - Shape: H100:4, TP4. NOT H100:1 — the checkpoint is 159 GiB, a single
    80 GiB H100 cannot hold it; the assignment's "H100:1" is physically
    impossible for this arch and TP4 is the reference lane's validated shape.
  - Vector: msuiche/Inkling-Small-abliterated-cyber-GLP-41 (gated, hf-token
    secret), Inkling-Small-abliterated-cyber-GLP-41-L1-41-a0.25.gguf, layers
    1-41, width 4096. alpha=0.25 is calibrated and is the file's
    glp.alpha_default; alpha 0.5 already garbles this model — the most
    dose-sensitive in the program. NEVER exceed 0.25.
  - VLLM_USE_V2_MODEL_RUNNER=1: the reference lane's env (Inkling's sconv
    metadata plumbing lives in the MRV2 runner).

Two arms, one deploy each (alpha is a model-init buffer; no runtime override):
    modal run cloud_serve_inkling.py::ensure_vector    # CPU, seconds
    modal run cloud_serve_inkling.py::ensure_weights   # CPU, verify-only
    modal run cloud_serve_inkling.py::preflight        # CPU, entry point + gguf
    WEIGHTLESS_STEER_ALPHA=0.0 modal deploy cloud_serve_inkling.py
    python3 modal/eval_inkling_plugin.py --base-url <url> --alpha 0.0
    modal app stop weightless-inkling-plugin-test
    WEIGHTLESS_STEER_ALPHA=0.25 modal deploy cloud_serve_inkling.py
    python3 modal/eval_inkling_plugin.py --base-url <url> --alpha 0.25
    modal app stop weightless-inkling-plugin-test

First boot runs WITHOUT --enforce-eager on purpose (the reference lane ran
eager, so compiled mode is the untested variable; the glm5next plugin lane
wants the compile path exercised). If output is garbage or the steering line
is missing in compiled mode, redeploy with INK_ENFORCE_EAGER=1 to isolate.

Logging trap, inherited from the glm5next lane (2026-09-18): vLLM's
dictConfig attaches a handler ONLY to the `vllm` logger (propagate=False), so
the plugin's INFO lines ("weightless GLP steering active ...") may never
render even when steering works. The sitecustomize shim
(modal/sitecustomize_inkling.py -> /opt/weightless-shim/sitecustomize.py,
PYTHONPATH-prepended for the vllm process only) re-runs register() at every
interpreter start, pre-parses the vector fail-closed, and prints
WEIGHTLESS-SHIM markers straight to stderr — boot evidence no logging config
can eat. "No log line" is no information, not failure.

Cost discipline: max_containers=1, scaledown_window=180, stop the app the
moment no driver is running against it; verify `modal container list` empty
at the end. Expected spend: 2 boots x ~1 h x 4 H100 (weights are warm).
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]           # weightless/
MODEL_ID = "thinkingmachines/Inkling-Small-NVFP4"
SERVED_MODEL = "inkling-small-nvfp4"
DATA_VOLUME = "inkling-small"                        # reference lane's cache
VOLUME_NAME = "weightless-inkling-plugin-test"       # this lane's outputs
VECTOR_REPO = "msuiche/Inkling-Small-abliterated-cyber-GLP-41"
VECTOR_FILE = "Inkling-Small-abliterated-cyber-GLP-41-L1-41-a0.25.gguf"
# sha256 of the file as actually published in the gated repo. The
# derivation-era local copy (refusal-research/experiments/20260901-...
# /out/) hashes differently (4bebe7a6...) because the publish re-export
# added provenance metadata — the direction tensors are byte-identical
# between the two and both carry glp.content_sha256 2a229d56... (the hash
# that covers the directions), verified 2026-09-19 by parsing both files
# through weightless_steer.container.load_control_vector.
VECTOR_SHA256 = ("e0d9f72dc611b285ef2ac1db1faa8444d3606dbb096693084a1e5551"
                 "280dac46")
VECTOR_PATH = "/out/vector/" + VECTOR_FILE
IMAGE = "vllm/vllm-openai:v0.28.0"
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
PY = "/usr/bin/python3.12"

# Arm/shape switches, read at deploy time (same convention as
# cloud_serve_glm53).
GPU = os.environ.get("INK_GPU", "H100:4")
TP = os.environ.get("INK_TP", "4")
ENFORCE_EAGER = os.environ.get("INK_ENFORCE_EAGER", "")  # "1" to isolate
GMU = os.environ.get("INK_GMU", "0.92")
ALPHA = os.environ.get("WEIGHTLESS_STEER_ALPHA", "0.0")

app = modal.App("weightless-inkling-plugin-test")
data_vol = modal.Volume.from_name(DATA_VOLUME)   # warm HF cache; do not write
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
         .add_local_file(ROOT / "modal" / "inkling_chat_template.jinja",
                         "/work/chat_template.jinja", copy=True)
         .add_local_file(ROOT / "modal" / "vllm_logging_config.json",
                         "/work/vllm_logging_config.json", copy=True)
         .add_local_file(ROOT / "modal" / "sitecustomize_inkling.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         .run_commands(
             f"{PY} -m pip --version || {PY} -m ensurepip --upgrade",
             "cd /opt/weightless-src/vllm-plugin && "
             f"{PY} -m pip install --no-deps ."))


@app.function(image=download_image, volumes={"/data": data_vol},
              timeout=3600, secrets=[hf_secret], env=ENV)
def ensure_weights():
    """Verify the warm 159 GiB NVFP4 snapshot; download only if incomplete.

    The reference lane cached and served from this snapshot on 2026-09-01/02,
    so this is a fail-closed gate, not a fetch: 10 safetensors + index, zero
    .incomplete blobs, >150 GB. If the gate fails the function resumes the
    download (ungated repo; the secret rides along for rate limits).
    """
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID)   # offline-complete: no traffic
    snap = Path(snapshot)
    shards = sorted(snap.glob("*.safetensors"))
    assert (snap / "model.safetensors.index.json").is_file(), "index missing"
    hub = (Path(os.environ["HF_HOME"]) / "hub"
           / "models--thinkingmachines--Inkling-Small-NVFP4")
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    print(f"shards: {len(shards)}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e9:.2f} GB", flush=True)
    assert len(shards) == 10, f"expected 10 safetensors, found {len(shards)}"
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 150e9, f"size-gate failed: {total / 1e9:.1f} GB"
    data_vol.commit()
    print("weights ready:", snapshot, flush=True)


@app.function(image=download_image, volumes={"/out": vol}, timeout=900,
              secrets=[hf_secret], env={"HF_HOME": "/out/hf"})
def ensure_vector():
    """Fetch the gated GLP-41 GGUF and stage it at the stable VECTOR_PATH.

    The HF-cache snapshot path embeds a revision hash; the server env needs a
    path known at deploy time, so the file is copied out of the cache.
    Fail-closed twice: <500 KB means a truncated or LFS-pointer download
    (real file is ~660 KB), and the sha256 must equal the verified local
    copy's (a wrong vector is worse than no vector).
    """
    import hashlib

    from huggingface_hub import hf_hub_download

    dst = Path(VECTOR_PATH)
    if not dst.is_file():
        fetched = hf_hub_download(repo_id=VECTOR_REPO, filename=VECTOR_FILE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(fetched).read_bytes())
        vol.commit()
    data = dst.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert len(data) > 500e3, f"{VECTOR_PATH} looks truncated ({len(data)} B)"
    assert digest == VECTOR_SHA256, f"{VECTOR_PATH} sha256 {digest} != " \
        f"{VECTOR_SHA256} — refusing to serve an unverified vector"
    print(f"vector ready: {VECTOR_PATH} ({len(data)} B, sha256 ok)", flush=True)


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
want = [
    ("InklingForCausalLM",
     "weightless_steer.archs.inkling:SteeredInklingForCausalLM"),
    ("InklingForConditionalGeneration",
     "weightless_steer.archs.inkling:SteeredInklingForConditionalGeneration"),
]
for w in want:
    assert w in calls, f"shadow registration missing: {w}; calls: {calls}"
print("SHADOW OK: register_model called with", want)
# The real compatibility gate: import the adapter against the image's vLLM.
import weightless_steer.archs.inkling as g
print("ADAPTER IMPORT OK:",
      g.SteeredInklingForCausalLM.__mro__[1].__module__,
      g.SteeredInklingForConditionalGeneration.__mro__[1].__module__)
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

# NOTE: WEIGHTLESS_STEER_ALPHA deliberately unset — the published file's
# glp.alpha_default=0.25 must resolve on its own (the env-unset boot IS the
# calibrated-dose boot; the hotfix defaulted to 1.0, which garbles Inkling).
_PREFLIGHT_CORE = f"""
import os
os.environ["WEIGHTLESS_STEER_PATH"] = {VECTOR_PATH!r}
os.environ.pop("WEIGHTLESS_STEER_ALPHA", None)
from weightless_steer.core import SteeringCore
core = SteeringCore.from_env(hook="residual_stream_post_layer",
                             num_layers=42, hidden_size=4096)
assert core is not None and len(core.dirs) == 41, core
assert min(core.dirs) == 1 and max(core.dirs) == 41, sorted(core.dirs)
assert core.alpha == 0.25, core.alpha
print("CORE OK: real GGUF parses, 41 dirs on layers 1..41, "
      "alpha_default 0.25 resolves with env unset")
"""


@app.function(image=image, volumes={"/data": data_vol, "/out": vol},
              timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version and the adapter's upstream import path;
    2. pip metadata: the `vllm.general_plugins` entry point is registered;
    3. with the env unset the registry stays stock (installed-but-
       unconfigured has zero effect);
    4. vLLM's real plugin loader runs register() and BOTH inkling arch
       names shadow to the steered classes when WEIGHTLESS_STEER_PATH is
       set, then the adapter imports against the image's vLLM — an
       anchor/API mismatch dies here, on CPU;
    5. the real GLP-41 GGUF parses through SteeringCore.from_env at the
       model's geometry (42 layers, 4096-wide stream) and the calibrated
       alpha resolves from the file.
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
         "import vllm.models.inkling.nvidia.model as m; "
         "print('inkling at', m.__file__)"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "vllm.models.inkling.nvidia.model import failed"

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
        "--max-model-len", "16384",          # the reference lane's len
        "--gpu-memory-utilization", GMU,
        "--tokenizer-mode", "inkling",       # validated lane flags
        "--trust-remote-code",
        "--chat-template", "/work/chat_template.jinja",
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
        "/data/hf/hub/models--thinkingmachines--Inkling-Small-NVFP4"
        "/snapshots/*"))
    assert len(snaps) == 1, f"expected exactly one snapshot, got {snaps}"
    assert os.path.isdir(snaps[0]) and \
        any(f.endswith(".safetensors") for f in os.listdir(snaps[0])), snaps
    return snaps[0]


@app.function(image=image, volumes={"/data": data_vol, "/out": vol}, gpu=GPU,
              min_containers=0, max_containers=1, scaledown_window=180,
              startup_timeout=3600, timeout=6 * 3600,
              env=dict(ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       VLLM_USE_V2_MODEL_RUNNER="1",
                       # compile caches on MY volume: arm B's boot reuses
                       # arm A's (identical graph, one less compile)
                       TORCHINDUCTOR_CACHE_DIR="/out/cache/inductor",
                       VLLM_CACHE_ROOT="/out/cache/vllm",
                       TRITON_CACHE_DIR="/out/cache/triton",
                       WEIGHTLESS_STEER_PATH=VECTOR_PATH,
                       WEIGHTLESS_STEER_ALPHA=ALPHA,
                       # Force third-party INFO logs (the plugin's
                       # steering-active line) to print in EVERY process —
                       # vllm's default dictConfig only attaches a handler to
                       # the "vllm" logger (propagate=False); third-party
                       # loggers fall through to root, which stays WARNING.
                       VLLM_CONFIGURE_LOGGING="1",
                       VLLM_LOGGING_CONFIG_PATH="/work/vllm_logging_config.json"))
@modal.web_server(port=8000, startup_timeout=3600)
def serve():
    """vllm serve with the plugin's entry point active.

    Alpha is baked into the steering buffer at model init (the plugin refuses
    per-request controls), so one deploy = one arm. The model path is the
    resolved snapshot dir (HF_HUB_OFFLINE=1: the path is on the volume, no
    network).

    vllm's stdout goes to a file ON MY VOLUME in APPEND mode (Modal
    auto-restarts crashed web tasks; truncate mode erased a dying boot's last
    words once already on the glm53 lane; Modal's app-log window is also just
    the last 100 entries). No mid-boot vol.commit(): two glm53 boots died mid
    weight-load with 30 s commits running — the commit is suspected of
    stalling the 9P mount under a 159 GiB read stream. The log is committed
    on the death path and from the SIGTERM handler (so `modal app stop`
    preserves it).

    The function MUST NOT block: Modal's web_server only starts routing
    traffic to the port once the function returns (a blocking monitor left
    two healthy boots unreachable on the glm53 lane, 2026-09-17/18). It
    watches the boot log for the plugin's steering-active line (fail-closed:
    the line missing by the deadline = unsteered = failed boot), then returns
    so routing can start; uvicorn binds :8000 many minutes later (the
    reference lane's engine load alone was ~35 min on the 9P volume).
    """
    import signal
    import subprocess
    import time

    model_path = _model_snapshot_path()
    cmd = _serve_cmd(model_path)
    log_dir = "/out/out-inkling-plugin-test"
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
                                # module config mess makes the plugin's own
                                # INFO lines unreliable as boot evidence.
                                # See modal/sitecustomize_inkling.py.
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
