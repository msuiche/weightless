"""ByteDance Ouro-2.6B + the weightless-steer vLLM plugin: GPU boot test.

The plugin (vllm-plugin/, dist `weightless-steer`) shadows OuroForCausalLM
in vLLM's ModelRegistry through the `vllm.general_plugins` entry point — no
in-container file patching, unlike the hotfix lane this test is compared
against (patches/hotfix-ouro-steering-projective.py; reference numbers from
refusal-research/experiments/20260906-ouro-glp/STATE.md). This app boots the
v0.26.0 serving image — the LAST vLLM image carrying OuroForCausalLM
natively (removed upstream in #49786 ~2h after the image was built) — with
the plugin pip-installed and the published GLP-192 vector, and exposes an
OpenAI-compatible endpoint for the local eval driver
(modal/eval_ouro_plugin.py).

Stack (all validated facts, see the experiment STATE.md):
  - Image: vllm/vllm-openai:v0.26.0 EXACTLY (newer vLLM refuses the arch).
  - Model: ByteDance/Ouro-2.6B (single safetensors, ~5.2 GB bf16; ungated;
    48 physical layers x total_ut_steps=4 = 192 execution steps, hidden
    2048, KV 192 slots ~1.5 MB/token).
  - Shape: H100:1, TP=1.
  - Vector: msuiche/Ouro-2.6B-abliterated-cyber-GLP-192-L1-192-a1.0 (gated,
    hf-token secret), glp.ouro26-GLP-192-L1-192-a1.gguf, 192 directions of
    width 2048, container convention direction.N = execution step N-1.
    alpha=1.0 is the shipped dose; usable band [0.5, 2.0], alpha=3.0
    collapses.
  - Reference row (hotfix lane, greedy, mt4096, n=32): stock refusal32
    C4/D1/R27, cyber32 C30/R2, benign32 C32; a0.0 no-op gate refusal32
    C5/R27, cyber32 C28/D2/R2, benign32 C32; a1.0 refusal32 C32/32, cyber32
    C31/D1, benign32 C32.

Two arms, one deploy each (alpha is a model-init buffer; no runtime override):
    modal run cloud_serve_ouro.py::ensure_vector    # CPU, seconds
    modal run cloud_serve_ouro.py::ensure_weights   # CPU, ~5.2 GB once
    modal run cloud_serve_ouro.py::preflight        # CPU, entry point + gguf
    WEIGHTLESS_STEER_ALPHA=0.0 modal deploy cloud_serve_ouro.py
    python3 modal/eval_ouro_plugin.py --base-url <url> --alpha 0.0
    modal app stop weightless-ouro-plugin-test
    WEIGHTLESS_STEER_ALPHA=1.0 modal deploy cloud_serve_ouro.py
    python3 modal/eval_ouro_plugin.py --base-url <url> --alpha 1.0
    modal app stop weightless-ouro-plugin-test

First boot runs WITHOUT --enforce-eager on purpose: OuroModel is
@support_torch_compile-decorated at v0.26.0, so the adapter's compile
rebind is live and wants the exercise. If output is garbage or the
steering-active line is missing in compiled mode, redeploy with
OURO_ENFORCE_EAGER=1 to isolate.

Logging evidence: same discipline as the glm53 lane — vllm's dictConfig
attaches a handler only to the `vllm` logger, so plugin INFO lines
("weightless GLP steering active ...") may never render even when steering
works. serve() prepends /opt/weightless-shim to the vllm process's
PYTHONPATH: modal/sitecustomize_ouro.py auto-imports at EVERY python
startup, runs register() itself, pre-parses the vector through
exec_core_from_env (fail-closed), and prints WEIGHTLESS-SHIM markers
straight to stderr — boot evidence no logging config can eat.

Compile-cache discipline (the 2026-09-06 cublas incident on THIS lane):
every boot that READ a volume-cached inductor artifact written by another
boot died at cublasCreate inside the cached graph. Workaround is
structural: compile caches live in container-local /tmp, never on the
volume — no boot ever reads another boot's cache. Cold compile costs
~2-3 min/boot.

Cost discipline: max_containers=1, scaledown_window=180, stop the app when
done. Expected spend: 2 boots x (load + ~3 min compile + eval) x 1 H100.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]           # weightless/
MODEL_ID = "ByteDance/Ouro-2.6B"
SERVED_MODEL = "ouro-2.6b"
VOLUME_NAME = "ouro-plugin-test"
VECTOR_REPO = "msuiche/Ouro-2.6B-abliterated-cyber-GLP-192-L1-192-a1.0"
VECTOR_FILE = "glp.ouro26-GLP-192-L1-192-a1.gguf"
VECTOR_PATH = "/data/vector/" + VECTOR_FILE
IMAGE = "vllm/vllm-openai:v0.26.0"
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
PY = "/usr/bin/python3.12"

# Ouro-2.6B geometry: 48 physical layers x 4 UT passes, hidden 2048.
NUM_PHYSICAL_LAYERS = 48
TOTAL_UT_STEPS = 4
EXEC_STEPS = NUM_PHYSICAL_LAYERS * TOTAL_UT_STEPS    # 192
HIDDEN_SIZE = 2048

# Arm/shape switches, read at deploy time (same convention as cloud_serve_glm53).
GPU = os.environ.get("OURO_GPU", "H100:1")
TP = os.environ.get("OURO_TP", "1")
ENFORCE_EAGER = os.environ.get("OURO_ENFORCE_EAGER", "")  # "1" to isolate
GMU = os.environ.get("OURO_GMU", "0.85")   # the experiment lane's hedge
ALPHA = os.environ.get("WEIGHTLESS_STEER_ALPHA", "0.0")

app = modal.App("weightless-ouro-plugin-test")
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
         .add_local_file(ROOT / "modal" / "sitecustomize_ouro.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         .run_commands(
             f"{PY} -m pip --version || {PY} -m ensurepip --upgrade",
             "cd /opt/weightless-src/vllm-plugin && "
             f"{PY} -m pip install --no-deps ."))


@app.function(image=download_image, volumes={"/data": vol},
              timeout=24 * 3600, secrets=[hf_secret], env=ENV)
def ensure_weights():
    """Resume/cache the ~5.2 GB snapshot on CPU; no GPU allocation.

    Fail-closed size gate: exactly one model.safetensors, zero .incomplete
    blobs, >4 GB before any GPU spend. Repo is ungated; the secret rides
    along for rate limits only."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID)
    snap = Path(snapshot)
    shards = sorted(snap.glob("*.safetensors"))
    hub = (Path(os.environ["HF_HOME"]) / "hub"
           / "models--ByteDance--Ouro-2.6B")
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    print(f"shards: {len(shards)}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e9:.2f} GB", flush=True)
    assert len(shards) == 1, f"expected 1 safetensors, found {len(shards)}"
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 4e9, f"size-gate failed: {total / 1e9:.1f} GB"
    vol.commit()
    print("weights ready:", snapshot, flush=True)


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              secrets=[hf_secret], env=ENV)
def ensure_vector():
    """Fetch the gated GLP-192 GGUF and stage it at the stable VECTOR_PATH.

    The HF-cache snapshot path embeds a revision hash; the server env needs a
    path known at deploy time, so the file is copied out of the cache.
    Fail-closed: <1 MB means a truncated or LFS-pointer download (real file
    is 1,584,672 B)."""
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
want = ("OuroForCausalLM",
        "weightless_steer.archs.ouro:SteeredOuroForCausalLM")
assert want in calls, f"shadow registration missing; register_model calls: {calls}"
print("SHADOW OK: register_model called with", want)
# The real compatibility gate: import the adapter against the image's vLLM.
import weightless_steer.archs.ouro as o
print("ADAPTER IMPORT OK:", o.SteeredOuroForCausalLM.__mro__[1].__module__)
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
os.environ["WEIGHTLESS_STEER_ALPHA"] = "1.0"
from weightless_steer.archs.ouro import exec_core_from_env
core = exec_core_from_env(hook="residual_stream_post_layer",
                          exec_steps={EXEC_STEPS}, hidden_size={HIDDEN_SIZE})
assert core is not None and len(core.dirs) == {EXEC_STEPS}, len(core.dirs)
# The looped container shift: direction.1..192 -> exec steps 0..191.
assert min(core.dirs) == 0 and max(core.dirs) == {EXEC_STEPS - 1}, \\
    (min(core.dirs), max(core.dirs))
assert core.alpha == 1.0, core.alpha
print("CORE OK: real GGUF parses, 192 dirs on exec steps 0..191, alpha 1.0")
"""


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version and that the ouro arch module EXISTS in it
       (the whole reason for the v0.26.0 pin — newer images refuse);
    2. pip metadata: the `vllm.general_plugins` entry point is registered;
    3. vLLM's real plugin loader runs register() and the registry shadows
       OuroForCausalLM to the STEERED class when WEIGHTLESS_STEER_PATH is
       set (this import-drags the adapter against the image's vLLM — the
       API-mismatch failure mode dies here, on CPU);
    4. with the env unset the registry stays stock (installed-but-
       unconfigured has zero effect);
    5. the real GLP-192 GGUF parses through exec_core_from_env at the
       model's geometry (192 execution steps, 2048-wide plain stream) —
       including the direction.N -> exec N-1 container shift.
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
         "import vllm.model_executor.models.ouro as m; "
         "print('ouro at', m.__file__)"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "vllm.model_executor.models.ouro import failed"
    assert r.stdout.splitlines()[0].endswith("0.26.0"), \
        f"not the v0.26.0 image: {r.stdout.splitlines()[0]}"

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
        "--max-model-len", "8192",           # eval speed; 64k not needed
        "--gpu-memory-utilization", GMU,
        "--max-num-seqs", "8",               # the KV-constrained eval batch
        "--port", "8000",
        # No --chat-template override: Ouro-2.6B ships ChatML in
        # tokenizer_config.json (verified in the reference lane), and the
        # base model has no think block to suppress.
    ]
    if ENFORCE_EAGER:
        cmd += ["--enforce-eager"]
    return cmd


def _model_snapshot_path() -> str:
    """Resolve the on-volume snapshot dir with stdlib only (huggingface_hub
    is not importable from Modal's runtime python on this image)."""
    import glob

    snaps = sorted(glob.glob(
        "/data/hf/hub/models--ByteDance--Ouro-2.6B/snapshots/*"))
    assert len(snaps) == 1, f"expected exactly one snapshot, got {snaps}"
    assert os.path.isdir(snaps[0]) and \
        any(f.endswith(".safetensors") for f in os.listdir(snaps[0])), snaps
    return snaps[0]


@app.function(image=image, volumes={"/data": vol}, gpu=GPU,
              min_containers=0, max_containers=1, scaledown_window=180,
              startup_timeout=3600, timeout=6 * 3600,
              env=dict(ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       # Compile caches are CONTAINER-LOCAL on this lane:
                       # every boot that read a volume-cached inductor
                       # artifact written by another boot died at
                       # cublasCreate (the 2026-09-06 incident); /tmp costs
                       # one cold compile per boot (~2-3 min) instead.
                       TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-cache",
                       VLLM_CACHE_ROOT="/tmp/vllm-cache",
                       TRITON_CACHE_DIR="/tmp/triton-cache",
                       WEIGHTLESS_STEER_PATH=VECTOR_PATH,
                       WEIGHTLESS_STEER_ALPHA=ALPHA,
                       # Force third-party INFO logs (the plugin's
                       # steering-active line) to print in EVERY process —
                       # vllm's default dictConfig only attaches a handler to
                       # the "vllm" logger (propagate=False); third-party
                       # loggers fall through to root, which stays WARNING.
                       # This config puts a root INFO handler in every
                       # process. The stderr shim below is the evidence
                       # channel that no logging config can eat.
                       VLLM_CONFIGURE_LOGGING="1",
                       VLLM_LOGGING_CONFIG_PATH="/work/vllm_logging_config.json"))
@modal.web_server(port=8000, startup_timeout=3600)
def serve():
    """vllm serve with the plugin's entry point active.

    Alpha is baked into the steering buffer at model init (the plugin refuses
    per-request controls), so one deploy = one arm.

    vllm's stdout goes to a file ON THE VOLUME in APPEND mode (Modal
    auto-restarts crashed web tasks; truncate mode erased a previous
    attempt's dying words once already on the glm53 lane). No mid-boot
    vol.commit(); the log is committed on the death path and from the
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
    log_dir = "/data/out-ouro-plugin-test"
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
                                # and prints stderr markers. See
                                # modal/sitecustomize_ouro.py.
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
