"""Qwen3.8-27B + the weightless-steer vLLM plugin: GPU boot test + mini-eval.

The plugin (vllm-plugin/, dist `weightless-steer`) shadows
Qwen3_5ForConditionalGeneration AND Qwen3_5ForCausalLM in vLLM's
ModelRegistry through the `vllm.general_plugins` entry point — no
in-container file patching, unlike the hotfix lane this test is compared
against (patches/hotfix-qwen38-steering-projective.py, reference numbers in
recipe/qwen/README.md: refusal32 4/32 stock -> 24/32 steered, benign clean,
on the GB10 NVFP4 stack). This app boots the STOCK vllm/vllm-openai:0.28.0
image with the plugin pip-installed and the published GLP-49 vector, and
exposes an OpenAI-compatible endpoint for the local eval driver
(modal/eval_qwen38_plugin.py).

Stack:
  - Image: vllm/vllm-openai:0.28.0 (vanilla; qwen3_5.py + qwen3_next.py ship
    in it — the preflight asserts the adapter's module paths AND the
    vendored qwen3_next_v0280.py anchors in-image before any GPU spend).
  - Model: unsloth/Qwen3.8-27B-NVFP4 (compressed-tensors mixed NVFP4/FP8,
    model.safetensors + model_mtp.safetensors, ~13.5 GB; declares
    Qwen3_5ForConditionalGeneration — multimodal wrapper, vision tower
    rides along; text-only prompts need no mm inputs).
  - Shape: H100:1, TP1 (weights ~13.5 GB).
  - Vector: msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49 (gated, hf-token
    secret), Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf, 49
    directions over layers 10..58, width 5120. alpha=1.0 is calibrated —
    alpha=2 INVERTS this model (37.5% benign refusal, QWEN38-27B.md 5.0.8),
    so the env default here is the file's glp.alpha_default, not a guess.
  - MTP is NOT enabled (no --speculative-config): greedy decoding makes it
    a pure speedup, and leaving it out keeps the boot surface stock.
  - enable_thinking=false via --default-chat-template-kwargs, matching the
    reference protocol (greedy, 400 max new tokens); the checkpoint's own
    chat_template.jinja handles the kwarg.

Two arms, one deploy each (alpha is a model-init buffer; no runtime override):
    modal run cloud_serve_qwen38.py::ensure_vector    # CPU, seconds
    modal run cloud_serve_qwen38.py::ensure_weights   # CPU, ~13.5 GB once
    modal run cloud_serve_qwen38.py::preflight        # CPU, entry point + gguf
    WEIGHTLESS_STEER_ALPHA=0.0 modal deploy cloud_serve_qwen38.py
    python3 modal/eval_qwen38_plugin.py --base-url <url> --alpha 0.0
    modal app stop weightless-qwen38-plugin-test
    modal deploy cloud_serve_qwen38.py                # alpha unset -> file default 1.0
    python3 modal/eval_qwen38_plugin.py --base-url <url> --alpha 1.0
    modal app stop weightless-qwen38-plugin-test

First boot runs WITHOUT --enforce-eager on purpose: Qwen3_5Model is
@support_torch_compile'd, so the adapter's compile-rebind fix is part of
what is being tested. If output is garbage or the steering-active line is
missing in compiled mode, redeploy with QWEN38_ENFORCE_EAGER=1 to isolate.

Boot evidence: the sitecustomize shim (modal/sitecustomize_qwen38.py) is on
the serve process's PYTHONPATH, so every interpreter of the serve stack
re-runs register(), pre-parses the vector fail-closed, and prints
WEIGHTLESS-SHIM markers straight to stderr — the evidence channel no
logging config can eat (vLLM's default dictConfig attaches a handler only
to the `vllm` logger; the plugin's own INFO lines may never render even
when steering works, so "no log line" is no information, not failure).

Cost discipline: max_containers=1, scaledown_window=180, stop the app when
done. Expected spend: 2 boots x (load + eval) x 1 H100.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]           # weightless/
MODEL_ID = "unsloth/Qwen3.8-27B-NVFP4"
SERVED_MODEL = "qwen38"
VOLUME_NAME = "qwen38-plugin-test"
VECTOR_REPO = "msuiche/Qwen3.8-27B-abliterated-cyber-GLP-49"
VECTOR_FILE = "Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf"
VECTOR_PATH = "/data/vector/" + VECTOR_FILE
IMAGE = "vllm/vllm-openai:0.28.0"
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
PY = "/usr/bin/python3.12"

# Arm/shape switches, read at deploy time (same convention as cloud_serve_glm53).
GPU = os.environ.get("QWEN38_GPU", "H100:1")
ENFORCE_EAGER = os.environ.get("QWEN38_ENFORCE_EAGER", "")  # "1" to isolate
GMU = os.environ.get("QWEN38_GMU", "0.92")
# Unset -> the GGUF's glp.alpha_default (1.0). The control arm sets "0.0".
ALPHA = os.environ.get("WEIGHTLESS_STEER_ALPHA", "")

app = modal.App("weightless-qwen38-plugin-test")
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
         .add_local_file(ROOT / "modal" / "sitecustomize_qwen38.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         .run_commands(
             f"{PY} -m pip --version || {PY} -m ensurepip --upgrade",
             "cd /opt/weightless-src/vllm-plugin && "
             f"{PY} -m pip install --no-deps ."))


@app.function(image=download_image, volumes={"/data": vol},
              timeout=6 * 3600, secrets=[hf_secret], env=ENV)
def ensure_weights():
    """Resume/cache the ~13.5 GB NVFP4 snapshot on CPU; no GPU allocation.

    Fail-closed size gate: model.safetensors + model_mtp.safetensors +
    index, zero .incomplete blobs, >12 GB before any GPU spend. Repo is
    ungated; the secret rides along for rate limits only."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID)
    snap = Path(snapshot)
    shards = sorted(snap.glob("*.safetensors"))
    assert (snap / "model.safetensors.index.json").is_file(), "index missing"
    hub = (Path(os.environ["HF_HOME"]) / "hub"
           / "models--unsloth--Qwen3.8-27B-NVFP4")
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    print(f"shards: {len(shards)}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e9:.2f} GB", flush=True)
    assert len(shards) == 2, f"expected 2 safetensors, found {len(shards)}"
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 12e9, f"size-gate failed: {total / 1e9:.1f} GB"
    vol.commit()
    print("weights ready:", snapshot, flush=True)


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              secrets=[hf_secret], env=ENV)
def ensure_vector():
    """Fetch the gated GLP-49 GGUF and stage it at the stable VECTOR_PATH.

    The HF-cache snapshot path embeds a revision hash; the server env needs a
    path known at deploy time, so the file is copied out of the cache.
    Fail-closed: <500 KB means a truncated or LFS-pointer download (the real
    file is 1,007,904 B)."""
    from huggingface_hub import hf_hub_download

    dst = Path(VECTOR_PATH)
    if not dst.is_file():
        fetched = hf_hub_download(repo_id=VECTOR_REPO, filename=VECTOR_FILE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(fetched).read_bytes())
        vol.commit()
    size = dst.stat().st_size
    assert size > 500_000, f"{VECTOR_PATH} looks truncated ({size} B)"
    print(f"vector ready: {VECTOR_PATH} ({size} B)", flush=True)


# The adapter's copied forward, as anchor strings (kept in sync by
# tests/test_archs/test_qwen38.py's StructureTests, which pin them against
# the vendored patches/reference/qwen3_next_v0280.py AND the hotfix's v0.28
# anchor). Here they are checked against the IMAGE's own qwen3_next.py:
# the file the serve actually executes. A drifted image fails here, on CPU,
# before any GPU spend.
_PREFLIGHT_ANCHOR = r"""
import hashlib, sys
p = sys.argv[1]
src = open(p).read()
anchor = (
    "        for layer_idx, layer in enumerate(\n"
    "            islice(self.layers, self.start_layer, self.end_layer),\n"
    "            start=self.start_layer,\n"
    "        ):\n"
    "            hidden_states, residual = layer(\n"
    "                positions=positions,\n"
    "                hidden_states=hidden_states,\n"
    "                residual=residual,\n"
    "            )\n"
    "            self._maybe_add_hidden_state(\n"
    "                aux_hidden_states, layer_idx + 1, hidden_states, residual\n"
    "            )\n"
)
assert anchor in src, f"{p}: the v0.28.0 layer-loop anchor is GONE (drifted upstream)"
tail = (
    "        hidden_states, _ = self.norm(hidden_states, residual)\n"
    "        if self.use_sequence_parallel:\n"
)
assert tail in src, f"{p}: the v0.28.0 forward tail anchor is GONE (drifted upstream)"
sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
print("ANCHOR OK:", p, "sha256", sha)
print("vendored reference sha256 is",
      "5aeac6c81bfa7c680e517106aecb7741a0a4e90fa52de77f0be1eadebc7516fa",
      "(informational; anchors are the gate)")
"""

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
    ("Qwen3_5ForCausalLM",
     "weightless_steer.archs.qwen38:SteeredQwen3_5ForCausalLM"),
    ("Qwen3_5ForConditionalGeneration",
     "weightless_steer.archs.qwen38:SteeredQwen3_5ForConditionalGeneration"),
]
for w in want:
    assert w in calls, f"shadow registration missing: {w}; calls: {calls}"
print("SHADOW OK: register_model called with", want)
# The real compatibility gate: import the adapter against the image's vLLM.
import weightless_steer.archs.qwen38 as q
print("ADAPTER IMPORT OK:", q.SteeredQwen3_5Model.__mro__[1].__module__)
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
                             num_layers=64, hidden_size=5120)
assert core is not None and len(core.dirs) == 49, core
assert min(core.dirs) == 10 and max(core.dirs) == 58, sorted(core.dirs)
assert core.alpha == 1.0, core.alpha
print("CORE OK: real GGUF parses, 49 dirs on layers 10..58, alpha 1.0")
"""


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version, the adapter's upstream import path, and the
       vendored v0.28.0 anchors present in the image's own qwen3_next.py
       (the drift failure mode dies here, not at GPU boot);
    2. pip metadata: the `vllm.general_plugins` entry point is registered;
    3. vLLM's real plugin loader runs register() and BOTH qwen38 shadows
       land (the multimodal wrapper is the arch the checkpoint declares);
       the adapter module import-drags against the image's vLLM;
    4. with the env unset the registry stays stock (installed-but-
       unconfigured has zero effect);
    5. the real GLP-49 GGUF parses through SteeringCore.from_env at the
       model's geometry (64 layers, 5120-wide stream, alpha_default 1.0).
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
         "import vllm.model_executor.models.qwen3_5 as m35; "
         "print('qwen3_5 at', m35.__file__); "
         "import vllm.model_executor.models.qwen3_next as m3n; "
         "print('qwen3_next at', m3n.__file__); "
         "print('has wrapper:', hasattr(m35, 'Qwen3_5ForConditionalGeneration'))"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "vllm.model_executor.models.qwen3_5 import failed"
    assert "has wrapper: True" in r.stdout, r.stdout

    # Anchor check against the image's own copy of the model file.
    r = subprocess.run(
        [PY, "-c",
         "import vllm.model_executor.models.qwen3_next as m; print(m.__file__)"],
        capture_output=True, text=True)
    assert r.returncode == 0, "qwen3_next path probe failed"
    sh(_PREFLIGHT_ANCHOR, r.stdout.strip().splitlines()[-1])

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
        "--max-model-len", "32768",          # eval speed; full 256K not needed
        "--gpu-memory-utilization", GMU,
        "--max-num-seqs", "8",
        # The reference protocol runs with thinking OFF (greedy, 400 max new
        # tokens); the checkpoint's shipped chat_template.jinja handles the
        # kwarg (thinking defaults ON otherwise and eats the budget).
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--port", "8000",
    ]
    if ENFORCE_EAGER:
        cmd += ["--enforce-eager"]
    # GB10/SM121-only items deliberately absent: no kpool bind-mount, no
    # --moe-backend (dense model), no MTP speculative config, no fp8 KV flag.
    return cmd


def _model_snapshot_path() -> str:
    """Resolve the on-volume snapshot dir with stdlib only (huggingface_hub
    is not importable from Modal's runtime python on this image)."""
    import glob

    snaps = sorted(glob.glob(
        "/data/hf/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/*"))
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
                       # Empty string would be an explicit override; omit the
                       # var entirely when unset so the file's
                       # glp.alpha_default (1.0) resolves.
                       **({"WEIGHTLESS_STEER_ALPHA": ALPHA} if ALPHA else {}),
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
                       QWEN38_GMU=GMU,
                       QWEN38_ENFORCE_EAGER=ENFORCE_EAGER))
@modal.web_server(port=8000, startup_timeout=3600)
def serve():
    """vllm serve with the plugin's entry point active.

    Alpha is baked into the steering buffer at model init (the plugin refuses
    per-request controls), so one deploy = one arm.

    vllm's stdout goes to a file ON THE VOLUME in APPEND mode (Modal
    auto-restarts crashed web tasks; truncate mode erased a previous
    attempt's dying words once already; Modal's app-log window is also just
    the last 100 entries). No mid-boot vol.commit(): commits under a heavy
    read stream are suspected of stalling the 9P mount. The log is committed
    on the death path and from the SIGTERM handler (so `modal app stop`
    preserves it).

    The function MUST NOT block: Modal's web_server only starts routing
    traffic to the port once the function returns (a blocking monitor left
    healthy boots unreachable on the glm53 lane — 2026-09-17/18). It watches
    the boot log for the steering-active evidence line (fail-closed: the
    line missing by the deadline = unsteered = failed boot), then returns so
    routing can start; uvicorn binds :8000 a few minutes later."""
    import signal
    import subprocess
    import time

    model_path = _model_snapshot_path()
    cmd = _serve_cmd(model_path)
    log_dir = "/data/out-qwen38-plugin-test"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/server-a{ALPHA or 'file'}.log"
    logf = open(log_path, "a")
    logf.write(f"\n\n===== boot attempt {time.strftime('%H:%M:%S')} "
               f"alpha={ALPHA or 'file-default'} =====\n")
    logf.flush()
    print("+", " ".join(cmd), flush=True)
    print(f"serve: arm alpha={ALPHA or 'file-default'} gpu={GPU} "
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
                                # See modal/sitecustomize_qwen38.py.
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
