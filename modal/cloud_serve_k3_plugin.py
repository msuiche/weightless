"""Kimi-K3 + the weightless-steer vLLM plugin: GPU boot + smoke test.

The plugin (vllm-plugin/, dist `weightless-steer`) shadows
KimiLinearForCausalLM in vLLM's ModelRegistry through the
`vllm.general_plugins` entry point — no in-container model-file patching,
unlike the hotfix lane this test replaces
(patches/hotfix-kimi-k3-steering-projective.py). The steering semantics are
identical (post-layer accumulated stream, prefix_sum + hidden_states,
7168-wide, alpha 1.0), so the published GLP-92 vector transfers without
recalibration. KimiK3ForConditionalGeneration (moonshotai/Kimi-K3's arch)
builds its language model through the registry, so the shadow covers it.

One deliberate carry-over from the hotfix: the MLA DCP sentinel fix
(patch_mla in the hotfix file) is applied AT IMAGE BUILD. It is a fork-infra
bug fix (the FA3 kernel rejects cp_world_size<=0 under PP=2; experiment boots
#14/#15), not steering semantics — without it this shape does not boot. The
hotfix's model.py probe/steering patch and runner patch are NOT applied.

Stack (all validated facts, see refusal-research/experiments/20260903-kimi-k3-glp):
  - Image: vllm/vllm-openai:kimi-k3 (day-0; ships
    vllm/models/kimi_k3/nvidia/model.py, the module the adapter imports).
  - Model: moonshotai/Kimi-K3 (~1.56 TB, 96 shards, on volume 'kimi-k3';
    check_weights VERIFIES only — never re-download).
  - Shape: 2x H200:8, PP2xTP8, external_launcher lockstep driver
    (modal/k3_plugin_serve_driver.py — the fork's multi-node serving wedges;
    offline LLM() per rank with ZMQ fanout is the validated shape). Compiled
    mode (PIECEWISE cudagraphs), never eager.
  - Vector: msuiche/Kimi-K3-abliterated-cyber-GLP-92-L1-92-a1.0 (gated,
    hf-token), glp.kimi-k3-GLP-92-L1-92-a1.gguf, layers 1-92 of 93. The repo
    also ships the .pt directions the HOTFIX consumed; the plugin reads only
    the GGUF.
  - alpha=1.0 (baked at model init; one deploy = one arm).

Run order (GPU cap: one boot, stop immediately after the smoke):
    modal run cloud_serve_k3_plugin.py::ensure_vector   # CPU, seconds
    modal run cloud_serve_k3_plugin.py::check_weights   # CPU, verify-only
    modal run cloud_serve_k3_plugin.py::preflight       # CPU, entry pt + gguf
    modal deploy cloud_serve_k3_plugin.py
    modal run cloud_serve_k3_plugin.py::tail_log        # CPU, boot progress
    python3 modal/smoke_k3_plugin.py --base-url <url>
    modal app stop --yes weightless-k3-plugin-test

Boot evidence: the sitecustomize shim (modal/sitecustomize_k3.py at
/opt/weightless-shim, on PYTHONPATH for the torchrun process and every
spawned worker) registers the shadow in EVERY interpreter, pre-parses the
vector fail-closed, and prints WEIGHTLESS-SHIM markers straight to stderr —
vllm's dictConfig attaches a handler only to the `vllm` logger, so the
plugin's own INFO lines are not a reliable channel on day-0 builds (glm5next
lane, 2026-09-18). The driver additionally asserts, post-build, that the
registry resolved KimiLinearForCausalLM to the steered adapter, and refuses
to serve otherwise.

Cost discipline: this lane is 16 H200s. The driver logs to the volume
(append mode; a crashed container's dying words survive), the SIGTERM
handler commits it, and `modal app stop` is part of the runbook, not an
afterthought.
"""
import os
from pathlib import Path

import modal
import modal.experimental

ROOT = Path(__file__).resolve().parents[1]           # weightless/
MODEL_ID = "moonshotai/Kimi-K3"
SERVED_MODEL = "kimi-k3"
VOLUME_NAME = "kimi-k3"
VECTOR_REPO = "msuiche/Kimi-K3-abliterated-cyber-GLP-92-L1-92-a1.0"
VECTOR_FILE = "glp.kimi-k3-GLP-92-L1-92-a1.gguf"
VECTOR_PATH = "/data/vector/" + VECTOR_FILE
IMAGE = "vllm/vllm-openai:kimi-k3"
HOTFIX_NAME = "hotfix-kimi-k3-steering-projective.py"
DRIVER_NAME = "k3_plugin_serve_driver.py"
ENV = {"HF_HOME": "/data/hf", "HF_XET_HIGH_PERFORMANCE": "1"}
PY = "/usr/bin/python3.12"
ALPHA = os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0")

app = modal.App("weightless-k3-plugin-test")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

download_image = (modal.Image.debian_slim(python_version="3.12")
                  .pip_install("huggingface_hub", "hf_xet"))

# The plugin installs from a copied source tree (whatever is on disk at
# deploy/build time is what boots). The pyproject packages.find looks in
# [".", ".."] for weightless_steer* AND weightless_runtime*, so the runtime
# package sits NEXT TO the plugin dir in the image. Install with the image's
# own python — the one the driver runs under — NOT modal's add_python shim,
# or the entry point lands in the wrong site-packages.
image = (modal.Image.from_registry(IMAGE, add_python="3.12",
                                   setup_dockerfile_commands=[
                                       "ENTRYPOINT []"])
         .add_local_dir(ROOT / "vllm-plugin", "/opt/weightless-src/vllm-plugin",
                        copy=True)
         .add_local_dir(ROOT / "weightless_runtime",
                        "/opt/weightless-src/weightless_runtime", copy=True)
         .add_local_file(ROOT / "patches" / HOTFIX_NAME,
                         "/work/" + HOTFIX_NAME, copy=True)
         .add_local_file(ROOT / "modal" / DRIVER_NAME,
                         "/work/" + DRIVER_NAME, copy=True)
         .add_local_file(ROOT / "modal" / "vllm_logging_config.json",
                         "/work/vllm_logging_config.json", copy=True)
         .add_local_file(ROOT / "modal" / "sitecustomize_k3.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         .run_commands(
             f"{PY} -m pip --version || {PY} -m ensurepip --upgrade",
             "cd /opt/weightless-src/vllm-plugin && "
             f"{PY} -m pip install --no-deps .",
             # The MLA DCP sentinel fix, and ONLY that fix, from the hotfix
             # (fail-closed on anchor drift; idempotent marker). The
             # steering/probe patches are the plugin's job now.
             f"cd /work && {PY} -c \""
             "import importlib.util; "
             "s = importlib.util.spec_from_file_location('k3hotfix', "
             f"'/work/{HOTFIX_NAME}'); "
             "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
             "m.patch_mla(m.MLA_FILE)\""))


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              secrets=[hf_secret], env=ENV)
def ensure_vector():
    """Fetch the gated GLP-92 GGUF and stage it at the stable VECTOR_PATH.

    The HF-cache snapshot path embeds a revision hash; the server env needs a
    path known at deploy time, so the file is copied out of the cache (same
    pattern as cloud_serve_k3.ensure_dirs). Fail-closed: <2 MB means a
    truncated or LFS-pointer download (real file is ~2.6 MB)."""
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


@app.function(image=download_image, volumes={"/data": vol}, timeout=600,
              env=ENV)
def check_weights():
    """VERIFY the ~1.56 TB snapshot is on the volume. Never downloads.

    Fail-closed size-gate (same discipline as cloud_serve_k3.ensure_weights):
    96 shards + index, zero .incomplete blobs, >1.5 TB before any GPU spend."""
    snap_glob = sorted((Path(os.environ["HF_HOME"]) / "hub"
                        / "models--moonshotai--Kimi-K3" / "snapshots").glob("*"))
    assert len(snap_glob) == 1, f"expected exactly one snapshot, got {snap_glob}"
    snap = snap_glob[0]
    shards = sorted(snap.glob("*.safetensors"))
    hub = Path(os.environ["HF_HOME"]) / "hub" / "models--moonshotai--Kimi-K3"
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    print(f"shards: {len(shards)}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e12:.4f} TB", flush=True)
    assert len(shards) == 96, f"expected 96 shards, found {len(shards)}"
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 1.5e12, f"size-gate failed: {total / 1e12:.3f} TB"
    print("weights present:", snap, flush=True)


_PREFLIGHT_REGISTER_ONLY = r"""
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
want = ("KimiLinearForCausalLM",
        "weightless_steer.archs.kimi_k3:SteeredKimiLinearForCausalLM")
assert want in calls, f"shadow registration missing; register_model calls: {calls}"
print("SHADOW OK: register_model called with", want)
# NOTE: the adapter IMPORT is deliberately not done here — this probe runs on
# CPU, and the adapter's upstream import chain needs libcuda (see
# preflight_gpu, which runs this file's resolve WITH the import).
"""

_PREFLIGHT_RESOLVE = _PREFLIGHT_REGISTER_ONLY + r"""
# The real compatibility gate: import the adapter against the image's vLLM.
import weightless_steer.archs.kimi_k3 as k
print("ADAPTER IMPORT OK:", k.SteeredKimiLinearForCausalLM.__mro__[1].__module__)
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
                             num_layers=93, hidden_size=7168)
assert core is not None and len(core.dirs) == 92, core
assert min(core.dirs) == 1 and max(core.dirs) == 92, sorted(core.dirs)
assert core.alpha == 1.0, core.alpha
print("CORE OK: real GGUF parses, 92 dirs on layers 1..92, alpha 1.0")
"""


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version and that the adapter's upstream module file
       exists (the MODULE IMPORT itself needs a GPU container: this image's
       kimi_k3 import chain touches torch.ops._C.rotary_embedding at module
       scope, which requires libcuda — matcher_utils.py:32 — so the
       import-drag lives in preflight_gpu below);
    2. the MLA DCP fix is in place and the model file is UNPATCHED (the
       plugin lane runs stock model.py);
    3. pip metadata: the `vllm.general_plugins` entry point is registered;
    4. vLLM's real plugin loader runs register() and the registry shadows
       KimiLinearForCausalLM with the STEERED lazy string when
       WEIGHTLESS_STEER_PATH is set, and stays stock with it unset;
    5. the real GLP-92 GGUF parses through SteeringCore.from_env at the
       model's geometry (93 layers, 7168-wide stream).
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
         "import pathlib; "
         "p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/"
         "models/kimi_k3/nvidia/model.py'); "
         "assert p.is_file(), p; print('kimi_k3 module file OK:', p)"],
        capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="", flush=True)
    assert r.returncode == 0, "vllm import / kimi_k3 module file check failed"

    sh("src = open('/usr/local/lib/python3.12/dist-packages/vllm/model_executor"
       "/layers/attention/mla_attention.py').read()\n"
       "assert 'dspark k3 probe' in src, 'MLA DCP fix missing'\n"
       "print('MLA DCP FIX OK')\n"
       "model = open('/usr/local/lib/python3.12/dist-packages/vllm/models/"
       "kimi_k3/nvidia/model.py').read()\n"
       "assert 'dspark k3 probe' not in model, 'model.py is hotfix-patched; "
       "the plugin lane must run stock'\n"
       "print('MODEL.PY STOCK OK')")

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
    sh(_PREFLIGHT_REGISTER_ONLY, "/nonexistent-sentinel.gguf")
    sh(_PREFLIGHT_CORE)
    print("PREFLIGHT PASSED", flush=True)


@app.function(image=image, volumes={"/data": vol}, gpu="T4:1", timeout=1800,
              env=ENV)
def preflight_gpu():
    """The import-drag gate, on the cheapest GPU Modal has.

    The adapter imports vllm.models.kimi_k3.nvidia.model, whose chain touches
    torch.ops._C.rotary_embedding at module scope — that op registers only
    when vllm._C loads, and vllm._C needs libcuda. One T4 for ~2 minutes
    instead of learning this at 16x H200 boot time. No model weights are
    touched: imports and the registry resolve only."""
    import subprocess

    def sh(code, *args):
        r = subprocess.run([PY, "-c", code, *args],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr, sep="", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"preflight_gpu step failed "
                               f"(rc={r.returncode})")

    sh("import vllm.models.kimi_k3.nvidia.model as m\n"
       "print('KIMI_K3 MODULE IMPORT OK:', m.__file__)\n"
       "print('classes:', m.KimiLinearModel.__name__, "
       "m.KimiLinearForCausalLM.__name__)")
    sh(_PREFLIGHT_RESOLVE, "/nonexistent-sentinel.gguf")
    print("PREFLIGHT GPU PASSED", flush=True)


@app.function(image=image, volumes={"/data": vol}, timeout=600, env=ENV)
def tail_log(n: int = 200):
    """Print the serve log tail (markers, errors) from the volume.

    CPU-only: safe to run while the GPU app boots and after it stops."""
    import glob

    logs = sorted(glob.glob("/data/out-kimi-k3-plugin-test/serve-*.log"),
                  key=os.path.getmtime)
    if not logs:
        print("no serve log yet", flush=True)
        return
    path = logs[-1]
    txt = open(path, errors="replace").read()
    print(f"--- {path} ({len(txt)} B), last {n} lines ---", flush=True)
    print("\n".join(txt.splitlines()[-n:]), flush=True)
    hits = [l for l in txt.splitlines()
            if "WEIGHTLESS-SHIM" in l or "weightless GLP steering active" in l
            or "registry KimiLinearForCausalLM" in l]
    print(f"--- steering evidence lines ({len(hits)}) ---", flush=True)
    for l in hits:
        print(l, flush=True)


@app.server(image=image, volumes={"/data": vol}, gpu="H200:8",
            min_containers=0, max_containers=2, scaledown_window=300,
            startup_timeout=45 * 60, port=8000, unauthenticated=True,
            env=dict(ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                     # compile caches on the volume: re-boots reuse them
                     TORCHINDUCTOR_CACHE_DIR="/data/cache/inductor",
                     VLLM_CACHE_ROOT="/data/cache/vllm",
                     TRITON_CACHE_DIR="/data/cache/triton",
                     # forked workers inherit EngineCore store fds; spawn
                     # sidesteps that wedge class (experiment boot #4)
                     VLLM_WORKER_MULTIPROC_METHOD="spawn",
                     # pynccl's torch symm-mem rendezvous cannot cross
                     # nodes on this TCP-only fabric (boots #2-#5)
                     VLLM_ALLREDUCE_USE_SYMM_MEM="0",
                     WEIGHTLESS_STEER_PATH=VECTOR_PATH,
                     WEIGHTLESS_STEER_ALPHA=ALPHA,
                     # Third-party INFO (the plugin's steering-active line)
                     # may still be eaten by this build's dictConfig — the
                     # shim's stderr markers are the evidence channel; this
                     # config asks nicely anyway (harmless if ignored).
                     VLLM_CONFIGURE_LOGGING="1",
                     VLLM_LOGGING_CONFIG_PATH="/work/vllm_logging_config.json"))
@modal.experimental.clustered(size=2, rdma=False)
class K3PluginServer:
    """Cold starts load ~1.56 TB; allow up to 45 minutes for readiness.

    Clustered Functions cannot be web functions, so this is a Modal Server
    (rank 0's :8000 is the public endpoint; followers proxy their :8000 to
    the leader — Server health checks hit every container). The fork's MP
    leader/headless choreography wedges cross-node (experiment boots #2-#9),
    so serving is the collective lockstep driver: every rank builds LLM()
    and generates together; rank 0 owns the HTTP endpoint.
    """

    @modal.enter()
    def start(self):
        import subprocess
        import time

        ci = modal.experimental.get_cluster_info()
        rank = ci.rank
        master = ci.container_ips[0]
        print(f"serve: rank={rank} master={master} "
              f"ips={ci.container_ips}", flush=True)

        log_dir = "/data/out-kimi-k3-plugin-test"
        os.makedirs(log_dir, exist_ok=True)
        log_path = f"{log_dir}/serve-a{ALPHA}.log"
        logf = open(log_path, "a")
        logf.write(f"\n\n===== boot attempt {time.strftime('%H:%M:%S')} "
                   f"rank={rank} alpha={ALPHA} (plugin lane) =====\n")
        logf.flush()

        # external_launcher under torchrun: ALL 16 processes run the same
        # driver symmetrically; rank 0 also serves HTTP on :8000.
        # PYTHONPATH prepends the sitecustomize shim: every python of the
        # serve stack (driver, EngineCore, spawned workers) registers the
        # shadow and prints stderr markers — the logging-config-proof boot
        # evidence channel (see modal/sitecustomize_k3.py).
        cmd = [
            PY, "-m", "torch.distributed.run",
            "--nnodes=2", f"--node-rank={rank}", "--nproc-per-node=8",
            f"--master-addr={master}", "--master-port=29511",
            "/work/" + DRIVER_NAME,
        ]
        print("+", " ".join(cmd), f"-> {log_path}", flush=True)
        self._logf = logf
        self._proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1",
                     PYTHONPATH="/opt/weightless-shim:"
                     + os.environ.get("PYTHONPATH", "")))

        def _sigterm(signum, frame):
            vol.commit()
            self._proc.terminate()
            raise SystemExit(0)

        import signal
        signal.signal(signal.SIGTERM, _sigterm)

    @modal.exit()
    def stop(self):
        self._proc.terminate()
        self._logf.close()
        vol.commit()


@app.local_entrypoint()
def main():
    print(__doc__)
