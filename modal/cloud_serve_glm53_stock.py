"""GLM-5.3-Flash + the weightless-steer plugin on STOCK upstream vLLM 0.31.

The glm5next arch landed upstream via PR #53906 (v0.30+), so this lane boots
the stock `vllm/vllm-openai:v0.31.0` image — NOT the day-0 fork image
(`cloud_serve_glm53.py` covers that one; the fork forces breakable
cudagraphs that crash deterministically at first inference with a modified
checkpoint, see TROUBLESHOOTING.md). The plugin is pip-installed from the
local tree and shadows Glm5NextForCausalLM through the
`vllm.general_plugins` entry point exactly as on the fork lane; the
checkpoint's multimodal wrapper (Glm5NextForConditionalGeneration) builds
its language model through the registry, so the one shadow covers both.

Stack:
  - Image: vllm/vllm-openai:v0.31.0 (stock, cu129). Adapter imports
    vllm.models.glm5next.common.model (the merged-upstream layout).
  - Model: RedHatAI/GLM-5.3-Flash-NVFP4 (compressed-tensors W4A4 NVFP4,
    ~198 GB on the glm53-flash volume, staged by the fork lane's
    ensure_weights; arch = Glm5NextForConditionalGeneration).
  - Shape: H100:4 single container, TP4 (~50 GiB weights/rank), marlin MoE
    backend, bf16 KV (stock 0.31's sparse-MLA sm90 fa3 path rejects the
    uint8 indexer kpool under fp8 KV — fork-lane fp8 KV does not port).
  - Vector: msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44 (gated,
    hf-token secret), GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf.
  - LoRA arm: --enable-lora with a generated random-init rank-8 LoRA on
    every decoder layer's self_attn.o_proj (KDA in=8192, MLA in=16384,
    out=4096, std=0.01) — stock 0.31's multimodal wrapper inherits
    SupportsLoRA from Glm4vForConditionalGeneration, so no plugin changes
    are needed for LoRA; this arm proves the mechanics before a real RL
    artifact exists.

Arms (one deploy/run each; alpha is a model-init buffer):
    modal run cloud_serve_glm53_stock.py::preflight    # CPU gates, incl.
                                                       # the plugin test
                                                       # suite in-image
    modal run cloud_serve_glm53_stock.py::ensure_lora  # CPU, ~21 MB
    GLM53_ARM=a2   modal run cloud_serve_glm53_stock.py::eval_arm  # dose
    GLM53_ARM=lora modal run cloud_serve_glm53_stock.py::eval_arm  # LoRA
    GLM53_ARM=stock modal deploy cloud_serve_glm53_stock.py        # endpoint

NOTE 2026-10-06: the web_server endpoint mode (serve()) is degraded on
current Modal — the container is recycled minutes after serve() returns at
the boot marker, before/while eval traffic flows (observed: five boot loops
in a row, SIGTERM right after "Application startup complete" + one 200).
eval_arm sidesteps the routing layer entirely: vllm runs as a subprocess of
the GPU function and requests go over localhost, so one `modal run` = one
boot + one eval, no endpoint. serve() is kept for endpoint parity with the
other lanes but do not run long evals through it until this is understood.

Boot evidence: same sitecustomize shim discipline as the fork lane
(WEIGHTLESS-SHIM stderr markers in every serve-stack process); eval_arm
fails closed on the steering-active line missing before it serves a single
request.

Cost discipline: max_containers=1, scaledown_window=180, stop the app
between arms. Expected spend: 4 boots x ~25 min x 4 H100.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]           # weightless/
SPARK = ROOT.parent                                  # spark workspace
MODEL_ID = "RedHatAI/GLM-5.3-Flash-NVFP4"
SERVED_MODEL = "glm53-flash"
VOLUME_NAME = "glm53-flash"
VECTOR_PATH = ("/data/vector/"
               "GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf")
IMAGE = "vllm/vllm-openai:v0.31.0"
CHAT_TEMPLATE = (SPARK / "refusal-research" / "experiments"
                 / "20260830-glp44-dflash2-acceptance" / "chat_template_mm.jinja")
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}
LORA_DIR = "/data/lora/smoke-o_proj-r8"
LORA_NAME = "smoke-lora"

# Arm/shape switches, read at deploy time (same convention as
# cloud_serve_glm53.py: the serve cmd is built inside the container, so the
# values are ALSO passed through the function env below).
ARM = os.environ.get("GLM53_ARM", "stock")   # stock | a0 | a2 | lora
GPU = os.environ.get("GLM53_GPU", "H100:4")
TP = os.environ.get("GLM53_TP", "4")
# Stock 0.31, fp8 KV REJECTED: the sparse-MLA (indexer) sm90 path stores its
# kpool as uint8 under fp8 KV and flashinfer's fa3 backend refuses uint8 at
# cudagraph capture (flashinfer_mla_sparse_sm90.py, "MLA kv_data_type
# torch.uint8 is not supported by the fa3 backend"). The day-0 fork shipped
# a patched sparse path that tolerated it. bf16 KV costs ~3 GB here (11 MLA
# layers share a 576-float compressed cache), so the flag is simply dropped.
KV_DTYPE = os.environ.get("GLM53_KV_DTYPE", "")
ENFORCE_EAGER = os.environ.get("GLM53_ENFORCE_EAGER", "")
GMU = os.environ.get("GLM53_GMU", "0.92")
ALPHA = {"a0": "0.0", "a2": "2.0", "lora": "0.0"}.get(ARM)
assert ARM in ("stock", "a0", "a2", "lora"), ARM

app = modal.App("weightless-glm53-stock031")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

download_image = (modal.Image.debian_slim(python_version="3.12")
                  .pip_install("huggingface_hub", "hf_transfer",
                               "numpy", "safetensors"))

# The stock image's vllm is importable by exactly one interpreter; discover
# it at build time (vllm-openai images have shipped both /usr/bin/python3.12
# dist-packages and /usr/local/bin layouts) and pin it in /opt/vllm-py.env
# for every later layer and the serve function.
_PY_DISCOVER = (
    "set -e; for c in /usr/bin/python3.12 /usr/local/bin/python3.12 "
    "/usr/bin/python3 /usr/local/bin/python3; do "
    "if $c -c 'import vllm' 2>/dev/null; then "
    "echo \"VLLM_PY=$c\" > /opt/vllm-py.env; break; fi; done; "
    "test -f /opt/vllm-py.env && cat /opt/vllm-py.env")

image = (modal.Image.from_registry(IMAGE, add_python="3.12",
                                   setup_dockerfile_commands=["ENTRYPOINT []"])
         .add_local_dir(ROOT / "vllm-plugin", "/opt/weightless-src/vllm-plugin",
                        copy=True)
         .add_local_dir(ROOT / "weightless_runtime",
                        "/opt/weightless-src/weightless_runtime", copy=True)
         # The structure tests pin the adapter's copied forwards against the
         # vendored references; mount them so the pin runs in-image too.
         .add_local_file(ROOT / "patches" / "reference" / "glm5next.py",
                         "/opt/weightless-src/patches/reference/glm5next.py",
                         copy=True)
         .add_local_file(ROOT / "patches" / "reference"
                         / "glm5next_upstream_v0310.py",
                         "/opt/weightless-src/patches/reference"
                         "/glm5next_upstream_v0310.py",
                         copy=True)
         .add_local_file(CHAT_TEMPLATE, "/work/chat_template_mm.jinja",
                         copy=True)
         .add_local_file(ROOT / "modal" / "vllm_logging_config.json",
                         "/work/vllm_logging_config.json", copy=True)
         .add_local_file(ROOT / "modal" / "sitecustomize.py",
                         "/opt/weightless-shim/sitecustomize.py", copy=True)
         # Eval suites for the in-container eval_arm (read-only data).
         .add_local_file(SPARK / "refusal-research" / "suites" / "contrasts"
                         / "refusal32-suite.json",
                         "/work/suites/refusal32-suite.json", copy=True)
         .add_local_file(SPARK / "refusal-research" / "suites" / "core"
                         / "cyber32-suite.json",
                         "/work/suites/cyber32-suite.json", copy=True)
         .add_local_file(SPARK / "refusal-research" / "suites" / "core"
                         / "benign32-suite.json",
                         "/work/suites/benign32-suite.json", copy=True)
         .run_commands(
             _PY_DISCOVER,
             # rm -rf: a stale setuptools build/ tree next to the sources
             # (produced by any local `pip install .`) makes build_py reuse
             # ancient files — the wheel then installs a month-old
             # weightless_steer with no error. Never trust it in-image.
             ". /opt/vllm-py.env && cd /opt/weightless-src/vllm-plugin && "
             "rm -rf build weightless_steer.egg-info && "
             "$VLLM_PY -m pip install --no-deps ."))


@app.function(image=download_image, volumes={"/data": vol}, timeout=900,
              env=ENV)
def ensure_lora():
    """Generate the smoke LoRA: random-init rank-8 on every o_proj.

    Nothing is trained — the point is the serving mechanics (load, activate,
    deactivate). A and B are both nonzero (peft's B=0 init would be a
    numeric no-op); std=0.01 makes outputs diverge without garbling.
    Deterministic seed so the artifact is reproducible. Layer shapes from
    the checkpoint's own config.json: KDA o_proj in = linear_num_heads *
    linear_head_dim (64*128=8192), MLA o_proj in = num_attention_heads *
    v_head_dim (64*256=16384), out = hidden_size (4096). Names follow the
    checkpoint's HF convention (model.language_model.layers.N....) under
    the peft base_model.model. prefix; vLLM's weights mapper
    (model.language_model. -> language_model.model.) resolves them to the
    served module tree.
    """
    import json
    import glob

    import numpy as np
    from safetensors.numpy import save_file

    snap = sorted(glob.glob(
        "/data/hf/hub/models--RedHatAI--GLM-5.3-Flash-NVFP4/snapshots/*"))
    assert len(snap) == 1, snap
    cfg = json.load(open(Path(snap[0]) / "config.json"))["text_config"]
    hidden = cfg["hidden_size"]
    kda_in = cfg["linear_num_heads"] * cfg["linear_head_dim"]
    mla_in = cfg["num_attention_heads"] * cfg["v_head_dim"]
    r = 8
    rng = np.random.default_rng(20261006)
    tensors = {}
    for i, kind in enumerate(cfg["layer_types"]):
        in_dim = kda_in if kind == "linear_attention" else mla_in
        base = (f"base_model.model.model.language_model.layers.{i}"
                f".self_attn.o_proj")
        tensors[f"{base}.lora_A.weight"] = (
            rng.standard_normal((r, in_dim), dtype=np.float32) * 0.01)
        tensors[f"{base}.lora_B.weight"] = (
            rng.standard_normal((hidden, r), dtype=np.float32) * 0.01)
    dst = Path(LORA_DIR)
    dst.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dst / "adapter_model.safetensors"))
    (dst / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": MODEL_ID,
        "r": r,
        "lora_alpha": r,
        "target_modules": ["o_proj"],
        "bias": "none",
        "modules_to_save": None,
    }, indent=1))
    vol.commit()
    total = sum(f.stat().st_size for f in dst.iterdir())
    print(f"lora ready: {dst} ({len(tensors)} tensors, "
          f"{total / 1e6:.1f} MB)", flush=True)


_PREFLIGHT_RESOLVE = r"""
import os, sys
os.environ["WEIGHTLESS_STEER_PATH"] = sys.argv[1]   # sentinel or real gguf
from vllm.model_executor.models import ModelRegistry
calls = []
orig = ModelRegistry.register_model
def spy(arch, target, *a, **k):
    calls.append((arch, target))
    return orig(arch, target, *a, **k)
ModelRegistry.register_model = spy
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

# The smoke-LoRA names must resolve through the wrapper's weights mapper to
# real module paths. Pure CPU string work; catches a naming drift before
# the GPU boots.
_PREFLIGHT_LORA_NAMES = r"""
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.models.glm5next.common.model import (
    Glm5NextForConditionalGeneration as G)
mapper = G.hf_to_vllm_mapper
print("mapper:", mapper)
name = ("base_model.model.model.language_model.layers.3"
        ".self_attn.o_proj.lora_A.weight")
mod, is_a = parse_fine_tuned_lora_name(name, mapper)
print("parsed:", mod, "is_lora_a:", is_a)
assert is_a
# 0.31 returns the bare mapped module path (no base_model.model. re-prefix).
want = "language_model.model.layers.3.self_attn.o_proj"
assert mod == want, f"{mod} != {want}"
# stock 0.31: the wrapper inherits SupportsLoRA from Glm4v; the text-only
# Glm5NextForCausalLM does NOT declare it (LoRA on text-only checkpoints
# would need a declaration we deliberately do not ship in the plugin).
from vllm.model_executor.models.interfaces import supports_lora
from vllm.models.glm5next.common.model import Glm5NextForCausalLM
print("wrapper supports_lora:", supports_lora(G),
      "| text CausalLM supports_lora:", supports_lora(Glm5NextForCausalLM))
assert supports_lora(G)
print("LORA NAMES OK")
"""


@app.function(image=image, volumes={"/data": vol}, timeout=900, env=ENV)
def debug_register():
    """CPU: what register() actually does inside the image, verbosely."""
    import subprocess

    code = r"""
import os, traceback
os.environ["WEIGHTLESS_STEER_PATH"] = "/nonexistent-sentinel.gguf"
import weightless_steer.plugin as p
print("plugin file:", p.__file__)
print("SHADOWED_ARCHS:", len(p.SHADOWED_ARCHS))
import weightless_steer
print("pkg:", weightless_steer.__file__,
      getattr(weightless_steer, "__version__", "?"))
from importlib.metadata import version, files
print("dist version:", version("weightless-steer"))
from vllm.model_executor.models import ModelRegistry
print("ModelRegistry type:", type(ModelRegistry))
calls = []
orig = ModelRegistry.register_model
def spy(arch, target, *a, **k):
    calls.append((arch, target))
    return orig(arch, target, *a, **k)
ModelRegistry.register_model = spy
try:
    p.register()
except Exception:
    traceback.print_exc()
print("calls:", len(calls), calls[:3])
from vllm.model_executor.models import ModelRegistry as R2
print("registry entry:", R2.models.get("Glm5NextForCausalLM"))
"""
    vllm_py = open("/opt/vllm-py.env").read().strip().split("=", 1)[1]
    r = subprocess.run([vllm_py, "-c", code], capture_output=True, text=True)
    print(r.stdout, r.stderr[-3000:], sep="", flush=True)


@app.function(image=image, volumes={"/data": vol}, timeout=1800, env=ENV)
def preflight():
    """CPU-only gates, BEFORE any GPU spend:

    1. the image's vLLM version is stock 0.31.x and glm5next resolves to
       the merged-upstream common.model module;
    2. pip metadata: the `vllm.general_plugins` entry point is registered;
    3. vLLM's real plugin loader runs register() and the registry shadows
       Glm5NextForCausalLM when WEIGHTLESS_STEER_PATH is set (import-drags
       the adapter against the image's vLLM);
    4. with the env unset the registry stays stock;
    5. the real GLP-44 GGUF parses through SteeringCore.from_env;
    6. the plugin's glm5next test suites run IN the image — the offline
       functional/structure tests plus the real-vLLM guard
       (TestRealVllmGlm5Next), so adapter/upstream drift fails here, in a
       CPU container, not at a GPU boot;
    7. the smoke-LoRA naming convention resolves through the wrapper's
       weights mapper, and the wrapper (not the text CausalLM) carries
       SupportsLoRA on this stock build.
    """
    import subprocess

    VLLM_PY = open("/opt/vllm-py.env").read().strip().split("=", 1)[1]

    def sh(code, *args):
        r = subprocess.run([VLLM_PY, "-c", code, *args],
                           capture_output=True, text=True)
        print(r.stdout, r.stderr, sep="", flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"preflight step failed (rc={r.returncode})")

    sh("import vllm; print('vllm', vllm.__version__); "
       "assert vllm.__version__.startswith('0.31'), vllm.__version__; "
       "import vllm.models.glm5next.common.model as m; "
       "print('glm5next at', m.__file__); "
       "import vllm.model_executor.layers.mhc as mhc; "
       "assert hasattr(mhc, 'hc_contract') and hasattr(mhc, 'hc_expand'); "
       "print('hc_contract/hc_expand OK')")

    sh("from importlib.metadata import entry_points\n"
       "eps = {e.name: e.value for e in entry_points("
       "group='vllm.general_plugins')}\n"
       "print('general_plugins:', eps)\n"
       "assert eps.get('weightless_steer') == "
       "'weightless_steer.plugin:register', eps\n"
       "print('ENTRY POINT OK')\n"
       # Stale-wheel tripwire: a cached setuptools build/ tree once made the
       # image install a one-arch weightless_steer (2026-10-06). The entry
       # point string is identical across vintages, so check CONTENT.
       "from weightless_steer.plugin import SHADOWED_ARCHS\n"
       "assert SHADOWED_ARCHS.get('Glm5NextForCausalLM') == "
       "'weightless_steer.archs.glm5next:SteeredGlm5NextForCausalLM', "
       "SHADOWED_ARCHS\n"
       "assert len(SHADOWED_ARCHS) >= 15, len(SHADOWED_ARCHS)\n"
       "print('PLUGIN CONTENT OK:', len(SHADOWED_ARCHS), 'archs')\n")

    sh(_PREFLIGHT_STOCK)
    sh(_PREFLIGHT_RESOLVE, "/nonexistent-sentinel.gguf")
    sh(_PREFLIGHT_CORE)
    sh(_PREFLIGHT_LORA_NAMES)

    r = subprocess.run(
        [VLLM_PY, "-m", "unittest", "-v",
         "tests.test_archs.test_glm5next",
         "tests.test_archs.test_real_vllm.TestRealVllmGlm5Next"],
        capture_output=True, text=True, cwd="/opt/weightless-src/vllm-plugin")
    print(r.stdout, r.stderr[-4000:], sep="", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"in-image test suite failed (rc={r.returncode})")
    print("PREFLIGHT PASSED", flush=True)


def _serve_cmd(model_path: str) -> list:
    import json

    vllm_py = open("/opt/vllm-py.env").read().strip().split("=", 1)[1]
    # The container re-reads the arm switches from its own env (the deploy
    # shell's values reach this function body only through the function env
    # dict below — learned the hard way on the fork lane).
    kv_dtype = os.environ.get("GLM53_KV_DTYPE", "")
    gmu = os.environ.get("GLM53_GMU", "0.92")
    cmd = [
        vllm_py, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", SERVED_MODEL,
        "--tensor-parallel-size", os.environ.get("GLM53_TP", "4"),
        "--max-model-len", "32768",
        "--moe-backend", "marlin",
        "--gpu-memory-utilization", gmu,
        "--max-num-seqs", "8",
        "--chat-template", "/work/chat_template_mm.jinja",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
        "--port", "8000",
    ]
    if kv_dtype:
        cmd += ["--kv-cache-dtype", kv_dtype]
    if os.environ.get("GLM53_ENFORCE_EAGER"):
        cmd += ["--enforce-eager"]
    if os.environ.get("GLM53_ARM") == "lora":
        cmd += [
            "--enable-lora", "--max-loras", "1",
            "--lora-modules", json.dumps(
                {"name": LORA_NAME, "path": LORA_DIR,
                 "base_model_name": SERVED_MODEL}),
        ]
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


_serve_env = dict(ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TORCHINDUCTOR_CACHE_DIR="/data/cache/inductor",
                  VLLM_CACHE_ROOT="/data/cache/vllm",
                  TRITON_CACHE_DIR="/data/cache/triton",
                  VLLM_CONFIGURE_LOGGING="1",
                  VLLM_LOGGING_CONFIG_PATH="/work/vllm_logging_config.json",
                  GLM53_ARM=ARM, GLM53_KV_DTYPE=KV_DTYPE, GLM53_GMU=GMU,
                  GLM53_ENFORCE_EAGER=ENFORCE_EAGER, GLM53_TP=TP)
if ALPHA is not None:
    _serve_env["WEIGHTLESS_STEER_PATH"] = VECTOR_PATH
    _serve_env["WEIGHTLESS_STEER_ALPHA"] = ALPHA


SUITES = {
    "cyber32": "/work/suites/cyber32-suite.json",
    "refusal32": "/work/suites/refusal32-suite.json",
    "benign32": "/work/suites/benign32-suite.json",
}


@app.function(image=image, volumes={"/data": vol}, gpu=GPU,
              timeout=4 * 3600, env=_serve_env)
def eval_arm():
    """One boot + one eval, entirely in-container: no endpoint lifecycle.

    serve() (web_server deploy) is degraded on current Modal — the container
    is recycled minutes after the function returns at the boot marker, so an
    external eval driver loses the server mid-run (2026-10-06: five boot
    loops, SIGTERM right after "Application startup complete"). Here vllm is
    a subprocess of the GPU function and requests go over localhost; the
    function stays alive for the whole eval. Steered arms fail closed: the
    steering-active line must appear before the first request is sent.

    Arm contents (results land in /data/out-glm53-stock031/ and print to
    stdout; scoring happens locally with the repo harness):
      a2   — cyber32 + refusal32 + benign32 (the dose run)
      lora — the 12-prompt probe set as base AND as smoke-lora (divergence),
             plus cyber32 + refusal32 as the base model: with LoRA modules
             loaded but inactive the base path must stay numerically
             identical to the a0 arm (cross-checked locally against
             probe-a0.json), so this doubles as the a0 suite read without a
             third boot.
      a0   — cyber32 + refusal32 (standalone behavioral-parity read)
      stock — the 12-prompt probe set + cyber32 + refusal32
    """
    import concurrent.futures as cf
    import json
    import signal
    import subprocess
    import time
    import urllib.request

    arm = os.environ.get("GLM53_ARM", "stock")

    def load_suite(path):
        d = json.load(open(path))
        rows = d if isinstance(d, list) else (d.get("results")
                                              or d.get("items"))
        return [r["prompt"] if isinstance(r, dict) else r for r in rows]

    cyber = load_suite(SUITES["cyber32"])
    refusal = load_suite(SUITES["refusal32"])
    benign = load_suite(SUITES["benign32"])
    probes = cyber[:8] + benign[:4]          # == probe_glm53_stock.PROMPTS

    if arm == "a2":
        plan = [("cyber32", cyber, SERVED_MODEL, "cyber32-a2.0.json"),
                ("refusal32", refusal, SERVED_MODEL, "refusal32-a2.0.json"),
                ("benign32", benign, SERVED_MODEL, "benign32-a2.0.json")]
    elif arm == "lora":
        plan = [("probe", probes, SERVED_MODEL, "probe-lora-base.json"),
                ("probe", probes, LORA_NAME, "probe-lora-smoke.json"),
                ("cyber32", cyber, SERVED_MODEL, "cyber32-a0.0.json"),
                ("refusal32", refusal, SERVED_MODEL, "refusal32-a0.0.json")]
    elif arm == "a0":
        plan = [("cyber32", cyber, SERVED_MODEL, "cyber32-a0.0.json"),
                ("refusal32", refusal, SERVED_MODEL, "refusal32-a0.0.json")]
    else:
        plan = [("probe", probes, SERVED_MODEL, "probe-stock-inc.json"),
                ("cyber32", cyber, SERVED_MODEL, "cyber32-stock.json"),
                ("refusal32", refusal, SERVED_MODEL, "refusal32-stock.json")]

    model_path = _model_snapshot_path()
    cmd = _serve_cmd(model_path)
    log_dir = "/data/out-glm53-stock031"
    os.makedirs(log_dir, exist_ok=True)
    stamp = time.strftime("%m%d-%H%M%S")
    log_path = f"{log_dir}/eval-{arm}-{stamp}.log"
    logf = open(log_path, "a")
    logf.write(f"===== eval_arm {arm} {stamp} =====\n")
    logf.flush()
    print("+", " ".join(cmd), flush=True)
    print(f"eval_arm: arm={arm} gpu={GPU} -> {log_path}", flush=True)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONUNBUFFERED="1",
                                     PYTHONPATH="/opt/weightless-shim:"
                                     + os.environ.get("PYTHONPATH", "")))

    def _cleanup():
        proc.terminate()
        try:
            proc.wait(120)
        except Exception:
            proc.kill()
        logf.close()
        vol.commit()

    def _sigterm(signum, frame):
        _cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    def post_chat(model, prompt, max_tokens, timeout=900):
        body = {"model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "max_tokens": max_tokens}
        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.load(r)
        ch = resp["choices"][0]
        usage = resp.get("usage") or {}
        return {"completion": ch["message"].get("content") or "",
                "finish_reason": ch.get("finish_reason"),
                "completion_tokens": usage.get("completion_tokens")}

    # Readiness: port answers AND (steered arms) the steering line is in the
    # log. The shim marker lands at worker spawn; the plugin's own
    # "weightless GLP steering active" lands at model init — either proves
    # the steered class booted; NEITHER by the deadline = fail closed.
    steered = arm in ("a0", "a2", "lora")
    t0 = time.time()
    printed_marker = False
    models = []
    while True:
        if proc.poll() is not None:
            _cleanup()
            raise RuntimeError(f"vllm died rc={proc.returncode}; "
                               f"log {log_path}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models",
                                        timeout=10) as r:
                models = [m["id"] for m in json.load(r)["data"]]
            ok = True
            if steered:
                txt = open(log_path, errors="replace").read()
                hits = [l for l in txt.splitlines()
                        if "weightless GLP steering active" in l
                        or "WEIGHTLESS-SHIM: steering core loaded" in l]
                ok = bool(hits)
                if ok and not printed_marker:
                    print("STEERING-LINE:", hits[0], flush=True)
                    printed_marker = True
            if ok:
                break
        except Exception:
            pass
        if time.time() - t0 > 75 * 60:
            _cleanup()
            raise RuntimeError("never ready (port or steering line); "
                               f"log {log_path}")
        time.sleep(20)
    print(f"ready after {time.time() - t0:.0f}s: models={models}",
          flush=True)
    if arm == "lora":
        assert SERVED_MODEL in models and LORA_NAME in models, models
        print(f"LORA OK: both models listed: {models}", flush=True)

    post_chat(SERVED_MODEL, "Say OK.", 8)
    print("warm-up ok", flush=True)

    alpha = os.environ.get("WEIGHTLESS_STEER_ALPHA", "")
    for suite_name, prompts, model, out_name in plan:
        items = [None] * len(prompts)
        t1 = time.time()
        with cf.ThreadPoolExecutor(4) as ex:
            futs = {ex.submit(post_chat, model, p, 400): i
                    for i, p in enumerate(prompts)}
            for fut in cf.as_completed(futs):
                i = futs[fut]
                r = fut.result()
                items[i] = {"i": i, "prompt": prompts[i], **r}
                print(f"  [{out_name}] #{i}: {r['completion_tokens']} tok, "
                      f"fr={r['finish_reason']}", flush=True)
        rec = {"suite": suite_name, "alpha": float(alpha or 0.0),
               "model": model, "max_new": 400, "decoding": "greedy",
               "items": items}
        out_path = f"{log_dir}/{out_name}"
        json.dump(rec, open(out_path, "w"), indent=1)
        vol.commit()
        print(f"{out_name}: {len(items)} completions in "
              f"{time.time() - t1:.0f}s -> {out_path}", flush=True)

    _cleanup()
    print(f"EVAL_ARM {arm} DONE", flush=True)


@app.function(image=image, volumes={"/data": vol}, gpu=GPU,
              min_containers=0, max_containers=1, scaledown_window=180,
              startup_timeout=3600, timeout=6 * 3600,
              env=_serve_env)
@modal.web_server(port=8000, startup_timeout=3600)
def serve():
    """vllm serve on stock 0.31, one arm per deploy.

    Steered arms (a0/a2/lora) fail closed on the steering-active line
    missing from the boot log; the stock arm (plugin installed but
    WEIGHTLESS_STEER_PATH unset — byte-for-byte stock serving) instead
    waits for uvicorn's startup-complete line as the boot proof. The
    function returns once the marker lands so Modal starts routing; vllm's
    stdout appends to a log ON THE VOLUME (Modal auto-restarts crashed web
    tasks; append preserves each attempt's dying words; no mid-boot
    vol.commit — suspected 9P stall under a 198 GB read stream, committed
    on the death path and on SIGTERM instead).
    """
    import signal
    import subprocess
    import time

    arm = os.environ.get("GLM53_ARM", "stock")
    model_path = _model_snapshot_path()
    cmd = _serve_cmd(model_path)
    log_dir = "/data/out-glm53-stock031"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/server-{arm}.log"
    logf = open(log_path, "a")
    logf.write(f"\n\n===== boot attempt {time.strftime('%H:%M:%S')} "
               f"arm={arm} =====\n")
    logf.flush()
    print("+", " ".join(cmd), flush=True)
    print(f"serve: arm={arm} gpu={GPU} tp={TP} "
          f"kv_dtype={KV_DTYPE or 'auto'} eager={bool(ENFORCE_EAGER)} "
          f"-> {log_path}", flush=True)
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            env=dict(
                                os.environ, PYTHONUNBUFFERED="1",
                                PYTHONPATH="/opt/weightless-shim:"
                                + os.environ.get("PYTHONPATH", "")))

    def _sigterm(signum, frame):
        vol.commit()
        proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm)

    if arm == "stock":
        markers = ("Application startup complete",)
    else:
        markers = ("weightless GLP steering active",
                   "WEIGHTLESS-SHIM: steering core loaded")
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
        hit = [l for l in txt.splitlines() if any(m in l for m in markers)]
        if hit:
            for line in hit[:4]:
                print("MARKER:", line, flush=True)
            print("serve: boot marker confirmed; returning so Modal starts "
                  "routing (port binds when uvicorn comes up)", flush=True)
            return
    txt = open(log_path, errors="replace").read()
    print(f"MARKER {markers} MISSING after 30 min — refusing to serve. "
          "Log tail:\n", txt[-4000:], sep="", flush=True)
    proc.terminate()
    vol.commit()
    raise RuntimeError(f"boot marker {markers} never appeared in boot log")


@app.local_entrypoint()
def main():
    print(__doc__)
