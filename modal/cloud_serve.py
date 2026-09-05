"""GLM-5.3 flagship serving on Modal H100:8; run through the Modal CLI.

Reuses refusal-research/experiments/20260829-glm53-flagship's volume,
vLLM image, marlin backend and calibrated GLP-77 directions. Serving runs
CUDA graphs (eager is the capture/eval-lane discipline, never serving).
Set WEIGHTLESS_STEER_ALPHA before deployment to override the 1.0 default.
"""
import json
import os
from pathlib import Path
import subprocess

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "RadixArk/GLM-5.3-NVFP4"
SERVED_MODEL = "glm-5.3"
VOLUME_NAME = "glm53-nvfp4"
DIRS_PATH = "/data/out/glm53-32perlayer-dirs.pt"
PATCH_NAME = "hotfix-glm53-modal-residual.py"
EXPERIMENT = "refusal-research/experiments/20260829-glm53-flagship"
ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}

app = modal.App("weightless-cloud")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
download_image = (modal.Image.debian_slim(python_version="3.12")
                  .pip_install("huggingface_hub", "hf_transfer"))
image = (modal.Image.from_registry("vllm/vllm-openai:v0.28.0", add_python="3.12")
         .entrypoint([])
         .add_local_file(ROOT / "patches" / PATCH_NAME, "/work/" + PATCH_NAME, copy=True))


def fix_kv_scheme(snapshot):
    """Hopper sparse MLA requires bf16 KV; the checkpoint declares fp8.

    Replace the config symlink, preserving the original HF cache blob.
    This is the experiment's fix_kv_scheme, applied to the resolved snapshot.
    """
    path = Path(snapshot) / "config.json"
    config = json.loads(path.read_text())
    quant = config.get("quantization_config") or {}
    if "kv_cache_scheme" in quant:
        quant.pop("kv_cache_scheme")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(path)


@app.function(image=download_image, volumes={"/data": vol}, timeout=6 * 3600,
              secrets=[modal.Secret.from_name("hf-token")], env=ENV)
def ensure_weights():
    """Resume/cache the ~465 GB snapshot on CPU; no GPU allocation."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID)
    fix_kv_scheme(snapshot)
    vol.commit()
    print("weights ready:", snapshot)


def require_dirs():
    if not Path(DIRS_PATH).is_file():
        raise FileNotFoundError(
            f"Missing {DIRS_PATH} on volume {VOLUME_NAME}. Restore the existing "
            f"GLP-77 .pt from {EXPERIMENT} to /out/glm53-32perlayer-dirs.pt "
            "on that volume. This serving lane does not derive directions.")


@app.function(image=download_image, volumes={"/data": vol}, timeout=60)
def ensure_dirs():
    """Verify the experiment's existing .pt; never capture or re-derive."""
    require_dirs()
    print("directions ready:", DIRS_PATH)


@app.function(image=image, volumes={"/data": vol}, gpu="H100:8",
              min_containers=0, max_containers=1, scaledown_window=300,
              timeout=30 * 60,
              env=dict(ENV, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       WEIGHTLESS_STEER_PATH=DIRS_PATH,
                       WEIGHTLESS_STEER_ALPHA=os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0")))
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=30 * 60)
def serve():
    """Cold starts load ~465 GB; allow up to 30 minutes for readiness."""
    require_dirs()
    subprocess.run(["/usr/bin/python3.12", "/work/" + PATCH_NAME], check=True)
    # Use the image's Python, as in the proven experiment (not Modal's shim).
    # Serving profile: CUDA graphs ON (eager is a capture/eval-lane
    # discipline, never a serving one), prefix caching on, agentic ctx.
    subprocess.Popen([
        "/usr/bin/python3.12", "-m", "vllm.entrypoints.cli.main", "serve", MODEL_ID,
        "--served-model-name", SERVED_MODEL,
        "--host", "0.0.0.0", "--port", "8000",
        "--tensor-parallel-size", "8", "--moe-backend", "marlin",
        "--max-model-len", os.environ.get("MAX_MODEL_LEN", "131072"),
        "--gpu-memory-utilization", "0.92",
        "--enable-auto-tool-choice", "--tool-call-parser", "glm47",
        "--reasoning-parser", "glm45",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
    ])
