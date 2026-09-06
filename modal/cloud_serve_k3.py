"""Kimi K3 serving on Modal 2x H200:8; run through the Modal CLI.

Reuses refusal-research/experiments/20260903-kimi-k3-glp's volume (kimi-k3),
day-0 image (vllm/vllm-openai:kimi-k3), fail-closed steering hotfix, and the
validated PP=2 x TP=8 external_launcher shape (16 boots; the fork's
leader/headless multi-node serving wedges and its SO_REUSEPORT serving shape
hangs 7/8 of requests -- see the experiment's STATE.md). Serving is the
collective lockstep driver in k3_serve_driver.py: offline LLM() on every
rank, rank 0 fans requests out over ZMQ. CUDA graphs on (PIECEWISE); eager
is a capture-lane discipline, never serving.

Set WEIGHTLESS_STEER_ALPHA before `modal deploy` to override the 1.0 default
(0.0 = hooks installed, projection x 0 = stock behavior).
"""
import os
from pathlib import Path
import subprocess

import modal
import modal.experimental

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "moonshotai/Kimi-K3"
SERVED_MODEL = "kimi-k3"
VOLUME_NAME = "kimi-k3"
DIRS_PATH = "/data/out/k3-perlayer-dirs.pt"
VECTOR_REPO = "msuiche/Kimi-K3-abliterated-cyber-GLP-92-L1-92-a1.0"
VECTOR_FILE = "glp.kimi-k3.dirs.pt"
PATCH_NAME = "hotfix-kimi-k3-steering-projective.py"
DRIVER_NAME = "k3_serve_driver.py"
EXPERIMENT = "refusal-research/experiments/20260903-kimi-k3-glp"
ENV = {"HF_HOME": "/data/hf", "HF_XET_HIGH_PERFORMANCE": "1"}
PY = "/usr/bin/python3.12"

app = modal.App("weightless-cloud-k3")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
download_image = (modal.Image.debian_slim(python_version="3.12")
                  .pip_install("huggingface_hub", "hf_xet"))
image = (modal.Image.from_registry("vllm/vllm-openai:kimi-k3",
                                   add_python="3.12",
                                   setup_dockerfile_commands=[
                                       "ENTRYPOINT []"])
         .add_local_file(ROOT / "patches" / PATCH_NAME,
                         "/work/" + PATCH_NAME, copy=True)
         .add_local_file(ROOT / "modal" / DRIVER_NAME,
                         "/work/" + DRIVER_NAME, copy=True))


@app.function(image=download_image, volumes={"/data": vol},
              timeout=24 * 3600,
              secrets=[modal.Secret.from_name("hf-token")], env=ENV)
def ensure_weights():
    """Resume/cache the ~1.56 TB snapshot on CPU; no GPU allocation.

    Fail-closed size-gate (METHODOLOGY section 16): 96 shards + index,
    zero .incomplete blobs, >1.5 TB before any GPU spend."""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(MODEL_ID)
    snap = Path(snapshot)
    shards = sorted(snap.glob("*.safetensors"))
    hub = Path(os.environ["HF_HOME"]) / "hub" / "models--moonshotai--Kimi-K3"
    incomplete = list((hub / "blobs").glob("*.incomplete"))
    total = sum(f.stat().st_size for f in (hub / "blobs").iterdir()
                if not f.name.endswith(".incomplete"))
    print(f"shards: {len(shards)}, incomplete: {len(incomplete)}, "
          f"bytes: {total / 1e12:.4f} TB")
    assert len(shards) == 96, f"expected 96 shards, found {len(shards)}"
    assert not incomplete, f"INCOMPLETE BLOBS: {incomplete[:5]}"
    assert total > 1.5e12, f"size-gate failed: {total / 1e12:.3f} TB"
    vol.commit()
    print("weights ready:", snapshot)


@app.function(image=download_image, volumes={"/data": vol}, timeout=300,
              secrets=[modal.Secret.from_name("hf-token")], env=ENV)
def ensure_dirs():
    """Verify the GLP-92 per-layer stack on the volume; fetch it from the
    gated HF vector repo if missing. This serving lane never derives
    directions."""
    dst = Path(DIRS_PATH)
    if not dst.is_file():
        from huggingface_hub import hf_hub_download

        fetched = hf_hub_download(repo_id=VECTOR_REPO, filename=VECTOR_FILE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(fetched).read_bytes())
        vol.commit()
    size = dst.stat().st_size
    assert size > 1e6, f"{DIRS_PATH} looks truncated ({size} B)"
    print("directions ready:", DIRS_PATH)


@app.server(image=image, volumes={"/data": vol}, gpu="H200:8",
            min_containers=0, max_containers=2, scaledown_window=600,
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
                     WEIGHTLESS_STEER_PATH=DIRS_PATH,
                     WEIGHTLESS_STEER_ALPHA=os.environ.get(
                         "WEIGHTLESS_STEER_ALPHA", "1.0")))
@modal.experimental.clustered(size=2, rdma=False)
class K3Server:
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
        ci = modal.experimental.get_cluster_info()
        rank = ci.rank
        master = ci.container_ips[0]
        print(f"serve: rank={rank} master={master} "
              f"ips={ci.container_ips}", flush=True)

        # Fail-closed patch on BOTH nodes (each patches its own fs).
        subprocess.run([PY, "/work/" + PATCH_NAME], check=True)

        # external_launcher under torchrun: ALL 16 processes run the same
        # driver symmetrically; rank 0 also serves HTTP on :8000.
        cmd = [
            PY, "-m", "torch.distributed.run",
            "--nnodes=2", f"--node-rank={rank}", "--nproc-per-node=8",
            f"--master-addr={master}", "--master-port=29511",
            "/work/" + DRIVER_NAME,
        ]
        self._proc = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self._proc.terminate()
