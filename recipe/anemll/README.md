# Anemll (MiaAI 2x DGX Spark) recipe — our canonical copies

The live serving stack is the
[MiaAI-Lab 2x recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark),
cloned at `~/dspark-miaai` on **both** cluster nodes, running the Anemll
image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2, blobs archived
at `~/anemll-oci` on the head). This directory holds the canonical copies of
**our local state on top of that clone** — the files we modified, as
modified. If a node is rebuilt: clone upstream, copy these three files over
it, plus `../../patches/hotfix-dsv4-steering-projective.py` into `patches/`.

| file | our changes vs upstream (as of 2026-08-21, upstream merged @6d00e4a) |
|---|---|
| `docker-compose.dspark.yml` | steering hotfix mount, `DSPARK_STEER_*` env passthrough, entrypoint runs the hotfix (`\|\| exit 1`) |
| `start-deepseek-v4-flash-dspark.sh` | worker-sync block for the steering hotfix |
| `.env.dspark` | full live config: dual-rail fabric (GID index 3 pinned), `DSPARK_REVISION=7872f01b`, served name `deepseek-v4-flash-dspark`, 1M `MAX_MODEL_LEN`, spec decode k=5, `DSPARK_STEER_PATH/_ALPHA=4.0/_LAYERS=10..38` |

Gotchas:

- The start script syncs hotfixes and `.env.dspark` to the worker but **not**
  `docker-compose.dspark.yml` — sync it manually when it changes.
- NCCL GID indexes drift across reboots; re-verify per `.env.dspark` notes.
- Upstream merged 2026-08-21 (10 commits: NCCL fabric passthrough with
  empty-to-unset normalization, `DRAFT_SAMPLE_METHOD` probabilistic|greedy
  gate — the k=7/greedy A/B lever is now one env line). Contract tests
  29/29 + 7/7 green; takes effect at next boot, no restart was needed.
