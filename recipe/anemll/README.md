# Anemll (MiaAI 2x DGX Spark) recipe — our canonical copies

The live serving stack is the
[MiaAI-Lab 2x recipe](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark),
cloned at `~/dspark-miaai` on **both** cluster nodes, running the Anemll
image `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` (vLLM 0.25.2, blobs archived
at `~/anemll-oci` on the head). This directory holds the canonical copies of
**our local state on top of that clone** — the files we modified, as
modified. If a node is rebuilt: clone upstream, copy `docker-compose.dsv4.yml`
and `start-deepseek-v4-flash-dspark.sh` over it, copy `.env.dsv4.example`
to `.env.dsv4` and fill in the `<...>` placeholders, plus
`../../patches/hotfix-dsv4-steering-projective.py` into `patches/`.

| file | our changes vs upstream (as of 2026-08-21, upstream merged @6d00e4a) |
|---|---|
| `docker-compose.dsv4.yml` | steering hotfix mount, `WEIGHTLESS_STEER_*` env passthrough, entrypoint runs the hotfix (`\|\| exit 1`) |
| `start-deepseek-v4-flash-dspark.sh` | worker-sync block for the steering hotfix |
| `.env.dsv4.example` | full live config with site values as `<...>` placeholders: dual-rail fabric (GID index 3 pinned), `DSPARK_REVISION=7872f01b`, served name `deepseek-v4-flash-dspark`, 1M `MAX_MODEL_LEN`, spec decode k=5, `WEIGHTLESS_STEER_PATH/_ALPHA=4.0/_LAYERS=10..38` |

## Steering vector

The vector is a 478 KB GGUF control vector (spec:
[`../../spec/GLP.md`](../../spec/GLP.md)), read by the
hotfix from `/cache/huggingface` inside the container (= `$HF_CACHE` on the
host, both nodes). Published artifact:
[`msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29`](https://huggingface.co/msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29)
(gated — fetch with an HF token):

```sh
huggingface-cli download msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29 \
  --local-dir ~/.cache/huggingface   # on BOTH nodes
```

`.env.dsv4` currently points `WEIGHTLESS_STEER_PATH` at a general-contrast
variant (`...-general-abliterated-cvec-L10-38-a4-keysdir.gguf`, from the same
derivation); the cyber file above is the documented swap. Both were verified
tensor-identical to their `.pt` sources (cos 1.0000/layer) and carry base rev
`7872f01b` = the pinned `DSPARK_REVISION`. Off = empty `WEIGHTLESS_STEER_PATH`.

Gotchas:

- The start script syncs hotfixes and `.env.dsv4` to the worker but **not**
  `docker-compose.dsv4.yml` — sync it manually when it changes.
- NCCL GID indexes drift across reboots; re-verify per `.env.dsv4` notes.
- Upstream merged 2026-08-21 (10 commits: NCCL fabric passthrough with
  empty-to-unset normalization, `DRAFT_SAMPLE_METHOD` probabilistic|greedy
  gate — the k=7/greedy A/B lever is now one env line). Contract tests
  29/29 + 7/7 green; takes effect at next boot, no restart was needed.
- Known flake (2026-08-22): the serving layer intermittently emits invalid
  JSON on the tool-call path — raw control characters (literal newlines)
  inside a string, failing strict JSON parsers client-side. Not content-
  determined (the identical bytes serialize fine before and after), nothing
  in the server logs, and it comes in **bursts**: minutes where every
  tool-call response is malformed, then long clean stretches (25+/25).
  That signature points at a concurrency bug in the image's custom response
  assembly (the response carries nonstandard fields like `routed_experts`).
  `tests/smoke/03-tool-call.sh` retries once and reports byte offset + context
  when both attempts fail — during a burst both do, which is the signal.
  Reported upstream: https://github.com/Anemll/dspark-vllm-gx10/issues/10

## Ops: wedge self-healing (armed 2026-09-10)

The 2026-09-10 incident: the worker wedged at 21:05 UTC (silent freeze —
journal stops mid-cron, no panic/OOM; the known GB10 wedge class). The head
lost its fabric address, vLLM exited fast against the dead peer, and
`unless-stopped` crash-looped it 2,825 times in 9h. Fixes, in layers:

- **Peer gate** (`docker-compose.dsv4.yml`, top of the boot command): rank 0
  waits for `WORKER_VLLM_HOST_IP`, rank 1 for `MASTER_ADDR`, pinging every
  30s instead of crash-looping. Takes effect at next container recreate —
  and remember the start script does NOT sync this file to the nodes; copy
  it over manually when it changes.
- **Hardware watchdog** (both Sparks): `/etc/systemd/system.conf.d/watchdog.conf`
  sets `RuntimeWatchdogSec=60` — systemd pings the SBSA watchdog at
  `/dev/watchdog0`; a full OS freeze hard-resets the board within 60s.
- **Panic-on-hang sysctls** (both Sparks, `/etc/sysctl.d/99-wedge-heal.conf`):
  `softlockup_panic=1`, `hung_task_panic=1`, `panic=30` — kernel lockups and
  hung tasks (the GPU-driver D-state class) panic and reboot instead of
  sitting frozen.
- Diagnose: wizard option 5 checks restart counts (>50 = crash loop), the
  node's fabric address, and per-peer pings, with interpretation.

Untested live: the watchdog has not been fired deliberately (needs a worker
reboot window). kdump/pstore for post-mortem evidence is NOT armed —
`crashkernel=` needs a boot-param change and a maintenance window.
