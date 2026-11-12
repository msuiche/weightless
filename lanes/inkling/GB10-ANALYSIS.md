# Inkling-Small-NVFP4 on 2× DGX Spark (GB10, sm_121) — boot-failure root-cause analysis

**Scope:** source analysis only, against `staging/srcdl-inkling/inkling/` (exact source of the
`vllm/vllm-openai:v0.28.0` image). No GPU jobs were run.
**Date:** 2026-09-03.

## 0. In-image path mapping

The staging tree is flattened. The imports in `model.py` pin the real layout inside the image:

| staging file | in-image path (reconstructed from imports) |
|---|---|
| `inkling/model.py` | `vllm/models/inkling/<nvidia>/model.py` (`model.py:38-45` imports `vllm.models.inkling.common.*`; `model.py:49` imports `..configs`) |
| `inkling/configs.py` | `vllm/models/inkling/configs.py` |
| `inkling/{lamport,norm,sconv,qkvr_prep,fa4_rel_attention,silu_and_mul}.py` | `vllm/models/inkling/<nvidia>/ops/*.py` (`model.py:55-56`, `short_conv.py:34`, `attention.py:37-42`, `__init__.py:12-18`) |
| `inkling/{attention,sconv_swa_attn,short_conv,moe,mtp,layernorm,logits_processor,mlp,towers}.py` | `vllm/models/inkling/<nvidia>/*.py` |

All `file:line` citations below refer to the staging files.

---

## 1. The cross-node init/forward sequence as the code defines it

### 1.1 Construction (before weight load)

1. Each worker builds `InklingForCausalLM._build` (`model.py:418-454`): 41 `InklingDecoderLayer`s.
   Each layer registers **two** entries in `compilation_config.static_forward_context`:
   - `InklingConvState` (`sconv_swa_attn.py:147-218`) — an `AttentionLayerBase` owning the layer's
     paged short-conv state as a `SlidingWindowSpec` (`sconv_swa_attn.py:211-218`), block_size =
     `sconv_kernel_size` (4), 4 streams (K, V, attn-out, mlp-out) packed head-major
     (`sconv_swa_attn.py:188-193`). Asserts `tp_size <= num_kv_heads` (`sconv_swa_attn.py:168-171`)
     — holds at TP=2.
   - `InklingAttention` (`attention.py:171-174`), backend hard-wired to `FlashAttentionBackend`
     for **metadata only** (`attention.py:177-178`); the real kernel is the custom FA4 rel kernel.
2. `initialize_lamport_rs_conv(hidden, sconv_kernel, max_num_batched_tokens)` runs at
   `model.py:434-438`. With `LAMPORT_RS_SCONV=0` it returns immediately (`lamport.py:775-776`) and
   `_STATE` stays `None` **forever**. Without the kill switch, cross-node TP would raise
   `"cross-node TP is supported only on MNNVL fabric"` (`lamport.py:541-542`), which is *caught*
   (`lamport.py:780-783`) — same NCCL fallback, just logged. Either way the fused Lamport path is
   dead on this deployment and nothing downstream can resurrect it:
   `get_lamport_rs_conv` returns `None` (`lamport.py:785-791`) → `model.py:94-95` short-circuits →
   the fallback at `model.py:115-123` always runs.
3. **The fallback is fully cross-node-safe at the inkling level.** It is:
   NCCL `tensor_model_parallel_reduce_scatter` → local `fused_sconv` Triton kernel → NCCL
   `tensor_model_parallel_all_gather` → fused `add_rmsnorm` (`model.py:115-123`). No MNNVL, no
   NVSHMEM, no fabric-mapped buffer, no `symm_mem` anywhere in the fallback. The only cross-node
   traffic in the whole forward is these RS/AG pairs (4 per layer × 41 layers) through vLLM's
   `GroupCoordinator` → pynccl over RoCE. MoE sets `skip_final_all_reduce = True` (`moe.py:457`)
   because the RS *is* the reduction; `wo_ud` is `reduce_results=False` (`attention.py:120-130`).
4. The HF config exposes `mamba2_cache_params` (`configs.py:185-215`, 6 conv streams,
   `layers=conv_layer_ids`) — the vLLM-core hybrid/Mamba state-cache hook — **in addition** to the
   per-layer `SlidingWindowSpec` from (1). See H5.

### 1.2 Weight load

`model.py:647-703`: remapped stacked loads, MoE expert translation (`moe.py:546-617`), replicated
full-vocab embedding (~2.3 GiB/rank, `model.py:243-261`), `finalize_load` (`moe.py:619-641`).
Last DEBUG line observed in the field = layer 41/41 — i.e. control returns from `load_weights`.

### 1.3 Post-load phases (the crash window, ~28 s)

In vLLM v1 order after `load_weights` returns:

- **(a) `process_weights_after_loading` / quant post-processing** for the ModelOpt NVFP4 methods
  (weight repacking / scale swizzling on device — vLLM core, not in this tree).
- **(b) AOT JIT warmup of `INKLING_FA4_REL_ATTENTION_KERNEL`** (`fa4_rel_attention.py:436`,
  instantiated at import; `VllmJitKernel` registered for ahead-of-time CuTeDSL compile — the ROCm
  shim confirms "The NVIDIA path registers ahead-of-time CuTeDSL units", `fa4_warmup.py:5-7`).
  `get_warmup_keys` (`fa4_rel_attention.py:258-325`) enumerates ~`log2(max_num_batched_tokens)`
  query buckets × `num_reqs` buckets × 2 layer types (local SWA + global), each a full CuTeDSL
  (cutlass Python → MLIR → cubin) compile under `FakeTensorMode` (`fa4_rel_attention.py:327-398`).
  Dozens of native compiles = tens of seconds, re-run every boot (fresh containers, no warm
  compile cache). **This alone matches the ~28 s post-load delay.**
- **(c) Memory-profiling dummy forward** (T = `max_num_batched_tokens`): first *real* execution of
  every custom kernel — Triton first-call JIT for `embed_rmsnorm` (`norm.py:227-268`),
  `fused_qkvr_prep` (`qkvr_prep.py:794-918`, aux-stream dual launch), the gate select
  (`moe.py:80-189`), fused-MoE NVFP4, `fused_sconv` (`sconv.py:152-220`), `add_rmsnorm`
  (`norm.py:146-185`) — plus the first real **FA4 rel-attention launch**
  (`attention.py:292-307`) and the first **real cross-node NCCL RS/AG collectives**
  (`model.py:118,120`). (NCCL "working at init" = connection setup only; the first data-path
  buffers are allocated here.)
- **(d) KV-cache allocation**: paged attention blocks + the sconv SWA caches
  (`sconv_swa_attn.py:116-125`) after page unification, and possibly the Mamba-style state cache
  from `mamba2_cache_params` (vLLM core).
- **(e) CUDA-graph capture + decode-shape warmup** (skipped under `--enforce-eager`; crash
  persists under eager, so this phase is *not required* for the failure).

### 1.4 Decode-path note (reached only if boot succeeded)

T ≤ 64 rows route the gate GEMM through `ll_bf16.ll_bf16_gemm` (`moe.py:60-77`), gated by
`current_platform.has_device_capability(90)` (`moe.py:71`) — **true on sm_121** (121 ≥ 90).
See H4.

---

## 2. Platform-gate inventory (the entire tree)

Exhaustive sweep for architecture branching in the NVIDIA tree:

| site | gate | sm_121 (GB10) result |
|---|---|---|
| `fa4_rel_attention.py:30-33` | `_use_sheared_bias()`: `capability.major in (10, 11)` | **False → Hopper path selected on a Blackwell-family chip** |
| `fa4_rel_attention.py:81-83` | `inkling_fa4_num_splits`: `major == 9 → 1` | falls into Blackwell split-KV heuristic (intended) |
| `moe.py:71` | `has_device_capability(90)` + `ll_bf16.is_available()` | True → CuTeDSL decode GEMM enabled (T ≤ 64 only) |
| `lamport.py:521` | PDL `capability >= 9` | dead code here (Lamport disabled) |

The ROCm material — `rel_mha_decode_gfx950.py`, `rel_mha_extend_gfx950.py`,
`rel_attention_decode.py`, `fa4_warmup.py` — has **zero importers** in this tree (verified by
grep): it cannot fire on NVIDIA aarch64. Lead "AMD kernels mis-selected on GB10" is **cleared**.

Lead 2 from the tasking ("does the NCCL fallback actually cover cross-node TP=2?") is also
**cleared at the inkling level** — §1.1 (3): the fallback contains no MNNVL/fabric assumption;
the only cross-node assumption left is pynccl RS/AG itself.

So the tree contains exactly **one** place where a Blackwell-family GPU is mis-classified by
version enumeration: `fa4_rel_attention.py:33`.

---

## 3. Ranked root-cause hypotheses

### H1 (primary, codepath-level): sm_121 falls through `_use_sheared_bias()` and Inkling runs the Hopper FA4 score-mod path on GB10

`fa4_rel_attention.py:30-33`:

```python
@cache
def _use_sheared_bias() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major in (10, 11)
```

GB10 reports compute capability **(12, 1)** → `major == 12` → `False`. The kernel then takes the
"Hopper" branch (`fa4_rel_attention.py:177-186`): `vllm.vllm_flash_attn.cute.flash_attn_varlen_func`
+ the Python `score_mod` closure (`fa4_rel_attention.py:37-68`). The code's own comment states
the intended split (`fa4_rel_attention.py:162-164`): *"Hopper uses standard FA4's score-mod
gather. Blackwell uses tml-fa4's sheared relative-bias layout."* GB10 **is** Blackwell
(Grace-Blackwell GB10, sm_121) — the `(10, 11)` enumeration simply predates/omits sm_12x, and the
model is silently routed onto a code path its authors did not intend for this silicon.

Why this can kill the worker **with no Python traceback**:

- The mis-selected path is exercised in the post-load window twice: AOT-compiled for every warmup
  key (§1.3b) and really-launched in the profiling dummy forward (§1.3c). CuTeDSL compilation is
  native code (cutlass Python → MLIR → ptxas). A native fault (segfault/abort in the compiler for
  an arch its tables half-support, or a cubin/ptxas mismatch for `sm_121a` family features) kills
  the worker process outright: EngineCore then reports exactly
  "WorkerProc initialization failed due to an exception in a background process" with no Python
  traceback, on both nodes deterministically, in the same post-load window.
- If the mis-compiled kernel instead launches, a device-side fault poisons the CUDA context; on
  the GB10 unified-memory driver the channel teardown surfaces in dmesg as NVRM
  `NV_ERR_NO_MEMORY`, and the process can die in a native call before torch ever converts the
  sticky context error into a Python exception.
- A manual `torch.empty` 115 GB probe exercises *none* of this (no DSL compiler, no custom
  kernel), which is exactly why it passes.

**Candid caveat:** the NVRM `NV_ERR_NO_MEMORY` dmesg line is the weakest link in H1 — it is a
driver-level message, and H1 explains it only as teardown fallout. This is why §6 exists: the
standalone probe discriminates H1/H2 from H3 in one 2-minute run.

### H2 (kernel-level sibling): *neither* FA4 path supports sm_121 in this image

If the probe (§6.1) shows the tml-fa4 sheared path *also* fails on sm_121, then the image's
CuTeDSL/tml-fa4 build simply has no sm_121 (or sm_120-family) support — sm_100a cubins do not load
on sm_121, and the JIT tables predate the arch. H1's patch then converts the failure mode but not
the failure; the fix is upstream (§7 bug report). H1 and H2 share the same signature and the same
elimination table; they are separated only by the probe.

### H3 (secondary, infra-level): the first *real* cross-node NCCL collective dies at the driver level

With Lamport off, the first cross-node data collectives are the RS/AG pairs in the profiling
forward (`model.py:118,120`). On GB10 (unified memory, no nvidia-peermem/GDR over RoCE), NCCL's
buffer/window registration (cuMem VMM APIs, symmetric-window registration in recent NCCL) can be
refused by NVRM with `NV_ERR_NO_MEMORY`. **Against:** `NCCL_IB_DISABLE=1` (socket transport) still
dies, and NCCL failures characteristically print `NCCL WARN ...` and abort loudly rather than
silently. Not fully eliminated (IB_DISABLE does not disable NCCL's VMM window registration), hence
kept as the fallback hypothesis if the §6.1 probe clears FA4 entirely.

### H4 (decode-only, cannot be the primary): `ll_bf16` decode GEMM on sm_121

`moe.py:60-77`: the CuTeDSL `ll_bf16_gemm` is selected when `has_device_capability(90)`
(true on sm_121) **and** T ≤ 64. It cannot fire in the profiling forward (T = 2048/8192) and
`--enforce-eager` still dies, so it is not the boot killer. It is a live landmine for the
*first decode step / cudagraph capture after boot is fixed* — keep the one-line guard in §5.3 in
the back pocket.

### H5 (accounting risk, wrong signature): sconv state double-allocation

The config advertises `mamba2_cache_params` (`configs.py:185-215`) *and* every layer registers a
paged `SlidingWindowSpec` for the same conv state (`sconv_swa_attn.py:211-218`). If vLLM core
honors both hooks, the sconv state is allocated twice (paged blocks + a Mamba-style
`max_num_seqs × layers` state tensor). The mamba-style tensor is small here (~MBs; `conv_len=3`,
streams ≤ 6144 wide) and any oversubscription would raise a **torch `OutOfMemoryError` with a
traceback** — not the observed signature. Logged for the follow-up audit, not a crash candidate.

---

## 4. Why H1/H2 match every elimination result

| eliminated variable | why it cannot matter under H1/H2 |
|---|---|
| `gpu_memory_utilization` 0.835/0.89/0.90 | the failure is not an allocation-size problem; kernel selection is capability-gated, memory-agnostic. The 115 GB torch probe already proved headroom. |
| `max_model_len` 262K/128K/32K | changes KV sizing and the split-KV `max_kv_len` heuristic input only (`fa4_rel_attention.py:71-109`); the mis-selected backend fires regardless. |
| `max_num_seqs` 8/4, `max_num_batched_tokens` 8192/2048 | changes *which* warmup keys enumerate (`fa4_rel_attention.py:258-325`) and the profiling T, but every configuration compiles and launches the same mis-selected kernel. |
| page caches / stale containers / full reboots | deterministic codepath, not state; a fresh container only forces a *cold* CuTeDSL compile cache, i.e. guarantees the ~28 s compile window is hit every boot. |
| NCCL HCA typo fix, `NCCL_IB_DISABLE=1` | H1 needs no collective at all — the mis-selected kernel is reached in the profiling forward / AOT warmup independent of transport. |
| `--enforce-eager` | AOT `VllmJitKernel` warmup and the profiling dummy forward both run with cudagraphs off (the kernels are invoked from `attention.py` directly). |
| MTP spec-config on/off | MTP reuses the same `InklingDecoderLayer`/FA4 kernel (`mtp.py:71-…`, `mtp.py:42`); the 41-layer backbone hits it regardless. |
| `VLLM_USE_V2_MODEL_RUNNER` on/off | the FA4 kernel is called from model code (`attention.py:292-307`); both runners publish the required `FlashAttentionMetadata` + `InklingSconvMetadata` (`attention.py:225-228`). |
| stock unpatched `model.py` | the bug *is* stock code (`fa4_rel_attention.py:33`). |
| `LAMPORT_RS_SCONV=0` | removes the only other MNNVL-dependent path (`lamport.py:541-542`); the fallback is pure NCCL + arch-generic Triton (§1.1). |
| H100 x86 single-node TP=4 boots fine | sm_90 → `_use_sheared_bias()` is *also* False there, but for Hopper the score-mod cute path is the **intended, supported** one (`fa4_rel_attention.py:162-164`). The mis-selection is only fatal where the hardware is Blackwell-family but the version check says otherwise — i.e. exactly sm_121. The cross-node/H100 comparison is confounded by arch, and H1 predicts GB10 would fail single-node too (untestable at 159 GB — see §6.2 note). |
| ~28 s after layer 41/41 | the post-load window is (a) NVFP4 post-load processing, (b) dozens of AOT CuTeDSL compiles (`fa4_rel_attention.py:258-325`), (c) first-call Triton JIT + first real FA4 launch in profiling. All three live in the seconds-to-tens-of-seconds range after the last weight lands; a cold per-boot compile cache makes the delay deterministic. |

---

## 5. Candidate patch (codepath-level; hotfix against the image)

### 5.1 The fix — `ops/fa4_rel_attention.py` (in-image: `vllm/models/inkling/<nvidia>/ops/fa4_rel_attention.py`)

```diff
+import os
+
 from vllm.platforms import current_platform

 @cache
 def _use_sheared_bias() -> bool:
+    # GB10 (DGX Spark) is sm_121 — Blackwell family. The previous
+    # ``major in (10, 11)`` enumeration silently routed sm_12x to the Hopper
+    # score-mod path (vllm.vllm_flash_attn.cute), which is not the supported
+    # Blackwell path for this kernel. Treat every major >= 10 as Blackwell.
+    # INKLING_FA4_SHEARED_BIAS=0/1 overrides the probe for bring-up triage.
+    override = os.environ.get("INKLING_FA4_SHEARED_BIAS")
+    if override is not None:
+        return override.strip().lower() not in ("", "0", "false", "no")
     capability = current_platform.get_device_capability()
-    return capability is not None and capability.major in (10, 11)
+    return capability is not None and capability.major >= 10
```

The env override is deliberate: it lets the operator test **both** branches on sm_121 without
rebuilding the image (`0` = old behavior, `1` = forced tml-fa4), which is also the A/B control
that converts the patch into the hypothesis test.

Do **not** change `inkling_fa4_num_splits` (`fa4_rel_attention.py:81-83`): `major == 9 → 1` is a
perf heuristic; sm_121 correctly falls into the Blackwell branch.

### 5.2 Apply in the running container (same pattern as the previous steering hotfix)

```bash
F=$(python -c "import vllm.models.inkling as i, pathlib, pkgutil; \
print(next(p for p in pathlib.Path(i.__path__[0]).rglob('fa4_rel_attention.py')))")
python - <<'EOF'
import re, pathlib
f = pathlib.Path("$F")  # resolved above
src = f.read_text()
old = "    capability = current_platform.get_device_capability()\n    return capability is not None and capability.major in (10, 11)"
assert old in src, "gate not found — image drifted"
# (apply the diff from §5.1)
EOF
```

### 5.3 Back-pocket guard (only if boot then dies at first decode/capture)

```python
# moe.py:71 — restrict ll_bf16 to the arches it was validated on
and current_platform.has_device_capability(90)
and current_platform.get_device_capability().major in (9, 10, 11)
```

---

## 6. The exact discriminating runs for the operator

### 6.0 First, zero-cost: re-read the DEBUG logs you already have

Take any captured failed boot and print the **last ~50 worker lines before** "WorkerProc
initialization failed". The last phase logged decides the ranking before any new run:

- last lines are JIT/`VllmJitKernel`/CuTeDSL compile messages → H1/H2 (compile-time native crash).
- last lines are memory-profiling / dummy-run → profiling-forward fault (H1 at first real launch, or H3).
- last lines are NCCL `NET`/transport INFO → H3.
- Also run `dmesg -T | grep -iE "xid|nvrm|oom-kill"` on both nodes to rule the host OOM-killer
  in/out definitively (unified memory = host RAM; SIGKILL also leaves no traceback).

### 6.1 Standalone FA4 probe — 2 minutes, no server, no weights, single node

Run **inside the `vllm/vllm-openai:v0.28.0` container on one Spark node**. This directly executes
the two candidate kernels with toy tensors and is the decisive H1/H2-vs-H3 discriminator:

```bash
docker run --rm --gpus all vllm/vllm-openai:v0.28.0 python - <<'EOF'
import faulthandler, torch
faulthandler.enable()
print("device_capability =", torch.cuda.get_device_capability(), flush=True)  # expect (12, 1)

import vllm.models.inkling as ink, pkgutil
print("inkling subpackages:", [m.name for m in pkgutil.iter_modules(ink.__path__)], flush=True)
import importlib
m = importlib.import_module("vllm.models.inkling.nvidia.ops.fa4_rel_attention")  # adjust to printed name

T, H, KVH, D, EXT, BLK = 8, 16, 8, 128, 64, 16
q   = torch.randn(T, H, D, dtype=torch.bfloat16, device="cuda")
kv  = torch.randn(1, 2, BLK, KVH, D, dtype=torch.bfloat16, device="cuda")
kc, vc = kv.unbind(1)
rel = torch.randn(T, H, EXT, dtype=torch.bfloat16, device="cuda")
kw = dict(block_table=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
          cache_seqlens=torch.tensor([BLK], dtype=torch.int32, device="cuda"),
          cu_seqlens_q=torch.tensor([0, T], dtype=torch.int32, device="cuda"),
          max_seqlen_q=T, softmax_scale=1.0 / D, causal=True,
          window_size=(-1, -1), rel_extent=EXT, rel_logits=rel, num_splits=1)

print("sheared_bias probe =", m._use_sheared_bias(), flush=True)  # must print False on sm121 -> bug
INK = m.INKLING_FA4_REL_ATTENTION_KERNEL
INK(q, kc, vc, **kw); torch.cuda.synchronize()
print("PATH-A (cute score-mod, today's sm121 selection): OK", flush=True)

m._use_sheared_bias = lambda: True   # monkeypatch the @cache'd gate
INK(q, kc, vc, **kw); torch.cuda.synchronize()
print("PATH-B (tml-fa4 sheared, intended Blackwell path): OK", flush=True)
EOF
```

Reading the result:

| PATH-A (cute) | PATH-B (tml-fa4) | conclusion |
|---|---|---|
| dies / native error | OK | **H1 confirmed** → apply §5.1 patch, boot. |
| dies / native error | dies / native error | **H2 confirmed** → kernel-level sm_121 gap → file §7 report. |
| OK | OK | FA4 selection is *not* the boot killer → H3; run §6.2 and watch NCCL phase. |
| `sheared_bias probe = True` | — | capability is not (12,1) or image drifted; re-derive the gate from the printed value. |

("dies" here = process killed / hang / CUDA fault without a clean Python exception — the same
silent signature as the server boot. A clean Python exception with traceback is *also* diagnostic:
it means the path is unusable on sm_121 but fails loudly, which would shift weight toward H3 for
the server crash.)

### 6.2 If §6.1 is inconclusive — instrumented boot with dummy weights

`--load-format dummy` removes the 159 GB load entirely, so anything that still crashes is by
definition in the post-load phases (this also decouples "28 s" from load time):

```bash
LAMPORT_RS_SCONV=0 VLLM_LOGGING_LEVEL=DEBUG CUDA_LAUNCH_BLOCKING=1 \
TORCH_SHOW_CPP_STACKTRACES=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
vllm serve thinkingmachines/Inkling-Small-NVFP4 \
  --load-format dummy --enforce-eager \
  --tensor-parallel-size 2 --nnodes 2 <usual mp/RoCE args> \
  --max-num-batched-tokens 512 --max-num-seqs 2 \
  2>&1 | tee /tmp/inkling-boot-dummy.log
```

- `CUDA_LAUNCH_BLOCKING=1` makes the faulting kernel the *last* DEBUG line (no async lag).
- Crash during jit-warmup compile lines → H1/H2. Crash on first RS/AG after profiling starts → H3
  (then retry once with `NCCL_WIN_ENABLE=0` and once with `NCCL_BUFFSIZE` default + `NCCL_P2P_DISABLE=1`
  to isolate NCCL's VMM window registration on GB10 unified memory).
- If the dummy-weight boot **succeeds**, re-run with real weights: a real-weights-only failure in
  the same window points at NVFP4 post-load processing (§1.3a), not the model ops.

---

## 7. If kernel-level (H2): upstream bug report text to file

> **Title:** [Inkling][GB10/sm_121] worker dies silently ~30 s after weight load — FA4
> rel-attention backend mis-selected on sm_121 (`_use_sheared_bias` omits major 12)
>
> **Environment:** `vllm/vllm-openai:v0.28.0` (Inkling day-0), 2× NVIDIA DGX Spark (GB10,
> aarch64, sm_121, 121.7 GB unified), driver <fill from `nvidia-smi`>, TP=2 over RoCE (mp backend,
> pynccl), `LAMPORT_RS_SCONV=0`.
>
> **Model:** `thinkingmachines/Inkling-Small-NVFP4` (159 GB, 41-layer MoE hybrid, short-conv +
> rel-attention).
>
> **Symptom:** EngineCore dies ~28 s after the last layer (41/41) finishes loading, on all
> workers, with no Python traceback — "WorkerProc initialization failed due to an exception in a
> background process"; NVRM `NV_ERR_NO_MEMORY` in dmesg at the same instant. A manual torch probe
> allocates 115 GB on the worker cleanly (not a capacity issue). The same image on x86 H100
> single-node TP=4 boots and serves fine.
>
> **Root cause (codepath):** `vllm/models/inkling/<nvidia>/ops/fa4_rel_attention.py`,
> `_use_sheared_bias()`: `capability.major in (10, 11)` returns False on sm_121, so a
> Blackwell-family GPU is routed to the Hopper score-mod path
> (`vllm.vllm_flash_attn.cute.flash_attn_varlen_func`), contrary to the documented split ("Hopper
> uses standard FA4's score-mod gather. Blackwell uses tml-fa4's sheared relative-bias layout").
> Probing both paths standalone on GB10 shows: PATH-A (cute score-mod): <result>; PATH-B
> (tml-fa4 sheared): <result>.
>
> **Suggested fix:** `capability.major >= 10` (Blackwell family), plus an env escape hatch
> (`INKLING_FA4_SHEARED_BIAS`). If PATH-B also fails on sm_121, this becomes a CuTeDSL/tml-fa4
> sm_121 (sm_120-family) support request: the Inkling rel-attention kernel has no runnable backend
> on GB10.
>
> **Repro (no server needed):** <§6.1 script, verbatim>.
>
> **Elimination table (each a controlled single-variable boot):** gpu_memory_utilization
> 0.835/0.89/0.90; max_model_len 262144/131072/32768; max_num_seqs 8/4; max_num_batched_tokens
> 8192/2048; page caches; stale containers; full node reboots (117 GiB free); NCCL HCA fix;
> NCCL_IB_DISABLE=1; --enforce-eager; MTP on/off; VLLM_USE_V2_MODEL_RUNNER on/off; stock model.py.
> All die identically ~28 s after layer 41/41.

---

## 8. Summary

- The inkling NVIDIA tree contains exactly one arch-mis-selection reachable on GB10:
  `fa4_rel_attention.py:33` (`major in (10, 11)` excludes sm_121). Everything else custom is
  arch-generic Triton; the gfx950/ROCm files are dead code on this platform; the Lamport
  kill-switch provably removes all MNNVL/fabric assumptions and the NCCL fallback is topologically
  correct for cross-node TP=2 (`model.py:115-123`, `lamport.py:766-791`).
- H1 (mis-selected FA4 path is fatal on sm_121 at AOT-compile or first real launch, both in the
  post-load window) is the only hypothesis consistent with *all* eliminations, the H100 control,
  the ~28 s delay, and the no-traceback death.
- Patch (§5.1) is one gate plus an env override that doubles as the A/B test.
- The §6.1 standalone probe discriminates H1/H2/H3 in one short run without a server; §6.2 with
  `--load-format dummy` isolates the post-load phases if the probe is inconclusive.

## 8. CONFIRMED root cause (2026-09-03, live probes on the DGX)

The §6 probes resolved H1 and then uncovered the full chain — the boot can
fail on EITHER of two interface-level blockers, and it hits both depending
on the gate:

1. **Gate mis-selection (confirmed):** device reports `(12, 1)` and
   `_use_sheared_bias()` returns False → sm_121 is routed to the Hopper
   cute score-mod path. Standalone probe of that path:
   `AssertionError: Paged KV not supported on SM 12.0 in this PR`
   (vllm_flash_attn/cute/interface.py:1101). Inkling uses paged KV → the
   selected path is guaranteed to fail. In the server this fires inside the
   JIT kernel-warmup busy-loop, which is why the engine dies ~28 s after
   layer 41/41 with no traceback.
2. **The intended path is also blocked (confirmed with dummy-weights boot
   + instrumented tml_fa4):** with the gate patched (`major >= 10`), warmup
   compiles much further — KV cache sizing completes (27.56 GiB, 257k
   tokens) — then dies at `tml_fa4/interface.py:673 assert tile_n == 128`
   in the rel_bias metadata. Instrumented values:
   `tile_m=128 tile_n=64 rel_extent=1024`. Inkling is a diff-headdim model
   (head_dim 128, v_head_dim 64, rel_extent 1024); tml_fa4's SplitKV
   heuristic shrinks tile_n to 64 for smem reasons, and the rel_bias path
   hard-requires 128×128 tiles. So the sheared path is incompatible with
   this model's shape in this image — on this arch, and likely any arch
   where the shrink fires.

**Net: neither FA4 backend in vllm/vllm-openai:v0.28.0 can run
Inkling-Small on GB10.** This is not configurable away — every one of the
15 eliminated knobs was correctly ruled out. The fix is upstream:
(a) cute gains SM 12 paged-KV support (the code literally says "in this
PR" — pending), or (b) tml_fa4's rel_bias metadata learns tile_n=64 for
diff-headdim shapes, or (c) vLLM routes inkling to a non-FA4 fallback for
the rel-bias attention (would need a numerics-verified custom path).

Evidence trail: standalone gate probe (device (12,1), gate False) →
PATH-A assert (paged KV SM12) → gate patch → dummy boot reaches KV sizing
→ PATH-B assert (tile_n==128, instrumented tile_n=64, rel_extent=1024).
All probe scripts and the patched fa4_rel_attention.py (with
INKLING_FA4_SHEARED_BIAS override) are in ~/dspark-inkling/files/ on
spark-4687.
