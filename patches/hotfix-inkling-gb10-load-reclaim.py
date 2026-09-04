#!/usr/bin/env python3
"""Hotfix Inkling weight loading for GB10 unified-memory systems.

vLLM 0.28.0 constructs the TP-local Inkling model on CUDA before iterating the
checkpoint.  On a 121.7 GiB GB10 this leaves about 30 GiB available.  The lazy
safetensors iterator keeps the current shard mapped while Inkling fills the
preallocated parameters, and CUDA copies leave temporary allocator blocks
cached.  The combined file-backed and CUDA staging working set trips the
8 GiB safety watchdog even though the completed model fits.

This patch adds an env-gated reclaim point after each sufficiently large
checkpoint tensor has been consumed:

* synchronize the copy before releasing its source pages;
* MADV_DONTNEED the consumed CPU tensor range;
* collect Python references and release unused CUDA allocator blocks;
* optionally pause briefly so kernel reclaim is not outrun by the next fill.

The target is vllm/models/inkling/nvidia/model.py.  Anchors are exact and the
patch fails closed on source drift.  Reapplication is a no-op.

Environment inside the server:

  INKLING_GB10_LOAD_RECLAIM=1       enable the guard (default off)
  INKLING_LOAD_RECLAIM_MIN_MIB=64   minimum source tensor size
  INKLING_LOAD_RECLAIM_SLEEP_MS=20  pause after a reclaim point

For offline staging set WEIGHTLESS_INKLING_MODEL_PY to the copied model.py.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


P = Path(
    os.environ.get(
        "WEIGHTLESS_INKLING_MODEL_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/models/inkling/nvidia/model.py",
    )
)
MARK = "# [gb10-load-reclaim-hotfix]"

IMPORT_ANCHOR = """from collections.abc import Iterable
from typing import Any, TypeAlias
"""
IMPORT_REPLACEMENT = """from collections.abc import Iterable
from typing import Any, TypeAlias

import ctypes
import gc
import logging
import os
import time
"""

HELPER_ANCHOR = """_MOE_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<mlp>.*\\.mlp)\\.(?P<rest>(?:shared_)?experts\\..+)$"
)


def _load_inkling_weights(
"""
HELPER_REPLACEMENT = """_MOE_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<mlp>.*\\.mlp)\\.(?P<rest>(?:shared_)?experts\\..+)$"
)


""" + MARK + """ Reclaim consumed checkpoint staging on GB10 unified memory.
_GB10_LOAD_LOGGER = logging.getLogger(__name__)
_GB10_LOAD_RECLAIM = os.environ.get(
    "INKLING_GB10_LOAD_RECLAIM", "0"
).strip().lower() in ("1", "true", "yes", "on")
_GB10_LOAD_RECLAIM_MIN_BYTES = int(
    float(os.environ.get("INKLING_LOAD_RECLAIM_MIN_MIB", "64")) * 1024 * 1024
)
_GB10_LOAD_RECLAIM_SLEEP_S = max(
    0.0, float(os.environ.get("INKLING_LOAD_RECLAIM_SLEEP_MS", "20")) / 1000.0
)


def _gb10_mem_available_mib() -> int:
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _gb10_reclaim_consumed_weight(name: str, weight: torch.Tensor) -> None:
    if not _GB10_LOAD_RECLAIM:
        return
    nbytes = weight.numel() * weight.element_size()
    if nbytes < _GB10_LOAD_RECLAIM_MIN_BYTES:
        return

    # All stock Inkling loaders use synchronous copies, but explicitly wait
    # before invalidating the mmap range so this stays correct if a loader
    # starts using non_blocking=True later.
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    madvise_rc = -1
    madvise_errno = 0
    if weight.device.type == "cpu" and weight.is_contiguous() and nbytes:
        page = os.sysconf("SC_PAGE_SIZE")
        addr = weight.data_ptr()
        start = addr - (addr % page)
        length = ((addr + nbytes - start + page - 1) // page) * page
        libc = ctypes.CDLL(None, use_errno=True)
        madvise_rc = libc.madvise(
            ctypes.c_void_p(start), ctypes.c_size_t(length), ctypes.c_int(4)
        )
        if madvise_rc != 0:
            madvise_errno = ctypes.get_errno()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() // (1024 * 1024)
        reserved = torch.cuda.memory_reserved() // (1024 * 1024)
    else:
        allocated = reserved = 0
    _GB10_LOAD_LOGGER.info(
        "Inkling GB10 load reclaim: %s source=%d MiB mem_available=%d MiB "
        "cuda_allocated=%d MiB cuda_reserved=%d MiB madvise_rc=%d errno=%d",
        name,
        nbytes // (1024 * 1024),
        _gb10_mem_available_mib(),
        allocated,
        reserved,
        madvise_rc,
        madvise_errno,
    )
    if _GB10_LOAD_RECLAIM_SLEEP_S:
        time.sleep(_GB10_LOAD_RECLAIM_SLEEP_S)


def _load_inkling_weights(
"""

MOE_ANCHOR = """                for rel in moe.load_expert_weight(moe_match.group("rest"), weight):
                    loaded.add(f"{moe_match.group('mlp')}.{rel}")
                continue

            yield name, weight
"""
MOE_REPLACEMENT = """                for rel in moe.load_expert_weight(moe_match.group("rest"), weight):
                    loaded.add(f"{moe_match.group('mlp')}.{rel}")
                _gb10_reclaim_consumed_weight(name, weight)
                del weight
                continue

            yield name, weight
            _gb10_reclaim_consumed_weight(name, weight)
            del weight
"""


def fail(message: str) -> None:
    print(f"hotfix-inkling-gb10-load-reclaim: {message}", file=sys.stderr)
    raise SystemExit(1)


if not P.is_file():
    fail(f"target not found: {P}")

source = P.read_text()
if "--status" in sys.argv:
    print("patched" if MARK in source else "stock")
    raise SystemExit(0)

if MARK in source:
    print(f"already patched: {P}")
    raise SystemExit(0)

for label, anchor in (
    ("import", IMPORT_ANCHOR),
    ("helper", HELPER_ANCHOR),
    ("MoE/iterator", MOE_ANCHOR),
):
    count = source.count(anchor)
    if count != 1:
        fail(f"{label} anchor count is {count}, expected 1; refusing source drift")

patched = source.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
patched = patched.replace(HELPER_ANCHOR, HELPER_REPLACEMENT, 1)
patched = patched.replace(MOE_ANCHOR, MOE_REPLACEMENT, 1)

if MARK not in patched:
    fail("internal error: marker absent after rewrite")

P.write_text(patched)
print(f"patched: {P}")
