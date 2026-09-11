"""Synthetic GLP GGUF writer shared by the offline tests.

Mirrors tests/test_nemotron35_hotfix.py's write_gguf: string metadata and
1-D F32 tensors, exactly the subset of GGUF v3 the container reader parses.
"""
import struct

import numpy as np


def write_gguf(path, meta: dict, tensors: dict) -> None:
    """Write a minimal GGUF v3 file.

    tensors: {name: (numpy float32 array, ggml_type)} — ggml_type 0 is F32.
    """
    out = bytearray()

    def w_str(s):
        b = s.encode()
        return struct.pack("<Q", len(b)) + b

    out += b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        out += w_str(k) + struct.pack("<I", 8) + w_str(v)  # string metadata
    blobs = []
    off = 0
    for name, (arr, gtype) in tensors.items():
        data = np.asarray(arr, dtype="<f4").tobytes()
        out += w_str(name)
        out += struct.pack("<I", 1) + struct.pack("<Q", arr.size)
        out += struct.pack("<I", gtype) + struct.pack("<Q", off)
        blobs.append(data)
        off += (len(data) + 31) // 32 * 32
    out += b"\0" * ((-len(out)) % 32)  # data base alignment
    for data in blobs:
        out += data
        out += b"\0" * ((-len(data)) % 32)
    with open(path, "wb") as f:
        f.write(bytes(out))


def good_meta(layers=(1, 2, 3), alpha="2.5"):
    return {
        "glp.mode": "project",
        "glp.hook_point": "residual_stream_post_layer",
        "glp.derived_at": "residual_stream_post_layer",
        "glp.spec_version": "1.0",
        "glp.alpha_default": alpha,
        "glp.layer_ids_zero_based": ",".join(str(i) for i in layers),
    }


def good_tensors(layers=(1, 2, 3), width=8):
    rng = np.random.default_rng(0)
    return {
        f"direction.{i}": (rng.standard_normal(width).astype(np.float32), 0)
        for i in layers
    }
