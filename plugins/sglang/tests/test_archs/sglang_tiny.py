"""Tiny random-weight GLM-5.3-Flash and Qwen3.8-27B models built from SGLang's
REAL model classes, CPU only, in whichever SGLang this interpreter imports.

The served class is constructed by its own __init__ from the checkpoint's
config.json (WEIGHTLESS_TEST_GLM_CONFIG, WEIGHTLESS_TEST_QWEN_CONFIG) with the
width shrunk (hidden_size), on the meta device, under SGLang's dummy runtime
context (override_server_args) and one-rank parallel fields, or the given
pipeline rank. Then, per decoder layer, the attention and the MLP/MoE are
replaced by small deterministic CPU stand-ins (their real kernels need a GPU
and a KV cache), every other parameter of the backbone is materialised on the
CPU from a generator seeded by its name, and the backbone's own forward runs:
the real decoder-layer forward, the tree's real residual plumbing (the layer
communicator, or the stage boundaries and the forward batch's residual
stream), the real mHC torch math, the real norms, the real model loop and the
real pipeline hand-over. Everything generated depends only on the layer id,
the parameter name and the seed, so two SGLang trees hold the same weights.

``WHY`` is None when this SGLang can build both, else the reason.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import types
import zlib
from unittest import mock

os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "0")

import torch
from torch import nn

try:
    import sglang.srt.runtime_context as rc
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.configs.glm5_next import Glm5NextConfig
    from sglang.srt.configs.qwen3_5 import Qwen3_5Config
    from sglang.srt.models import glm5_next as G
    from sglang.srt.models import qwen3_5 as Q
    WHY = None if hasattr(rc.get_context(), "override_server_args") else \
        "this SGLang's runtime context has no override_server_args"
except Exception as e:  # no SGLang, or one without these models
    WHY = f"{type(e).__name__}: {e}"

NEW_LAYOUT = False
try:
    import sglang.srt.layers.layer_boundary.residual.batch as _rb  # noqa: F401
    NEW_LAYOUT = True
except Exception:
    pass


def config_path(arch):
    """The checkpoint config.json the tests build from, or None."""
    p = os.environ.get("WEIGHTLESS_TEST_GLM_CONFIG" if arch == "glm" else
                       "WEIGHTLESS_TEST_QWEN_CONFIG", "").strip()
    if p and os.path.isdir(p):
        p = os.path.join(p, "config.json")
    return p if p and os.path.isfile(p) else None


class OneRankGroup:
    """A process group of ``size`` ranks seen from rank ``rank`` (one process:
    collectives are never called on it with more than one rank here)."""
    device_group = cpu_group = None

    def __init__(self, rank=0, size=1):
        self.world_size, self.rank_in_group, self.rank = size, rank, rank
        self.ranks = list(range(size))
        self.first_rank, self.last_rank = 0, size - 1
        self.is_first_rank, self.is_last_rank = rank == 0, rank == size - 1

    def all_reduce(self, x):
        return x


class _Rope(nn.Identity):
    def __init__(self, *a, **k):
        super().__init__()
        self.head_size = k.get("head_size", a[0] if a else 0)
        self.rotary_dim = k.get("rotary_dim", a[1] if len(a) > 1 else self.head_size)


@contextlib.contextmanager
def _meta():
    with contextlib.ExitStack() as st:
        for m in list(sys.modules.values()):
            if getattr(m, "__name__", "").startswith("sglang"):
                for n in ("get_rope", "get_rope_wrapper"):
                    if callable(getattr(m, n, None)):
                        st.enter_context(mock.patch.object(m, n, _Rope))
        st.enter_context(mock.patch.object(torch.cuda, "current_device", lambda: "meta"))
        st.enter_context(mock.patch.object(torch.cuda, "Stream", lambda *a, **k: None))
        st.enter_context(torch.device("meta"))
        yield


class Context:
    """The dummy runtime context and parallel fields, entered for the build and
    for every forward (the stage boundaries read them at run time)."""

    def __init__(self, pp_rank=0, pp_size=1):
        self.pp_rank, self.pp_size = pp_rank, pp_size
        self.sa = dict(tp_size=1, pp_size=pp_size)
        if "boundary_reduction" in getattr(ServerArgs, "__struct_fields__", ()):
            self.sa["boundary_reduction"] = "ar"
        self.ov = {"pp_rank": pp_rank, "pp_size": pp_size, "tp_rank": 0, "tp_size": 1,
                   "pp_group": OneRankGroup(pp_rank, pp_size)}
        self._stack = None

    def _fields(self):
        f = rc._parallel_fields()
        return {k: v for k, v in self.ov.items() if k in f}

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(rc.get_context().override_server_args(**self.sa))
        self._stack.enter_context(rc.get_parallel().override(**self._fields()))
        return self

    def __exit__(self, *e):
        self._stack.close()

    def learn(self, err):
        m = (re.search(r"parallel name '(\w+)' has not been written", str(err))
             or re.search(r"derived parallel width '(\w+)'", str(err)))
        if not m:
            return False
        n = m.group(1)
        self.ov[n] = OneRankGroup() if n.endswith("group") else 0 if n.endswith("rank") else 1
        return True


def _seeded(name, seed):
    return torch.Generator().manual_seed((zlib.crc32(name.encode()) + 7919 * seed) % (2 ** 31))


class StandInAttn(nn.Module):
    """The attention (DSA/MLA, KDA, GDN or full attention) replaced by a
    deterministic token-wise map that depends only on the layer id."""

    def __init__(self, layer_id, hidden, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(1000 + 31 * layer_id + seed)
        self.register_buffer("w", 0.5 + torch.rand(hidden, generator=g), persistent=False)
        self.register_buffer("b", 0.1 * torch.randn(hidden, generator=g), persistent=False)

    def forward(self, *args, **kw):
        x = kw["hidden_states"] if "hidden_states" in kw else args[0]
        return torch.tanh(x * self.w.to(x.dtype) + self.b.to(x.dtype))


class StandInMLP(nn.Module):
    """The dense MLP or the MoE replaced by a deterministic token-wise map."""

    def __init__(self, layer_id, hidden, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(2000 + 31 * layer_id + seed)
        self.register_buffer("w", torch.randn(hidden, generator=g), persistent=False)

    def forward(self, x, *a, **k):
        return torch.nn.functional.silu(x) * self.w.to(x.dtype)


def _fill(module, seed):
    """Every parameter of ``module`` on the CPU, from its name: norms near 1
    (Gemma norms near 0), the mHC parameters small."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            g = _seeded(name, seed)
            v = torch.randn(p.shape, generator=g, dtype=torch.float32)
            leaf = name.rsplit(".", 1)[-1]
            if "hc_" in name and name.endswith("_scale"):
                v = 1.0 + 0.1 * v
            elif "hc_" in name and name.endswith("_fn"):
                v = 0.05 * v
            elif "hc_" in name and name.endswith("_base"):
                v = 0.1 * v
            elif leaf == "weight" and "norm" in name:
                v = 0.1 * v + (0.0 if getattr(module, "_gemma_norms", False) else 1.0)
            else:
                v = 0.02 * v
            p.copy_(v.to(p.dtype))


def _attn_name(layer):
    """The attribute the layer's forward calls for its attention: a module
    (GLM ``self_attn``, Qwen GDN ``linear_attn``) or, on Qwen's full-attention
    layer, the ``self_attention`` method of the layer itself."""
    for n in ("self_attn", "linear_attn"):
        if isinstance(getattr(layer, n, None), nn.Module):
            return n
    if callable(getattr(type(layer), "self_attention", None)):
        return "self_attention"
    raise RuntimeError(f"no attention on {type(layer).__name__}")


def _set_attn(layer, name, mod):
    if name == "self_attention":  # a method: an instance attribute shadows it
        layer.add_module("_standin_attention", mod)
        object.__setattr__(layer, "self_attention", mod)
    else:
        setattr(layer, name, mod)


def _materialise(model, bb, hidden, dtype, seed, gemma):
    names = {}
    for i in range(bb.start_layer, bb.end_layer):
        layer = bb.layers[i]
        names[i] = _attn_name(layer)
        # drop the real attention and MLP/MoE (meta) before anything is materialised
        if names[i] != "self_attention":
            setattr(layer, names[i], nn.Identity())
        layer.mlp = nn.Identity()
    for n in ("embed_tokens",):
        if isinstance(getattr(bb, n, None), nn.Module):
            setattr(bb, n, nn.Identity())
    bb.to_empty(device="cpu")
    bb._gemma_norms = gemma
    _fill(bb, seed)
    # The hc_* parameters stay fp32, as SGLang makes them; everything else in dtype.
    for name, p in bb.named_parameters():
        if "hc_" not in name:
            p.data = p.data.to(dtype)
    for i in range(bb.start_layer, bb.end_layer):
        layer = bb.layers[i]
        _set_attn(layer, names[i], StandInAttn(i, hidden, seed).to(dtype))
        layer.mlp = StandInMLP(i, hidden, seed).to(dtype)
    return model


def _shrunk_glm_config(cfg_path, hidden):
    raw = json.load(open(cfg_path))
    raw.pop("quantization_config", None)
    t = dict(raw["text_config"])
    t["hidden_size"] = hidden
    raw["text_config"] = t
    return Glm5NextConfig(**{k: v for k, v in raw.items() if k != "architectures"})


def _shrunk_qwen_config(cfg_path, hidden):
    raw = json.load(open(cfg_path))
    raw.pop("quantization_config", None)
    t = dict(raw["text_config"])
    t["hidden_size"] = hidden
    raw["text_config"] = t
    return Qwen3_5Config(**{k: v for k, v in raw.items() if k != "architectures"})


def build(arch, cfg_path, *, hidden=64, dtype=torch.float32, seed=0, ctx=None):
    """(model, backbone, context) for arch "glm" or "qwen"."""
    ctx = ctx or Context()
    for _ in range(80):
        try:
            with ctx, _meta():
                if arch == "glm":
                    model = G.Glm5NextForConditionalGeneration(_shrunk_glm_config(cfg_path, hidden))
                else:
                    model = Q.Qwen3_5ForConditionalGeneration(_shrunk_qwen_config(cfg_path, hidden))
            break
        except RuntimeError as e:
            if not ctx.learn(e):
                raise
    else:
        raise RuntimeError(f"too many parallel fields: {sorted(ctx.ov)}")
    bb = model.model
    _materialise(model, bb, hidden, dtype, seed, gemma=(arch == "qwen"))
    return model, bb, ctx


def forward_batch(tokens):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    fb = types.SimpleNamespace(
        can_run_tbo=False, forward_mode=ForwardMode.EXTEND, residual_stream=None,
        input_ids=torch.zeros(tokens, dtype=torch.long), dp_padding_mode=None, residual=None,
        positions=None,
        global_num_tokens_gpu=None, global_num_tokens_cpu=None, attn_cp_metadata=None)
    return fb


def run_backbone(arch, bb, ctx, x=None, proxy=None):
    """One backbone forward on input embeddings ``x`` (first rank) or a
    pipeline hand-over ``proxy`` (later ranks), inside ``ctx``."""
    fb = forward_batch(int((x if x is not None else proxy["hidden_states"]).shape[0]))
    with ctx, torch.no_grad():
        if arch == "glm":
            return bb(None, None, fb, input_embeds=x, pp_proxy_tensors=proxy)
        return bb(None, None, fb, input_embeds=x, pp_proxy_tensors=proxy)


def runner_of(model, hf_model_type, dtype, *, ranks=None, ctx=None):
    """A ModelRunner stand-in. ``ranks`` = (tp_rank, tp_size, pp_rank, pp_size)
    as attributes, the way the older ModelRunner carries them; None: no rank
    attributes, the way the newer one has none (the plugin reads
    get_parallel())."""
    run = types.SimpleNamespace(
        model=model, is_draft_worker=False,
        server_args=types.SimpleNamespace(enable_two_batch_overlap=False),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(model_type=hf_model_type), dtype=dtype))
    if ranks is not None:
        run.tp_rank, run.tp_size, run.pp_rank, run.pp_size = ranks
    return run


def capture_streams(arch, bb, store):
    """Forward hooks that record each layer's post-layer stream into
    ``store[layer_id]``: GLM, the written mHC stream (the layer's first
    output); Qwen, the pair (output, residual) from the layer's tuple (older
    layout) or from the forward batch's residual stream (stage boundaries).
    Register them AFTER the steering is installed so they see its edit."""
    handles = []
    for i in range(bb.start_layer, bb.end_layer):
        layer = bb.layers[i]
        if arch == "glm":
            h = layer.register_forward_hook(
                lambda m, a, o, _i=i: store.__setitem__(_i, (o[0].detach().clone(), None)))
        elif hasattr(layer, "ffn_boundary"):
            def hook(m, a, kw, o, _i=i):
                fb = kw.get("forward_batch")
                st = fb.residual_stream
                x = o if isinstance(o, torch.Tensor) else st.pending.value
                store[_i] = (x.detach().clone(), st.residual.detach().clone())
            h = layer.register_forward_hook(hook, with_kwargs=True)
        else:
            h = layer.register_forward_hook(
                lambda m, a, o, _i=i: store.__setitem__(
                    _i, (o[0].detach().clone(), o[1].detach().clone())))
        handles.append(h)
    return handles
