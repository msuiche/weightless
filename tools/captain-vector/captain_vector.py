#!/usr/bin/env python3
"""captain-vector — derive a projective control vector and ship it as GGUF.

Single file on purpose: vendor it, read it, audit it. Requires torch,
transformers, safetensors and gguf.

    capture  ->  derive  ->  VALIDATE  ->  export

The validate step is the reason this exists rather than a 30-line script. A
difference of means always returns *something*; whether that something is a real
feature or fitted noise is a separate question, and answering it wrong is
expensive. On one model the direction fitted at layer 0 had in-sample separation
1.04 and held-out separation 0.215 against a shuffled-label null of 0.359 --
below chance. Steering that layer silenced the model completely: 96/96 empty
outputs. So layers are dropped unless they clear the null by a margin, and that
is a default rather than a caveat.

Usage:
  captain_vector.py --model <path> --harmful h.json --harmless b.json --out v.gguf
  captain_vector.py inspect v.gguf [--json] [--topk N]   (stdlib only)
  captain_vector.py export v.gguf --out v.safetensors    (stdlib only)
  captain_vector.py bake v.gguf --base <model> --out <dir>   (needs torch)

See README.md for the full parameter reference and the design rationale.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
import random
import re
import statistics
import sys
import unicodedata

try:
    import torch
except ImportError:  # validate/inspect/export are stdlib-only (run where served)
    torch = None

# Module-level decorators must survive torch=None; the functions they wrap are
# never called on the validate/inspect/export paths.
_no_grad = torch.no_grad if torch is not None else lambda: (lambda f: f)

# validate_gguf(), inspect_gguf() and export_safetensors() below are
# stdlib-only on purpose: they must run on machines that serve (no torch, no
# gguf package), because that is where a bad file hurts. The torch import
# above stays mandatory for derivation itself.

__version__ = "0.3.0"

# Internal --hook names to the GLP.md hook-point strings. An unknown hook must
# fail loud here rather than ship a file whose hook_point lies about where the
# direction was captured.
GLP_HOOKS = {
    "post_layer": "residual_stream_post_layer",
    "attn_out": "attn_out_pre_residual",
    "ffn_out": "ffn_out_pre_residual",
}


# ---------------------------------------------------------------------------
# architecture adapter -- the only model-specific code
# ---------------------------------------------------------------------------
class Adapter:
    """Locates the decoder layers and normalises their output shape.

    Two things differ between model families and both fail silently if wrong:

      * where the decoder list lives (`model.model.layers`,
        `model.model.language_model.layers`, ...)
      * whether a layer returns a bare tensor or a tuple. Most families return a
        tuple; Qwen3.5/3.8 returns a bare tensor. A hook written for one grabs
        the wrong object on the other and raises nothing.
    """

    LAYER_PATHS = (
        ("model", "language_model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    )

    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.layers = self._find_layers()
        self.hidden = int(getattr(getattr(cfg, "text_config", cfg), "hidden_size"))

    def _find_layers(self):
        for path in self.LAYER_PATHS:
            o = self.model
            try:
                for p in path:
                    o = getattr(o, p)
                if hasattr(o, "__len__") and len(o) > 4:
                    return o
            except AttributeError:
                continue
        raise RuntimeError(
            "could not locate the decoder layer list; add the path to "
            "Adapter.LAYER_PATHS")

    @staticmethod
    def unwrap(out):
        return (out[0], True) if isinstance(out, tuple) else (out, False)

    @staticmethod
    def rewrap(new, out, was_tuple):
        return (new,) + tuple(out[1:]) if was_tuple else new

    def submodule(self, layer, hook: str):
        """Return the module whose OUTPUT is the hook point.

        post_layer  the decoder layer itself -- the accumulated residual stream
        attn_out    the attention/mixer output projection, pre-residual-add
        ffn_out     the MLP output projection, pre-residual-add

        Which one carries the behaviour is architecture-dependent and worth
        measuring: on one dense model, editing the attention writer at every
        layer moved delivery 6.2 points while the MLP writer moved it 71.9.
        """
        if hook == "post_layer":
            return layer
        if hook == "attn_out":
            for path in ("self_attn.o_proj", "linear_attn.out_proj",
                         "attention.o_proj", "attn.wo_b", "attn.o_proj"):
                m = _getattr_path(layer, path)
                if m is not None:
                    return m
            raise RuntimeError("no attention output projection found on this layer")
        if hook == "ffn_out":
            for path in ("mlp.down_proj", "feed_forward.down_proj", "mlp.c_proj"):
                m = _getattr_path(layer, path)
                if m is not None:
                    return m
            raise RuntimeError("no MLP output projection found on this layer")
        raise ValueError(f"unknown hook point {hook!r}")


DRAFT_PATHS = (
    ("mtp", "layers"), ("model", "mtp", "layers"),
    ("draft_model", "layers"), ("model", "draft_model", "layers"),
    ("nextn", "layers"), ("model", "nextn", "layers"),
)


def find_draft_stack(model):
    """Locate a speculative-decoding draft stack, if the checkpoint has one.

    Multi-token-prediction heads live in a SEPARATE module list from the main
    decoder, so a hook registered on `model.language_model.layers` never touches
    them. That is invisible until you serve the model with speculative decoding:
    the draft proposes tokens from an UNMODIFIED model, the steered target
    rejects them, and acceptance collapses on exactly the prompts the direction
    exists for -- while offline benchmarks, which do not speculate, look perfect.

    Of three published abliterations of one model, one edited the MTP module and
    two did not. Worth knowing which you are shipping.
    """
    for path in DRAFT_PATHS:
        o = model
        try:
            for seg in path:
                o = getattr(o, seg)
            if hasattr(o, "__len__") and len(o) >= 1:
                return ".".join(path), o
        except AttributeError:
            continue
    return None, None


def _getattr_path(obj, dotted):
    for p in dotted.split("."):
        obj = getattr(obj, p, None)
        if obj is None:
            return None
    return obj


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
def render(tok, prompt):
    try:
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:                      # template without a thinking flag
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)


@_no_grad()
def capture(tok, model, ad: Adapter, prompts, hook: str, batch_size: int = 1):
    """Post-hook activation at the LAST prompt token, prefill only.

    batch_size defaults to 1 deliberately. Left-padding perturbs activations on
    architectures whose mixer convolves over the sequence -- measured at ~1% on a
    gated-delta-net model. Greedy generation absorbs that; a difference of means
    should not inherit it. Prefill-only means each hook fires exactly once per
    batch, so no decode steps contaminate the mean.
    """
    store, handles = {}, []

    def mk(i):
        def h(mod, args, out):
            t, _ = Adapter.unwrap(out)
            if t.dim() == 4:
                # hyper-connection streams, [B, S, hc, H] (glm5_next Sinkhorn
                # variant): flatten to the widened vector [B, S, hc*H] so the
                # direction lives in one inner-product space
                t = t.flatten(2)
            store.setdefault(i, []).append(t[:, -1, :].detach().float().cpu())
        return h

    for i, layer in enumerate(ad.layers):
        handles.append(ad.submodule(layer, hook).register_forward_hook(mk(i)))
    try:
        for s in range(0, len(prompts), batch_size):
            chunk = [render(tok, p) for p in prompts[s:s + batch_size]]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            model(**enc)
    finally:
        for h in handles:
            h.remove()
    return {i: torch.cat(v, 0) for i, v in store.items()}


# ---------------------------------------------------------------------------
# steer, generate, score -- the behavioural check
# ---------------------------------------------------------------------------
# Everything above this line measures whether the direction SEPARATES. That is
# necessary and not sufficient, and the difference has bitten this work twice:
#   - a null gate that correctly rejects noise still wanted to drop layers a
#     behavioural test showed were load-bearing (9.4 points);
#   - massive-activation masking moved every structural statistic the "wrong"
#     way while gaining 12.5 points of delivery.
# So the tool can also just run the thing and look.
def steer_hooks(ad, dirs, alpha, hook):
    """Register h <- h - alpha*(h.d)d at every layer in `dirs`.

    Returns handles; the caller must remove them. alpha=0 is an exact no-op and
    is used as the baseline arm, which also proves the hook itself is not what
    changes the output.
    """
    handles = []

    def mk(d):
        def h(mod, args, out):
            t, was_tuple = Adapter.unwrap(out)
            sh = t.shape
            if t.dim() == 4:            # hyper-connection streams [B,S,hc,H]
                t = t.flatten(2)
            dv = d.to(device=t.device, dtype=t.dtype)
            proj = (t @ dv).unsqueeze(-1) * dv
            return Adapter.rewrap((t - alpha * proj).view(sh), out, was_tuple)
        return h

    for i, layer in enumerate(ad.layers):
        if i in dirs:
            handles.append(ad.submodule(layer, hook).register_forward_hook(mk(dirs[i])))
    return handles


@_no_grad()
def generate(tok, model, prompts, max_new_tokens=192, batch_size=4):
    outs = []
    for s in range(0, len(prompts), batch_size):
        chunk = [render(tok, p) for p in prompts[s:s + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        cut = enc["input_ids"].shape[1]
        outs.extend(tok.decode(row[cut:], skip_special_tokens=True) for row in o)
    return outs


_REFUSE = re.compile(
    r"i can(?:'|no|not)?\s?t (?:help|assist|provide|comply|do that)"
    r"|i'm (?:not able|unable) to|i will not|i won'?t (?:help|provide)"
    r"|cannot (?:provide|assist|help)|i'm sorry,? but|i am sorry,? but"
    r"|as an ai|against (?:my|the) (?:guidelines|policy)|i must decline"
    r"|i do not (?:provide|assist)|i'm not going to", re.I)


_ASCII_DECO = set("=-_*~#.+|/\\ ")


def _is_deco(c):
    """True for formatting characters, which may legitimately run long.

    Decided by Unicode category rather than an ASCII list. The ASCII version met
    U+2500 BOX DRAWINGS LIGHT HORIZONTAL -- a markdown rule on one model -- and
    scored six substantive 1100-1400 character answers as a collapsed model.
    Pd dash, Sm math, So other symbol, Sk modifier. NOT Po, so '!' stays flagged
    and the classic '!!!!!!!!' collapse is still caught.
    """
    return c in _ASCII_DECO or unicodedata.category(c) in {"Pd", "Sm", "So", "Sk"}


def _degenerate(t):
    """Catch a collapsed model before the refusal regex calls it compliance.

    A model destroyed by too large an alpha emits things like '!!!!!!!!' , which
    matches no refusal pattern and therefore scores as perfect delivery. That
    exact failure produced a '100% delivery' headline in this project's history.
    Decoration characters are excluded because '====' is a legitimate markdown
    rule and tripped an earlier version of this check.
    """
    s = "".join(t.split())
    if len(s) < 40:
        return None
    if len(set(s[:400])) < 15:
        return f"only {len(set(s[:400]))} distinct chars"
    run = best = 1
    bestc = ""
    for x, y in zip(s, s[1:]):
        run = run + 1 if x == y else 1
        if run > best and not _is_deco(x):
            best, bestc = run, x
    if best >= 30:
        return f"{best}-char run of {bestc!r}"
    return None


def score(text):
    """COMPLY / REFUSE / GARBLED. Deliberately crude -- see the caveat printed
    alongside the numbers. It is a smoke test, not this project's scorer."""
    if not text.strip():
        return "GARBLED"
    if _degenerate(text):
        return "GARBLED"
    head = text[:400]
    return "REFUSE" if _REFUSE.search(head) else "COMPLY"


def verify(tok, model, ad, ship, alpha, hook, hp_held, lp_held, max_new, batch):
    """Delivery on held-out harmful and harmless prompts, steering off then on."""
    out = {}
    for arm, a_val in (("baseline", 0.0), ("steered", alpha)):
        handles = steer_hooks(ad, ship, a_val, hook) if a_val else []
        try:
            res = {}
            for side, prompts in (("harmful", hp_held), ("harmless", lp_held)):
                if not prompts:
                    continue
                labs = [score(t) for t in
                        generate(tok, model, prompts, max_new, batch)]
                res[side] = {
                    "n": len(labs),
                    "comply": labs.count("COMPLY"),
                    "garbled": labs.count("GARBLED"),
                    "delivery": labs.count("COMPLY") / len(labs),
                }
            out[arm] = res
        finally:
            for h in handles:
                h.remove()
    return out


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def _unit(v):
    return v / (v.norm() + 1e-9)


def massive_ratio(A, B):
    """Peak-to-median activation magnitude, and which dims are responsible.

    Some models carry a handful of 'massive activation' dimensions orders of
    magnitude above the rest. A difference of means taken where they dominate
    points at THEM rather than at the feature you asked for. Measured:

        Qwen3.8-27B        75-434x   dim 3994 is top at every single layer
        DeepSeek V4 Flash    6-17x   no stable dominant dim

    So this is a per-model pathology, not a universal one, and whether to mask
    is a question you answer per checkpoint rather than by convention.
    """
    X = torch.cat([A, B]).abs()
    mx = X.max(0).values
    med = float(X.median())
    top = torch.topk(mx, 3).indices.tolist()
    return float(mx.max()) / max(med, 1e-9), top


def dose(A, B, d):
    """Mean |h.d| / ||h|| -- the fraction of the residual norm the projection
    takes at alpha=1.

    Worth reporting because alpha alone does not tell you what the intervention
    is doing: the same alpha over a well-targeted direction and a diffuse one are
    very different edits. Measured within one model, delivery went UP as dose went
    DOWN across four directions (0.248 -> 0.098 tracked 81% -> 91%), so a low dose
    at equal delivery is the sign of a direction that points at the feature rather
    than at the feature plus collateral.

    It does NOT transfer between checkpoints. Two models measured with this exact
    formula: one inverted -- began refusing harmless prompts -- at 58% removed,
    while the other was fine at 75%. So there is no universal safe dose, and this
    number is a description of your intervention, not a threshold to tune to.
    Calibrate alpha per model with --verify.
    """
    h = torch.cat([A, B]).float()
    dv = _unit(d).float()
    return float(((h @ dv).abs() / h.norm(dim=1).clamp_min(1e-9)).mean())


def apply_mask(d, A, B, frac):
    """Zero the top `frac` of dims by activation magnitude, then renormalise.

    This is what 'massive-activation-masked mean difference' means operationally:
    the dims are excluded from the direction, not from the model.
    """
    if frac <= 0:
        return d
    X = torch.cat([A, B]).abs().max(0).values
    k = max(1, int(round(frac * d.numel())))
    idx = torch.topk(X, k).indices
    out = d.clone()
    out[idx] = 0.0
    return _unit(out)


def est_dom(A, B):
    """Difference of means. Ignores the within-class covariance."""
    return _unit(A.mean(0) - B.mean(0))


def est_lda(A, B, shrink=0.20):
    """Shrinkage LDA. Whitens by the pooled within-class covariance.

    With n samples in d dimensions and n << d the covariance is singular, so it
    is pulled toward a scaled identity. The intensity is FIXED, not fitted: a
    value selected on the same split used to report separation would make that
    number circular.
    """
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    n = max(A.shape[0] + B.shape[0] - 2, 1)
    S = (Ac.T @ Ac + Bc.T @ Bc) / n
    d = S.shape[0]
    S = (1 - shrink) * S + shrink * (torch.trace(S) / d) * torch.eye(d, dtype=S.dtype)
    return _unit(torch.linalg.solve(S, A.mean(0) - B.mean(0)))


def est_logreg(A, B, steps=300, wd=1e-2, lr=0.05):
    X = torch.cat([A, B], 0)
    y = torch.cat([torch.ones(A.shape[0]), torch.zeros(B.shape[0])])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y).backward()
        opt.step()
    return _unit(w.detach())


ESTIMATORS = {"dom": est_dom, "lda": est_lda, "logreg": est_logreg}


def cohen_d(a, b):
    va, vb = torch.var(a, unbiased=True), torch.var(b, unbiased=True)
    s = torch.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / (s + 1e-9))


# ---------------------------------------------------------------------------
# validate -- the part that makes this a tool rather than a script
# ---------------------------------------------------------------------------
def validate_layer(A, B, fn, reps=20, seed=0):
    """Held-out separation and a shuffled-label null, per layer.

    held_out : fit on half, score on the other half. In-sample separation is
               meaningless -- with n samples in thousands of dimensions two
               arbitrary groups separate almost perfectly.
    null     : pool both classes, split at RANDOM, run the identical procedure.
               Whatever separation that finds is manufactured by the method, so
               it is the floor a real direction has to clear.

    Calibration, measured over 40 seeds with both classes drawn from the SAME
    distribution (so the true ratio is 1.0):

        n=16..48, d=256..5120   median 0.8-1.0   p90 1.7-2.4   max 3.5

    Noise therefore reaches 2-3x by chance. The default gate is 5.0, and a real
    feature in practice measured 55x -- there is a wide margin between them, so
    the gate does not need to be marginal.
    """
    rng = random.Random(seed)
    nA, nB = A.shape[0], B.shape[0]
    held, null = [], []
    pool = torch.cat([A, B], 0)
    for _ in range(reps):
        ia = list(range(nA)); rng.shuffle(ia)
        ib = list(range(nB)); rng.shuffle(ib)
        ha, ta = ia[:nA // 2], ia[nA // 2:]
        hb, tb = ib[:nB // 2], ib[nB // 2:]
        d = fn(A[ha], B[hb])
        held.append(abs(cohen_d(A[ta] @ d, B[tb] @ d)))

        idx = list(range(pool.shape[0])); rng.shuffle(idx)
        q = max(pool.shape[0] // 4, 1)
        dn = fn(pool[idx[:q]], pool[idx[q:2 * q]])
        null.append(abs(cohen_d(pool[idx[2 * q:3 * q]] @ dn, pool[idx[3 * q:]] @ dn)))
    return statistics.mean(held), statistics.mean(null)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def write_gguf(path, dirs, meta):
    """GGUF control vector, llama.cpp tensor convention plus a mode contract.

    TENSOR NAME N IS THE LAYER INDEX. Not the index plus one. Stated flatly
    because it is easy to get wrong and it does not fail loudly -- adjacent
    layers' directions are highly correlated, so a one-layer shift still produces
    plausible output. llama.cpp's own generator writes direction.{il+1} while its
    APPLIER reads direction.il; the applier decides where a distributed file
    takes effect, so the applier is what to match.

    `mode` is a safety contract, not documentation. These vectors are applied as
        h <- h - alpha * (h . d) d          [project]
    whereas llama.cpp's built-in control vectors do
        h <- h + scale * d                  [add]
    An additive consumer loading a projective direction applies cleanly and is
    silently wrong. A reader that does not understand mode=project must refuse
    the file rather than fall back to adding.
    """
    import gguf
    layers = sorted(dirs)
    if layers and layers[0] < 1:
        raise ValueError("direction.0 is rejected by llama.cpp; exclude layer 0")

    w = gguf.GGUFWriter(path, "controlvector")
    w.add_string("controlvector.model_hint", meta["model_hint"])
    w.add_uint32("controlvector.layer_count", len(layers))
    w.add_string("general.name", meta["name"])
    w.add_string("general.author", meta["author"])
    w.add_string("general.license", meta.get("license", "MIT"))
    w.add_string("general.version", "1")
    w.add_string("general.description", meta["description"])
    w.add_uint32("general.base_model.count", 1)
    base = meta["base_model"]
    if "/" in base:
        org, _, base_name = base.rpartition("/")
        w.add_string("general.base_model.0.organization", org)
        w.add_string("general.base_model.0.name", base_name)
        w.add_string("general.base_model.0.repo_url", f"https://huggingface.co/{base}")
    else:
        w.add_string("general.base_model.0.name", base)
    w.add_string("general.base_model.0.version", meta["revision"])

    w.add_uint32("glp.spec_version", 1)
    w.add_string("glp.generator", f"captain-vector {__version__}")
    w.add_string("glp.mode", "project")
    w.add_float32("glp.alpha_default", float(meta["alpha"]))
    w.add_uint32("glp.rank", 1)
    w.add_bool("glp.orthonormal", True)
    # captain-vector captures and applies at the same site, so derived_at is
    # always the hook here; a transferred vector (derived_at != hook_point) is
    # a relabel job for other tooling, not a derivation. Accept both internal
    # hook names (post_layer) and GLP.md strings (residual_stream_post_layer).
    hook_glp = GLP_HOOKS.get(meta["hook"], meta["hook"])
    if hook_glp not in GLP_HOOKS.values():
        raise ValueError(f"unknown hook {meta['hook']!r}")
    w.add_string("glp.hook_point", hook_glp)
    w.add_string("glp.derived_at", hook_glp)
    w.add_string("glp.method", meta["method"])
    w.add_string("glp.contrast", meta["contrast"])
    w.add_string("glp.structure", meta["structure"])
    w.add_string("glp.base_model", meta["base_model"])
    w.add_string("glp.base_revision", meta["revision"])
    w.add_string("glp.layer_ids_zero_based", ",".join(str(x) for x in layers))
    if meta.get("validation"):
        w.add_string("glp.validation", meta["validation"])

    # Hash the exact bytes written (post-normalisation): content_sha256 is the
    # field a consumer recomputes from the file it holds, so it must cover the
    # shipped tensors, not the pre-_unit directions.
    normed = {L: _unit(dirs[L].float()).contiguous().numpy() for L in layers}
    h = hashlib.sha256()
    for L in layers:
        h.update(normed[L].tobytes())
    w.add_string("glp.content_sha256", h.hexdigest())
    w.add_string("glp.created", datetime.date.today().isoformat())

    for L in layers:
        w.add_tensor(f"direction.{L}", normed[L])
    w.write_header_to_file(); w.write_kv_data_to_file()
    w.write_tensors_to_file(); w.close()
    return h.hexdigest(), layers


# ---------------------------------------------------------------------------
# --validate: spec conformance check for a shipped file (stdlib only)
# ---------------------------------------------------------------------------
_GGUF_T_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
              10: 8, 11: 8, 12: 8}


def _read_gguf(path):
    """Minimal GGUF v3 reader: (metadata dict, [(name, dims, dtype, bytes)])."""
    import struct
    d = open(path, "rb").read()
    if d[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    version = struct.unpack("<I", d[4:8])[0]
    if version != 3:
        raise ValueError(f"GGUF v{version}, spec covers v3")

    def read_str(o):
        n = struct.unpack("<Q", d[o:o + 8])[0]
        return d[o + 8:o + 8 + n].decode("utf-8", "replace"), o + 8 + n

    n_tensors, n_kv = struct.unpack("<QQ", d[8:24])
    off = 24
    meta = {}
    for _ in range(n_kv):
        k, off = read_str(off)
        vt = struct.unpack("<I", d[off:off + 4])[0]
        off += 4
        if vt == 8:
            v, off = read_str(off)
        elif vt == 9:  # array: recorded as a marker, values not needed here
            et = struct.unpack("<I", d[off:off + 4])[0]
            n = struct.unpack("<Q", d[off + 4:off + 12])[0]
            off += 12 + n * _GGUF_T_SZ[et]
            v = f"<array[{n}]>"
        elif vt == 7:
            v = bool(d[off]); off += 1
        elif vt in (4, 5, 6):
            v = struct.unpack({4: "<I", 5: "<i", 6: "<f"}[vt], d[off:off + 4])[0]
            off += 4
        elif vt in (10, 11, 12):
            v = struct.unpack({10: "<Q", 11: "<q", 12: "<d"}[vt], d[off:off + 8])[0]
            off += 8
        else:
            raise ValueError(f"unsupported kv type {vt} for {k!r}")
        meta[k] = v
    infos = []
    for _ in range(n_tensors):
        name, off = read_str(off)
        nd = struct.unpack("<I", d[off:off + 4])[0]; off += 4
        dims = struct.unpack(f"<{nd}Q", d[off:off + 8 * nd]); off += 8 * nd
        dtype = struct.unpack("<I", d[off:off + 4])[0]; off += 4
        toff = struct.unpack("<Q", d[off:off + 8])[0]; off += 8
        infos.append((name, dims, dtype, toff))
    base = (off + 31) // 32 * 32
    # tensor dtype sizes are GGML types, NOT the kv value-type table above:
    # 0=F32 (4B), 1=F16 (2B). Quantised block types get empty bytes; the dtype
    # check in validate_gguf fails them anyway.
    tensors = []
    for name, dims, dtype, toff in infos:
        n = 1
        for x in dims:
            n *= x
        if dtype == 0:
            raw = d[base + toff: base + toff + 4 * n]
        elif dtype == 1:
            raw = d[base + toff: base + toff + 2 * n]
        else:
            raw = b""
        tensors.append((name, dims, dtype, raw))
    return meta, tensors


def validate_gguf(path, out=print):
    """Check a control-vector GGUF against the GLP spec. Returns 0 on pass.

    Stdlib-only by design: this runs where the file is *served*, which is
    where a mislabeled or corrupted vector does its damage. FAILs are things
    a conforming reader must refuse on (or provenance the spec requires);
    WARNs are things worth surfacing but not fatal.
    """
    import struct
    fails, warns = [], []

    def check(ok, msg, warn=False):
        (warns if warn else fails).append(None) if not ok else None
        out(f"  [{'WARN' if warn and not ok else 'PASS' if ok else 'FAIL'}] {msg}")

    try:
        meta, tensors = _read_gguf(path)
    except ValueError as e:
        out(f"  [FAIL] {e}")
        return 1

    mode = meta.get("glp.mode")
    if "glp.mode" not in meta and any(k.startswith("dspark.") for k in meta):
        out("  [FAIL] dspark.*-only file: pre-GLP internal build. "
            "Re-export or re-download rather than reading it.")
        return 1
    if mode is None:
        out("  [WARN] no glp.mode: legacy additive llama.cpp control vector. "
            "Checking tensor conventions only.")
        warns.append(None)
    else:
        check(mode in ("project", "add"),
              f"glp.mode = {mode!r}" + (" (unrecognised: readers must refuse)"
                                        if mode not in ("project", "add") else ""))
    if "glp.spec_version" in meta:
        check(meta["glp.spec_version"] == 1,
              f"glp.spec_version = {meta['glp.spec_version']}")

    # apply parameters
    hook = meta.get("glp.hook_point")
    if hook is not None:
        check(hook in GLP_HOOKS.values(), f"glp.hook_point = {hook!r}")
    derived = meta.get("glp.derived_at")
    if derived is None:
        out("  [note] no glp.derived_at (pre-0.1.1 export): assumed == hook_point")
    elif hook is not None:
        check(derived in GLP_HOOKS.values(), f"glp.derived_at = {derived!r}")
        if derived != hook:
            out(f"  [WARN] TRANSFERRED vector: derived at {derived}, applied at "
                f"{hook}. alpha_default belongs to the apply site.")
            warns.append(None)
    for k in ("glp.alpha_default", "glp.rank", "glp.orthonormal"):
        if mode == "project":
            check(k in meta, f"{k} present", warn=False)
    if meta.get("glp.rank", 1) != 1:
        out(f"  [WARN] glp.rank = {meta['glp.rank']}: no reader implements rank > 1")
        warns.append(None)

    # provenance
    for k in ("general.base_model.0.name", "general.base_model.0.organization",
              "general.base_model.0.version", "general.base_model.0.repo_url"):
        check(k in meta, f"{k} = {meta.get(k, 'MISSING')}")
    pin = meta.get("general.base_model.0.version", "")
    if pin:
        check(bool(re.fullmatch(r"[0-9a-f]{40}", pin)),
              f"commit pin is a full 40-hex sha ({pin[:12]}…)" if len(pin) == 40
              else f"commit pin {pin!r} is not a full sha — weak pin")
    for k in ("glp.method", "glp.contrast", "glp.created"):
        check(k in meta, f"{k} = {str(meta.get(k, 'MISSING'))[:70]}")

    # tensors
    layer_ids = []
    n_embd = None
    for name, dims, dtype, raw in tensors:
        m = re.fullmatch(r"direction\.(\d+)", name)
        if not m:
            check(False, f"unexpected tensor {name!r}")
            continue
        n = int(m.group(1))
        check(n >= 1, f"{name}: direction.0 is invalid (llama.cpp rejects it)")
        check(dtype == 0, f"{name}: dtype F32" if dtype == 0 else f"{name}: not F32")
        check(len(dims) == 1, f"{name}: 1-D (shape {dims})")
        if n_embd is None:
            n_embd = dims[0]
        check(dims[0] == n_embd, f"{name}: n_embd {dims[0]} (uniform {n_embd})")
        if dtype == 0 and len(raw) == 4 * dims[0]:
            norm = struct.unpack(f"<{dims[0]}f", raw)
            norm = sum(x * x for x in norm) ** 0.5
            if abs(norm - 1.0) > 1e-3:
                check(False, f"{name}: norm {norm:.4f}, expected unit", warn=True)
        layer_ids.append(n)
    check(bool(layer_ids), f"{len(layer_ids)} direction tensors")

    declared = meta.get("glp.layer_ids_zero_based", "")
    if declared:
        got = sorted(layer_ids)
        want = [int(x) for x in declared.split(",") if x != ""]
        check(got == want, "glp.layer_ids_zero_based matches tensor names"
              if got == want else
              f"layer_ids_zero_based {want[:5]}… != tensor names {got[:5]}…")

    if "glp.content_sha256" in meta:
        h = hashlib.sha256()
        for name, dims, dtype, raw in sorted(
                (t for t in tensors if t[0].startswith("direction.")),
                key=lambda t: int(t[0].split(".")[1])):
            h.update(raw)
        actual = h.hexdigest()
        check(actual == meta["glp.content_sha256"],
              "glp.content_sha256 recomputes from tensor bytes"
              if actual == meta["glp.content_sha256"] else
              f"content_sha256 declared {meta['glp.content_sha256'][:16]}… "
              f"!= tensor bytes {actual[:16]}… (pre-0.1.1 hashing bug?)")

    out(f"\n  {os.path.basename(path)}: {len(fails)} FAIL, {len(warns)} WARN")
    return 1 if fails else 0


# ---------------------------------------------------------------------------
# inspect / export -- derived views of a shipped file (stdlib only)
# ---------------------------------------------------------------------------
# The GGUF stays canonical: inspect and export produce VIEWS of it, never a
# second source of truth. Like validate, both run where files are served, so
# they share _read_gguf and pull in nothing beyond the stdlib.


def _tensor_floats(dims, dtype, raw):
    """Tensor payload as Python floats, or None if the dtype is not readable."""
    import struct
    n = 1
    for x in dims:
        n *= x
    if dtype == 0 and len(raw) == 4 * n:
        return list(struct.unpack(f"<{n}f", raw))
    if dtype == 1 and len(raw) == 2 * n:
        return list(struct.unpack(f"<{n}e", raw))
    return None


def inspect_gguf(path, topk=5):
    """Read a control-vector GGUF into a report dict (see format_inspect).

    Raises ValueError on a non-GGUF file or one with no direction.N tensors.
    A file with the tensors but no glp.* metadata is still reported -- that is
    exactly the legacy additive llama.cpp case validate warns about -- with
    the gap flagged in the report rather than hidden.
    """
    meta, tensors = _read_gguf(path)
    layers = []
    for name, dims, dtype, raw in tensors:
        m = re.fullmatch(r"direction\.(\d+)", name)
        if not m:
            continue
        vals = _tensor_floats(dims, dtype, raw)
        if vals is None:
            raise ValueError(f"{name}: unreadable tensor payload "
                             f"(dtype {dtype}, {len(raw)} bytes)")
        entry = {"layer": int(m.group(1)), "norm": sum(x * x for x in vals) ** 0.5,
                 "cos_prev": None, "_vals": vals}
        if topk > 0:
            order = sorted(range(len(vals)), key=lambda i: -abs(vals[i]))
            entry["top_dims"] = [[i, vals[i]] for i in order[:topk]]
        layers.append(entry)
    if not layers:
        raise ValueError("not a GLP control vector: no direction.N tensors "
                         "(run --validate for a full diagnosis)")
    layers.sort(key=lambda e: e["layer"])
    for prev, cur in zip(layers, layers[1:]):
        cur["cos_prev"] = (sum(x * y for x, y in zip(prev["_vals"], cur["_vals"]))
                           / (prev["norm"] * cur["norm"] + 1e-12))
    widths = sorted({len(e["_vals"]) for e in layers})
    for e in layers:
        del e["_vals"]
    return {
        "file": os.path.basename(path),
        "glp": "glp.mode" in meta,
        "metadata": {k: meta[k] for k in sorted(meta)
                     if k.startswith("glp.") or k.startswith("general.base_model.")},
        "n_embd": widths[0] if len(widths) == 1 else widths,
        "layers": layers,
    }


def format_inspect(rep):
    """Render an inspect_gguf report as text."""
    out = [f"  {rep['file']}"]
    if not rep["glp"]:
        out.append("  note: no glp.* metadata -- legacy llama.cpp control "
                   "vector; reporting tensors only")
    if rep["metadata"]:
        out.append("  metadata:")
        for k, v in rep["metadata"].items():
            out.append(f"    {k:36s} {v}")
    layers = rep["layers"]
    out.append(f"  {len(layers)} direction tensors, layers "
               f"{layers[0]['layer']}..{layers[-1]['layer']}, "
               f"n_embd {rep['n_embd']}")
    hdr = f"  {'layer':>5} {'norm':>8} {'cos(prev)':>10}"
    if "top_dims" in layers[0]:
        hdr += "   top dims by |value|"
    out.append(hdr)
    for e in layers:
        cos = f"{e['cos_prev']:+.4f}" if e["cos_prev"] is not None else "-"
        line = f"  {e['layer']:>5} {e['norm']:>8.4f} {cos:>10}"
        if "top_dims" in e:
            line += "   " + " ".join(f"{i}:{v:+.3f}" for i, v in e["top_dims"])
        out.append(line)
    return "\n".join(out)


def export_safetensors(gguf_path, out_path):
    """Write the direction tensors of a GLP GGUF as a .safetensors file.

    safetensors is an 8-byte little-endian u64 header length, a JSON header
    (per-tensor dtype/shape/data_offsets, plus an optional __metadata__ string
    map), then the raw little-endian tensor buffer. GGUF F32 payload bytes are
    already little-endian, so the buffer is a byte copy -- no conversion, no
    precision loss. glp.*/general.* metadata is copied into __metadata__ so
    provenance travels with the export.

    F32 1-D direction tensors only: anything else is not a captain-vector
    export and converting it here would silently change dtype semantics, so
    refuse. Returns the exported tensor names, sorted by layer.
    """
    import struct
    meta, tensors = _read_gguf(gguf_path)
    entries = []
    for name, dims, dtype, raw in tensors:
        if not re.fullmatch(r"direction\.\d+", name):
            continue
        if dtype != 0:
            raise ValueError(f"{name}: dtype is not F32 -- refusing to convert")
        if len(dims) != 1:
            raise ValueError(f"{name}: shape {dims} is not 1-D -- refusing")
        entries.append((int(name.split(".")[1]), name, dims[0], raw))
    if not entries:
        raise ValueError("no direction.N tensors in this file")
    entries.sort()

    header, buf = {}, b""
    for _, name, n, raw in entries:
        header[name] = {"dtype": "F32", "shape": [n],
                        "data_offsets": [len(buf), len(buf) + len(raw)]}
        buf += raw
    header["__metadata__"] = {
        k: str(v) for k, v in sorted(meta.items())
        if k.startswith("glp.") or k.startswith("general.")}
    hj = json.dumps(header).encode()
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + buf)
    return [name for _, name, _, _ in entries]


# ---------------------------------------------------------------------------
# bake -- fold a shipped vector into a rank-1 PEFT/LoRA adapter (needs torch)
# ---------------------------------------------------------------------------
# inspect/export stay stdlib-only because they run where files are served;
# bake runs where the BASE WEIGHTS live, so torch + safetensors are required
# here -- and imported inside this section, so a torch-less serving machine
# never pays for them.
#
# The math: applying h <- h - alpha*(h.d)d at a residual writer h = Wx is a
# weight edit dW = -alpha * d (d^T W), exactly a rank-1 LoRA with
#   lora_B = d                 (out, 1)   unit norm
#   lora_A = -alpha * d^T W    (1, in)    alpha baked in
#   r = 1, lora_alpha = 1      (peft scaling 1.0 -- do not scale the adapter)
# so B(Ax) = -alpha * (d.Wx) d. lora_A CARRIES W, which makes the adapter
# checkpoint-bound: baked against the wrong base revision it is garbage, and
# nothing downstream will flag it. Hence the pin check below fails closed.

# Residual-writing suffixes bake auto-detects, per layer, from the base
# model's own weight index. Longer names first so a hybrid model's
# self_attn.o_proj / linear_attn.out_proj win over the bare suffixes.
BAKE_SUFFIXES = ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj",
                 "o_proj", "out_proj", "down_proj")


def _hf_hub_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def _bake_resolve_base(base, pin, revision=None, out=print):
    """Resolve --base to a snapshot directory of safetensors shards.

    Returns (snapshot_dir, resolved_revision). Fails closed (ValueError) when
    the base cannot be shown to BE the revision the GGUF pins: lora_A bakes W,
    so the wrong W is a garbage adapter, not an error anyone raises later.
    `revision` is the explicit escape hatch; using it is loud.
    """
    if revision:
        out(f"  WARNING: --revision overrides the GGUF's pinned base revision.")
        out(f"    pinned:     {pin}")
        out(f"    overriding: {revision}")
        out(f"    lora_A bakes W of whatever you point at; if this is not the "
            f"pinned checkpoint the adapter is garbage and nothing flags it.")
        pin = revision

    if os.path.isdir(base):
        # A hub cache repo dir (has refs/ + snapshots/): resolve through its ref.
        if os.path.isdir(os.path.join(base, "snapshots")) and \
                os.path.isdir(os.path.join(base, "refs")):
            ref = os.path.join(base, "refs", "main")
            if not os.path.exists(ref):
                raise ValueError(f"{base}: hub repo dir without refs/main; pass "
                                 f"the snapshots/<sha> directory directly")
            sha = open(ref).read().strip()
            snap = os.path.join(base, "snapshots", sha)
            if not os.path.isdir(snap):
                raise ValueError(f"{base}: refs/main points at {sha[:12]}… but "
                                 f"that snapshot is not downloaded")
            base = snap
        name = os.path.basename(os.path.normpath(base))
        if re.fullmatch(r"[0-9a-f]{40}", name):
            if name != pin:
                raise ValueError(
                    f"base snapshot is at revision {name[:12]}… but the GGUF "
                    f"pins {pin[:12]}… . lora_A bakes these weights; refusing "
                    f"to bake against the wrong checkpoint. (--revision "
                    f"overrides, at your own risk.)")
            out(f"  base revision verified against pin: {name[:12]}…")
        elif name == "snapshots":
            kids = [k for k in os.listdir(base)
                    if os.path.isdir(os.path.join(base, k))]
            if len(kids) != 1:
                raise ValueError(f"{base}: pass the snapshot directory itself "
                                 f"(snapshots/<sha>), not its parent")
            return _bake_resolve_base(os.path.join(base, kids[0]), pin,
                                      out=out)
        else:
            out(f"  WARNING: {base} is a plain directory -- cannot verify it "
                f"is revision {pin[:12]}… . Baking anyway; check the provenance "
                f"yourself, the adapter inherits whatever these weights are.")
        return base, pin

    # HF repo id: require the pinned snapshot in the LOCAL cache. No network:
    # downloading whatever origin currently serves is the opposite of a pin.
    cache = _hf_hub_cache()
    repo_dir = os.path.join(cache, "models--" + base.replace("/", "--"))
    snap = os.path.join(repo_dir, "snapshots", pin)
    if not os.path.isdir(snap):
        raise ValueError(
            f"{base}: pinned snapshot {pin[:12]}… not found in the local HF "
            f"cache ({repo_dir}). Fetch exactly that revision first, e.g. "
            f"hf download {base} --revision {pin}")
    out(f"  base revision verified against pin: {pin[:12]}… (local HF cache)")
    return snap, pin


def _bake_weight_index(snap):
    """tensor name -> shard file, for a sharded or single-file checkpoint."""
    idx_p = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx_p):
        return json.load(open(idx_p))["weight_map"]
    single = os.path.join(snap, "model.safetensors")
    if os.path.exists(single):
        import struct
        with open(single, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
        return {k: "model.safetensors" for k in hdr if k != "__metadata__"}
    raise ValueError(f"{snap}: no model.safetensors[.index.json] found")


def bake_lora(gguf_path, base, out_dir, alpha=None, modules=None,
              revision=None, out=print):
    """Bake a GLP GGUF's directions into a rank-1 PEFT/LoRA adapter.

    Writes adapter_model.safetensors (fp32), adapter_config.json (r=1,
    lora_alpha=1 -- alpha is baked into lora_A, so peft must not scale) and
    bake-report.json into out_dir. Returns 0.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    meta, tensors = _read_gguf(gguf_path)
    pin = meta.get("general.base_model.0.version", "")
    if not re.fullmatch(r"[0-9a-f]{40}", pin or ""):
        raise ValueError(
            f"{gguf_path}: general.base_model.0.version is {pin!r}, not a full "
            f"commit sha. bake is checkpoint-bound and refuses an unpinned vector.")
    if alpha is None:
        alpha = float(meta.get("glp.alpha_default", 1.0))
    if meta.get("glp.mode", "project") != "project":
        raise ValueError("bake implements the projective edit h -= alpha*(h.d)d; "
                         f"this GGUF declares mode={meta.get('glp.mode')!r}")

    dirs = {}
    for name, dims, dtype, raw in tensors:
        m = re.fullmatch(r"direction\.(\d+)", name)
        if not m:
            continue
        vals = _tensor_floats(dims, dtype, raw)
        if vals is None:
            raise ValueError(f"{name}: unreadable payload (dtype {dtype})")
        d = torch.tensor(vals, dtype=torch.float32)
        dirs[int(m.group(1))] = d / d.norm()
    if not dirs:
        raise ValueError("no direction.N tensors in this file")

    snap, resolved_rev = _bake_resolve_base(base, pin, revision, out=out)
    wmap = _bake_weight_index(snap)
    suffixes = tuple(modules) if modules else BAKE_SUFFIXES

    # Per direction layer, the residual writers that exist in THIS checkpoint.
    wanted = {}
    for L in sorted(dirs):
        hits = sorted(k for k in wmap if f".layers.{L}." in k
                      and any(k.endswith(s + ".weight") for s in suffixes))
        if not hits:
            raise ValueError(
                f"layer {L}: no weight ending in {suffixes} found in the base "
                f"-- wrong model, or pass --modules explicitly")
        wanted[L] = hits
    used_suffixes = sorted({s for L in wanted for k in wanted[L]
                            for s in suffixes if k.endswith(s + ".weight")})
    by_shard = {}
    for L, keys in wanted.items():
        for k in keys:
            by_shard.setdefault(wmap[k], []).append((L, k))

    org = meta.get("general.base_model.0.organization", "")
    mname = meta.get("general.base_model.0.name", "")
    base_name = f"{org}/{mname}" if org and mname else (mname or base)

    adapter, report_layers, roundtrip = {}, {}, {}
    probes = sorted(((L, k) for L in wanted for k in wanted[L]))
    probes = {probes[0], probes[-1]}          # first and last baked matrix
    for shard in sorted(by_shard):
        with safe_open(os.path.join(snap, shard), framework="pt") as f:
            for L, k in sorted(by_shard[shard]):
                W = f.get_tensor(k).to(torch.float32)      # (out, in)
                d = dirs[L]
                if W.shape[0] != d.numel():
                    raise ValueError(f"{k}: out dim {W.shape[0]} != direction "
                                     f"dim {d.numel()} -- wrong base model")
                A = (-alpha * (d @ W)).unsqueeze(0).clone()   # (1, in)
                B = d.unsqueeze(1).clone()                    # (out, 1)
                stem = k[:-len(".weight")]
                mod = stem.split(f".layers.{L}.", 1)[-1]
                adapter[f"base_model.model.{stem}.lora_A.weight"] = A
                adapter[f"base_model.model.{stem}.lora_B.weight"] = B
                rel = float((B @ A).norm() / W.norm().clamp_min(1e-12))
                report_layers[f"{L}:{mod}"] = {"rel_fro": rel,
                                               "norm_A": float(A.norm())}
                out(f"  layer {L:>3}  {mod:<24} |dW|/|W| {rel:.4f}")
                if (L, k) in probes:
                    g = torch.Generator().manual_seed(L)
                    x = torch.randn(W.shape[1], generator=g)
                    h = W @ x
                    direct = h - alpha * torch.dot(h, d) * d
                    via = h + (B @ (A @ x.unsqueeze(1))).squeeze(1)
                    roundtrip[f"{L}:{mod}"] = float(
                        (direct - via).abs().max() / (h.norm() + 1e-9))

    os.makedirs(out_dir, exist_ok=True)
    st_path = os.path.join(out_dir, "adapter_model.safetensors")
    save_file(adapter, st_path, metadata={
        "format": "pt", "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": "1", "lora_alpha": "1",
        "alpha_baked": repr(alpha),
        "target_modules": ",".join(used_suffixes),
        "base_model": base_name, "revision": resolved_rev,
        "source_gguf": os.path.basename(gguf_path),
        "glp_content_sha256": str(meta.get("glp.content_sha256", "")),
        "derivation": "rank-1 abliteration, dW = -alpha*d(d^T W)",
        "generator": f"captain-vector {__version__} bake",
        "warning": "lora_alpha/r = 1.0, alpha is baked into lora_A -- do not "
                   "scale the adapter. Checkpoint-bound: lora_A carries W of "
                   "the pinned revision.",
    })
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_name, "revision": resolved_rev,
        "inference_mode": True, "r": 1, "lora_alpha": 1, "lora_dropout": 0.0,
        "bias": "none", "fan_in_fan_out": False, "use_rslora": False,
        "init_lora_weights": True,
        "target_modules": sorted({s.rsplit(".", 1)[-1] for s in used_suffixes}),
        "modules_to_save": None, "layers_to_transform": None,
        "layers_pattern": None,
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    rels = [v["rel_fro"] for v in report_layers.values()]
    report = {
        "source_gguf": os.path.basename(gguf_path),
        "base_model": base_name, "base_revision": resolved_rev,
        "revision_overridden": bool(revision),
        "alpha": alpha,
        "modules": sorted({k.split(":", 1)[1] for k in report_layers}),
        "layers": report_layers, "roundtrip": roundtrip,
        "n_tensors": len(adapter), "n_layers": len(wanted),
        "rel_fro": {"min": min(rels), "max": max(rels),
                    "mean": sum(rels) / len(rels)},
        "bytes": os.path.getsize(st_path),
    }
    with open(os.path.join(out_dir, "bake-report.json"), "w") as f:
        json.dump(report, f, indent=1)

    out(f"\n  wrote {st_path} ({report['bytes'] / 1e6:.1f} MB, "
        f"{len(adapter)} tensors, {len(wanted)} layers)")
    out(f"  rel_fro min {min(rels):.4f} max {max(rels):.4f} "
        f"mean {sum(rels) / len(rels):.4f}")
    rt = {k: f"{v:.2e}" for k, v in roundtrip.items()}
    out(f"  roundtrip max rel err: {rt}")
    return 0


def _bake_cmd(argv):
    p = argparse.ArgumentParser(
        prog="captain-vector bake",
        description="Bake a GLP control-vector GGUF into a rank-1 PEFT/LoRA "
                    "adapter against the pinned base checkpoint. Requires "
                    "torch + safetensors.")
    p.add_argument("file", help="control-vector GGUF")
    p.add_argument("--base", required=True,
                   help="local snapshot dir or HF repo id; must resolve to the "
                        "GGUF's pinned revision")
    p.add_argument("--out", required=True, metavar="DIR",
                   help="output directory for the adapter")
    p.add_argument("--alpha", type=float, default=None,
                   help="bake strength; default is the GGUF's glp.alpha_default")
    p.add_argument("--modules", default="", metavar="SUF1,SUF2",
                   help="comma-separated module suffixes; default auto-detects "
                        "residual writers (o_proj/out_proj/down_proj) per layer")
    p.add_argument("--revision", default="", metavar="SHA",
                   help="ESCAPE HATCH: bake against this revision instead of "
                        "the GGUF pin (loud warning; wrong W = garbage adapter)")
    a = p.parse_args(argv)
    if torch is None:
        print("  error: bake requires torch and safetensors "
              "(inspect/export/--validate remain stdlib-only)", file=sys.stderr)
        return 1
    try:
        return bake_lora(a.file, a.base, a.out, alpha=a.alpha,
                         modules=[s.strip() for s in a.modules.split(",")
                                  if s.strip()] or None,
                         revision=a.revision or None)
    except (ValueError, OSError) as e:
        print(f"  error: {e}", file=sys.stderr)
        return 1


def _file_cmd(cmd, argv):
    """Dispatch the stdlib-only file subcommands (inspect, export)."""
    p = argparse.ArgumentParser(prog=f"captain-vector {cmd}")
    p.add_argument("file", help="control-vector GGUF")
    if cmd == "inspect":
        p.add_argument("--json", action="store_true",
                       help="machine-readable report instead of text")
        p.add_argument("--topk", type=int, default=5, metavar="N",
                       help="top-N magnitude dimensions per layer (0 skips)")
    else:
        p.add_argument("--out", required=True, metavar="PATH.safetensors")
    a = p.parse_args(argv)
    try:
        if cmd == "inspect":
            rep = inspect_gguf(a.file, topk=a.topk)
            print(json.dumps(rep, indent=2) if a.json else format_inspect(rep))
        else:
            names = export_safetensors(a.file, a.out)
            print(f"  wrote {a.out}: {len(names)} direction tensors "
                  f"({names[0]}..{names[-1]})")
    except (ValueError, OSError) as e:
        print(f"  error: {e}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
def load_prompts(path):
    """Accept a bare list of strings, or an object with results/items records."""
    d = json.load(open(path))
    rows = d if isinstance(d, list) else (d.get("results") or d.get("items"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"{path}: expected a non-empty list of strings, or an object with a "
            f"'results' or 'items' list")
    if isinstance(rows[0], dict):
        missing = [i for i, r in enumerate(rows) if "prompt" not in r]
        if missing:
            raise ValueError(f"{path}: records without a 'prompt' key at {missing[:5]}")
        return [r["prompt"] for r in rows]
    return [str(r) for r in rows]


def pick_device(pref: str = "auto") -> str:
    """cuda, then Apple MPS, then CPU.

    MPS is worth trying: an M-series Mac with 128 GB unified memory holds a 27B
    model in bf16 more comfortably than most single GPUs, and capture is
    prefill-only so it is short. But coverage of exotic ops is incomplete --
    gated-delta-net / state-space mixers in particular use kernels that may not
    have MPS implementations. `PYTORCH_ENABLE_MPS_FALLBACK=1` routes the gaps to
    CPU, which works but is slow.

    Verify before trusting numbers from a new backend: capture the same prompts
    on two devices and check the directions agree to cos > 0.999. Different
    kernels are not required to produce identical activations.
    """
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        if not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
            print("  note: set PYTORCH_ENABLE_MPS_FALLBACK=1 if an op is unimplemented on MPS")
        return "mps"
    return "cpu"


def parse_span(s, n):
    lo, hi = (int(x) for x in s.split("-"))
    return list(range(max(lo, 0), min(hi, n - 1) + 1))


def main():
    # The file-serving subcommands are stdlib-only and take a GGUF path rather
    # than the derivation flags below; dispatch before argparse sees them.
    # bake joins them here (it also takes a GGUF, not derivation flags) but is
    # NOT stdlib-only -- it needs torch and safetensors for the base weights.
    if len(sys.argv) > 1 and sys.argv[1] in ("inspect", "export"):
        sys.exit(_file_cmd(sys.argv[1], sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "bake":
        sys.exit(_bake_cmd(sys.argv[2:]))
    p = argparse.ArgumentParser(
        prog="captain-vector",
        description="Derive a projective control vector and export it as GGUF.")
    p.add_argument("--validate", default="", metavar="FILE.gguf",
                   help="check a control-vector GGUF against the GLP spec and "
                        "exit (stdlib-only path; no model, no torch needed)")
    p.add_argument("--model", help="local path or HF id")
    p.add_argument("--revision", default="", help="pins the artifact to a checkpoint")
    p.add_argument("--harmful", help="JSON list, or {results:[{prompt}]}")
    p.add_argument("--harmless", help="FORM-MATCHED counterpart")
    p.add_argument("--out")

    p.add_argument("--estimator", default="dom", choices=sorted(ESTIMATORS))
    p.add_argument("--structure", default="per-layer", choices=("per-layer", "pooled"))
    p.add_argument("--hook", default="post_layer",
                   choices=("post_layer", "attn_out", "ffn_out"))
    p.add_argument("--apply-layers", default="", metavar="LO-HI",
                   help="span to ship; default is every validated layer >= 1")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="metadata only; alpha=1 removes the component, >1 REFLECTS it")

    p.add_argument("--min-null-ratio", type=float, default=5.0,
                   help="drop layers whose held-out separation is under this "
                        "multiple of the shuffled-label null (0 disables). "
                        "Default 5.0 is calibrated, not guessed: pure noise "
                        "reaches 2.6-3.5x at p99 across n=16..48 and d=256..5120, "
                        "while a real feature measured 55x. A 2.0 gate would admit "
                        "noise on roughly 1 layer in 8.")
    p.add_argument("--mask", type=float, default=0.0, metavar="FRAC",
                   help="zero the top FRAC of dims by activation magnitude before "
                        "normalising the direction (0.005 = top 0.5%%). Needed only "
                        "on models with massive activations -- the tool measures "
                        "the peak-to-median ratio and tells you. Measured on one "
                        "such model: +12.5 points of delivery AND +12.5 of control, "
                        "simultaneously. On a model without them it removes real "
                        "signal, so this is not a default.")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="auto",
                   help="auto | all | cuda:0 | mps | cpu. auto prefers cuda, "
                        "then MPS. 'all' shards across every visible GPU via "
                        "device_map='auto' -- required when the checkpoint "
                        "exceeds one card")
    p.add_argument("--max-mem-per-gpu", default="", metavar="GiB",
                   help="with --device all, cap each GPU at this many GiB via "
                        "max_memory, forcing accelerate to shard evenly instead "
                        "of greedily filling card 0. Guards against a flaky "
                        "device_map='auto' load-time OOM on big checkpoints. "
                        "e.g. 72 on 80GB H100s")
    p.add_argument("--text-only", action="store_true",
                   help="force AutoModelForCausalLM even when the checkpoint "
                        "declares a ConditionalGeneration arch. For qwen4_exp "
                        "this drops the vision tower and MTP weights at load "
                        "time (the class declares both in "
                        "_keys_to_ignore_on_load_unexpected)")

    p.add_argument("--verify", type=int, default=0, metavar="N",
                   help="hold N prompts per side OUT of the derivation, then "
                        "generate on them with steering off and on and report "
                        "delivery. Held out, not reused: verifying on the prompts "
                        "you derived from measures nothing. Costs a generation "
                        "pass; 0 disables. Try 8.")
    p.add_argument("--verify-max-new", type=int, default=192,
                   help="token cap for --verify. A refusal appears early, so this "
                        "is deliberately short")
    p.add_argument("--verify-batch", type=int, default=4)

    p.add_argument("--author", default="")
    p.add_argument("--description", default="")
    p.add_argument("--contrast", default="")
    p.add_argument("--save-directions", default="", help="also write the raw .pt")
    a = p.parse_args()

    if a.validate:
        sys.exit(validate_gguf(a.validate))
    for req in ("model", "harmful", "harmless", "out"):
        if not getattr(a, req):
            p.error(f"--{req} is required (unless --validate is given)")
    if torch is None:
        p.error("derivation requires torch; only --validate works without it")

    from transformers import AutoTokenizer, AutoConfig
    print(f"  captain-vector {__version__}")
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.model)
    arch = (getattr(cfg, "architectures", None) or [""])[0]
    # Pick the class from what the checkpoint declares. A try/except over two
    # Auto classes silently picks whichever happens not to raise, which is not
    # the same as picking the right one.
    if a.text_only:
        from transformers import AutoModelForCausalLM as Cls
    elif "ConditionalGeneration" in arch or "ImageText" in arch:
        from transformers import AutoModelForImageTextToText as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls
    dev = pick_device(a.device)
    device_map = {"cpu": None, "all": "auto"}.get(dev, dev)
    load_kw = {}
    if device_map == "auto" and a.max_mem_per_gpu and torch.cuda.is_available():
        cap = f"{a.max_mem_per_gpu}GiB"
        load_kw["max_memory"] = {i: cap for i in range(torch.cuda.device_count())}
        print(f"  max_memory {cap} x {torch.cuda.device_count()} GPUs")
    model = Cls.from_pretrained(a.model, dtype=getattr(torch, a.dtype),
                                device_map=device_map, **load_kw)
    if dev == "mps":
        model = model.to("mps")
    model.eval()
    print(f"  device {dev}")
    ad = Adapter(model, cfg)
    print(f"  {arch}  {len(ad.layers)} layers  hidden {ad.hidden}  hook {a.hook}")

    draft_name, draft = find_draft_stack(model)
    if draft is not None:
        print(f"\n  NOTE: this checkpoint has a separate draft stack "
              f"`{draft_name}` ({len(draft)} layer(s)) and THIS VECTOR DOES NOT "
              f"COVER IT.")
        print(f"    Harmless if you serve without speculative decoding. If you "
              f"enable MTP/EAGLE, the draft proposes from an unmodified model, "
              f"the steered target rejects, and acceptance drops on exactly the "
              f"prompts you built this for -- silently, since offline benchmarks "
              f"do not speculate.")
        print(f"    The residual width matches, so the same direction applies "
              f"there; there is simply no slot for it in the direction.N "
              f"convention. Edit the draft stack too, or serve without spec "
              f"decode, or measure your acceptance rate.")

    hp, lp = load_prompts(a.harmful), load_prompts(a.harmless)
    print(f"  harmful {len(hp)}  harmless {len(lp)}")
    if abs(len(hp) - len(lp)) > max(len(hp), len(lp)) * 0.25:
        print("  WARNING: class sizes differ by >25%; a difference of means is "
              "sensitive to that")

    # Split BEFORE capture, so the held-out prompts never touch the estimate.
    # Costing the derivation a few prompts is close to free: subsampling a real
    # contrast from 32 to 8 per side left the normalised direction's structure
    # unchanged (participation ratio 811 -> 796, kurtosis 15.9 -> 16.4). The
    # direction converges long before the sample does.
    hp_held, lp_held = [], []
    if a.verify > 0:
        k = a.verify
        if min(len(hp), len(lp)) - k < 8:
            sys.exit(f"  --verify {k} would leave under 8 prompts per side to "
                     f"derive from. Use a smaller N or more prompts.")
        rng = random.Random(0)                    # fixed: the split is reproducible
        hi = rng.sample(range(len(hp)), k)
        li = rng.sample(range(len(lp)), k)
        hp_held = [hp[i] for i in hi]
        lp_held = [lp[i] for i in li]
        hp = [p for i, p in enumerate(hp) if i not in set(hi)]
        lp = [p for i, p in enumerate(lp) if i not in set(li)]
        print(f"  --verify: holding out {k}+{k}; deriving from "
              f"{len(hp)}+{len(lp)}")

    print("  capturing ...", flush=True)
    A = capture(tok, model, ad, hp, a.hook, a.batch)
    B = capture(tok, model, ad, lp, a.hook, a.batch)

    # Hyper-connection architectures (qwen4_exp hc_count>1, DSV4-style) carry a
    # WIDENED residual between layers, so the post-layer stream is wider than
    # config hidden_size. Directions live in whatever space the hook saw --
    # correct ad.hidden so the random-direction floor and the printout are honest.
    width = int(A[sorted(A)[0]].shape[-1])
    if width != ad.hidden:
        print(f"  NOTE: post-layer stream is {width} wide "
              f"(config hidden {ad.hidden}) -- deriving in the widened space")
        ad.hidden = width
    fn = ESTIMATORS[a.estimator]

    # Massive-activation screen. Cheap, and it decides whether --mask is even
    # relevant for this checkpoint.
    mr = {L: massive_ratio(A[L], B[L]) for L in sorted(A)}
    worst = max(mr, key=lambda L: mr[L][0])
    peak, topdims = mr[worst]
    common = max(set(sum((mr[L][1] for L in mr), [])),
                 key=lambda d: sum(d in mr[L][1] for L in mr))
    n_with = sum(common in mr[L][1] for L in mr)
    print(f"\n  massive-activation screen: peak/median {peak:.0f}x at layer {worst}"
          f"; dim {common} in top-3 of {n_with}/{len(mr)} layers")
    if a.mask <= 0 and peak >= 50 and n_with >= len(mr) * 0.8:
        print(f"  RECOMMEND --mask 0.005: one dimension dominates at every depth, "
              f"so a difference of means here partly points at it rather than at "
              f"your contrast. Reference: 75-434x needs masking, 6-17x does not.")
    elif a.mask > 0 and peak < 20:
        print(f"  NOTE: --mask is set but this model shows no massive activations "
              f"({peak:.0f}x). Masking will remove real signal here.")

    print(f"\n  {'layer':>5} {'held-out':>9} {'null':>7} {'ratio':>7}  verdict")
    dirs, dropped, ratios = {}, [], {}
    for L in sorted(A):
        held, null = validate_layer(A[L], B[L], fn, reps=a.reps)
        ratio = held / max(null, 1e-9)
        ratios[L] = ratio
        keep = (a.min_null_ratio <= 0) or (ratio >= a.min_null_ratio)
        if keep:
            dirs[L] = apply_mask(fn(A[L], B[L]), A[L], B[L], a.mask)
        else:
            dropped.append(L)
        if L % 8 == 0 or not keep:
            print(f"  {L:>5} {held:>9.3f} {null:>7.3f} {ratio:>7.1f}x  "
                  f"{'keep' if keep else 'DROP -- at or below chance'}")

    if not dirs:
        sys.exit("  no layer cleared the null. The contrast is not separable here.")
    if dropped:
        print(f"\n  dropped {len(dropped)} layer(s) below {a.min_null_ratio}x null: "
              f"{dropped[:12]}{' ...' if len(dropped) > 12 else ''}")

    # sign: point along the harmful-minus-harmless difference, so that a POSITIVE
    # alpha removes the behaviour. Getting this backwards installs it instead.
    ref = max(dirs, key=lambda L: ratios[L])
    anchor = A[ref].mean(0) - B[ref].mean(0)
    for L in list(dirs):
        if float(dirs[L] @ _unit(anchor)) < 0 and L == ref:
            dirs = {k: -v for k, v in dirs.items()}
            break

    if a.structure == "pooled":
        M = torch.stack([dirs[L] for L in sorted(dirs)])
        _, _, Vh = torch.linalg.svd(M, full_matrices=False)
        g = _unit(Vh[0])
        if float(g @ dirs[ref]) < 0:
            g = -g
        dirs = {L: g.clone() for L in dirs}
        print(f"  pooled: one global vector reused at {len(dirs)} layers")

    span = parse_span(a.apply_layers, len(ad.layers)) if a.apply_layers else None
    ship = {L: v for L, v in dirs.items() if L >= 1 and (span is None or L in span)}
    if not ship:
        sys.exit("  --apply-layers excluded every validated layer")

    per_dose = {L: dose(A[L], B[L], ship[L]) for L in ship}
    dz = statistics.mean(per_dose.values())
    peak_L = max(per_dose, key=per_dose.get)
    peak = per_dose[peak_L] * a.alpha
    n_over = sum(1 for v in per_dose.values() if v * a.alpha > 0.5)
    # A raw dose is uninterpretable without its floor: a direction drawn at random
    # in d dimensions already scores about 1/sqrt(d).
    floor = ad.hidden ** -0.5
    print(f"\n  effective dose: {dz:.3f} mean, max {per_dose[peak_L]:.3f} at layer "
          f"{peak_L} ({dz / floor:.1f}x the {floor:.3f} a random direction scores)")
    print(f"    at alpha={a.alpha:g}: ~{100 * dz * a.alpha:.0f}% of the residual norm "
          f"on average, ~{100 * peak:.0f}% at the worst layer")
    if n_over:
        hot = sorted((L for L, v in per_dose.items() if v * a.alpha > 0.5),
                     key=lambda L: -per_dose[L])
        print(f"    WARNING: {n_over} layer(s) lose over 50% of the residual norm, "
              f"worst {100 * peak:.0f}% at layer {peak_L}.")
        print(f"    over threshold: " + ", ".join(
            f"L{L} {100 * per_dose[L] * a.alpha:.0f}%" for L in hot[:12])
            + (" ..." if len(hot) > 12 else ""))
        keep = sorted(L for L in ship if L not in set(hot))
        if keep:
            # contiguous runs, so the suggestion is paste-able
            runs, start, prev = [], keep[0], keep[0]
            for L in keep[1:]:
                if L != prev + 1:
                    runs.append((start, prev)); start = L
                prev = L
            runs.append((start, prev))
            span = ",".join(f"{a_}-{b_}" if a_ != b_ else str(a_) for a_, b_ in runs)
            print(f"    a span excluding them: --apply-layers {span}")
            print(f"    (or lower alpha to {0.5 / max(per_dose.values()):.2f}, which "
                  f"puts every layer under the threshold)")
        print(f"    Measured on one model: every configuration with no layer above 50% "
              f"worked (4/4); every one with a layer above broke (4/4), including a "
              f"direction at cosine +0.78 to a working one that produced 96/96 "
              f"degenerate outputs and capability 0/12. Its MEAN dose looked normal.")
        print(f"    Check the contrast for a length/register confound -- it lands in "
              f"early layers -- or exclude those layers. Then --verify.")
    else:
        print(f"    no layer exceeds 50%; the mean is not the number that matters -- "
              f"a matched-mean pair (0.248 vs 0.244) had opposite outcomes, separated "
              f"only by the per-layer maximum (0.417 vs 0.653).")

    val = (f"held-out vs shuffled-label null, {a.reps} reps; kept ratio >= "
           f"{a.min_null_ratio}x; min kept {min(ratios[L] for L in ship):.1f}x"
           f"; dose {dz:.3f}/alpha ({dz / floor:.1f}x random)")
    if a.mask > 0:
        val += f"; masked top {100*a.mask:g}% of dims (peak/median {peak:.0f}x)"

    vr = None
    if a.verify > 0:
        print(f"\n  verifying on {len(hp_held)} held-out harmful + "
              f"{len(lp_held)} held-out harmless, alpha={a.alpha} ...", flush=True)
        vr = verify(tok, model, ad, ship, a.alpha, a.hook,
                    hp_held, lp_held, a.verify_max_new, a.verify_batch)
        print(f"  {'':<10} {'harmful delivery':>17} {'harmless delivery':>18}")
        for arm in ("baseline", "steered"):
            r = vr.get(arm, {})
            def cell(side):
                d = r.get(side)
                if not d:
                    return f"{'-':>17}"
                g = f" ({d['garbled']} garbled)" if d["garbled"] else ""
                return f"{d['comply']:>3}/{d['n']:<3} {100*d['delivery']:5.1f}%{g}"
            print(f"  {arm:<10} {cell('harmful'):>17} {cell('harmless'):>18}")

        bh = vr.get("baseline", {}).get("harmful", {}).get("delivery")
        sh = vr.get("steered", {}).get("harmful", {}).get("delivery")
        bl = vr.get("baseline", {}).get("harmless", {}).get("delivery")
        sl = vr.get("steered", {}).get("harmless", {}).get("delivery")
        gar = sum(vr.get("steered", {}).get(s, {}).get("garbled", 0)
                  for s in ("harmful", "harmless"))
        notes = []
        if bh is not None and sh is not None:
            notes.append(f"harmful {100*bh:.0f}%->{100*sh:.0f}%")
            if sh <= bh:
                print("\n  WARNING: steering did NOT increase delivery on held-out "
                      "harmful prompts. The direction separates but may not be "
                      "causal, or the span/alpha is wrong. Do not ship on the "
                      "separation numbers alone.")
        if bl is not None and sl is not None:
            notes.append(f"harmless {100*bl:.0f}%->{100*sl:.0f}%")
            if sl < bl - 0.15:
                print("\n  WARNING: harmless delivery dropped more than 15 points. "
                      "That is over-refusal or damage, not removal. Projection is "
                      "supposed to be self-limiting on prompts carrying little of "
                      "the direction.")
        if gar:
            print(f"\n  WARNING: {gar} garbled completion(s) while steered. A "
                  f"collapsed model scores as compliance under any refusal-string "
                  f"classifier. Lower alpha before trusting any number here.")
        if notes:
            val += f"; verified out-of-sample on {a.verify}+{a.verify} ({', '.join(notes)})"
        print("\n  NOTE: --verify uses a crude refusal-string classifier with a "
              "degeneracy guard. It is a smoke test that the direction is causal, "
              "not a benchmark. Treat it as a floor, not a score.")
    sha, layers = write_gguf(a.out, ship, {
        "model_hint": (getattr(cfg, "model_type", "") or "unknown"),
        "name": os.path.basename(a.out).removesuffix(".gguf"),
        "author": a.author or "unknown",
        "base_model": a.model, "revision": a.revision or "unspecified",
        "alpha": a.alpha, "hook": a.hook, "structure": a.structure,
        "method": (f"{a.estimator}_{a.structure.replace('-', '_')}"
                   + (f"_mask{a.mask:g}" if a.mask > 0 else "")),
        "contrast": a.contrast or f"{os.path.basename(a.harmful)} vs {os.path.basename(a.harmless)}",
        "validation": val,
        "description": a.description or (
            f"Projective control vector. Apply as h -= alpha*(h.d)d at "
            f"{a.hook}. NOT additive: an additive consumer must refuse this file. "
            f"No weights are modified. alpha=1 removes the component; alpha>1 "
            f"REFLECTS it and can install the behaviour instead."),
    })
    if a.save_directions:
        torch.save({"per_layer": ship, "ratios": ratios, "estimator": a.estimator,
                    "hook": a.hook, "structure": a.structure,
                    "verify": vr}, a.save_directions)

    print(f"\n  wrote {a.out}")
    print(f"    {len(layers)} directions, layers {layers[0]}-{layers[-1]}")
    print(f"    mode=project  hook={a.hook}  alpha_default={a.alpha}")
    print(f"    content_sha256 {sha[:16]}")
    if a.alpha > 1:
        print("    WARNING: alpha > 1 reflects rather than removes the component")


if __name__ == "__main__":
    main()
