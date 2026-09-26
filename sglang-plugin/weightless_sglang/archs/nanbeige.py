"""Nanbeige4.2-3B (``NanbeigeForCausalLM``), and execution ids ``looped``:
steering per execution step of a looped model.

A looped model runs the same physical decoder layers several times per
forward (Nanbeige4.2: 22 layers, ``num_loops`` 2, 44 execution steps). Its
GLP file has one direction per EXECUTION STEP, not per physical layer, so
the stack is sized ``per_loop x num_loops`` and indexed by

    exec_id = loop_idx x per_loop + physical_layer_id

(SGLang's own unrolled id: nanbeige.py's ``logical_id`` for the draft
capture and its per-loop KV layer id ``base_layer_id + loop_idx x
total_layers``). The loop index reaches the layer as an argument
(``layer(positions, hidden_states, forward_batch, loop_idx, residual)``);
the row says which positional slot and keyword carry it.

The file-to-step mapping is the looped-model container convention of the
vLLM plugin (vllm-plugin/weightless_steer/archs/ouro.py,
``exec_core_from_env``): tensor ``direction.N`` holds execution step N-1,
because step 0 is steered and a GLP container cannot name layer 0. The
Nanbeige model card states the same mapping (physical layer (N-1) mod
22, loop pass (N-1) div 22). ``exec_core_from_env`` below is that function
ported gate for gate (it cannot be imported: the vLLM module imports vLLM).
The file must say ``glp.structure = per-execution-step`` (install.py,
``_check_structure``), and a looped file on a normal row is refused there
too, so the shift is applied to exactly the files that carry it.

One hook (or forward wrapper) per physical layer dispatches on the loop
index, which is a Python int in the model loop: each call site is fixed
when a CUDA graph is captured, and the edit of each step is recorded into
the graph with that step's stack row.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import os

import torch

from ..core import SiteError, make_post_layer_hook, wrap_forward
from .base import HOOK_POINT, ArchRow, hidden_size

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LoopSpec:
    """The loop geometry of one served model.

    per_loop    physical decoder layers run in each loop pass
    num_loops   loop passes per forward
    arg         positional slot of the loop index in the layer's call
                (``args`` as the forward sees them, without ``self``)
    kwarg       keyword name of the loop index, if it is passed by name
    """

    per_loop: int
    num_loops: int
    arg: int
    kwarg: str

    @property
    def steps(self):
        return self.per_loop * self.num_loops

    def physical(self, exec_id):
        return int(exec_id) % self.per_loop

    def loop_of(self, exec_id):
        return int(exec_id) // self.per_loop

    def exec_id(self, loop_idx, layer_id):
        return int(loop_idx) * self.per_loop + int(layer_id)


def config_loop(*, per_loop, num_loops, arg, kwarg):
    """A row's ``loop`` field: config -> LoopSpec, reading the two counts
    from the named config attributes."""

    def loop(cfg):
        for a in (per_loop, num_loops):
            if not hasattr(cfg, a):
                raise RuntimeError(f"weightless: the model config has no {a!r}; the loop "
                                   f"geometry of this looped row is unknown. Failing closed.")
        return LoopSpec(per_loop=int(getattr(cfg, per_loop)),
                        num_loops=int(getattr(cfg, num_loops)), arg=int(arg), kwarg=str(kwarg))

    loop.fields = (per_loop, num_loops, arg, kwarg)
    return loop


# Nanbeige4.2 (looped). 22 physical NanbeigeDecoderLayer run num_loops (2)
# times; the model loop calls ``layer(positions, hidden_states,
# forward_batch, loop_idx, residual)``, which returns (hidden, residual)
# (Qwen convention). The file has one direction per execution step
# (glp.structure per-execution-step, direction.N = step N-1), indexed
# loop_idx x num_hidden_layers + layer. o_proj and down_proj are
# RowParallelLinear with their default all-reduce and the model uses no
# layer communicator, so the layer output is reduced inside the layer
# (pinned in both SGLang trees by the structure tests) and TP > 1 is
# allowed; SGLang itself refuses PP > 1 for this model.
_NANBEIGE = dict(
    layers=frozenset({"NanbeigeDecoderLayer"}),
    hooks=frozenset({HOOK_POINT}),
    width=hidden_size,
    arity=2, hidden_index=0, residual_index=1,
    install="hook", exec_id="looped", tp="ok",
    hint=frozenset({"nanbeige"}),
    loop=config_loop(per_loop="num_hidden_layers", num_loops="num_loops", arg=3,
                     kwarg="loop_idx"),
)

ROWS = {
    "NanbeigeForCausalLM": ArchRow(backbone=("model",), **_NANBEIGE),
}


def loop_geometry(row, row_name, backbone, meta, path):
    """The LoopSpec of a looped row on this backbone, checked against the
    model and the file. Raises on any mismatch."""
    if row.loop is None:
        raise RuntimeError(f"weightless: {row_name} is looped but declares no loop geometry; "
                           f"execution ids are not implemented for it. Failing closed.")
    spec = row.loop(backbone.config)
    n = len(backbone.layers)
    if spec.per_loop != n:
        raise RuntimeError(
            f"weightless: {row_name} runs {spec.per_loop} layers per loop pass by its config, "
            f"but the backbone holds {n}; the execution ids would land on the wrong steps. "
            f"Failing closed.")
    if spec.num_loops < 1 or spec.arg < 0:
        raise RuntimeError(f"weightless: {row_name} has loop geometry {spec}. Failing closed.")
    method = str(meta.get("glp.method") or "").strip().lower()
    if method and "per_layer" in method and "per_execution_step" not in method:
        raise RuntimeError(
            f"weightless: {path} says glp.structure=per-execution-step but "
            f"glp.method={meta.get('glp.method')!r}, a per-layer derivation; the file "
            f"contradicts itself. Refusing.")
    return spec


def exec_core_from_env(*, hook, exec_steps, hidden_size):
    """SteeringCore.from_env for a looped model: ``direction.N`` = execution
    step N-1. A port of vllm-plugin archs/ouro.py ``exec_core_from_env``:
    the same gates in the same order (mode, hook point and spec gates in
    load_control_vector, the alpha-multiplier and rank-k refusals, alpha
    resolution, the layer filter, the width check, finite non-zero unit
    directions, the range check, an empty selection), with the one shift
    applied before filtering and range checks. WEIGHTLESS_STEER_LAYERS
    therefore names 0-based EXECUTION steps. Returns None when
    WEIGHTLESS_STEER_PATH is unset; raises on everything else."""
    from weightless_steer.container import load_control_vector
    from weightless_steer.core import SteeringCore

    path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
    if not path:
        return None

    env_hook = (os.environ.get("WEIGHTLESS_STEER_HOOK") or hook).strip()
    if env_hook != hook:
        raise RuntimeError(
            f"WEIGHTLESS_STEER_HOOK={env_hook!r} is not implemented at "
            f"this arch adapter; the only site here is {hook}. Refusing "
            f"to serve with a silently wrong hook site.")

    try:
        meta, raw = load_control_vector(path, hook=hook)
        scales = sorted(set(meta) & {"glp.dir_scales", "glp.layer_scales"})
        if scales:
            raise ValueError(
                f"{path}: carries {', '.join(scales)}, which this plugin "
                f"does not implement. Refusing to ignore alpha "
                f"multipliers.")
        if any(v.dim() > 1 for v in raw.values()):
            raise RuntimeError(
                f"{path}: rank-k (subspace) GLP vector: this plugin "
                f"implements rank 1 only. Refusing to serve a partially "
                f"applied subspace.")

        alpha_env = os.environ.get("WEIGHTLESS_STEER_ALPHA", "").strip()
        alpha = (float(alpha_env) if alpha_env
                 else float(meta.get("glp.alpha_default", 1.0)))
        if not math.isfinite(alpha):
            raise ValueError("steering alpha must be finite")

        want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
        selected = ({int(t) for t in want.replace(" ", "").split(",") if t}
                    if want else None)

        dirs = {}
        for container_id, vec in raw.items():
            exec_id = int(container_id) - 1  # direction.N steers execution step N-1
            if selected is not None and exec_id not in selected:
                continue
            vec = vec.detach().to(torch.float32).reshape(-1)
            if vec.numel() != hidden_size:
                raise RuntimeError(
                    f"steering vector exec step {exec_id} width "
                    f"{vec.numel()} != {hidden_size} "
                    f"(hidden_size; plain single stream)")
            norm = vec.double().norm()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError(
                    f"{path}: direction exec step {exec_id} must be finite "
                    f"and nonzero; refusing to steer along a direction "
                    f"that is neither")
            dirs[exec_id] = (vec.double() / norm).float()

        out_of_range = sorted(int(k) - 1 for k in raw if not 0 <= int(k) - 1 < exec_steps)
        if out_of_range:
            raise RuntimeError(
                f"{path}: direction exec steps {out_of_range} out of range "
                f"for this model ({exec_steps} execution steps)")
        if not dirs:
            raise RuntimeError(
                f"WEIGHTLESS_STEER_PATH={path} matched no execution steps; "
                f"refusing to run unsteered")
    except Exception as exc:
        logger.error("GLP steering load failed (%s); failing closed", exc)
        raise

    logger.info("weightless GLP steering active: hook=%s alpha=%.3f exec_steps=%d..%d (%d) "
                "width=%d", hook, alpha, min(dirs), max(dirs), len(dirs), hidden_size)
    return SteeringCore(dirs, alpha, hook, exec_steps, hidden_size, path=path)


def read_loop_idx(spec, args, kwargs, layer_id):
    """The loop index of one layer call, or raise."""
    if len(args) > spec.arg:
        v = args[spec.arg]
    elif spec.kwarg in kwargs:
        v = kwargs[spec.kwarg]
    else:
        raise SiteError(
            f"weightless: looped decoder layer {layer_id} was called without its loop index "
            f"(positional slot {spec.arg} or {spec.kwarg}=); the execution step is unknown. "
            f"Failing closed.")
    if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < spec.num_loops:
        raise SiteError(
            f"weightless: looped decoder layer {layer_id} got loop index {v!r}, not an int in "
            f"[0, {spec.num_loops}); the call style changed. Failing closed.")
    return v


def make_looped_codec(owner, layer_id, spec, steps, **kw):
    """The codec for physical layer ``layer_id``: ``codec(module, args,
    kwargs, output)``. ``steps`` maps the steered execution ids of this
    layer to their next steered execution id (or None). Loop passes whose
    step is not steered pass the output through unchanged."""
    by_loop = {}
    for exec_id, nxt in steps.items():
        by_loop[spec.loop_of(exec_id)] = make_post_layer_hook(
            owner, exec_id, next_layer_id=nxt, **kw)

    def codec(module, args, kwargs, output):
        hook = by_loop.get(read_loop_idx(spec, args, kwargs, layer_id))
        return output if hook is None else hook(module, args, output)

    codec.exec_ids = sorted(steps)
    return codec


def install_looped(backbone, local, spec, install, **kw):
    """Install the looped sites for the local execution ids ``local``
    (sorted, which is the execution order). One hook or forward wrapper per
    physical layer. Returns the undo handles."""
    per_layer = {}
    for n, e in enumerate(local):
        nxt = local[n + 1] if n + 1 < len(local) else None
        per_layer.setdefault(spec.physical(e), {})[e] = nxt
    handles = []
    for i in sorted(per_layer):
        layer = backbone.layers[i]
        codec = make_looped_codec(backbone, i, spec, per_layer[i], **kw)
        if install == "hook":
            handles.append(layer.register_forward_hook(codec, with_kwargs=True))
        elif install == "forward_wrap":
            handles.append(wrap_forward(layer, codec, layer_id=i, with_kwargs=True))
        else:
            raise RuntimeError(f"weightless: install mode {install!r} has no looped form. "
                               f"Failing closed.")
    return handles
