"""Install GLP steering on a loaded SGLang ModelRunner.

Called once per rank process, after the weights load and before the memory
pools are sized and the CUDA graphs are captured, from the plugin's AFTER
hook on ModelRunner.load_model (plugin.py). It uses the vLLM plugin's
loader and gates (SteeringCore). ``source="server_args"`` takes an explicit
path, alpha and layers instead of the environment, for a caller that is
not the plugin hook (for example a server flag); both share one install.

What differs between architectures is data, not code: one ``ArchRow`` per
served model class in ``archs.ARCH`` says where the steering site is and
how to read it. ``install_steering`` is generic.

Two things moved in SGLang's layer-boundary series (2026-09-24 to 09-30):
the decoder layers of many models are built from stage boundaries (see
core.stage_boundaries; a row serves them only with ``ArchRow.boundary``),
and ModelRunner no longer carries tp_rank, tp_size, pp_rank and pp_size
(#41813): the ranks are read from the runner when it has them, else from
``sglang.srt.runtime_context.get_parallel()``, and the boot fails when
neither answers (``read_ranks``).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sys

import torch

from .archs import ARCH, BOUNDARY_SITE, COMPANION, EDIT, PREFLIGHT, REFUSED, SPECIAL, VARIANTS
from .archs.base import EXEC_IDS, INSTALL_MODES
from .core import (FiredCheck, LayerDiag, make_post_layer_hook, make_site, make_stream_hook,
                   stage_boundaries, uses_stage_boundaries, wrap_forward)

logger = logging.getLogger(__name__)

_STEER_ENV = ("WEIGHTLESS_STEER_PATH", "WEIGHTLESS_STEER_ALPHA", "WEIGHTLESS_STEER_LAYERS")


@contextlib.contextmanager
def _steer_env(path, alpha, layers):
    """Present explicit (flag) values to SteeringCore.from_env, which reads
    the WEIGHTLESS_STEER_* environment. Restores the environment after."""
    saved = {k: os.environ.get(k) for k in _STEER_ENV}
    try:
        os.environ["WEIGHTLESS_STEER_PATH"] = str(path)
        for key, val in (("WEIGHTLESS_STEER_ALPHA", alpha), ("WEIGHTLESS_STEER_LAYERS", layers)):
            if val is None or str(val).strip() == "":
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(val)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def resolve_row(model, runner=None):
    """Return (row name, ArchRow) for a served model, or raise.

    A served class with variants (``archs.VARIANTS``: a model SGLang serves
    through another model's class, picked from the model's config) gets the
    first variant whose test passes, else its own row."""
    name = type(model).__name__
    for picks, key in VARIANTS.get(name, ()):
        if picks(model, runner):
            return key, ARCH[key]
    row = ARCH.get(name)
    if row is None:
        if name in REFUSED:
            raise RuntimeError(f"weightless: served model class {name} is not supported: "
                               f"{REFUSED[name]} Refusing to serve unsteered.")
        raise RuntimeError(
            f"weightless: served model class {name} has no SGLang steering "
            f"adapter (supported: {sorted(ARCH)}). Refusing to serve unsteered."
        )
    return name, row


_MISSING = object()


def resolve_backbone(model, row=None):
    """Return (backbone, row) for a served model, or raise. The backbone is
    None when the row's path exists but holds None on this rank."""
    name, row = resolve_row(model) if row is None else (type(model).__name__, row)
    for path in row.backbone:
        bb = model
        for part in filter(None, path.split(".")):
            bb = getattr(bb, part, _MISSING)
            if bb is _MISSING or bb is None:
                break
        if bb is _MISSING:
            continue
        if bb is None:
            return None, row
        for attr in ("layers", "start_layer", "end_layer", "config"):
            if not hasattr(bb, attr):
                raise RuntimeError(
                    f"weightless: {name} backbone {type(bb).__name__} has no "
                    f".{attr}; the SGLang model layout changed. Failing closed."
                )
        return bb, row
    raise RuntimeError(
        f"weightless: {name} has none of the backbone paths {list(row.backbone)}; "
        f"the SGLang model layout changed. Failing closed."
    )


def _model_types(runner, backbone):
    out = set()
    for cfg in (
        getattr(getattr(runner, "model_config", None), "hf_config", None),
        getattr(backbone, "config", None),
    ):
        mt = getattr(cfg, "model_type", None)
        if mt:
            out.add(str(mt))
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _emit(line):
    if logger.isEnabledFor(logging.INFO) and logging.getLogger().handlers:
        logger.info(line)
    else:
        print(line, file=sys.stderr, flush=True)


KERNELS = ("auto", "triton", "torch")


def _act_dtype(runner):
    """The dtype of the hidden states (the model dtype), bf16 if unknown."""
    dt = getattr(getattr(runner, "model_config", None), "dtype", None)
    return dt if isinstance(dt, torch.dtype) else torch.bfloat16


def pick_kernel(choice, device, dtype, width, with_residual=True, check=None, form=None):
    """Pick the kernel for the edit. Returns (kernel, reason).

    WEIGHTLESS_STEER_KERNEL: "auto" (default) uses the fused Triton kernel
    when Triton and a CUDA device are there and its start-up self-check
    gives the same bits as the torch path, and the torch path otherwise.
    "triton" forces the fused kernel and fails the boot when it cannot run.
    "torch" forces the torch path.

    ``width`` and ``with_residual`` are those of the row's steering site
    (its stream width, and whether it passes a residual), so the
    self-check runs exactly the kernel variant the model will run. A row
    with its own edit passes that edit's start-up check as ``check``
    (``check(device, dtype=, width=) -> bool``) and its text as ``form``.
    """
    choice = (choice or "").strip().lower() or "auto"
    if choice not in KERNELS:
        raise RuntimeError(
            f"weightless: WEIGHTLESS_STEER_KERNEL={choice!r} is not one of {', '.join(KERNELS)}")
    if choice == "torch":
        return "torch", "forced by WEIGHTLESS_STEER_KERNEL=torch"
    from . import fused

    form = (f"{form}, width {int(width)}" if form else
            f"width {int(width)}, {'with' if with_residual else 'no'} residual")
    why = None
    if not fused.available():
        why = "Triton or a CUDA device is not available"
    elif torch.device(device).type != "cuda":
        why = f"the model is on {device}, not on a CUDA device"
    else:
        try:
            ok = (check(device, dtype=dtype, width=width) if check is not None else
                  fused.self_check(device, dtype=dtype, width=width, with_residual=with_residual))
            if not ok:
                why = (f"the fused kernel did not give the same bits as the torch path at "
                       f"start-up ({form})")
        except Exception as e:  # compile or launch failure
            why = f"the fused kernel failed at start-up ({type(e).__name__}: {e})"
    if why is None:
        return "triton", f"{choice}: fused Triton kernel, start-up self-check passed ({form})"
    if choice == "triton":
        raise RuntimeError(f"weightless: WEIGHTLESS_STEER_KERNEL=triton but {why}")
    return "torch", f"auto: torch path, because {why}"


_RANKS = ("tp_rank", "tp_size", "pp_rank", "pp_size")


def read_ranks(runner):
    """This rank's (tp_rank, tp_size, pp_rank, pp_size, source).

    The runner's own fields when it has all four (SGLang before #41813, the
    serving build b62df575), else ``get_parallel()``, the process's parallel
    state that SGLang reads them from since. Never a default: a rank that
    cannot be read fails the boot, so a pipeline rank cannot pass as rank 0
    of 1 (the manifests and log lines are keyed by it, and TP > 1 is refused
    on it).
    """
    if all(hasattr(runner, k) for k in _RANKS):
        return tuple(int(getattr(runner, k)) for k in _RANKS) + ("runner",)
    try:
        from sglang.srt.runtime_context import get_parallel
        par = get_parallel()
        vals = tuple(int(getattr(par, k)) for k in _RANKS)
    except Exception as e:
        missing = [k for k in _RANKS if not hasattr(runner, k)]
        raise RuntimeError(
            f"weightless: this rank's tp/pp ranks cannot be read: the ModelRunner has no "
            f"{', '.join(missing)} and get_parallel() did not answer ({type(e).__name__}: {e}). "
            f"Failing closed.") from e
    return vals + ("get_parallel",)


def _refuse_server_args(runner, row_name):
    sa = getattr(runner, "server_args", None)
    if sa is not None and getattr(sa, "enable_two_batch_overlap", False):
        raise RuntimeError(
            f"weightless: two-batch overlap (--enable-two-batch-overlap) runs {row_name}'s "
            f"layers through a path the steering sites are not on. Refusing to serve "
            f"unsteered; turn two-batch overlap off."
        )


def _check_structure(row, row_name, meta, path):
    structure = str(meta.get("glp.structure") or "").strip().lower()
    per_step = structure == "per-execution-step"
    if row.exec_id == "looped" and not per_step:
        raise RuntimeError(
            f"weightless: {row_name} is a looped model steered per execution step, but "
            f"{path} has glp.structure={meta.get('glp.structure')!r}, not "
            f"'per-execution-step'. Refusing."
        )
    if row.exec_id != "looped" and per_step:
        raise RuntimeError(
            f"weightless: {path} is per-execution-step (a looped model's file), but "
            f"{row_name} runs each layer once. Refusing."
        )


def _layout(backbone, local, phys, row, row_name, start=None):
    """The SGLang layer layout of this rank's steered layers: "stage-boundary"
    (core.stage_boundaries) or "layer-output" (everything older); one for
    all of them. A row without ``boundary`` refuses stage-boundary layers,
    and so does a non-special row that the residual-stream codec cannot
    read (anything but a one-stream (hidden, residual) "hook" row).
    A rank with no steered layer reports the layout of its first layer
    (``start``) and refuses nothing: nothing is hooked there."""
    if not local:
        try:
            first = backbone.layers[start] if start is not None else None
        except (IndexError, TypeError):
            first = None
        return "stage-boundary" if first is not None and uses_stage_boundaries(first) else "layer-output"
    kinds = {}
    for i in local:
        kinds.setdefault(
            "stage-boundary" if uses_stage_boundaries(backbone.layers[phys(i)])
            else "layer-output", []).append(i)
    if len(kinds) > 1:
        raise RuntimeError(f"weightless: {row_name} mixes SGLang layer layouts on this rank "
                           f"({ {k: v[:4] for k, v in kinds.items()} }). Failing closed.")
    layout = next(iter(kinds), "layer-output")
    if layout == "stage-boundary" and not row.boundary:
        raise RuntimeError(
            f"weightless: {row_name}'s decoder layers are built from SGLang's stage boundaries "
            f"(attn_boundary / ffn_boundary: the layer returns its output alone and the residual "
            f"stays in the forward batch), and this row's site is only validated on the older "
            f"layout where the layer returns it. Refusing to serve unsteered; this row needs a "
            f"port to the layer-boundary API.")
    if layout == "stage-boundary" and not row.install.startswith("special:"):
        # The residual-stream codec (core.make_stream_hook) edits the layer's output as ONE
        # stream with r = the stream's residual: the (hidden, residual) pair of the older
        # layout and nothing else. A row read another way there would be edited wrongly.
        shape = dict(install=row.install, exec_id=row.exec_id, arity=row.arity,
                     hidden_index=row.hidden_index, residual_index=row.residual_index,
                     per_stream=row.per_stream, residual_may_be_none=row.residual_may_be_none)
        if shape != dict(install="hook", exec_id="layer", arity=2, hidden_index=0,
                         residual_index=1, per_stream=False, residual_may_be_none=False):
            raise RuntimeError(
                f"weightless: {row_name} serves SGLang's stage-boundary layers through the "
                f"residual-stream codec, which edits one stream with the stream's residual (a "
                f"'hook' row whose older output is the (hidden, residual) pair, one direction "
                f"per layer); this row is {shape}. Failing closed.")
    for i in local if layout == "stage-boundary" else ():
        stage_boundaries(backbone.layers[phys(i)])  # the pair this plugin knows, or raise
    return layout


def install_steering(runner, *, source="env", path=None, alpha=None, layers=None):
    """Load the GLP vector and install the steering sites of `runner`.

    source="env": read WEIGHTLESS_STEER_* (the vLLM plugin's contract).
    source="server_args": use the explicit path/alpha/layers.
    Returns the installed record, or None when there is nothing to do.
    Raises (fail closed) on every problem.
    """
    from weightless_steer.core import SteeringCore
    from weightless_steer.container import read_gguf_cvec

    if source == "env":
        path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip() or None
    if not path:
        return None
    if getattr(runner, "is_draft_worker", False):
        # The MTP/NEXTN draft stays stock (vllm-plugin archs/qwen38.py).
        return None

    prev = getattr(runner, "_weightless_steer_installed", None)
    if prev is not None and getattr(runner, "_weightless_steer_model_id", None) == id(runner.model):
        if os.path.realpath(prev["path"]) != os.path.realpath(path):
            raise RuntimeError(
                f"weightless: steering already installed from {prev['path']} "
                f"({prev['source']}); refusing a second vector {path} ({source})"
            )
        return prev  # same file through both routes: install once

    model = runner.model
    row_name, row = resolve_row(model, runner)
    _refuse_server_args(runner, row_name)
    tp_rank, tp_size, pp_rank, pp_size, rank_source = read_ranks(runner)
    if tp_size > 1 and row.tp != "ok":
        raise RuntimeError(
            f"weightless: {row_name} is not validated at TP={tp_size} (its layer outputs may "
            f"be partial sums at the steering site). Refusing."
        )

    install = row.install
    special = None
    if install.startswith("special:"):
        special = SPECIAL.get(install.split(":", 1)[1])
        if special is None:
            raise RuntimeError(f"weightless: {row_name} needs install handler {install!r}, "
                               f"which this build does not have. Failing closed.")
    elif install not in INSTALL_MODES:
        raise RuntimeError(f"weightless: {row_name} has unknown install mode {install!r}")

    backbone, _ = resolve_backbone(model, row)
    if backbone is None:
        _emit(f"weightless GLP steering: {row_name} has no backbone on this rank "
              f"(pp_rank={pp_rank} tp_rank={tp_rank}); nothing is steered here")
        return None
    cfg = backbone.config
    width = int(row.width(cfg))
    hidden = int(getattr(cfg, "hidden_size", width))
    num_layers = len(backbone.layers)

    # The file's hook point picks the site; the row must serve it.
    meta, _ = read_gguf_cvec(path)
    if special is not None:  # row checks before anything is hooked (archs/base.py)
        preflight = PREFLIGHT.get(install.split(":", 1)[1])
        if preflight is not None:
            preflight(runner=runner, backbone=backbone, meta=meta, path=path, row=row,
                      row_name=row_name)
    file_hook = meta.get("glp.hook_point")
    if file_hook not in row.hooks:
        raise RuntimeError(
            f"weightless: {path} has glp.hook_point={file_hook!r}, which {row_name} does "
            f"not serve (it serves {sorted(row.hooks)}). Refusing to apply at the wrong site."
        )
    _check_structure(row, row_name, meta, path)
    if row.exec_id not in EXEC_IDS:
        raise RuntimeError(f"weightless: execution ids {row.exec_id!r} are not implemented in "
                           f"this build ({row_name}). Failing closed.")
    loop = None
    if row.exec_id == "looped":
        # One direction per execution step: direction.N is step N-1 (archs/nanbeige.py).
        from .archs.nanbeige import exec_core_from_env, loop_geometry
        loop = loop_geometry(row, row_name, backbone, meta, path)

    ctx = contextlib.nullcontext() if source == "env" else _steer_env(path, alpha, layers)
    with ctx:
        if loop is None:
            core = SteeringCore.from_env(hook=file_hook, num_layers=num_layers, hidden_size=width)
        else:
            core = exec_core_from_env(hook=file_hook, exec_steps=loop.steps, hidden_size=width)
    if core is None:  # cannot happen with a path set; keep fail-closed
        raise RuntimeError("weightless: steering requested but no core was built")

    hint = meta.get("controlvector.model_hint")
    types = _model_types(runner, backbone)
    if hint and (types or row.hint) and hint not in types and hint not in row.hint:
        raise RuntimeError(
            f"weightless: {core.path} is for model_hint={hint!r} but this "
            f"model is {sorted(types | set(row.hint))}. Refusing."
        )

    start, end = int(backbone.start_layer), int(backbone.end_layer)
    phys = (lambda i: i) if loop is None else loop.physical  # stack row -> decoder layer
    local = [i for i in sorted(core.dirs) if start <= phys(i) < end]
    for i in local:
        cls = type(backbone.layers[phys(i)]).__name__
        if cls not in row.layers:
            raise RuntimeError(
                f"weightless: layer {i} is {cls}, not one of {sorted(row.layers)}; "
                f"its output convention is not validated. Failing closed."
            )
    layout = _layout(backbone, local, phys, row, row_name, start=start if start < end else None)
    if special is not None:
        site_name = special[1]
        if layout == "stage-boundary":
            site_name = BOUNDARY_SITE.get(install.split(":", 1)[1])
            if site_name is None:  # cannot happen with row.boundary checked; keep fail-closed
                raise RuntimeError(f"weightless: {row_name}'s handler {install!r} names no site on "
                                   f"stage-boundary layers. Failing closed.")
    elif install == "hook":
        site_name = ("decoder-layer output (residual stream)" if layout == "stage-boundary"
                     else "decoder-layer output")
    else:
        site_name = "decoder-layer forward"
    # A special row may run its own edit (archs/base.py EDIT), made here, after
    # its preflight has checked the settings it reads.
    special_name = install.split(":", 1)[1] if special is not None else None
    edit = EDIT[special_name]() if special_name in EDIT else None
    if edit is not None and getattr(edit, "label", None):
        site_name = f"{site_name} [{edit.label}]"

    try:
        dev = next(backbone.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    with torch.device(dev):
        core.register_buffers(backbone, torch.float32)

    diag = None
    if os.environ.get("WEIGHTLESS_STEER_DIAG", "") == "1" and local:
        diag = LayerDiag(os.environ.get("WEIGHTLESS_STEER_DIAG_DIR") or os.getcwd(), local)
    kernel, kernel_why = pick_kernel(
        os.environ.get("WEIGHTLESS_STEER_KERNEL", ""), dev, _act_dtype(runner), width,
        with_residual=row.residual_index is not None,
        check=getattr(edit, "self_check", None), form=getattr(edit, "form", None))

    check = FiredCheck(local, what=f"steered {site_name}")
    handles = []
    codec_kw = dict(diag=diag, kernel=kernel, arity=row.arity, hidden_index=row.hidden_index,
                    residual_index=row.residual_index, width=width, per_stream=row.per_stream,
                    check=check, row=row_name, residual_may_be_none=row.residual_may_be_none)
    if loop is not None:  # one hook or wrapper per physical layer, dispatching on the loop index
        from .archs.nanbeige import install_looped
        handles = install_looped(backbone, local, loop, install, **codec_kw)
    for n, i in enumerate(local if loop is None else ()):
        nxt = local[n + 1] if n + 1 < len(local) else None
        if install == "hook" and layout == "stage-boundary":
            handles.append(backbone.layers[i].register_forward_hook(make_stream_hook(
                backbone, i, next_layer_id=nxt, diag=diag, kernel=kernel, width=width,
                check=check, row=row_name), with_kwargs=True))
        elif install == "hook":
            handles.append(backbone.layers[i].register_forward_hook(make_post_layer_hook(
                backbone, i, next_layer_id=nxt, **codec_kw)))
        elif install == "forward_wrap":  # the loop calls layer.forward(...): wrap the instance
            handles.append(wrap_forward(backbone.layers[i], make_post_layer_hook(
                backbone, i, next_layer_id=nxt, **codec_kw), layer_id=i))
        else:
            site = make_site(backbone, i, next_layer_id=nxt, diag=diag, kernel=kernel,
                             width=width, per_stream=row.per_stream, check=check, row=row_name,
                             hidden_slot=site_name, edit=edit)
            handles.append(special[0](backbone.layers[i], site, layer_id=i, backbone=backbone,
                                      row=row))
    undo_check = check.wrap_forward(backbone) if local else None

    # A companion site some files need next to the main one (archs/base.py
    # COMPANION): the layer at position k gets stack row k + row_offset, so it
    # can be a layer the file has no row for. Its own fired check.
    writer = None
    comp = COMPANION.get(special_name) if special_name is not None else None
    if comp is not None and file_hook in comp.hooks:
        pairs = [(k, k + comp.row_offset) for k in range(start, end)
                 if (k + comp.row_offset) in core.dirs]
        for k, _ in pairs:
            cls = type(backbone.layers[k]).__name__
            if cls not in row.layers:
                raise RuntimeError(f"weightless: layer {k} (companion {comp.site}) is {cls}, not "
                                   f"one of {sorted(row.layers)}. Failing closed.")
        wkernel, wkernel_why = pick_kernel(os.environ.get("WEIGHTLESS_STEER_KERNEL", ""), dev,
                                           _act_dtype(runner), width, with_residual=False)
        wcheck = FiredCheck([r for _, r in pairs],
                            what=f"companion {comp.site} (stack row = layer + {comp.row_offset})")
        for k, r in pairs:
            wsite = make_site(backbone, r, kernel=wkernel, width=width, per_stream=False,
                              check=wcheck, row=row_name, hidden_slot=comp.site)
            handles.append(comp.install(backbone.layers[k], wsite, layer_id=k, row_id=r,
                                        backbone=backbone, row=row))
        writer = {"site": comp.site, "row_offset": comp.row_offset,
                  "layers": [k for k, _ in pairs], "rows": [r for _, r in pairs],
                  "kernel": wkernel, "kernel_reason": wkernel_why,
                  "fired_check": "armed" if pairs else "no local sites",
                  "fired_at_capture": wcheck.fired_at_capture,
                  "undo_check": wcheck.wrap_forward(backbone) if pairs else None,
                  "check": wcheck}

    ltypes = [type(backbone.layers[phys(i)]).__name__ for i in local]
    if row.kind is not None:
        lkinds = [row.kind(backbone.layers[phys(i)]) for i in local]
        ltypes = ["Attention" if k == "full" else "Linear" if k == "linear" else "" for k in lkinds]
    n_full = sum(1 for t in ltypes if "Attention" in t)
    n_lin = sum(1 for t in ltypes if "Linear" in t)
    kinds = f"; {n_full} full-attn, {n_lin} linear" if local and n_full + n_lin == len(local) else ""
    if tp_size > 1:
        logger.warning("weightless: TP=%d is not validated for the SGLang plugin on GPUs yet", tp_size)
    sha = _sha256(core.path)
    rng = f"{local[0]}..{local[-1]}" if local else "none"
    line = (
        f"weightless GLP steering active (sglang): source={source} tp_rank={tp_rank} "
        f"layers={rng} ({len(local)}{kinds}) "
        f"alpha={core.alpha:.3f} hook={file_hook} file_sha256={sha} diag={int(diag is not None)} "
        f"pp_rank={pp_rank} row={row_name} install={install} site={site_name} width={width} "
        f"layout={layout} "
        + (f"exec=looped({loop.num_loops}x{loop.per_loop}, direction.N=step N-1) "
           if loop is not None else "")
        + f"kernel={kernel} ({kernel_why})"
    )
    if writer is not None:
        wl = writer["layers"]
        line += (f" companion={writer['site']} layers="
                 f"{(str(wl[0]) + '..' + str(wl[-1])) if wl else 'none'} ({len(wl)}, stack row = "
                 f"layer + {writer['row_offset']}) companion_kernel={writer['kernel']}")
    _emit(line)

    rec = {
        "pid": os.getpid(),
        "source": source,
        "path": core.path,
        "file_sha256": sha,
        "alpha": core.alpha,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "pp_rank": pp_rank,
        "pp_size": pp_size,
        "rank_source": rank_source,
        "layout": layout,
        "hooked_layers": local,
        "local_layer_ids": local,
        "full_attn_layers": [i for i, t in zip(local, ltypes) if "Attention" in t],
        "num_layers": num_layers,
        "hidden_size": hidden,
        "width": width,
        "model_class": type(model).__name__,
        "row": row_name,
        "install": install,
        "site": site_name,
        "hook_point": file_hook,
        "model_types": sorted(types),
        "model_hint": hint,
        "buffer_device": str(dev),
        "diag": diag.path if diag is not None else None,
        "kernel": kernel,
        "kernel_reason": kernel_why,
        "fired_check": "armed" if local else "no local sites",
        "fired_at_capture": check.fired_at_capture,
        "exec_id": row.exec_id,
    }
    if edit is not None:
        rec["edit"] = getattr(edit, "label", type(edit).__name__)
    if writer is not None:
        rec["companion"] = {k: v for k, v in writer.items() if k not in ("undo_check", "check")}
    if loop is not None:
        rec["loop"] = {"per_loop": loop.per_loop, "num_loops": loop.num_loops,
                       "exec_steps": loop.steps, "loop_arg": loop.arg, "loop_kwarg": loop.kwarg,
                       "file_entry": "direction.N = execution step N-1"}
        rec["physical_layers"] = sorted({phys(i) for i in local})
    runner._weightless_steer_installed = rec
    runner._weightless_steer_model_id = id(runner.model)
    runner._weightless_steer_handles = handles
    runner._weightless_steer_check = check
    runner._weightless_steer_undo_check = undo_check
    runner._weightless_steer_companion_check = writer["check"] if writer is not None else None
    runner._weightless_steer_undo_companion_check = (writer["undo_check"] if writer is not None
                                                     else None)
    mdir = os.environ.get("WEIGHTLESS_STEER_MANIFEST_DIR", "").strip()
    if mdir:
        os.makedirs(mdir, exist_ok=True)
        mpath = os.path.join(mdir, f"weightless-manifest-{os.getpid()}.json")
        rec["manifest"] = mpath

        def write_manifest():
            tmp = mpath + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rec, f, indent=1)
            os.replace(tmp, mpath)

        write_manifest()
        check.on_capture = write_manifest
        if writer is not None:
            writer["check"].on_capture = write_manifest
    return rec
