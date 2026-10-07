"""What an architecture row is, and the registries of the special install
handlers.

What differs between architectures is data, not code: one ``ArchRow`` per
served model class says where the steering site is and how to read it, and
``install.install_steering`` is generic. Each module in this package defines
the rows of one architecture (``ROWS``); ``archs/__init__.py`` collects them
into ``ARCH`` and lists the classes the plugin refuses by name.

A row whose site is not the decoder layer's output tuple names a special
handler (``install = "special:<name>"``). A handler is registered with
``@register_special(name, site=...)`` and is called once per steered local
layer:

    handler(layer, site, *, layer_id, backbone, row) -> undo

``layer`` is the decoder-layer module, ``site`` is the steering function
from ``core.make_site`` (``site(x, r=None) -> x'``; it also does the fired
count, the refusals, the kernel choice and the diagnostics), and ``undo``
is a callable (or an object with ``.remove()``) that takes the handler's
wrapping off again. A handler must wrap an INSTANCE attribute, never a
class, and must raise if the attribute it wraps is missing or has another
shape than the row expects: the boot then stops. ``site`` is the text
written to the log line and the manifest.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

HOOK_POINT = "residual_stream_post_layer"

INSTALL_MODES = ("hook", "forward_wrap")  # plus "special:<name>" (below)
EXEC_IDS = ("layer", "looped")
TP_MODES = ("ok", "refuse")


@dataclasses.dataclass(frozen=True)
class ArchRow:
    """How one served model class is steered.

    backbone        candidate attribute paths from ``runner.model`` to the
                    module that owns ``.layers`` (indexed by GLOBAL layer
                    id), ``.start_layer``, ``.end_layer`` and ``.config``;
                    tried in order ("" is the model itself). A path that
                    resolves to None means this rank holds no backbone.
    layers          exact decoder-layer class names (never isinstance).
    hooks           the ``glp.hook_point`` values this row can serve.
    width           config -> the width of the steered stream.
    arity, hidden_index, residual_index
                    the layer's output tuple: its length, the slot that is
                    edited and the slot that holds the residual added to it
                    by the next layer (None: the edited slot is the whole
                    stream). Every other slot passes through unchanged.
    residual_may_be_none
                    the residual slot may hold None (the edited slot is
                    then the whole stream); otherwise the site's first
                    forward refuses a None there.
    per_stream      the site's stream is [T, streams, width].
    install         "hook" (a forward hook on the layer), "forward_wrap"
                    (the layer instance's forward is wrapped) or
                    "special:<name>" (a registered handler, see above).
    exec_id         "layer" (one direction per decoder layer) or "looped"
                    (one per execution step of a looped model).
    tp              "ok" or "refuse" (TP > 1 fails the boot).
    hint            ``controlvector.model_hint`` values accepted besides the
                    model's own ``model_type``.
    kind            optional: decoder layer -> "full" or "linear", for rows
                    whose attention kind is not in the class name (the log
                    line's split and the manifest's ``full_attn_layers``).
    loop            looped rows only: config -> LoopSpec (layers per loop
                    pass, passes, and where the loop index is in the
                    layer's call); see nanbeige.py.
    """

    backbone: tuple
    layers: frozenset
    hooks: frozenset
    width: Callable
    arity: int
    hidden_index: int
    residual_index: Optional[int]
    residual_may_be_none: bool = False
    per_stream: bool = False
    install: str = "hook"
    exec_id: str = "layer"
    tp: str = "refuse"
    hint: frozenset = frozenset()
    kind: Optional[Callable] = None
    loop: Optional[Callable] = None


def hidden_size(cfg):
    return int(cfg.hidden_size)


SPECIAL = {}


def register_special(name, *, site):
    def deco(fn):
        if name in SPECIAL:
            raise RuntimeError(f"weightless: special install handler {name!r} registered twice")
        SPECIAL[name] = (fn, site)
        return fn

    return deco


# Row checks that run once, before anything is hooked, for rows whose special
# handler has one: ``preflight(*, runner, backbone, meta, path, row, row_name)``
# raises to fail the boot (for example a model or a file the row refuses).
PREFLIGHT = {}


def register_preflight(name):
    def deco(fn):
        if name in PREFLIGHT:
            raise RuntimeError(f"weightless: preflight for {name!r} registered twice")
        PREFLIGHT[name] = fn
        return fn

    return deco
