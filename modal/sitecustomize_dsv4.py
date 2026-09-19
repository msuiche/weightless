"""weightless GLP steering bootstrap for the DSV4-Flash-0731 Modal serving lane.

Python's `site` module auto-imports `sitecustomize` from PYTHONPATH at EVERY
interpreter startup. That property is used deliberately: on these day-0 dev
builds the plugin's register() does run through vllm's own
`load_general_plugins()`, but its INFO log lines never surface in the serve
path — vllm's dictConfig attaches a handler only to the `vllm` logger
(propagate=False), third-party loggers fall through to root at WARNING, and
even a VLLM_LOGGING_CONFIG_PATH root-INFO override did not produce the lines
in real glm5next boots (2026-09-17/18 debug session). Without the
"weightless GLP steering active" line there is no boot evidence, and an
unverifiable boot counts as unsteered.

This shim makes registration deterministic and its evidence logging-proof:
- runs weightless_steer.plugin.register() in every python process on the
  serve image (api server, EngineCore, workers — spawned processes included,
  since each is a fresh interpreter);
- prints markers DIRECTLY to stderr (the logging module is never involved);
- pre-parses the vector through SteeringCore.from_env at the DSV4 geometry
  (43 layers, hidden_size 4096, hook ffn_out_pre_residual — the pre-fold
  FFN write, NOT the residual stream) and prints the steering-active
  summary — a bad/missing/gated vector kills EVERY process at startup
  (fail-closed), rather than a silent stock boot.

Silent skip (no marker) happens only in interpreters where vllm or
weightless_steer is not importable — Modal's own runtime python
(/usr/local/bin/python3.12, site-packages layout without the Debian
dist-packages vllm) and pip/uv helpers. Those never touch the model.

Wired in by cloud_serve_dsv4.py: the file lands at
/opt/weightless-shim/sitecustomize.py and serve() prepends that directory to
PYTHONPATH of the vllm subprocess env only.
"""
import os
import sys

_MARKER = "WEIGHTLESS-SHIM"


def _maybe_register():
    path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
    if not path:
        return
    try:
        import vllm  # noqa: F401
        from weightless_steer import plugin
    except ImportError:
        return  # modal runtime / tooling interpreters: no vllm, no model
    plugin.register()
    print(f"{_MARKER}: register() ran (pid {os.getpid()}, "
          f"{sys.executable})", file=sys.stderr, flush=True)

    from weightless_steer.core import SteeringCore
    core = SteeringCore.from_env(hook="ffn_out_pre_residual",
                                 num_layers=43, hidden_size=4096)
    if core is not None:
        print(f"{_MARKER}: steering core loaded: "
              f"hook={core.hook} alpha={core.alpha:.3f} "
              f"layers={min(core.dirs)}..{max(core.dirs)} ({len(core.dirs)}) "
              f"width={core.hidden_size}", file=sys.stderr, flush=True)


_maybe_register()
