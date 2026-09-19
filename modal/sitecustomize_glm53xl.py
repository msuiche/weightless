"""weightless GLP steering bootstrap for the GLM-5.3 743B Modal serving lane.

Python's `site` module auto-imports `sitecustomize` from PYTHONPATH at EVERY
interpreter startup. That property is used deliberately: vLLM's dictConfig
attaches a handler only to the `vllm` logger (propagate=False), third-party
loggers fall through to root at WARNING, so the plugin's INFO lines (the
"weightless GLP steering active" boot evidence) may never render even when
steering works. Without that line there is no boot evidence, and an
unverifiable boot counts as unsteered.

This shim makes registration deterministic and its evidence logging-proof:
- runs weightless_steer.plugin.register() in every python process on the
  serve image (api server, EngineCore, workers — spawned processes included,
  since each is a fresh interpreter);
- prints markers DIRECTLY to stderr (the logging module is never involved);
- pre-parses the vector through SteeringCore.from_env at the glm53xl
  geometry (78 layers, plain 6144-wide stream) and prints the
  steering-active summary — a bad/missing/gated vector kills EVERY process
  at startup (fail-closed), rather than a silent stock boot.

Silent skip (no marker) happens only in interpreters where vllm or
weightless_steer is not importable — Modal's own runtime python
(/usr/local/bin/python3.12, site-packages layout without the Debian
dist-packages vllm) and pip/uv helpers. Those never touch the model.

Wired in by cloud_serve_glm53xl.py: the file lands at
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
    core = SteeringCore.from_env(hook="residual_stream_post_layer",
                                 num_layers=78, hidden_size=6144)
    if core is not None:
        print(f"{_MARKER}: steering core loaded: "
              f"hook={core.hook} alpha={core.alpha:.3f} "
              f"layers={min(core.dirs)}..{max(core.dirs)} ({len(core.dirs)}) "
              f"width={core.hidden_size}", file=sys.stderr, flush=True)


_maybe_register()
