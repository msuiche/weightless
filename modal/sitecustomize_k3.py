"""weightless GLP steering bootstrap for the Kimi-K3 Modal serving lane.

Python's `site` module auto-imports `sitecustomize` from PYTHONPATH at EVERY
interpreter startup. That property is used deliberately on the K3 lane: the
engine runs as offline LLM() built inside torchrun-spawned workers
(VLLM_WORKER_MULTIPROC_METHOD=spawn — 16 fresh interpreters across 2 nodes),
where the plugin's own INFO log lines are unreliable as boot evidence (vllm's
dictConfig attaches a handler only to the `vllm` logger; third-party loggers
fall through to a WARNING root — the glm5next lane proved this on
2026-09-18). Without the "weightless GLP steering active" line there is no
boot evidence, and an unverifiable boot counts as unsteered.

This shim makes registration deterministic and its evidence logging-proof:
- runs weightless_steer.plugin.register() in every python process on the
  serve image (torchrun driver, EngineCore, workers — spawned processes
  included, since each is a fresh interpreter);
- prints markers DIRECTLY to stderr (the logging module is never involved);
- pre-parses the vector through SteeringCore.from_env and prints the
  steering-active summary — a bad/missing/gated vector kills EVERY process
  at startup (fail-closed), rather than a silent stock boot.

Silent skip (no marker) happens only in interpreters where vllm or
weightless_steer is not importable — Modal's own runtime python and pip/uv
helpers. Those never touch the model.

Wired in by cloud_serve_k3_plugin.py: the file lands at
/opt/weightless-shim/sitecustomize.py and K3Server prepends that directory
to PYTHONPATH of the torchrun process env (inherited by every worker).
Geometry is Kimi-K3's: 93 layers, 7168-wide plain residual stream.
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
                                 num_layers=93, hidden_size=7168)
    if core is not None:
        print(f"{_MARKER}: steering core loaded: "
              f"hook={core.hook} alpha={core.alpha:.3f} "
              f"layers={min(core.dirs)}..{max(core.dirs)} ({len(core.dirs)}) "
              f"width={core.hidden_size}", file=sys.stderr, flush=True)


_maybe_register()
