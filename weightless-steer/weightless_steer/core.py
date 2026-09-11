"""SteeringCore — the CUDA-graph-safe projective apply, shared by all archs.

Ports the discipline the hotfix fleet converged on (spec/GLP.md, "Apply, in
graph ops" and the vLLM implementation note):

- The direction stack is a **dense zero-padded tensor indexed by GLOBAL
  layer id** (correct under pipeline parallelism: each rank's forward loop
  visits its own layers and looks them up by the same global index; zero
  rows are a numeric no-op so the traced graph is identical for every
  layer set).
- Alpha is a **registered tensor buffer**, not a Python float — a float
  gets baked into the torch.compile cache as a graph constant (the cache
  key does not include these values), and alpha=0 would trace a graph
  without the op at all.
- The apply is **unconditional** — no Python ``if`` inside the traced
  region. Everything under ``@support_torch_compile`` runs its Python only
  at compile warmup.
- Buffers are **non-persistent** so ``load_weights`` never sees them as
  unexpected state-dict keys.

All four were measured failure modes on the hotfix lanes; they port as-is.
"""
from __future__ import annotations

import logging
import os

import torch

from .container import load_control_vector

logger = logging.getLogger(__name__)


class SteeringCore:
    """A loaded GLP vector plus the buffers it is applied from.

    Attributes:
        dirs: {global_layer_id: unit F32 direction}, width- and
            range-checked against the model. Empty for a disabled core.
        alpha: projection strength (a Python float here — it becomes a
            tensor at register_buffers time).
        hook: the GLP hook point this core was gated for.
        num_layers / hidden_size: model geometry the buffers are sized to.
    """

    def __init__(self, dirs, alpha, hook, num_layers, hidden_size, path=None):
        self.dirs = dirs
        self.alpha = float(alpha)
        self.hook = hook
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.path = path
        self._owner = None

    @classmethod
    def from_env(cls, *, hook, num_layers, hidden_size):
        """Build a core from WEIGHTLESS_STEER_*, or None if steering is off.

        Fail-closed: WEIGHTLESS_STEER_PATH set but the file missing,
        non-project, wrong hook, wrong width, out of range, or filtered to
        nothing raises — a boot asked for steering must not serve unsteered.
        """
        path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
        if not path:
            return None

        env_hook = (os.environ.get("WEIGHTLESS_STEER_HOOK") or hook).strip()
        if env_hook != hook:
            raise RuntimeError(
                f"WEIGHTLESS_STEER_HOOK={env_hook!r} is not implemented at "
                f"this arch adapter; the only site here is {hook}. Refusing "
                f"to serve with a silently wrong hook site."
            )

        try:
            meta, raw = load_control_vector(path, hook=hook)

            # This lane implements rank 1: one scalar alpha buffer per
            # model, one direction per layer. A rank-k (subspace) vector
            # needs per-direction alphas; serving only direction 0 would be
            # the silent partial apply the spec's refusal rule exists to
            # prevent.
            if any(v.dim() > 1 for v in raw.values()):
                raise RuntimeError(
                    f"{path}: rank-k (subspace) GLP vector — this serving "
                    f"lane implements rank 1 only. Refusing to serve a "
                    f"partially applied subspace."
                )

            alpha_env = os.environ.get("WEIGHTLESS_STEER_ALPHA", "").strip()
            alpha = (float(alpha_env) if alpha_env
                     else float(meta.get("glp.alpha_default", 1.0)))

            want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
            selected = (
                {int(t) for t in want.replace(" ", "").split(",") if t}
                if want else None
            )

            dirs = {}
            for layer_id, vec in raw.items():
                layer_id = int(layer_id)
                if selected is not None and layer_id not in selected:
                    continue
                vec = vec.detach().to(torch.float32).reshape(-1)
                # Width guard: a direction that does not match this arch's
                # stream width must fail here, not as an opaque broadcast
                # error at serve time.
                if vec.numel() != hidden_size:
                    raise RuntimeError(
                        f"steering vector layer {layer_id} width "
                        f"{vec.numel()} != {hidden_size} "
                        f"(hidden_size; plain single stream)"
                    )
                # The published vector ships unit directions; normalise
                # anyway so a non-unit export cannot silently scale alpha.
                dirs[layer_id] = vec / (vec.norm() + 1e-9)

            out_of_range = sorted(
                int(k) for k in raw
                if not 0 <= int(k) < num_layers
            )
            if out_of_range:
                raise RuntimeError(
                    f"{path}: direction layers {out_of_range} out of range "
                    f"for this model ({num_layers} layers)"
                )
            if not dirs:
                raise RuntimeError(
                    f"WEIGHTLESS_STEER_PATH={path} matched no layers; "
                    f"refusing to run unsteered"
                )
        except Exception as exc:
            # Fail closed: a boot asked for steering must not serve
            # unsteered.
            logger.error("GLP steering load failed (%s); failing closed",
                         exc)
            raise

        logger.info(
            "weightless GLP steering active: hook=%s alpha=%.3f "
            "layers=%d..%d (%d) width=%d",
            hook, alpha, min(dirs), max(dirs), len(dirs), hidden_size,
        )
        return cls(dirs, alpha, hook, num_layers, hidden_size, path=path)

    @classmethod
    def disabled(cls, *, hook, num_layers, hidden_size):
        """A core that steers nothing: empty stack rows, alpha 0.

        register_buffers still runs, so the traced graph is identical
        whether steering is on or off — the apply below is unconditional.
        """
        return cls({}, 0.0, hook, num_layers, hidden_size)

    def register_buffers(self, module, dtype):
        """Register (and fill) _steer_stack / _steer_alpha on `module`.

        Dense [num_layers, 1, hidden_size] stack indexed by global layer id
        plus a scalar alpha; both non-persistent so they never enter the
        state dict, which would make load_weights report them as unexpected
        keys. The core keeps a reference to the owning module and reads the
        buffers through it on every apply — nn.Module._apply (to/cuda)
        replaces buffer objects on device moves, so caching the tensors
        themselves would go stale.
        """
        module.register_buffer(
            "_steer_stack",
            torch.zeros(self.num_layers, 1, self.hidden_size, dtype=dtype),
            persistent=False,
        )
        module.register_buffer(
            "_steer_alpha",
            torch.zeros((), dtype=dtype),
            persistent=False,
        )
        for layer_id, vec in self.dirs.items():
            module._steer_stack[layer_id, 0] = vec.to(dtype)
        module._steer_alpha.fill_(self.alpha)
        self._owner = module

    def apply(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        """h <- h - alpha * (h . d) d at GLOBAL layer id `layer_idx`.

        Unconditional and branch-free: layers we do not steer have a zero
        stack row, making this a numeric no-op there while the traced graph
        stays identical for every layer set.
        """
        dirs = self._owner._steer_stack[layer_idx]
        coef = torch.einsum("...h,kh->...k", h, dirs)
        return h - self._owner._steer_alpha * torch.einsum(
            "...k,kh->...h", coef, dirs)
