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

    def __init__(self, dirs, alpha, hook, num_layers, hidden_size, path=None,
                 max_num_tokens=None, max_num_reqs=None):
        self.dirs = dirs
        self.alpha = float(alpha)
        self.hook = hook
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.path = path
        # Per-request geometry. None on both = the scalar lane: one alpha
        # for the whole batch, the original apply, byte-identical graph.
        if (max_num_tokens is None) != (max_num_reqs is None):
            raise ValueError(
                "per-request geometry is both-or-neither: got "
                f"max_num_tokens={max_num_tokens!r}, "
                f"max_num_reqs={max_num_reqs!r}"
            )
        self.max_num_tokens = max_num_tokens
        self.max_num_reqs = max_num_reqs
        self._owner = None

    @property
    def per_request(self) -> bool:
        """True when this core carries per-token alpha / per-request masks.

        Decided once at construction, never from tensor data, so the branch
        in apply() is resolved at trace time and the captured graph is the
        same on every step.
        """
        return self.max_num_tokens is not None

    @classmethod
    def from_env(cls, *, hook, num_layers, hidden_size,
                 max_num_tokens=None, max_num_reqs=None):
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
        return cls(dirs, alpha, hook, num_layers, hidden_size, path=path,
                   max_num_tokens=max_num_tokens, max_num_reqs=max_num_reqs)

    @classmethod
    def disabled(cls, *, hook, num_layers, hidden_size,
                 max_num_tokens=None, max_num_reqs=None):
        """A core that steers nothing: empty stack rows, alpha 0.

        register_buffers still runs, so the traced graph is identical
        whether steering is on or off — the apply below is unconditional.
        """
        return cls({}, 0.0, hook, num_layers, hidden_size,
                   max_num_tokens=max_num_tokens, max_num_reqs=max_num_reqs)

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

        if self.per_request:
            # Per-token alpha, per-token request slot, per-request layer
            # gate. All three are PRE-FILLED with the values that reproduce
            # the scalar lane exactly: alpha rows hold the server alpha and
            # the gate is all-ones. A step where nothing writes them (no
            # control plumbing attached, a warmup/profile forward, padded
            # rows past the batch) therefore steers exactly as this core
            # would without per-request support -- the failure direction is
            # "steered as configured", never "silently unsteered".
            #
            # Slot 0 is the no-request slot and its gate row is all-ones for
            # the same reason.
            module.register_buffer(
                "_steer_alpha_rows",
                torch.full((self.max_num_tokens,), self.alpha, dtype=dtype),
                persistent=False,
            )
            module.register_buffer(
                "_steer_slot_rows",
                torch.zeros(self.max_num_tokens, dtype=torch.long),
                persistent=False,
            )
            # LAYER-MAJOR [num_layers, max_num_reqs + 1], matching
            # _steer_stack's "indexed by global layer id first" shape. The
            # transpose is not cosmetic: gating as bank[layer_idx][slots]
            # keeps layer_idx an ordinary integer index, which dynamo
            # generalises over, whereas bank[slots, layer_idx] makes it
            # part of an advanced index and forces a recompile per layer
            # (measured: 6 graphs for 6 layers vs 2).
            module.register_buffer(
                "_steer_layer_bank",
                torch.ones(self.num_layers, self.max_num_reqs + 1,
                           dtype=dtype),
                persistent=False,
            )
        self._owner = module

    def set_control_rows(self, alpha_rows=None, slot_rows=None,
                         layer_bank=None):
        """Install one step's per-request control plan.

        Copies IN PLACE (`copy_`) rather than rebinding: a CUDA graph
        captures buffer addresses, so replacing the tensor objects would
        leave the captured graph reading the old memory. Short rows are
        written at the front and the tail keeps its pre-filled default, so
        a partial write still steers the untouched rows as configured.

        Raises on a core that was not built for per-request control -- the
        caller asked for something this lane cannot honour, and silently
        ignoring it would serve an unsteered-or-miscalibrated request while
        reporting success.
        """
        if not self.per_request:
            raise RuntimeError(
                "this SteeringCore was built without per-request control "
                "geometry (max_num_tokens/max_num_reqs); refusing to accept "
                "a control plan it cannot apply"
            )
        owner = self._owner
        if owner is None:
            raise RuntimeError("set_control_rows before register_buffers")
        for name, value, default in (
            ("_steer_alpha_rows", alpha_rows, self.alpha),
            ("_steer_slot_rows", slot_rows, 0),
            ("_steer_layer_bank", layer_bank, 1),
        ):
            if value is None:
                continue
            target = getattr(owner, name)
            if value.shape == target.shape:
                target.copy_(value)
                continue
            if value.dim() != target.dim():
                raise ValueError(
                    f"{name}: plan has {value.dim()} dims, buffer has "
                    f"{target.dim()}"
                )
            if any(v > t for v, t in zip(value.shape, target.shape)):
                raise ValueError(
                    f"{name}: plan {tuple(value.shape)} exceeds buffer "
                    f"{tuple(target.shape)}"
                )
            # Reset first, then write the front. Without the reset the tail
            # would keep the PREVIOUS step's plan -- a shorter batch would
            # inherit the last batch's alphas and masks on the rows it does
            # not cover.
            target.fill_(default)
            target[tuple(slice(0, v) for v in value.shape)].copy_(value)

    def reset_control_rows(self):
        """Return every control buffer to its scalar-lane default."""
        if not self.per_request or self._owner is None:
            return
        self._owner._steer_alpha_rows.fill_(self.alpha)
        self._owner._steer_slot_rows.zero_()
        self._owner._steer_layer_bank.fill_(1.0)

    def apply(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        """h <- h - alpha * (h . d) d at GLOBAL layer id `layer_idx`.

        Unconditional and branch-free: layers we do not steer have a zero
        stack row, making this a numeric no-op there while the traced graph
        stays identical for every layer set.
        """
        owner = self._owner
        dirs = owner._steer_stack[layer_idx]
        coef = torch.einsum("...h,kh->...k", h, dirs)
        proj = torch.einsum("...k,kh->...h", coef, dirs)
        if not self.per_request:
            return h - owner._steer_alpha * proj
        # Per-request lane. `self.per_request` is a Python constant fixed at
        # construction, so this branch is resolved when the region is traced
        # and the captured graph still contains exactly one apply.
        #
        # h is the flattened [num_tokens, hidden] batch, so row i is token i
        # of the concatenated requests: alpha comes from the per-token row
        # and the layer gate from this token's request slot. Both index the
        # front of a fixed-size buffer, which keeps every shape in the graph
        # a function of the captured token count alone.
        if h.dim() != 2:
            # Resolved at trace time (h.dim() is static), so this never
            # becomes a branch inside a captured graph. A 3-D [batch, seq,
            # hidden] stream would index rows by BATCH here and silently
            # steer every request at the first request's alpha.
            raise RuntimeError(
                f"per-request steering needs a flattened [num_tokens, "
                f"hidden] stream; this adapter passed {h.dim()} dims "
                f"({tuple(h.shape)})"
            )
        n = h.shape[0]
        alpha = owner._steer_alpha_rows[:n].unsqueeze(-1)
        gate = owner._steer_layer_bank[layer_idx][
            owner._steer_slot_rows[:n]
        ].unsqueeze(-1)
        return h - alpha * gate * proj
