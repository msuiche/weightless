"""SteeredModelMixin — wires a SteeringCore into an upstream model class.

One mixin, one contract: the arch adapter's inner-model subclass calls
``_wire_steering(dtype)`` once after construction, and its forward-loop
override calls ``_steer_post_layer`` after each decoder layer returns. The
adapter owns everything upstream-shaped (the forward signature, the layer
loop, the return convention); the mixin owns everything GLP-shaped.
"""
from __future__ import annotations

import torch

from weightless_runtime.controls import _enabled

from ..core import SteeringCore

#: Per-request controls are opt-in, server-side, exactly once.
_PER_REQUEST_ENV = "WEIGHTLESS_ENABLE_MILESTONE_2"


def _per_request_enabled() -> bool:
    """Whether this server registers per-request control buffers.

    Reuses the request parser's own truthiness test
    (weightless_runtime.controls) rather than restating it: two copies
    that disagree on, say, "on" would leave the server accepting control
    xargs it has no buffers to apply, or registering buffers no request
    can reach.
    """
    return _enabled(_PER_REQUEST_ENV)


class SteeredModelMixin:
    """GLP wiring for an inner model whose loop sees the residual stream.

    Class attribute STEER_HOOK names the one site the adapter implements;
    a vector gated for a different hook_point fails closed in
    SteeringCore.from_env.
    """

    STEER_HOOK = "residual_stream_post_layer"

    def _wire_steering(
        self,
        *,
        dtype: torch.dtype,
        max_num_tokens: int | None = None,
        max_num_reqs: int | None = None,
    ) -> None:
        """Load the vector named by WEIGHTLESS_STEER_PATH (if any) and
        register the steering buffers on this module.

        A disabled core (env unset) still registers zeroed buffers: the
        apply is unconditional in the forward loop, and the traced graph
        must be identical whether steering is on or off.
        """
        config = self.config
        # Per-request control geometry is registered only when the server
        # opted in AND the adapter passed the batch shape. Either missing
        # leaves the scalar lane exactly as it was: same buffers, same
        # apply, same traced graph.
        if not _per_request_enabled():
            max_num_tokens = max_num_reqs = None
        elif max_num_tokens is None or max_num_reqs is None:
            raise RuntimeError(
                "WEIGHTLESS_ENABLE_MILESTONE_2 is set but this arch adapter "
                "did not pass the batch geometry (max_num_tokens / "
                "max_num_reqs) to _wire_steering; refusing to accept "
                "per-request controls the model cannot apply."
            )
        geometry = {
            "max_num_tokens": max_num_tokens,
            "max_num_reqs": max_num_reqs,
        }
        core = SteeringCore.from_env(
            hook=self.STEER_HOOK,
            num_layers=len(self.layers),
            hidden_size=config.hidden_size,
            **geometry,
        )
        if core is None:
            core = SteeringCore.disabled(
                hook=self.STEER_HOOK,
                num_layers=len(self.layers),
                hidden_size=config.hidden_size,
                **geometry,
            )
        core.register_buffers(self, dtype)
        self._steer_core = core

    # -- per-request control plane -------------------------------------
    # The names the runner-side adapter binds to. Kept on the mixin (not
    # the core) so every arch lane exposes the identical contract.

    @property
    def weightless_per_request(self) -> bool:
        return self._steer_core.per_request

    @property
    def weightless_steer_layer_ids(self) -> tuple[int, ...]:
        """Global layer ids this model actually carries a direction for.

        The control plane refuses a request whose layer mask is not a
        subset of these -- a mask naming an unloaded layer would otherwise
        silently steer nothing there.
        """
        return tuple(sorted(self._steer_core.dirs))

    def set_weightless_control_rows(
        self,
        alpha_rows: torch.Tensor | None = None,
        slot_rows: torch.Tensor | None = None,
        layer_bank: torch.Tensor | None = None,
    ) -> None:
        self._steer_core.set_control_rows(alpha_rows, slot_rows, layer_bank)

    def reset_weightless_control_rows(self) -> None:
        self._steer_core.reset_control_rows()

    def _steer_post_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Project the post-layer residual stream at GLOBAL id `layer_idx`.

        For archs on vLLM's fused add+norm convention (nemotron_h,
        qwen3_next): the layer returns (mixer_output, residual) with the
        fold deferred to the next norm, so the post-layer stream is
        h = hidden_states + residual. Steering projects h and writes back
        ``hidden_states <- h' - residual`` so the next fold reproduces h'.
        """
        h = hidden_states + residual
        return self._steer_core.apply(layer_idx, h) - residual
