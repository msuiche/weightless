"""SteeredModelMixin — wires a SteeringCore into an upstream model class.

One mixin, one contract: the arch adapter's inner-model subclass calls
``_wire_steering(dtype)`` once after construction, and its forward-loop
override calls ``_steer_post_layer`` after each decoder layer returns. The
adapter owns everything upstream-shaped (the forward signature, the layer
loop, the return convention); the mixin owns everything GLP-shaped.
"""
from __future__ import annotations

import torch

from ..core import SteeringCore


class SteeredModelMixin:
    """GLP wiring for an inner model whose loop sees the residual stream.

    Class attribute STEER_HOOK names the one site the adapter implements;
    a vector gated for a different hook_point fails closed in
    SteeringCore.from_env.
    """

    STEER_HOOK = "residual_stream_post_layer"

    def _wire_steering(self, *, dtype: torch.dtype) -> None:
        """Load the vector named by WEIGHTLESS_STEER_PATH (if any) and
        register the steering buffers on this module.

        A disabled core (env unset) still registers zeroed buffers: the
        apply is unconditional in the forward loop, and the traced graph
        must be identical whether steering is on or off.
        """
        config = self.config
        core = SteeringCore.from_env(
            hook=self.STEER_HOOK,
            num_layers=len(self.layers),
            hidden_size=config.hidden_size,
        )
        if core is None:
            core = SteeringCore.disabled(
                hook=self.STEER_HOOK,
                num_layers=len(self.layers),
                hidden_size=config.hidden_size,
            )
        core.register_buffers(self, dtype)
        self._steer_core = core

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
