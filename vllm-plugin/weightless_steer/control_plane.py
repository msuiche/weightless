"""Per-request control plane: request descriptors -> steering row tensors.

The half of per-request steering that has no vLLM in it. Given what the
engine scheduled for one forward step -- which requests, how many tokens
each, where each sits in its own sequence -- this builds the three tensors
``SteeringCore`` applies from:

``alpha_rows``   [max_num_tokens]                per-token alpha
``slot_rows``    [max_num_tokens] int32          per-token request slot
``layer_bank``   [max_num_reqs + 1, num_layers]  per-slot layer gate

Policy (what a request is allowed to ask for) lives in
``weightless_runtime.controls``; this module only maps decisions onto rows.
Engine glue -- reading req_ids and scheduled token counts out of whatever
shape the runner hands over -- stays at the call site, so this is testable
on CPU with no vLLM and no GPU.

Slot 0 is reserved for "no request": its gate row stays all-ones so a
padded or unassigned token steers exactly as the server is configured to.
Every buffer is rebuilt from its default each step, so a slot freed by a
finished request cannot leak its old control plan into whoever reuses it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from weightless_runtime.controls import (
    WeightlessEffectiveControl,
    WeightlessResolvedXArgs,
    effective_weightless_control,
    weightless_alpha_at,
)


@dataclass(frozen=True, slots=True)
class ScheduledRequest:
    """One request's slice of a scheduled forward step.

    Attributes:
        req_id: engine request id, for error messages only.
        resolved: the validated controls for this request (an empty
            ``WeightlessResolvedXArgs`` means "server defaults").
        token_count: tokens this request contributes to this step.
        start_ordinal: index, within this request's own sequence, of its
            first token in this step (vLLM's ``num_computed_tokens``).
        prompt_length: this request's prompt length, which is what makes a
            token prefill or decode.
    """

    req_id: str
    resolved: WeightlessResolvedXArgs
    token_count: int
    start_ordinal: int
    prompt_length: int


class WeightlessControlPlane:
    """Builds one step's control rows into reusable pinned buffers."""

    def __init__(self, *, max_num_tokens: int, max_num_reqs: int,
                 num_layers: int, dtype: torch.dtype = torch.float32,
                 default_alpha: float = 1.0,
                 loaded_layers: tuple[int, ...] | None = None):
        self.max_num_tokens = max_num_tokens
        self.max_num_reqs = max_num_reqs
        self.num_layers = num_layers
        self.dtype = dtype
        self.default_alpha = float(default_alpha)
        self.loaded_layers = (
            None if loaded_layers is None else tuple(sorted(loaded_layers))
        )
        pin = torch.cuda.is_available()
        self.alpha_rows = torch.empty(max_num_tokens, dtype=dtype,
                                      pin_memory=pin)
        self.slot_rows = torch.empty(max_num_tokens, dtype=torch.int32,
                                     pin_memory=pin)
        self.layer_bank = torch.empty(max_num_reqs + 1, num_layers,
                                      dtype=dtype, pin_memory=pin)
        self.reset()

    def reset(self) -> None:
        """Back to the defaults that reproduce the scalar lane exactly."""
        self.alpha_rows.fill_(self.default_alpha)
        self.slot_rows.zero_()
        self.layer_bank.fill_(1.0)

    def _effective(self, request: ScheduledRequest
                   ) -> WeightlessEffectiveControl:
        control = effective_weightless_control(
            request.resolved,
            default_alpha=self.default_alpha,
            default_layers=self.loaded_layers,
        )
        out_of_range = tuple(
            layer for layer in (control.layers or ())
            if not 0 <= layer < self.num_layers
        )
        if out_of_range:
            raise ValueError(
                f"request {request.req_id}: intervention layers "
                f"{list(out_of_range)} outside this model's depth "
                f"({self.num_layers} layers)"
            )
        # A mask naming a layer the vector never loaded would steer nothing
        # there while the response claims the mask was honoured.
        if (request.resolved.layers_override is not None
                and self.loaded_layers is not None):
            missing = sorted(
                set(request.resolved.layers_override)
                .difference(self.loaded_layers)
            )
            if missing:
                raise ValueError(
                    f"request {request.req_id}: intervention layers "
                    f"{missing} are not loaded by this steering vector"
                )
        return control

    def build(self, requests) -> int:
        """Fill the buffers for one step. Returns the token count written.

        Requests are laid out back to back in the order given, which is the
        order the engine flattens them into the batch.
        """
        self.reset()
        cursor = 0
        for index, request in enumerate(requests):
            slot = index + 1
            if slot > self.max_num_reqs:
                raise ValueError(
                    f"step schedules {slot} requests, capacity is "
                    f"{self.max_num_reqs}"
                )
            count = int(request.token_count)
            if count < 0:
                raise ValueError(
                    f"request {request.req_id}: negative token_count {count}"
                )
            if cursor + count > self.max_num_tokens:
                raise ValueError(
                    f"step schedules {cursor + count} tokens, capacity is "
                    f"{self.max_num_tokens}"
                )
            control = self._effective(request)
            for offset in range(count):
                self.alpha_rows[cursor + offset] = weightless_alpha_at(
                    control,
                    token_ordinal=request.start_ordinal + offset,
                    prompt_length=request.prompt_length,
                )
            self.slot_rows[cursor:cursor + count].fill_(slot)
            # Narrow the gate only when the REQUEST asked for a mask.
            # Falling back to the loaded-layer set here would look
            # equivalent -- unloaded layers have a zero stack row and are
            # already a numeric no-op -- but it would silently rewrite the
            # gate of every ordinary request, so a default request would no
            # longer be bit-identical to the scalar lane.
            if request.resolved.layers_override is not None:
                self.layer_bank[slot].zero_()
                if control.layers:
                    self.layer_bank[slot, list(control.layers)] = 1.0
            cursor += count
        return cursor

    def install(self, model) -> None:
        """Push the built rows into a steered model's buffers."""
        model.set_weightless_control_rows(
            self.alpha_rows, self.slot_rows, self.layer_bank,
        )
