"""Nemotron-H adapter: GLP steering by subclassing, replacing the hotfix.

`SteeredNemotronHModel.forward` is upstream's `NemotronHModel.forward`
copied verbatim with one line added in the layer loop — the projection at
the post-layer residual stream. This is the coupling the design doc
accepts for path (1): the loop shape and the (hidden_states, residual)
return convention are upstream internals that can change in any release,
but a break now fails loudly at import or first forward instead of
silently serving unsteered (the anchor-matched hotfix's failure mode).
The copy is pinned against patches/reference/nemotron_h_v0280.py by
tests/test_archs/test_nemotron_h.py and was checked identical in the local
v0.27 tree (vllm/model_executor/models/nemotron_h.py @ dspark-steering-v027).
"""
from __future__ import annotations

from itertools import islice

import torch
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.nemotron_h import (
    NemotronHForCausalLM,
    NemotronHModel,
)
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


class SteeredNemotronHModel(NemotronHModel, SteeredModelMixin):
    """NemotronHModel with h <- h - alpha*(h.d)d applied per decoder layer.

    nemotron_h uses vLLM's fused add+norm convention: each decoder layer
    returns (mixer_output, residual) with the fold deferred to the NEXT
    layer's norm, so the post-layer stream at the loop is
    h = hidden_states + residual; _steer_post_layer writes back
    hidden_states <- h' - residual so the next fold reproduces h'.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # ---- upstream forward, verbatim -----------------------------------
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            # [weightless-steer] the one added line: unconditional per-layer
            # projection at the post-layer residual stream (zero stack rows
            # make it a numeric no-op on unsteered layers).
            hidden_states = self._steer_post_layer(
                self.start_layer + idx, hidden_states, residual
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm_f(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
        # ---- end upstream forward -----------------------------------------


class SteeredNemotronHForCausalLM(NemotronHForCausalLM):
    """NemotronHForCausalLM whose inner model applies GLP steering.

    Registered as a lazy shadow of the stock arch by plugin.register() when
    WEIGHTLESS_STEER_PATH is set; every weight-loading and interface method
    is inherited untouched.
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        if _per_request_enabled():
            raise RuntimeError(
                "WEIGHTLESS_ENABLE_MILESTONE_2 requires request validation and "
                "runner integration, which this plugin does not implement. "
                "Unset it to serve with scalar steering."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Swap the already-constructed inner model onto the steered class
        # rather than rebuilding it: make_layers would allocate the whole
        # layer stack a second time. A subclass with an identical (pure
        # Python) layout is a legal __class__ target, and all buffers the
        # steered forward needs are registered by _wire_steering below.
        # Weight loading happens after __init__, so the swapped class is in
        # place before any checkpoint tensors arrive.
        self.model.__class__ = SteeredNemotronHModel
        scheduler_config = vllm_config.scheduler_config
        self.model._wire_steering(
            dtype=vllm_config.model_config.dtype,
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_num_reqs=scheduler_config.max_num_seqs,
        )
        # The upstream constructor captures its bound forward in the compile
        # wrapper. Rebind after the class swap and buffer registration, before
        # warmup can compile or capture a stock, unsteered forward.
        if not getattr(self.model, "do_not_compile", True):
            from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

            # Drop the first init's dynamo bytecode hook before registering
            # the second: register_bytecode_hook appends to a process-global
            # dict and the wrapper only ever removes the handle it last
            # stored, so re-initialising without this leaves a hook behind
            # for the life of the process.
            cleanup = getattr(TorchCompileWithNoGuardsWrapper, "cleanup", None)
            if cleanup is not None:
                cleanup(self.model)
            TorchCompileWithNoGuardsWrapper.__init__(
                self.model,
                compile_prefix=self.model._compile_prefix,
                is_encoder=self.model._is_encoder,
            )
