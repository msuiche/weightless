"""Inkling adapter: GLP steering on the flushed post-layer residual stream.

`SteeredInklingModel.forward` is upstream's `InklingModel.forward` (vLLM
v0.28.0, vllm/models/inkling/nvidia/model.py) copied verbatim with one block
added in the layer loop — flush the deferred MLP add, then the projection at
the materialized post-layer stream. This is the coupling the design doc
accepts for path (1): the loop shape and the (hidden_states, pending) return
convention are upstream internals that can change in any release, but a break
now fails loudly at import or first forward instead of silently serving
unsteered (the anchor-matched hotfix's failure mode). The copy is pinned
against patches/reference/inkling_v0280.py by tests/test_archs/test_inkling.py.

Replaces patches/hotfix-inkling-steering-projective.py, with the same
steering semantics so the published GLP-41 vector
(Inkling-Small-abliterated-cyber-GLP-41-L1-41-a0.25.gguf, layers 1-41) and
its alpha transfer without recalibration:

- Inkling DEFERS the MLP residual add: each decoder layer called with
  defer_mlp_add=True returns (hidden_states, pending) where pending carries
  this layer's pre-reduce, pre-sconv MLP delta (an InklingDelta — a plain
  tensor, or the MoE (routed, shared) partial pair), fused into the NEXT
  layer's sconv+add+rmsnorm (_sconv_add_norm). The true post-layer residual
  stream at iteration i is only materialized by flushing pending — the same
  call the file's own PP-boundary branch uses. The adapter flushes after
  every layer, projects the materialized [T, hidden] stream via the core,
  and continues with pending=None: the next layer takes its pending=None
  path (attn_norm of the materialized stream) — mathematically the same
  input, one kernel less fused, the tradeoff the hotfix documents. The flush
  is unconditional (pending is structurally non-None in this loop: every
  layer defers), so the traced graph does not depend on the layer set.
- NO hyper-connection widening: the stream is plain [T, hidden_size] (4096
  on Inkling-Small) — the width the GLP vector is derived at.
- Layer ids are GLOBAL: the loop enumerates the PP slice and indexes the
  steer stack by start_layer + offset (InklingDecoderLayer takes layer_id
  but does not store it).

Semantic differences from the hotfix, called out honestly:

- Alpha resolution is SteeringCore's: WEIGHTLESS_STEER_ALPHA, else the
  file's glp.alpha_default, else 1.0. The hotfix defaulted to 1.0 when the
  env was unset — and alpha 1.0 GARBLES Inkling (28/32 garbled at refusal32
  in the dose ladder; 0.5 garbles everything). The published a0.25 file
  carries glp.alpha_default=0.25, so an env-unset boot lands on the
  calibrated dose — but a vector file WITHOUT glp.alpha_default now lands on
  1.0, not 0.25; set WEIGHTLESS_STEER_ALPHA=0.25 explicitly for those.
  Never exceed 0.25 on this arch.
- Rank-k files are REFUSED (the hotfix QR-orthogonalized a rank-k basis on
  load; the plugin serves rank 1 only — GLP-41 is rank 1, unaffected).
- The container is GGUF-only (the hotfix also accepted a .pt
  {layer: tensor} dump), matching the other plugin lanes.
- The hotfix's DSPARK_PROBE capture lane is not ported: capture stays on
  the hotfix lane, the plugin is the serving lane.

Parallelism: _sconv_add_norm reduce-scatters/all-gathers internally, so the
materialized stream is full hidden width on every TP rank and the projection
is per-token over the full stream with no collective — safe at TP4 (the
validated 4xH100 NVFP4 lane). PP is supported upstream
(make_empty_intermediate_tensors carries only "hidden_states"; the rank's
last pending is flushed before the boundary — unchanged here, since the
loop's flush already leaves pending=None). Per-request controls are refused
at construction on every arch (no runner binding exists).

Entry classes: the registry shadows BOTH InklingForCausalLM and
InklingForConditionalGeneration. Unlike glm5next, the multimodal wrapper
does NOT build its language model through the registry — both entry classes
construct InklingModel directly in _TmlForCausalLMBase._build — so shadowing
only the text class would silently serve multimodal checkpoints unsteered,
and thinkingmachines/Inkling-Small-NVFP4 (GLP-41's own base) resolves to
InklingForConditionalGeneration. The swap-and-wire is identical for both;
every weight-loading and interface method is inherited untouched.
"""
from __future__ import annotations

import torch
from vllm.distributed import get_pp_group
from vllm.models.inkling.nvidia.model import (
    InklingDelta,
    InklingForCausalLM,
    InklingForConditionalGeneration,
    InklingModel,
    InklingShortConv,
    _sconv_add_norm,
    compute_log_scaling_tau,
    embed_rmsnorm,
)
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled

_PER_REQUEST_REFUSAL = (
    "WEIGHTLESS_ENABLE_MILESTONE_2 requires request validation and "
    "runner integration, which this plugin does not implement. "
    "Unset it to serve with scalar steering."
)


class SteeredInklingModel(InklingModel, SteeredModelMixin):
    """InklingModel with h <- h - alpha*(h.d)d applied per decoder layer.

    Inkling defers each layer's MLP residual add into the next layer's fused
    sconv+add+rmsnorm, so the post-layer stream at the loop only exists after
    flushing `pending` — the exact call the PP-boundary branch uses. The one
    added block flushes, projects the materialized [T, hidden] stream at the
    GLOBAL layer id, and continues with pending=None (the next layer takes
    its standalone attn_norm path — same math, one kernel less fused). With
    the flush unconditional, pending is always None at loop exit: the stock
    PP/final-norm fused-flush branches are kept verbatim but never fire, and
    the forward returns self.norm(hidden_states) — the same decomposition
    the hotfix lane validated at alpha 0.25.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # ---- upstream forward, verbatim -----------------------------------
        attn_in0: torch.Tensor | None = None
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                # embed_norm was already applied when producing inputs_embeds.
                hidden_states = inputs_embeds
            else:
                # Gather + embed_norm + the first layer's attn_norm, one launch.
                norm = self.embed_norm
                hidden_states, attn_in0 = embed_rmsnorm(
                    input_ids,
                    self.embed_tokens.weight,
                    norm.weight if norm is not None else None,
                    self.config.rms_norm_eps,
                    chain_weight=self.layers[self.start_layer].attn_norm.weight,
                )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        log_scaling = None
        if self.config.log_scaling_n_floor is not None:
            log_scaling = compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )

        pending: tuple[InklingDelta, InklingShortConv] | None = None
        for _wl_off, layer in enumerate(
            self.layers[self.start_layer : self.end_layer]
        ):
            hidden_states, pending = layer(
                positions,
                hidden_states,
                pending=pending,
                defer_mlp_add=True,
                attn_in=attn_in0,
                log_scaling=log_scaling,
            )
            attn_in0 = None
            # [weightless-steer] the one added block: materialize the
            # post-layer stream by flushing the deferred MLP add (the same
            # _sconv_add_norm call the PP-boundary branch below uses), steer
            # it at the GLOBAL layer id, and continue with pending=None.
            # Unconditional: pending is structurally non-None here (every
            # layer defers), so the traced graph is identical for every
            # layer set; zero stack rows make the apply a numeric no-op on
            # unsteered layers.
            if pending is not None:
                hidden_states = _sconv_add_norm(
                    pending[0], hidden_states, pending[1], None, positions
                )[1]
                pending = None
            hidden_states = self._steer_core.apply(
                self.start_layer + _wl_off, hidden_states
            )

        if not get_pp_group().is_last_rank:
            if pending is not None:
                hidden_states = _sconv_add_norm(
                    pending[0], hidden_states, pending[1], None, positions
                )[1]
            return IntermediateTensors({"hidden_states": hidden_states})
        if pending is not None:
            # Final RS/sconv/AG + residual add fused with the final rmsnorm.
            norm_out = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.norm, positions
            )[0]
            assert norm_out is not None
            return norm_out
        return self.norm(hidden_states)
        # ---- end upstream forward -----------------------------------------


def _swap_and_wire(outer, vllm_config) -> None:
    """Swap the built InklingModel onto the steered class and wire the core.

    Shared by both entry classes: _TmlForCausalLMBase._build constructs
    self.model identically in each. The __class__ swap idiom is the one
    nemotron_h documents — a subclass with an identical (pure Python) layout
    is a legal target, all buffers the steered forward needs are registered
    by _wire_steering, and weight loading happens after __init__, so the
    swap is in place before any checkpoint tensors arrive.
    """
    outer.model.__class__ = SteeredInklingModel
    scheduler_config = vllm_config.scheduler_config
    outer.model._wire_steering(
        dtype=vllm_config.model_config.dtype,
        max_num_tokens=scheduler_config.max_num_batched_tokens,
        max_num_reqs=scheduler_config.max_num_seqs,
    )
    # The upstream constructor captures its bound forward in the compile
    # wrapper. Rebind after the class swap and buffer registration, before
    # warmup can compile or capture a stock, unsteered forward. inkling
    # carries no @support_torch_compile today (do_not_compile is unset), so
    # this is a no-op until upstream decorates the class.
    if not getattr(outer.model, "do_not_compile", True):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        # Drop the first init's dynamo bytecode hook before registering the
        # second: register_bytecode_hook appends to a process-global dict and
        # the wrapper only ever removes the handle it last stored, so
        # re-initialising without this leaves a hook behind for the life of
        # the process.
        cleanup = getattr(TorchCompileWithNoGuardsWrapper, "cleanup", None)
        if cleanup is not None:
            cleanup(outer.model)
        TorchCompileWithNoGuardsWrapper.__init__(
            outer.model,
            compile_prefix=outer.model._compile_prefix,
            is_encoder=outer.model._is_encoder,
        )


class SteeredInklingForCausalLM(InklingForCausalLM):
    """InklingForCausalLM whose inner model applies GLP steering.

    Registered as a lazy shadow of the stock arch by plugin.register() when
    WEIGHTLESS_STEER_PATH is set; every weight-loading and interface method
    is inherited untouched.
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        if _per_request_enabled():
            raise RuntimeError(_PER_REQUEST_REFUSAL)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _swap_and_wire(self, vllm_config)


class SteeredInklingForConditionalGeneration(InklingForConditionalGeneration):
    """InklingForConditionalGeneration whose inner model applies GLP steering.

    The multimodal entry class — the one thinkingmachines/Inkling-Small-NVFP4
    actually resolves to. The text backbone (and its steering) is identical;
    the vision/audio towers are untouched side branches.
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        if _per_request_enabled():
            raise RuntimeError(_PER_REQUEST_REFUSAL)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _swap_and_wire(self, vllm_config)
