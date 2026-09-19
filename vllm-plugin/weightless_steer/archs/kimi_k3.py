"""Kimi-K3 (kimi_k3) adapter: GLP steering on the post-layer accumulated stream.

`SteeredKimiLinearModel.forward` is upstream's `KimiLinearModel.forward`
copied verbatim with one block added in the layer loop — the projection at
the post-layer stream. This is the coupling the design doc accepts for path
(1): the loop shape and the (hidden_states, prefix_sum, residual) return
convention are upstream internals that can change in any release, but a break
now fails loudly at import or first forward instead of silently serving
unsteered (the anchor-matched hotfix's failure mode). The copy is pinned
against patches/reference/kimi_k3_nvidia_model.py by
tests/test_archs/test_kimi_k3.py.

Replaces the steering half of patches/hotfix-kimi-k3-steering-projective.py
(the runner probe gate/flush and the MLA DCP sentinel fix are capture-lane
and fork-infra concerns, not steering semantics; the serving lane still
applies the MLA fix at image build — see modal/cloud_serve_k3_plugin.py).
Same steering semantics, so the published GLP-92 vector
(glp.kimi-k3-GLP-92-L1-92-a1.gguf, layers 1-92 of 93) and its alpha transfer
without recalibration:

- kimi_k3 carries the residual stream PLAIN, [T, hidden] (7168 on Kimi-K3 —
  the GLP-92 derivation space); there is no hyper-connection widening and no
  flatten order to preserve. With attn_res on (Kimi-K3,
  attn_res_block_size=12) the post-layer accumulated stream at the loop is
  prefix_sum + hidden_states — the attn_res block side stream rides in
  `residual` — and without attn_res it is hidden_states + residual. The
  hotfix's capture probe measured exactly this point (STATE.md §4), and it is
  the glp.hook_point=residual_stream_post_layer site.
- The projection is the mixin's: h <- h - alpha*(h.d)d with the side stream
  passed as the write-back base, so hidden_states <- h' - prefix_sum
  (attn_res) or h' - residual (plain), and the loop continues steered. The
  use_attn_res branch is a structural attribute fixed at model build — it is
  resolved at trace time, so the traced graph is identical either way.
- The aux_hidden_states capture (EAGLE/MTP) reads prefix_sum + hidden_states
  AFTER the steering block, exactly the hotfix's ordering: aux taps see the
  steered stream.
- No last-layer trap: unlike glm5next there is no deferred terminal
  contract — every layer returns the same 3-tuple and the loop's final fold
  (attn_res or hidden_states + residual) runs after steering.

Semantic differences from the hotfix, called out honestly:

- The hotfix applied the projection under `if layer_idx in self._steer_layers`;
  the plugin applies it unconditionally from a dense zero-padded stack (zero
  rows are a numeric no-op) so the traced graph is identical for every layer
  set. Same numbers, one less Python branch in the loop.
- Alpha resolution is SteeringCore's: WEIGHTLESS_STEER_ALPHA, else the file's
  glp.alpha_default, else 1.0. The hotfix defaulted to 1.0 and the published
  a1 file carries glp.alpha_default=1.0, so an env-unset boot matches.
- Rank-k files are REFUSED (the hotfix never saw one; the plugin serves rank
  1 only — GLP-92 is rank 1, unaffected).
- The container is GGUF-only. The vector repo also ships
  glp.kimi-k3.dirs.pt ({layer: tensor} — the hotfix's input format); the
  plugin does NOT read it. Use the GGUF.

Parallelism: the stream is full-width and replicated at the layer boundary
(TP sharding is internal to KDA/MLA attention and KimiMoE and reduced before
the layer returns), so the projection is per-token over the full 7168-wide
stream on every rank with no collective — the validated 2x H200:8 PP2xTP8
shape. PP is native upstream (make_empty_intermediate_tensors, and
self.layers is full-length with PPMissingLayer placeholders, so the dense
stack stays indexed by GLOBAL layer id on every stage); the boundary crossing
(hidden_states + prefix_sum folded into intermediate tensors) is upstream's,
untouched. Under sequence parallelism the in-loop stream is token-sharded,
which the scalar lane tolerates (the projection is per-token); per-request
controls are refused at construction on every arch, so the token-ordinal
misalignment SP would cause there is unreachable.

The multimodal wrapper (KimiK3ForConditionalGeneration — moonshotai/Kimi-K3's
registered arch) builds its language model through the registry with
architectures=["KimiLinearForCausalLM"], so shadowing that name covers both
the plain text model and the wrapper. Upstream classes are imported from
vllm.models.kimi_k3.nvidia.model — the day-0 image layout the hotfix patches
and the vendored reference mirrors.
"""
from __future__ import annotations

import torch
import vllm.envs as envs
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.kimi_k3.nvidia.model import (
    KimiLinearForCausalLM,
    KimiLinearModel,
)
from vllm.models.kimi_k3.nvidia.ops import attn_res
from vllm.models.kimi_k3.nvidia.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


class SteeredKimiLinearModel(KimiLinearModel, SteeredModelMixin):
    """KimiLinearModel with h <- h - alpha*(h.d)d applied per decoder layer.

    kimi_k3's convention: each decoder layer returns (hidden_states,
    prefix_sum, residual). With attn_res on, the post-layer accumulated
    stream is prefix_sum + hidden_states; without it, hidden_states +
    residual. _steer_post_layer projects that stream with the side stream as
    the write-back base (hidden_states <- h' - prefix_sum, resp. h' -
    residual), so the next layer's fold reproduces the steered stream.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
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
        assert hidden_states is not None

        aux_hidden_states: list[torch.Tensor] = []
        if self.start_layer in self.aux_hidden_state_layers:
            if self.use_attn_res or residual is None:
                aux_hidden_states.append(hidden_states)
            else:
                aux_hidden_states.append(hidden_states + residual)

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            assert residual is None, "Currently, SP is not supported with PP"

        prefix_sum = None
        if self.use_attn_res:
            block_residual = hidden_states.new_empty(
                hidden_states.size(0),
                self.num_attn_res_blocks,
                hidden_states.size(1),
            )
            if residual is not None:
                block_residual[:, : residual.size(1), :].copy_(residual)
            prefix_sum = hidden_states
            hidden_states = None
            residual = block_residual

        for layer_idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer],
            start=self.start_layer,
        ):
            hidden_states, prefix_sum, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                prefix_sum=prefix_sum,
                residual=residual,
            )
            # [weightless-steer] the one added block: unconditional per-layer
            # projection on the post-layer accumulated stream (zero stack rows
            # make it a numeric no-op on unsteered layers). The use_attn_res
            # branch is structural (fixed at model build), resolved at trace
            # time: with attn_res the stream is prefix_sum + hidden_states and
            # prefix_sum is the write-back base; otherwise the side stream is
            # residual. Sits before the aux tap, matching the hotfix's
            # ordering (aux hidden states see the steered stream).
            if self.use_attn_res:
                hidden_states = self._steer_post_layer(
                    layer_idx, hidden_states, prefix_sum
                )
            else:
                hidden_states = self._steer_post_layer(
                    layer_idx, hidden_states, residual
                )
            if (layer_idx + 1) in self.aux_hidden_state_layers:
                if self.use_attn_res:
                    assert prefix_sum is not None
                    aux_hidden_state = prefix_sum + hidden_states
                else:
                    assert residual is not None
                    aux_hidden_state = hidden_states + residual

                if self.use_sequence_parallel:
                    # Gather SP-sharded aux hidden states.
                    # TODO: Optimize this.
                    aux_hidden_state = sp_all_gather(aux_hidden_state)
                    aux_hidden_state = aux_hidden_state[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)

        assert hidden_states is not None
        assert residual is not None
        if not get_pp_group().is_last_rank:
            assert not self.use_sequence_parallel, (
                "Currently, SP is not supported with PP"
            )
            if prefix_sum is not None:
                hidden_states = hidden_states + prefix_sum
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.use_attn_res:
            assert prefix_sum is not None
            hidden_states = attn_res(
                prefix_sum,
                hidden_states,
                residual,
                self.output_attn_res_norm.weight,
                self.output_attn_res_proj.weight.squeeze(0),
                None,
                num_blocks=self.num_attn_res_blocks,
                block_write_idx=-1,
                eps=self.output_attn_res_norm.variance_epsilon,
                output_norm_eps=0.0,
            )
        else:
            hidden_states = hidden_states + residual

        if self.use_sequence_parallel:
            # Gather SP-sharded hidden states.
            hidden_states = sp_all_gather(hidden_states)
            hidden_states = hidden_states[:full_num_tokens]

        # NOTE: the final norm is applied in compute_logits instead of here, so
        # the MTP draft model receives the pre-norm hidden states.
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
        # ---- end upstream forward -----------------------------------------


class SteeredKimiLinearForCausalLM(KimiLinearForCausalLM):
    """KimiLinearForCausalLM whose inner model applies GLP steering.

    Registered as a lazy shadow of the stock arch by plugin.register() when
    WEIGHTLESS_STEER_PATH is set. The multimodal wrapper
    (KimiK3ForConditionalGeneration) builds its language model through the
    registry, so the shadow covers it too. Every weight-loading and interface
    method is inherited untouched.
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
        self.model.__class__ = SteeredKimiLinearModel
        scheduler_config = vllm_config.scheduler_config
        self.model._wire_steering(
            dtype=vllm_config.model_config.dtype,
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_num_reqs=scheduler_config.max_num_seqs,
        )
        # The upstream constructor captures its bound forward in the compile
        # wrapper. Rebind after the class swap and buffer registration, before
        # warmup can compile or capture a stock, unsteered forward. kimi_k3
        # carries no @support_torch_compile today (do_not_compile is unset),
        # so this is a no-op until upstream decorates the class.
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
