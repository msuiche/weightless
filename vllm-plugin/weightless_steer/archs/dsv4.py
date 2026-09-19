"""DeepSeek-V4 (deepseek_v4) adapter: GLP steering on the pre-fold FFN write.

`SteeredDeepseekV4Model.forward` is upstream's `DeepseekV4Model.forward`
copied verbatim with one line added in the layer loop — the projection on
``hidden_states``, which at this anchor is the layer's PENDING FFN WRITE,
shape (tokens, hidden_size): the decoder layer returns
``(ffn_out, residual, post_mix, res_mix)`` and the mHC fold of that write
into the hc_mult streams is deferred to the next layer's fused
mhc_fused_post_pre call. This is the coupling the design doc accepts for
path (1): the loop shape and the 4-tuple return convention are upstream
internals that can change in any release, but a break now fails loudly at
import or first forward instead of silently serving unsteered (the
anchor-matched hotfix's failure mode). The copy is pinned against
patches/reference/deepseek_v4_nvidia_model.py by
tests/test_archs/test_dsv4.py.

Replaces patches/hotfix-dsv4-steering-projective.py, with the same steering
semantics so the published GLP-29 vector
(DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4.gguf, layers
10-38, width 4096) and its alpha transfer without recalibration:

- The hook is ``ffn_out_pre_residual``, NOT the post-layer residual stream
  — the 2026-09-04 site correction. An earlier hotfix revision was labelled
  residual_stream_post_layer but measured at this anchor (the model loop's
  hidden_states IS the pending FFN write; shape-probe verified against the
  0.25.2 mHC kernel signatures). Every published vLLM-lane DSV4 number was
  measured here, and the measured site ordering on DSV4-Flash-0731 is
  FFN >> residual >> attention (FFN window alpha 4-6; the residual site
  garbles at 4.0), so the anchor stays. A residual_stream_post_layer file
  is the mismatch case: SteeringCore.from_env fails closed on it.
- DSV4 is a hyper-connection arch (hc_mult=4) but the steering site is NOT
  the widened stream: the pending FFN write is the contracted single-stream
  (tokens, hidden_size), so the GLP directions are hidden_size-wide. A
  hc-widened vector fails the width gate — that file belongs to the
  residual-site experiment (20260905-dsv4-residual-glp), not this lane.
- The apply is one unconditional line in the model loop,
  ``hidden_states = self._steer_core.apply(idx, hidden_states)`` with idx
  the GLOBAL layer id (the loop enumerates from start_layer), before the
  aux-hidden-state reconstruction — the hotfix's exact block order, so
  eagle3 aux taps see the steered stream, same as the hotfix lane.

Semantic differences from the hotfix, called out honestly:

- Alpha resolution is SteeringCore's: WEIGHTLESS_STEER_ALPHA, else the
  file's glp.alpha_default, else 1.0. The hotfix hardcoded 1.0 when the env
  was unset and deployments pinned 4.0 (keysdir) or 6.0 (GLP-29 cyber,
  2026-09-04). The published GLP-29 file carries glp.alpha_default=6.0 (the
  rig's serving dose), so an env-unset boot now lands on 6.0 — set
  WEIGHTLESS_STEER_ALPHA=4.0 explicitly to reproduce the BENCHMARK.md
  GLP-29 reference row (refusal32 0/32 -> 19/32 at alpha 4.0).
- GLP-29 is a TRANSFERRED vector (glp.derived_at=residual_stream_post_layer,
  glp.hook_point=ffn_out_pre_residual): legal, and the loader warns loudly,
  exactly as the hotfix did.
- Rank-k files are REFUSED (the hotfix QR-orthogonalized a rank-k basis on
  load; the plugin serves rank 1 only — GLP-29 is rank 1, unaffected).
- The container is GGUF-only (the hotfix also accepted a .pt
  {layer: tensor} dump), matching the other plugin lanes.
- WEIGHTLESS_STEER_HOOK set to anything but ffn_out_pre_residual fails
  closed, as in the hotfix.

Parallelism: the pending FFN write is full-width and replicated at the
layer boundary (TP sharding is internal to attention/MoE and reduced before
the layer returns), so the projection is per-token over the full
hidden_size stream on every rank with no collective — validated at TP4
(H100:4). PP is supported upstream (make_empty_intermediate_tensors); the
direction stack is dense, zero-padded and indexed by global layer id, so
each rank steers exactly its own layers. Under sequence parallelism the
in-loop stream is token-sharded, which the scalar lane tolerates (the
projection is per-token); per-request controls are refused at construction
on every arch (no runner binding exists), so the token-ordinal misalignment
SP would cause there is unreachable.

No last-layer trap: unlike glm5next, every DSV4 decoder layer defers
identically (the final mhc_post fold lives in the model loop, after the
last steering apply), so no decoder-layer __class__ swap is needed.

The MTP drafter (DeepSeekV4MTP) and the DSpark draft model
(DSparkDraftModel) are separate registry archs and are NOT shadowed — they
are constructed only when speculative decoding is enabled, which the
reference eval lane does not use. The multimodal wrapper
(DeepseekV4ForConditionalGeneration, the Vision-Exp arch) builds its
language model through the registry with
architectures=["DeepseekV4ForCausalLM"], so the shadow covers it too.

Upstream classes are imported from vllm.models.deepseek_v4.nvidia.model —
the day-0 image layout (vllm/vllm-openai:deepseekv4-flash-vision, the tag
the 20260905 capture/eval lane validated on H100:4; the same module path
exists in the v0.27 tree). The vendored reference is a byte-identical pull
from that image (md5 f897a3354a9ac7508c20be7c5d5f7d63).
"""
from __future__ import annotations

from itertools import islice

import torch
import vllm.envs as envs
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_shard,
)
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


class SteeredDeepseekV4Model(DeepseekV4Model, SteeredModelMixin):
    """DeepseekV4Model with h <- h - alpha*(h.d)d applied per decoder layer.

    DSV4 uses vLLM's deferred-mHC convention: each decoder layer returns
    (ffn_out, residual, post_mix, res_mix) with the fold of ffn_out into
    the hyper-connection streams deferred to the next layer's fused
    post/pre call, so hidden_states at the loop is the layer's pending FFN
    write, (tokens, hidden_size) — the glp.hook_point=ffn_out_pre_residual
    site. The projection applies to it directly (NOT h + residual: the
    stream fold has not happened yet and the pending write is the measured
    site); the next layer's fused post/pre then folds the steered write.
    """

    STEER_HOOK = "ffn_out_pre_residual"

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # ---- upstream forward, verbatim -----------------------------------
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        residual, post_mix, res_mix = None, None, None
        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None  # avoid duplicate mhc_post call
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix = layer(
                hidden_states,
                positions,
                input_ids,
                post_mix,
                res_mix,
                residual,
            )
            # [weightless-steer] the one added line: unconditional per-layer
            # projection on the layer's pending FFN write (the
            # ffn_out_pre_residual site; zero stack rows make it a numeric
            # no-op on unsteered layers). Before the aux-hidden-state
            # reconstruction, matching the hotfix's block order.
            hidden_states = self._steer_core.apply(idx, hidden_states)
            if idx + 1 in self.aux_hidden_state_layers:
                # Reconstruct the aux hidden state for draft models
                aux_recon = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )
                aux_hidden_state = aux_recon.mean(dim=1)
                if self.use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon
        if layer is not None:
            # Reuse if the last layer was captured as an aux hidden state
            if self.end_layer in self.aux_hidden_state_layers:
                hidden_states = final_aux_recon
            else:
                hidden_states = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        hidden_states = hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
        # ---- end upstream forward -----------------------------------------


class SteeredDeepseekV4ForCausalLM(DeepseekV4ForCausalLM):
    """DeepseekV4ForCausalLM whose inner model applies GLP steering.

    Registered as a lazy shadow of the stock arch by plugin.register() when
    WEIGHTLESS_STEER_PATH is set. The multimodal wrapper
    (DeepseekV4ForConditionalGeneration) builds its language model through
    the registry, so the shadow covers it too. Every weight-loading and
    interface method is inherited untouched.
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
        self.model.__class__ = SteeredDeepseekV4Model
        scheduler_config = vllm_config.scheduler_config
        self.model._wire_steering(
            dtype=vllm_config.model_config.dtype,
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_num_reqs=scheduler_config.max_num_seqs,
        )
        # The upstream constructor captures its bound forward in the compile
        # wrapper. Rebind after the class swap and buffer registration, before
        # warmup can compile or capture a stock, unsteered forward. The day-0
        # image's DeepseekV4Model carries no @support_torch_compile
        # (do_not_compile is unset), so this is a no-op until upstream
        # decorates the class.
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
