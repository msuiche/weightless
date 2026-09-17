"""GLM-5.3 (glm5next) adapter: GLP steering on the materialized mHC stream.

`SteeredGlm5NextModel.forward` is upstream's `Glm5NextModel.forward` copied
verbatim with one block added in the layer loop, and
`SteeredGlm5NextDecoderLayer.forward` is upstream's copied verbatim with the
last-layer terminal contract removed. This is the coupling the design doc
accepts for path (1): the loop shape and the (hidden_states, residual,
post, comb) return convention are upstream internals that can change in any
release, but a break now fails loudly at import or first forward instead of
silently serving unsteered (the anchor-matched hotfix's failure mode). The
copies are pinned against patches/reference/glm5next.py by
tests/test_archs/test_glm5next.py.

Replaces patches/hotfix-glm53-steering-projective.py, with the same
steering semantics so the published GLP-44 vector
(GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf, layers 1-44) and its
alpha transfer without recalibration:

- glm5next carries the residual stream WIDENED to [T, n, hidden]
  (n = config.mhc_num_residual_streams; 4x4096 = 16384 on GLM-5.3-Flash —
  the GLP-44 derivation space), split across the layer's return tuple:
  hidden_states is the layer output [T, hidden] and residual/post/comb are
  the deferred hc_post mixes. Each layer's hc_post is fused into the next
  layer's pre (MHCFusedPostPreOp, documented upstream as exactly MHCPostOp
  + MHCPreOp), so materializing the parameter-free
  layer.hc_post(hidden_states, residual, post, comb) and letting the next
  layer take its standalone hc_pre path is the documented-equivalent
  decomposition, one kernel less fused. That materialized stream is the
  post-layer residual stream an HF forward hook sees — the
  glp.hook_point=residual_stream_post_layer site.
- The projection flattens the stream HC-outer to [T, n*hidden] (stream k
  occupies columns [k*hidden:(k+1)*hidden] — the derivation convention,
  carried over by analogy with the qwen38fn capture validated at cos
  0.9931; as the hotfix documents, NOT re-measured for GLM-5.3), steers via
  the core, reshapes back, and continues materialized (residual/post/comb
  cleared).
- L44, the deferred-contract caveat: stock Glm5NextDecoderLayer runs the
  LAST mHC layer's hc_post + hc_contract inside the decoder and returns the
  contracted [T, hidden] stream, which would escape steering — GLP-44
  covers layer 44, the last of 45. The adapter swaps every decoder layer's
  __class__ onto SteeredGlm5NextDecoderLayer, whose forward defers like
  every other layer; the model loop materializes, steers and contracts
  there instead.

Semantic differences from the hotfix, called out honestly:

- Alpha resolution is SteeringCore's: WEIGHTLESS_STEER_ALPHA, else the
  file's glp.alpha_default, else 1.0. The hotfix hardcoded 2.0 when the env
  was unset (GLP-44 is calibrated AT alpha 2.0; alpha >= 2.5 garbles this
  model, the cliff is abrupt). The published a2 file carries
  glp.alpha_default=2.0, so an env-unset boot matches the hotfix — but a
  vector file WITHOUT glp.alpha_default now lands on 1.0, not 2.0; set
  WEIGHTLESS_STEER_ALPHA=2.0 explicitly for those.
- Rank-k files are REFUSED (the hotfix QR-orthogonalized a rank-k basis on
  load; the plugin serves rank 1 only — GLP-44 is rank 1, unaffected).
- The container is GGUF-only (the hotfix also accepted a .pt
  {layer: tensor} dump), matching the other plugin lanes.

Parallelism: the widened stream is full-width and replicated at the layer
boundary (TP sharding is internal to attention/MoE and reduced before the
loop state), so the projection is per-token over the full n*hidden stream
on every rank with no collective — safe at TP4/TP2 (the 2x DGX Spark
lanes). PP is gated off upstream (no make_empty_intermediate_tensors).
Under sequence parallelism the in-loop stream is token-sharded, which the
scalar lane tolerates (the projection is per-token); per-request controls
are refused at construction on every arch (no runner binding exists), so
the token-ordinal misalignment SP would cause there is unreachable.

Non-mHC layers (MTP draft, the 70B fallback path) return post=None and are
never steered; GLM-5.3-Flash's 45 base layers are all mHC. The 743B
flagship is a different arch (GlmMoeDsaForCausalLM on deepseek_v2, a plain
single-stream residual) and is NOT covered by this adapter.

Upstream classes are imported from vllm.models.glm5next.nvidia.model — the
day-0 image layout the hotfix patches and the vendored reference mirrors.
Merged upstream main rehomed the same classes to
vllm.models.glm5next.common.model with identical forward bodies (checked
2026-09-17); update the import when the serving base moves.
"""
from __future__ import annotations

import torch
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.mhc import hc_contract, hc_expand
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.glm5next.nvidia.model import (
    Glm5NextDecoderLayer,
    Glm5NextForCausalLM,
    Glm5NextModel,
)
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


class SteeredGlm5NextDecoderLayer(Glm5NextDecoderLayer):
    """Glm5NextDecoderLayer with the last-layer terminal contract removed.

    Stock materializes the last mHC layer's hc_post and contracts it inside
    the decoder, returning the [T, hidden] stream — which would escape the
    model-loop projection (GLP-44 covers the last layer). This copy defers
    exactly like every other layer; the steered model forward materializes,
    steers and contracts there.
    """

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        post: torch.Tensor | None = None,
        comb: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # ---- upstream forward, verbatim (last-layer contract removed) -----
        # 70B or MTP layers: KDA + MoE without HC.
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(
                attn_output, residual=residual
            )
            hidden_states = self.mlp(hidden_states)
            if self.is_mtp_layer:
                # Return the unsummed pair: the MTP caller feeds it straight
                # into shared_head's fused_add_rms_norm (one kernel instead of
                # a separate residual-add + norm). The sum itself is unchanged
                # (fp32-accumulated inside the fused kernel).
                return hidden_states, residual, None, None
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        # mHC start. `post`/`comb` carry the previous layer's deferred
        # hc_post inputs (its ffn-pre outputs); when present, fuse that
        # hc_post with this layer's attn hc_pre into one kernel (inter-layer
        # fusion). Layer 0 has no incoming state -> standalone hc_pre.
        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                x = hc_expand(x, self.n)
            residual = x
            post, comb, x = self.hc_pre(
                x,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x,
                residual,
                post,
                comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )

        # Attention needs the full token sequence; mHC above ran on the SP
        # shard. Gather for attention, scatter back afterward (DSv4 pattern).
        if self.is_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        if self.is_sequence_parallel:
            x = sp_reduce_scatter(x)

        # Fuse post-attn hc_post + pre-FFN hc_pre (+ RMSNorm) into one kernel.
        residual, post, comb, x = self.hc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        # Fully Connected
        if self._mlp_is_moe:
            x = self.mlp(x, already_sequence_parallel=self.is_sequence_parallel)
        else:
            x = self.mlp(x)

        # [weightless-steer] the one removed block: stock's last mHC layer
        # runs hc_post + hc_contract here and returns the contracted stream,
        # which would escape steering. Defer like every other layer —
        # SteeredGlm5NextModel.forward materializes, steers and contracts.
        return x, residual, post, comb
        # ---- end upstream forward -----------------------------------------


class SteeredGlm5NextModel(Glm5NextModel, SteeredModelMixin):
    """Glm5NextModel with h <- h - alpha*(h.d)d applied per mHC decoder layer.

    glm5next uses vLLM's deferred mHC convention: each decoder layer returns
    (hidden_states, residual, post, comb) with its hc_post fused into the
    next layer's pre, so the post-layer stream at the loop is the
    parameter-free materialization layer.hc_post(hidden_states, residual,
    post, comb), [T, n, hidden]. The projection flattens it HC-outer to
    [T, n*hidden] — the width the GLP vector is derived at — steers via the
    core, and continues materialized (the deferred state is cleared, so the
    next layer takes its standalone hc_pre path). The last mHC layer is
    contracted with hc_contract after steering. Non-mHC layers return
    post=None and are skipped.
    """

    def _steer_mhc_stream(
        self,
        layer: Glm5NextDecoderLayer,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        """Project the materialized post-layer mHC stream at layer.layer_idx.

        Returns what the loop continues with: the steered stream (contracted
        to [T, hidden] on the last mHC layer, widened [T, n, hidden]
        otherwise) and the deferred hc_post state cleared.
        """
        stream = layer.hc_post(hidden_states, residual, post, comb)
        flat = stream.flatten(-2)
        hidden_states = self._steer_core.apply(
            layer.layer_idx, flat
        ).reshape(stream.shape)
        if layer.layer_idx == self.config.num_hidden_layers - 1:
            hidden_states = hc_contract(hidden_states, layer.n)
        return hidden_states, None, None, None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # ---- upstream forward, verbatim -----------------------------------
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            post = None
            comb = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            # post/comb (deferred mHC hc_post state) are not propagated across
            # PP ranks; the receiving rank's first mHC layer uses standalone pre.
            post = None
            comb = None

        full_num_tokens = positions.shape[0]
        if self.is_sequence_parallel:
            hidden_states = sp_shard(hidden_states)

        for layer in self._active_layers:
            hidden_states, residual, post, comb = layer(
                positions, hidden_states, residual, post, comb
            )
            # [weightless-steer] the one added block: unconditional per-layer
            # projection on the materialized post-layer mHC stream (zero
            # stack rows make it a numeric no-op on unsteered layers). The
            # `post is not None` guard is structural, not steering state:
            # mHC layers always return their deferred mixes and non-mHC
            # layers never do, so the traced graph is identical for every
            # layer set.
            if post is not None:
                hidden_states, residual, post, comb = self._steer_mhc_stream(
                    layer, hidden_states, residual, post, comb
                )

        if not get_pp_group().is_last_rank:
            # PP is gated off for GLM-5.3-Flash (no make_empty_intermediate_tensors),
            # so this branch is not exercised. post/comb are the deferred
            # hc_post state of this rank's last mHC layer; a future PP path
            # would need to propagate them, but for now they are dropped (the
            # receiving rank's first layer would fall back to standalone pre).
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if self.is_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        hidden_states = self.norm(hidden_states)
        return hidden_states
        # ---- end upstream forward -----------------------------------------


class SteeredGlm5NextForCausalLM(Glm5NextForCausalLM):
    """Glm5NextForCausalLM whose inner model applies GLP steering.

    Registered as a lazy shadow of the stock arch by plugin.register() when
    WEIGHTLESS_STEER_PATH is set. The multimodal wrapper
    (Glm5NextForConditionalGeneration) builds its language model through the
    registry, so the shadow covers it too. Every weight-loading and
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
        # Same __class__ swap idiom as nemotron_h, applied twice: the inner
        # model onto the steered model class, and every decoder layer onto
        # the class that defers the last layer's terminal contract. A
        # subclass with an identical (pure Python) layout is a legal
        # __class__ target, all buffers the steered forward needs are
        # registered by _wire_steering below, and weight loading happens
        # after __init__, so both swaps are in place before any checkpoint
        # tensors arrive.
        self.model.__class__ = SteeredGlm5NextModel
        for layer in self.model.layers:
            if isinstance(layer, Glm5NextDecoderLayer):
                layer.__class__ = SteeredGlm5NextDecoderLayer
        config = self.model.config
        scheduler_config = vllm_config.scheduler_config
        self.model._wire_steering(
            dtype=vllm_config.model_config.dtype,
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_num_reqs=scheduler_config.max_num_seqs,
            stream_width=config.hidden_size * config.mhc_num_residual_streams,
        )
        # The upstream constructor captures its bound forward in the compile
        # wrapper. Rebind after the class swap and buffer registration, before
        # warmup can compile or capture a stock, unsteered forward. glm5next
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
