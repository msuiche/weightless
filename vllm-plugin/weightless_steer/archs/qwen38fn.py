"""Qwen3.8-Flash-Next (qwen38fn) adapter: GLP steering on the materialized
delayed-combine hyper-connection stream.

`SteeredQwen3_8FlashNextModel.forward` is upstream's
`Qwen3_8FlashNextModel.forward` copied verbatim with one block added in the
layer loop and the final mixer guarded for it. This is the coupling the
design doc accepts for path (1): the loop shape and the (hidden_states,
block_output, injection) delayed-combine return convention are upstream
internals that can change in any release, but a break now fails loudly at
import or first forward instead of silently serving unsteered (the
anchor-matched hotfix's failure mode). The copy is pinned against
patches/reference/qwen3_8_flash_next.py by tests/test_archs/test_qwen38fn.py.

Replaces patches/hotfix-qwen38fn-steering-projective.py, with the same
steering semantics so the published GLP-47 vector
(Qwen3.8-Flash-Next-abliterated-cyber-GLP-47-L1-47-a1.gguf, layers 1-47,
alpha=1.0 calibrated) transfers without recalibration:

- qwen38fn carries the residual stream WIDENED and FLAT:
  [T, hc_count*hidden] = [T, 10240] (4x2560, HC-outer / H-inner — the GLP-47
  derivation space, reproduced on this serving stack at cos 0.9931 by the
  capture lane). Decoder layers return (hidden_states, block_output,
  injection) with the layer's MLP output still PENDING; the post-layer
  stream is the parameter-free mlp_hyper_connection.combine() of the three.
  The apply materializes per layer, steers the materialized stream via the
  core (no flatten needed — the stream is already the derivation-width
  vector), and continues materialized (pending combine consumed): the next
  layer takes its mix() path, mathematically the same input, one kernel
  less fused.
- The final-mixer guard IS this arch's deferred-contract trap: stock ends
  the loop with the last layer's combine still pending and the final mixer
  consumes it via combine_and_mix(). The unconditional per-layer apply
  consumes it first, so the mixer must take its mix() path (same input for
  an already-materialized state — multi_hidden is the materialized
  multi-stream itself). Unguarded, combine_and_mix(h, None, None) would
  crash or corrupt the one layer GLP-47 needs most: layer 47, the last of
  48.
- The deepstack branch (multimodal only) already materializes the state
  before its external addition, so the apply reads hidden_states directly
  when block_output is None. The `is None` guard is structural, not
  steering state: the traced graph is identical for every layer set.

Semantic differences from the hotfix, called out honestly:

- Alpha resolution is SteeringCore's: WEIGHTLESS_STEER_ALPHA, else the
  file's glp.alpha_default, else 1.0. The hotfix defaulted to 1.0 and the
  shipping a1 file is calibrated AT alpha 1.0 (1.5+ over-projects), so an
  env-unset boot matches the hotfix whether or not the file carries the
  key.
- Rank-k files are REFUSED (the hotfix QR-orthogonalized a rank-k basis on
  load; the plugin serves rank 1 only — GLP-47 is rank 1, unaffected).
- The container is GGUF-only (the hotfix also accepted a .pt
  {layer: tensor} dump), matching the other plugin lanes.

Parallelism: the widened stream is full-width and replicated at the layer
boundary (TP sharding is internal to attention/MoE/PLE and reduced before
the loop state), so the projection is per-token over the full
hc_count*hidden stream on every rank with no collective — validated at TP2
(2x DGX Spark lane) and TP8/TP1 (the capture and NVFP4-serve lanes). The
steer stack is dense and indexed by GLOBAL layer id (the loop's islice
enumerates the full layers list), correct under PP; the PP transport
contract is satisfied because the apply leaves the state materialized,
which is exactly what the transport combines-and-sends. Sequence-parallel
MoE is refused by upstream itself for this arch (NotImplementedError in
the layer constructor).

Serving classes: the RadixArk NVFP4 checkpoint declares
architectures=["Qwen4ExpForConditionalGeneration"], so BOTH wrapper
classes get a steered subclass here and plugin.SHADOWED_ARCHS registers
all three day-0 names. Unlike glm5next, the multimodal wrapper builds its
language model by DIRECTLY instantiating Qwen3_8FlashNextForCausalLM (not
through the registry), so shadowing the CausalLM name alone would leave
the actually-served multimodal arch unsteered — the wrapper subclass swaps
its language_model.model onto the steered inner class instead.

Upstream classes are imported from vllm.models.qwen3_8_flash_next.nvidia.model —
the day-0 image layout (vllm/vllm-openai:qwen38-flash-next, fork
1CatAI/1Cat-vLLM) that the hotfix patches and the vendored reference
mirrors. The PR branch later rehomed the same arch to
vllm.models.qwen4_exp with the classes renamed Qwen4Exp*; update the
import when the serving base moves (the PLE patch already probes both).
"""
from __future__ import annotations

from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.models.qwen3_8_flash_next.nvidia.model import (
    Qwen3_8FlashNextForCausalLM,
    Qwen3_8FlashNextForConditionalGeneration,
    Qwen3_8FlashNextModel,
)
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


class SteeredQwen3_8FlashNextModel(Qwen3_8FlashNextModel, SteeredModelMixin):
    """Qwen3_8FlashNextModel with h <- h - alpha*(h.d)d per decoder layer.

    qwen38fn uses vLLM's delayed-combine hyper-connection convention: each
    decoder layer returns (hidden_states, block_output, injection) with its
    MLP output pending, so the post-layer stream at the loop is the
    parameter-free layer.mlp_hyper_connection.combine(hidden_states,
    block_output, injection), [T, hc_count*hidden] — already the flat
    derivation-width stream. The projection applies on it directly and the
    loop continues materialized (deferred state cleared); the final mixer
    is guarded onto its mix() path for the consumed combine.
    """

    def _steer_delayed_combine(
        self,
        layer_idx: int,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor | None,
        injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None, None]:
        """Project the materialized post-layer stream at GLOBAL layer_idx.

        Returns what the loop continues with: the steered stream and the
        deferred combine state cleared, so the next layer takes its
        standalone mix() path (same math, one kernel less fused).
        """
        if block_output is None:
            # deepstack already materialized the state (multimodal only)
            steer_stream = hidden_states
        else:
            steer_stream = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
        return self._steer_core.apply(layer_idx, steer_stream), None, None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # ---- upstream forward, verbatim -----------------------------------
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        block_output = None
        injection = None
        last_layer = None
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            last_layer = layer
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                prev_block_output=block_output,
                prev_injection=injection,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                deepstack_embed = (
                    deepstack_embed.unsqueeze(-2)
                    .expand(
                        *deepstack_embed.shape[:-1],
                        self.config.hc_count,
                        self.config.hidden_size,
                    )
                    .flatten(-2)
                )
                # Deepstack is an external addition to the materialized
                # multi-stream state and therefore terminates delayed combine.
                hidden_states = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                block_output = None
                injection = None
                hidden_states = hidden_states + deepstack_embed
            # [weightless-steer] the one added block: unconditional per-layer
            # projection on the materialized post-layer multi-stream
            # [T, hc_count*hidden] (zero stack rows make it a numeric no-op
            # on unsteered layers). The `block_output is None` guard is
            # structural, not steering state: it fires only when the
            # deepstack branch above already materialized the state, so the
            # traced graph is identical for every layer set.
            hidden_states, block_output, injection = (
                self._steer_delayed_combine(
                    layer_idx, layer, hidden_states, block_output, injection
                )
            )

        if not get_pp_group().is_last_rank:
            # PP transports one tensor, not the delayed HC tuple. Materialize
            # with the HC module that produced the pending injection.
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return IntermediateTensors({"hidden_states": hidden_states})

        # The final mixer consumes the last pending combine and returns both
        # the sampled single stream and the materialized multi-stream state.
        final_mixer = self.hyper_connection_mixer
        assert final_mixer is not None
        # [weightless-steer] mixer guard: the unconditional apply consumed
        # the last layer's pending combine, so the mixer takes its mix()
        # path — the same input path for an already-materialized state
        # (multi_hidden is the materialized multi-stream itself). Stock's
        # unconditional combine_and_mix would re-combine None.
        if block_output is not None:
            multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        else:
            multi_hidden, sample_hidden_states, _ = final_mixer.mix(
                hidden_states
            )
        if self._mtp_hidden_buffer is not None:
            # Capture the pre-final-mixer multi-stream hidden state
            # [T, hc_count*H] for the MTP drafter (zero extra compute:
            # this tensor is needed by the final mixer regardless).
            num_tokens = multi_hidden.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states
        # ---- end upstream forward -----------------------------------------


def _steer_inner_model(inner: Qwen3_8FlashNextModel, vllm_config) -> None:
    """Swap an already-constructed inner model onto the steered class and
    wire the GLP core. Shared by both serving-class shadows.

    Same __class__ swap idiom as nemotron_h: a subclass with an identical
    (pure Python) layout is a legal __class__ target, all buffers the
    steered forward needs are registered by _wire_steering, and weight
    loading happens after __init__, so the swap is in place before any
    checkpoint tensors arrive.
    """
    inner.__class__ = SteeredQwen3_8FlashNextModel
    config = inner.config
    scheduler_config = vllm_config.scheduler_config
    inner._wire_steering(
        dtype=vllm_config.model_config.dtype,
        max_num_tokens=scheduler_config.max_num_batched_tokens,
        max_num_reqs=scheduler_config.max_num_seqs,
        stream_width=config.hidden_size * config.hc_count,
    )
    # The upstream constructor captures its bound forward in the compile
    # wrapper (Qwen3_8FlashNextModel IS @support_torch_compile-decorated, so
    # this rebind is live on every compiled boot, unlike glm5next's dormant
    # one). Rebind after the class swap and buffer registration, before
    # warmup can compile or capture a stock, unsteered forward.
    if not getattr(inner, "do_not_compile", True):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        # Drop the first init's dynamo bytecode hook before registering the
        # second: register_bytecode_hook appends to a process-global dict
        # and the wrapper only ever removes the handle it last stored, so
        # re-initialising without this leaves a hook behind for the life of
        # the process.
        cleanup = getattr(TorchCompileWithNoGuardsWrapper, "cleanup", None)
        if cleanup is not None:
            cleanup(inner)
        TorchCompileWithNoGuardsWrapper.__init__(
            inner,
            compile_prefix=inner._compile_prefix,
            is_encoder=inner._is_encoder,
        )


def _refuse_per_request() -> None:
    if _per_request_enabled():
        raise RuntimeError(
            "WEIGHTLESS_ENABLE_MILESTONE_2 requires request validation and "
            "runner integration, which this plugin does not implement. "
            "Unset it to serve with scalar steering."
        )


class SteeredQwen3_8FlashNextForCausalLM(Qwen3_8FlashNextForCausalLM):
    """Qwen3_8FlashNextForCausalLM whose inner model applies GLP steering.

    Registered as a lazy shadow of the stock arch by plugin.register() when
    WEIGHTLESS_STEER_PATH is set. Every weight-loading and interface method
    is inherited untouched.
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        _refuse_per_request()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _steer_inner_model(self.model, vllm_config)


class SteeredQwen3_8FlashNextForConditionalGeneration(
    Qwen3_8FlashNextForConditionalGeneration
):
    """The multimodal wrapper, with its language model's inner model steered.

    This is the arch the RadixArk NVFP4 checkpoint actually resolves to
    (architectures=["Qwen4ExpForConditionalGeneration"]). The wrapper builds
    self.language_model by directly instantiating Qwen3_8FlashNextForCausalLM
    — not through the registry — so shadowing the CausalLM name alone would
    leave this arch unsteered; the swap happens here instead, on
    self.language_model.model, after the upstream __init__ completes.
    """

    def __init__(self, *, vllm_config, prefix: str = "model"):
        _refuse_per_request()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _steer_inner_model(self.language_model.model, vllm_config)
