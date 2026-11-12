"""Qwen3.8-27B (qwen3_5) adapter: GLP steering on the decomposed stream.

`SteeredQwen3_5Model.forward` is upstream's `Qwen3NextModel.forward` (vLLM
0.28.0, which `Qwen3_5Model` inherits unchanged) copied verbatim with one
block added in the layer loop — the projection at the post-layer residual
stream. This is the coupling the design doc accepts for path (1): the loop
shape and the (hidden_states, residual) return convention are upstream
internals that can change in any release, but a break now fails loudly at
import or first forward instead of silently serving unsteered (the
anchor-matched hotfix's failure mode). The copy is pinned against the
hotfix's v0.28.0 anchor text by tests/test_archs/test_qwen38.py, and the
Modal preflight re-checks the same anchor against the image's own
qwen3_next.py before any GPU spend.

Replaces patches/hotfix-qwen38-steering-projective.py, with the same
steering semantics so the published GLP-49 vector
(Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf, layers 10-58, width
5120, alpha_default 1.0) and its alpha transfer without recalibration:

- qwen3_5 uses vLLM's fused add+norm convention: each decoder layer returns
  (mixer_output, residual) with the fold deferred to the NEXT layer's norm,
  so the post-layer stream at the loop is h = hidden_states + residual. The
  derivation measured the FULL post-layer stream (HF layer outputs), so the
  apply steers the sum and writes back hidden_states <- h' - residual so the
  next fold reproduces h' (SteeredModelMixin._steer_post_layer). The mixers
  differ per layer — 48 gated-delta-net linear_attention layers whose
  out_proj writes into the residual and 16 full-attention layers whose
  o_proj does — but both write into the same stream and the direction is
  mixer-indifferent (refusal-research/QWEN38-27B.md 5.0.1), so one loop-site
  projection covers both.
- Hybrid-state note: the gated delta net's conv/SSM state is recurrent
  state, not residual stream; steering the stream between layers leaves it
  untouched, same as the hotfix and the transformers reference harness.

Semantic differences from the hotfix, called out honestly:

- None on alpha: the hotfix's env-unset default was already 1.0 and the
  published file carries glp.alpha_default=1.0, so SteeringCore's
  resolution (WEIGHTLESS_STEER_ALPHA, else the file's, else 1.0) agrees with
  the hotfix in every case. alpha=2 INVERTS this model (reflection installs
  refusal; 37.5% benign refusal — QWEN38-27B.md 5.0.8); the shipped alpha is
  1.0.
- Rank-k files are REFUSED (the hotfix QR-orthogonalized a rank-k basis on
  load; the plugin serves rank 1 only — GLP-49 is rank 1, unaffected).
- The container is GGUF-only (the hotfix also accepted a .pt
  {layer: tensor} dump), matching the other plugin lanes.

Registration covers TWO arch names, and the reason is the hotfix's two-file
story restated at class granularity: Qwen3.8-27B's checkpoint declares
Qwen3_5ForConditionalGeneration (multimodal; the vision tower rides along),
and that wrapper builds its language model as a DIRECT Qwen3_5ForCausalLM
construction in qwen3_5.py — NOT through the registry — so shadowing only
Qwen3_5ForCausalLM would leave the actually-served arch unsteered. Both
names are shadowed; both wrappers swap their inner Qwen3_5Model onto
SteeredQwen3_5Model after construction (buffers are registered post-swap by
_wire_steering, before weight loading). The inner swap is also what fixes
the hotfix's qwen3_5.py half: Qwen3_5Model.__init__ skips its parent's, but
the plugin never relies on upstream __init__ for buffers at all.

Qwen3NextForCausalLM (Qwen3-Next-80B) is NOT registered: same inherited
forward, but no published vector exists for it and the assignment is
qwen38. Qwen3_5MTP is deliberately left stock: under greedy verification
the draft head only proposes, so speculative decoding does not change the
served distribution; shadowing it would steer a stream the GLP-49 layer
ids do not describe.

Parallelism: the stream at the loop is the full-width replicated residual
(TP sharding is internal to the layers and reduced before return), so the
projection is per-token over the full 5120-wide stream on every rank with
no collective. The dense stack is indexed by GLOBAL layer id and make_layers
pads with PPMissingLayer, so PP ranks steer their own window correctly.
Under use_sequence_parallel (MoE-only upstream gate; this dense model never
sets it) the in-loop stream is token-sharded, which the scalar lane
tolerates (the projection is per-token).

Upstream classes are imported from vllm.model_executor.models.qwen3_5 — the
vllm/vllm-openai:0.28.0 layout (the GitHub v0.28.0 tag and the image agree;
the Modal preflight asserts the module path in-image before boot).
"""
from __future__ import annotations

from itertools import islice

import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


def _steer_inner_model(model: Qwen3_5Model, vllm_config) -> None:
    """Swap a constructed Qwen3_5Model onto the steered class and wire it.

    Same __class__ swap idiom as nemotron_h: a subclass with an identical
    (pure Python) layout is a legal __class__ target, all buffers the
    steered forward needs are registered by _wire_steering below, and weight
    loading happens after __init__, so the swapped class is in place before
    any checkpoint tensors arrive.
    """
    model.__class__ = SteeredQwen3_5Model
    scheduler_config = vllm_config.scheduler_config
    model._wire_steering(
        dtype=vllm_config.model_config.dtype,
        max_num_tokens=scheduler_config.max_num_batched_tokens,
        max_num_reqs=scheduler_config.max_num_seqs,
    )
    # Qwen3_5Model IS @support_torch_compile'd (unlike glm5next): its
    # constructor captured the bound STOCK forward in the compile wrapper.
    # Rebind after the class swap and buffer registration, before warmup can
    # compile or capture an unsteered graph.
    if not getattr(model, "do_not_compile", True):
        from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper

        # Drop the first init's dynamo bytecode hook before registering the
        # second: register_bytecode_hook appends to a process-global dict
        # and the wrapper only ever removes the handle it last stored, so
        # re-initialising without this leaves a hook behind for the life of
        # the process.
        cleanup = getattr(TorchCompileWithNoGuardsWrapper, "cleanup", None)
        if cleanup is not None:
            cleanup(model)
        TorchCompileWithNoGuardsWrapper.__init__(
            model,
            compile_prefix=model._compile_prefix,
            is_encoder=model._is_encoder,
        )


class SteeredQwen3_5Model(Qwen3_5Model, SteeredModelMixin):
    """Qwen3_5Model with h <- h - alpha*(h.d)d applied per decoder layer.

    Qwen3_5Model inherits Qwen3NextModel.forward unchanged (v0.28.0): each
    decoder layer returns (hidden_states, residual) with the fold deferred,
    so the post-layer stream at the loop is h = hidden_states + residual;
    _steer_post_layer writes back hidden_states <- h' - residual so the next
    layer's fused add+norm reproduces h'. The loop's enumerate starts at
    start_layer, so layer_idx is the GLOBAL id the dense stack is indexed
    by.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        # ---- upstream forward, verbatim (Qwen3NextModel.forward @0.28.0) --
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

        full_num_tokens = positions.shape[-1]
        if self.use_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)
            assert residual is None

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            # [weightless-steer] the one added block: unconditional per-layer
            # projection at the post-layer residual stream (zero stack rows
            # make it a numeric no-op on unsteered layers).
            hidden_states = self._steer_post_layer(
                layer_idx, hidden_states, residual
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if self.use_sequence_parallel:
            if aux_hidden_states:
                hidden_size = hidden_states.shape[-1]
                hidden_states = torch.cat([hidden_states, *aux_hidden_states], dim=-1)
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                hidden_states = hidden_states[:full_num_tokens]
                hidden_states, *aux_hidden_states = hidden_states.split(
                    hidden_size, dim=-1
                )
            else:
                hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
                hidden_states = hidden_states[:full_num_tokens]
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states
        # ---- end upstream forward -----------------------------------------


class SteeredQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3_5ForCausalLM whose inner model applies GLP steering.

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
        _steer_inner_model(self.model, vllm_config)


class SteeredQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3_5ForConditionalGeneration whose language model steers.

    The arch Qwen3.8-27B checkpoints actually declare. The multimodal
    wrapper constructs Qwen3_5ForCausalLM directly (not via the registry),
    so the shadow must live on THIS class; the swap then reaches
    self.language_model.model. prefix defaults to "model" upstream.
    """

    def __init__(self, *, vllm_config, prefix: str = "model"):
        if _per_request_enabled():
            raise RuntimeError(
                "WEIGHTLESS_ENABLE_MILESTONE_2 requires request validation and "
                "runner integration, which this plugin does not implement. "
                "Unset it to serve with scalar steering."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        _steer_inner_model(self.language_model.model, vllm_config)
