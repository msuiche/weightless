"""GLM-5.3 743B (glm53xl) adapter: GLP steering on the deepseek_v2 path.

`SteeredGlmMoeDsaModel.forward` is stock vLLM v0.28.0's
`DeepseekV2Model.forward` copied verbatim with one line added in the layer
loop — the projection at the post-layer residual stream. This is the
coupling the design doc accepts for path (1): the loop shape and the
(hidden_states, residual) return convention are upstream internals that can
change in any release, but a break now fails loudly at import or first
forward instead of silently serving unsteered (the anchor-matched hotfix's
failure mode). The copy is pinned against
patches/reference/deepseek_v2_v0280.py (byte-identical to the
vllm/vllm-openai:v0.28.0 image's
vllm/model_executor/models/deepseek_v2.py, md5 61da370634b4dfbe7f0158aaffa55202;
the same file the GLP-77 capture and eval ran on, refusal-research
experiments/20260829-glm53-flagship) by tests/test_archs/test_glm53xl.py.

Replaces patches/hotfix-glm53xl-steering-projective.py, with the same
steering semantics so the published GLP-77 vector
(msuiche/GLM-5.3-abliterated-cyber-GLP-77,
GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf, layers 1-77 of 78, plain
6144-wide directions) and its alpha transfer without recalibration:

- glm_moe_dsa is the plain deepseek_v2 residual convention — NO mHC
  widening (that is glm5next/GLM-5.3-Flash, a different arch and adapter).
  Each decoder layer returns the decomposed (hidden_states, residual) pair
  with the fold fused into the next layer's input_layernorm, so the
  post-layer stream at the loop is h = hidden_states + residual (the file's
  own aux_hidden_states code does the same sum). _steer_post_layer steers
  the sum and writes back hidden_states <- h' - residual, leaving residual
  untouched, so the next fold reproduces h'.
- The apply sits immediately after the layer call, the hotfix's exact
  anchor position. Stock v0.28.0 (unlike the tonyd2wild GB10 overlay the
  hotfix patches) additionally carries sequence-parallel all-gather blocks
  at the top of the loop and after it; those are token-sharding data
  movement, and the projection is per-token, so steering before them is
  shard-invariant.
- No last-layer trap: unlike glm5next's deferred mHC contract, every
  deepseek_v2 layer returns the unfused pair and the terminal self.norm
  folds after the loop, so layer 77 (GLP-77's last) is steered by the same
  loop line as every other layer. The post-loop aux_hidden_state capture
  sees the steered stream, matching the hotfix's relative ordering.
- The MTP draft layer (num_nextn_predict_layers=1) is not part of
  self.layers (make_layers spans config.num_hidden_layers) and is never
  steered; GLP-77 covers the full base stack 1..77.

Semantic differences from the hotfix, called out honestly:

- Alpha resolution is SteeringCore's: WEIGHTLESS_STEER_ALPHA, else the
  file's glp.alpha_default, else 1.0. GLP-77 ships glp.alpha_default=1.0 —
  the calibrated value (higher alpha makes refusal WORSE at full length on
  this model; measured 2026-08-30) — so an env-unset boot matches the
  hotfix's hardcoded 1.0 exactly.
- Rank-k files are REFUSED (the hotfix QR-orthogonalized a rank-k basis on
  load; the plugin serves rank 1 only — GLP-77 is rank 1, unaffected).
- The container is GGUF-only (the hotfix also accepted a .pt
  {layer: tensor} dump), matching the other plugin lanes.

Parallelism: the stream at the loop is full-width and replicated across TP
ranks (TP sharding is internal to attention/MoE and reduced before the loop
state), so the projection needs no collective — the Modal lane runs TP8.
PP is supported upstream (SupportsPP): the dense direction stack is indexed
by GLOBAL layer id, so each rank steers exactly its own slice. Under
sequence-parallel MoE the in-loop pair is token-sharded, which the scalar
lane tolerates (the projection is per-token); per-request controls are
refused at construction on every arch (no runner binding exists), so the
token-ordinal misalignment SP would cause there is unreachable.

Compile: DeepseekV2Model DOES carry @support_torch_compile (unlike
glm5next, where the rebind is dormant), so the wrapper rebind in
SteeredGlmMoeDsaForCausalLM.__init__ is load-bearing here: the decorated
stock __init__ captures torch.compile(self.forward) — the STOCK bound
forward — during super().__init__(), and without the rebind compiled
serving would run unsteered while the eager path steered.
"""
from __future__ import annotations

from itertools import islice

import torch
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2Model,
    GlmMoeDsaForCausalLM,
    _get_llama_4_scaling,
)
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


class SteeredGlmMoeDsaModel(DeepseekV2Model, SteeredModelMixin):
    """DeepseekV2Model with h <- h - alpha*(h.d)d applied per decoder layer.

    deepseek_v2 uses the decomposed convention: each decoder layer returns
    (hidden_states, residual) with the fold deferred to the next layer's
    input_layernorm, so the post-layer stream at the loop is
    h = hidden_states + residual; _steer_post_layer writes back
    hidden_states <- h' - residual so the next fold reproduces h'. The
    stream is plain hidden_size wide (6144 on GLM-5.3) — the mixin's
    default stream_width, no mHC widening.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # ---- upstream forward, verbatim -----------------------------------
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "Either input_ids or inputs_embeds must be provided "
                        "to DeepseekV2Model.forward"
                    )
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        llama_4_scaling: torch.Tensor | None
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        aux_hidden_states = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # all gather if we need to use the whole states
            if (
                hidden_states.shape[0] != positions.shape[0]
                and not layer.use_sequence_parallel_moe
            ):
                combined_states = torch.cat([hidden_states, residual], dim=-1)
                combined_states = tensor_model_parallel_all_gather(combined_states, 0)
                combined_states = combined_states[: positions.shape[0]]
                hidden_states, residual = combined_states.split(
                    [self.hidden_size, self.hidden_size], dim=-1
                )
            if idx in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states + residual
                if aux_hidden_state.shape[0] != positions.shape[0]:
                    aux_hidden_state = tensor_model_parallel_all_gather(
                        aux_hidden_state, 0
                    )
                    aux_hidden_state = aux_hidden_state[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )
            # [weightless-steer] the one added line: unconditional per-layer
            # projection at the post-layer residual stream (zero stack rows
            # make it a numeric no-op on unsteered layers). idx is the GLOBAL
            # layer id, correct under pipeline parallelism.
            hidden_states = self._steer_post_layer(idx, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != positions.shape[0]:
            combined_states = torch.cat([hidden_states, residual], dim=-1)
            combined_states = tensor_model_parallel_all_gather(combined_states, 0)
            combined_states = combined_states[: positions.shape[0]]
            hidden_states, residual = combined_states.split(
                [self.hidden_size, self.hidden_size], dim=-1
            )

        if self.end_layer in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states
        # ---- end upstream forward -----------------------------------------


class SteeredGlmMoeDsaForCausalLM(GlmMoeDsaForCausalLM):
    """GlmMoeDsaForCausalLM whose inner model applies GLP steering.

    Registered as a lazy shadow of the stock arch by plugin.register() when
    WEIGHTLESS_STEER_PATH is set; every weight-loading and interface method
    is inherited untouched. GlmMoeDsaForCausalLM is a plain `pass` subclass
    of DeepseekV2ForCausalLM whose model_cls is DeepseekV2Model — no
    skip-parent trap: super().__init__ builds a stock DeepseekV2Model as
    self.model, and the swap below reclasses it.
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
        # 78-layer stack a second time. A subclass with an identical (pure
        # Python) layout is a legal __class__ target, and all buffers the
        # steered forward needs are registered by _wire_steering below.
        # Weight loading happens after __init__, so the swapped class is in
        # place before any checkpoint tensors arrive.
        self.model.__class__ = SteeredGlmMoeDsaModel
        scheduler_config = vllm_config.scheduler_config
        self.model._wire_steering(
            dtype=vllm_config.model_config.dtype,
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_num_reqs=scheduler_config.max_num_seqs,
        )
        # The upstream constructor captures its bound forward in the compile
        # wrapper (DeepseekV2Model is @support_torch_compile'd, so unlike
        # glm5next this rebind is live). Rebind after the class swap and
        # buffer registration, before warmup can compile or capture a stock,
        # unsteered forward.
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
