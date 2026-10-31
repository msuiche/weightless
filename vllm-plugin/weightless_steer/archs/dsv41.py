"""DeepSeek-V4.1 (deepseek_v4_1) adapter: GLP steering at the hc fold site.

Implements GLP `spec_version 3`'s `hc_fold_pre_engram` hook — the site the
published GLP-27 vector (DeepSeek-V4.1-Flash-1017-abliterated-cyber-GLP-27,
L11-37, alpha 3.5) is calibrated at, and the one the spec's
"Multi-copy streams and the fold site" section defines. Verified live on the
four-node DGX ring by the spark-ds41f glp-specv3 lane (PR christopherowen/
spark-ds41f#738); this adapter brings the same math to stock-registry serving.

The site, exactly (docs/spec/GLP.md):

- Each token carries `hc_mult` hyper-connection copies, `[T, C, H]`, and a
  layer's FFN write folds into the stream at the START of the *next* layer
  (`mhc.post`, gated by post_mix/res_mix). `residual_stream_post_layer`
  names no tensor between layers on this arch; the steerable post-layer
  stream materialises only inside that fold.
- The projection operates on the model's own canonical reduction — the mean
  over copies — and subtracts the rank-1 term from every copy, which shifts
  the mean by exactly `alpha*(h.d)d`:

      h        <-  mean over the C copies              [T, H]
      copy_c   <-  copy_c - alpha * (h . d) d          for each copy c

- Layer mapping at the fold site: `direction.N` steers the fold consuming
  layer N's FFN write — the fold that runs at the start of layer N+1, and
  after the decoder loop for the final layer. `direction.0` stays invalid
  (row 0 zero-padded); the `direction.N` <-> layer N contract is unchanged,
  only the apply location sits one fold later.
- The companion attention writer is part of the site: `o_proj`'s output is
  projected with the same stack and the same alpha, attenuating the
  component's re-deposit on the next pass. A fold-only reader under-doses —
  the silent-degradation class the spec kills — so this adapter arms both
  halves or refuses the file.

Where this differs from the V4-Flash-0731 row (archs/dsv4.py,
`ffn_out_pre_residual`): that lane steers the layer's pending FFN write
BEFORE the fold; GLP-27 is natively derived AT the fold
(`glp.derived_at == glp.hook_point`), and the 2026-10-24 ladder measured the
two sites are not interchangeable (FFN-writer window alpha 4-6, the fold
site's operating point 3.5, corruption past 4.0). Two different files, two
different rows: a `ffn_out_pre_residual` file is refused here, and an
`hc_fold_pre_engram` file is refused on the dsv4 row.

Off is stock (the r6v lesson, restated in the spec): until a vector arms
steering, every fold site runs the tree's own fused path — the fold is
normally fused into `mhc.pre` and only materialises on Engram layers, so an
unarmed boot performs byte-identical compute with zero extra kernels and no
CUDA-graph delta. The first successful load (WEIGHTLESS_STEER_PATH at boot)
sets `_steer_on`; the apply is otherwise unconditional tensor math whose
zero rows and zero alpha make it a numeric no-op, keeping the traced graph
identical for every layer set.

What the forward changes, honestly:

- `SteeredDeepseekV41Model.forward` is upstream's copied verbatim with the
  layer-loop call wrapped: the loop hands each layer's inputs to
  `_steer_fold(layer, ...)` instead of calling `layer(...)` directly. That
  helper reproduces the tree's own Engram-layer materialization
  ("do not fuse across it") at EVERY armed layer — materialize
  `layer.mhc.post(x, residual, post_mix, res_mix)`, steer, continue with
  previous_output cleared — and forwards untouched when steering is off.
  Wrapping the call (rather than copying the decoder layer) keeps the
  upstream layer class exactly as loaded.
- The post-loop fold (`layer.mhc.post` after the last layer, and the
  aux-hidden-state reuse branch) steers identically: `direction[N]` at the
  fold consuming the final layer's write.
- The companion writer wraps each `layer.attn._o_proj` bound method with a
  kwargs-forwarding closure (this tree's call site passes keywords), active
  only while steering is armed; alpha 0 or a zero row is a numeric no-op.

The DSpark drafter builds its own `DeepseekV4DecoderLayer` stack
(nvidia/dspark.py); `_tag_layers` walks `model.modules()` so every instance —
including any rebuilt after weight loading — carries the steer attributes,
and the unarmed fallback is inert rather than an AttributeError.

Parallelism: the fold-site stream is full-width and replicated at the layer
boundary (TP sharding is internal to attention/MoE and reduced before the
layer returns), so the projection is per-token over the full hidden_size on
every rank with no collective — the same property the dsv4 row documents.
Per-request controls are refused at construction (no runner binding), and
under sequence parallelism the token-sharded stream makes the per-request
lane unreachable by construction.

Upstream classes are imported from vllm.models.deepseek_v4_1.nvidia.model
(the 1022 candidate tree; the same module path exists in earlier V4.1
trees). Served as `DeepseekV41ForCausalLM` (the multimodal wrapper builds
its language model through the registry, so the shadow covers it).
"""
from __future__ import annotations

from itertools import islice

import torch
import torch.nn as nn

from vllm.distributed import get_pp_group
from vllm.models.deepseek_v4_1.nvidia.model import (
    DeepseekV41LLMForCausalLM,
    DeepseekV4Model,
)
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


class _FoldSteer:
    """The fold-site op, as a helper the loop calls (kept off the steered
    model class so the copied forward stays recognizably upstream)."""

    __slots__ = ("dirs", "alpha")

    def __init__(self, owner: "SteeredDeepseekV41Model"):
        # Views alias the registered buffers; the captured graph reads the
        # buffers, so ctrl-side buffer mutation reaches the replayed graph.
        self.dirs = owner._steer_stack
        self.alpha = owner._steer_alpha

    def fold(self, layer, x, residual, post_mix, res_mix):
        """Materialize the fold, steer it, hand the loop the steered stream.

        Returns (residual, previous_output cleared state) — the caller
        passes what follows to mhc.pre exactly as the tree's own Engram
        branch does. Steering-off layers never reach here: the steered
        forward keeps the stock `layer(...)` call.
        """
        stream = layer.mhc.post(x, residual, post_mix, res_mix)
        h = stream.mean(dim=1)
        d = layer._ds41_steer_dir
        # The rank-1 term subtracted from every copy shifts the mean by
        # exactly alpha*(h.d)d: the spec's op, graph-safe (pure tensor math
        # on registered buffers; zero rows make it a numeric no-op).
        stream = stream - (
            (self.alpha * (h @ d)).unsqueeze(-1) * d
        ).unsqueeze(1)
        return stream


class SteeredDeepseekV41Model(DeepseekV4Model, SteeredModelMixin):
    """DeepseekV4Model with the GLP v3 fold-site projection applied.

    Off is stock: `_steer_on` False keeps the loop's `layer(...)` call
    untouched (the fused mhc.pre path, zero extra kernels). Armed, each
    layer with an incoming residual materializes the fold, steers
    post-fold pre-Engram (`hc_fold_pre_engram`), and continues exactly the
    way the tree's own Engram branch continues after materializing
    (previous_output cleared so the fused pre does not re-fold).
    """

    STEER_HOOK = "hc_fold_pre_engram"

    def _tag_layers(self) -> int:
        """Point every DeepseekV4DecoderLayer under this model at the shared
        steer state. dspark.py builds its OWN layer ModuleList for the draft
        stack, and loaders may rebuild layers after __init__; this walks
        model.modules() so every instance carries the attributes. Fold rows:
        the DecoderLayer at loop position i applies direction[i-1] (its
        start fold consumes layer i-1's FFN write)."""
        lays = [
            m for m in self.modules()
            if type(m).__name__ == "DeepseekV4DecoderLayer"
            or (isinstance(m, nn.Module)
                and hasattr(m, "mhc") and hasattr(m, "attn")
                and (hasattr(m, "ffn_write") or hasattr(m, "block_write")))
        ]
        for i, lay in enumerate(lays):
            lay._ds41_steer_dir = (
                self._steer_stack[i - 1, 0]
                if i >= 1 else self._steer_stack.new_zeros(
                    self._steer_stack.shape[-1]
                )
            )
            lay._ds41_steer_alpha = self._steer_alpha
        return len(lays)

    def _steer_fold(self, layer, x, residual, post_mix, res_mix):
        """The armed fold-site step: materialize, steer, return the stream
        the loop continues with (deferred state cleared, matching the
        tree's Engram materialization contract)."""
        stream = self._foldop.fold(layer, x, residual, post_mix, res_mix)
        return stream, None, None, None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        ced_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        # ---- upstream forward, verbatim -----------------------------------
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            pre_mix = None
            post_mix = None
            res_mix = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            pre_mix = intermediate_tensors["pre_mix"]
            post_mix = None
            res_mix = None

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            from vllm.models.common.ops.sequence_parallel import sp_shard

            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if self._steer_on and residual is not None and hidden_states \
                    is not None:
                # [weightless-steer] the armed fold site runs BEFORE the
                # layer's blocks consume the fold: materialize the fold
                # this tree normally fuses into mhc.pre, steer post-fold
                # pre-Engram, and pass the steered stream with the deferred
                # state cleared — the exact continuation the tree's own
                # Engram branch uses after materializing.
                hidden_states, residual, post_mix, res_mix = self._steer_fold(
                    layer, hidden_states, residual, post_mix, res_mix
                )
                pre_mix = None
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes=kwargs.get("engram_hashes"),
                engram_mask=kwargs.get("engram_mask"),
                ced_indices=kwargs.get("ced_indices"),
                sp_rows=kwargs.get("sp_rows"),
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # Reconstruct the aux hidden state for draft models. Under
                # the armed path the layer returned the pending write and
                # the streams; fold them (unsteered — this is the NEXT
                # layer's fold, armed on its own iteration) then collapse.
                aux_recon = layer.mhc.post(
                    hidden_states, residual, post_mix, res_mix) \
                    if residual is not None else hidden_states
                aux_hidden_state = layer.mhc.collapse(aux_recon)
                if self.use_sequence_parallel:
                    from vllm.models.common.ops.sequence_parallel import (
                        sp_all_gather,
                    )

                    aux_hidden_state = sp_all_gather(aux_hidden_state)[
                        :full_num_tokens
                    ]
                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon

        if layer is not None:
            if self.end_layer in self.aux_hidden_state_layers:
                hidden_states = final_aux_recon
            elif residual is not None:
                hidden_states = layer.mhc.post(
                    hidden_states, residual, post_mix, res_mix
                )
            # When the armed loop already materialized the final fold,
            # hidden_states IS the folded stream; the post-loop mhc.post
            # is the stock path and must not re-fold on top of it.
            if self._steer_on and self.end_layer - 1 in range(
                self._steer_stack.shape[0]
            ):
                # [weightless-steer] the post-loop fold: direction[N] at the
                # fold consuming the final layer's FFN write.
                d = self._steer_stack[self.end_layer - 1, 0]
                h = hidden_states.mean(dim=1)
                hidden_states = hidden_states - (
                    (self._steer_alpha * (h @ d)).unsqueeze(-1) * d
                ).unsqueeze(1)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )

        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(
                hidden_states.flatten(1)
            )

        hidden_states = layer.mhc.collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class SteeredDeepseekV41LLMForCausalLM(DeepseekV41LLMForCausalLM):
    """Registry shadow for `DeepseekV41ForCausalLM`'s language model.

    The `DeepseekV41LLMForCausalLM` inner class owns the decoder; the
    multimodal wrapper (DeepseekV41ForCausalLM) builds it through the
    registry with architectures=["DeepseekV41ForCausalLM"], so this shadow
    plus the plugin's registry entry covers both serving shapes.
    """

    def __init__(self, *, vllm_config, prefix: str = ""):
        if _per_request_enabled():
            raise RuntimeError(
                "per-request controls need the runner binding this plugin "
                "does not implement on dsv41. Unset "
                "WEIGHTLESS_ENABLE_MILESTONE_2 to serve with scalar steering."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model.__class__ = SteeredDeepseekV41Model
        scheduler_config = vllm_config.scheduler_config
        self.model._wire_steering(
            dtype=vllm_config.model_config.dtype,
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_num_reqs=scheduler_config.max_num_seqs,
        )
        self.model._foldop = _FoldSteer(self.model)
        self.model._tag_layers()
        armed = self.model._steer_core.dirs != {}
        self.model._steer_on = armed
        if armed:
            self._arm_companion_writer(self.model)

    def _arm_companion_writer(self, model) -> None:
        """Wrap each layer's attn._o_proj with the spec-mandated companion
        projection (same stack, same alpha). The candidate tree's call site
        passes keywords, so the closure forwards *args/**kwargs. Zero row or
        alpha 0 is a numeric no-op; the wrap is inert until buffers fill."""
        stack = model._steer_stack
        alpha = model._steer_alpha
        layers = [lay for lay in model.layers]
        for i, lay in enumerate(layers, start=1):
            lay._ds41_lid = i
            attn = getattr(lay, "attn", None)
            if attn is None or getattr(attn, "_ds41_wrapped", False):
                continue
            orig = attn._o_proj

            def wrap(o, pos, *args, _self=lay, _f=orig, _stack=stack,
                     _alpha=alpha, **kwargs):
                out = _f(o, pos, *args, **kwargs)
                lid = getattr(_self, "_ds41_lid", None)
                if lid is None or not (1 <= lid < _stack.shape[0]):
                    return out
                d = _stack[lid, 0]
                if float(d.norm()) <= 0.0 or float(_alpha) == 0.0:
                    return out
                h = out.float()
                return (
                    h - (float(_alpha) * (h @ d)).unsqueeze(-1) * d
                ).to(out.dtype)

            attn._o_proj = wrap
            attn._ds41_wrapped = True