"""Tencent Hy4-preview (hy_v4) adapter: GLP steering on the materialized iHC stream.

`SteeredHy4Model.forward` is upstream's `HYV4Model.forward` copied verbatim
with one block added in the layer loop. This is the coupling the design doc
accepts for path (1): the loop shape and the (hidden_states, residual) return
convention are upstream internals that can change in any release, but a break
now fails loudly at import or first forward instead of silently serving
unsteered (the anchor-matched hotfix's failure mode). The copy is pinned
against patches/reference/hy_v4_nvidia_model.py by
tests/test_archs/test_hy4.py.

Replaces patches/hotfix-hy4-steering-projective.py, with the same steering
semantics so the published GLP-77 vector
(Hy4-preview-abliterated-cyber-GLP-77-L1-77-a2.0.gguf, layers 1-77) and its
alpha transfer without recalibration:

- hy_v4 runs iHC (identity hyper-connections): the decoder layer carries the
  FULL multi-stream state in hidden_states as [T, hc_mult=4, hidden=6144] and
  each sub-block's hc post merges immediately — no pending combine (unlike
  glm5next's deferred mHC, no materialization step and no last-layer terminal
  contract; every layer's post-layer stream is simply the hidden_states it
  returns, residual=None). The apply therefore runs directly on the returned
  hidden_states — the glp.hook_point=residual_stream_post_layer site an HF
  forward hook sees.
- The projection keeps the [T, 4, 6144] shape: SteeringCore.apply's einsum
  contracts only the last axis, so each of the 4 iHC streams is steered
  independently with the SAME 6144-wide direction — the hotfix's semantics
  exactly (direction width == hidden_size, never the flattened hc*hidden;
  the derivation reduces the captured streams to one direction per layer).
  stream_width stays at the base mixin default, config.hidden_size.
- No per-layer trap: the loop indexes the dense stack with layer.layer_idx
  (global id, set per decoder layer from its prefix). The islice over
  [start_layer, end_layer) never contains a PPMissingLayer (make_layers
  places those strictly outside the window), so the hotfix's isinstance
  guard is dead code and is not carried over.

Semantic differences from the hotfix, called out honestly:

- Alpha resolution is SteeringCore's: WEIGHTLESS_STEER_ALPHA, else the
  file's glp.alpha_default, else 1.0. The hotfix hardcoded 1.0 when the env
  was unset (its eval lane always set the env explicitly). The published a2.0
  file carries glp.alpha_default=2.0 — the calibration point (ladder
  1→11→19→24 delivered on refusal32, no garble at any dose) — so an env-unset
  boot now serves alpha 2.0, not 1.0; set WEIGHTLESS_STEER_ALPHA explicitly
  for anything else.
- Rank-k files are REFUSED (the hotfix QR-orthogonalized a rank-k basis on
  load; the plugin serves rank 1 only — GLP-77 is rank 1, unaffected).
- The container is GGUF-only (the hotfix also accepted a .pt
  {layer: tensor} dump), matching the other plugin lanes.
- A hypothetical non-iHC hy_v4 config (enable_ihc unset): _forward_normal
  returns (mlp_output, residual) with the fold deferred, and this adapter —
  like the hotfix — projects the returned hidden_states directly, NOT the
  folded h = hidden_states + residual that _steer_post_layer would compute.
  Preserved for hotfix parity; no shipped hy_v4 checkpoint is non-iHC and
  the GLP-77 vector was derived and calibrated on the iHC stream only.

Parallelism: the iHC stream is full-width and replicated at the layer
boundary (TP sharding is internal to MLA/MoE and reduced before the loop
state), so the projection is per-token over [T, 4, 6144] on every rank with
no collective — safe at TP8 (the validated 8xH200 lane). PP is supported
upstream (make_empty_intermediate_tensors flattens the stream to
[T, hc*hidden] for the transfer; the receiving rank's first layer reshapes
via prepare_input): the dense stack is indexed by global layer id, so each
rank steers its own window correctly. Per-request controls are refused at
construction on every arch (no runner binding exists) — and could never
apply here anyway: the 3-D stream trips SteeringCore.apply's 2-D guard by
design.

The MTP draft stack (nvidia/mtp.py, HYV4MTP) is a separate module serving
spec decode; it is not covered by this adapter. The 78 backbone layers are.

Upstream classes are imported from vllm.models.hy_v4.nvidia.model — the
day-0 image layout the hotfix patches and the vendored reference mirrors
(vllm.models.hy_v4 itself only dispatches on current_platform).
"""
from __future__ import annotations

from itertools import islice

import torch
from vllm.distributed import get_pp_group
from vllm.models.hy_v4.nvidia.model import HYV4ForCausalLM, HYV4Model
from vllm.sequence import IntermediateTensors

from .base import SteeredModelMixin, _per_request_enabled


class SteeredHy4Model(HYV4Model, SteeredModelMixin):
    """HYV4Model with h <- h - alpha*(h.d)d applied per decoder layer.

    hy_v4's iHC layers return the fully merged multi-stream state:
    hidden_states is [T, hc_mult, hidden] and residual is None, so the
    post-layer stream at the loop is hidden_states itself. The projection
    runs on it directly — SteeringCore.apply's ellipsis contracts only the
    last axis, removing the component from each iHC stream independently
    with one hidden_size-wide direction per layer (zero stack rows make it
    a numeric no-op on unsteered layers).
    """

    def forward(
        self,
        input_ids: torch.Tensor,
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
            # In iHC mode the flattened [num_tokens, hc*h] tensor from the
            # previous PP stage is reshaped back to 3D by the first layer's
            # prepare_input, and residual is unused.
            residual = None if self.enable_ihc else intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)
            # [weightless-steer] the one added block: unconditional per-layer
            # projection on the post-layer stream (zero stack rows make it a
            # numeric no-op on unsteered layers). Under iHC hidden_states is
            # [T, hc_mult, hidden] and the apply steers each stream
            # independently; layer.layer_idx is the global id, correct under
            # a PP split (the islice window never holds a PPMissingLayer).
            hidden_states = self._steer_core.apply(layer.layer_idx,
                                                   hidden_states)

        if not get_pp_group().is_last_rank:
            if self.enable_ihc:
                # hidden_states is [num_tokens, hc, h]; flatten the channel dim
                # for PP transfer (matches the 2D receive buffer).
                return IntermediateTensors({"hidden_states": hidden_states.flatten(1)})
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        if self.enable_ihc:
            hidden_states = self.hc_head(hidden_states)
        else:
            hidden_states = hidden_states + residual

        return self.norm(hidden_states)
        # ---- end upstream forward -----------------------------------------


class SteeredHy4ForCausalLM(HYV4ForCausalLM):
    """HYV4ForCausalLM whose inner model applies GLP steering.

    Registered as a lazy shadow of the stock arch by plugin.register() when
    WEIGHTLESS_STEER_PATH is set. Every weight-loading and interface method
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
        # Same __class__ swap idiom as nemotron_h/glm5next: swap the
        # already-constructed inner model onto the steered class rather than
        # rebuilding it (make_layers would allocate the whole layer stack a
        # second time). A subclass with an identical (pure Python) layout is
        # a legal __class__ target, all buffers the steered forward needs are
        # registered by _wire_steering below, and weight loading happens
        # after __init__, so the swap is in place before any checkpoint
        # tensors arrive.
        self.model.__class__ = SteeredHy4Model
        scheduler_config = vllm_config.scheduler_config
        # stream_width stays at the default config.hidden_size: one
        # 6144-wide direction per layer, applied per iHC stream — NOT the
        # flattened hc_mult*hidden (that is glm5next's derivation space, and
        # the width guard fails closed on such a vector here).
        self.model._wire_steering(
            dtype=vllm_config.model_config.dtype,
            max_num_tokens=scheduler_config.max_num_batched_tokens,
            max_num_reqs=scheduler_config.max_num_seqs,
        )
        # The upstream constructor captures its bound forward in the compile
        # wrapper. Rebind after the class swap and buffer registration, before
        # warmup can compile or capture a stock, unsteered forward. hy_v4
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
