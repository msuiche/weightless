"""ByteDance Ouro adapter: GLP steering per EXECUTION STEP on a looped LM.

Ouro (LoopLM, arXiv 2510.25741) iterates 48 shared physical layers
total_ut_steps=4 times — 192 execution steps over the SAME weights. The
loop has no execution index upstream, so the steered forward enumerates
the layer slice and keys on

    exec_id = current_ut * num_hidden_layers + start_layer + layer_idx

which is also the KV-slot indexing (OuroAttention builds one Attention per
(ut, layer): unique_layer_idx = ut_step * num_hidden_layers + base_layer_idx).

`SteeredOuroModel.forward` is upstream v0.26.0 `OuroModel.forward` copied
verbatim with one block added in the layer loop — the projection at the
post-layer residual stream of every execution step. This is the coupling
the design doc accepts for path (1): the loop shape and the
(hidden_states, residual) return convention are upstream internals, but a
break now fails loudly at import or first forward instead of silently
serving unsteered (the anchor-matched hotfix's failure mode). The copy is
pinned against patches/reference/ouro_v0260.py by
tests/test_archs/test_ouro.py.

Replaces patches/hotfix-ouro-steering-projective.py, with the same steering
semantics so the published GLP-192 vector (msuiche/Ouro-2.6B-abliterated-
cyber-GLP-192-L1-192-a1.0, exec steps 1..192 as container ids) and its
alpha transfer without recalibration:

- ouro uses vLLM's fused add+norm convention: the sandwich-normed decoder
  layer returns (pa2(mlp_output), residual) with the fold deferred to the
  next norm (or the inter-pass self.norm at pass end), so the post-layer
  stream at the model loop is h = hidden_states + residual, and the mixin's
  _steer_post_layer writes back hidden_states <- h' - residual so the next
  fold reproduces h'. derived_at == hook_point == residual_stream_post_layer.
- ouro is a plain single-stream arch (no hyper-connection widening): the
  stream width is config.hidden_size (2048 on Ouro-2.6B).

THE LOOPED-MODEL CONTAINER SHIFT — the one semantic trap of this lane.
Every other plugin arch follows container.py's convention direction.N =
global layer id N (layer 0 inexpressible). The ouro vector was exported by
captain-vector 0.1.2 under the looped-model convention: direction.N =
EXECUTION STEP N-1 (step 0 must be expressible — it is steered), and the
file's glp.layer_ids_zero_based carries the CONTAINER ids 1..192 despite
the key name (documented in the hotfix and in
experiments/20260906-ouro-glp/STATE.md; NOT fixed upstream). The validated
numbers (hotfix docstring: alpha=1.0 refusal32 4->32/32 comply, cyber32
30->31/32, benign32 32/32 unchanged; alpha=0.0 no-op gate passes; alpha=3.0
collapses; usable band [0.5, 2.0]) were produced with exec = N-1, so that
mapping is load-bearing. SteeringCore.from_env cannot be reused here: it
would read direction.192 as layer 192 and reject it as out of range for a
192-step stack BEFORE this adapter could shift. exec_core_from_env below
mirrors from_env's gates (mode/hook/spec gates in load_control_vector,
alpha-scaling and rank-k refusals, alpha resolution, layer filter, width
and range checks, unit re-normalisation, fail-closed log-and-raise) with
exactly one change: container id N maps to execution step N-1 before
filtering and range-checking. WEIGHTLESS_STEER_LAYERS therefore names
0-based EXECUTION ids, as the hotfix documented.

Looped-model caveat (METHODOLOGY §"Known limitation: looped models"):
per-layer GLP semantics assume distinct weights per layer. Here the same
weights serve 4 effective depths and the shipped vector is
per-execution-step (dom_per_execution_step), which is the meaningful axis —
but the degenerate-risk caveat stands: this lane's evidence is the
published vector's own validation row, not a fresh derivation.

Version pin: vLLM removed OuroForCausalLM upstream in #49786 (2026-07-25),
~2h after the v0.26.0 image was built. This adapter imports
vllm.model_executor.models.ouro and therefore requires
vllm/vllm-openai:v0.26.0 EXACTLY — the last image carrying the arch.
Unlike glm5next, the compile rebind is LIVE here: OuroModel carries
@support_torch_compile at v0.26.0, so the wrapper's captured bound forward
must be re-initialised after the __class__ swap or compiled serving runs
the stock forward unsteered.

Parallelism: the residual stream is full-width and replicated at the layer
boundary (o_proj/down_proj all-reduce before returning), so the projection
is per-token on every rank with no collective — TP-safe (the hotfix
validated TP=1; the math is TP-invariant). PP: upstream's forward has no
pipeline-rank branches (intermediate_tensors is accepted but unused), so PP
is not a real lane for this arch; the start_layer/end_layer terms keep exec
ids global if one ever is.

Semantic differences from the hotfix, called out honestly:

- GGUF-only (the hotfix also accepted a .pt {exec_id: tensor} dump),
  matching the other plugin lanes.
- Rank-k files are REFUSED (rank 1 only — GLP-192 is rank 1, unaffected),
  as are spec-version-2 alpha multipliers (glp.dir_scales/layer_scales).
- Per-request controls are refused at construction on every arch, this one
  included.
"""
from __future__ import annotations

import logging
import math
import os

import torch
from vllm.model_executor.models.ouro import OuroForCausalLM, OuroModel
from vllm.sequence import IntermediateTensors

from ..container import load_control_vector
from ..core import SteeringCore
from .base import SteeredModelMixin, _per_request_enabled

logger = logging.getLogger(__name__)


def exec_core_from_env(*, hook, exec_steps, hidden_size,
                       max_num_tokens=None, max_num_reqs=None):
    """SteeringCore.from_env for a looped arch: direction.N = exec step N-1.

    Mirrors SteeringCore.from_env gate for gate, with the looped-model
    container shift applied right after load_control_vector: the published
    GLP-192 file names its tensors direction.1..direction.192 for execution
    steps 0..191 (execution step 0 must be expressible; the container cannot
    express a 0 id). Returns None when WEIGHTLESS_STEER_PATH is unset.
    Fail-closed: set-but-bad file, wrong hook, alpha-scaling keys, rank-k,
    wrong width, out-of-range steps, or a layer filter that empties the set
    all raise — a boot asked for steering must not serve unsteered.
    """
    path = os.environ.get("WEIGHTLESS_STEER_PATH", "").strip()
    if not path:
        return None

    env_hook = (os.environ.get("WEIGHTLESS_STEER_HOOK") or hook).strip()
    if env_hook != hook:
        raise RuntimeError(
            f"WEIGHTLESS_STEER_HOOK={env_hook!r} is not implemented at "
            f"this arch adapter; the only site here is {hook}. Refusing "
            f"to serve with a silently wrong hook site."
        )

    try:
        meta, raw = load_control_vector(path, hook=hook)
        # Spec-version-2 alpha multipliers rescale alpha per direction and
        # per layer; this lane applies ONE scalar alpha buffer to every
        # steered step, so a file carrying them would be served at the
        # wrong strength — with no error, and no way to tell from the
        # output. Refuse rather than ignore.
        scales = sorted(set(meta) & {"glp.dir_scales", "glp.layer_scales"})
        if scales:
            raise ValueError(
                f"{path}: carries {', '.join(scales)}, which this serving "
                f"lane does not implement. Refusing to ignore alpha "
                f"multipliers."
            )
        # Rank 1 only: one scalar alpha buffer per model, one direction per
        # execution step. A rank-k (subspace) vector needs per-direction
        # alphas; serving only direction 0 would be the silent partial
        # apply the spec's refusal rule exists to prevent.
        if any(v.dim() > 1 for v in raw.values()):
            raise RuntimeError(
                f"{path}: rank-k (subspace) GLP vector — this serving lane "
                f"implements rank 1 only. Refusing to serve a partially "
                f"applied subspace."
            )

        alpha_env = os.environ.get("WEIGHTLESS_STEER_ALPHA", "").strip()
        alpha = (float(alpha_env) if alpha_env
                 else float(meta.get("glp.alpha_default", 1.0)))
        if not math.isfinite(alpha):
            raise ValueError("steering alpha must be finite")

        want = os.environ.get("WEIGHTLESS_STEER_LAYERS", "").strip()
        selected = (
            {int(t) for t in want.replace(" ", "").split(",") if t}
            if want else None
        )

        dirs = {}
        for container_id, vec in raw.items():
            # direction.N steers execution step N-1 (looped convention).
            exec_id = int(container_id) - 1
            if selected is not None and exec_id not in selected:
                continue
            vec = vec.detach().to(torch.float32).reshape(-1)
            # Width guard: a direction that does not match this arch's
            # stream width must fail here, not as an opaque broadcast error
            # at serve time.
            if vec.numel() != hidden_size:
                raise RuntimeError(
                    f"steering vector exec step {exec_id} width "
                    f"{vec.numel()} != {hidden_size} "
                    f"(hidden_size; plain single stream)"
                )
            # The published vector ships unit directions; normalise anyway
            # so a non-unit export cannot silently scale alpha. In float64:
            # a finite but extreme f32 export (1e30) would overflow its own
            # squared norm in f32 and normalise to inf or 0.
            norm = vec.double().norm()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError(
                    f"{path}: direction exec step {exec_id} must be finite "
                    f"and nonzero; refusing to steer along a direction "
                    f"that is neither"
                )
            dirs[exec_id] = (vec.double() / norm).float()

        out_of_range = sorted(
            int(k) - 1 for k in raw
            if not 0 <= int(k) - 1 < exec_steps
        )
        if out_of_range:
            raise RuntimeError(
                f"{path}: direction exec steps {out_of_range} out of range "
                f"for this model ({exec_steps} execution steps)"
            )
        if not dirs:
            raise RuntimeError(
                f"WEIGHTLESS_STEER_PATH={path} matched no execution steps; "
                f"refusing to run unsteered"
            )
    except Exception as exc:
        # Fail closed: a boot asked for steering must not serve unsteered.
        logger.error("GLP steering load failed (%s); failing closed", exc)
        raise

    logger.info(
        "weightless GLP steering active: hook=%s alpha=%.3f "
        "exec_steps=%d..%d (%d) width=%d",
        hook, alpha, min(dirs), max(dirs), len(dirs), hidden_size,
    )
    return SteeringCore(dirs, alpha, hook, exec_steps, hidden_size,
                        path=path, max_num_tokens=max_num_tokens,
                        max_num_reqs=max_num_reqs)


class SteeredOuroModel(OuroModel, SteeredModelMixin):
    """OuroModel with h <- h - alpha*(h.d)d applied per execution step.

    The core's stack is indexed by EXECUTION id (192 rows on Ouro-2.6B),
    not by physical layer — _wire_steering's num_layers=len(self.layers)
    would size it to the 48 physical layers, so this adapter wires the core
    itself in SteeredOuroForCausalLM.__init__. The apply site is the
    mixin's _steer_post_layer: ouro's fused add+norm returns
    (pa2(mlp_output), residual) with the fold deferred, so the post-layer
    stream is h = hidden_states + residual and the write-back is
    hidden_states <- h' - residual.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        # ---- upstream forward, verbatim (vLLM v0.26.0) --------------------
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)

        for current_ut in range(self.total_ut_steps):
            residual = None
            for _ouro_lidx, layer in enumerate(
                self.layers[self.start_layer : self.end_layer]
            ):
                hidden_states, residual = layer(
                    positions, hidden_states, current_ut, residual
                )
                # [weightless-steer] the one added block: unconditional
                # per-execution-step projection at the post-layer residual
                # stream (zero stack rows make it a numeric no-op on steps
                # we do not steer, so the traced graph is identical for
                # every step set). exec id = ut * num_hidden_layers +
                # global physical layer — the same id as the KV slot
                # (unique_layer_idx = ut_step * num_hidden_layers + base).
                hidden_states = self._steer_post_layer(
                    current_ut * self.config.num_hidden_layers
                    + self.start_layer + _ouro_lidx,
                    hidden_states,
                    residual,
                )
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
        # ---- end upstream forward -----------------------------------------


class SteeredOuroForCausalLM(OuroForCausalLM):
    """OuroForCausalLM whose inner model applies GLP steering.

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
        # steered forward needs are registered below. Weight loading happens
        # after __init__, so the swapped class is in place before any
        # checkpoint tensors arrive.
        self.model.__class__ = SteeredOuroModel
        config = self.model.config
        # The stack is indexed by EXECUTION step: physical layers iterated
        # total_ut_steps times (48 x 4 = 192 on Ouro-2.6B). Upstream reads
        # total_ut_steps with the same default in OuroModel.__init__.
        exec_steps = config.num_hidden_layers * getattr(
            config, "total_ut_steps", 4)
        core = exec_core_from_env(
            hook=SteeredOuroModel.STEER_HOOK,
            exec_steps=exec_steps,
            hidden_size=config.hidden_size,
        )
        if core is None:
            core = SteeringCore.disabled(
                hook=SteeredOuroModel.STEER_HOOK,
                num_layers=exec_steps,
                hidden_size=config.hidden_size,
            )
        core.register_buffers(self.model, vllm_config.model_config.dtype)
        self.model._steer_core = core
        # The upstream constructor captures its bound forward in the compile
        # wrapper. Rebind after the class swap and buffer registration, before
        # warmup can compile or capture a stock, unsteered forward. Unlike
        # glm5next this is LIVE: OuroModel is @support_torch_compile-decorated
        # at v0.26.0, so do_not_compile is set (and False under a compiling
        # config) on every boot.
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
