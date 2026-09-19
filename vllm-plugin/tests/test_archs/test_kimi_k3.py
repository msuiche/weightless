"""Offline tests for the kimi_k3 steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce kimi_k3's 3-tuple convention (each decoder layer returns
  (hidden_states, prefix_sum, residual); with attn_res on the post-layer
  accumulated stream is prefix_sum + hidden_states, without it hidden_states
  + residual — plain [T, hidden] either way, no widening). The adapter is
  imported against the stub, the model is built, and its forward output is
  compared against a hand-computed  h <- h - alpha*(h.d)d  per steered layer.
  This exercises the registry-shadowed class end to end: the __class__ swap,
  buffer wiring at hidden_size width, global-layer indexing, and the
  write-back against the side stream (prefix_sum resp. residual). Both the
  attn_res (Kimi-K3) and plain-residual (Kimi-Linear-style) conventions are
  covered.
- Structural: the adapter's copied forward loop is pinned verbatim against
  the vendored upstream reference (patches/reference/kimi_k3_nvidia_model.py),
  so upstream drift in our copy is caught here rather than at serve time.
"""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))          # vllm-plugin/
sys.path.insert(0, str(_HERE.parents[1]))          # vllm-plugin/tests/
sys.path.insert(0, str(_HERE.parents[3]))          # weightless_runtime/

from glpfiles import good_meta, good_tensors, write_gguf  # noqa: E402

REFERENCE = (_HERE.parents[3] / "patches" / "reference"
             / "kimi_k3_nvidia_model.py")

HIDDEN = 8
NUM_LAYERS = 4
ATTN_RES_BLOCK = 2
# Deterministic per-layer "writes": what each fake decoder layer deposits
# into the accumulated stream.
WRITES = [np.random.default_rng(100 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]


def _fake_attn_res(prefix_sum, delta, residual, norm_w, proj_w, out_norm_w,
                   *, num_blocks, block_write_idx, eps, output_norm_eps):
    """The final attn_res fold, reduced to the stream it materializes.

    The real op folds the block side stream through learned norms and
    projections; for the steering math what matters is the accumulated
    stream it returns, prefix_sum + delta.
    """
    return prefix_sum + delta


class FakeNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = 1e-5


class FakeProj(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, hidden))


class FakeKimiDecoderLayer(nn.Module):
    """The upstream 3-tuple convention with a trivial sublayer write.

    attn_res mode: the post-layer stream is prefix_sum + hidden_states; the
    layer deposits its write and returns (h_new - prefix_sum, prefix_sum,
    residual). Plain mode: fused add+norm with the fold deferred, returns
    (write, None, h) so hidden_states + residual reproduces h + write.
    """

    def __init__(self, layer_idx, write, use_attn_res):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_attn_res = use_attn_res
        self.register_buffer("write", torch.from_numpy(write))

    def forward(self, positions, hidden_states, residual, prefix_sum=None,
                **kwargs):
        if self.use_attn_res:
            # Layer 0 enters the loop with hidden_states=None (the real
            # attn_res op treats a missing delta as zero).
            h = prefix_sum if hidden_states is None else prefix_sum + hidden_states
            h = h + self.write
            return h - prefix_sum, prefix_sum, residual
        h = hidden_states if residual is None else hidden_states + residual
        new = h + self.write
        return new - h, None, h


class FakeKimiLinearModel(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer=0,
                 end_layer=NUM_LAYERS):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.attn_res_block_size = self.config.attn_res_block_size
        self.use_attn_res = self.attn_res_block_size is not None
        self.use_sequence_parallel = False
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(
            FakeKimiDecoderLayer(i, WRITES[i], self.use_attn_res)
            for i in range(NUM_LAYERS))
        self.num_attn_res_blocks = (
            (end_layer + self.attn_res_block_size - 1)
            // self.attn_res_block_size
            if self.use_attn_res else 0)
        self.aux_hidden_state_layers = ()
        if self.use_attn_res:
            self.output_attn_res_norm = FakeNorm(HIDDEN)
            self.output_attn_res_proj = FakeProj(HIDDEN)

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeKimiLinearForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeKimiLinearModel(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)


class _PPGroup:
    is_first_rank = True
    is_last_rank = True


MAX_NUM_TOKENS = 16
MAX_NUM_REQS = 4


def _vllm_config(attn_res=True):
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(
                hidden_size=HIDDEN,
                num_hidden_layers=NUM_LAYERS,
                attn_res_block_size=ATTN_RES_BLOCK if attn_res else None,
            ),
            dtype=torch.float32,
        ),
        # The adapter reads the batch shape from here to size the
        # per-request control buffers; it is unused on the scalar lane but
        # the real VllmConfig always carries it, so the stub does too.
        scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_TOKENS,
            max_num_seqs=MAX_NUM_REQS,
        ),
    )


def _import_adapter():
    """Install vllm stubs in sys.modules and import the adapter fresh."""
    stubs = {}
    # Parent packages as empty modules: the adapter's verbatim-copied forward
    # references `envs.VLLM_MOE_SKIP_PADDING`, so it binds the name with
    # `import vllm.envs as envs` — a form whose import machinery resolves the
    # TOP-level package (plain `from x.y import z` leaves never need it).
    for name in ("vllm", "vllm.models", "vllm.models.kimi_k3",
                 "vllm.models.kimi_k3.nvidia"):
        stubs.setdefault(name, types.ModuleType(name))
    k3 = types.ModuleType("vllm.models.kimi_k3.nvidia.model")
    k3.KimiLinearModel = FakeKimiLinearModel
    k3.KimiLinearForCausalLM = FakeKimiLinearForCausalLM
    stubs["vllm.models.kimi_k3.nvidia.model"] = k3
    ops = types.ModuleType("vllm.models.kimi_k3.nvidia.ops")
    ops.attn_res = _fake_attn_res
    stubs["vllm.models.kimi_k3.nvidia.ops"] = ops
    sp = types.ModuleType("vllm.models.kimi_k3.nvidia.ops.sequence_parallel")
    sp.sp_shard = lambda x: x
    sp.sp_all_gather = lambda x: x
    sp.sp_reduce_scatter = lambda x: x
    sp.sp_padding_mask = lambda mask, h: mask
    stubs["vllm.models.kimi_k3.nvidia.ops.sequence_parallel"] = sp
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: _PPGroup
    stubs["vllm.distributed"] = distributed
    envs = types.ModuleType("vllm.envs")
    envs.VLLM_MOE_SKIP_PADDING = False
    stubs["vllm.envs"] = envs
    fc = types.ModuleType("vllm.forward_context")
    fc.get_forward_context = lambda: None
    fc.is_forward_context_available = lambda: False
    stubs["vllm.forward_context"] = fc
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    # Child attributes on the parent stubs: `import vllm.envs as envs`
    # resolves `envs` by getattr on the top-level package, which ignores
    # sys.modules when the attribute exists on a real parent.
    stubs["vllm"].envs = envs
    stubs["vllm"].distributed = distributed
    stubs["vllm"].forward_context = fc
    stubs["vllm"].sequence = sequence
    stubs["vllm"].models = stubs["vllm.models"]
    stubs["vllm.models"].kimi_k3 = stubs["vllm.models.kimi_k3"]
    stubs["vllm.models.kimi_k3"].nvidia = stubs["vllm.models.kimi_k3.nvidia"]
    stubs["vllm.models.kimi_k3.nvidia"].model = k3
    stubs["vllm.models.kimi_k3.nvidia"].ops = ops
    ops.sequence_parallel = sp
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.kimi_k3", None)
    return importlib.import_module("weightless_steer.archs.kimi_k3")


def manual_forward(embed, dirs, alpha, layers=range(NUM_LAYERS),
                   steer_layers=None):
    """Hand-computed steered forward over the accumulated stream.

    Both conventions reduce to the same recurrence in the fakes: the stream
    gains the layer's write, then the projection for steered layers; the
    final fold (attn_res or hidden + residual) returns the last stream.
    `layers` selects which decoder layers RUN; `steer_layers`, when given,
    further restricts which of them steer.
    """
    h = embed.clone()
    for i in layers:
        h = h + torch.from_numpy(WRITES[i])
        if i in dirs and (steer_layers is None or i in steer_layers):
            d = dirs[i] / dirs[i].norm()
            h = h - alpha * (h @ d).unsqueeze(-1) * d
    return h


class SteeredForwardTests(unittest.TestCase):
    def setUp(self):
        self.adapter = _import_adapter()
        self.addCleanup(self._unimport)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "vec.gguf")
        for k in ("WEIGHTLESS_STEER_PATH", "WEIGHTLESS_STEER_ALPHA",
                  "WEIGHTLESS_STEER_LAYERS", "WEIGHTLESS_STEER_HOOK",
                  "WEIGHTLESS_ENABLE_MILESTONE_2"):
            os.environ.pop(k, None)

    def _unimport(self):
        sys.modules.pop("weightless_steer.archs.kimi_k3", None)
        for name in ("vllm.models.kimi_k3.nvidia.model",
                     "vllm.models.kimi_k3.nvidia.ops.sequence_parallel",
                     "vllm.models.kimi_k3.nvidia.ops",
                     "vllm.models.kimi_k3.nvidia",
                     "vllm.models.kimi_k3", "vllm.models",
                     "vllm.distributed", "vllm.envs", "vllm.forward_context",
                     "vllm.sequence", "vllm"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="1.0", width=HIDDEN):
        write_gguf(self.path, good_meta(layers=layers, alpha=alpha),
                   good_tensors(layers=layers, width=width))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=HIDDEN)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, attn_res=True, **env):
        with mock.patch.dict(os.environ, env):
            return self.adapter.SteeredKimiLinearForCausalLM(
                vllm_config=_vllm_config(attn_res=attn_res))

    def build_control_harness(self):
        """Wire the control primitives directly.

        Serving refuses the per-request env (no runner glue exists), so
        the control-plane tests drive _wire_steering themselves.
        """
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        env = {"WEIGHTLESS_STEER_PATH": self.path,
               "WEIGHTLESS_ENABLE_MILESTONE_2": "1"}
        with mock.patch.dict(os.environ, env):
            model.model._wire_steering(dtype=torch.float32,
                                       max_num_tokens=MAX_NUM_TOKENS,
                                       max_num_reqs=MAX_NUM_REQS)
        return model

    def test_forward_applies_projection_per_layer(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model was swapped onto the steered class and wired at
        # the plain stream width (no widening on this arch).
        self.assertIsInstance(model.model,
                              self.adapter.SteeredKimiLinearModel)
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (NUM_LAYERS, 1, HIDDEN))
        self.assertAlmostEqual(float(model.model._steer_alpha), 2.0)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=2.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_forward_plain_residual_mode(self):
        """Without attn_res the stream is hidden_states + residual."""
        layers = (1, 3)
        self.write_vector(layers, alpha="2.0")
        model = self.build(attn_res=False, WEIGHTLESS_STEER_PATH=self.path)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=2.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_last_layer_is_steered(self):
        """GLP-92 covers L92, the last of 93: the final layer must not
        escape steering (the glm5next L44 trap class). kimi_k3 has no
        deferred terminal contract, so this pins the plain-loop case."""
        layers = (NUM_LAYERS - 1,)
        self.write_vector(layers, alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=1.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")
        unsteered = manual_forward(embed, {}, alpha=0.0)
        self.assertFalse(torch.allclose(out, unsteered, atol=1e-4))

    def test_per_request_forward_end_to_end(self):
        """Two requests, two alphas, through the real adapter forward."""
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )

        layers = (1, 3)
        self.write_vector(layers, alpha="2.0")
        model = self.build_control_harness()
        inner = model.model
        self.assertTrue(inner.weightless_per_request)
        self.assertEqual(inner.weightless_steer_layer_ids, layers)

        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=NUM_LAYERS, default_alpha=2.0, loaded_layers=layers,
        )
        plane.build([
            ScheduledRequest("a", WeightlessResolvedXArgs(alpha_override=0.0),
                             token_count=2, start_ordinal=0, prompt_length=2),
            ScheduledRequest("b", WeightlessResolvedXArgs(alpha_override=1.0),
                             token_count=3, start_ordinal=0, prompt_length=3),
        ])
        plane.install(inner)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        dirs = self.file_dirs(layers)
        # Request "a" is unsteered, request "b" runs at alpha 1.
        want_a = manual_forward(embed[:2], dirs, alpha=0.0)
        want_b = manual_forward(embed[2:], dirs, alpha=1.0)
        self.assertTrue(torch.allclose(out[:2], want_a, atol=1e-4),
                        f"req a max err {(out[:2] - want_a).abs().max()}")
        self.assertTrue(torch.allclose(out[2:], want_b, atol=1e-4),
                        f"req b max err {(out[2:] - want_b).abs().max()}")

    def test_per_request_serving_fails_before_model_allocation(self):
        """Refuse the unwired flag, and refuse it before loading weights."""
        with mock.patch.object(FakeKimiLinearForCausalLM,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build(WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()

    def test_compilation_rebinds_to_steered_forward(self):
        """The compile wrapper must capture the STEERED forward.

        kimi_k3 carries no @support_torch_compile today, so the rebind is
        dormant; this pins its behaviour for the day upstream decorates the
        class — without it the compiled callable would run the stock
        forward and compiled serving would be silently unsteered.
        """
        original_init = FakeKimiLinearModel.__init__
        captured = []
        # Mirrors the real wrapper: each __init__ registers a dynamo
        # bytecode hook in a process-global registry, and cleanup() removes
        # only the handle that instance last stored.
        live_hooks = {}

        class Wrapper:
            def __init__(self, compile_prefix="", is_encoder=False):
                captured.append(self.forward.__func__)
                self._hook_handle = len(captured)
                live_hooks[self._hook_handle] = self
                self._compiled_callable = torch.compile(
                    self.forward, backend="eager", fullgraph=True)

            def cleanup(self):
                live_hooks.pop(getattr(self, "_hook_handle", None), None)

        def init(model, **kwargs):
            original_init(model, **kwargs)
            model.do_not_compile = False
            model._compile_prefix = ""
            model._is_encoder = False
            Wrapper.__init__(model)

        wrapper_module = types.ModuleType("vllm.compilation.wrapper")
        wrapper_module.TorchCompileWithNoGuardsWrapper = Wrapper
        self.write_vector((1, 3), alpha="2.0")
        previous = sys.modules.get("vllm.compilation.wrapper")
        sys.modules["vllm.compilation.wrapper"] = wrapper_module
        try:
            with mock.patch.object(FakeKimiLinearModel, "__init__", init):
                model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        finally:
            if previous is None:
                sys.modules.pop("vllm.compilation.wrapper", None)
            else:
                sys.modules["vllm.compilation.wrapper"] = previous
        self.assertEqual(captured, [FakeKimiLinearModel.forward,
                                    self.adapter.SteeredKimiLinearModel.forward])
        # The stock init's hook was dropped, not left registered
        # alongside the new one for the life of the process.
        self.assertEqual(list(live_hooks), [2])
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        out = model.model._compiled_callable(None, positions, None,
                                             inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((1, 3)), alpha=2.0)
        torch.testing.assert_close(out, want, atol=1e-4, rtol=1e-4)

    def test_alpha_env_overrides_file_default(self):
        layers = (2,)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_STEER_ALPHA="0.5")
        embed = torch.randn(4, HIDDEN)
        positions = torch.arange(4)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=0.5)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_unsteered_path_is_passthrough(self):
        model = self.build()  # no WEIGHTLESS_STEER_PATH: disabled core
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, {}, alpha=0.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_global_layer_indexing_under_pipeline_split(self):
        # This rank runs global layers 2..3 (start_layer=2). A vector
        # covering layers 1..3 must steer 2 and 3 by GLOBAL id — a
        # local-index bug would apply direction.1 to the first local layer.
        layers = (1, 2, 3)
        self.write_vector(layers, alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        # Move this rank's window onto global layers 2..3 (make_layers sets
        # these on the inner model at build; the dense stack stays global).
        model.model.start_layer, model.model.end_layer = 2, 4
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=1.0,
                              layers=range(2, 4))
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_gate_off_registers_no_per_request_buffers(self):
        """The default deployment is untouched by any of this."""
        self.write_vector((1,), alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertFalse(model.model.weightless_per_request)
        for name in ("_steer_alpha_rows", "_steer_slot_rows",
                     "_steer_layer_bank"):
            self.assertNotIn(name, dict(model.model.named_buffers()))

    def test_widened_stream_vector_fails_closed(self):
        """A hyper-connection-width vector is not this arch's stream."""
        self.write_vector((1, 2), alpha="1.0", width=HIDDEN * 4)
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_bad_vector_fails_closed_at_model_build(self):
        self.write_vector((1, 2), alpha="1.0")
        # Corrupt: declare a layer set that disagrees with the tensors.
        write_gguf(self.path,
                   {**good_meta(layers=(1, 2)),
                    "glp.layer_ids_zero_based": "5,6"},
                   good_tensors(layers=(1, 2), width=HIDDEN))
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)


class StructureTests(unittest.TestCase):
    """Pin the adapter's copied forward against the vendored reference."""

    # The model layer-loop anchor, identical to the hotfix's capture anchor.
    LOOP_ANCHOR = (
        "        for layer_idx, layer in enumerate(\n"
        "            self.layers[self.start_layer : self.end_layer],\n"
        "            start=self.start_layer,\n"
        "        ):\n"
        "            hidden_states, prefix_sum, residual = layer(\n"
        "                positions=positions,\n"
        "                hidden_states=hidden_states,\n"
        "                prefix_sum=prefix_sum,\n"
        "                residual=residual,\n"
        "            )\n"
    )
    # The final attn_res fold head — pins the tail of the copied forward.
    FINAL_FOLD_ANCHOR = (
        "        if self.use_attn_res:\n"
        "            assert prefix_sum is not None\n"
        "            hidden_states = attn_res(\n"
        "                prefix_sum,\n"
        "                hidden_states,\n"
        "                residual,\n"
    )

    def test_copied_loop_matches_vendored_reference(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "kimi_k3.py").read_text()
        reference_src = REFERENCE.read_text()
        # The anchors exist verbatim in BOTH the adapter's forward copy and
        # the upstream reference it was copied from.
        self.assertIn(self.LOOP_ANCHOR, reference_src)
        self.assertIn(self.LOOP_ANCHOR, adapter_src)
        self.assertIn(self.FINAL_FOLD_ANCHOR, reference_src)
        self.assertIn(self.FINAL_FOLD_ANCHOR, adapter_src)
        # The steering block sits immediately after the layer call.
        self.assertIn(
            self.LOOP_ANCHOR
            + "            # [weightless-steer] the one added block",
            adapter_src,
        )


if __name__ == "__main__":
    unittest.main()
