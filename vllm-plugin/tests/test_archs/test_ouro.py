"""Offline tests for the ouro steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce ouro's looped structure (48-layer block iterated total_ut_steps
  times; here 4 layers x 2 passes = 8 execution steps) and its fused
  add+norm convention (each decoder layer returns (mixer_output, residual),
  fold deferred; a pure-fold norm runs at every pass end). The adapter is
  imported against the stub, the model is built, and its forward output is
  compared against a hand-computed  h <- h - alpha*(h.d)d  per steered
  execution step. This exercises the registry-shadowed class end to end:
  the __class__ swap, the exec-step-sized buffer wiring, the direction.N ->
  exec step N-1 container shift, exec-id indexing across passes, and the
  hidden_states <- h' - residual write-back.
- Structural: the adapter's copied forward loop is pinned verbatim against
  the vendored upstream reference (patches/reference/ouro_v0260.py — the
  hotfix's anchor file), so upstream drift in our copy is caught here
  rather than at serve time.
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
             / "ouro_v0260.py")

HIDDEN = 8
NUM_LAYERS = 4
TOTAL_UT = 2
EXEC_STEPS = NUM_LAYERS * TOTAL_UT   # 8: the looped stand-in for 48 x 4 = 192
# Deterministic per-physical-layer "writes": what each fake decoder layer
# deposits into the residual stream. Scaled by (current_ut + 1) in the fake
# so the UT passes are distinguishable — an exec-id mapping that confused
# passes would fail the comparison.
WRITES = [np.random.default_rng(100 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]


class FakeOuroDecoderLayer(nn.Module):
    """Fused add+norm convention: returns (mixer_output, residual) with the
    fold deferred, so mixer_output + residual is the post-layer stream."""

    def __init__(self, write):
        super().__init__()
        self.register_buffer("write", torch.from_numpy(write))

    def forward(self, positions, hidden_states, current_ut, residual=None):
        h = hidden_states if residual is None else hidden_states + residual
        new = h + self.write * (current_ut + 1)
        return new - h, h


class FakeNorm(nn.Module):
    """The inter-pass norm: a pure fold in the fake (identity beyond it)."""

    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


class FakeOuroModel(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer=0,
                 end_layer=NUM_LAYERS):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(FakeOuroDecoderLayer(w) for w in WRITES)
        self.norm = FakeNorm()
        self.total_ut_steps = getattr(self.config, "total_ut_steps", 4)

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeOuroForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeOuroModel(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)


MAX_NUM_TOKENS = 16
MAX_NUM_REQS = 4


def _vllm_config():
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(hidden_size=HIDDEN,
                                            num_hidden_layers=NUM_LAYERS,
                                            total_ut_steps=TOTAL_UT),
            dtype=torch.float32,
        ),
        # The per-request harness reads the batch shape from here; it is
        # unused on the scalar lane but the real VllmConfig always carries
        # it, so the stub does too.
        scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_TOKENS,
            max_num_seqs=MAX_NUM_REQS,
        ),
    )


def _import_adapter():
    """Install vllm stubs in sys.modules and import the adapter fresh."""
    stubs = {}
    ouro = types.ModuleType("vllm.model_executor.models.ouro")
    ouro.OuroModel = FakeOuroModel
    ouro.OuroForCausalLM = FakeOuroForCausalLM
    stubs["vllm.model_executor.models.ouro"] = ouro
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.ouro", None)
    return importlib.import_module("weightless_steer.archs.ouro")


def manual_forward(embed, dirs, alpha, total_ut=TOTAL_UT,
                   layers=range(NUM_LAYERS)):
    """Hand-computed steered forward over the looped stack.

    dirs keys are 0-based EXECUTION ids (exec = ut * NUM_LAYERS + layer);
    the pass-end norm is a pure fold in the fake, so it drops out.
    `layers` selects which physical layers RUN (pipeline split).
    """
    h = embed.clone()
    for ut in range(total_ut):
        for i in layers:
            h = h + torch.from_numpy(WRITES[i]) * (ut + 1)
            e = ut * NUM_LAYERS + i
            if e in dirs:
                d = dirs[e] / dirs[e].norm()
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
        sys.modules.pop("weightless_steer.archs.ouro", None)
        for name in ("vllm.model_executor.models.ouro", "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, exec_ids, alpha="2.0"):
        """Write a GGUF under the looped convention: direction.N = exec N-1."""
        containers = tuple(e + 1 for e in exec_ids)
        write_gguf(self.path, good_meta(layers=containers, alpha=alpha),
                   good_tensors(layers=containers, width=HIDDEN))

    def file_dirs(self, exec_ids, written=None):
        """{exec_id: direction} for a file written with write_vector(written).

        good_tensors' draws are position-dependent, so the WRITE set (not
        the compared subset) must reproduce the file's tensor values.
        """
        written = exec_ids if written is None else written
        containers = tuple(e + 1 for e in written)
        tensors = good_tensors(layers=containers, width=HIDDEN)
        return {e: torch.from_numpy(tensors[f"direction.{e + 1}"][0])
                for e in exec_ids}

    def build(self, **env):
        with mock.patch.dict(os.environ, env):
            return self.adapter.SteeredOuroForCausalLM(
                vllm_config=_vllm_config())

    def build_control_harness(self, exec_ids):
        """Wire the per-request control primitives directly.

        Serving refuses the per-request env (no runner glue exists), so the
        control-plane tests build a geometry-carrying core themselves and
        re-register the buffers on the already-built model (nn.Module
        register_buffer overwrites same-name buffers, so the second wiring
        replaces the scalar-lane ones).
        """
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        env = {"WEIGHTLESS_STEER_PATH": self.path,
               "WEIGHTLESS_ENABLE_MILESTONE_2": "1"}
        with mock.patch.dict(os.environ, env):
            core = self.adapter.exec_core_from_env(
                hook="residual_stream_post_layer", exec_steps=EXEC_STEPS,
                hidden_size=HIDDEN, max_num_tokens=MAX_NUM_TOKENS,
                max_num_reqs=MAX_NUM_REQS)
        core.register_buffers(model.model, torch.float32)
        model.model._steer_core = core
        return model

    def test_forward_applies_projection_per_execution_step(self):
        # Exec ids span both passes: 1 = (ut0, layer1), 6 = (ut1, layer2).
        exec_ids = (1, 6)
        self.write_vector(exec_ids, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model was swapped onto the steered class and wired with
        # an EXECUTION-step-sized stack (8), not a physical-layer one (4).
        self.assertIsInstance(model.model, self.adapter.SteeredOuroModel)
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (EXEC_STEPS, 1, HIDDEN))
        self.assertAlmostEqual(float(model.model._steer_alpha), 2.0)
        self.assertEqual(sorted(model.model._steer_core.dirs),
                         list(exec_ids))

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(exec_ids), alpha=2.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_container_shift_maps_direction_N_to_exec_N_minus_1(self):
        """THE looped-lane pin: direction.1 steers exec step 0, not step 1.

        A missing shift (the generic container.py reading, N = id) would
        steer exec step 1 instead; an unshifted stack would also have
        rejected the published file's direction.192 as out of range. Both
        failure shapes are caught here by comparing against both candidate
        mappings.
        """
        self.write_vector((0,), alpha="1.0")   # container direction.1
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertEqual(sorted(model.model._steer_core.dirs), [0])
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((0,)), alpha=1.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))
        # The unshifted mapping would steer exec step 1 with the same
        # direction — a different output (exec 1's stream has layer 0's
        # write in it).
        wrong = manual_forward(embed, {1: self.file_dirs((0,))[0]},
                               alpha=1.0)
        self.assertFalse(torch.allclose(out, wrong, atol=1e-4))

    def test_last_execution_step_is_steered(self):
        """Exec step 191 on the real model (7 here): direction.192's row.

        The last step of the last pass feeds the pass-end norm directly;
        a loop that escaped it would leave the vector's final direction
        unapplied.
        """
        exec_ids = (EXEC_STEPS - 1,)
        self.write_vector(exec_ids, alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(exec_ids), alpha=1.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")
        unsteered = manual_forward(embed, {}, alpha=0.0)
        self.assertFalse(torch.allclose(out, unsteered, atol=1e-4))

    def test_steer_layers_env_selects_exec_ids(self):
        """WEIGHTLESS_STEER_LAYERS names 0-based EXECUTION ids (hotfix)."""
        self.write_vector((1, 4, 6), alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_STEER_LAYERS="1,6")
        self.assertEqual(sorted(model.model._steer_core.dirs), [1, 6])
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((1, 6),
                                                    written=(1, 4, 6)),
                              alpha=2.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_alpha_env_overrides_file_default(self):
        self.write_vector((2,), alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_STEER_ALPHA="0.5")
        embed = torch.randn(4, HIDDEN)
        positions = torch.arange(4)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((2,)), alpha=0.5)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_alpha_default_comes_from_the_file(self):
        """The published vector carries glp.alpha_default=1.0; an env-unset
        boot must land on the file's dose, not a hardcoded one."""
        self.write_vector((2,), alpha="1.5")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertAlmostEqual(float(model.model._steer_alpha), 1.5)
        embed = torch.randn(4, HIDDEN)
        positions = torch.arange(4)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((2,)), alpha=1.5)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_unsteered_path_is_passthrough(self):
        model = self.build()  # no WEIGHTLESS_STEER_PATH: disabled core
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (EXEC_STEPS, 1, HIDDEN))
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, {}, alpha=0.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_exec_step_out_of_range_fails_closed(self):
        """direction.9 = exec step 8, but this model has 8 steps (0..7).

        The published file's direction.192 would hit exactly this gate on a
        hypothetical 191-step model; on Ouro-2.6B (192 steps) it is the
        row that makes the full-range file legal.
        """
        self.write_vector((EXEC_STEPS,), alpha="1.0")  # container id 9
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_wrong_hook_env_fails_closed(self):
        self.write_vector((1,), alpha="1.0")
        with self.assertRaisesRegex(RuntimeError, "WEIGHTLESS_STEER_HOOK"):
            self.build(WEIGHTLESS_STEER_PATH=self.path,
                       WEIGHTLESS_STEER_HOOK="residual_stream_pre_layer")

    def test_layer_filter_to_empty_fails_closed(self):
        self.write_vector((1,), alpha="1.0")
        with self.assertRaisesRegex(RuntimeError, "matched no execution"):
            self.build(WEIGHTLESS_STEER_PATH=self.path,
                       WEIGHTLESS_STEER_LAYERS="3")

    def test_bad_vector_fails_closed_at_model_build(self):
        self.write_vector((1, 2), alpha="2.0")
        # Corrupt: declare a container id list that disagrees with the
        # tensors (the looped convention carries CONTAINER ids here).
        write_gguf(self.path,
                   {**good_meta(layers=(2, 3)),
                    "glp.layer_ids_zero_based": "5,6"},
                   good_tensors(layers=(2, 3), width=HIDDEN))
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_wrong_width_fails_closed(self):
        containers = (2,)
        write_gguf(self.path, good_meta(layers=containers),
                   {f"direction.{containers[0]}": (
                       np.random.default_rng(0).standard_normal(
                           HIDDEN * 2).astype(np.float32), 0)})
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_per_request_serving_fails_before_model_allocation(self):
        """Refuse the unwired flag, and refuse it before loading weights."""
        with mock.patch.object(FakeOuroForCausalLM,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build(WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()

    def test_per_request_forward_end_to_end(self):
        """Two requests, two alphas, through the real adapter forward."""
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )

        exec_ids = (1, 6)
        self.write_vector(exec_ids, alpha="2.0")
        model = self.build_control_harness(exec_ids)
        inner = model.model
        self.assertTrue(inner.weightless_per_request)
        self.assertEqual(inner.weightless_steer_layer_ids, exec_ids)

        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=EXEC_STEPS, default_alpha=2.0,
            loaded_layers=exec_ids,
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
        dirs = self.file_dirs(exec_ids)
        # Request "a" is unsteered, request "b" runs at alpha 1.
        want_a = manual_forward(embed[:2], dirs, alpha=0.0)
        want_b = manual_forward(embed[2:], dirs, alpha=1.0)
        self.assertTrue(torch.allclose(out[:2], want_a, atol=1e-4),
                        f"req a max err {(out[:2] - want_a).abs().max()}")
        self.assertTrue(torch.allclose(out[2:], want_b, atol=1e-4),
                        f"req b max err {(out[2:] - want_b).abs().max()}")

    def test_per_request_layer_mask_through_the_forward(self):
        """A per-request layer mask narrows which EXECUTION ids steer."""
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )

        exec_ids = (1, 6)
        self.write_vector(exec_ids, alpha="2.0")
        model = self.build_control_harness(exec_ids)
        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=EXEC_STEPS, default_alpha=2.0,
            loaded_layers=exec_ids,
        )
        plane.build([ScheduledRequest(
            "a", WeightlessResolvedXArgs(layers_override=(6,)),
            token_count=3, start_ordinal=0, prompt_length=3)])
        plane.install(model.model)
        embed = torch.randn(3, HIDDEN)
        positions = torch.arange(3)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        # Only exec step 6 steers, even though step 1 carries a direction.
        want = manual_forward(embed, {6: self.file_dirs(exec_ids)[6]},
                              alpha=2.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_gate_off_registers_no_per_request_buffers(self):
        """The default deployment is untouched by any of this."""
        self.write_vector((1,), alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertFalse(model.model.weightless_per_request)
        for name in ("_steer_alpha_rows", "_steer_slot_rows",
                     "_steer_layer_bank"):
            self.assertNotIn(name, dict(model.model.named_buffers()))

    def test_compilation_rebinds_to_steered_forward(self):
        """The compile wrapper must capture the STEERED forward.

        OuroModel is @support_torch_compile-decorated at v0.26.0, so the
        rebind is live on every compiled boot: upstream's constructor
        captures its bound forward before the class swap, and without the
        rebind the compiled callable runs the stock forward and the whole
        steering lane is a no-op under compilation.
        """
        original_init = FakeOuroModel.__init__
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
        self.write_vector((1, 6), alpha="2.0")
        previous = sys.modules.get("vllm.compilation.wrapper")
        sys.modules["vllm.compilation.wrapper"] = wrapper_module
        try:
            with mock.patch.object(FakeOuroModel, "__init__", init):
                model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        finally:
            if previous is None:
                sys.modules.pop("vllm.compilation.wrapper", None)
            else:
                sys.modules["vllm.compilation.wrapper"] = previous
        self.assertEqual(captured, [FakeOuroModel.forward,
                                    self.adapter.SteeredOuroModel.forward])
        # The stock init's hook was dropped, not left registered
        # alongside the new one for the life of the process.
        self.assertEqual(list(live_hooks), [2])
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        out = model.model._compiled_callable(None, positions, None,
                                             inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((1, 6)), alpha=2.0)
        torch.testing.assert_close(out, want, atol=1e-4, rtol=1e-4)

    def test_global_exec_indexing_under_pipeline_split(self):
        # This rank runs physical layers 2..3 (start_layer=2). exec id =
        # ut * num_hidden_layers + start_layer + local idx must stay GLOBAL:
        # a local-index bug would apply direction.1 to the first local layer.
        # (Upstream's forward has no PP rank branches; the split is simulated
        # by moving the window, as in test_nemotron_h.)
        exec_ids = (2, 3, 6, 7)
        self.write_vector(exec_ids, alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        model.model.start_layer, model.model.end_layer = 2, 4
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(exec_ids), alpha=1.0,
                              layers=range(2, 4))
        self.assertTrue(torch.allclose(out, want, atol=1e-4))


class StructureTests(unittest.TestCase):
    """Pin the adapter's copied forward against the vendored reference."""

    # The layer-loop anchor, identical to the hotfix's ANCHOR_FORWARD.
    LOOP_HEAD = (
        "        for current_ut in range(self.total_ut_steps):\n"
        "            residual = None\n"
    )
    UPSTREAM_LOOP = LOOP_HEAD + (
        "            for layer in self.layers[self.start_layer : self.end_layer]:\n"
        "                hidden_states, residual = layer(\n"
        "                    positions, hidden_states, current_ut, residual\n"
        "                )\n"
    )
    # The hotfix's REPLACEMENT_FORWARD loop head: enumerate added, same call.
    ADAPTER_LOOP = LOOP_HEAD + (
        "            for _ouro_lidx, layer in enumerate(\n"
        "                self.layers[self.start_layer : self.end_layer]\n"
        "            ):\n"
        "                hidden_states, residual = layer(\n"
        "                    positions, hidden_states, current_ut, residual\n"
        "                )\n"
    )
    PASS_END = "            hidden_states, _ = self.norm(hidden_states, residual)\n"

    def test_copied_loop_matches_vendored_reference(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "ouro.py").read_text()
        reference_src = REFERENCE.read_text()
        # The upstream loop exists verbatim in the vendored v0.26.0 file
        # (this is also the hotfix's ANCHOR_FORWARD).
        self.assertIn(self.UPSTREAM_LOOP, reference_src)
        self.assertIn(self.PASS_END, reference_src)
        # The adapter carries the enumerate'd loop (the hotfix's replacement
        # shape) with the steering block immediately after the layer call,
        # and the pass-end norm verbatim.
        self.assertIn(self.ADAPTER_LOOP, adapter_src)
        self.assertIn(
            self.ADAPTER_LOOP
            + "                # [weightless-steer] the one added block",
            adapter_src,
        )
        self.assertIn(self.PASS_END, adapter_src)


if __name__ == "__main__":
    unittest.main()
