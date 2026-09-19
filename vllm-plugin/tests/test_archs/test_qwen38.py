"""Offline tests for the qwen38 (Qwen3.8-27B / qwen3_5) steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce qwen3_5's fused add+norm convention (each decoder layer returns
  (mixer_output, residual) with the fold deferred, so the post-layer stream
  is hidden_states + residual). The adapter is imported against the stub,
  the model is built through BOTH registered wrappers (the text-only
  Qwen3_5ForCausalLM and the multimodal Qwen3_5ForConditionalGeneration —
  the arch the real checkpoint declares), and forward outputs are compared
  against a hand-computed  h <- h - alpha*(h.d)d  per steered layer. This
  exercises the registry-shadowed classes end to end: the __class__ swap,
  buffer wiring, global-layer indexing, and the h' - residual write-back.
- Structural: the adapter's copied forward is pinned against the vendored
  upstream reference (patches/reference/qwen3_next_v0280.py) AND against
  the hotfix's v0.28.0 anchor constant (imported, not re-read as text — the
  anchor is a string literal there), so drift in any of the three copies is
  caught here rather than at serve time.
"""
import importlib
import importlib.util
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
             / "qwen3_next_v0280.py")
HOTFIX = (_HERE.parents[3] / "patches"
          / "hotfix-qwen38-steering-projective.py")

HIDDEN = 8
NUM_LAYERS = 4
# Deterministic per-layer "writes": what each fake decoder layer deposits
# into the residual stream.
WRITES = [np.random.default_rng(100 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]


class FakeQwen3DecoderLayer(nn.Module):
    """Fused add+norm convention: returns (mixer_output, residual) with the
    fold deferred, so mixer_output + residual is the post-layer stream. The
    linear/full mixer distinction lives inside the layer and is invisible at
    this site, so one fake covers both."""

    def __init__(self, write):
        super().__init__()
        self.register_buffer("write", torch.from_numpy(write))

    def forward(self, positions, hidden_states, residual):
        h = hidden_states if residual is None else hidden_states + residual
        new = h + self.write
        return new - h, h


class FakeNorm(nn.Module):
    """The final fused norm: folds and returns (folded, None)."""

    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


class FakeQwen3_5Model(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer=0,
                 end_layer=NUM_LAYERS):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(FakeQwen3DecoderLayer(w) for w in WRITES)
        self.norm = FakeNorm()
        self.use_sequence_parallel = False

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def _maybe_add_hidden_state(self, aux, idx, hidden_states, residual):
        return aux

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeQwen3_5ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeQwen3_5Model(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)


class FakeQwen3_5ForConditionalGeneration(nn.Module):
    """The arch the real checkpoint declares: builds its language model
    DIRECTLY (as upstream's wrapper does), so only shadowing THIS class
    reaches the served model."""

    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__()
        self.language_model = FakeQwen3_5ForCausalLM(
            vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.language_model.model(input_ids, positions,
                                         intermediate_tensors, inputs_embeds)


class _PPGroup:
    is_first_rank = True
    is_last_rank = True


MAX_NUM_TOKENS = 16
MAX_NUM_REQS = 4


def _vllm_config():
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(hidden_size=HIDDEN,
                                            num_hidden_layers=NUM_LAYERS),
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
    qwen35 = types.ModuleType("vllm.model_executor.models.qwen3_5")
    qwen35.Qwen3_5Model = FakeQwen3_5Model
    qwen35.Qwen3_5ForCausalLM = FakeQwen3_5ForCausalLM
    qwen35.Qwen3_5ForConditionalGeneration = (
        FakeQwen3_5ForConditionalGeneration)
    stubs["vllm.model_executor.models.qwen3_5"] = qwen35
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: _PPGroup
    distributed.tensor_model_parallel_all_gather = lambda x, dim: x
    stubs["vllm.distributed"] = distributed
    utils = types.ModuleType("vllm.model_executor.models.utils")
    utils.sequence_parallel_chunk = lambda x: x
    stubs["vllm.model_executor.models.utils"] = utils
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.qwen38", None)
    return importlib.import_module("weightless_steer.archs.qwen38")


def manual_forward(embed, dirs, alpha, layers=range(NUM_LAYERS),
                   steer_layers=None):
    """Hand-computed steered forward: fold, deposit, project per layer.

    `layers` selects which decoder layers RUN (and so deposit their write);
    `steer_layers`, when given, further restricts which of them steer. The
    two are different questions -- a per-request layer mask narrows the
    second without touching the first.
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
        sys.modules.pop("weightless_steer.archs.qwen38", None)
        for name in ("vllm.model_executor.models.qwen3_5",
                     "vllm.distributed",
                     "vllm.model_executor.models.utils",
                     "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="1.0", width=HIDDEN):
        write_gguf(self.path, good_meta(layers=layers, alpha=alpha),
                   good_tensors(layers=layers, width=width))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=HIDDEN)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, cls_name="SteeredQwen3_5ForCausalLM", **env):
        with mock.patch.dict(os.environ, env):
            return getattr(self.adapter, cls_name)(vllm_config=_vllm_config())

    def inner(self, model):
        """The Qwen3_5Model, whichever wrapper shadow built it."""
        if isinstance(model, FakeQwen3_5ForConditionalGeneration):
            return model.language_model.model
        return model.model

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
        self.write_vector(layers, alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model was swapped onto the steered class and wired.
        self.assertIsInstance(model.model, self.adapter.SteeredQwen3_5Model)
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (NUM_LAYERS, 1, HIDDEN))
        self.assertAlmostEqual(float(model.model._steer_alpha), 1.0)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=1.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_multimodal_wrapper_is_steered(self):
        """The arch the checkpoint actually declares: shadowing the wrapper
        must reach language_model.model (built directly, not via registry)."""
        layers = (1, 3)
        self.write_vector(layers, alpha="1.0")
        model = self.build("SteeredQwen3_5ForConditionalGeneration",
                           WEIGHTLESS_STEER_PATH=self.path)
        inner = model.language_model.model
        self.assertIsInstance(inner, self.adapter.SteeredQwen3_5Model)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=1.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_per_request_forward_end_to_end(self):
        """Two requests, two alphas, through the real adapter forward."""
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )

        layers = (1, 3)
        self.write_vector(layers, alpha="1.0")
        model = self.build_control_harness()
        inner = model.model
        self.assertTrue(inner.weightless_per_request)
        self.assertEqual(inner.weightless_steer_layer_ids, layers)

        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=NUM_LAYERS, default_alpha=1.0, loaded_layers=layers,
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

    def test_per_request_layer_mask_through_the_forward(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="1.0")
        model = self.build_control_harness()
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )
        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=NUM_LAYERS, default_alpha=1.0, loaded_layers=layers,
        )
        plane.build([ScheduledRequest(
            "a", WeightlessResolvedXArgs(layers_override=(3,)),
            token_count=3, start_ordinal=0, prompt_length=3)])
        plane.install(model.model)
        embed = torch.randn(3, HIDDEN)
        positions = torch.arange(3)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        # Only layer 3 steers, even though layer 1 carries a direction.
        want = manual_forward(embed, self.file_dirs(layers), alpha=1.0,
                              steer_layers=(3,))
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_per_request_serving_fails_before_model_allocation(self):
        """Refuse the unwired flag, and refuse it before loading weights —
        on BOTH wrappers (the multimodal one is the served arch)."""
        with mock.patch.object(FakeQwen3_5ForCausalLM,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build(WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()
        with mock.patch.object(FakeQwen3_5ForConditionalGeneration,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build("SteeredQwen3_5ForConditionalGeneration",
                           WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()

    def test_compilation_rebinds_to_steered_forward(self):
        """The compile wrapper must capture the STEERED forward.

        Qwen3_5Model IS @support_torch_compile'd (the glm5next lane's
        dormant rebind is live here): without the rebind the compiled
        callable would run the stock forward and compiled serving would be
        silently unsteered.
        """
        original_init = FakeQwen3_5Model.__init__
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
        self.write_vector((1, 3), alpha="1.0")
        previous = sys.modules.get("vllm.compilation.wrapper")
        sys.modules["vllm.compilation.wrapper"] = wrapper_module
        try:
            with mock.patch.object(FakeQwen3_5Model, "__init__", init):
                model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        finally:
            if previous is None:
                sys.modules.pop("vllm.compilation.wrapper", None)
            else:
                sys.modules["vllm.compilation.wrapper"] = previous
        self.assertEqual(captured, [FakeQwen3_5Model.forward,
                                    self.adapter.SteeredQwen3_5Model.forward])
        # The stock init's hook was dropped, not left registered
        # alongside the new one for the life of the process.
        self.assertEqual(list(live_hooks), [2])
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        out = model.model._compiled_callable(None, positions, None,
                                             inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((1, 3)), alpha=1.0)
        torch.testing.assert_close(out, want, atol=1e-4, rtol=1e-4)

    def test_alpha_env_overrides_file_default(self):
        layers = (2,)
        self.write_vector(layers, alpha="1.0")
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
        """A glm5next-lane (widened-stream) vector is not this arch's
        stream: the width guard refuses it at model build."""
        self.write_vector((1, 2), alpha="1.0", width=4 * HIDDEN)
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
    """Pin the adapter's copied forward against the vendored v0.28.0
    reference and the hotfix's v0.28.0 anchor — three-way agreement."""

    # The loop head, present verbatim in reference and adapter (the hotfix
    # anchors start at the layer call, one block later).
    LOOP_HEAD = (
        "        for layer_idx, layer in enumerate(\n"
        "            islice(self.layers, self.start_layer, self.end_layer),\n"
        "            start=self.start_layer,\n"
        "        ):\n"
    )
    STEER_MARKER = "            # [weightless-steer] the one added block"

    def test_copied_forward_matches_reference_and_hotfix(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "qwen38.py").read_text()
        reference_src = REFERENCE.read_text()
        # Import the hotfix for its anchor CONSTANTS — the file holds them
        # as string literals, so reading it as text would compare escapes.
        spec = importlib.util.spec_from_file_location(
            "hotfix_qwen38_structure", HOTFIX)
        hotfix = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hotfix)
        v28 = hotfix.NEXT_ANCHOR_FORWARD_V28

        # The hotfix's v0.28.0 anchor exists verbatim in the vendored
        # reference (hotfix <-> reference agreement).
        self.assertIn(v28, reference_src)
        # The reference's loop head immediately precedes the anchor.
        self.assertIn(self.LOOP_HEAD + v28, reference_src)

        # The adapter carries the same loop head + layer call, with the
        # steering block inserted between the call and the aux tail.
        call, _, tail = v28.partition(
            "            self._maybe_add_hidden_state")
        self.assertTrue(tail)
        self.assertIn(self.LOOP_HEAD + call, adapter_src)
        self.assertIn(call + self.STEER_MARKER, adapter_src)
        self.assertIn(self.STEER_MARKER, adapter_src)
        self.assertIn("            self._maybe_add_hidden_state" + tail,
                      adapter_src)
        # And the steered call itself targets the GLOBAL layer id.
        self.assertIn(
            "            # [weightless-steer] the one added block: "
            "unconditional per-layer\n"
            "            # projection at the post-layer residual stream "
            "(zero stack rows\n"
            "            # make it a numeric no-op on unsteered layers).\n"
            "            hidden_states = self._steer_post_layer(\n"
            "                layer_idx, hidden_states, residual\n"
            "            )\n",
            adapter_src,
        )

    def test_forward_head_and_tail_survive_verbatim(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "qwen38.py").read_text()
        reference_src = REFERENCE.read_text()
        for block in (
            "        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)\n",
            "        hidden_states, _ = self.norm(hidden_states, residual)\n",
            "            hidden_states = sequence_parallel_chunk(hidden_states)\n",
        ):
            self.assertIn(block, reference_src)
            self.assertIn(block, adapter_src)


if __name__ == "__main__":
    unittest.main()
