"""Offline tests for the nemotron_h steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce nemotron_h's fused add+norm convention (each decoder layer
  returns (mixer_output, residual), fold deferred). The adapter is
  imported against the stub, the model is built, and its forward output is
  compared against a hand-computed  h <- h - alpha*(h.d)d  per steered
  layer. This exercises the registry-shadowed class end to end: the
  __class__ swap, buffer wiring, global-layer indexing, and the
  hidden_states <- h' - residual write-back.
- Structural: the adapter's copied forward loop is pinned verbatim against
  the vendored upstream reference
  (patches/reference/nemotron_h_v0280.py), so upstream drift in our copy
  is caught here rather than at serve time.
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

from glpfiles import good_meta, good_tensors, write_gguf  # noqa: E402

REFERENCE = (_HERE.parents[3] / "patches" / "reference"
             / "nemotron_h_v0280.py")

HIDDEN = 8
NUM_LAYERS = 4
# Deterministic per-layer "writes": what each fake decoder layer deposits
# into the residual stream.
WRITES = [np.random.default_rng(100 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]


class FakeDecoderLayer(nn.Module):
    """Fused add+norm convention: returns (mixer_output, residual) with the
    fold deferred, so mixer_output + residual is the post-layer stream."""

    def __init__(self, write):
        super().__init__()
        self.register_buffer("write", torch.from_numpy(write))

    def forward(self, positions, hidden_states, residual):
        h = hidden_states if residual is None else hidden_states + residual
        new = h + self.write
        return new - h, h


class FakeNormF(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


class FakeNemotronHModel(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer=0,
                 end_layer=NUM_LAYERS):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(FakeDecoderLayer(w) for w in WRITES)
        self.norm_f = FakeNormF()

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def _maybe_add_hidden_state(self, aux, idx, hidden_states, residual):
        return aux

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeNemotronHForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeNemotronHModel(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)


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
    nemotron = types.ModuleType("vllm.model_executor.models.nemotron_h")
    nemotron.NemotronHModel = FakeNemotronHModel
    nemotron.NemotronHForCausalLM = FakeNemotronHForCausalLM
    stubs["vllm.model_executor.models.nemotron_h"] = nemotron
    distributed = types.ModuleType("vllm.distributed.parallel_state")
    distributed.get_pp_group = lambda: _PPGroup
    stubs["vllm.distributed.parallel_state"] = distributed
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.nemotron_h", None)
    return importlib.import_module("weightless_steer.archs.nemotron_h")


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
        sys.modules.pop("weightless_steer.archs.nemotron_h", None)
        for name in ("vllm.model_executor.models.nemotron_h",
                     "vllm.distributed.parallel_state", "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="2.0"):
        write_gguf(self.path, good_meta(layers=layers, alpha=alpha),
                   good_tensors(layers=layers, width=HIDDEN))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=HIDDEN)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, **env):
        with mock.patch.dict(os.environ, env):
            return self.adapter.SteeredNemotronHForCausalLM(
                vllm_config=_vllm_config())

    def test_forward_applies_projection_per_layer(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model was swapped onto the steered class and wired.
        self.assertIsInstance(model.model, self.adapter.SteeredNemotronHModel)
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (NUM_LAYERS, 1, HIDDEN))
        self.assertAlmostEqual(float(model.model._steer_alpha), 2.0)

        embed = torch.randn(5, HIDDEN)
        with torch.no_grad():
            out = model(None, None, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=2.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_per_request_forward_end_to_end(self):
        """Two requests, two alphas, through the real adapter forward.

        Covers the whole chain the bare-core tests skip: the gate env, the
        adapter passing batch geometry, _wire_steering registering the
        rows, the control plane building them, and the steered forward
        loop reading them at every layer.
        """
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )

        layers = (1, 3)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_ENABLE_MILESTONE_2="1")
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
        with torch.no_grad():
            out = model(None, None, inputs_embeds=embed)
        dirs = self.file_dirs(layers)
        # Request "a" is unsteered, request "b" runs at alpha 1.
        want_a = manual_forward(embed[:2], dirs, alpha=0.0)
        want_b = manual_forward(embed[2:], dirs, alpha=1.0)
        self.assertTrue(torch.allclose(out[:2], want_a, atol=1e-4),
                        f"req a max err {(out[:2] - want_a).abs().max()}")
        self.assertTrue(torch.allclose(out[2:], want_b, atol=1e-4),
                        f"req b max err {(out[2:] - want_b).abs().max()}")
        # And the two really did differ.
        self.assertFalse(torch.allclose(out[:2], embed[:2] * 0 + out[2:3]))

    def test_per_request_layer_mask_through_the_forward(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_ENABLE_MILESTONE_2="1")
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )
        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=NUM_LAYERS, default_alpha=2.0, loaded_layers=layers,
        )
        plane.build([ScheduledRequest(
            "a", WeightlessResolvedXArgs(layers_override=(3,)),
            token_count=3, start_ordinal=0, prompt_length=3)])
        plane.install(model.model)
        embed = torch.randn(3, HIDDEN)
        with torch.no_grad():
            out = model(None, None, inputs_embeds=embed)
        # Only layer 3 steers, even though layer 1 carries a direction.
        want = manual_forward(embed, self.file_dirs(layers), alpha=2.0,
                              steer_layers=(3,))
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

    def test_alpha_env_overrides_file_default(self):
        layers = (2,)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_STEER_ALPHA="0.5")
        embed = torch.randn(4, HIDDEN)
        with torch.no_grad():
            out = model(None, None, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=0.5)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_unsteered_path_is_passthrough(self):
        model = self.build()  # no WEIGHTLESS_STEER_PATH: disabled core
        embed = torch.randn(5, HIDDEN)
        with torch.no_grad():
            out = model(None, None, inputs_embeds=embed)
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
        with torch.no_grad():
            out = model(None, None, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=1.0,
                              layers=range(2, 4))
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_bad_vector_fails_closed_at_model_build(self):
        self.write_vector((1, 2), alpha="2.0")
        # Corrupt: declare a layer set that disagrees with the tensors.
        write_gguf(self.path,
                   {**good_meta(layers=(1, 2)),
                    "glp.layer_ids_zero_based": "5,6"},
                   good_tensors(layers=(1, 2), width=HIDDEN))
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)


class StructureTests(unittest.TestCase):
    """Pin the adapter's copied forward against the vendored reference."""

    # The layer-loop anchor, identical to the hotfix's ANCHOR_FORWARD.
    LOOP_ANCHOR = (
        "        for idx, layer in enumerate(\n"
        "            islice(self.layers, self.start_layer, self.end_layer)\n"
        "        ):\n"
        "            hidden_states, residual = layer(\n"
        "                positions=positions,\n"
        "                hidden_states=hidden_states,\n"
        "                residual=residual,\n"
        "            )\n"
    )

    def test_copied_loop_matches_vendored_reference(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "nemotron_h.py").read_text()
        reference_src = REFERENCE.read_text()
        # The anchor exists verbatim in BOTH the adapter's forward copy and
        # the upstream reference it was copied from.
        self.assertIn(self.LOOP_ANCHOR, reference_src)
        self.assertIn(self.LOOP_ANCHOR, adapter_src)
        # The steering line sits immediately after the layer call.
        self.assertIn(
            self.LOOP_ANCHOR
            + "            # [weightless-steer] the one added line",
            adapter_src,
        )


if __name__ == "__main__":
    unittest.main()
