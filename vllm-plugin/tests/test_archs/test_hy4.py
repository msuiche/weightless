"""Offline tests for the hy4 (Tencent Hy4-preview, hy_v4) steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce hy_v4's iHC convention (each decoder layer carries the FULL
  multi-stream state in hidden_states as [T, hc_mult, hidden], merges each
  sub-block's post immediately, and returns (hidden_states, None)). The
  adapter is imported against the stub, the model is built, and its forward
  output is compared against a hand-computed  h <- h - alpha*(h.d)d  per
  steered layer, per iHC stream (one hidden_size-wide direction, ellipsis
  contraction over the last axis only). This exercises the registry-shadowed
  class end to end: the __class__ swap, buffer wiring at hidden_size (NOT
  the flattened hc*hidden), and global-layer indexing via layer.layer_idx.
- Structural: the adapter's copied forward loop is pinned verbatim against
  the vendored upstream reference (patches/reference/hy_v4_nvidia_model.py,
  byte-identical to the day-0 vllm/vllm-openai:hy4-preview image file), so
  upstream drift in our copy is caught here rather than at serve time.
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
             / "hy_v4_nvidia_model.py")

HIDDEN = 8
HC = 3                      # hc_mult; the real arch runs 4
NUM_LAYERS = 4
# Deterministic per-layer sublayer "writes" and per-stream post gates, so
# the hc streams diverge even unsteered and per-stream steering is
# load-bearing in every comparison.
ATTN_WRITES = [np.random.default_rng(200 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
FFN_WRITES = [np.random.default_rng(300 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
ATTN_GATES = [np.random.default_rng(400 + i).standard_normal(HC).astype(
    np.float32) for i in range(NUM_LAYERS)]
FFN_GATES = [np.random.default_rng(500 + i).standard_normal(HC).astype(
    np.float32) for i in range(NUM_LAYERS)]


class FakeHYV4DecoderLayer(nn.Module):
    """The upstream iHC convention with trivial sublayers.

    Each boundary reduces the streams (mean), runs the sub-block (a fixed
    write), and scatters back post-gated per stream — the shape of
    HYV4HCLayer.pre/post. iHC layers return (stream, None); the stream is
    the post-layer residual stream. The non-iHC branch mirrors upstream's
    _forward_normal: returns (mlp_output, residual) with the fold deferred.
    """

    def __init__(self, layer_idx, enable_ihc=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.enable_ihc = enable_ihc
        self.register_buffer("attn_write",
                             torch.from_numpy(ATTN_WRITES[layer_idx]))
        self.register_buffer("ffn_write",
                             torch.from_numpy(FFN_WRITES[layer_idx]))
        self.register_buffer("attn_gate",
                             torch.from_numpy(ATTN_GATES[layer_idx]))
        self.register_buffer("ffn_gate",
                             torch.from_numpy(FFN_GATES[layer_idx]))

    def forward(self, positions, hidden_states, residual):
        if self.enable_ihc:
            s = hidden_states
            if s.dim() == 2:  # prepare_input: broadcast over the channels
                s = s.unsqueeze(1).expand(-1, HC, -1).contiguous()
            attn_out = s.mean(dim=1) + self.attn_write
            s = s + attn_out.unsqueeze(1) * self.attn_gate.view(1, HC, 1)
            mlp_out = s.mean(dim=1) + self.ffn_write
            s = s + mlp_out.unsqueeze(1) * self.ffn_gate.view(1, HC, 1)
            return s, None
        if residual is not None:
            hidden_states = hidden_states + residual
        residual = hidden_states
        attn_out = hidden_states + self.attn_write
        hidden_states = attn_out + residual
        residual = hidden_states
        mlp_out = hidden_states + self.ffn_write
        return mlp_out, residual


class FakeHCHead(nn.Module):
    """Stands in for HYV4HCHeadLayer: merge the channels to one stream."""

    def forward(self, hidden_states):
        return hidden_states.mean(dim=1)


class FakeHYV4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer=0,
                 end_layer=NUM_LAYERS):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.enable_ihc = self.config.enable_ihc
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(
            FakeHYV4DecoderLayer(i, enable_ihc=self.enable_ihc)
            for i in range(NUM_LAYERS)
        )
        self.hc_head = FakeHCHead()
        self.norm = nn.Identity()

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeHYV4ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeHYV4Model(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)


class _PPGroup:
    is_first_rank = True
    is_last_rank = True


MAX_NUM_TOKENS = 16
MAX_NUM_REQS = 4


def _vllm_config(enable_ihc=True):
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                hidden_size=HIDDEN,
                num_hidden_layers=NUM_LAYERS,
                enable_ihc=enable_ihc,
                hc_mult=HC,
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
    hy = types.ModuleType("vllm.models.hy_v4.nvidia.model")
    hy.HYV4Model = FakeHYV4Model
    hy.HYV4ForCausalLM = FakeHYV4ForCausalLM
    stubs["vllm.models.hy_v4.nvidia.model"] = hy
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: _PPGroup
    stubs["vllm.distributed"] = distributed
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.hy4", None)
    return importlib.import_module("weightless_steer.archs.hy4")


def manual_forward(embed, dirs, alpha, layers=range(NUM_LAYERS),
                   steer_layers=None):
    """Hand-computed steered forward over the materialized iHC stream.

    Per running layer: reduce, attn write, per-stream gated scatter, reduce,
    ffn write, per-stream gated scatter — then the projection applied to
    EACH stream independently with one hidden-wide direction (the hotfix's
    ellipsis semantics). The head merges streams by mean.
    """
    if embed.dim() == 2:
        s = embed.unsqueeze(1).expand(-1, HC, -1).clone()
    else:
        s = embed.clone()
    for i in layers:
        attn_out = s.mean(dim=1) + torch.from_numpy(ATTN_WRITES[i])
        s = s + attn_out.unsqueeze(1) * torch.from_numpy(
            ATTN_GATES[i]).view(1, HC, 1)
        mlp_out = s.mean(dim=1) + torch.from_numpy(FFN_WRITES[i])
        s = s + mlp_out.unsqueeze(1) * torch.from_numpy(
            FFN_GATES[i]).view(1, HC, 1)
        if i in dirs and (steer_layers is None or i in steer_layers):
            d = dirs[i] / dirs[i].norm()
            coef = torch.einsum("tch,h->tc", s, d)
            s = s - alpha * coef.unsqueeze(-1) * d
    return s.mean(dim=1)


def manual_forward_non_ihc(embed, dirs, alpha):
    """The non-iHC lane: _forward_normal returns (mlp_out, residual) with
    the fold deferred, and the hotfix-parity apply steers the RETURNED
    hidden_states (mlp_out) directly, not the folded stream."""
    h = embed.clone()
    residual = None
    for i in range(NUM_LAYERS):
        if residual is not None:
            h = h + residual
        residual = h
        attn_out = h + torch.from_numpy(ATTN_WRITES[i])
        h = attn_out + residual
        residual = h
        mlp_out = h + torch.from_numpy(FFN_WRITES[i])
        if i in dirs:
            d = dirs[i] / dirs[i].norm()
            mlp_out = mlp_out - alpha * (mlp_out @ d).unsqueeze(-1) * d
        h, residual = mlp_out, residual
    return h + residual


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
        sys.modules.pop("weightless_steer.archs.hy4", None)
        for name in ("vllm.models.hy_v4.nvidia.model",
                     "vllm.distributed", "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="2.0", width=HIDDEN):
        write_gguf(self.path, good_meta(layers=layers, alpha=alpha),
                   good_tensors(layers=layers, width=width))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=HIDDEN)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, enable_ihc=True, **env):
        with mock.patch.dict(os.environ, env):
            return self.adapter.SteeredHy4ForCausalLM(
                vllm_config=_vllm_config(enable_ihc=enable_ihc))

    def test_forward_applies_projection_per_stream_per_layer(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model was swapped onto the steered class and the buffers
        # are sized to hidden_size — one direction per layer, applied per
        # iHC stream, never the flattened hc*hidden.
        self.assertIsInstance(model.model, self.adapter.SteeredHy4Model)
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

    def test_apply_steers_each_stream_independently(self):
        """A one-hot direction moves only that component, in every stream.

        Pins the hotfix's per-stream semantics: the projection contracts
        only the last axis of the [T, hc, hidden] stream, so streams never
        mix and the direction width is hidden_size — a glm5next-style
        flattened hc*hidden apply would look completely different here.
        """
        one_hot = np.zeros(HIDDEN, dtype=np.float32)
        one_hot[3] = 1.0
        write_gguf(self.path, good_meta(layers=(1,), alpha="1.0"),
                   {"direction.1": (one_hot, 0)})
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        s = torch.randn(4, HC, HIDDEN)
        with torch.no_grad():
            out = model.model._steer_core.apply(1, s)
        want = s.clone()
        want[..., 3] = 0.0          # alpha 1.0 removes the whole component
        self.assertTrue(torch.allclose(out, want, atol=1e-5),
                        f"max err {(out - want).abs().max()}")
        # An unsteered layer's zero stack row is a numeric no-op.
        with torch.no_grad():
            out0 = model.model._steer_core.apply(0, s)
        self.assertTrue(torch.allclose(out0, s, atol=1e-6))

    def test_last_layer_is_steered(self):
        """No deferred-contract trap (glm5next's L44 case): every iHC layer
        returns its materialized stream, the last one included."""
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

    def test_non_ihc_lane_steers_returned_hidden_states(self):
        """Hotfix parity on the non-iHC branch: the apply steers the
        RETURNED hidden_states (the mlp output, fold deferred), not the
        folded h = hidden_states + residual a fused-add+norm adapter would
        compute. No shipped hy_v4 checkpoint is non-iHC; the published
        GLP-77 vector is iHC-only."""
        layers = (1, 2)
        self.write_vector(layers, alpha="2.0")
        model = self.build(enable_ihc=False,
                           WEIGHTLESS_STEER_PATH=self.path)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward_non_ihc(embed, self.file_dirs(layers),
                                      alpha=2.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_per_request_serving_fails_before_model_allocation(self):
        """Refuse the unwired flag, and refuse it before loading weights."""
        with mock.patch.object(FakeHYV4ForCausalLM,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build(WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()

    def test_per_request_apply_fails_closed_on_3d_stream(self):
        """Per-request controls are structurally impossible on this arch:
        the iHC stream is [T, hc, hidden], and SteeringCore.apply refuses
        anything but the flattened [num_tokens, hidden] per-request layout
        (batch rows would index streams, not requests)."""
        self.write_vector((1,), alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        env = {"WEIGHTLESS_STEER_PATH": self.path,
               "WEIGHTLESS_ENABLE_MILESTONE_2": "1"}
        with mock.patch.dict(os.environ, env):
            model.model._wire_steering(dtype=torch.float32,
                                       max_num_tokens=MAX_NUM_TOKENS,
                                       max_num_reqs=MAX_NUM_REQS)
        self.assertTrue(model.model.weightless_per_request)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "flattened"):
                model(None, positions, inputs_embeds=embed)

    def test_compilation_rebinds_to_steered_forward(self):
        """The compile wrapper must capture the STEERED forward.

        hy_v4 carries no @support_torch_compile today, so the rebind is
        dormant; this pins its behaviour for the day upstream decorates the
        class — without it the compiled callable would run the stock
        forward and compiled serving would be silently unsteered.
        """
        original_init = FakeHYV4Model.__init__
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
            with mock.patch.object(FakeHYV4Model, "__init__", init):
                model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        finally:
            if previous is None:
                sys.modules.pop("vllm.compilation.wrapper", None)
            else:
                sys.modules["vllm.compilation.wrapper"] = previous
        self.assertEqual(captured, [FakeHYV4Model.forward,
                                    self.adapter.SteeredHy4Model.forward])
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
        # covering layers 1..3 must steer 2 and 3 by GLOBAL id (the apply
        # reads layer.layer_idx) — a local-index bug would apply
        # direction.1 to the first local layer.
        layers = (1, 2, 3)
        self.write_vector(layers, alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        # Move this rank's window onto global layers 2..3 (make_layers sets
        # these on the inner model at build; the dense stack stays global).
        model.model.start_layer, model.model.end_layer = 2, 4
        # Mid-stack state is the materialized iHC stream, [T, hc, hidden].
        embed = torch.randn(5, HC, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=1.0,
                              layers=range(2, 4))
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_gate_off_registers_no_per_request_buffers(self):
        """The default deployment is untouched by any of this."""
        self.write_vector((1,), alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertFalse(model.model.weightless_per_request)
        for name in ("_steer_alpha_rows", "_steer_slot_rows",
                     "_steer_layer_bank"):
            self.assertNotIn(name, dict(model.model.named_buffers()))

    def test_flattened_width_vector_fails_closed(self):
        """A glm5next-style flattened hc*hidden vector is not this arch's
        stream: one 6144-wide direction per layer, per iHC stream."""
        self.write_vector((1, 2), alpha="2.0", width=HIDDEN * HC)
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

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

    # The model layer-loop anchor, identical to the hotfix's ANCHOR_FORWARD.
    LOOP_ANCHOR = (
        "        for layer in islice(self.layers, self.start_layer, self.end_layer):\n"
        "            hidden_states, residual = layer(positions, hidden_states, residual)\n"
    )

    def test_copied_loop_matches_vendored_reference(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "hy4.py").read_text()
        reference_src = REFERENCE.read_text()
        # The anchor exists verbatim in BOTH the adapter's forward copy and
        # the upstream reference it was copied from.
        self.assertIn(self.LOOP_ANCHOR, reference_src)
        self.assertIn(self.LOOP_ANCHOR, adapter_src)
        # The steering block sits immediately after the layer call.
        self.assertIn(
            self.LOOP_ANCHOR
            + "            # [weightless-steer] the one added block",
            adapter_src,
        )


if __name__ == "__main__":
    unittest.main()
