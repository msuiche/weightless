"""Offline tests for the inkling steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce inkling's DEFERRED MLP-add convention (each decoder layer called
  with defer_mlp_add=True returns (hidden_states, pending) where pending
  carries the pre-reduce, pre-sconv MLP delta that the NEXT layer's fused
  sconv+add+rmsnorm consumes). The adapter is imported against the stub, the
  model is built through BOTH entry classes, and its forward output is
  compared against two hand-computed recurrences: the stock deferred
  pipeline (flush fused with the next layer's norm) and the steered
  flushed-per-layer pipeline (h <- h - alpha*(h.d)d on the materialized
  [T, hidden] stream). This exercises the registry-shadowed classes end to
  end: the __class__ swap, buffer wiring at plain hidden_size width,
  global-layer indexing, and the flush/steer ordering — the norms are real
  (non-identity) RMS norms so a flush that landed on the wrong side of a
  norm would not cancel out of the comparison.
- Structural: the adapter's copied forward is pinned against the vendored
  upstream reference (patches/reference/inkling_v0280.py — byte-identical
  to vLLM v0.28.0's vllm/models/inkling/nvidia/model.py), so upstream drift
  in our copy is caught here rather than at serve time.
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
             / "inkling_v0280.py")

HIDDEN = 8
NUM_LAYERS = 4
# Deterministic per-layer sublayer "writes": what each fake decoder layer's
# attention and MLP deposit into the residual stream.
ATTN_WRITES = [np.random.default_rng(500 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
MLP_WRITES = [np.random.default_rng(600 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
# Non-identity norm scales: which norm fired, and on what input, is part of
# what the recurrence comparisons pin.
ATTN_NORM_W = [np.random.default_rng(700 + i).uniform(0.7, 1.4, HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
MLP_NORM_W = [np.random.default_rng(800 + i).uniform(0.7, 1.4, HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
FINAL_NORM_W = np.random.default_rng(900).uniform(0.7, 1.4, HIDDEN).astype(
    np.float32)
EPS = 1e-5


def _rms(x, weight):
    return x * weight / (x.float().pow(2).mean(-1, keepdim=True) + EPS).sqrt()


class FakeRMSNorm(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(torch.from_numpy(weight))
        self.variance_epsilon = EPS

    def forward(self, x):
        return _rms(x, self.weight)


class FakeAttn(nn.Module):
    def __init__(self, write):
        super().__init__()
        self.register_buffer("write", torch.from_numpy(write))

    def forward(self, positions, hidden_states, log_scaling=None):
        return hidden_states + self.write


class FakeMLP(nn.Module):
    def __init__(self, write):
        super().__init__()
        self.register_buffer("write", torch.from_numpy(write))

    def forward(self, x):
        return x + self.write


def _fake_sconv_add_norm(delta, hidden, sconv, norm, positions):
    """The upstream flush, reduced to its math: h = hidden + TP-sum(delta),
    optionally normed. The RS/sconv/AG machinery and the MoE (routed, shared)
    tuple partial collapse to plain adds — the adapter only depends on the
    (normed, hidden) return convention and the flush being THE
    materialization of the post-layer stream.
    """
    if isinstance(delta, tuple):
        delta = delta[0] + delta[1]
    h = hidden + delta
    return (norm(h) if norm is not None else None), h


class FakeInklingShortConv:
    """Marker for the pending tuple's second slot; the fake flush ignores it."""


class FakeInklingDecoderLayer(nn.Module):
    """The upstream deferred-MLP-add convention with trivial sublayers.

    Faithful copy of InklingDecoderLayer.forward's control flow: an incoming
    pending is flushed fused with this layer's attn_norm; the attention delta
    is flushed fused with mlp_norm; with defer_mlp_add the MLP delta is
    RETURNED (hidden_states, (mlp_output, mlp_sconv)) for the caller to fold.
    """

    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = FakeRMSNorm(ATTN_NORM_W[layer_idx])
        self.mlp_norm = FakeRMSNorm(MLP_NORM_W[layer_idx])
        self.attn = FakeAttn(ATTN_WRITES[layer_idx])
        self.mlp = FakeMLP(MLP_WRITES[layer_idx])
        self.attn_sconv = FakeInklingShortConv()
        self.mlp_sconv = FakeInklingShortConv()

    def forward(self, positions, hidden_states, pending=None,
                defer_mlp_add=False, attn_in=None, log_scaling=None):
        if pending is None:
            if attn_in is None:
                attn_in = self.attn_norm(hidden_states)
        else:
            attn_in, hidden_states = _fake_sconv_add_norm(
                pending[0], hidden_states, pending[1], self.attn_norm,
                positions)
        attn_output = self.attn(positions, attn_in, log_scaling)
        mlp_in, hidden_states = _fake_sconv_add_norm(
            attn_output, hidden_states, self.attn_sconv, self.mlp_norm,
            positions)
        mlp_output = self.mlp(mlp_in)
        if defer_mlp_add:
            return hidden_states, (mlp_output, self.mlp_sconv)
        return _fake_sconv_add_norm(
            mlp_output, hidden_states, self.mlp_sconv, None, positions)[1]


def _fake_config():
    return types.SimpleNamespace(
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        rms_norm_eps=EPS,
        log_scaling_n_floor=None,   # keeps log_scaling=None in the loop
        log_scaling_alpha=None,
    )


class FakeInklingModel(nn.Module):
    def __init__(self, *, config, start_layer=0, end_layer=NUM_LAYERS):
        super().__init__()
        self.config = config
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(
            FakeInklingDecoderLayer(i) for i in range(NUM_LAYERS))
        self.norm = FakeRMSNorm(FINAL_NORM_W)
        self.embed_tokens = None      # tests pass inputs_embeds
        self.embed_norm = None

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeInklingForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeInklingModel(
            config=vllm_config.model_config.hf_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)


class FakeInklingForConditionalGeneration(FakeInklingForCausalLM):
    """The multimodal entry: towers are side branches the adapter never
    touches, so the fake is the same backbone build (what _build does)."""


class _PPGroup:
    is_first_rank = True
    is_last_rank = True


MAX_NUM_TOKENS = 16
MAX_NUM_REQS = 4


def _vllm_config():
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=_fake_config(),
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
    ink = types.ModuleType("vllm.models.inkling.nvidia.model")
    ink.InklingDelta = torch.Tensor  # annotation alias; never evaluated
    ink.InklingShortConv = FakeInklingShortConv
    ink.InklingModel = FakeInklingModel
    ink.InklingForCausalLM = FakeInklingForCausalLM
    ink.InklingForConditionalGeneration = FakeInklingForConditionalGeneration
    ink._sconv_add_norm = _fake_sconv_add_norm
    ink.compute_log_scaling_tau = lambda *a: None
    ink.embed_rmsnorm = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("tests pass inputs_embeds"))
    stubs["vllm.models.inkling.nvidia.model"] = ink
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: _PPGroup
    stubs["vllm.distributed"] = distributed
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.inkling", None)
    return importlib.import_module("weightless_steer.archs.inkling")


def manual_forward_stock(embed, layers=range(NUM_LAYERS)):
    """Hand-computed STOCK forward: the deferred-MLP pipeline, where each
    layer's flush is fused with the next layer's attn_norm and the last
    layer's with the final norm. Ground truth for the adapter's
    decomposition: steering aside, flush-per-layer must reproduce this
    exactly (same math, one kernel less fused).
    """
    h = embed.clone()
    pending = None
    for i in layers:
        if pending is not None:
            h = h + pending
        attn_in = _rms(h, torch.from_numpy(ATTN_NORM_W[i]))
        attn_out = attn_in + torch.from_numpy(ATTN_WRITES[i])
        h = h + attn_out
        mlp_in = _rms(h, torch.from_numpy(MLP_NORM_W[i]))
        pending = mlp_in + torch.from_numpy(MLP_WRITES[i])
    h = h + pending
    return _rms(h, torch.from_numpy(FINAL_NORM_W))


def manual_forward(embed, dirs, alpha, layers=range(NUM_LAYERS),
                   steer_layers=None):
    """Hand-computed ADAPTER forward: flush-per-layer (pending=None into the
    next layer, so attn_norm runs on the materialized stream), projection on
    the materialized post-layer stream per steered layer, final norm.
    """
    h = embed.clone()
    for i in layers:
        attn_in = _rms(h, torch.from_numpy(ATTN_NORM_W[i]))
        attn_out = attn_in + torch.from_numpy(ATTN_WRITES[i])
        h = h + attn_out
        mlp_in = _rms(h, torch.from_numpy(MLP_NORM_W[i]))
        h = h + mlp_in + torch.from_numpy(MLP_WRITES[i])
        if i in dirs and (steer_layers is None or i in steer_layers):
            d = dirs[i] / dirs[i].norm()
            h = h - alpha * (h @ d).unsqueeze(-1) * d
    return _rms(h, torch.from_numpy(FINAL_NORM_W))


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
        sys.modules.pop("weightless_steer.archs.inkling", None)
        for name in ("vllm.models.inkling.nvidia.model",
                     "vllm.distributed", "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="0.25", width=HIDDEN):
        write_gguf(self.path, good_meta(layers=layers, alpha=alpha),
                   good_tensors(layers=layers, width=width))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=HIDDEN)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, cls_name="SteeredInklingForCausalLM", **env):
        with mock.patch.dict(os.environ, env):
            return getattr(self.adapter, cls_name)(vllm_config=_vllm_config())

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
        self.write_vector(layers, alpha="0.25")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model was swapped onto the steered class and wired at
        # plain hidden_size width (no hyper-connection widening).
        self.assertIsInstance(model.model, self.adapter.SteeredInklingModel)
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (NUM_LAYERS, 1, HIDDEN))
        self.assertAlmostEqual(float(model.model._steer_alpha), 0.25)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, None, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=0.25)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_unsteered_path_reproduces_stock_deferred_pipeline(self):
        """THE inkling invariant: flush-per-layer == fused deferred add.

        With the disabled core (no vector), the adapter's forward must be
        numerically identical to the stock deferred pipeline — the flush
        moves the MLP add out of the next layer's fused kernel without
        changing the math. This is what makes the alpha=0.0 control arm a
        valid stock control.
        """
        model = self.build()  # no WEIGHTLESS_STEER_PATH: disabled core
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, None, inputs_embeds=embed)
        want = manual_forward_stock(embed)
        self.assertTrue(torch.allclose(out, want, atol=1e-5),
                        f"max err {(out - want).abs().max()}")
        # And the two hand-computed recurrences agree at alpha 0 — the test
        # machinery itself is not smuggling a difference.
        self.assertTrue(torch.allclose(
            manual_forward(embed, {}, alpha=0.0), want, atol=1e-6))

    def test_last_layer_is_steered(self):
        """GLP-41 covers layer 41, the last of 42: the final layer's MLP add
        is fused with the FINAL norm in stock, and must still be steered.
        """
        layers = (NUM_LAYERS - 1,)
        self.write_vector(layers, alpha="0.25")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, None, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=0.25)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")
        self.assertFalse(torch.allclose(out, manual_forward_stock(embed),
                                        atol=1e-4))

    def test_conditional_generation_entry_is_steered(self):
        """Inkling-Small-NVFP4 resolves to InklingForConditionalGeneration —
        the multimodal shadow must wire the same steering."""
        layers = (1, 2)
        self.write_vector(layers, alpha="0.25")
        model = self.build("SteeredInklingForConditionalGeneration",
                           WEIGHTLESS_STEER_PATH=self.path)
        self.assertIsInstance(model.model, self.adapter.SteeredInklingModel)
        embed = torch.randn(4, HIDDEN)
        positions = torch.arange(4)
        with torch.no_grad():
            out = model(None, positions, None, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=0.25)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_per_request_serving_fails_before_model_allocation(self):
        """Refuse the unwired flag, and refuse it before loading weights."""
        for cls_name, fake in (
                ("SteeredInklingForCausalLM", FakeInklingForCausalLM),
                ("SteeredInklingForConditionalGeneration",
                 FakeInklingForConditionalGeneration)):
            with mock.patch.object(fake, "__init__") as allocate:
                with self.assertRaisesRegex(RuntimeError,
                                            "runner integration"):
                    self.build(cls_name, WEIGHTLESS_ENABLE_MILESTONE_2="1")
            allocate.assert_not_called()

    def test_compilation_rebinds_to_steered_forward(self):
        """The compile wrapper must capture the STEERED forward.

        inkling carries no @support_torch_compile today, so the rebind is
        dormant; this pins its behaviour for the day upstream decorates the
        class — without it the compiled callable would run the stock
        forward and compiled serving would be silently unsteered.
        """
        original_init = FakeInklingModel.__init__
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
        self.write_vector((1, 3), alpha="0.25")
        previous = sys.modules.get("vllm.compilation.wrapper")
        sys.modules["vllm.compilation.wrapper"] = wrapper_module
        try:
            with mock.patch.object(FakeInklingModel, "__init__", init):
                model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        finally:
            if previous is None:
                sys.modules.pop("vllm.compilation.wrapper", None)
            else:
                sys.modules["vllm.compilation.wrapper"] = previous
        self.assertEqual(captured, [FakeInklingModel.forward,
                                    self.adapter.SteeredInklingModel.forward])
        # The stock init's hook was dropped, not left registered
        # alongside the new one for the life of the process.
        self.assertEqual(list(live_hooks), [2])
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        out = model.model._compiled_callable(None, positions, None,
                                             inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((1, 3)), alpha=0.25)
        torch.testing.assert_close(out, want, atol=1e-4, rtol=1e-4)

    def test_alpha_env_overrides_file_default(self):
        layers = (2,)
        self.write_vector(layers, alpha="0.25")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_STEER_ALPHA="0.125")
        embed = torch.randn(4, HIDDEN)
        positions = torch.arange(4)
        with torch.no_grad():
            out = model(None, positions, None, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=0.125)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_per_request_forward_end_to_end(self):
        """Two requests, two alphas, through the real adapter forward."""
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )

        layers = (1, 3)
        self.write_vector(layers, alpha="0.25")
        model = self.build_control_harness()
        inner = model.model
        self.assertTrue(inner.weightless_per_request)
        self.assertEqual(inner.weightless_steer_layer_ids, layers)

        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=NUM_LAYERS, default_alpha=0.25, loaded_layers=layers,
        )
        plane.build([
            ScheduledRequest("a", WeightlessResolvedXArgs(alpha_override=0.0),
                             token_count=2, start_ordinal=0, prompt_length=2),
            ScheduledRequest("b", WeightlessResolvedXArgs(alpha_override=0.5),
                             token_count=3, start_ordinal=0, prompt_length=3),
        ])
        plane.install(inner)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, None, inputs_embeds=embed)
        dirs = self.file_dirs(layers)
        # Request "a" is unsteered, request "b" runs at alpha 0.5.
        want_a = manual_forward(embed[:2], dirs, alpha=0.0)
        want_b = manual_forward(embed[2:], dirs, alpha=0.5)
        self.assertTrue(torch.allclose(out[:2], want_a, atol=1e-4),
                        f"req a max err {(out[:2] - want_a).abs().max()}")
        self.assertTrue(torch.allclose(out[2:], want_b, atol=1e-4),
                        f"req b max err {(out[2:] - want_b).abs().max()}")

    def test_per_request_layer_mask_through_the_forward(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="0.25")
        model = self.build_control_harness()
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )
        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=NUM_LAYERS, default_alpha=0.25, loaded_layers=layers,
        )
        plane.build([ScheduledRequest(
            "a", WeightlessResolvedXArgs(layers_override=(3,)),
            token_count=3, start_ordinal=0, prompt_length=3)])
        plane.install(model.model)
        embed = torch.randn(3, HIDDEN)
        positions = torch.arange(3)
        with torch.no_grad():
            out = model(None, positions, None, inputs_embeds=embed)
        # Only layer 3 steers, even though layer 1 carries a direction.
        want = manual_forward(embed, self.file_dirs(layers), alpha=0.25,
                              steer_layers=(3,))
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_gate_off_registers_no_per_request_buffers(self):
        """The default deployment is untouched by any of this."""
        self.write_vector((1,), alpha="0.25")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertFalse(model.model.weightless_per_request)
        for name in ("_steer_alpha_rows", "_steer_slot_rows",
                     "_steer_layer_bank"):
            self.assertNotIn(name, dict(model.model.named_buffers()))

    def test_global_layer_indexing_under_pipeline_split(self):
        # This rank runs global layers 2..3 (start_layer=2). A vector
        # covering layers 1..3 must steer 2 and 3 by GLOBAL id — a
        # local-index bug would apply direction.1 to the first local layer.
        layers = (1, 2, 3)
        self.write_vector(layers, alpha="0.25")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        # Move this rank's window onto global layers 2..3 (make_layers sets
        # these on the inner model at build; the dense stack stays global).
        model.model.start_layer, model.model.end_layer = 2, 4
        # Mid-stack state is the materialized stream, [T, hidden] — the
        # previous rank flushed its last pending before the boundary.
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, None, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=0.25,
                              layers=range(2, 4))
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_widened_width_vector_fails_closed(self):
        """A 2*hidden-wide vector is not this arch's stream (that is the
        glm5next derivation width, not inkling's)."""
        self.write_vector((1, 2), alpha="0.25", width=2 * HIDDEN)
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_bad_vector_fails_closed_at_model_build(self):
        self.write_vector((1, 2), alpha="0.25")
        # Corrupt: declare a layer set that disagrees with the tensors.
        write_gguf(self.path,
                   {**good_meta(layers=(1, 2)),
                    "glp.layer_ids_zero_based": "5,6"},
                   good_tensors(layers=(1, 2), width=HIDDEN))
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)


class StructureTests(unittest.TestCase):
    """Pin the adapter's copied forward against the vendored reference."""

    # The layer call + loop tail, identical to the hotfix's ANCHOR_FORWARD
    # body (the hotfix wraps the same call in its enumerate rewrite).
    LOOP_CALL_ANCHOR = (
        "            hidden_states, pending = layer(\n"
        "                positions,\n"
        "                hidden_states,\n"
        "                pending=pending,\n"
        "                defer_mlp_add=True,\n"
        "                attn_in=attn_in0,\n"
        "                log_scaling=log_scaling,\n"
        "            )\n"
        "            attn_in0 = None\n"
    )
    # The stock loop header (the adapter enumerates the same slice).
    STOCK_LOOP_ANCHOR = (
        "        pending: tuple[InklingDelta, InklingShortConv] | None = None\n"
        "        for layer in self.layers[self.start_layer : self.end_layer]:\n"
    )
    # The flush idiom — the PP-boundary branch the steering flush mirrors.
    PP_FLUSH_ANCHOR = (
        "            if pending is not None:\n"
        "                hidden_states = _sconv_add_norm(\n"
        "                    pending[0], hidden_states, pending[1], None, positions\n"
        "                )[1]\n"
    )
    # The fused final-norm tail (kept verbatim; never fires once the loop
    # flushes unconditionally, and must still match upstream byte-for-byte).
    FINAL_NORM_ANCHOR = (
        "        if pending is not None:\n"
        "            # Final RS/sconv/AG + residual add fused with the final rmsnorm.\n"
        "            norm_out = _sconv_add_norm(\n"
        "                pending[0], hidden_states, pending[1], self.norm, positions\n"
        "            )[0]\n"
        "            assert norm_out is not None\n"
        "            return norm_out\n"
        "        return self.norm(hidden_states)\n"
    )

    def test_copied_loops_match_vendored_reference(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "inkling.py").read_text()
        reference_src = REFERENCE.read_text()
        # The anchors exist verbatim in BOTH the adapter's forward copy and
        # the upstream reference it was copied from.
        self.assertIn(self.LOOP_CALL_ANCHOR, reference_src)
        self.assertIn(self.LOOP_CALL_ANCHOR, adapter_src)
        self.assertIn(self.PP_FLUSH_ANCHOR, reference_src)
        self.assertIn(self.PP_FLUSH_ANCHOR, adapter_src)
        self.assertIn(self.FINAL_NORM_ANCHOR, reference_src)
        self.assertIn(self.FINAL_NORM_ANCHOR, adapter_src)
        # The stock loop header pins the reference's loop shape; the adapter
        # iterates the same slice under enumerate.
        self.assertIn(self.STOCK_LOOP_ANCHOR, reference_src)
        self.assertIn(
            "        for _wl_off, layer in enumerate(\n"
            "            self.layers[self.start_layer : self.end_layer]\n"
            "        ):\n",
            adapter_src,
        )
        # The steering block sits immediately after the layer call.
        self.assertIn(
            self.LOOP_CALL_ANCHOR
            + "            # [weightless-steer] the one added block",
            adapter_src,
        )


if __name__ == "__main__":
    unittest.main()
