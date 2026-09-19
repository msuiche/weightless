"""Offline tests for the qwen38fn (Qwen3.8-Flash-Next) steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce qwen3_8_flash_next's delayed-combine hyper-connection
  convention (each decoder layer returns (hidden_states, block_output,
  injection) with the MLP output still pending; the post-layer stream is
  the parameter-free mlp_hyper_connection.combine() materialization
  [T, hc_count*hidden], already flat HC-outer). The adapter is imported
  against the stub, the model is built, and its forward output is compared
  against a hand-computed  h <- h - alpha*(h.d)d  per steered layer on the
  flat widened stream. This exercises the registry-shadowed classes end to
  end: the __class__ swap of the inner model (via BOTH serving classes —
  the CausalLM and the multimodal ConditionalGeneration that the RadixArk
  checkpoint actually resolves to), buffer wiring at the widened stream
  width, global-layer indexing, and the final-mixer guard for the consumed
  pending combine.
- Structural: the adapter's copied forward is pinned against the vendored
  upstream reference (patches/reference/qwen3_8_flash_next.py), so upstream
  drift in our copy is caught here rather than at serve time.
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
             / "qwen3_8_flash_next.py")

HIDDEN = 8
HC = 3
STREAM_WIDTH = HIDDEN * HC
NUM_LAYERS = 4
# Deterministic per-layer sublayer "writes": what each fake decoder layer's
# attention and MLP deposit into the stream.
ATTN_WRITES = [np.random.default_rng(200 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
MLP_WRITES = [np.random.default_rng(300 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
# Per-layer, per-HC-module stream scales: the fold of a pending block output
# lands on each stream with a distinct gain, so the streams diverge even
# unsteered and the HC-outer flatten order is load-bearing in every
# comparison.
ATTN_SCALES = [np.random.default_rng(400 + i).standard_normal(HC).astype(
    np.float32) for i in range(NUM_LAYERS)]
MLP_SCALES = [np.random.default_rng(500 + i).standard_normal(HC).astype(
    np.float32) for i in range(NUM_LAYERS)]
# Per-HC-module injection markers ([HC*H] vectors), so the delayed-combine
# state is visible in the stream values and the mixer path is pinned.
ATTN_BIAS = [np.random.default_rng(600 + i).standard_normal(
    STREAM_WIDTH).astype(np.float32) * 0.1 for i in range(NUM_LAYERS)]
MLP_BIAS = [np.random.default_rng(700 + i).standard_normal(
    STREAM_WIDTH).astype(np.float32) * 0.1 for i in range(NUM_LAYERS)]
MIXER_SCALE = np.random.default_rng(800).standard_normal(HC).astype(np.float32)
MIXER_BIAS = (np.random.default_rng(801).standard_normal(STREAM_WIDTH)
              .astype(np.float32) * 0.1)


class FakeHC(nn.Module):
    """Delayed-combine hyper-connection, minimal deterministic semantics.

    mix:        h [T, C*H] -> (h, per-stream mean [T, H], injection marker)
    combine:    fold a pending [T, H] block output into the stream with a
                per-stream gain, plus the carried injection marker
    combine_and_mix: combine, then mix (the fused inter-layer path)

    The same class serves the final mixer, whose mix()/combine_and_mix()
    second return value is the sampled single stream.
    """

    def __init__(self, scale, bias):
        super().__init__()
        self.register_buffer("scale", torch.from_numpy(scale))    # [C]
        self.register_buffer("bias", torch.from_numpy(bias))      # [C*H]

    def mix(self, h):
        block_input = h.view(h.shape[0], HC, HIDDEN).mean(dim=1)
        return h, block_input, self.bias

    def combine(self, h, block_output, injection):
        spread = (block_output.unsqueeze(1)
                  * self.scale.view(1, -1, 1)).flatten(-2)
        return h + spread + injection

    def combine_and_mix(self, h, block_output, injection):
        h2 = self.combine(h, block_output, injection)
        return h2, h2.view(h2.shape[0], HC, HIDDEN).mean(dim=1), self.bias


class FakeQwen38DecoderLayer(nn.Module):
    """The upstream 3-tuple delayed-combine convention with trivial sublayers.

    Mirrors the reference layer's structure (sans PLE): a pending combine
    from the previous layer is fused into this layer's attention mix
    (combine_and_mix) when present, else the standalone mix() path is taken;
    attention and MLP read the per-stream mean and their writes land back on
    the stream via the mlp HC's combine; the layer returns
    (hidden_states, mlp_out, injection) with the MLP write still pending.
    """

    def __init__(self, layer_idx, attn_write, mlp_write,
                 attn_scale, mlp_scale, attn_bias, mlp_bias):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_hyper_connection = FakeHC(attn_scale, attn_bias)
        self.mlp_hyper_connection = FakeHC(mlp_scale, mlp_bias)
        self.register_buffer("attn_write", torch.from_numpy(attn_write))
        self.register_buffer("mlp_write", torch.from_numpy(mlp_write))

    def forward(self, hidden_states, prev_block_output, prev_injection,
                positions, *, input_ids, query_start_loc, ngram_context):
        attn_hc = self.attn_hyper_connection
        if prev_block_output is not None and prev_injection is not None:
            hidden_states, block_input, injection = attn_hc.combine_and_mix(
                hidden_states, prev_block_output, prev_injection)
        else:
            hidden_states, block_input, injection = attn_hc.mix(hidden_states)
        attn_out = block_input + self.attn_write
        hidden_states, block_input, injection = (
            self.mlp_hyper_connection.combine_and_mix(
                hidden_states, attn_out, injection))
        mlp_out = block_input + self.mlp_write
        return hidden_states, mlp_out, injection


class FakeQwen38Model(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.start_layer, self.end_layer = 0, NUM_LAYERS
        self.layers = nn.ModuleList(
            FakeQwen38DecoderLayer(i, ATTN_WRITES[i], MLP_WRITES[i],
                                   ATTN_SCALES[i], MLP_SCALES[i],
                                   ATTN_BIAS[i], MLP_BIAS[i])
            for i in range(NUM_LAYERS)
        )
        self.hyper_connection_mixer = FakeHC(MIXER_SCALE, MIXER_BIAS)
        self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeQwen38ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeQwen38Model(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds, **kw)


class FakeQwen38ForConditionalGeneration(nn.Module):
    """The multimodal wrapper: builds its language model DIRECTLY (the reason
    the CausalLM shadow alone cannot cover this arch)."""

    def __init__(self, *, vllm_config, prefix="model"):
        super().__init__()
        self.language_model = FakeQwen38ForCausalLM(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kw)


class _PPGroup:
    """Mutable PP state: tests flip the ranks to exercise the mid-stack and
    transport branches."""
    is_first_rank = True
    is_last_rank = True


MAX_NUM_TOKENS = 16
MAX_NUM_REQS = 4


def _vllm_config():
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                hidden_size=HIDDEN,
                num_hidden_layers=NUM_LAYERS,
                hc_count=HC,
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
    mod = types.ModuleType("vllm.models.qwen3_8_flash_next.nvidia.model")
    mod.Qwen3_8FlashNextModel = FakeQwen38Model
    mod.Qwen3_8FlashNextForCausalLM = FakeQwen38ForCausalLM
    mod.Qwen3_8FlashNextForConditionalGeneration = (
        FakeQwen38ForConditionalGeneration)
    stubs["vllm.models.qwen3_8_flash_next.nvidia.model"] = mod
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: _PPGroup
    stubs["vllm.distributed"] = distributed
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.qwen38fn", None)
    return importlib.import_module("weightless_steer.archs.qwen38fn")


def _norm_dir(d):
    """SteeringCore.from_env normalises in float64 then casts to f32."""
    v = d.double()
    return (v / v.norm()).float()


def manual_forward(embed, dirs, alpha, layers=range(NUM_LAYERS),
                   steer_layers=None):
    """Hand-computed steered forward over the flat widened stream.

    embed is the already-widened [T, C*H] entry state (the model's first
    rank repeat(1, hc_count) of inputs_embeds). Per running layer: the
    fake's delayed-combine semantics (pending fold at entry, attn write,
    mlp write kept pending), then the apply materializes the post-layer
    stream via the layer's mlp HC combine, projects on it, and continues
    materialized. The tail takes the guarded final-mixer mix() path:
    sample = per-stream mean of the materialized stream.
    """
    s = embed.clone()
    prev = None  # (block_output, injection)
    for i in layers:
        attn_hc_scale = torch.from_numpy(ATTN_SCALES[i]).view(1, -1, 1)
        mlp_hc_scale = torch.from_numpy(MLP_SCALES[i]).view(1, -1, 1)
        attn_bias = torch.from_numpy(ATTN_BIAS[i])
        mlp_bias = torch.from_numpy(MLP_BIAS[i])
        if prev is not None:
            bo, inj = prev
            s = s + (bo.unsqueeze(1) * attn_hc_scale).flatten(-2) + inj
        block_input = s.view(s.shape[0], HC, HIDDEN).mean(dim=1)
        attn_out = block_input + torch.from_numpy(ATTN_WRITES[i])
        s = s + (attn_out.unsqueeze(1) * mlp_hc_scale).flatten(-2) + attn_bias
        block_input = s.view(s.shape[0], HC, HIDDEN).mean(dim=1)
        mlp_out = block_input + torch.from_numpy(MLP_WRITES[i])
        # the adapter's per-layer apply: materialize, project, continue
        stream = s + (mlp_out.unsqueeze(1) * mlp_hc_scale).flatten(-2) + mlp_bias
        if i in dirs and (steer_layers is None or i in steer_layers):
            d = _norm_dir(dirs[i])
            stream = stream - alpha * (stream @ d).unsqueeze(-1) * d
        s = stream
        prev = None
    return s.view(s.shape[0], HC, HIDDEN).mean(dim=1)


class SteeredForwardTests(unittest.TestCase):
    def setUp(self):
        self.adapter = _import_adapter()
        self.addCleanup(self._unimport)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "vec.gguf")
        _PPGroup.is_first_rank = True
        _PPGroup.is_last_rank = True
        for k in ("WEIGHTLESS_STEER_PATH", "WEIGHTLESS_STEER_ALPHA",
                  "WEIGHTLESS_STEER_LAYERS", "WEIGHTLESS_STEER_HOOK",
                  "WEIGHTLESS_ENABLE_MILESTONE_2"):
            os.environ.pop(k, None)

    def _unimport(self):
        sys.modules.pop("weightless_steer.archs.qwen38fn", None)
        for name in ("vllm.models.qwen3_8_flash_next.nvidia.model",
                     "vllm.distributed", "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="1.0", width=STREAM_WIDTH):
        write_gguf(self.path, good_meta(layers=layers, alpha=alpha),
                   good_tensors(layers=layers, width=width))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=STREAM_WIDTH)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, cls_name="SteeredQwen3_8FlashNextForCausalLM", **env):
        with mock.patch.dict(os.environ, env):
            return getattr(self.adapter, cls_name)(
                vllm_config=_vllm_config())

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
                                       max_num_reqs=MAX_NUM_REQS,
                                       stream_width=STREAM_WIDTH)
        return model

    def widen(self, embed):
        """The first-rank entry transform: repeat(1, hc_count)."""
        return embed.repeat(1, HC)

    def test_forward_applies_projection_per_layer(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model was swapped onto the steered class and the buffers
        # are sized to the widened stream.
        self.assertIsInstance(model.model,
                              self.adapter.SteeredQwen3_8FlashNextModel)
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (NUM_LAYERS, 1, STREAM_WIDTH))
        self.assertAlmostEqual(float(model.model._steer_alpha), 1.0)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(self.widen(embed), self.file_dirs(layers),
                              alpha=1.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_last_layer_is_steered_and_mixer_guarded(self):
        """The L47 case: the last layer's pending combine must not escape
        steering, and the final mixer must take mix() for the
        already-materialized state.

        Without the guard, combine_and_mix(h, None, None) would crash or
        fold garbage on exactly the layer GLP-47 needs (47 = last of 48).
        """
        layers = (NUM_LAYERS - 1,)
        self.write_vector(layers, alpha="1.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(self.widen(embed), self.file_dirs(layers),
                              alpha=1.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")
        unsteered = manual_forward(self.widen(embed), {}, alpha=0.0)
        self.assertFalse(torch.allclose(out, unsteered, atol=1e-4))

    def test_steer_delayed_combine_site(self):
        """The site helper on a real (fake) layer: materialize via the
        layer's mlp HC combine, project the flat stream, clear the pending
        state."""
        layers = (1,)
        self.write_vector(layers, alpha="1.5")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        inner = model.model
        d = _norm_dir(self.file_dirs(layers)[1])

        x = torch.randn(4, STREAM_WIDTH)
        bo = torch.randn(4, HIDDEN)
        layer = inner.layers[1]
        out, nbo, ninj = inner._steer_delayed_combine(1, layer, x, bo,
                                                      layer.mlp_hyper_connection.bias)
        stream = (x + (bo.unsqueeze(1)
                       * layer.mlp_hyper_connection.scale.view(1, -1, 1)
                       ).flatten(-2) + layer.mlp_hyper_connection.bias)
        want = stream - 1.5 * (stream @ d).unsqueeze(-1) * d
        self.assertEqual(out.shape, (4, STREAM_WIDTH))
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")
        self.assertIsNone(nbo)
        self.assertIsNone(ninj)

        # An unsteered layer's zero stack row is a numeric no-op.
        out0, _, _ = inner._steer_delayed_combine(0, layer, x, bo,
                                                  layer.mlp_hyper_connection.bias)
        self.assertTrue(torch.allclose(out0, stream, atol=1e-6))

        # block_output=None (deepstack already materialized): the stream is
        # read directly.
        out1, _, _ = inner._steer_delayed_combine(1, layer, x, None, None)
        want1 = x - 1.5 * (x @ d).unsqueeze(-1) * d
        self.assertTrue(torch.allclose(out1, want1, atol=1e-4))

    def test_conditional_generation_shadow_steers_language_model(self):
        """The arch the RadixArk checkpoint actually resolves to: the
        wrapper builds its language model directly, so the shadowed wrapper
        must swap language_model.model itself."""
        layers = (2,)
        self.write_vector(layers, alpha="1.0")
        model = self.build("SteeredQwen3_8FlashNextForConditionalGeneration",
                           WEIGHTLESS_STEER_PATH=self.path)
        inner = model.language_model.model
        self.assertIsInstance(inner, self.adapter.SteeredQwen3_8FlashNextModel)
        self.assertEqual(tuple(inner._steer_stack.shape),
                         (NUM_LAYERS, 1, STREAM_WIDTH))
        embed = torch.randn(4, HIDDEN)
        positions = torch.arange(4)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(self.widen(embed), self.file_dirs(layers),
                              alpha=1.0, layers=range(NUM_LAYERS))
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
        want_a = manual_forward(self.widen(embed[:2]), dirs, alpha=0.0)
        want_b = manual_forward(self.widen(embed[2:]), dirs, alpha=1.0)
        self.assertTrue(torch.allclose(out[:2], want_a, atol=1e-4),
                        f"req a max err {(out[:2] - want_a).abs().max()}")
        self.assertTrue(torch.allclose(out[2:], want_b, atol=1e-4),
                        f"req b max err {(out[2:] - want_b).abs().max()}")

    def test_per_request_serving_fails_before_model_allocation(self):
        """Refuse the unwired flag, and refuse it before loading weights."""
        with mock.patch.object(FakeQwen38ForCausalLM,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build(WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()
        with mock.patch.object(FakeQwen38ForConditionalGeneration,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build("SteeredQwen3_8FlashNextForConditionalGeneration",
                           WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()

    def test_compilation_rebinds_to_steered_forward(self):
        """The compile wrapper must capture the STEERED forward.

        qwen38fn IS @support_torch_compile-decorated (unlike glm5next
        today), so the rebind runs on every compiled boot — without it the
        compiled callable would run the stock forward and compiled serving
        would be silently unsteered.
        """
        original_init = FakeQwen38Model.__init__
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
            with mock.patch.object(FakeQwen38Model, "__init__", init):
                model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        finally:
            if previous is None:
                sys.modules.pop("vllm.compilation.wrapper", None)
            else:
                sys.modules["vllm.compilation.wrapper"] = previous
        self.assertEqual(captured, [FakeQwen38Model.forward,
                                    self.adapter.SteeredQwen3_8FlashNextModel.forward])
        # The stock init's hook was dropped, not left registered
        # alongside the new one for the life of the process.
        self.assertEqual(list(live_hooks), [2])
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        out = model.model._compiled_callable(None, positions, None,
                                             inputs_embeds=embed)
        want = manual_forward(self.widen(embed), self.file_dirs((1, 3)),
                              alpha=1.0)
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
        want = manual_forward(self.widen(embed), self.file_dirs(layers),
                              alpha=0.5)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_unsteered_path_is_passthrough(self):
        model = self.build()  # no WEIGHTLESS_STEER_PATH: disabled core
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(self.widen(embed), {}, alpha=0.0)
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
        # A mid-stack rank receives the materialized stream via
        # intermediate_tensors.
        model.model.start_layer, model.model.end_layer = 2, 4
        _PPGroup.is_first_rank = False
        embed = torch.randn(5, STREAM_WIDTH)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions,
                        intermediate_tensors={"hidden_states": embed})
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

    def test_single_stream_width_vector_fails_closed(self):
        """A plain hidden_size-wide vector is not this arch's stream."""
        self.write_vector((1, 2), alpha="1.0", width=HIDDEN)
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_bad_vector_fails_closed_at_model_build(self):
        self.write_vector((1, 2), alpha="1.0")
        # Corrupt: declare a layer set that disagrees with the tensors.
        write_gguf(self.path,
                   {**good_meta(layers=(1, 2)),
                    "glp.layer_ids_zero_based": "5,6"},
                   good_tensors(layers=(1, 2), width=STREAM_WIDTH))
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)


class StructureTests(unittest.TestCase):
    """Pin the adapter's copied forward against the vendored reference."""

    # The model layer-loop anchor.
    LOOP_ANCHOR = (
        "        for layer_idx, layer in islice(\n"
        "            enumerate(self.layers), self.start_layer, self.end_layer\n"
        "        ):\n"
    )
    LAYER_CALL_ANCHOR = (
        "            hidden_states, block_output, injection = layer(\n"
        "                hidden_states=hidden_states,\n"
        "                prev_block_output=block_output,\n"
        "                prev_injection=injection,\n"
    )
    # First-rank widening: the flat stream is born here.
    WIDEN_ANCHOR = (
        "            hidden_states = hidden_states.repeat(1, self.config.hc_count)\n"
    )
    # End of the deepstack block: the steering block follows immediately.
    DEEPSTACK_ANCHOR = (
        "                hidden_states = hidden_states + deepstack_embed\n"
    )
    # Stock's UNGUARDED final mixer — the deferred-contract trap. Present in
    # the reference, gone from the adapter (replaced by the guarded form).
    STOCK_MIXER_ANCHOR = (
        "        multi_hidden, sample_hidden_states, _ = final_mixer.combine_and_mix(\n"
        "            hidden_states, block_output, injection\n"
        "        )\n"
    )
    # PP transport and MTP capture blocks must survive verbatim.
    PP_ANCHOR = (
        "        if not get_pp_group().is_last_rank:\n"
        "            # PP transports one tensor, not the delayed HC tuple. Materialize\n"
    )
    MTP_ANCHOR = (
        "            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)\n"
    )

    def test_copied_forward_matches_vendored_reference(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "qwen38fn.py").read_text()
        reference_src = REFERENCE.read_text()
        # The anchors exist verbatim in BOTH the adapter's forward copy and
        # the upstream reference it was copied from.
        for anchor in (self.LOOP_ANCHOR, self.LAYER_CALL_ANCHOR,
                       self.WIDEN_ANCHOR, self.DEEPSTACK_ANCHOR,
                       self.PP_ANCHOR, self.MTP_ANCHOR):
            self.assertIn(anchor, reference_src)
            self.assertIn(anchor, adapter_src)
        # The steering block sits immediately after the deepstack block.
        self.assertIn(
            self.DEEPSTACK_ANCHOR
            + "            # [weightless-steer] the one added block",
            adapter_src,
        )
        # The final-mixer guard: stock's unguarded combine_and_mix is in the
        # reference; the adapter replaces it with the guarded form whose
        # else branch is mix().
        self.assertIn(self.STOCK_MIXER_ANCHOR, reference_src)
        self.assertNotIn(self.STOCK_MIXER_ANCHOR, adapter_src)
        self.assertIn("multi_hidden, sample_hidden_states, _ = final_mixer.mix(",
                      adapter_src)


if __name__ == "__main__":
    unittest.main()
