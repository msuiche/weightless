"""Offline tests for the glm5next steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce glm5next's deferred mHC convention (each decoder layer returns
  (hidden_states, residual, post, comb) with hc_post fused into the next
  layer's pre; the post-layer stream is the parameter-free
  layer.hc_post(...) materialization [T, n, hidden]). The adapter is
  imported against the stub, the model is built, and its forward output is
  compared against a hand-computed  h <- h - alpha*(h.d)d  per steered
  layer on the HC-outer-flattened stream. This exercises the
  registry-shadowed class end to end: both __class__ swaps (the inner
  model, and every decoder layer onto the class that defers the last
  layer's terminal contract), buffer wiring at the widened stream width,
  global-layer indexing, and the contract-after-steering on the final
  layer.
- Structural: the adapter's copied forward loops are pinned against the
  vendored upstream reference (patches/reference/glm5next.py), so upstream
  drift in our copies is caught here rather than at serve time.
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
             / "glm5next.py")

HIDDEN = 8
N_STREAMS = 3
STREAM_WIDTH = HIDDEN * N_STREAMS
NUM_LAYERS = 4
# Deterministic per-layer sublayer "writes": what each fake decoder layer's
# attention and MLP deposit into the mHC stream.
ATTN_WRITES = [np.random.default_rng(200 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
FFN_WRITES = [np.random.default_rng(300 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
# Per-layer, per-stream comb biases: the fake's deferred-state mix marker,
# so the n streams diverge even unsteered and the HC-outer flatten order is
# load-bearing in every comparison.
COMB_BIASES = [np.random.default_rng(400 + i).standard_normal(
    N_STREAMS).astype(np.float32) for i in range(NUM_LAYERS)]

# The deferred post-mix marker: with post=1 the fake's hc_post is
# residual + x broadcast across streams + the comb bias — pure and
# parameter-free like the real MHCPostOp, trivial to reproduce in the
# hand-computed reference.
_ONE = torch.tensor(1.0)


def _hc_expand(x, n):
    return x.unsqueeze(1).expand(-1, n, -1).contiguous()


def _hc_contract(x, n):
    return x.mean(dim=1)


class FakeNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = 1e-5

    def forward(self, x, residual=None):
        if residual is None:
            return x
        fold = x + residual
        return fold, fold


class FakeAttn(nn.Module):
    def __init__(self, write):
        super().__init__()
        self.register_buffer("write", torch.from_numpy(write))

    def forward(self, hidden_states, positions):
        return hidden_states + self.write


class FakeMLP(nn.Module):
    def __init__(self, write):
        super().__init__()
        self.register_buffer("write", torch.from_numpy(write))

    def forward(self, x, already_sequence_parallel=False):
        return x + self.write


class FakeGlm5NextDecoderLayer(nn.Module):
    """The upstream 4-tuple mHC convention with trivial sublayers.

    Entry folds the incoming deferral (fused post+pre) or starts standalone
    (post=None); attention and MLP read the contracted stream and their
    writes land on every stream via the post/comb mixes; the layer returns
    (ffn_out, stream, 1, 0) so hc_post reassembles the post-layer stream.
    The last mHC layer materializes and contracts in-decoder — the stock
    special case the adapter's swap removes.
    """

    def __init__(self, layer_idx, attn_write, ffn_write, comb_bias,
                 mhc=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_hidden_layers = NUM_LAYERS
        self.mhc = mhc
        self.is_mtp_layer = False
        self.is_sequence_parallel = False
        self.n = N_STREAMS
        self.hidden_size = HIDDEN
        self.input_layernorm = FakeNorm(HIDDEN)
        self.post_attention_layernorm = FakeNorm(HIDDEN)
        self.self_attn = FakeAttn(attn_write)
        self.mlp = FakeMLP(ffn_write)
        self._mlp_is_moe = False
        self.register_buffer("comb_bias", torch.from_numpy(comb_bias))
        self.hc_attn_fn = self.hc_attn_scale = self.hc_attn_base = None
        self.hc_ffn_fn = self.hc_ffn_scale = self.hc_ffn_base = None

    def hc_post(self, x, residual, post, comb):
        return residual + x.unsqueeze(1) * post + comb.view(1, -1, 1)

    def hc_pre(self, x, hc_fn, hc_scale, hc_base,
               norm_weight=None, norm_eps=0.0):
        return _ONE, self.comb_bias, x.mean(dim=1) if x.dim() == 3 else x

    def hc_fused_post_pre(self, x, residual, post, comb, hc_fn, hc_scale,
                          hc_base, norm_weight=None, norm_eps=0.0):
        residual = self.hc_post(x, residual, post, comb)
        return residual, _ONE, self.comb_bias, residual.mean(dim=1)

    def forward(self, positions, hidden_states, residual=None, post=None,
                comb=None):
        if not self.mhc or self.is_mtp_layer:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
            hidden_states, residual = self.post_attention_layernorm(
                attn_output, residual=residual
            )
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states, residual, None, None

        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                x = _hc_expand(x, self.n)
            residual = x
            post, comb, x = self.hc_pre(
                x, None, None, None,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )
        else:
            residual, post, comb, x = self.hc_fused_post_pre(
                x, residual, post, comb, None, None, None,
                norm_weight=self.input_layernorm.weight.data,
                norm_eps=self.input_layernorm.variance_epsilon,
            )

        x = self.self_attn(
            hidden_states=x,
            positions=positions,
        )

        residual, post, comb, x = self.hc_fused_post_pre(
            x, residual, post, comb, None, None, None,
            norm_weight=self.post_attention_layernorm.weight.data,
            norm_eps=self.post_attention_layernorm.variance_epsilon,
        )

        x = self.mlp(x)

        if self.layer_idx == self.num_hidden_layers - 1:
            x = self.hc_post(x, residual, post, comb)
            x = _hc_contract(x, self.n)
            return x, None, None, None

        return x, residual, post, comb


class FakeGlm5NextModel(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer=0,
                 end_layer=NUM_LAYERS):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(
            FakeGlm5NextDecoderLayer(i, ATTN_WRITES[i], FFN_WRITES[i],
                                     COMB_BIASES[i], mhc=self.config.mhc)
            for i in range(NUM_LAYERS)
        )
        self._active_layers = self.layers[start_layer:end_layer]
        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )
        self.norm = FakeNorm(HIDDEN)

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeGlm5NextForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeGlm5NextModel(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kw):
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)


class _PPGroup:
    is_first_rank = True
    is_last_rank = True


MAX_NUM_TOKENS = 16
MAX_NUM_REQS = 4


def _vllm_config(mhc=True):
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(
                hidden_size=HIDDEN,
                num_hidden_layers=NUM_LAYERS,
                mhc_num_residual_streams=N_STREAMS,
                mhc=mhc,
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
        parallel_config=types.SimpleNamespace(
            use_sequence_parallel_moe=False,
        ),
    )


def _import_adapter():
    """Install vllm stubs in sys.modules and import the adapter fresh."""
    stubs = {}
    glm = types.ModuleType("vllm.models.glm5next.nvidia.model")
    glm.Glm5NextDecoderLayer = FakeGlm5NextDecoderLayer
    glm.Glm5NextModel = FakeGlm5NextModel
    glm.Glm5NextForCausalLM = FakeGlm5NextForCausalLM
    stubs["vllm.models.glm5next.nvidia.model"] = glm
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: _PPGroup
    stubs["vllm.distributed"] = distributed
    mhc = types.ModuleType("vllm.model_executor.layers.mhc")
    mhc.hc_expand = _hc_expand
    mhc.hc_contract = _hc_contract
    stubs["vllm.model_executor.layers.mhc"] = mhc
    sp = types.ModuleType("vllm.models.common.ops.sequence_parallel")
    sp.sp_shard = lambda x: x
    sp.sp_all_gather = lambda x: x
    sp.sp_reduce_scatter = lambda x: x
    stubs["vllm.models.common.ops.sequence_parallel"] = sp
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.glm5next", None)
    return importlib.import_module("weightless_steer.archs.glm5next")


def manual_forward(embed, dirs, alpha, layers=range(NUM_LAYERS),
                   steer_layers=None):
    """Hand-computed steered forward over the widened mHC stream.

    Per running layer: the attn write lands on every stream plus the
    per-stream comb bias, then the ffn write the same way (the fake's
    hc_post decomposition), then the projection on the HC-outer-flattened
    stream for steered layers; the last layer contracts (mean over
    streams). `layers` selects which decoder layers RUN; `steer_layers`,
    when given, further restricts which of them steer.
    """
    if embed.dim() == 2:
        s = _hc_expand(embed, N_STREAMS).clone()
    else:
        s = embed.clone()
    for i in layers:
        c = torch.from_numpy(COMB_BIASES[i]).view(1, N_STREAMS, 1)
        a = s.mean(dim=1) + torch.from_numpy(ATTN_WRITES[i])
        s = s + a.unsqueeze(1) + c
        f = s.mean(dim=1) + torch.from_numpy(FFN_WRITES[i])
        s = s + f.unsqueeze(1) + c
        if i in dirs and (steer_layers is None or i in steer_layers):
            d = dirs[i] / dirs[i].norm()
            flat = s.flatten(-2)
            s = (flat - alpha * (flat @ d).unsqueeze(-1) * d).reshape(
                s.shape)
    return s.mean(dim=1)


def manual_forward_non_mhc(embed):
    """The non-mHC stack: per layer the fused add+norm fold doubles the
    stream (attn passthrough + fold + mlp passthrough + fold)."""
    h = embed.clone()
    for i in range(NUM_LAYERS):
        fold = 2 * h + torch.from_numpy(ATTN_WRITES[i])
        h = 2 * fold + torch.from_numpy(FFN_WRITES[i])
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
        sys.modules.pop("weightless_steer.archs.glm5next", None)
        for name in ("vllm.models.glm5next.nvidia.model",
                     "vllm.distributed",
                     "vllm.model_executor.layers.mhc",
                     "vllm.models.common.ops.sequence_parallel",
                     "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="2.0", width=STREAM_WIDTH):
        write_gguf(self.path, good_meta(layers=layers, alpha=alpha),
                   good_tensors(layers=layers, width=width))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=STREAM_WIDTH)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, **env):
        with mock.patch.dict(os.environ, env):
            return self.adapter.SteeredGlm5NextForCausalLM(
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

    def test_forward_applies_projection_per_layer(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model and every decoder layer were swapped onto the
        # steered classes, and the buffers are sized to the widened stream.
        self.assertIsInstance(model.model, self.adapter.SteeredGlm5NextModel)
        for layer in model.model.layers:
            self.assertIsInstance(layer,
                                  self.adapter.SteeredGlm5NextDecoderLayer)
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (NUM_LAYERS, 1, STREAM_WIDTH))
        self.assertAlmostEqual(float(model.model._steer_alpha), 2.0)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=2.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_last_layer_is_steered(self):
        """The L44 case: the final mHC layer must not escape steering.

        Stock contracts the last layer's stream inside the decoder and
        returns post=None, which the loop guard would skip. The layer-class
        swap defers that contract, so the loop materializes, steers and
        contracts the last layer like any other.
        """
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

    def test_steer_mhc_stream_site(self):
        """The site helper on a mock layer: materialize, flatten HC-outer,
        project, reshape; contract only on the last layer."""
        layers = (1,)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        inner = model.model
        d = self.file_dirs(layers)[1]
        d = d / d.norm()

        x = torch.randn(4, HIDDEN)
        res = torch.randn(4, N_STREAMS, HIDDEN)
        layer = inner.layers[1]
        out, r, p, c = inner._steer_mhc_stream(layer, x, res,
                                               _ONE, layer.comb_bias)
        stream = res + x.unsqueeze(1) + layer.comb_bias.view(1, -1, 1)
        flat = stream.flatten(-2)
        want = (flat - 2.0 * (flat @ d).unsqueeze(-1) * d).reshape(
            stream.shape)
        self.assertEqual(out.shape, (4, N_STREAMS, HIDDEN))
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")
        self.assertIsNone(r)
        self.assertIsNone(p)
        self.assertIsNone(c)

        # An unsteered layer's zero stack row is a numeric no-op.
        layer0 = inner.layers[0]
        out0, *_ = inner._steer_mhc_stream(layer0, x, res,
                                           _ONE, layer0.comb_bias)
        stream0 = res + x.unsqueeze(1) + layer0.comb_bias.view(1, -1, 1)
        self.assertTrue(torch.allclose(out0, stream0, atol=1e-6))

        # The last layer contracts after steering: [T, n, hidden] ->
        # [T, hidden].
        layers = (NUM_LAYERS - 1,)
        self.write_vector(layers, alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        inner = model.model
        d = self.file_dirs(layers)[NUM_LAYERS - 1]
        d = d / d.norm()
        last = inner.layers[NUM_LAYERS - 1]
        out, r, p, c = inner._steer_mhc_stream(last, x, res,
                                               _ONE, last.comb_bias)
        stream_last = res + x.unsqueeze(1) + last.comb_bias.view(1, -1, 1)
        flat = stream_last.flatten(-2)
        want = (flat - 2.0 * (flat @ d).unsqueeze(-1) * d).reshape(
            stream_last.shape).mean(dim=1)
        self.assertEqual(out.shape, (4, HIDDEN))
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

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
        with mock.patch.object(FakeGlm5NextForCausalLM,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build(WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()

    def test_compilation_rebinds_to_steered_forward(self):
        """The compile wrapper must capture the STEERED forward.

        glm5next carries no @support_torch_compile today, so the rebind is
        dormant; this pins its behaviour for the day upstream decorates the
        class — without it the compiled callable would run the stock
        forward and compiled serving would be silently unsteered.
        """
        original_init = FakeGlm5NextModel.__init__
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
            with mock.patch.object(FakeGlm5NextModel, "__init__", init):
                model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        finally:
            if previous is None:
                sys.modules.pop("vllm.compilation.wrapper", None)
            else:
                sys.modules["vllm.compilation.wrapper"] = previous
        self.assertEqual(captured, [FakeGlm5NextModel.forward,
                                    self.adapter.SteeredGlm5NextModel.forward])
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
        model.model._active_layers = model.model.layers[2:4]
        # Mid-stack state is the widened materialized stream, [T, n, hidden].
        embed = torch.randn(5, N_STREAMS, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=1.0,
                              layers=range(2, 4))
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_non_mhc_layers_are_never_steered(self):
        """MTP/70B-style layers return post=None; the guard skips them."""
        self.write_vector((1, 2), alpha="2.0")
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_STEER_PATH": self.path}):
            model = self.adapter.SteeredGlm5NextForCausalLM(
                vllm_config=_vllm_config(mhc=False))
        # The vector loaded (the core is armed) but every layer returned
        # post=None, so nothing was projected out.
        self.assertTrue(model.model._steer_core.dirs)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        self.assertTrue(torch.allclose(out, manual_forward_non_mhc(embed),
                                       atol=1e-5))

    def test_gate_off_registers_no_per_request_buffers(self):
        """The default deployment is untouched by any of this."""
        self.write_vector((1,), alpha="2.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertFalse(model.model.weightless_per_request)
        for name in ("_steer_alpha_rows", "_steer_slot_rows",
                     "_steer_layer_bank"):
            self.assertNotIn(name, dict(model.model.named_buffers()))

    def test_single_stream_width_vector_fails_closed(self):
        """A plain hidden_size-wide vector is not this arch's stream."""
        self.write_vector((1, 2), alpha="2.0", width=HIDDEN)
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_bad_vector_fails_closed_at_model_build(self):
        self.write_vector((1, 2), alpha="2.0")
        # Corrupt: declare a layer set that disagrees with the tensors.
        write_gguf(self.path,
                   {**good_meta(layers=(1, 2)),
                    "glp.layer_ids_zero_based": "5,6"},
                   good_tensors(layers=(1, 2), width=STREAM_WIDTH))
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)


class StructureTests(unittest.TestCase):
    """Pin the adapter's copied forwards against the vendored reference."""

    # The model layer-loop anchor.
    LOOP_ANCHOR = (
        "        for layer in self._active_layers:\n"
        "            hidden_states, residual, post, comb = layer(\n"
        "                positions, hidden_states, residual, post, comb\n"
        "            )\n"
    )
    # The head of the decoder-layer mHC branch, pinning the layer copy.
    LAYER_PRE_ANCHOR = (
        "        x = hidden_states\n"
        "        if post is None:\n"
        "            if self.layer_idx == 0:\n"
        "                x = hc_expand(x, self.n)\n"
        "            residual = x\n"
    )
    # Stock's last-layer terminal contract — the block the adapter removes.
    LAYER_LAST_ANCHOR = (
        "        if self.layer_idx == self.num_hidden_layers - 1:\n"
        "            x = self.hc_post(x, residual, post, comb)\n"
        "            x = hc_contract(x, self.n)\n"
        "            return x, None, None, None\n"
    )

    def test_copied_loops_match_vendored_reference(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "glm5next.py").read_text()
        reference_src = REFERENCE.read_text()
        # The anchors exist verbatim in BOTH the adapter's forward copies
        # and the upstream reference they were copied from.
        self.assertIn(self.LOOP_ANCHOR, reference_src)
        self.assertIn(self.LOOP_ANCHOR, adapter_src)
        self.assertIn(self.LAYER_PRE_ANCHOR, reference_src)
        self.assertIn(self.LAYER_PRE_ANCHOR, adapter_src)
        # The steering block sits immediately after the layer call.
        self.assertIn(
            self.LOOP_ANCHOR
            + "            # [weightless-steer] the one added block",
            adapter_src,
        )
        # The L44 deferral: the stock terminal contract is in the
        # reference, and exactly that block is gone from the adapter's
        # layer copy (replaced by the plain defer return).
        self.assertIn(self.LAYER_LAST_ANCHOR, reference_src)
        self.assertNotIn(self.LAYER_LAST_ANCHOR, adapter_src)
        self.assertIn("        return x, residual, post, comb\n",
                      adapter_src)


if __name__ == "__main__":
    unittest.main()
