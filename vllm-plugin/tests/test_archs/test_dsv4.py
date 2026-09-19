"""Offline tests for the deepseek_v4 steering adapter.

Two layers of verification, both GPU-free:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce DSV4's deferred-mHC convention (each decoder layer returns
  (ffn_out, residual, post_mix, res_mix) with the fold of ffn_out into the
  hyper-connection streams deferred to the next layer's fused post/pre
  call, so the model loop's hidden_states IS the pending FFN write
  [T, hidden] — the ffn_out_pre_residual site of the 2026-09-04 hook-site
  correction). The adapter is imported against the stub, the model is
  built, and its forward output is compared against a hand-computed
  h <- h - alpha*(h.d)d per steered layer on that pending write. This
  exercises the registry-shadowed class end to end: the __class__ swap,
  buffer wiring at the UNWIDENED stream width (hidden_size — a hc-widened
  vector must fail closed, that file belongs to the residual-site
  experiment), and global-layer indexing.
- Structural: the adapter's copied forward is pinned against the vendored
  upstream reference (patches/reference/deepseek_v4_nvidia_model.py, a
  byte-identical pull from the day-0 image), so upstream drift in our copy
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
sys.path.insert(0, str(_HERE.parents[3]))          # weightless_runtime/

from glpfiles import good_meta, good_tensors, write_gguf  # noqa: E402

REFERENCE = (_HERE.parents[3] / "patches" / "reference"
             / "deepseek_v4_nvidia_model.py")

HIDDEN = 8
NUM_LAYERS = 4
# Deterministic per-layer sublayer "writes": what each fake decoder layer's
# attention and FFN deposit, plus a per-layer fold bias so the deferred
# state (residual/post_mix/res_mix) is load-bearing in every comparison.
ATTN_WRITES = [np.random.default_rng(200 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
FFN_WRITES = [np.random.default_rng(300 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
MIX_BIASES = [np.random.default_rng(400 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]

_ONE = torch.tensor(1.0)


def _mhc_post(x, residual, post_mix, res_mix):
    """The deferred fold, reduced to its load-bearing algebra."""
    return residual + x * post_mix + res_mix


def _hc_head(h, fn, scale, base, rms_eps, hc_eps):
    return h


class FakeNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = 1e-5

    def forward(self, x):
        return x


class FakeDeepseekV4DecoderLayer(nn.Module):
    """The upstream 4-tuple deferred-mHC convention with trivial sublayers.

    First layer starts standalone (residual=None); later layers fold the
    incoming pending write (fused post+pre), attention and FFN deposit
    fixed writes, and the layer returns (ffn_out, residual, post_mix,
    res_mix) — the FFN write's own fold deferred to the next layer, which
    is exactly why the model loop's hidden_states is the
    ffn_out_pre_residual site.
    """

    def __init__(self, layer_idx, attn_write, ffn_write, mix_bias):
        super().__init__()
        self.layer_idx = layer_idx
        self.register_buffer("attn_write", torch.from_numpy(attn_write))
        self.register_buffer("ffn_write", torch.from_numpy(ffn_write))
        self.register_buffer("mix_bias", torch.from_numpy(mix_bias))

    def forward(self, x, positions, input_ids, post_mix=None, res_mix=None,
                residual=None):
        if residual is None:
            # Standalone pre on the first layer: the stream starts at the
            # embedding.
            residual = x
            post_mix, res_mix = _ONE, self.mix_bias
        else:
            # Fused post+pre: fold the previous layer's pending FFN write,
            # then contract for this layer's attention input.
            residual = residual + x * post_mix + res_mix
            post_mix, res_mix = _ONE, self.mix_bias
            x = residual
        x = x + self.attn_write
        # Fused post+pre: fold the attention write, contract for the FFN.
        residual = residual + x * post_mix + res_mix
        post_mix, res_mix = _ONE, self.mix_bias
        x = residual + self.ffn_write  # the pending FFN write
        return x, residual, post_mix, res_mix


class FakeDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer=0,
                 end_layer=NUM_LAYERS):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(
            FakeDeepseekV4DecoderLayer(i, ATTN_WRITES[i], FFN_WRITES[i],
                                       MIX_BIASES[i])
            for i in range(NUM_LAYERS)
        )
        self.use_mega_moe = False
        self.use_sequence_parallel = False
        self.aux_hidden_state_layers = ()
        self._mtp_hidden_buffer = None
        self.hc_head_fn = self.hc_head_scale = self.hc_head_base = None
        self.rms_norm_eps = 1e-5
        self.hc_eps = 1e-5
        self.norm = FakeNorm(HIDDEN)

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def forward(self, *a, **k):  # replaced by the adapter's override
        raise AssertionError("stock forward must not run after the swap")


class FakeDeepseekV4ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeDeepseekV4Model(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
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
            hf_config=types.SimpleNamespace(
                hidden_size=HIDDEN,
                num_hidden_layers=NUM_LAYERS,
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
    # The adapter's `import vllm.envs as envs` resolves the PARENT package
    # through __import__ (from-imports short-circuit on the full dotted
    # name; the plain-import-as form does not), so the parent is stubbed
    # too. IMPORT_FROM then finds the leaf via the sys.modules fallback.
    stubs["vllm"] = types.ModuleType("vllm")
    dsv4 = types.ModuleType("vllm.models.deepseek_v4.nvidia.model")
    dsv4.DeepseekV4Model = FakeDeepseekV4Model
    dsv4.DeepseekV4ForCausalLM = FakeDeepseekV4ForCausalLM
    stubs["vllm.models.deepseek_v4.nvidia.model"] = dsv4
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: _PPGroup
    stubs["vllm.distributed"] = distributed
    tl = types.ModuleType("vllm.model_executor.kernels.mhc.tilelang")
    tl.mhc_post_tilelang = _mhc_post
    tl.hc_head_fused_kernel_tilelang = _hc_head
    stubs["vllm.model_executor.kernels.mhc.tilelang"] = tl
    sp = types.ModuleType("vllm.models.common.ops.sequence_parallel")
    sp.sp_shard = lambda x: x
    sp.sp_all_gather = lambda x: x
    sp.sp_padding_mask = lambda mask, h: mask
    stubs["vllm.models.common.ops.sequence_parallel"] = sp
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
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.dsv4", None)
    return importlib.import_module("weightless_steer.archs.dsv4")


def manual_forward(embed, dirs, alpha, layers=range(NUM_LAYERS),
                   site="ffn_out_pre_residual"):
    """Hand-computed steered forward over the deferred-mHC loop.

    Per running layer: fold the incoming pending write, the fixed attn
    write, the attn fold, then the pending FFN write x — and the projection
    h <- h - alpha*(h.d)d on THAT pending write for steered layers (the
    ffn_out_pre_residual site). The final mhc_post fold closes the loop;
    hc_head and the final norm are identities in the fake.

    site="post_layer_residual" is the anti-site: the projection lands on
    h + residual (the mixin's fused add+norm convention), the site the
    2026-09-04 hook-site correction rejected for this arch. A correct
    adapter must NOT match it.
    """
    h = embed.clone()
    residual = None
    post_mix = res_mix = None
    for i in layers:
        mb = torch.from_numpy(MIX_BIASES[i])
        if residual is None:
            residual = h
            post_mix, res_mix = _ONE, mb
        else:
            residual = residual + h * post_mix + res_mix
            post_mix, res_mix = _ONE, mb
            h = residual
        h = h + torch.from_numpy(ATTN_WRITES[i])
        residual = residual + h * post_mix + res_mix
        post_mix, res_mix = _ONE, mb
        h = residual + torch.from_numpy(FFN_WRITES[i])
        if i in dirs:
            d = dirs[i] / dirs[i].norm()
            if site == "ffn_out_pre_residual":
                h = h - alpha * (h @ d).unsqueeze(-1) * d
            else:  # post_layer_residual: steer the fold, write back
                hs = h + residual
                hs = hs - alpha * (hs @ d).unsqueeze(-1) * d
                h = hs - residual
    return residual + h * post_mix + res_mix


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
        sys.modules.pop("weightless_steer.archs.dsv4", None)
        for name in ("vllm",
                     "vllm.models.deepseek_v4.nvidia.model",
                     "vllm.distributed",
                     "vllm.model_executor.kernels.mhc.tilelang",
                     "vllm.models.common.ops.sequence_parallel",
                     "vllm.envs",
                     "vllm.forward_context",
                     "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="4.0", width=HIDDEN, **meta_over):
        meta = {**good_meta(layers=layers, alpha=alpha),
                "glp.hook_point": "ffn_out_pre_residual",
                "glp.derived_at": "residual_stream_post_layer",
                **meta_over}
        write_gguf(self.path, meta, good_tensors(layers=layers, width=width))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=HIDDEN)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, **env):
        with mock.patch.dict(os.environ, env):
            return self.adapter.SteeredDeepseekV4ForCausalLM(
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
                                       max_num_reqs=MAX_NUM_REQS)
        return model

    def test_forward_applies_projection_per_layer(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="4.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        # The inner model was swapped onto the steered class, and the
        # buffers are sized to the UNWIDENED stream (hidden_size).
        self.assertIsInstance(model.model, self.adapter.SteeredDeepseekV4Model)
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (NUM_LAYERS, 1, HIDDEN))
        self.assertAlmostEqual(float(model.model._steer_alpha), 4.0)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=4.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_site_is_the_pending_ffn_write(self):
        """Not the post-layer residual: the 2026-09-04 hook-site correction.

        The same vector/alpha at the anti-site (h + residual, the mixin's
        fused add+norm convention) produces a materially different output;
        the adapter must match the pending-write computation and only that.
        """
        layers = (1, 2, 3)
        self.write_vector(layers, alpha="4.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        dirs = self.file_dirs(layers)
        want = manual_forward(embed, dirs, alpha=4.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")
        wrong = manual_forward(embed, dirs, alpha=4.0,
                               site="post_layer_residual")
        self.assertFalse(torch.allclose(out, wrong, atol=1e-4))

    def test_last_layer_is_steered(self):
        """The final fold happens after the last steering apply.

        DSV4 has no in-decoder terminal contract (unlike glm5next): the
        loop steers layer N-1's pending write and the mhc_post fold after
        the loop consumes the steered value.
        """
        layers = (NUM_LAYERS - 1,)
        self.write_vector(layers, alpha="4.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=4.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")
        unsteered = manual_forward(embed, {}, alpha=0.0)
        self.assertFalse(torch.allclose(out, unsteered, atol=1e-4))

    def test_alpha_env_overrides_file_default(self):
        layers = (2,)
        self.write_vector(layers, alpha="6.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_STEER_ALPHA="4.0")
        embed = torch.randn(4, HIDDEN)
        positions = torch.arange(4)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=4.0)
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_alpha_default_comes_from_the_file(self):
        """GLP-29 carries glp.alpha_default=6.0 (the rig's serving dose):
        an env-unset boot must land on the file's value, not 1.0."""
        layers = (2,)
        self.write_vector(layers, alpha="6.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertAlmostEqual(float(model.model._steer_alpha), 6.0)

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
        self.write_vector(layers, alpha="4.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        # Move this rank's window onto global layers 2..3 (make_layers sets
        # these on the inner model at build; the dense stack stays global).
        model.model.start_layer, model.model.end_layer = 2, 4
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        # A PP mid-rank receives the widened stream and starts with
        # residual=None; the fake's standalone-pre branch reproduces that.
        want = manual_forward(embed, self.file_dirs(layers), alpha=4.0,
                              layers=range(2, 4))
        self.assertTrue(torch.allclose(out, want, atol=1e-4))

    def test_per_request_forward_end_to_end(self):
        """Two requests, two alphas, through the real adapter forward."""
        from weightless_runtime.controls import WeightlessResolvedXArgs
        from weightless_steer.control_plane import (
            ScheduledRequest, WeightlessControlPlane,
        )

        layers = (1, 3)
        self.write_vector(layers, alpha="4.0")
        model = self.build_control_harness()
        inner = model.model
        self.assertTrue(inner.weightless_per_request)
        self.assertEqual(inner.weightless_steer_layer_ids, layers)

        plane = WeightlessControlPlane(
            max_num_tokens=MAX_NUM_TOKENS, max_num_reqs=MAX_NUM_REQS,
            num_layers=NUM_LAYERS, default_alpha=4.0, loaded_layers=layers)
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
        with mock.patch.object(FakeDeepseekV4ForCausalLM,
                               "__init__") as allocate:
            with self.assertRaisesRegex(RuntimeError, "runner integration"):
                self.build(WEIGHTLESS_ENABLE_MILESTONE_2="1")
        allocate.assert_not_called()

    def test_compilation_rebinds_to_steered_forward(self):
        """The compile wrapper must capture the STEERED forward.

        The day-0 image's DeepseekV4Model carries no @support_torch_compile,
        so the rebind is dormant; this pins its behaviour for the day
        upstream decorates the class — without it the compiled callable
        would run the stock forward and compiled serving would be silently
        unsteered.
        """
        original_init = FakeDeepseekV4Model.__init__
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
        self.write_vector((1, 3), alpha="4.0")
        previous = sys.modules.get("vllm.compilation.wrapper")
        sys.modules["vllm.compilation.wrapper"] = wrapper_module
        try:
            with mock.patch.object(FakeDeepseekV4Model, "__init__", init):
                model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        finally:
            if previous is None:
                sys.modules.pop("vllm.compilation.wrapper", None)
            else:
                sys.modules["vllm.compilation.wrapper"] = previous
        self.assertEqual(captured, [FakeDeepseekV4Model.forward,
                                    self.adapter.SteeredDeepseekV4Model.forward])
        # The stock init's hook was dropped, not left registered
        # alongside the new one for the life of the process.
        self.assertEqual(list(live_hooks), [2])
        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        out = model.model._compiled_callable(None, positions, None,
                                             inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs((1, 3)), alpha=4.0)
        torch.testing.assert_close(out, want, atol=1e-4, rtol=1e-4)

    def test_gate_off_registers_no_per_request_buffers(self):
        """The default deployment is untouched by any of this."""
        self.write_vector((1,), alpha="4.0")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertFalse(model.model.weightless_per_request)
        for name in ("_steer_alpha_rows", "_steer_slot_rows",
                     "_steer_layer_bank"):
            self.assertNotIn(name, dict(model.model.named_buffers()))

    def test_widened_stream_width_vector_fails_closed(self):
        """A hc_mult-widened vector is not this arch's steering site.

        DSV4 IS a hyper-connection arch (hc_mult=4), but the site is the
        pre-fold single-stream FFN write: the GLP-29 directions are
        hidden_size-wide. A widened file belongs to the residual-site
        experiment and must die here, not broadcast at serve time.
        """
        self.write_vector((1, 2), alpha="4.0", width=HIDDEN * 4)
        with self.assertRaisesRegex(RuntimeError, "width"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_residual_hook_file_fails_closed(self):
        """A residual_stream_post_layer file is the mismatch case.

        The hook the pre-2026-09-04 hotfix claimed to be; the corrected
        lane refuses it rather than editing the wrong stream.
        """
        write_gguf(self.path,
                   good_meta(layers=(1, 2)),  # hook_point=residual_stream...
                   good_tensors(layers=(1, 2), width=HIDDEN))
        with self.assertRaisesRegex(ValueError, "hook"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_transferred_vector_warns_but_loads(self):
        """GLP-29 itself: hook ffn_out_pre_residual, derived_at residual.

        A transferred vector is legal (GLP.md) and must not be silent.
        """
        self.write_vector((1, 2), alpha="4.0")
        with self.assertLogs("weightless_steer.container", level="WARNING") as cm:
            model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        self.assertTrue(any("transferred vector" in line
                            for line in cm.output), cm.output)
        self.assertEqual(sorted(model.model._steer_core.dirs), [1, 2])

    def test_bad_vector_fails_closed_at_model_build(self):
        self.write_vector((1, 2), alpha="4.0")
        # Corrupt: declare a layer set that disagrees with the tensors.
        self.write_vector((1, 2), alpha="4.0",
                          **{"glp.layer_ids_zero_based": "5,6"})
        with self.assertRaisesRegex(ValueError, "layer_ids_zero_based"):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_wrong_hook_env_fails_closed(self):
        """WEIGHTLESS_STEER_HOOK other than ffn_out_pre_residual: refuse."""
        self.write_vector((1, 2), alpha="4.0")
        with self.assertRaisesRegex(RuntimeError, "WEIGHTLESS_STEER_HOOK"):
            self.build(WEIGHTLESS_STEER_PATH=self.path,
                       WEIGHTLESS_STEER_HOOK="residual_stream_post_layer")


class StructureTests(unittest.TestCase):
    """Pin the adapter's copied forward against the vendored reference."""

    # The model layer loop (the hotfix's forward anchor context).
    LOOP_ANCHOR = (
        "        for idx, layer in enumerate(\n"
        "            islice(self.layers, self.start_layer, self.end_layer),\n"
        "            start=self.start_layer,\n"
        "        ):\n"
        "            hidden_states, residual, post_mix, res_mix = layer(\n"
        "                hidden_states,\n"
        "                positions,\n"
        "                input_ids,\n"
        "                post_mix,\n"
        "                res_mix,\n"
        "                residual,\n"
        "            )\n"
    )
    # The hotfix's forward anchor (patches/hotfix-dsv4-steering-projective.py
    # ANCHOR_FORWARD): the steering block sits between these two lines.
    HOTFIX_FORWARD_ANCHOR = (
        "                residual,\n"
        "            )\n"
        "            if idx + 1 in self.aux_hidden_state_layers:"
    )
    # The post-loop fold: consumes the (steered) last pending write.
    FOLD_ANCHOR = (
        "        if layer is not None:\n"
        "            # Reuse if the last layer was captured as an aux hidden state\n"
        "            if self.end_layer in self.aux_hidden_state_layers:\n"
        "                hidden_states = final_aux_recon\n"
        "            else:\n"
        "                hidden_states = mhc_post_tilelang(\n"
        "                    hidden_states, residual, post_mix, res_mix\n"
        "                )\n"
    )

    def test_copied_loop_matches_vendored_reference(self):
        adapter_src = (_HERE.parents[2] / "weightless_steer" / "archs"
                       / "dsv4.py").read_text()
        reference_src = REFERENCE.read_text()
        # The anchors exist verbatim in BOTH the adapter's forward copy and
        # the upstream reference it was copied from.
        self.assertIn(self.LOOP_ANCHOR, reference_src)
        self.assertIn(self.LOOP_ANCHOR, adapter_src)
        self.assertIn(self.FOLD_ANCHOR, reference_src)
        self.assertIn(self.FOLD_ANCHOR, adapter_src)
        # The vendored reference is the file the hotfix patches: its forward
        # anchor must be present, tying reference <-> hotfix lane.
        self.assertIn(self.HOTFIX_FORWARD_ANCHOR, reference_src)
        # The steering block sits immediately after the layer call...
        self.assertIn(
            self.LOOP_ANCHOR
            + "            # [weightless-steer] the one added line",
            adapter_src,
        )
        # ...and before the post-loop fold.
        self.assertLess(adapter_src.index("self._steer_core.apply(idx"),
                        adapter_src.index(self.FOLD_ANCHOR))
        # The upstream module path: catches a future rehoming (glm5next's
        # nvidia.model -> common.model move is the live example).
        self.assertIn("from vllm.models.deepseek_v4.nvidia.model import",
                      adapter_src)


if __name__ == "__main__":
    unittest.main()
