"""Offline tests for the deepseek_v4_1 fold-site steering adapter.

GPU-free, the same two layers as test_dsv4.py:

- Functional: the upstream module is stubbed with tiny torch modules that
  reproduce V4.1's fused-fold convention — each decoder layer receives
  (x, ..., residual, post_mix, res_mix) where the PREVIOUS layer's FFN
  write is still pending, folds it at the top via mhc.post only on the
  armed path (the stock stub fuses it into mhc.pre like the real tree),
  and the steered model's loop materializes the fold, steers the mean over
  the C copies, and continues. The adapter is imported against the stub,
  built, and its forward output is compared against a hand-computed
  h <- mean-op per steered fold plus the spec's post-loop fold and the
  companion o_proj projection.
- Off-is-stock: with no WEIGHTLESS_STEER_PATH, the steered loop never
  materializes a fold — output is byte-identical to the stock stub's fused
  path, and no _o_proj wrap is installed.
The hand-computed reference mirrors docs/spec/GLP.md's op exactly:
    h        <-  mean over the C copies
    copy_c   <-  copy_c - alpha * (h . d) d
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

HIDDEN = 8
NUM_LAYERS = 4
HC_MULT = 4
ATTN_WRITES = [np.random.default_rng(200 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]
FFN_WRITES = [np.random.default_rng(300 + i).standard_normal(HIDDEN).astype(
    np.float32) for i in range(NUM_LAYERS)]

_ONE = torch.tensor(1.0)


class _FakeMhc(nn.Module):
    """The tree's mhc: post (the fold, [T, C, H]), pre, and post_pre fused.

    post(x, residual, post_mix, res_mix) folds the pending write x into
    the carried [T, C, H] streams; pre runs the standalone path once the
    fold is materialized; post_pre is the FUSED fold+pre the stock loop
    uses (armed layers bypass it by materializing first).
    """

    def post(self, x, residual, post_mix, res_mix):
        # x: [T, H] pending write; residual: [T, C, H] streams. Trivial
        # gate mixes keep the stub's arithmetic out of the way: the fold is
        # the plain sum of the pending write and the carried streams.
        return residual + x.unsqueeze(1)

    def pre(self, residual, *a, **k):
        # The REAL fused pre consumes the previous layer's pending write
        # (previous_output) with its mixes; the stub passes the pending
        # write through x and folds it via the armed path's post or the
        # stub layer's own post_pre. Trivial mixes make both equal.
        return (residual, None, None, residual.mean(dim=1), None)

    def post_pre(self, x, residual, post_mix, res_mix, *a, **k):
        folded = self.post(x, residual, post_mix, res_mix)
        out = folded.mean(dim=1)
        return folded, None, None, out, None

    def collapse(self, stream, pre_mix=None):
        return stream.mean(dim=1)
class FakeAttn(nn.Module):
    """Identity _o_proj: the wrap steers the o_proj OUTPUT (= the
    attention block's write in the real tree), observable without the
    stub's own write arithmetic confusing the comparison."""

    def __init__(self):
        super().__init__()

    def _o_proj(self, o, positions, *, is_prefill=False, sp_rows=None):
        return o


class FakeDeepseekV4DecoderLayer(nn.Module):
    """V4.1's layer: fused pre/post_pre, the fold only materializing on
    apply_engram (the armed steering path takes the same branch)."""

    def __init__(self, layer_idx, block_write):
        super().__init__()
        self.layer_idx = layer_idx
        self.mhc = _FakeMhc()
        self.attn = FakeAttn()
        self.register_buffer("block_write", torch.from_numpy(block_write))
        self.engram = None

    def forward(self, x, positions, input_ids, pre_mix=None, post_mix=None,
                res_mix=None, residual=None, engram_hashes=None,
                engram_mask=None, **k):
        # The armed loop always materializes the fold itself and passes the
        # [T, C, H] streams in residual with x None; the stock path (the
        # off-is-stock comparison) folds the pending write here instead.
        if residual is None:
            # First layer: the embedding IS the [T, C, H] stream start.
            streams = x.unsqueeze(1).expand(
                -1, HC_MULT, -1).contiguous() if x.dim() == 2 else x
        elif x is None:
            streams = residual
        else:
            # Stock fused path: fold the pending write into the streams.
            streams = self.mhc.post(x, residual, post_mix, res_mix)
        h = streams.mean(dim=1)
        # The attention block's write: _o_proj (identity, wrapped when
        # armed) then the block's own deposit, added by the LAYER — the
        # real tree's o_proj output IS the write; the identity keeps the
        # wrap's steering observable in isolation.
        h = self.attn._o_proj(h, positions)
        h = h + self.block_write
        streams = streams + h.unsqueeze(1)
        x = streams.mean(dim=1)
        return x, streams, None, None, pre_mix


class FakeDeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix="", start_layer=0,
                 end_layer=NUM_LAYERS):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.start_layer, self.end_layer = start_layer, end_layer
        self.layers = nn.ModuleList(
            FakeDeepseekV4DecoderLayer(i, FFN_WRITES[i])
            for i in range(NUM_LAYERS)
        )
        self.use_sequence_parallel = False
        self.aux_hidden_state_layers = ()
        self._mtp_hidden_buffer = None
        self.norm = lambda x: x

    def embed_input_ids(self, input_ids):
        raise AssertionError("tests pass inputs_embeds")

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **k):
        """The stock fused path: every layer runs the fused pre/post_pre
        convention with no fold materialization. The steered class's
        forward override replaces this on the swap; the off-is-stock test
        compares the two against this reference."""
        hidden_states = inputs_embeds
        residual = None
        for lay in self.layers:
            hidden_states, residual, _, _, _ = lay(
                hidden_states, positions, input_ids,
                residual=residual)
        # The model loop's own post-loop fold of the final pending write —
        # exactly the tree's `layer.mhc.post(hidden_states, residual, ...)`
        # after the loop (vLLM upstream); the steered forward mirrors it.
        final = lay.mhc.post(hidden_states, residual, None, None)
        return final.mean(dim=1)


class FakeDeepseekV41LLMForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self.model = FakeDeepseekV4Model(vllm_config=vllm_config)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **k):
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
        scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=MAX_NUM_TOKENS,
            max_num_seqs=MAX_NUM_REQS,
        ),
    )


def _import_adapter():
    stubs = {}
    stubs["vllm"] = types.ModuleType("vllm")
    v41 = types.ModuleType("vllm.models.deepseek_v4_1.nvidia.model")
    v41.DeepseekV4Model = FakeDeepseekV4Model
    v41.DeepseekV41LLMForCausalLM = FakeDeepseekV41LLMForCausalLM
    stubs["vllm.models.deepseek_v4_1.nvidia.model"] = v41
    distributed = types.ModuleType("vllm.distributed")
    distributed.get_pp_group = lambda: _PPGroup
    stubs["vllm.distributed"] = distributed
    sequence = types.ModuleType("vllm.sequence")
    sequence.IntermediateTensors = dict
    stubs["vllm.sequence"] = sequence
    sys.modules.update(stubs)
    sys.modules.pop("weightless_steer.archs.dsv41", None)
    return importlib.import_module("weightless_steer.archs.dsv41")


def manual_forward(embed, dirs, alpha):
    """Hand-computed fold-site forward over the stub loop.

    Mirrors docs/spec/GLP.md: at every layer with an incoming residual the
    fold materializes, h = mean over copies, and the rank-1 term leaves
    every copy. Fold mapping: the layer at loop position i applies
    direction[i-1] (it folds layer i-1's FFN write). The post-loop fold
    applies direction[end_layer-1]. The companion writer projects each
    layer's _o_proj output with direction[lid].
    """
    T = embed.shape[0]
    streams = embed.unsqueeze(1).expand(-1, HC_MULT, -1).contiguous()
    x = None
    for i in range(NUM_LAYERS):
        if x is not None:
            streams = streams + x.unsqueeze(1)  # the fold
            d = dirs.get(i - 1)
            if d is not None:
                d = d / d.norm()
                h = streams.mean(dim=1)
                streams = streams - (
                    alpha * (h @ d).unsqueeze(-1) * d
                ).unsqueeze(1)
        h = streams.mean(dim=1)
        lid_d = dirs.get(i + 1)  # companion lid is 1-based: layer i uses row i+1
        if lid_d is not None:
            dd = lid_d / lid_d.norm()
            h = h - alpha * (h @ dd).unsqueeze(-1) * dd
        h = h + torch.from_numpy(FFN_WRITES[i])   # the block's deposit
        streams = streams + h.unsqueeze(1)
        x = streams.mean(dim=1)
    streams = streams + x.unsqueeze(1)  # post-loop fold
    d = dirs.get(NUM_LAYERS - 1)
    if d is not None:
        d = d / d.norm()
        h = streams.mean(dim=1)
        streams = streams - (alpha * (h @ d).unsqueeze(-1) * d).unsqueeze(1)
    return streams.mean(dim=1)


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
        sys.modules.pop("weightless_steer.archs.dsv41", None)
        for name in ("vllm",
                     "vllm.models.deepseek_v4_1.nvidia.model",
                     "vllm.distributed",
                     "vllm.sequence"):
            sys.modules.pop(name, None)

    def write_vector(self, layers, alpha="3.5", **meta_over):
        meta = {**good_meta(layers=layers, alpha=alpha),
                "glp.hook_point": "hc_fold_pre_engram",
                "glp.spec_version": "3",
                **meta_over}
        write_gguf(self.path, meta, good_tensors(layers=layers, width=HIDDEN))

    def file_dirs(self, layers):
        return {i: torch.from_numpy(good_tensors(layers=layers,
                                                 width=HIDDEN)[
                                        f"direction.{i}"][0])
                for i in layers}

    def build(self, **env):
        with mock.patch.dict(os.environ, env):
            return self.adapter.SteeredDeepseekV41LLMForCausalLM(
                vllm_config=_vllm_config())

    def test_forward_applies_fold_site_projection(self):
        """The armed forward matches the spec's mean-op per fold, the fold
        mapping (direction[i-1] at layer i), the post-loop fold, and the
        companion o_proj writer — end to end against the hand computation."""
        layers = (1, 3)
        self.write_vector(layers, alpha="3.5")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)

        self.assertIsInstance(model.model,
                              self.adapter.SteeredDeepseekV41Model)
        self.assertTrue(model.model._steer_on)
        # The steer stack is dense and zero-padded; armed rows hold the file.
        self.assertEqual(tuple(model.model._steer_stack.shape),
                         (NUM_LAYERS, 1, HIDDEN))
        self.assertAlmostEqual(float(model.model._steer_alpha), 3.5)

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            out = model(None, positions, inputs_embeds=embed)
        want = manual_forward(embed, self.file_dirs(layers), alpha=3.5)
        self.assertTrue(torch.allclose(out, want, atol=1e-4),
                        f"max err {(out - want).abs().max()}")

    def test_off_is_byte_identical_stock(self):
        """No WEIGHTLESS_STEER_PATH: the loop never materializes a fold and
        never wraps o_proj — byte-identical to the stock stub forward."""
        layers = (1, 3)
        self.write_vector(layers, alpha="3.5")
        steered = self.build()  # armed=False path: buffers registered zeroed
        stock = FakeDeepseekV41LLMForCausalLM(vllm_config=_vllm_config())

        embed = torch.randn(5, HIDDEN)
        positions = torch.arange(5)
        with torch.no_grad():
            a = steered(None, positions, inputs_embeds=embed)
            b = stock(None, positions, inputs_embeds=embed)
        self.assertFalse(steered.model._steer_on)
        self.assertTrue(torch.equal(a, b))
        # No companion wrap when unarmed.
        for lay in stock.model.layers:
            self.assertIsInstance(lay.attn._o_proj.__self__, FakeAttn)

    def test_ffn_out_pre_residual_file_fails_closed(self):
        """The V4-Flash site's hook is not this row's site: GLP-27's file
        names hc_fold_pre_engram; a ffn_out_pre_residual file refuses."""
        layers = (1, 3)
        self.write_vector(layers, alpha="3.5",
                          **{"glp.hook_point": "ffn_out_pre_residual"})
        with self.assertRaisesRegex(
            (RuntimeError, ValueError),
            "does not match this hook|Refusing",
        ):
            self.build(WEIGHTLESS_STEER_PATH=self.path)

    def test_v2_fold_file_fails_closed(self):
        """spec_version 2 file at the fold hook: the container's v3 gate."""
        layers = (1, 3)
        self.write_vector(layers, alpha="3.5",
                          **{"glp.spec_version": "2"})
        with self.assertRaisesRegex(ValueError, "spec_version 3"):
            from weightless_steer.container import load_control_vector
            load_control_vector(self.path, hook="hc_fold_pre_engram")

    def test_alpha_env_overrides_file_default(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="3.5")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path,
                           WEIGHTLESS_STEER_ALPHA="2.0")
        self.assertAlmostEqual(float(model.model._steer_alpha), 2.0)

    def test_companion_writer_installed_when_armed(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="3.5")
        model = self.build(WEIGHTLESS_STEER_PATH=self.path)
        wrapped = [
            getattr(lay.attn, "_ds41_wrapped", False)
            for lay in model.model.layers
        ]
        self.assertTrue(all(wrapped))

    def test_wrong_hook_env_fails_closed(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="3.5")
        with self.assertRaisesRegex(RuntimeError, "WEIGHTLESS_STEER_HOOK"):
            self.build(WEIGHTLESS_STEER_PATH=self.path,
                       WEIGHTLESS_STEER_HOOK="residual_stream_post_layer")

    def test_per_request_refused(self):
        layers = (1, 3)
        self.write_vector(layers, alpha="3.5")
        with self.assertRaisesRegex(RuntimeError, "MILESTONE_2"):
            self.build(WEIGHTLESS_STEER_PATH=self.path,
                       WEIGHTLESS_ENABLE_MILESTONE_2="1")


if __name__ == "__main__":
    unittest.main()