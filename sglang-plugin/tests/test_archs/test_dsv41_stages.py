"""The DeepSeek-V4.1 row on SGLang's real model loop and real stage boundaries, on CPU, split over three pipeline stages.

Real, from the importable SGLang: DeepseekV4Model._forward_impl and _forward_layers_hc_pre_from_prev,
DeepseekV4CudaDecoderLayer.forward, make_pp_layers with its stage declarations and FinalRead, hc_boundary (HcUpdate, HcRead,
materialize), the residual stream, the FFN exit that writes the stream at a hand-off, run_attn_post, run_moe_post, post,
combine, get_pp_indices, stage_sources, pack_stage_outputs, unpack_stage_inputs, SourceTap. Stubs: the attention and MoE
compute (fixed functions of their input), the Engram lookup and the kernels only Triton serves (sinkhorn split, combine,
broadcast: SGLang's torch reference or one torch line). Needs an SGLang with the V4.1 pipeline wiring (the Thireus/sglang
fork; upstream main refuses PP in the loop) and skips otherwise. The width is the real 5120, three rows per forward.
"""
import contextlib
import functools
import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import glpfiles  # noqa: E402  (sets up the import paths)
from glpfiles import good_meta, write_gguf  # noqa: E402

from test_archs import unit_dirs  # noqa: E402
from weightless_sglang.archs import dsv41 as A  # noqa: E402
from weightless_sglang.core import steer_rows  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402

os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "0")

try:
    import sglang.kernels.ops.layernorm.mhc as RKERN
    from sglang.srt.distributed.utils import get_pp_indices
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.mem_cache.deepseek_v41_pp_sources import SourceTap, pipeline_stages, stage_sources
    from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode, PPProxyTensors
    from sglang.srt.models import deepseek_v4 as M
    from sglang.srt.models import deepseek_v4_boundary as RB
    from sglang.srt.models import deepseek_v4_mhc as RM
    from sglang.srt.models.deepseek_common import moe_transport
    from sglang.srt.runtime_context import get_parallel
    from sglang.test import boundary_fixtures as fixture
    from sglang.test.kernels.mhc import hc_split_sinkhorn_torch
    HAVE, WHY = True, ""
except Exception as e:  # no SGLang, or one without the V4.1 pipeline wiring
    HAVE, WHY = False, f"{type(e).__name__}: {e}"

GLP1017 = glpfiles.find_glp("glp.deepseek-v41-flash-1017-GLP-39-L1-39-a0.75.gguf")
GLP27 = glpfiles.find_glp("glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf")
FOLD = "hc_fold_pre_engram"
W, S, T, Q_LORA = 5120, 4, 3, 8
PART = "13,13,14"
RES = "residual_stream_post_layer"
COMPRESS = [0, 0] + [2] * 18 + [1] * 20
KV_SOURCES, INDEX_SOURCES, CANDIDATE, ENGRAM = [2, 8, 14, 20], [2, 8, 14, 20, 24, 28, 32, 36], 20, [1, 14]


def bits(t):
    return t.contiguous().view(-1).view(torch.uint8)


def same(a, b):
    return a.shape == b.shape and torch.equal(bits(a), bits(b))


def config():
    return types.SimpleNamespace(
        hidden_size=W, hc_mult=S, num_hidden_layers=40, model_type="deepseek_v41", hc_pre_from_prev_sublayer=True,
        compress_ratios=COMPRESS, kv_source_layer_ids=KV_SOURCES, index_source_layer_ids=INDEX_SOURCES,
        candidate_source_layer_id=CANDIDATE, q_lora_rank=Q_LORA, engram_layer_ids=ENGRAM, hc_sinkhorn_iters=20, hc_eps=1e-6,
        rms_norm_eps=1e-6)


class StubAttn(nn.Module):
    """A fixed function of its input; records the source tap the way the attention backend does."""

    def __init__(self, layer_id):
        super().__init__()
        g = torch.Generator().manual_seed(2000 + layer_id)
        self.register_buffer("w", 0.5 + torch.rand(W, generator=g), persistent=False)
        self.attn_tp_size = 1
        self.wo_b = types.SimpleNamespace(reduce_results=False)
        self.pp_source_tap = None
        self.seen = []

    def accepts_mxfp8_swizzled_input(self):
        return False

    def accepts_deepgemm_mxfp8_input(self):
        return False

    def maybe_use_decode_attn_tp(self, forward_batch):
        return contextlib.nullcontext()

    def forward(self, x, positions, forward_batch, x_quant=None, defer_all_reduce=False):
        self.seen.append(x.clone())
        if self.pp_source_tap is not None:
            self.pp_source_tap.record(x, x[:, :Q_LORA].clone())
        return torch.tanh(x.float() * self.w).to(x.dtype)


class StubMoe(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        g = torch.Generator().manual_seed(3000 + layer_id)
        self.register_buffer("w", 2.0 * torch.rand(W, generator=g) - 1.0, persistent=False)
        self.tp_size = 1
        self.reduce_results = False

    def forward(self, x, forward_batch, input_ids=None, input_ids_global=None, skip_shared_experts=False,
                return_moe_output=False, leave_output_sum=False):
        return (x.float() * self.w).to(x.dtype)


class StubEngram(nn.Module):
    layer_hash_index = 0

    def __init__(self, layer_id):
        super().__init__()
        g = torch.Generator().manual_seed(4000 + layer_id)
        self.register_buffer("w", 2.0 * torch.rand(W, generator=g) - 1.0, persistent=False)
        self.seen = []

    def forward(self, residual, hash_ids, forward_batch, input_ids=None, cp_all_tokens=False):
        self.seen.append(residual.clone())
        return (residual.float() + 0.1 * torch.tanh(residual.float() * self.w)).to(residual.dtype)


def sub_hc(cfg, seed, norm):
    g = torch.Generator().manual_seed(seed)
    return RM.HcSubLayer(cfg, 0.05 * torch.randn((2 + S) * S, S * W, generator=g), 1.0 + 0.1 * torch.randn(3, generator=g),
                         0.1 * torch.randn((2 + S) * S, generator=g), norm)


def make_classes():
    """The real classes' forward paths under the names the row requires; only the construction is replaced."""

    class _Layer(M.DeepseekV4CudaDecoderLayer):
        def __init__(self, cfg, layer_id):  # the base __init__ builds the quantized weights
            nn.Module.__init__(self)
            self.config, self.layer_id = cfg, layer_id
            self.hidden_size, self.hc_mult = W, S
            self.boundary_pre_from_prev, self.boundary_mhc_fusions = True, True
            self.hc_stats_stream, self.use_fused_mhc_post_pre, self.hc_pre_from_prev_sublayer = None, False, True
            self.self_attn, self.mlp = StubAttn(layer_id), StubMoe(layer_id)
            self.input_layernorm = RMSNorm(W, eps=1e-6)
            self.post_attention_layernorm = RMSNorm(W, eps=1e-6)
            self.hc_cfg = RM.HcConfig(mult=S, sinkhorn_iters=20, eps=1e-6, rms_eps=1e-6, hidden=W, pre_from_prev=True,
                                      cp_prefill=False)
            self.attn_hc = sub_hc(self.hc_cfg, 5000 + layer_id, self.input_layernorm)
            self.ffn_hc = sub_hc(self.hc_cfg, 6000 + layer_id, self.post_attention_layernorm)
            self.engram = StubEngram(layer_id) if layer_id in cfg.engram_layer_ids else None
            decl = RB.stages(cfg, layer_id, layer=self)
            self.attn_boundary, self.ffn_boundary = M.append_stages((decl[0], self.input_layernorm),
                                                                    (decl[1], self.post_attention_layernorm))

    layer_cls = type("DeepseekV4CudaDecoderLayer", (_Layer,), {})

    class _Model(M.DeepseekV4Model):
        def __init__(self, cfg, rank, size):
            nn.Module.__init__(self)
            self.config = cfg
            self.pp_group = types.SimpleNamespace(world_size=size, is_first_rank=rank == 0, is_last_rank=rank == size - 1,
                                                  rank_in_group=rank)
            self.hidden_size, self.hc_mult, self.hc_eps, self.norm_eps = W, S, 1e-6, 1e-6
            self.hc_pre_from_prev_sublayer = True
            # the model module's own import: utils/common.py before the layer_list move, layers/utils/layer_list.py after
            self.layers, self.start_layer, self.end_layer = M.make_pp_layers(
                cfg.num_hidden_layers, lambda idx, prefix: layer_cls(cfg, idx), prefix="layers", final_read=RB.FinalRead(),
                stage_facts=functools.partial(RB.stages, cfg))
            self.norm = RMSNorm(W, eps=1e-6) if self.pp_group.is_last_rank else M.PPMissingLayer()
            self.embed_tokens = nn.Identity()
            self.pp_sources, self.pp_source_plan = None, None
            if size > 1:
                stages = pipeline_stages(cfg.num_hidden_layers, size)
                self.pp_source_plan = stage_sources(cfg, stages, rank)
                self.pp_sources = nn.ModuleDict({str(i): nn.Module() for i in self.pp_source_plan.replayed})
                for i in self.pp_source_plan.replayed:
                    self.pp_sources[str(i)].seen = []
                for i in self.pp_source_plan.captured:
                    self.layers[i].self_attn.pp_source_tap = SourceTap()
            self.engram_hasher = lambda input_ids, batch: torch.zeros((input_ids.shape[0], 1), dtype=torch.int64)
            self.dspark_layers_to_capture, self.late_layer_start, self.decoder_replay_tail = None, None, None
            self.tail_writes_source_kv, self.late_layer_vl_routing, self.use_fused_mhc_post_pre = False, True, False

    model_cls = type("DeepseekV4Model", (_Model,), {})
    causal = type("DeepseekV4ForCausalLM", (nn.Module,), {})
    return model_cls, causal


class Backend:
    tail_forward_metadata = None

    def forward_low_ratio_sources(self, layer, x, q_lora, positions, forward_batch):
        layer.seen.append(x.clone())


def torch_combine(x_flat, pre, hc, dtype):
    n = x_flat.shape[0]
    return (pre.float().unsqueeze(-1) * x_flat.float().view(n, hc, x_flat.shape[1] // hc)).sum(1).to(dtype)


@contextlib.contextmanager
def rig(rank, size):
    par = fixture.parallel_of(attn_dp=1, attn_tp=1, pp_size=size, pp_rank=rank)
    none = types.SimpleNamespace(is_none=lambda: True, is_mori=lambda: False)
    patches = [
        fixture.planning(par),
        get_parallel().override(tp_size=1, tp_rank=0, pp_size=size, pp_rank=rank, attn_dp_size=1, attn_tp_size=1, attn_tp_rank=0,
                                attn_cp_size=1, attn_cp_rank=0),
        mock.patch.object(RB, "get_moe_a2a_backend", lambda: none),
        mock.patch.object(moe_transport, "get_moe_a2a_backend", lambda: none),
        mock.patch.object(RB, "get_parallel", lambda: par),
        mock.patch.object(moe_transport, "get_parallel", lambda: par),
        mock.patch.object(M, "_is_cuda", True),
        mock.patch.object(M, "get_moe_a2a_backend", lambda: none),
        mock.patch.object(M, "is_cross_layer_mhc_fusion_enabled", lambda: False),
        mock.patch.object(M, "_is_fused_mhc_post_pre_enabled_xpu", lambda: False),
        mock.patch.object(M, "get_attn_backend", lambda: Backend()),
        mock.patch.object(M, "_low_ratio_sources_break", lambda *a, **k: False),
        mock.patch.object(M, "is_cp_active", lambda fb: False),
        mock.patch.object(M, "is_in_breakable_cuda_graph", lambda: False),
        mock.patch.object(M, "check_cuda_graph_backend", lambda *a, **k: True),
        mock.patch.object(RKERN, "hc_split_sinkhorn", hc_split_sinkhorn_torch),
        mock.patch.object(RKERN, "mhc_combine", torch_combine),
        mock.patch.object(RKERN, "hc_broadcast", lambda x, hc: x.unsqueeze(1).repeat(1, hc, 1)),
    ]
    for p in patches:
        p.__enter__()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.__exit__(None, None, None)


def batch(ids):
    return types.SimpleNamespace(
        input_ids=ids, forward_mode=ForwardMode.EXTEND, attn_cp_metadata=None, residual_stream=None,
        capture_hidden_mode=CaptureHiddenMode.NULL, return_logprob=False, is_extend_in_batch=True,
        extend_seq_lens_cpu=[ids.shape[0]], extend_logprob_start_lens_cpu=[0], num_token_non_padded=ids.shape[0],
        dp_padding_mode=None, can_run_tbo=False, tbo_children=None, global_forward_mode=ForwardMode.EXTEND,
        moe_num_token_non_padded=lambda: None, attn_tp_sequence_sharded=False)


def embeds(seed=0, n=T):
    return (torch.randn(n, W, generator=torch.Generator().manual_seed(seed)) * 2.0).bfloat16()


def run_ranks(size, emb, steer=True):
    """Each stage in turn, the previous stage's PP tensors as its input. Returns the head's output and the stages."""
    model_cls, causal = make_classes()
    out, info = None, []
    ids = torch.arange(emb.shape[0])
    for rank in range(size):
        with rig(rank, size):
            bb = model_cls(config(), rank, size)
            bb.to(torch.bfloat16)
            top = causal()
            nn.Module.__init__(top)
            top.model, top.config = bb, bb.config
            run = types.SimpleNamespace(
                model=top, is_draft_worker=False, tp_rank=0, tp_size=1, pp_rank=rank, pp_size=size,
                server_args=types.SimpleNamespace(enable_two_batch_overlap=False),
                model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="deepseek_v41"), dtype=torch.bfloat16))
            rec = install_steering(run, source="env") if steer else None
            # backbone.forward: the plugin's fired check wraps it and raises when a site ran 0 or 2 times
            res = bb.forward(ids, ids, batch(ids), emb if rank == 0 else None, None if rank == 0 else PPProxyTensors(out))
        wcheck = getattr(run, "_weightless_steer_companion_check", None) if steer else None
        info.append({"bb": bb, "rec": rec, "fired": dict(run._weightless_steer_check.fired) if steer else {},
                     "cfired": dict(wcheck.fired) if wcheck is not None else {}})
        out = dict(res.tensors) if isinstance(res, PPProxyTensors) else (res[0] if isinstance(res, tuple) else res)
    return out, info


@unittest.skipUnless(HAVE, f"SGLang's DeepSeek-V4.1 pipeline wiring is not importable here: {WHY}")
class ThreeStages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": PART, "WEIGHTLESS_STEER_HC_COPIES": "mean", "WEIGHTLESS_STEER_ALPHA": "0.75"})
        p.start()
        self.addCleanup(p.stop)
        path = GLP1017
        if path is None:
            layers = tuple(range(1, 40))
            m = good_meta(layers=layers, alpha="0.75")
            m["controlvector.model_hint"], m["glp.hook_point"] = "deepseek_v41", RES
            path = os.path.join(self.tmp.name, "v.gguf")
            write_gguf(path, m, unit_dirs(layers, W, 0))
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        self.emb = embeds()

    def one_rank(self, steer=True):
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_PP_LAYER_PARTITION")
            return run_ranks(1, self.emb, steer)

    def test_layers_per_rank_and_each_post_once(self):
        self.assertEqual([get_pp_indices(40, k, 3) for k in range(3)], [(0, 13), (13, 26), (26, 40)])
        _, info = run_ranks(3, self.emb)
        want = [list(range(1, 13)), list(range(13, 26)), list(range(26, 40))]
        self.assertEqual([x["rec"]["hooked_layers"] for x in info], want)
        for k, x in enumerate(info):
            self.assertEqual(sorted(x["fired"]), want[k])
            self.assertEqual(set(x["fired"].values()), {1})
            for i in range(x["bb"].start_layer, x["bb"].end_layer):
                self.assertNotIn("post", vars(x["bb"].layers[i].attn_boundary.declaration.update))

    def test_three_ranks_equal_one_rank_and_alpha_zero_is_stock(self):
        stock3, _ = run_ranks(3, self.emb, steer=False)
        stock1, _ = self.one_rank(steer=False)
        self.assertTrue(same(stock3, stock1))
        steered3, _ = run_ranks(3, self.emb)
        steered1, _ = self.one_rank()
        self.assertTrue(same(steered3, steered1))
        self.assertFalse(same(steered3, stock3))
        os.environ["WEIGHTLESS_STEER_ALPHA"] = "0"
        zero, _ = run_ranks(3, self.emb)
        self.assertTrue(same(zero, stock3))

    def test_every_post_is_edited_before_the_engrams_and_the_sources(self):
        records, wr = {}, []
        orig_post, orig_wr = RB.HcUpdate.post, RM.HcState.with_residual

        def spy_post(upd, output, state, *, next=None):
            st = orig_post(upd, output, state, next=next)
            if not upd.attention:
                records.setdefault(upd.layer.layer_id, []).append(st.residual.clone())
            return st

        def spy_wr(st, residual):
            wr.append((st.residual.clone(), residual.clone()))
            return orig_wr(st, residual)

        with mock.patch.object(RB.HcUpdate, "post", spy_post), mock.patch.object(RM.HcState, "with_residual", spy_wr):
            _, info = run_ranks(3, self.emb)
        bb = {i: x["bb"] for x in info for i in range(x["bb"].start_layer, x["bb"].end_layer)}
        edited = {}
        for i, rec in records.items():
            if i >= 1:
                edited[i] = next(new for old, new in wr if same(old, rec[0]))
        self.assertEqual(sorted(edited), list(range(1, 40)))
        for i, new in edited.items():
            self.assertTrue(same(new, A.hc_mean_rows(records[i][0], bb[i]._steer_stack[i, 0], bb[i]._steer_alpha)), i)
        self.assertTrue(same(bb[14].layers[14].engram.seen[-1], edited[13]))
        self.assertTrue(same(bb[1].layers[1].engram.seen[-1], records[0][0]))
        replays = 0
        for x in info:
            for lid in x["bb"].pp_source_plan.replayed:
                replays += 1
                self.assertTrue(same(x["bb"].pp_sources[str(lid)].seen[-1], bb[lid].layers[lid].self_attn.seen[-1]), lid)
        self.assertGreater(replays, 0)

    def test_an_empty_forward_passes_the_sites_once(self):
        out, info = run_ranks(3, self.emb[:0])
        self.assertEqual(out.shape[0], 0)
        for x in info:
            self.assertEqual(set(x["fired"].values()), {1})



@unittest.skipUnless(HAVE, f"SGLang's DeepSeek-V4.1 pipeline wiring is not importable here: {WHY}")
class ThreeStagesFold(ThreeStages):
    """A GLP spec_version 3 fold-site file (GLP-27, rows 11..37, alpha 3.5): the same folds plus the companion writer on
    the attention write, the layer at position k with stack row k + 1."""

    def setUp(self):
        super().setUp()
        os.environ["WEIGHTLESS_STEER_ALPHA"] = "3.5"
        path = GLP27
        if path is None:
            layers = tuple(range(11, 38))
            m = good_meta(layers=layers, alpha="3.5")
            m["controlvector.model_hint"], m["glp.hook_point"], m["glp.derived_at"] = "deepseek_v41", FOLD, FOLD
            m["glp.spec_version"] = "3"
            path = os.path.join(self.tmp.name, "f.gguf")
            write_gguf(path, m, unit_dirs(layers, W, 0))
        os.environ["WEIGHTLESS_STEER_PATH"] = path

    def test_layers_per_rank_and_each_post_once(self):
        _, info = run_ranks(3, self.emb)
        folds = [[11, 12], list(range(13, 26)), list(range(26, 38))]
        writers = [[10, 11, 12], list(range(13, 26)), list(range(26, 37))]
        for k, x in enumerate(info):
            self.assertEqual(x["rec"]["hooked_layers"], folds[k])
            self.assertEqual(x["rec"]["companion"]["layers"], writers[k])
            self.assertEqual(sorted(x["fired"]), folds[k])
            self.assertEqual(sorted(x["cfired"]), [i + 1 for i in writers[k]])
            self.assertEqual(set(x["fired"].values()) | set(x["cfired"].values()), {1})

    def test_every_post_is_edited_before_the_engrams_and_the_sources(self):
        """The writer half on the real post: the attention write it receives is the stub's write edited with row k + 1."""
        got = {}
        orig_post = RB.HcUpdate.post

        def spy_post(upd, output, state, *, next=None):
            if upd.attention:
                got[upd.layer.layer_id] = output.clone()
            return orig_post(upd, output, state, next=next)

        with mock.patch.object(RB.HcUpdate, "post", spy_post):
            _, info = run_ranks(3, self.emb)
        bb = {i: x["bb"] for x in info for i in range(x["bb"].start_layer, x["bb"].end_layer)}
        for k in range(40):
            attn = bb[k].layers[k].self_attn
            raw = torch.tanh(attn.seen[-1].float() * attn.w).to(attn.seen[-1].dtype)
            if 10 <= k <= 36:
                want = steer_rows(raw, None, bb[k]._steer_stack[k + 1, 0], bb[k]._steer_alpha)
                self.assertTrue(same(got[k], want), k)
                self.assertFalse(same(got[k], raw), k)
            else:
                self.assertTrue(same(got[k], raw), k)

    def test_an_empty_forward_passes_the_sites_once(self):
        out, info = run_ranks(3, self.emb[:0])
        self.assertEqual(out.shape[0], 0)
        for x in info:
            self.assertEqual(set(x["fired"].values()) | set(x["cfired"].values()), {1})


if __name__ == "__main__":
    unittest.main()
