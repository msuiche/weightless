"""The DeepSeek-V4-Flash row (``DeepseekV4ForCausalLM``, model_type
deepseek_v4), CPU only.

The site is the FFN write before the mHC fold: ``output[0]`` of each
steered ``DeepseekV4DecoderLayer`` in fused mode, edited with r=None, the
other three slots passed through. The next layer (or, after the last
layer, the model loop) folds the edited write into the streams.

- Fakes shaped like SGLang's classes (the first section) and the row on
  them: install, the edit, the pipeline, refusals, diagnostics.
- With WEIGHTLESS_TEST_GLP_DIR set, the published files are installed at
  the real width (4096) on a 43-layer fake: the 0731 GLP-29 file and the
  Vision-Exp -ffn file install; the GLP-42 residual file, the plain
  Vision-Exp file and the DeepSeek-V4.1 file are refused with their named
  errors. With WEIGHTLESS_TEST_DSV41_CONFIG set to the DeepSeek-V4.1
  config.json, the V4.1 refusal is also proven on the config SGLang builds
  from it.
- ``RealLayers``: SGLang's real ``DeepseekV4DecoderLayer.forward`` in its
  fused mHC mode, the real ``hc_pre`` / ``hc_post`` /
  ``_run_moe_ffn_dp_sync``, the real ``DeepseekV4Model.forward`` loop with
  its last-layer ``hc_post`` and ``hc_head``, and ``PPProxyTensors``, on a
  small width. Built with ``__new__``, so no weights, no GPU and no
  distributed setup. Stubs: the attention, the MoE and the norms. The
  kernel entry points that have no CPU build are given SGLang's own torch
  paths (``_hc_split_sinkhorn_torch``, ``hc_head_torch``) or the one-line
  torch equivalent (``hc_combine``); the runtime context lookups answer
  "one rank, no DP, no CP, no TBO". The cross-layer fused kernel is absent
  on CPU, so the fused mode runs the layer's own fallback (hc_post then
  hc_pre) inside the fused branch, and the layer returns the fused-mode
  4-tuple. Skips when this interpreter cannot import SGLang's DeepSeek-V4
  model.
- ``Structure``: a parse of ``models/deepseek_v4.py`` in every SGLang tree
  (the installed one and every package folder in
  WEIGHTLESS_TEST_SGLANG_TREES) that pins what the row relies on.
"""
import ast
import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import glpfiles  # noqa: E402  (sets up the import paths)
from glpfiles import good_meta, write_gguf  # noqa: E402

from test_archs import _trees, _parse, _cls, _fn, _src, _calls, _flat, unit_dirs  # noqa: E402

from weightless_sglang.archs import ARCH, PREFLIGHT, SPECIAL  # noqa: E402
from weightless_sglang.archs.dsv4 import Dsv4SiteError  # noqa: E402
from weightless_sglang.core import layer_verdict  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402


# ---------------------------------------------------------------- fakes shaped like SGLang's classes
# They follow models/deepseek_v4.py call for call: the backbone at .model; a
# full-length layers list with stand-ins outside this rank's range; the model
# widens the embeddings to [T, hc_mult, hidden] on the first rank, loops
# hidden_states, prev_residual, prev_post, prev_comb = layer(...) with keyword
# arguments, runs the last layer's hc_post after the loop in fused mode and
# hands the flattened stream to the next pipeline rank. The layer in fused
# mode folds the previous layer's pending FFN write first (hc_post), runs
# hc_pre / attention / hc_post / hc_pre / MLP and returns (ffn_out [T, H],
# residual [T, n, H], post [T, n], comb [T, n, n]); in unfused mode it folds
# its own FFN write and returns (stream [T, n, H], None, None, None). hc_post
# is SGLang's torch formula; hc_pre is a small torch stand-in.


def _vec(gen, n, lo, hi):
    return lo + (hi - lo) * torch.rand(n, generator=gen)


class _DecoderLayer(nn.Module):
    """Weights depend only on the layer id, so pipeline ranks built one by
    one hold the same model."""

    def __init__(self, cfg, layer_id, fused=True):
        super().__init__()
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.layer_id = layer_id
        self.hc_mult = n = cfg.hc_mult
        self.use_fused_mhc_post_pre = fused
        self.hc_pre_from_prev_sublayer = bool(getattr(cfg, "hc_pre_from_prev_sublayer", False))
        if self.hc_pre_from_prev_sublayer:
            self.use_fused_mhc_post_pre = False  # as SGLang's __init__ does
        H = cfg.hidden_size
        g = torch.Generator().manual_seed(2000 + layer_id)
        self.w_attn = nn.Parameter(_vec(g, H, 0.5, 1.5), requires_grad=False)
        self.w_mlp = nn.Parameter(_vec(g, H, -1.0, 1.0), requires_grad=False)
        self.pre_attn = nn.Parameter(_vec(g, n, -1.0, 1.0), requires_grad=False)
        self.pre_ffn = nn.Parameter(_vec(g, n, -1.0, 1.0), requires_grad=False)
        self.post_w = nn.Parameter(_vec(g, n, -1.0, 1.0), requires_grad=False)
        mix = torch.eye(n) * 0.7 + 0.3 / n
        self.register_buffer("res_mix", mix[torch.randperm(n, generator=g)], persistent=False)
        self.hc_post_calls = []  # spy: the x each hc_post folds

    def hc_pre(self, x, pre):
        T, n = x.shape[0], self.hc_mult
        w = torch.softmax(pre.float(), 0)
        y = (w.view(1, n, 1) * x.float()).sum(1).to(x.dtype)
        post = (2.0 * torch.sigmoid(self.post_w.float())).expand(T, n)
        comb = self.res_mix.float().expand(T, n, n)
        return y, post, comb

    def hc_post(self, x, residual, post, comb):
        """SGLang's torch hc_post (deepseek_v4.py, hc_post_torch_impl)."""
        self.hc_post_calls.append(x)
        return (post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)).type_as(x)

    def forward(self, positions, hidden_states, input_ids, forward_batch, input_ids_global,
                prev_residual=None, prev_post=None, prev_comb=None):
        use_fused = self.use_fused_mhc_post_pre
        if prev_residual is not None and use_fused:
            hidden_states = self.hc_post(hidden_states, prev_residual, prev_post, prev_comb)
        residual = hidden_states
        h, post, comb = self.hc_pre(hidden_states, self.pre_attn)
        h = torch.tanh(h * self.w_attn)
        hidden_states = self.hc_post(h, residual, post, comb)
        residual = hidden_states
        h, post, comb = self.hc_pre(hidden_states, self.pre_ffn)
        h = h * self.w_mlp  # the MoE output: the FFN write
        if not use_fused:
            return self.hc_post(h, residual, post, comb), None, None, None
        return h, residual, post, comb


DeepseekV4DecoderLayer = type("DeepseekV4DecoderLayer", (_DecoderLayer,), {})
OtherLayer = type("DeepseekV2DecoderLayer", (_DecoderLayer,), {})
PPMissingLayer = type("PPMissingLayer", (nn.Identity,), {})


class DeepseekV4Model(nn.Module):
    def __init__(self, cfg, start, end, fused=True, layer_cls=DeepseekV4DecoderLayer):
        super().__init__()
        self.config = cfg
        self.hidden_size = cfg.hidden_size
        self.hc_mult = cfg.hc_mult
        L = cfg.num_hidden_layers
        self.layers = nn.ModuleList(layer_cls(cfg, i, fused) if start <= i < end
                                    else PPMissingLayer() for i in range(L))
        self.start_layer, self.end_layer = start, end
        self.hc_pre_from_prev_sublayer = bool(getattr(cfg, "hc_pre_from_prev_sublayer", False))
        self.use_fused_mhc_post_pre = fused
        self.layers_run = []  # spy: layer ids called by the loop

    def forward(self, input_embeds=None, pp_hidden_states=None, bypass=False):
        if self.start_layer == 0:
            hidden_states = input_embeds.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            hidden_states = pp_hidden_states.view(pp_hidden_states.shape[0], self.hc_mult,
                                                  self.hidden_size)
        use_fused = self.use_fused_mhc_post_pre
        prev_residual = prev_post = prev_comb = None
        last_layer = None
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            last_layer = layer
            self.layers_run.append(i)
            call = layer.forward if bypass else layer  # bypass: a path that skips hooks
            hidden_states, prev_residual, prev_post, prev_comb = call(
                positions=None, hidden_states=hidden_states, forward_batch=None,
                input_ids=None, input_ids_global=None, prev_residual=prev_residual,
                prev_post=prev_post, prev_comb=prev_comb)
        if use_fused and last_layer is not None:
            hidden_states = last_layer.hc_post(hidden_states, prev_residual, prev_post, prev_comb)
        if self.end_layer != self.config.num_hidden_layers:
            return {"hidden_states": hidden_states.flatten(1)}  # PPProxyTensors
        return hidden_states.mean(1)  # stand-in for hc_head + norm


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone


def config(hidden=8, hc_mult=4, num_layers=43, model_type="deepseek_v4", v41=False):
    return types.SimpleNamespace(hidden_size=hidden, hc_mult=hc_mult, num_hidden_layers=num_layers,
                                 model_type=model_type, hc_pre_from_prev_sublayer=v41)


def stage_bounds(partition, rank):
    start = sum(partition[:rank])
    return start, start + partition[rank]


def runner(partition=(43,), rank=0, hidden=8, hc_mult=4, fused=True, dtype=torch.float32,
           cfg=None, hf_model_type="deepseek_v4", layer_cls=DeepseekV4DecoderLayer, tp_size=1):
    cfg = cfg if cfg is not None else config(hidden, hc_mult, sum(partition))
    start, end = stage_bounds(partition, rank)
    bb = DeepseekV4Model(cfg, start, end, fused, layer_cls).to(dtype)
    return types.SimpleNamespace(
        model=DeepseekV4ForCausalLM(bb), is_draft_worker=False, tp_rank=0, tp_size=tp_size,
        pp_rank=rank, pp_size=len(partition),
        model_config=types.SimpleNamespace(
            hf_config=types.SimpleNamespace(model_type=hf_model_type), dtype=dtype))


def pipeline(runners, x):
    out = None
    for n, run in enumerate(runners):
        bb = run.model.model
        out = bb(input_embeds=x) if n == 0 else bb(pp_hidden_states=out["hidden_states"])
    return out


# ---------------------------------------------------------------- the row on the fakes


ROW = "DeepseekV4ForCausalLM"
FFN = "ffn_out_pre_residual"
RES = "residual_stream_post_layer"
GLP29_LAYERS = tuple(range(10, 39))  # GLP-29: layers 10-38 of 43
PARTITIONS = {  # possible pipeline splits of 43 layers, and the GLP-29 layers per rank
    (22, 21): [list(range(10, 22)), list(range(22, 39))],
    (15, 14, 14): [list(range(10, 15)), list(range(15, 29)), list(range(29, 39))],
}

# huggingface.co/msuiche/DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29 and siblings
GLP29 = glpfiles.find_glp("DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4.gguf")
GLP42 = glpfiles.find_glp("DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-42-residual-L1-42-a1.5.gguf")
VIS_FFN = glpfiles.find_glp("DeepSeek-V4-Flash-Vision-Exp-abliterated-cyber-GLP-29-L10-38-a1.0-ffn.gguf")
VIS_RES = glpfiles.find_glp("DeepSeek-V4-Flash-Vision-Exp-abliterated-cyber-GLP-29-L10-38-a1.0.gguf")
DSV41 = glpfiles.find_glp("glp.deepseek-v41-flash-GLP-39-L1-39-a0.5.gguf")


class Dsv4Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)

    def write(self, layers=GLP29_LAYERS, width=8, alpha="4.0", hint="deepseek_v4", hook=FFN,
              name="v.gguf", **extra):
        m = good_meta(layers=layers, alpha=alpha)
        m["controlvector.model_hint"] = hint
        m["glp.hook_point"] = hook
        m.update(extra)
        path = os.path.join(self.tmp.name, name)
        write_gguf(path, m, unit_dirs(layers, width))
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        return path

    def install(self, run):
        with redirect_stderr(io.StringIO()) as err:
            rec = install_steering(run, source="env")
        self.log = err.getvalue()
        return rec

    @staticmethod
    def hooked(run):
        bb = run.model.model
        return sorted(i for i in range(bb.start_layer, bb.end_layer)
                      if getattr(bb.layers[i], "_weightless_dsv4_hook", None) is not None)


class Row(unittest.TestCase):
    def test_row_is_declared(self):
        row = ARCH[ROW]
        self.assertEqual(row.backbone, ("model",))
        self.assertEqual(row.layers, frozenset({"DeepseekV4DecoderLayer"}))
        self.assertEqual(row.hooks, frozenset({FFN}))
        self.assertEqual((row.arity, row.hidden_index, row.residual_index), (4, 0, None))
        self.assertEqual(row.install, "special:dsv4_ffn")
        self.assertEqual((row.exec_id, row.tp, row.per_stream), ("layer", "ok", False))
        self.assertEqual(row.hint, frozenset({"deepseek_v4"}))
        self.assertEqual(row.width(config(4096)), 4096)  # the FFN write is hidden-wide
        self.assertIn("dsv4_ffn", SPECIAL)
        self.assertIn("dsv4_ffn", PREFLIGHT)
        # the GLM-5.3-Flash special row sits beside it
        self.assertEqual(ARCH["Glm5NextForConditionalGeneration"].install, "special:glm_mhc_combine")
        self.assertEqual(SPECIAL["glm_mhc_combine"][1], "mhc.mlp_combine")


class Install(Dsv4Base):
    def test_hooks_exactly_the_file_layers(self):
        self.write()
        run = runner()
        rec = self.install(run)
        self.assertEqual(rec["local_layer_ids"], list(GLP29_LAYERS))
        self.assertEqual(self.hooked(run), list(GLP29_LAYERS))
        bb = run.model.model
        self.assertEqual(tuple(bb._steer_stack.shape), (43, 1, 8))
        nz = [int(i) for i in (bb._steer_stack.abs().sum((1, 2)) > 0).nonzero().flatten()]
        self.assertEqual(nz, list(GLP29_LAYERS))
        self.assertEqual((rec["row"], rec["install"], rec["hook_point"]), (ROW, "special:dsv4_ffn", FFN))
        self.assertEqual(rec["site"], "ffn_out (layer output[0], pre-fold)")
        self.assertEqual((rec["width"], rec["hidden_size"], rec["num_layers"]), (8, 8, 43))
        self.assertEqual(rec["alpha"], 4.0)
        self.assertEqual(rec["kernel"], "torch")
        self.assertIn("hook=ffn_out_pre_residual", self.log)
        self.assertIn("layers=10..38 (29)", self.log)

    def test_edit_is_on_the_ffn_write_and_other_slots_pass_through(self):
        """At each steered layer: output[0]' = x - alpha (x.d) d (no residual),
        and residual/post/comb are the very objects the layer returned."""
        alpha = 4.0
        self.write(alpha=str(alpha))
        run = runner(dtype=torch.float64)
        bb = run.model.model
        raw = {}
        for i in range(43):
            def spy(mod, args, out, _i=i):
                raw[_i] = out
            bb.layers[i].register_forward_hook(spy)  # runs before the steering hook
        self.install(run)
        seen = {}
        for i in range(43):
            def after(mod, args, out, _i=i):
                seen[_i] = out
            bb.layers[i].register_forward_hook(after)  # runs after it
        stack = bb._steer_stack.double()
        for tokens in (5, 1):  # a prefill shape and a one-token (decode) shape
            with self.subTest(tokens=tokens):
                raw.clear()
                seen.clear()
                x = torch.randn(tokens, 8, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(1))
                bb(input_embeds=x)
                for i in range(43):
                    x0, r0, p0, c0 = raw[i]
                    x1, r1, p1, c1 = seen[i]
                    self.assertEqual(tuple(x0.shape), (tokens, 8))
                    self.assertTrue(r1 is r0 and p1 is p0 and c1 is c0, i)
                    if i in GLP29_LAYERS:
                        d = stack[i, 0]
                        want = x0 - alpha * (x0 @ d).unsqueeze(-1) * d
                        self.assertTrue(torch.allclose(x1, want, atol=1e-12), i)
                        self.assertTrue(torch.allclose(x1 @ d, (1 - alpha) * (x0 @ d), atol=1e-9), i)
                        self.assertFalse(torch.equal(x1, x0), i)
                    else:
                        self.assertTrue(x1 is x0, i)

    def test_next_layer_and_model_loop_fold_the_steered_write(self):
        """The next layer's first hc_post folds the edited write; for the last
        layer the model loop's hc_post after the loop does."""
        alpha = 4.0
        layers = tuple(range(1, 43))  # through the last layer (42)
        self.write(layers=layers, alpha=str(alpha))
        run = runner(dtype=torch.float64)
        bb = run.model.model
        raw = {}
        for i in range(43):
            bb.layers[i].register_forward_hook(lambda m, a, o, _i=i: raw.__setitem__(_i, o[0]))
        self.install(run)
        x = torch.randn(3, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
        y = bb(input_embeds=x)
        stack = bb._steer_stack.double()
        for i in layers:
            d = stack[i, 0]
            folded = (bb.layers[i + 1].hc_post_calls[0] if i < 42
                      else bb.layers[42].hc_post_calls[-1])  # the loop's fold of layer 42
            self.assertTrue(torch.allclose(folded @ d, (1 - alpha) * (raw[i] @ d), atol=1e-9), i)
            e = torch.randn(8, dtype=torch.float64, generator=torch.Generator().manual_seed(i))
            e = e - (e @ d) * d
            self.assertTrue(torch.allclose(folded @ e, raw[i] @ e, atol=1e-9), i)
        # layer 0 is not steered: layer 1 folds layer 0's write as it came out
        self.assertTrue(torch.equal(bb.layers[1].hc_post_calls[0], raw[0]))
        self.assertEqual(tuple(y.shape), (3, 8))

    def test_pipeline_equals_one_rank(self):
        self.write()
        x = torch.randn(4, 8, generator=torch.Generator().manual_seed(3))
        one = runner()
        self.install(one)
        ref = one.model.model(input_embeds=x)
        stock = runner().model.model(input_embeds=x)
        self.assertFalse(torch.allclose(ref, stock))
        for part, expect in PARTITIONS.items():
            with self.subTest(partition=part):
                runs = [runner(part, r) for r in range(len(part))]
                for r, run in enumerate(runs):
                    rec = self.install(run)
                    self.assertEqual(rec["local_layer_ids"], expect[r])
                    self.assertEqual(self.hooked(run), expect[r])
                    self.assertEqual(tuple(run.model.model._steer_stack.shape), (43, 1, 8))
                self.assertTrue(torch.equal(pipeline(runs, x), ref))

    def test_alpha_zero_is_bitwise_stock_bf16_at_4096(self):
        self.write(width=4096, alpha="0")
        x = torch.randn(3, 4096, generator=torch.Generator().manual_seed(4)).bfloat16()
        stock = runner(hidden=4096, dtype=torch.bfloat16).model.model(input_embeds=x)
        run = runner(hidden=4096, dtype=torch.bfloat16)
        rec = self.install(run)
        self.assertEqual((rec["alpha"], rec["width"]), (0.0, 4096))
        self.assertTrue(torch.equal(run.model.model(input_embeds=x), stock))
        os.environ["WEIGHTLESS_STEER_ALPHA"] = "4.0"
        run2 = runner(hidden=4096, dtype=torch.bfloat16)
        self.install(run2)
        self.assertFalse(torch.equal(run2.model.model(input_embeds=x), stock))

    def test_fired_check_catches_a_bypass(self):
        """A path that runs the layers without calling them through
        __call__ (as two-batch overlap does) skips the hooks: the first such
        forward fails."""
        self.write()
        run = runner()
        self.install(run)
        bb = run.model.model
        x = torch.randn(2, 8)
        bb(input_embeds=x)
        with self.assertRaisesRegex(RuntimeError, r"did not run in this forward \(layers \[10"):
            bb(input_embeds=x, bypass=True)

    def test_tp_is_allowed_and_partial_sum_marker_still_fails(self):
        self.write()
        run = runner(tp_size=2)
        rec = self.install(run)  # the site is linear: TP>1 installs
        self.assertEqual(rec["tp_size"], 2)
        layer = run.model.model.layers[10]
        orig = type(layer).forward

        def marked(self_, *a, **k):
            out = orig(self_, *a, **k)
            out[0]._sglang_needs_allreduce_fusion = True
            return out

        layer.forward = types.MethodType(marked, layer)
        with self.assertRaisesRegex(RuntimeError, "_sglang_needs_allreduce_fusion"):
            run.model.model(input_embeds=torch.randn(2, 8))

    def test_unfused_return_at_run_time_fails_closed(self):
        """If a layer ever returned the unfused form (stream, None, None,
        None) the first forward stops."""
        self.write()
        run = runner()
        self.install(run)
        bb = run.model.model
        bb.use_fused_mhc_post_pre = False
        for layer in bb.layers:
            layer.use_fused_mhc_post_pre = False
        with self.assertRaisesRegex(RuntimeError, "layer 10 returned the unfused form"):
            run.model.model(input_embeds=torch.randn(2, 8))

    def test_three_d_write_at_run_time_fails_closed(self):
        """A steered layer whose output[0] is [T, n, H] (the last axis still
        the row width, slots 1-3 live) stops the first forward instead of
        being edited as if it were the FFN write."""
        self.write()
        run = runner()
        self.install(run)
        layer = run.model.model.layers[10]
        real = type(layer).forward

        def forward(*a, **k):
            x, residual, post, comb = real(layer, *a, **k)
            return x.unsqueeze(1).expand(-1, layer.hc_mult, -1), residual, post, comb

        layer.forward = forward
        with self.assertRaisesRegex(RuntimeError, r"layer 10: output\[0\] has shape \(2, 4, 8\)"):
            run.model.model(input_embeds=torch.randn(2, 8))

    def test_wrong_arity_at_run_time_fails_closed(self):
        """A steered layer returning a 5-tuple stops the first forward with
        the site error, not an unpacking error somewhere else."""
        self.write()
        run = runner()
        self.install(run)
        layer = run.model.model.layers[10]
        real = type(layer).forward
        layer.forward = lambda *a, **k: (*real(layer, *a, **k), None)
        with self.assertRaisesRegex(Dsv4SiteError, "returned tuple of 5, expected the fused-mode"):
            run.model.model(input_embeds=torch.randn(2, 8))

    def test_manifest_and_undo(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = runner((22, 21), 1)
        self.install(run)
        m = json.load(open(os.path.join(self.tmp.name, f"weightless-manifest-{os.getpid()}.json")))
        self.assertEqual((m["row"], m["model_class"], m["install"]), (ROW, ROW, "special:dsv4_ffn"))
        self.assertEqual((m["hook_point"], m["width"], m["pp_rank"], m["pp_size"]), (FFN, 8, 1, 2))
        self.assertEqual(m["local_layer_ids"], list(range(22, 39)))
        self.assertEqual(m["model_types"], ["deepseek_v4"])
        self.assertEqual(m["kernel"], "torch")
        for undo in run._weightless_steer_handles:
            undo()
        self.assertEqual(self.hooked(run), [])
        self.assertFalse(any(l._forward_hooks for l in run.model.model.layers))

    def test_layer_map_diag(self):
        """WEIGHTLESS_STEER_DIAG=1: the alpha-aware layer map passes on every
        steered layer (the edit is exact, signed ratio 1 - alpha)."""
        alpha = 4.0
        self.write(alpha=str(alpha))
        ddir = os.path.join(self.tmp.name, "diag")
        os.environ.update(WEIGHTLESS_STEER_DIAG="1", WEIGHTLESS_STEER_DIAG_DIR=ddir)
        run = runner()
        self.install(run)
        run.model.model(input_embeds=torch.randn(6, 8, generator=torch.Generator().manual_seed(6)))
        f = json.load(open(os.path.join(ddir, f"weightless-diag-{os.getpid()}.json")))
        self.assertEqual(sorted(int(k) for k in f["layers"]), list(GLP29_LAYERS))
        for k, st in f["layers"].items():
            self.assertLess(st["sum_abs_err"], 1e-5 * st["sum_abs_pre"], k)
            self.assertAlmostEqual(st["sum_signed_post"] / st["sum_abs_pre"], 1 - alpha, places=5)
            for floor in ("norm", "elementwise"):
                v = layer_verdict(st, alpha, floor=floor)
                self.assertNotIn("fail", (v["edit"], v["sign"], v["probe"]), (k, v))


class Refusals(Dsv4Base):
    def test_unfused_mode_is_refused_at_install(self):
        self.write()
        with self.assertRaisesRegex(RuntimeError, "unfused mHC mode"):
            self.install(runner(fused=False))
        run = runner()
        run.model.model.layers[30].use_fused_mhc_post_pre = False
        with self.assertRaisesRegex(RuntimeError, r"layers \[30\].*unfused mHC mode"):
            self.install(run)
        self.assertEqual(self.hooked(run), [])  # nothing was hooked

    def test_dsv41_model_is_refused_before_anything_is_hooked(self):
        self.write()
        for name, mutate in (
                ("hf model_type", lambda run: setattr(run.model_config.hf_config, "model_type",
                                                       "deepseek_v41")),
                ("backbone model_type", lambda run: setattr(run.model.model.config, "model_type",
                                                             "deepseek_v41")),
                ("text model_type, no flag", lambda run: setattr(
                    run.model_config.hf_config, "model_type", "deepseek_v41_text")),
                ("hc_pre_from_prev_sublayer", lambda run: setattr(
                    run.model.model.config, "hc_pre_from_prev_sublayer", True)),
                ("backbone flag", lambda run: setattr(run.model.model, "hc_pre_from_prev_sublayer",
                                                      True))):
            with self.subTest(name):
                run = runner()
                mutate(run)
                with self.assertRaisesRegex(RuntimeError, "is DeepSeek-V4.1"):
                    self.install(run)
                self.assertEqual(self.hooked(run), [])
                self.assertFalse(hasattr(run.model.model, "_steer_stack"))

    def test_model_type_other_than_deepseek_v4_is_refused(self):
        """The row serves model_type deepseek_v4 only: another type that
        reuses the class, or no model_type at all, fails closed before
        anything is hooked."""
        self.write()

        def none_on(cfg):
            cfg.model_type = None

        for name, mutate in (
                ("hf deepseek_v42", lambda run: setattr(run.model_config.hf_config, "model_type",
                                                         "deepseek_v42")),
                ("backbone deepseek_v42", lambda run: setattr(run.model.model.config,
                                                               "model_type", "deepseek_v42")),
                ("both deepseek_v42", lambda run: (
                    setattr(run.model_config.hf_config, "model_type", "deepseek_v42"),
                    setattr(run.model.model.config, "model_type", "deepseek_v42"))),
                ("deepseek_v3", lambda run: setattr(run.model_config.hf_config, "model_type",
                                                     "deepseek_v3")),
                ("no model_type", lambda run: (none_on(run.model_config.hf_config),
                                               none_on(run.model.model.config)))):
            with self.subTest(name):
                run = runner()
                mutate(run)
                with self.assertRaisesRegex(RuntimeError,
                                            "serves model_type 'deepseek_v4' only.*Refusing"):
                    self.install(run)
                self.assertEqual(self.hooked(run), [])
                self.assertFalse(hasattr(run.model.model, "_steer_stack"))
        # one config without model_type is fine when the other says deepseek_v4
        run = runner()
        none_on(run.model.model.config)
        self.assertEqual(self.install(run)["local_layer_ids"], list(GLP29_LAYERS))

    def test_residual_site_files_are_refused_by_name(self):
        self.write(hook=RES, layers=tuple(range(1, 43)))
        with self.assertRaisesRegex(RuntimeError, "residual-site file.*GLP-42.*last-layer trap"):
            self.install(runner())
        self.write(hook=RES, **{"glp.base_model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"})
        with self.assertRaisesRegex(RuntimeError, "plain DeepSeek-V4-Flash-Vision-Exp GLP-29 file.*-ffn"):
            self.install(runner())

    def test_other_models_residual_file_is_not_named_a_deepseek_file(self):
        """A residual-site file of another model (a Qwen hint) is refused by
        the generic hook-point check, not called a DeepSeek-V4 file."""
        self.write(hint="qwen3_5", hook=RES)
        with self.assertRaisesRegex(RuntimeError, "glp.hook_point='residual_stream_post_layer', "
                                                  "which DeepseekV4ForCausalLM does not serve"):
            self.install(runner())
        self.write(hint="", hook=RES)  # no hint: treated as this row's own file
        with self.assertRaisesRegex(RuntimeError, "residual-site file"):
            self.install(runner())

    def test_dsv41_file_is_refused_by_name(self):
        self.write(hint="deepseek_v41", hook=RES, width=8, layers=tuple(range(1, 40)))
        with self.assertRaisesRegex(RuntimeError, "DeepSeek-V4.1 file"):
            self.install(runner())

    def test_boot_refusals(self):
        self.write()
        cases = (
            ("two-batch overlap", "two-batch overlap",
             lambda run: setattr(run, "server_args",
                                 types.SimpleNamespace(enable_two_batch_overlap=True))),
            ("layer id", "says layer_id=11",
             lambda run: setattr(run.model.model.layers[10], "layer_id", 11)),
            ("hidden size", "hidden_size 16 != row width 8",
             lambda run: setattr(run.model.model.layers[12], "hidden_size", 16)),
        )
        for name, pattern, mutate in cases:
            with self.subTest(name):
                run = runner()
                mutate(run)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self.install(run)

    def test_second_install_on_the_same_layers_is_refused(self):
        self.write()
        run = runner()
        self.install(run)
        run2 = types.SimpleNamespace(**vars(run))  # a new runner over the same model
        del run2._weightless_steer_installed
        with self.assertRaisesRegex(RuntimeError, "already steered"):
            self.install(run2)

    def test_wrong_layer_class(self):
        self.write()
        with self.assertRaisesRegex(RuntimeError, "not one of"):
            self.install(runner(layer_cls=OtherLayer))

    def test_file_refusals(self):
        for name, kw, pattern in (
                ("glm hint", dict(hint="glm5_next"), "model_hint"),
                ("looped file", {"glp.structure": "per-execution-step"}, "per-execution-step"),
                ("layer 43", dict(layers=(10, 43)), "43"),
                ("hc-widened file", dict(width=32), r"width 32 != 8")):
            with self.subTest(name):
                self.write(**kw)
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self.install(runner())

    def test_wrong_hook_env(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_HOOK"] = RES
        with self.assertRaisesRegex(RuntimeError, "WEIGHTLESS_STEER_HOOK"):
            self.install(runner())


@unittest.skipUnless(glpfiles.have(GLP29, GLP42, VIS_FFN, VIS_RES, DSV41),
                     "set WEIGHTLESS_TEST_GLP_DIR to a folder that holds the DeepSeek-V4 GLP files")
class RealFiles(Dsv4Base):
    """The published DeepSeek files through install_steering on a 43-layer
    fake at the real width (4096)."""

    def use(self, path):
        os.environ["WEIGHTLESS_STEER_PATH"] = path

    def test_glp29_installs_at_the_ffn_site(self):
        from weightless_steer.container import read_gguf_cvec
        from weightless_steer.core import SteeringCore
        self.use(GLP29)
        meta, _ = read_gguf_cvec(GLP29)
        self.assertEqual((meta["glp.hook_point"], meta["glp.derived_at"]), (FFN, RES))
        self.assertNotIn("glp.structure", meta)  # no structure: not refused for it
        core = SteeringCore.from_env(hook=FFN, num_layers=43, hidden_size=4096)
        self.assertEqual(core.alpha, 6.0)  # glp.alpha_default: the README's shipped dose
        x = torch.randn(2, 4096, generator=torch.Generator().manual_seed(1))
        one = runner(hidden=4096)
        rec = self.install(one)
        self.assertEqual(rec["local_layer_ids"], list(GLP29_LAYERS))
        self.assertEqual((rec["width"], rec["alpha"], rec["hook_point"]), (4096, 6.0, FFN))
        stack = one.model.model._steer_stack
        for i in GLP29_LAYERS:
            self.assertTrue(torch.equal(stack[i, 0], core.dirs[i]))
        ref = one.model.model(input_embeds=x)
        for part, expect in PARTITIONS.items():
            runs = [runner(part, r, hidden=4096) for r in range(len(part))]
            for r, run in enumerate(runs):
                self.assertEqual(self.install(run)["local_layer_ids"], expect[r])
            self.assertTrue(torch.equal(pipeline(runs, x), ref))
        print(f"  GLP-29: 29 layers 10..38 at width 4096, alpha 6.0, sha256 {rec['file_sha256'][:16]}")

    def test_vision_exp_ffn_file_installs(self):
        from weightless_steer.container import read_gguf_cvec
        self.use(VIS_FFN)
        meta, _ = read_gguf_cvec(VIS_FFN)
        self.assertIn("per-layer vectors", meta["glp.structure"])  # free text: not refused
        rec = self.install(runner(hidden=4096))
        self.assertEqual(rec["local_layer_ids"], list(GLP29_LAYERS))
        self.assertEqual((rec["width"], rec["alpha"]), (4096, 1.0))

    def test_residual_files_are_refused(self):
        from weightless_steer.container import read_gguf_cvec
        self.use(GLP42)
        with self.assertRaisesRegex(RuntimeError, "residual-site file.*GLP-42"):
            self.install(runner(hidden=4096))
        self.use(VIS_RES)
        with self.assertRaisesRegex(RuntimeError, "plain DeepSeek-V4-Flash-Vision-Exp"):
            self.install(runner(hidden=4096))
        # the plain file holds the same tensors as the -ffn file
        a, ta = read_gguf_cvec(VIS_RES)
        b, tb = read_gguf_cvec(VIS_FFN)
        self.assertEqual(a["glp.content_sha256"], b["glp.content_sha256"])
        self.assertEqual(sorted(ta), sorted(tb))
        for k in ta:
            self.assertTrue(np.array_equal(np.asarray(ta[k][0] if isinstance(ta[k], tuple) else ta[k]),
                                           np.asarray(tb[k][0] if isinstance(tb[k], tuple) else tb[k])))

    def test_dsv41_file_is_refused(self):
        self.use(DSV41)
        cfg = config(5120, 4, 40, model_type="deepseek_v41", v41=True)
        with self.assertRaisesRegex(RuntimeError, "is DeepSeek-V4.1"):
            self.install(runner(cfg=cfg, hf_model_type="deepseek_v41"))
        # on a DeepSeek-V4 model the V4.1 file is refused by its hint
        with self.assertRaisesRegex(RuntimeError, "DeepSeek-V4.1 file"):
            self.install(runner(hidden=5120))


def _dsv41_config():
    p = os.environ.get("WEIGHTLESS_TEST_DSV41_CONFIG", "").strip()
    if p and os.path.isdir(p):
        p = os.path.join(p, "config.json")
    return p if p and os.path.isfile(p) else None


@unittest.skipUnless(_dsv41_config(), "set WEIGHTLESS_TEST_DSV41_CONFIG to the DeepSeek-V4.1 config.json")
class Dsv41RealConfig(Dsv4Base):
    """DeepSeek-V4.1 on its real config.json: SGLang's own config class
    turns it into DeepseekV4ForCausalLM with model_type deepseek_v41 and
    hc_pre_from_prev_sublayer, and the row refuses it before anything is
    hooked."""

    def test_refused_on_the_real_config(self):
        raw = json.load(open(_dsv41_config()))
        self.assertEqual(raw["architectures"], ["DeepseekV41ForCausalLM"])
        try:
            from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
        except Exception as e:  # no SGLang here
            DeepseekV41Config = None
            why = f"{type(e).__name__}: {e}"
        if DeepseekV41Config is not None:
            cfg = DeepseekV41Config(**raw)
            self.assertEqual(cfg.architectures, [ROW])  # SGLang serves it through the V4 row
            self.assertEqual(cfg.model_type, "deepseek_v41")
            self.assertTrue(cfg.hc_pre_from_prev_sublayer)
            layers, hidden = int(cfg.num_hidden_layers), int(cfg.hidden_size)
        else:
            print(f"  (SGLang config class not importable: {why}; using the raw text_config)")
            t = raw["text_config"]
            cfg = types.SimpleNamespace(model_type="deepseek_v41", hc_pre_from_prev_sublayer=True,
                                        hidden_size=t["hidden_size"], hc_mult=t["hc_mult"],
                                        num_hidden_layers=t["num_hidden_layers"])
            layers, hidden = cfg.num_hidden_layers, cfg.hidden_size
        self.assertEqual((layers, hidden), (40, 5120))
        # the real file when there, else a synthetic V4.1-shaped one
        if glpfiles.have(DSV41):
            os.environ["WEIGHTLESS_STEER_PATH"] = DSV41
        else:
            self.write(layers=tuple(range(1, 40)), width=5120, hint="deepseek_v41", hook=RES)
        bb_cfg = types.SimpleNamespace(hidden_size=hidden, hc_mult=int(cfg.hc_mult),
                                       num_hidden_layers=layers, model_type=cfg.model_type,
                                       hc_pre_from_prev_sublayer=cfg.hc_pre_from_prev_sublayer)
        run = runner(cfg=bb_cfg, hf_model_type=cfg.model_type)
        run.model_config.hf_config = cfg
        with self.assertRaisesRegex(RuntimeError, "is DeepSeek-V4.1"):
            self.install(run)
        self.assertFalse(hasattr(run.model.model, "_steer_stack"))


# ---------------------------------------------------------------- SGLang's real classes

os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "0")
os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_POST", "0")

try:
    import sglang.srt.models.deepseek_v4 as D
    import sglang.kernels.ops.layernorm.mhc as MHC
    import sglang.kernels.ops.layernorm.mhc_head as MHC_HEAD
    import sglang.srt.layers.moe as MOE
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    HAVE_SGLANG = True
except Exception as e:  # no SGLang, or a build without the DeepSeek-V4 model
    HAVE_SGLANG, WHY = False, f"{type(e).__name__}: {e}"

H, N, NL = 8, 4, 12
STEERED = tuple(range(1, NL))  # every layer but 0, the last one (11) included


def _hc_combine_torch(x_flat, pre, hc, out_dtype):
    """y[m, h] = sum_k pre[m, k] x_flat[m, k H + h] (the Triton kernel's formula)."""
    m = x_flat.shape[0]
    return (pre.float().unsqueeze(-1) * x_flat.float().view(m, hc, -1)).sum(1).to(out_dtype)


def _fused_hc_head_torch(x, hc_fn, hc_scale, hc_base, *, norm_eps, hc_eps):
    return D.hc_head_torch(x, hc_fn, hc_scale, hc_base, norm_eps=norm_eps, hc_eps=hc_eps)


class _Parallel(types.SimpleNamespace):
    pass


def _parallel():
    return _Parallel(attn_dp_size=1, attn_tp_size=1, attn_tp_rank=0, tp_size=1, tp_rank=0,
                     tp_group=None, attn_cp_size=1, dwdp_size=1, moe_ep_size=1, moe_tp_size=1,
                     pp_group=types.SimpleNamespace(world_size=1))


@contextlib.contextmanager
def cpu_runtime():
    """One rank, no DP/CP/TBO, torch paths for the kernels with no CPU build."""
    ops = D.MhcOps(hc_split_sinkhorn=MHC._hc_split_sinkhorn_torch, mhc_fused_post_pre=None,
                   npu_hc_pre=None, mhc_pre=None, mhc_post=None, fused_hc_head=None)
    fwd = types.SimpleNamespace(scoped=lambda **k: contextlib.nullcontext(),
                                defer_moe_finalize=False, fuse_mlp_allreduce=False,
                                mlp_reduce_scatter=False)
    with contextlib.ExitStack() as st:
        for obj, name, val in (
                (D, "_get_mhc_ops", lambda: ops),
                (MHC, "hc_combine", _hc_combine_torch),
                (MHC_HEAD, "fused_hc_head", _fused_hc_head_torch),
                (D, "get_parallel", _parallel),
                (D, "use_symmetric_memory", lambda *a, **k: contextlib.nullcontext()),
                (D, "get_moe_a2a_backend", lambda: types.SimpleNamespace(is_none=lambda: True)),
                (D, "get_forward", lambda: fwd),
                (D, "get_attn_backend", lambda: types.SimpleNamespace()),
                (D, "check_cuda_graph_backend", lambda *a, **k: True),
                (D, "dsa_use_prefill_cp", lambda fb: False),
                (MOE, "is_tbo_enabled", lambda: False)):
            st.enter_context(mock.patch.object(obj, name, val))
        yield


def _cfg():
    return types.SimpleNamespace(hidden_size=H, hc_mult=N, num_hidden_layers=NL,
                                 model_type="deepseek_v4", hc_pre_from_prev_sublayer=False,
                                 rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20)


class _Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(H), requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, x, residual=None):
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight).to(x.dtype)
        return y if residual is None else (y, residual)


class _Attn(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(0.5, 1.5, H) * (1 + 0.02 * i), requires_grad=False)

    @contextlib.contextmanager
    def maybe_use_decode_attn_tp(self, forward_batch):
        yield

    def forward(self, x, positions=None, forward_batch=None, x_quant=None):
        return torch.tanh(x * self.w)


class _MoE(nn.Module):
    """Stands for DeepseekV2MoE: returns the (already reduced) FFN write."""

    def __init__(self, i):
        super().__init__()
        self.w = nn.Parameter(torch.linspace(-1, 1, H) * (1 + 0.01 * i), requires_grad=False)
        self.inputs = []

    def forward(self, x, forward_batch=None, input_ids=None, input_ids_global=None,
                skip_shared_experts=False):
        self.inputs.append(x)
        return x * self.w


def real_layer(i, cfg, dtype, fused=True):
    g = torch.Generator().manual_seed(300 + i)
    L = D.DeepseekV4DecoderLayer.__new__(D.DeepseekV4DecoderLayer)
    nn.Module.__init__(L)
    L.hc_stats_stream, L.config, L.hidden_size, L.layer_id = None, cfg, H, i
    L.self_attn, L.mlp = _Attn(i).to(dtype), _MoE(i).to(dtype)
    L.input_layernorm, L.post_attention_layernorm = _Norm().to(dtype), _Norm().to(dtype)
    L.hc_mult, L.hc_sinkhorn_iters, L.hc_eps, L.rms_norm_eps = N, 20, 1e-6, 1e-6
    params = D.make_hc_mixing_params(N, H)  # the real shapes, filled below
    names = ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale",
             "hc_ffn_scale")
    for name, p in zip(names, params):
        with torch.no_grad():
            if name.endswith("_fn"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            elif name.endswith("_base"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
            else:
                p.fill_(1.0)
        p.requires_grad_(False)
        setattr(L, name, p)
    L.dsa_enable_prefill_cp = False
    L.use_fused_mhc_post_pre = fused  # SGLang's default where the TileLang/aiter kernels are there
    L.hc_pre_from_prev_sublayer = False
    L.engram = None
    L._input_layernorm_weight_bf16 = None
    L._post_attention_layernorm_weight_bf16 = None
    return L


def real_runner(partition=(NL,), rank=0, dtype=torch.float32, fused=True):
    cfg = _cfg()
    start = sum(partition[:rank])
    end = start + partition[rank]
    M = D.DeepseekV4Model.__new__(D.DeepseekV4Model)
    nn.Module.__init__(M)
    M.config = cfg
    M.pp_group = types.SimpleNamespace(is_first_rank=start == 0, is_last_rank=end == NL,
                                       world_size=len(partition), rank_in_group=rank)
    M.hidden_size, M.embed_tokens, M.rms_norm_eps = H, nn.Identity(), 1e-6
    M.alt_streams = M.moe_routed_quant_stream = M.hc_stats_stream = None
    M.engram_layout = M.engram_hasher = None
    M.layers = nn.ModuleList(real_layer(i, cfg, dtype, fused) if start <= i < end
                             else nn.Identity() for i in range(NL))
    M.start_layer, M.end_layer = start, end
    M.norm = _Norm().to(dtype) if end == NL else nn.Identity()
    M.gemm_output_zero_allocator_size = 0
    M.hc_eps, M.hc_mult, M.norm_eps, M.hc_pre_from_prev_sublayer = 1e-6, N, 1e-6, False
    M.hc_head_fn = M.hc_head_base = M.hc_head_scale = None
    if end == NL:
        fn, base, scale = D.make_hc_head_params(N, H)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            fn.copy_(torch.randn(fn.shape, generator=g) * 0.05)
            base.copy_(torch.randn(base.shape, generator=g) * 0.1)
            scale.fill_(1.0)
        M.hc_head_fn, M.hc_head_base, M.hc_head_scale = (p.requires_grad_(False) for p in (fn, base, scale))
    M.use_fused_mhc_post_pre = fused
    M.dspark_layers_to_capture = None
    M.late_layer_start = None
    W = D.DeepseekV4ForCausalLM.__new__(D.DeepseekV4ForCausalLM)
    nn.Module.__init__(W)
    W.config, W.model = cfg, M
    return types.SimpleNamespace(
        model=W, is_draft_worker=False, tp_rank=0, tp_size=1, pp_rank=rank,
        pp_size=len(partition),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="deepseek_v4"),
                                           dtype=dtype))


def _fb():
    return types.SimpleNamespace(can_run_tbo=False, tbo_children=None, global_forward_mode=None,
                                 attn_cp_metadata=None, num_token_non_padded=None,
                                 forward_mode=types.SimpleNamespace(is_idle=lambda: False))


def run_model(run, x=None, proxy=None):
    T = (x if x is not None else proxy["hidden_states"]).shape[0]
    ids = torch.zeros(T, dtype=torch.long)
    with cpu_runtime():
        out = run.model.model(ids, torch.arange(T), _fb(), x, pp_proxy_tensors=proxy)
    return out


def real_pipeline(runs, x):
    out = None
    for n, run in enumerate(runs):
        out = run_model(run, x=x) if n == 0 else run_model(run, proxy=out)
    return out


@unittest.skipUnless(HAVE_SGLANG, "SGLang's DeepSeek-V4 model is not importable here")
class RealLayers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        self.write()

    def write(self, alpha="4.0", hook=FFN):
        m = good_meta(layers=STEERED, alpha=alpha)
        m["controlvector.model_hint"] = "deepseek_v4"
        m["glp.hook_point"] = hook
        rng = np.random.default_rng(0)
        t = {}
        for i in STEERED:
            v = rng.standard_normal(H)
            t[f"direction.{i}"] = ((v / np.linalg.norm(v)).astype(np.float32), 0)
        path = os.path.join(self.tmp.name, f"v{alpha}-{hook}.gguf")
        write_gguf(path, m, t)
        os.environ["WEIGHTLESS_STEER_PATH"] = path

    def install(self, run):
        with redirect_stderr(io.StringIO()):
            return install_steering(run, source="env")

    def test_real_classes_install(self):
        run = real_runner()
        rec = self.install(run)
        self.assertEqual(rec["row"], "DeepseekV4ForCausalLM")
        self.assertEqual(rec["local_layer_ids"], list(STEERED))
        self.assertEqual((rec["width"], rec["hook_point"]), (H, FFN))
        bb = run.model.model
        hooked = [i for i in range(NL) if getattr(bb.layers[i], "_weightless_dsv4_hook", None)]
        self.assertEqual(hooked, list(STEERED))

    def test_real_layer_returns_the_fused_four_tuple(self):
        """The real forward in fused mode returns (ffn_out [T,H], residual
        [T,n,H], post [T,n], comb [T,n,n]), and ffn_out is the MoE output."""
        run = real_runner()
        bb = run.model.model
        outs = {}
        for i in range(NL):
            bb.layers[i].register_forward_hook(lambda m, a, o, _i=i: outs.__setitem__(_i, o))
        x = torch.randn(3, H, generator=torch.Generator().manual_seed(1))
        run_model(run, x=x)
        for i in range(NL):
            ffn, res, post, comb = outs[i]
            self.assertEqual(tuple(ffn.shape), (3, H))
            self.assertEqual(tuple(res.shape), (3, N, H))
            self.assertEqual(tuple(post.shape), (3, N))
            self.assertEqual(tuple(comb.shape), (3, N, N))

    def test_steered_write_reaches_the_next_layer_and_the_last_fold(self):
        """Spy on each layer's output[0] before the edit and on the real
        hc_post: the next layer (or, after the last layer, the model loop)
        folds x' = x - alpha (x.d) d. Checked by recomputing the fold of the
        edited write with the real hc_post and comparing with the stream the
        next layer's attention pre-mix received."""
        alpha = 4.0
        run = real_runner(dtype=torch.float64)
        bb = run.model.model
        raw, folds = {}, {}
        for i in range(NL):
            L = bb.layers[i]
            L.register_forward_hook(lambda m, a, o, _i=i: raw.__setitem__(_i, o))
            orig = L.hc_post

            def spy(x, residual, post, comb, _i=i, _f=orig):
                out = _f(x, residual, post, comb)
                folds.setdefault(_i, []).append((x, out))
                return out

            L.hc_post = spy
        self.install(run)
        x = torch.randn(4, H, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
        y, pre_head = run_model(run, x=x)
        stack = bb._steer_stack.double()
        for i in STEERED:
            d = stack[i, 0]
            x0 = raw[i][0]  # the write before the edit (the spy runs before the steering hook)
            want = x0 - alpha * (x0 @ d).unsqueeze(-1) * d
            # the consumer's hc_post gets the edited write: layer i+1's first
            # fold, or for the last layer the model loop's fold (its last call)
            got = folds[i + 1][0][0] if i + 1 < NL else folds[i][-1][0]
            self.assertTrue(torch.allclose(got, want, atol=1e-12), i)
            self.assertTrue(torch.allclose(got @ d, (1 - alpha) * (x0 @ d), atol=1e-9), i)
        # the last layer: the stream before hc_head is the fold of the edited write
        last_fold = folds[NL - 1][-1][1]
        self.assertTrue(torch.equal(pre_head, last_fold.flatten(1)))
        # layer 0 is not steered: layer 1 folds layer 0's write as it came out
        self.assertTrue(torch.equal(folds[1][0][0], raw[0][0]))
        self.assertEqual(tuple(y.shape), (4, H))

    def test_steering_changes_the_output_and_alpha_zero_is_stock(self):
        for tokens in (3, 1):  # a prefill shape and a one-token (decode) shape
            with self.subTest(tokens=tokens):
                self.write(alpha="4.0")
                x = torch.randn(tokens, H, generator=torch.Generator().manual_seed(2)).bfloat16()
                stock = run_model(real_runner(dtype=torch.bfloat16), x=x)[0]
                self.write(alpha="0")
                run = real_runner(dtype=torch.bfloat16)
                self.assertEqual(self.install(run)["alpha"], 0.0)
                self.assertTrue(torch.equal(run_model(run, x=x)[0], stock))
                self.write(alpha="4.0")
                run2 = real_runner(dtype=torch.bfloat16)
                self.install(run2)
                self.assertFalse(torch.equal(run_model(run2, x=x)[0], stock))

    def test_pipeline_ranks_equal_one_rank(self):
        x = torch.randn(3, H, generator=torch.Generator().manual_seed(3))
        one = real_runner()
        self.install(one)
        ref = run_model(one, x=x)[0]
        for part, expect in (((6, 6), [list(range(1, 6)), list(range(6, 12))]),
                             ((4, 4, 4), [list(range(1, 4)), list(range(4, 8)), list(range(8, 12))])):
            with self.subTest(partition=part):
                runs = [real_runner(part, r) for r in range(len(part))]
                for r, run in enumerate(runs):
                    self.assertEqual(self.install(run)["local_layer_ids"], expect[r])
                mid = run_model(runs[0], x=x)
                self.assertIsInstance(mid, PPProxyTensors)
                self.assertEqual(tuple(mid["hidden_states"].shape), (3, N * H))
                self.assertTrue(torch.equal(real_pipeline(runs, x)[0], ref))

    def test_empty_batch(self):
        run = real_runner()
        self.install(run)
        y, _ = run_model(run, x=torch.empty(0, H))
        self.assertEqual(tuple(y.shape), (0, H))

    def test_unfused_mode_is_refused(self):
        """The real layer in unfused mode returns (stream, None, None, None);
        the row refuses it at install."""
        run = real_runner(fused=False)
        outs = {}
        run.model.model.layers[3].register_forward_hook(lambda m, a, o: outs.__setitem__(3, o))
        run_model(run, x=torch.randn(2, H))
        s, r, p, c = outs[3]
        self.assertEqual((tuple(s.shape), r, p, c), ((2, N, H), None, None, None))
        with self.assertRaisesRegex(RuntimeError, "unfused mHC mode"):
            self.install(real_runner(fused=False))

    def test_other_model_type_is_refused_on_the_real_classes(self):
        """The real classes under a model_type other than deepseek_v4 (a
        future type that reuses the class), or with none: refused before any
        hook, although the layers and the loop would run."""
        for name, mt in (("deepseek_v42", "deepseek_v42"), ("no model_type", None)):
            with self.subTest(name):
                run = real_runner()
                run.model_config.hf_config.model_type = mt
                run.model.model.config.model_type = mt  # the backbone's config (shared by W)
                run_model(run, x=torch.randn(2, H))  # the real loop runs as it is
                with self.assertRaisesRegex(RuntimeError,
                                            "serves model_type 'deepseek_v4' only"):
                    self.install(run)
                bb = run.model.model
                self.assertFalse(hasattr(bb, "_steer_stack"))
                self.assertFalse(any(getattr(bb.layers[i], "_weightless_dsv4_hook", None)
                                     for i in range(NL)))

    def test_fired_check_catches_the_tbo_path(self):
        """Two-batch overlap runs layer ops, not the layers: with the real
        model loop taking that branch, the first forward fails."""
        run = real_runner()
        self.install(run)
        run_model(run, x=torch.randn(2, H))
        bb = run.model.model
        with mock.patch.object(D.DeepseekV4Model, "_can_run_tbo", lambda self, fb: True), \
                mock.patch.object(D.DeepseekV4Model, "_forward_layers_tbo",
                                  lambda self, positions, hidden_states, forward_batch: hidden_states):
            with self.assertRaisesRegex(RuntimeError, r"did not run in this forward"):
                run_model(run, x=torch.randn(2, H))
        self.assertIsNotNone(bb)


# ---------------------------------------------------------------- structure


MODEL_FILE = "srt/models/deepseek_v4.py"


@unittest.skipUnless(_trees(MODEL_FILE), "no SGLang package with models/deepseek_v4.py found")
class Structure(unittest.TestCase):
    """Pins, per SGLang tree, the code paths the DeepSeek-V4 row relies on."""

    def assertIn(self, member, container, msg=None):  # noqa: N802
        if isinstance(member, str) and isinstance(container, str):
            member, container = _flat(member), _flat(container)
        super().assertIn(member, container, msg)

    def trees(self):
        return [(t, _parse(t, "srt/models/deepseek_v4.py")) for t in _trees(MODEL_FILE)]

    def test_model_loop_calls_each_layer_and_unpacks_four(self):
        for t, m in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(m, "DeepseekV4Model"), "forward")
                loops = [n for n in ast.walk(fwd) if isinstance(n, ast.For)]
                self.assertEqual(len(loops), 2)  # the layer loop and the freqs_cis reset
                loop = [lp for lp in loops if _src(lp.iter) == "range(self.start_layer, self.end_layer)"]
                self.assertEqual(len(loop), 1)
                loop = loop[0]
                self.assertIn("layer = self.layers[i]", _src(loop))
                assigns = [n for n in ast.walk(loop) if isinstance(n, ast.Assign)
                           and isinstance(n.value, ast.Call) and _src(n.value.func) == "layer"]
                self.assertEqual(len(assigns), 1)
                self.assertEqual(_src(assigns[0].targets[0]),
                                 "(hidden_states, prev_residual, prev_post, prev_comb)")
                kws = {k.arg: _src(k.value) for k in assigns[0].value.keywords}
                self.assertEqual(kws.get("hidden_states"), "hidden_states")
                self.assertEqual(kws.get("prev_residual"), "prev_residual")
                # no other call of a layer, and no layer.forward( in the model
                self.assertEqual(len(_calls(fwd, "layer")), 1)
                self.assertEqual(_calls(_cls(m, "DeepseekV4Model"), "forward"), [])

    def test_last_layer_is_folded_by_the_loop_from_the_loop_variable(self):
        for t, m in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(m, "DeepseekV4Model"), "forward")
                s = _src(fwd)
                self.assertIn("if use_fused and last_layer is not None:\n"
                              "                hidden_states = last_layer.hc_post(hidden_states, "
                              "prev_residual, prev_post, prev_comb)", s)
                self.assertIn("use_fused = self.use_fused_mhc_post_pre", s)
                # the DSpark aux capture completes the same (steered) loop variable
                self.assertIn("completed = layer.hc_post(hidden_states, prev_residual, prev_post, "
                              "prev_comb)", s)
                self.assertIn("return PPProxyTensors({'hidden_states': hidden_states.flatten(1)})", s)

    def test_only_other_paths_are_v41_and_tbo(self):
        """The layer loop is skipped only by the DeepSeek-V4.1 scheme (which
        calls forward_hc_pre_from_prev directly; refused) and by two-batch
        overlap (refused at boot; the fired check catches it too)."""
        for t, m in self.trees():
            with self.subTest(tree=t):
                fwd = _fn(_cls(m, "DeepseekV4Model"), "forward")
                s = _src(fwd)
                self.assertIn("if self.hc_pre_from_prev_sublayer:", s)
                self.assertIn("hidden_states, last_pre, tail = self._forward_layers_hc_pre_from_prev(", s)
                self.assertIn("elif run_tbo:", s)
                # and nothing else: the chain is exactly V4.1 / run_tbo / the layer loop
                # (the other `if self.hc_pre_from_prev_sublayer:` is the output tail, no elif)
                heads = [n for n in ast.walk(fwd) if isinstance(n, ast.If)
                         and _src(n.test) == "self.hc_pre_from_prev_sublayer"
                         and len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If)]
                self.assertEqual(len(heads), 1)
                tbo_if = heads[0].orelse
                self.assertEqual(len(tbo_if), 1)
                self.assertIsInstance(tbo_if[0], ast.If)
                self.assertEqual(_src(tbo_if[0].test), "run_tbo")
                loop_arm = tbo_if[0].orelse
                self.assertFalse(len(loop_arm) == 1 and isinstance(loop_arm[0], ast.If),
                                 "a further elif branch before the layer loop")
                loops = [n for n in loop_arm if isinstance(n, ast.For)]
                self.assertEqual([_src(n.iter) for n in loops],
                                 ["range(self.start_layer, self.end_layer)"])
                self.assertEqual([n for n in ast.walk(fwd) if isinstance(n, ast.Return)
                                  and n.lineno < heads[0].lineno], [],
                                 "a return before the branch chain")
                self.assertIn("run_tbo = self._can_run_tbo(forward_batch) and (not capture_dspark)", s)
                v41 = _src(_fn(_cls(m, "DeepseekV4Model"), "_forward_layers_hc_pre_from_prev"))
                self.assertIn(".forward_hc_pre_from_prev(", v41)
                self.assertIn("next_input", v41)  # the next layer's input from the unsteered stream
                tbo = _fn(_cls(m, "DeepseekV4Model"), "_can_run_tbo")
                self.assertIn("is_tbo_enabled()", _src(tbo))

    def test_layer_returns_the_fused_four_tuple_or_the_unfused_stream(self):
        for t, m in self.trees():
            with self.subTest(tree=t):
                layer = _cls(m, "DeepseekV4DecoderLayer")
                fwd = _fn(layer, "forward")
                rets = [_src(r.value) for r in sorted(
                    (n for n in ast.walk(fwd) if isinstance(n, ast.Return)), key=lambda n: n.lineno)]
                self.assertEqual(rets, ["(hidden_states, None, None, None)",
                                        "(hidden_states, residual, post, comb)"])
                s = _src(fwd)
                self.assertIn("use_fused = self.use_fused_mhc_post_pre", s)
                self.assertIn("if not use_fused:\n            hidden_states = self.hc_post("
                              "hidden_states, residual, post, comb)\n"
                              "            return (hidden_states, None, None, None)", s)
                # the pending write is the MoE output, and the next layer folds it first
                self.assertIn("hidden_states = self._run_moe_ffn_dp_sync(", s)
                self.assertIn("if prev_residual is not None and use_fused:", s)
                self.assertIn("hidden_states = self.hc_post(hidden_states, prev_residual, "
                              "prev_post, prev_comb)", s)
                init = _src(_fn(layer, "__init__"))
                self.assertIn("self.use_fused_mhc_post_pre = is_cross_layer_mhc_fusion_enabled() "
                              "or _is_fused_mhc_post_pre_enabled_xpu()", init)
                self.assertIn("if self.hc_pre_from_prev_sublayer:\n            "
                              "self.use_fused_mhc_post_pre = False", init)
                self.assertIn("self.layer_id = layer_id", init)
                model_init = _src(_fn(_cls(m, "DeepseekV4Model"), "__init__"))
                self.assertIn("self.use_fused_mhc_post_pre = is_cross_layer_mhc_fusion_enabled() "
                              "or _is_fused_mhc_post_pre_enabled_xpu()", model_init)
                self.assertIn("DeepseekV4DecoderLayer(", model_init)

    def test_the_ffn_write_is_reduced_inside_the_layer(self):
        """The partial-sum forms that decide tp: the MoE output is reduced (or
        reduce-scattered) inside _run_moe_ffn_dp_sync; nothing in this file
        uses UnreducedOutput, the attribute marker, or a deferred MoE
        finalize; the MhcPostFusion all-reduce fusion is only in the V4.1
        path. The site is linear anyway, so tp is 'ok'."""
        for t, m in self.trees():
            with self.subTest(tree=t):
                src = _src(m)
                for marker in ("UnreducedOutput", "_sglang_needs_allreduce_fusion",
                               "defer_moe_finalize", "fuse_mlp_allreduce="):
                    self.assertNotIn(marker, src, marker)
                layer = _cls(m, "DeepseekV4DecoderLayer")
                users = [f.name for f in layer.body if isinstance(f, ast.FunctionDef)
                         and "MhcPostFusion" in _src(f)]
                self.assertEqual(users, ["forward_hc_pre_from_prev"])
                sync = _src(_fn(layer, "_run_moe_ffn_dp_sync"))
                self.assertIn("get_forward().scoped(mlp_reduce_scatter=mlp_reduce_scatter)", sync)
                self.assertIn("return hidden_states", sync)

    def test_v41_config_is_served_by_the_v4_class(self):
        for t, _ in self.trees():
            with self.subTest(tree=t):
                c = _parse(t, "srt/configs/deepseek_v41.py")
                s = _src(c)
                self.assertIn("if values.get('architectures') == ['DeepseekV41ForCausalLM']:\n"
                              "        values['architectures'] = ['DeepseekV4ForCausalLM']", s)
                v41 = _src(_cls(c, "DeepseekV41Config"))
                self.assertIn("model_type = 'deepseek_v41'", v41)
                self.assertIn("hc_pre_from_prev_sublayer = True", v41)
                m = _parse(t, "srt/models/deepseek_v4.py")
                entry = [n for n in m.body if isinstance(n, ast.Assign)
                         and _src(n.targets[0]) == "EntryClass"]
                self.assertEqual([_src(e.value) for e in entry], ["[DeepseekV4ForCausalLM]"])
                cfg = _src(_cls(_parse(t, "srt/configs/deepseek_v4.py"), "DeepSeekV4Config"))
                self.assertIn("hc_pre_from_prev_sublayer: bool = False", cfg)
                self.assertIn("model_type: str = 'deepseek_v4'", cfg)
                self.assertIn("hidden_size: int = 4096", cfg)
                self.assertIn("num_hidden_layers: int = 43", cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
