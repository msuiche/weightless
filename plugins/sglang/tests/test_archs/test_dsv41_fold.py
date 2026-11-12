"""The DeepSeek-V4.1 row with a GLP spec_version 3 fold-site file
(glp.hook_point = hc_fold_pre_engram), CPU only, on the fakes of
test_dsv41.py (call for call with SGLang's V4.1 loop and stage boundaries).

What spec 3 adds to the row (docs/spec/GLP.md, "Multi-copy streams and the
fold site"): the same fold and the same edit of the mean, plus the companion
writer, the attention output (the write the attention boundary's
``HcUpdate.post`` folds in) projected with the same stack and alpha. The
layer at position k uses stack row k + 1 (the vLLM plugin's archs/dsv41.py).

- install: folds on the file's rows, writers on the layers one below them,
  the boot line and the manifest name both;
- the forward equals a hand-wired reference built from the plugin's own
  edit functions, bit for bit; the attention write the post receives is the
  projected write (float64 check);
- alpha 0 equals stock bit for bit with both halves armed;
- pipeline ranks 13/13/14 equal one rank, and each rank arms its own
  writers, including the one whose row the next rank folds;
- the writers' fired check; undo;
- refusals: a fold hook under spec 1 or 2, ``each``, attention TP, a
  non-tensor attention write;
- the published GLP-27 file at width 5120, when WEIGHTLESS_TEST_GLP_DIR
  holds it.
"""
import os
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import glpfiles  # noqa: E402
from glpfiles import good_meta, write_gguf  # noqa: E402

from test_archs import unit_dirs  # noqa: E402
from test_archs import test_dsv41 as T  # noqa: E402  (the fakes)

from weightless_sglang import core as wcore  # noqa: E402
from weightless_sglang.archs import ARCH, COMPANION  # noqa: E402
from weightless_sglang.archs import dsv41 as A  # noqa: E402
from weightless_sglang.install import install_steering  # noqa: E402

ROW = A.ROW_NAME
FOLD = "hc_fold_pre_engram"
RES = "residual_stream_post_layer"
GLP27 = glpfiles.find_glp("glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf")
ROWS27 = tuple(range(11, 38))


class FoldBase(T.Base):
    def write3(self, layers=ROWS27, width=16, alpha="3.5", version="3", hook=FOLD, seed=0):
        m = good_meta(layers=layers, alpha=alpha)
        m["controlvector.model_hint"] = "deepseek_v41"
        m["glp.hook_point"] = hook
        m["glp.derived_at"] = hook
        m["glp.spec_version"] = version
        path = os.path.join(self.tmp.name, f"f{seed}.gguf")
        write_gguf(path, m, unit_dirs(layers, width, seed))
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        return path

    @staticmethod
    def writers(run):
        bb = run.model.model
        out = {}
        for i in range(bb.start_layer, bb.end_layer):
            p = vars(bb.layers[i].attn_boundary.declaration.update).get("post")
            if p is not None:
                out[i] = p._weightless_row_id
        return out


def reference_forward(partition, x, rows, alpha, hidden=16, seed=0):
    """Stock fakes with the two halves wired by hand from the plugin's own
    edit functions: the FFN post of layer N edited with hc_mean_rows and
    row N, the attention write of layer k with steer_rows and row k + 1."""
    dirs = {int(k.split(".")[1]): torch.from_numpy(v) for k, (v, _) in
            unit_dirs(rows, hidden, seed).items()}
    a = torch.tensor(float(alpha))
    out = None
    for rank in range(len(partition)):
        run = T.runner(partition, rank, hidden=hidden)
        bb = run.model.model
        for i in range(bb.start_layer, bb.end_layer):
            lay = bb.layers[i]
            if i in dirs:
                fupd = lay.ffn_boundary.declaration.update
                forig = fupd.post

                def fpost(*args, _o=forig, _d=dirs[i], **kw):
                    st = _o(*args, **kw)
                    return st.with_residual(A.hc_mean_rows(st.streams, _d, a))
                fupd.post = fpost
            if i + 1 in dirs:
                aupd = lay.attn_boundary.declaration.update
                aorig = aupd.post

                def apost(output, state, *args, _o=aorig, _d=dirs[i + 1], **kw):
                    return _o(wcore.steer_rows(output, None, _d, a), state, *args, **kw)
                aupd.post = apost
        out = bb(input_embeds=x) if rank == 0 else bb(pp_proxy_tensors=out)
    return out


def steered_forward(case, partition, x, hidden=16):
    out = None
    for rank in range(len(partition)):
        run = T.runner(partition, rank, hidden=hidden)
        case.install(run)
        bb = run.model.model
        out = bb(input_embeds=x) if rank == 0 else bb(pp_proxy_tensors=out)
    return out


def bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t


class Fold(FoldBase):
    def test_row_serves_both_hooks_and_registers_the_writer(self):
        self.assertEqual(ARCH[ROW].hooks, frozenset({RES, FOLD}))
        comp = COMPANION["dsv41_hc_post"]
        self.assertEqual((comp.hooks, comp.row_offset), (frozenset({FOLD}), 1))
        self.assertIs(comp.install, A.dsv41_attn_writer)

    def test_install_arms_folds_and_writers(self):
        self.write3()
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = T.runner()
        rec = self.install(run)
        self.assertEqual(self.wrapped(run), list(ROWS27))
        self.assertEqual(self.writers(run), {k: k + 1 for k in range(10, 37)})
        self.assertEqual((rec["hook_point"], rec["alpha"], rec["edit"]),
                         (FOLD, 3.5, "hc copies: mean coefficient"))
        c = rec["companion"]
        self.assertEqual((c["layers"], c["rows"], c["row_offset"], c["kernel"]),
                         (list(range(10, 37)), list(ROWS27), 1, "torch"))
        self.assertIn("companion=" + A.WRITER_SITE + " layers=10..36 (27, stack row = layer + 1) "
                      "companion_kernel=torch", self.log)
        self.assertTrue(os.path.exists(rec["manifest"]))

    def test_a_residual_file_arms_no_writer(self):
        self.write(layers=ROWS27)
        run = T.runner()
        rec = self.install(run)
        self.assertEqual(self.writers(run), {})
        self.assertNotIn("companion", rec)
        self.assertNotIn("companion=", self.log)

    def test_forward_equals_the_hand_wired_reference(self):
        self.write3()
        x = T.embeds()
        got = steered_forward(self, (40,), x)
        want = reference_forward((40,), x, ROWS27, 3.5)
        self.assertTrue(torch.equal(bits(got), bits(want)))
        stock = T.runner()
        self.assertFalse(torch.equal(bits(got), bits(stock.model.model(input_embeds=x))))

    def test_the_post_receives_the_projected_attention_write(self):
        self.write3()
        seen = {}
        inner = T.HcUpdate.post

        def spy(self_, output, state, *, next=None):  # the original post, as the wrapper calls it
            if self_.attention and self_.layer.layer_id == 20:
                seen["edited"] = output.clone()
            return inner(self_, output, state, next=next)
        with mock.patch.object(T.HcUpdate, "post", spy):
            run = T.runner()
            self.install(run)
            upd = run.model.model.layers[20].attn_boundary.declaration.update
            wrapped = vars(upd)["post"]

            def capture(output, state, *args, **kw):  # outside the wrapper: the raw write
                seen["raw"] = output.clone()
                return wrapped(output, state, *args, **kw)
            upd.post = capture
            run.model.model(input_embeds=T.embeds())
        a, b = seen["raw"].double(), seen["edited"].double()
        d = torch.from_numpy(unit_dirs(ROWS27, 16, 0)["direction.21"][0]).double()
        want = a - 3.5 * (a @ d).unsqueeze(-1) * d
        step = a.abs().max().item() * 2 ** -7 + 1e-6  # one bf16 step at the largest magnitude
        self.assertLessEqual((b - want).abs().max().item(), 2 * step)
        self.assertFalse(torch.equal(a, b))

    def test_alpha_zero_is_stock(self):
        self.write3(alpha="0")
        x = T.embeds()
        got = steered_forward(self, (40,), x)
        stock = T.runner().model.model(input_embeds=x)
        self.assertTrue(torch.equal(bits(got), bits(stock)))

    def test_pipeline_ranks_equal_one_rank_and_arm_their_own_writers(self):
        self.write3()
        x = T.embeds()
        one = steered_forward(self, (40,), x)
        three = steered_forward(self, (13, 13, 14), x)
        self.assertTrue(torch.equal(bits(one), bits(three)))
        want = {0: (range(10, 13), [11, 12]), 1: (range(13, 26), list(range(13, 26))),
                2: (range(26, 37), list(range(26, 38)))}
        for rank, (wl, fl) in want.items():
            with self.subTest(rank=rank):
                run = T.runner((13, 13, 14), rank)
                rec = self.install(run)
                self.assertEqual(sorted(self.writers(run)), list(wl))
                self.assertEqual(rec["local_layer_ids"], fl)
        # layer 12's writer (rank 0) uses row 13, which rank 1 folds
        run = T.runner((13, 13, 14), 0)
        self.install(run)
        self.assertEqual(self.writers(run)[12], 13)

    def test_the_writers_fired_check(self):
        self.write3()
        run = T.runner()
        self.install(run)
        bb = run.model.model
        upd = bb.layers[15].attn_boundary.declaration.update
        wrapped = vars(upd)["post"]
        upd.post = lambda output, state, *a, **k: type(upd).post(upd, output, state, *a, **k)
        with self.assertRaisesRegex(RuntimeError, r"companion .* did not run .*layers \[16\]"):
            bb(input_embeds=T.embeds())

        def twice(output, state, *a, **k):
            wrapped(output, state, *a, **k)
            return wrapped(output, state, *a, **k)
        upd.post = twice
        with self.assertRaisesRegex(RuntimeError, r"companion .* ran more than once"):
            bb(input_embeds=T.embeds())
        self.assertEqual(run._weightless_steer_companion_check.layer_ids, list(ROWS27))

    def test_undo_restores_the_attention_posts(self):
        self.write3()
        run = T.runner()
        self.install(run)
        for undo in run._weightless_steer_handles:
            undo()
        self.assertEqual(self.writers(run), {})
        self.assertEqual(self.wrapped(run), [])


class FoldRefusals(FoldBase):
    def refuse(self, pattern, run=None):
        with self.assertRaisesRegex(Exception, pattern):
            self.install(run if run is not None else T.runner())

    def test_a_fold_hook_needs_spec_version_3(self):
        for v in ("1", "2"):
            with self.subTest(version=v):
                self.write3(version=v, seed=int(v))
                self.refuse(r"hc fold site.*spec_version")

    def test_each_is_refused_for_a_fold_file(self):
        self.write3()
        os.environ["WEIGHTLESS_STEER_HC_COPIES"] = "each"
        self.refuse(r"spec_version 3 fold-site file.*mean.*'each'")
        os.environ["WEIGHTLESS_STEER_HC_COPIES"] = "mean"
        self.install(T.runner())

    def test_attention_tp_is_refused(self):
        self.write3()
        run = T.runner()
        run.model.model.layers[10].self_attn = type("A", (), {"attn_tp_size": 2})()
        self.refuse(r"layer 10 \(companion writer, row 11\): attention TP is 2", run)

    def test_a_non_tensor_attention_write_fails_closed(self):
        self.write3()
        run = T.runner()
        self.install(run)
        upd = run.model.model.layers[20].attn_boundary.declaration.update
        with self.assertRaisesRegex(A.Dsv41SiteError, "attention write is object, not a tensor"):
            upd.post(object(), None)
        with self.assertRaisesRegex(A.Dsv41SiteError, r"shape \(2, 4, 16\)"):
            upd.post(torch.zeros(2, 4, 16), None)

    def test_a_writer_layer_of_another_class_is_refused(self):
        self.write3()
        run = T.runner()
        bb = run.model.model
        bad = type("DeepseekV4NextNLayer", (T._DecoderLayer,), {})(bb.config, 10)
        bb.layers[10] = bad
        self.refuse(r"layer 10 \(companion .*\) is DeepseekV4NextNLayer", run)


@unittest.skipIf(GLP27 is None, "glp.deepseek-v41-flash-1017-GLP-27-L11-37-a3.5.gguf is not "
                                "under WEIGHTLESS_TEST_GLP_DIR")
class PublishedGlp27(FoldBase):
    def test_the_published_file_installs_and_runs(self):
        os.environ["WEIGHTLESS_STEER_PATH"] = GLP27
        run = T.runner(hidden=5120)
        rec = self.install(run)
        self.assertEqual(rec["local_layer_ids"], list(ROWS27))
        self.assertEqual(rec["companion"]["layers"], list(range(10, 37)))
        self.assertEqual((rec["width"], rec["alpha"], rec["hook_point"], rec["model_hint"]),
                         (5120, 3.5, FOLD, "deepseek_v41"))
        out = run.model.model(input_embeds=T.embeds(T=2, W=5120))
        self.assertEqual(tuple(out.shape), (2, 5120))
        self.assertTrue(bool(torch.isfinite(out.float()).all()))
        print(f"  {os.path.basename(GLP27)} -> {ROW}: folds 11..37, writers 10..36, width 5120 "
              f"alpha 3.5, one forward")


if __name__ == "__main__":
    unittest.main()
