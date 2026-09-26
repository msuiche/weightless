"""The plugin as a whole, CPU only.

- ``register()`` and the entry point; ``install_steering`` on a fake
  64-layer qwen3_5 runner: the hooked-layer set keyed by global id, the env
  and the explicit (``source="server_args"``) routes, the draft skip,
  idempotency, and the fail-closed cases.
- The published-file matrix: every published GLP file through
  install_steering on a fake runner of the right width and depth.

The matrix has one row per (file, served class): the fake's geometry (the depth and
width SGLang's config gives that model), and the expectation, either the
installed range, width, hook point and alpha, or the named refusal.
The glp.structure rules are part of it: the files with no glp.structure
(Qwen GLP-49/63, DeepSeek-V4 0731 GLP-29) or free text in it (GLM-5.3
GLP-77, both Vision-Exp files) install on their normal rows; the looped
files (Nanbeige, Ouro) are refused on every row that is not looped.

A table row whose served class is not in this build's ARCH is skipped with
a reason naming the class and the row's status, so the matrix stays
complete in a build without that row. The cross check puts every file on every
other row this build serves and expects a refusal each time.

Needs WEIGHTLESS_TEST_GLP_DIR (the folders of the published files). With
WEIGHTLESS_TEST_CONFIG_DIR (folders <org>__<repo>/config.json, the public
config.json files), WEIGHTLESS_TEST_GLM_CONFIG, WEIGHTLESS_TEST_QWEN_CONFIG
(the Qwen3.8-27B config.json at the files' glp.base_revision) and
WEIGHTLESS_TEST_DSV41_CONFIG, each fake's geometry is also checked against
the model's real config.

The cross check builds the fake first, outside the refusal check, and
requires a refusal of a known family (width, model_hint, hook point,
structure, layer class or range, or a row's named refusal), so a broken
fake or an unrelated error cannot pass as a refusal.
"""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tests/
import glpfiles  # noqa: E402  (sets up the import paths)
from glpfiles import good_meta, good_tensors, write_gguf  # noqa: E402
from test_archs import test_dsv4 as fd  # noqa: E402  (the DeepSeek-V4 fake)
from test_archs import test_glm5next as fg  # noqa: E402  (the GLM-5.3-Flash fake)
from test_archs import test_qwen38 as fakes  # noqa: E402  (the Qwen3.8-27B fake)
from test_archs import test_refused as fa  # noqa: E402  (a fake for any served class)

from weightless_sglang.archs import ARCH, REFUSED  # noqa: E402
from weightless_sglang.install import install_steering, resolve_backbone  # noqa: E402


# ---------------------------------------------------------------- install and register


ENV = ["WEIGHTLESS_STEER_PATH", "WEIGHTLESS_STEER_ALPHA", "WEIGHTLESS_STEER_LAYERS",
       "WEIGHTLESS_STEER_HOOK", "WEIGHTLESS_STEER_MANIFEST_DIR", "WEIGHTLESS_STEER_DIAG",
       "WEIGHTLESS_STEER_KERNEL"]
LAYERS = (10, 11, 12, 30, 58)


def meta(layers=LAYERS, **extra):
    m = good_meta(layers=layers, alpha="1.0")
    m["controlvector.model_hint"] = "qwen3_5"
    m.update(extra)
    return m


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "vec.gguf")
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        for k in ENV:
            os.environ.pop(k, None)

    def write(self, m=None, t=None, path=None):
        write_gguf(path or self.path, meta() if m is None else m,
                   good_tensors(LAYERS) if t is None else t)
        return path or self.path

    def hooked(self, run):
        bb = run.model.model
        return sorted(i for i, l in enumerate(bb.layers) if l._forward_hooks)


class Installs(Base):
    def test_env_unset_installs_nothing(self):
        run = fakes.runner()
        self.assertIsNone(install_steering(run, source="env"))
        self.assertEqual(self.hooked(run), [])
        self.assertFalse(hasattr(run.model.model, "_steer_stack"))

    def test_hooked_set_is_exactly_the_file_layers(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        os.environ["WEIGHTLESS_STEER_MANIFEST_DIR"] = self.tmp.name
        run = fakes.runner()
        rec = install_steering(run, source="env")
        self.assertEqual(self.hooked(run), list(LAYERS))
        self.assertEqual(rec["hooked_layers"], list(LAYERS))
        self.assertEqual(rec["full_attn_layers"], [11])
        bb = run.model.model
        self.assertEqual(tuple(bb._steer_stack.shape), (64, 1, 8))
        self.assertEqual(bb._steer_stack.dtype, torch.float32)
        nz = sorted(int(i) for i in (bb._steer_stack.abs().sum((1, 2)) > 0).nonzero().flatten())
        self.assertEqual(nz, list(LAYERS))
        self.assertEqual(float(bb._steer_alpha), 1.0)
        self.assertNotIn("_steer_stack", bb.state_dict())
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, f"weightless-manifest-{os.getpid()}.json")))

    def test_forward_matches_reference_semantics(self):
        """Hooked fake model == manual h <- h - a (h.d) d at each steered layer."""
        self.write()
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        os.environ["WEIGHTLESS_STEER_ALPHA"] = "1.0"
        run = fakes.runner()
        bb = run.model.model
        x = torch.randn(5, 8, dtype=torch.float64)
        bb.double()
        # manual reference on a second, unhooked copy
        ref = fakes.runner().model.model.double()
        from weightless_steer.core import SteeringCore
        core = SteeringCore.from_env(hook="residual_stream_post_layer", num_layers=64, hidden_size=8)
        install_steering(run, source="env")
        h, r = x, None
        for i in range(64):
            h, r = ref.layers[i](hidden_states=h, residual=r)
            if i in core.dirs:
                d = core.dirs[i].double()
                s = h + r
                s = s - 1.0 * (s @ d).unsqueeze(-1) * d
                h = s - r
        want = h + r
        got = bb(x)
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

    def test_layer_subset_and_pp_rank_range(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        os.environ["WEIGHTLESS_STEER_LAYERS"] = "11,30,58"
        run = fakes.runner(start=20, end=64)
        rec = install_steering(run, source="env")
        self.assertEqual(self.hooked(run), [30, 58])
        self.assertEqual(rec["hooked_layers"], [30, 58])

    def test_draft_runner_is_left_stock(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        run = fakes.runner(draft=True)
        self.assertIsNone(install_steering(run, source="env"))
        self.assertEqual(self.hooked(run), [])

    def test_flag_route_equals_env_route_and_is_idempotent(self):
        self.write()
        run = fakes.runner()
        rec = install_steering(run, source="server_args", path=self.path, alpha=0.5, layers="10,58")
        self.assertEqual(rec["hooked_layers"], [10, 58])
        self.assertEqual(rec["alpha"], 0.5)
        self.assertNotIn("WEIGHTLESS_STEER_PATH", os.environ)  # env restored
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        self.assertIs(install_steering(run, source="env"), rec)  # same file: once
        self.assertEqual(sum(len(l._forward_hooks) for l in run.model.model.layers), 2)
        other = self.write(path=os.path.join(self.tmp.name, "other.gguf"))
        os.environ["WEIGHTLESS_STEER_PATH"] = other
        with self.assertRaises(RuntimeError):
            install_steering(run, source="env")

    def test_alpha_zero_is_still_hooked(self):
        self.write()
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        os.environ["WEIGHTLESS_STEER_ALPHA"] = "0"
        run = fakes.runner()
        rec = install_steering(run, source="env")
        self.assertEqual(rec["alpha"], 0.0)
        self.assertEqual(self.hooked(run), list(LAYERS))
        x = torch.randn(3, 8)
        stock = fakes.runner().model.model
        torch.testing.assert_close(run.model.model(x), stock(x), rtol=0, atol=0)


class FailsClosed(Base):
    def boot(self, run=None, **env):
        os.environ["WEIGHTLESS_STEER_PATH"] = self.path
        os.environ.update(env)
        with self.assertRaises(Exception) as cm:
            install_steering(run or fakes.runner(), source="env")
        return str(cm.exception)

    def test_model_hint_mismatch(self):
        self.write(m=meta(**{"controlvector.model_hint": "llama"}))
        self.assertIn("model_hint", self.boot())

    def test_wrong_width(self):
        self.write()
        self.assertIn("width", self.boot(run=fakes.runner(width=16)))

    def test_layer_out_of_range(self):
        self.write(m=meta(layers=(10, 70)), t=good_tensors((10, 70)))
        self.assertIn("out of range", self.boot())

    def test_add_mode(self):
        self.write(m=meta(**{"glp.mode": "add"}))
        self.assertIn("glp.mode", self.boot())

    def test_wrong_hook_point(self):
        self.write(m=meta(**{"glp.hook_point": "attn_out"}))
        self.assertIn("hook_point", self.boot())

    def test_rank2(self):
        t = {}
        for i in LAYERS:  # an orthonormal rank-2 basis, so only the rank gate can refuse
            t[f"direction.{i}"] = (np.eye(8, dtype=np.float32)[0], 0)
            t[f"direction.{i}.1"] = (np.eye(8, dtype=np.float32)[1], 0)
        self.write(m=meta(**{"glp.rank": "2", "glp.spec_version": "2", "glp.orthonormal": "true"}), t=t)
        self.assertIn("rank", self.boot())

    def test_scale_keys(self):
        self.write(m=meta(**{"glp.spec_version": "2", "glp.layer_scales": "1,1,1,1,1"}))
        self.assertIn("layer_scales", self.boot())

    def test_missing_file(self):
        self.boot()

    def test_nonfinite_alpha(self):
        self.write()
        self.boot(WEIGHTLESS_STEER_ALPHA="inf")

    def test_unsupported_arch(self):
        self.write()
        self.assertIn("no SGLang steering", self.boot(run=fakes.runner(cls=fakes.Unsupported)))

    def test_unvalidated_layer_class(self):
        self.write()
        self.assertIn("not validated", self.boot(run=fakes.runner(layer_cls=fakes.OtherLayer)))

    def test_layers_filter_selects_nothing(self):
        self.write()
        self.assertIn("matched no layers", self.boot(WEIGHTLESS_STEER_LAYERS="3"))


class Register(Base):
    def test_register_is_inert_without_env_and_hooks_load_model_with_it(self):
        try:
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType
        except Exception as e:  # pragma: no cover
            self.skipTest(f"sglang not importable: {e}")
        from weightless_sglang import plugin
        HookRegistry.reset()
        self.addCleanup(HookRegistry.reset)
        plugin.register()
        self.assertNotIn(plugin.TARGET, dict(HookRegistry._hooks))
        os.environ["WEIGHTLESS_STEER_PATH"] = "/nonexistent.gguf"
        plugin.register()
        hooks = HookRegistry._hooks[plugin.TARGET]
        self.assertEqual([(ht, h) for ht, h, _ in hooks],
                         [(HookType.AFTER, plugin._after_load_model)])

    def test_entry_point_is_declared(self):
        from importlib.metadata import entry_points
        eps = [e.value for e in entry_points(group="sglang.srt.plugins")]
        if not eps:
            self.skipTest("plugin not pip-installed in this interpreter")
        self.assertIn("weightless_sglang.plugin:register", eps)


# ---------------------------------------------------------------- the published-file matrix


def _qwen_layers(n):
    return ["Qwen3_5AttentionDecoderLayer" if i % 4 == 3 else "Qwen3_5LinearDecoderLayer"
            for i in range(n)]


def _qwen4_layers(n):
    return ["Qwen4ExpAttentionDecoderLayer" if i % 4 == 3 else "Qwen4ExpLinearDecoderLayer"
            for i in range(n)]


# Nemotron-3.5-Lightning's layers_block_type (52 entries; the public config)
_NEMO_KIND = {"mamba": "NemotronHMambaDecoderLayer", "moe": "NemotronHMoEDecoderLayer",
              "attention": "NemotronHAttentionDecoderLayer", "mlp": "NemotronHMLPDecoderLayer"}
_NEMO_PATTERN = ("mamba moe mamba moe mamba attention moe mamba moe mamba moe mamba attention "
                 "moe mamba moe mamba moe mamba attention moe mamba moe mamba moe mamba attention "
                 "moe mamba moe mamba moe mamba attention moe mamba moe mamba moe mamba moe mamba "
                 "attention moe mamba moe mamba moe mamba moe mamba moe").split()

# key: (file, served class, row status, how the fake is built, expectation)
#   fake: ("generic", backbone path, layer class names, config, hf model_type)
#         ("glm", hidden) / ("dsv4", config kwargs, hf model_type) / ("qwen", width)
#   expect: ("ok", steered ids, width, alpha)  or  ("refused", message regex)
# geometry: (config source, {attribute: value}) checked against the real config
MATRIX = {
    "qwen38-glp49": dict(
        file="Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf",
        row="Qwen3_5ForConditionalGeneration", status="gpu-validated",
        fake=("qwen", 5120), expect=("ok", list(range(10, 59)), 5120, 1.0),
        geometry=("qwen", {"num_hidden_layers": 64, "hidden_size": 5120,
                           "model_type": "qwen3_5_text",
                           "architectures": ["Qwen3_5ForConditionalGeneration"]})),
    "qwen38-glp63": dict(
        file="Qwen3.8-27B-abliterated-cyber-GLP-63-L1-63-a1.gguf",
        row="Qwen3_5ForConditionalGeneration", status="gpu-validated",
        fake=("qwen", 5120), expect=("ok", list(range(1, 64)), 5120, 1.0),
        geometry=("qwen", {"num_hidden_layers": 64, "hidden_size": 5120})),
    "qwen38-glp49-causal": dict(
        file="Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf",
        row="Qwen3_5ForCausalLM", status="gpu-validated",
        fake=("generic", "model", _qwen_layers(64),
              dict(hidden_size=5120, num_hidden_layers=64, model_type="qwen3_5_text"), "qwen3_5"),
        expect=("ok", list(range(10, 59)), 5120, 1.0)),
    "glm53flash-glp44": dict(
        file="GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf",
        row="Glm5NextForConditionalGeneration", status="gpu-validated",
        fake=("glm", 4096), expect=("ok", list(range(1, 45)), 16384, 2.0),
        geometry=("glm", {"num_hidden_layers": 45, "hidden_size": 4096, "hc_mult": 4})),
    "glm53-glp77": dict(
        file="GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf",
        row="GlmMoeDsaForCausalLM", status="structure-tested",
        fake=("generic", "model", ["DeepseekV2DecoderLayer"] * 78,
              dict(hidden_size=6144, num_hidden_layers=78, model_type="glm_moe_dsa"), "glm_moe_dsa"),
        expect=("ok", list(range(1, 78)), 6144, 1.0),
        geometry=("RadixArk__GLM-5.3-NVFP4", {"num_hidden_layers": 78, "hidden_size": 6144})),
    "dsv4-glp29": dict(
        file="DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4.gguf",
        row="DeepseekV4ForCausalLM", status="structure-tested",
        fake=("dsv4", dict(hidden=4096, hc_mult=4, num_layers=43), "deepseek_v4"),
        expect=("ok", list(range(10, 39)), 4096, 6.0),
        geometry=("deepseek-ai__DeepSeek-V4-Flash-0731",
                  {"num_hidden_layers": 43, "hidden_size": 4096, "hc_mult": 4,
                   "model_type": "deepseek_v4"})),
    "dsv4-glp42-residual": dict(
        file="DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-42-residual-L1-42-a1.5.gguf",
        row="DeepseekV4ForCausalLM", status="structure-tested",
        fake=("dsv4", dict(hidden=4096, hc_mult=4, num_layers=43), "deepseek_v4"),
        expect=("refused", r"residual-site file.*GLP-42.*last-layer trap.*vLLM plugin refuses"),
        geometry=("deepseek-ai__DeepSeek-V4-Flash-0731",
                  {"num_hidden_layers": 43, "hidden_size": 4096})),
    "dsv4-vision-ffn": dict(
        file="DeepSeek-V4-Flash-Vision-Exp-abliterated-cyber-GLP-29-L10-38-a1.0-ffn.gguf",
        row="DeepseekV4ForCausalLM", status="structure-tested",
        fake=("dsv4", dict(hidden=4096, hc_mult=4, num_layers=43), "deepseek_v4"),
        expect=("ok", list(range(10, 39)), 4096, 1.0),
        geometry=("deepseek-ai__DeepSeek-V4-Flash-Vision-Exp",
                  {"num_hidden_layers": 43, "hidden_size": 4096, "hc_mult": 4,
                   "model_type": "deepseek_v4", "architectures": ["DeepseekV4ForCausalLM"]})),
    "dsv4-vision-plain": dict(
        file="DeepSeek-V4-Flash-Vision-Exp-abliterated-cyber-GLP-29-L10-38-a1.0.gguf",
        row="DeepseekV4ForCausalLM", status="structure-tested",
        fake=("dsv4", dict(hidden=4096, hc_mult=4, num_layers=43), "deepseek_v4"),
        expect=("refused", r"plain DeepSeek-V4-Flash-Vision-Exp GLP-29 file.*Serve the -ffn file"),
        geometry=("deepseek-ai__DeepSeek-V4-Flash-Vision-Exp",
                  {"num_hidden_layers": 43, "hidden_size": 4096})),
    "dsv41-glp39": dict(
        file="glp.deepseek-v41-flash-GLP-39-L1-39-a0.5.gguf",
        row="DeepseekV4ForCausalLM", status="structure-tested",
        fake=("dsv4", dict(hidden=5120, hc_mult=4, num_layers=40, model_type="deepseek_v41",
                           v41=True), "deepseek_v41"),
        expect=("refused", r"is DeepSeek-V4.1 \(model_type='deepseek_v41'\).*forward_hc_pre_from_prev"),
        geometry=("dsv41", {"num_hidden_layers": 40, "hidden_size": 5120, "hc_mult": 4})),
    "hy4-glp77": dict(
        file="Hy4-preview-abliterated-cyber-GLP-77-L1-77-a2.0.gguf",
        row="HYV4ForCausalLM", status="structure-tested",
        fake=("generic", "model", ["HYV4DecoderLayer"] * 78,
              dict(hidden_size=6144, num_hidden_layers=78, hc_mult=4, enable_ihc=True,
                   model_type="hy_v4"), "hy_v4"),
        expect=("ok", list(range(1, 78)), 6144, 2.0),
        geometry=("tencent__Hy4-preview-FP8",
                  {"num_hidden_layers": 78, "hidden_size": 6144, "hc_mult": 4, "enable_ihc": True})),
    "kimi-k3-glp92": dict(
        file="glp.kimi-k3-GLP-92-L1-92-a1.gguf",
        row="KimiK3ForConditionalGeneration", status="structure-tested",
        fake=("generic", "language_model.model", ["KimiK3DecoderLayer"] * 93,
              dict(hidden_size=7168, num_hidden_layers=93, model_type="kimi_linear"), "kimi_k3"),
        expect=("ok", list(range(1, 93)), 7168, 1.0),
        geometry=("moonshotai__Kimi-K3",
                  {"num_hidden_layers": 93, "hidden_size": 7168, "model_type": "kimi_linear"})),
    "kimi-k3-glp92-linear": dict(
        file="glp.kimi-k3-GLP-92-L1-92-a1.gguf",
        row="KimiK3LinearForCausalLM", status="structure-tested",
        fake=("generic", "model", ["KimiK3DecoderLayer"] * 93,
              dict(hidden_size=7168, num_hidden_layers=93, model_type="kimi_linear"), "kimi_linear"),
        expect=("ok", list(range(1, 93)), 7168, 1.0)),
    "nanbeige-glp44": dict(
        file="glp.nanbeige42-GLP-44-L1-44-a2.gguf",
        row="NanbeigeForCausalLM", status="structure-tested",
        fake=("generic", "model", ["NanbeigeDecoderLayer"] * 22,
              dict(hidden_size=3072, num_hidden_layers=22, num_loops=2, model_type="nanbeige"),
              "nanbeige"),
        # 22 layers x 2 loops = 44 execution steps; direction.N is step N-1
        expect=("ok", list(range(0, 44)), 3072, 2.0),
        geometry=("Nanbeige__Nanbeige4.2-3B",
                  {"num_hidden_layers": 22, "hidden_size": 3072, "num_loops": 2})),
    "nemotron-glp51": dict(
        file="Nemotron-3.5-Lightning-30B-A3B-abliterated-GLP-51-L1-51-a1.0.gguf",
        row="NemotronHForCausalLM", status="structure-tested",
        fake=("generic", "model", [_NEMO_KIND[k] for k in _NEMO_PATTERN],
              dict(hidden_size=2688, num_hidden_layers=52, model_type="nemotron_h"), "nemotron_h"),
        expect=("ok", list(range(1, 52)), 2688, 1.0),
        geometry=("nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
                  {"num_hidden_layers": 52, "hidden_size": 2688, "layers_block_type": _NEMO_PATTERN})),
    "nemotron-glp51-puzzle": dict(
        file="Nemotron-3.5-Lightning-30B-A3B-abliterated-GLP-51-L1-51-a1.0.gguf",
        row="NemotronHPuzzleForCausalLM", status="structure-tested",
        fake=("generic", "model", [_NEMO_KIND[k] for k in _NEMO_PATTERN],
              dict(hidden_size=2688, num_hidden_layers=52, model_type="nemotron_h"), "nemotron_h"),
        expect=("ok", list(range(1, 52)), 2688, 1.0)),
    "ouro-glp192": dict(
        file="glp.ouro26-GLP-192-L1-192-a1.gguf",
        row="OuroForCausalLM", status="refused",
        fake=("generic", "model", ["OuroDecoderLayer"] * 48,
              dict(hidden_size=2048, num_hidden_layers=48, total_ut_steps=4, model_type="ouro"),
              "ouro"),
        expect=("refused", r"OuroForCausalLM is not supported: Ouro .*no SGLang model"),
        geometry=("ByteDance__Ouro-2.6B",
                  {"num_hidden_layers": 48, "hidden_size": 2048, "total_ut_steps": 4})),
    "qwen38fn-glp47": dict(
        file="Qwen3.8-Flash-Next-abliterated-cyber-GLP-47-L1-47-a1.gguf",
        row="Qwen4ExpForConditionalGeneration", status="structure-tested",
        fake=("generic", "model", _qwen4_layers(48),
              dict(hidden_size=2560, num_hidden_layers=48, hc_count=4, model_type="qwen4_exp_text"),
              "qwen4_exp"),
        expect=("ok", list(range(1, 48)), 10240, 1.0),
        geometry=("Qwen__Qwen3.8-Flash-Next",
                  {"num_hidden_layers": 48, "hidden_size": 2560, "hc_count": 4})),
}

# The looped files, and the files with no or free-text glp.structure
LOOPED_FILES = {"glp.nanbeige42-GLP-44-L1-44-a2.gguf", "glp.ouro26-GLP-192-L1-192-a1.gguf"}
NO_STRUCTURE = {"Qwen3.8-27B-abliterated-cyber-GLP-49-L10-58-a1.gguf",
                "Qwen3.8-27B-abliterated-cyber-GLP-63-L1-63-a1.gguf",
                "DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4.gguf"}
FREE_TEXT_STRUCTURE = {"GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf",
                       "DeepSeek-V4-Flash-Vision-Exp-abliterated-cyber-GLP-29-L10-38-a1.0-ffn.gguf",
                       "DeepSeek-V4-Flash-Vision-Exp-abliterated-cyber-GLP-29-L10-38-a1.0.gguf"}
ALL_FILES = sorted({e["file"] for e in MATRIX.values()})


def build(entry):
    kind = entry["fake"][0]
    if kind == "qwen":
        return fakes.runner(width=entry["fake"][1])
    if kind == "glm":
        return fg.runner((45,), 0, hidden=entry["fake"][1])
    if kind == "dsv4":
        _, kw, hf = entry["fake"]
        return fd.runner(cfg=fd.config(**{"hidden": kw["hidden"], "hc_mult": kw["hc_mult"],
                                          "num_layers": kw["num_layers"],
                                          "model_type": kw.get("model_type", "deepseek_v4"),
                                          "v41": kw.get("v41", False)}),
                         partition=(kw["num_layers"],), hf_model_type=hf)
    _, path, names, cfg, hf = entry["fake"]
    return fa.runner(entry["row"], path, names, cfg, hf)


def _config_path(src):
    if src == "glm":
        p = os.environ.get("WEIGHTLESS_TEST_GLM_CONFIG", "").strip()
    elif src == "dsv41":
        p = os.environ.get("WEIGHTLESS_TEST_DSV41_CONFIG", "").strip()
    elif src == "qwen":
        p = os.environ.get("WEIGHTLESS_TEST_QWEN_CONFIG", "").strip()
    else:
        p = ""
        for d in os.environ.get("WEIGHTLESS_TEST_CONFIG_DIR", "").split(os.pathsep):
            if d.strip() and os.path.isdir(os.path.join(d.strip(), src)):
                p = os.path.join(d.strip(), src)
                break
    if p and os.path.isdir(p):
        p = os.path.join(p, "config.json")
    return p if p and os.path.isfile(p) else None


def _missing_row(entry):
    row = entry["row"]
    if entry["status"] == "refused" or row in ARCH:
        return None
    return f"row {row} ({entry['status']}) is not in this build's ARCH"


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)

    def install(self, run, path):
        os.environ["WEIGHTLESS_STEER_PATH"] = path
        with redirect_stderr(io.StringIO()) as err:
            rec = install_steering(run, source="env")
        self.log = err.getvalue()
        return rec


@unittest.skipUnless(glpfiles.GLP_DIRS, "set WEIGHTLESS_TEST_GLP_DIR to the folder of the published GLP files")
class Matrix(_Base):
    """test_g_<key>: one file on its row. test_x_<file>: that file on every
    other row this build serves. test_c_<key>: the fake's geometry against
    the model's real config."""

    def test_every_published_file_is_in_the_matrix(self):
        """The 15 published files, each at least once."""
        found = sorted(os.path.basename(p) for d in glpfiles.GLP_DIRS for p in _walk(d))
        self.assertEqual(sorted(set(found)), ALL_FILES)
        self.assertEqual(len(ALL_FILES), 15)

    def test_structure_rules_on_the_real_files(self):
        from weightless_steer.container import read_gguf_cvec
        for name in ALL_FILES:
            meta, _ = read_gguf_cvec(_need(self, name))
            s = meta.get("glp.structure")
            with self.subTest(name):
                if name in LOOPED_FILES:
                    self.assertEqual(s, "per-execution-step")
                elif name in NO_STRUCTURE:
                    self.assertIsNone(s)
                elif name in FREE_TEXT_STRUCTURE:
                    self.assertNotIn(s, (None, "per-layer", "per-execution-step"))
                else:
                    self.assertEqual(s, "per-layer")


def _walk(d):
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.join(root, f) for f in files if f.endswith(".gguf")]
    return out


def _need(case, name):
    p = glpfiles.find_glp(name)
    if p is None:
        case.skipTest(f"{name} is not under WEIGHTLESS_TEST_GLP_DIR")
    return p


def _make_g(key, entry):
    def test(self):
        why = _missing_row(entry)
        if why:
            self.skipTest(why)
        path = _need(self, entry["file"])
        from weightless_steer.container import read_gguf_cvec
        meta, _ = read_gguf_cvec(path)
        run = build(entry)
        exp = entry["expect"]
        if exp[0] == "refused":
            with self.assertRaisesRegex(RuntimeError, exp[1]):
                self.install(run, path)
            if entry["row"] in ARCH:
                bb = run.model.model
                self.assertFalse(hasattr(bb, "_steer_stack"), "refused before anything was hooked")
            else:
                self.assertIn(entry["row"], REFUSED)
            return
        _, ids, width, alpha = exp
        rec = self.install(run, path)
        self.assertEqual(rec["row"], entry["row"])
        self.assertEqual(rec["local_layer_ids"], ids)
        self.assertEqual(rec["width"], width)
        self.assertEqual(rec["alpha"], alpha)
        self.assertEqual(rec["hook_point"], meta["glp.hook_point"])
        self.assertIn(meta["controlvector.model_hint"],
                      set(rec["model_types"]) | set(ARCH[entry["row"]].hint))
        self.assertIn(rec["hook_point"], ARCH[entry["row"]].hooks)
        self.assertAlmostEqual(float(meta.get("glp.alpha_default", 1.0)), alpha)
        print(f"  {key}: {entry['file']} -> {entry['row']} {len(ids)} sites "
              f"{ids[0]}..{ids[-1]} width {width} alpha {alpha} hook {rec['hook_point']}")

    test.__doc__ = f"G: {entry['file']} on {entry['row']} ({entry['expect'][0]})"
    return test


# What a cross-check refusal may say: the core's width check, the model_hint
# and hook-point checks, the structure check, the layer class or range, or a
# row's named refusal. Anything else (a fake that does not build, a
# TypeError) is not a refusal and fails the test.
REFUSAL_FAMILY = (r"width \d+ != \d+|model_hint=|glp\.hook_point=.* does not serve"
                  r"|per-execution-step|glp\.structure|DeepSeek-V4\.1 file"
                  r"|residual-site file|plain DeepSeek-V4-Flash-Vision-Exp"
                  r"|not one of|out of range|matched no layers")

# Served classes that are one model under two names (the same row data).
ALIASES = ({"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"},
           {"KimiK3ForConditionalGeneration", "KimiK3LinearForCausalLM"},
           {"NemotronHForCausalLM", "NemotronHPuzzleForCausalLM"})


def _make_x(name):
    home = {e["row"] for e in MATRIX.values() if e["file"] == name and e["expect"][0] == "ok"}
    for group in ALIASES:
        if home & group:
            home |= group

    def test(self):
        path = _need(self, name)
        seen = 0
        for key, entry in MATRIX.items():
            row = entry["row"]
            if row in home or entry["status"] == "refused" or _missing_row(entry):
                continue
            if entry["fake"][0] == "dsv4" and entry["fake"][1].get("v41"):
                continue  # the V4.1 model is refused whatever the file
            with self.subTest(file=name, row=row, fake=key):
                run = build(entry)  # outside the refusal check: a broken fake is an error
                with self.assertRaisesRegex(RuntimeError, REFUSAL_FAMILY):
                    self.install(run, path)
                bb, _ = resolve_backbone(run.model, ARCH[row])
                self.assertFalse(hasattr(bb, "_steer_stack"), "refused before anything was hooked")
                seen += 1
        self.assertGreater(seen, 0)

    test.__doc__ = f"X: {name} is refused on every row but its own"
    return test


def _make_c(key, entry):
    def test(self):
        src, geom = entry["geometry"]
        p = _config_path(src)
        if p is None:
            self.skipTest(f"no config.json for {src} (WEIGHTLESS_TEST_CONFIG_DIR, "
                          f"WEIGHTLESS_TEST_GLM_CONFIG, WEIGHTLESS_TEST_DSV41_CONFIG)")
        raw = json.load(open(p))
        cfg = dict(raw)
        for sub in ("text_config", "language_config"):
            if isinstance(raw.get(sub), dict):
                cfg = {**raw, **raw[sub]}
        for k, v in geom.items():
            with self.subTest(attribute=k):
                if k == "model_type":  # the text model's type (the file's hint)
                    self.assertEqual((raw.get("text_config") or raw).get(k), v)
                elif k == "architectures":
                    self.assertEqual(raw[k], v)
                else:
                    self.assertEqual(cfg.get(k), v)
        # the file's layer range fits the model
        path = glpfiles.find_glp(entry["file"])
        if path is not None:
            from weightless_steer.container import read_gguf_cvec
            _, tensors = read_gguf_cvec(path)
            ids = sorted(int(k.split(".")[1]) for k in tensors if k.startswith("direction."))
            steps = int(cfg["num_hidden_layers"]) * int(cfg.get("num_loops") or
                                                        cfg.get("total_ut_steps") or 1)
            looped = entry["file"] in LOOPED_FILES
            self.assertLessEqual(max(ids), steps if looped else steps - 1)

    test.__doc__ = f"C: {key} geometry against {entry['geometry'][0]}"
    return test


def _ident(s):
    return "".join(c if c.isalnum() else "_" for c in s)


for _key, _entry in MATRIX.items():
    setattr(Matrix, f"test_g_{_ident(_key)}", _make_g(_key, _entry))
    if "geometry" in _entry:
        setattr(Matrix, f"test_c_{_ident(_key)}", _make_c(_key, _entry))
for _name in ALL_FILES:
    setattr(Matrix, f"test_x_{_ident(_name[:-5])}", _make_x(_name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
