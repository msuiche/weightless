"""SGLang itself builds each served model from its public config.json on
the meta device (no weights, no memory), CPU only, and the plugin installs
on it. The other test files stub SGLang or run parts of it; this one proves
that the real classes, built by SGLang's own ``__init__`` from the real
configs, resolve to the rows' backbones, carry the expected layer classes
and counts, and take every site of the published file (a synthetic file of
the same shape where the published one is not there).

What the builds need, and nothing more: a runtime context with an
unresolved ``ServerArgs(device="cpu")`` (resolving it looks for a GPU),
reset after each test, or the context's own override helpers; the parallel
fields the constructors read, given as one rank; the rotary embedding
factory (or its CPU kernel import) stubbed; the current CUDA device and
stream read as the meta device.

- ``MetaBuild``: GLM-5.3, Hy4-preview and Qwen3.8-Flash-Next.
- ``MetaBuildLooped``: Nemotron-3.5-Lightning (forward wrappers),
  Kimi-K3 (attention split) and Nanbeige4.2-3B (44 execution steps).
- ``MetaBuildDsv4``: DeepSeek-V4-Flash, with the cross-layer mHC fusion
  switch set both ways: on, the GLP-29 file installs on layers 10..38 at
  width 4096; off, the boot is refused with the unfused-mode message and
  nothing is hooked. The Vision-Exp config builds the same model, and its
  -ffn file installs.

Skips without WEIGHTLESS_TEST_CONFIG_DIR (a folder of
``<org>__<repo>/config.json``), without the published files where a test
needs them (WEIGHTLESS_TEST_GLP_DIR), or when this SGLang cannot build the
model.
"""
import contextlib
import dataclasses
import io
import json
import os
import re
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

from weightless_sglang.install import install_steering  # noqa: E402

CONFIG_DIR = os.environ.get("WEIGHTLESS_TEST_CONFIG_DIR", "").strip() or None


# ---------------------------------------------------------------- GLM-5.3, Hy4-preview, Qwen3.8-Flash-Next

ROWS = {
    # row: (config folder, hf model_type, file, depth, width, first/last steered, layer classes)
    "GlmMoeDsaForCausalLM": ("RadixArk__GLM-5.3-NVFP4", "glm_moe_dsa",
                             "GLM-5.3-abliterated-cyber-GLP-77-L1-77-a1.0.gguf", 78, 6144, 1.0,
                             {"DeepseekV2DecoderLayer"}),
    "HYV4ForCausalLM": ("tencent__Hy4-preview-FP8", "hy_v4",
                        "Hy4-preview-abliterated-cyber-GLP-77-L1-77-a2.0.gguf", 78, 6144, 2.0,
                        {"HYV4DecoderLayer"}),
    "Qwen4ExpForConditionalGeneration": ("Qwen__Qwen3.8-Flash-Next", "qwen4_exp",
                                         "Qwen3.8-Flash-Next-abliterated-cyber-GLP-47-L1-47-a1.gguf",
                                         48, 10240, 1.0,
                                         {"Qwen4ExpLinearDecoderLayer", "Qwen4ExpAttentionDecoderLayer"}),
}


class _OneRank:
    world_size, rank_in_group, rank, ranks = 1, 0, 0, [0]
    is_first_rank = is_last_rank = True
    device_group = cpu_group = None

    def all_reduce(self, x):
        return x


class _Rope(nn.Identity):
    def __init__(self, *a, **k):
        super().__init__()
        self.head_size = k.get("head_size", a[0] if a else 0)
        self.rotary_dim = k.get("rotary_dim", a[1] if len(a) > 1 else self.head_size)
        f = k.get("partial_rotary_factor", 1.0)
        if f < 1.0:
            self.rotary_dim = int(self.rotary_dim * f)


def _build(row, cfg_json):
    if row == "GlmMoeDsaForCausalLM":
        from transformers import PretrainedConfig
        from sglang.srt.models import glm4_moe as G4
        c = {k: v for k, v in cfg_json.items() if k != "quantization_config"}
        return G4.GlmMoeDsaForCausalLM(PretrainedConfig(**c))
    c = {k: v for k, v in cfg_json.items() if k not in ("architectures", "model_type", "quantization_config")}
    if row == "HYV4ForCausalLM":
        from sglang.srt.configs.hy_v4 import HYV4Config
        from sglang.srt.models import hunyuan_v4 as HY
        return HY.HYV4ForCausalLM(HYV4Config(**c))
    from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
    from sglang.srt.models import qwen4_exp as Q
    c["language_model_only"] = True  # the text model: no vision tower on the meta device
    return Q.Qwen4ExpForConditionalGeneration(Qwen4ExpConfig(**c))


def _import_or_why():
    try:
        from sglang.srt import runtime_context  # noqa: F401
        from sglang.srt.server_args import ServerArgs  # noqa: F401
        import sglang.srt.models.glm4_moe  # noqa: F401
        import sglang.srt.models.hunyuan_v4  # noqa: F401
        import sglang.srt.models.qwen4_exp  # noqa: F401
        import sglang.srt.configs.hy_v4  # noqa: F401
        import sglang.srt.configs.qwen4_exp  # noqa: F401
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


WHY = _import_or_why()


@unittest.skipUnless(CONFIG_DIR, "set WEIGHTLESS_TEST_CONFIG_DIR to a folder of <org>__<repo>/config.json")
@unittest.skipIf(WHY, f"SGLang's models or runtime context are not importable here ({WHY})")
class MetaBuild(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_context, reset_context
        from sglang.srt.server_args import ServerArgs
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        get_context().set_server_args(ServerArgs(model_path="unused", device="cpu"))
        self.addCleanup(reset_context)

    @contextlib.contextmanager
    def meta(self):
        with contextlib.ExitStack() as st:
            for m in list(sys.modules.values()):
                if getattr(m, "__name__", "").startswith("sglang"):
                    for n in ("get_rope", "get_rope_wrapper"):
                        if callable(getattr(m, n, None)):
                            st.enter_context(mock.patch.object(m, n, _Rope))
            st.enter_context(mock.patch.object(torch.cuda, "current_device", lambda: "meta"))
            st.enter_context(torch.device("meta"))
            yield

    def build(self, row, cfg_json):
        from sglang.srt.runtime_context import get_parallel
        ov = {}
        for _ in range(40):  # give each parallel field a constructor reads as one rank
            try:
                with get_parallel().override(**ov), self.meta():
                    return _build(row, cfg_json), sorted(ov)
            except RuntimeError as e:
                m = re.search(r"parallel name '(\w+)' has not been written", str(e))
                if not m:
                    raise
                n = m.group(1)
                ov[n] = _OneRank() if n.endswith("group") else 0 if n.endswith("rank") else 1
        raise AssertionError(f"too many parallel fields: {sorted(ov)}")

    def check(self, row):
        folder, hf, fname, depth, width, alpha, classes = ROWS[row]
        path = os.path.join(CONFIG_DIR, folder, "config.json")
        if not os.path.isfile(path):
            self.skipTest(f"{folder}/config.json is not in {CONFIG_DIR}")
        with open(path) as f:
            cfg_json = json.load(f)
        self.assertEqual(cfg_json["architectures"], [row])
        model, fields = self.build(row, cfg_json)
        self.assertEqual(type(model).__name__, row)
        bb = model.model
        self.assertEqual(len(bb.layers), depth)
        self.assertEqual((bb.start_layer, bb.end_layer), (0, depth))
        self.assertEqual({type(layer).__name__ for layer in bb.layers}, classes)
        glp = glpfiles.find_glp(fname)
        if glp is None:  # a synthetic file of the published one's shape
            m = good_meta(layers=range(1, depth), alpha=str(alpha))
            m["controlvector.model_hint"] = hf
            glp = os.path.join(self.tmp.name, fname)
            rng = np.random.default_rng(0)
            write_gguf(glp, m, {f"direction.{i}": (rng.standard_normal(width).astype(np.float32), 0)
                                for i in range(1, depth)})
        os.environ["WEIGHTLESS_STEER_PATH"] = glp
        run = types.SimpleNamespace(
            model=model, is_draft_worker=False, tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
            server_args=types.SimpleNamespace(enable_two_batch_overlap=False),
            model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=hf),
                                               dtype=torch.bfloat16))
        with redirect_stderr(io.StringIO()):
            rec = install_steering(run, source="env")
        self.assertEqual((rec["row"], rec["width"], rec["num_layers"], rec["alpha"]),
                         (row, width, depth, alpha))
        self.assertEqual(rec["local_layer_ids"], list(range(1, depth)))
        self.assertEqual(tuple(bb._steer_stack.shape), (depth, 1, width))
        self.assertEqual([i for i, layer in enumerate(bb.layers) if layer._forward_hooks],
                         list(range(1, depth)))
        print(f"  {row}: built {depth} x {sorted(classes)} from {folder}/config.json on meta "
              f"(one-rank fields {fields}); {os.path.basename(glp)} installed on layers 1..{depth - 1}, "
              f"width {width}, alpha {alpha}")

    def test_glm53(self):
        self.check("GlmMoeDsaForCausalLM")

    def test_hy4(self):
        self.check("HYV4ForCausalLM")

    def test_qwen38fn(self):
        self.check("Qwen4ExpForConditionalGeneration")


# ---------------------------------------------------------------- Nemotron-H, Kimi-K3, Nanbeige

CONFIG_DIRS = [d.strip() for d in os.environ.get("WEIGHTLESS_TEST_CONFIG_DIR", "").split(os.pathsep)
               if d.strip()]
NEM_FILE = glpfiles.find_glp("Nemotron-3.5-Lightning-30B-A3B-abliterated-GLP-51-L1-51-a1.0.gguf")
KIMI_FILE = glpfiles.find_glp("glp.kimi-k3-GLP-92-L1-92-a1.gguf")
NAN_FILE = glpfiles.find_glp("glp.nanbeige42-GLP-44-L1-44-a2.gguf")


def config(repo):
    for d in CONFIG_DIRS:
        p = os.path.join(d, repo, "config.json")
        if os.path.isfile(p):
            with open(p) as f:
                raw = json.load(f)
            raw.pop("quantization_config", None)
            return raw
    return None


NEM_CFG = config("nvidia__NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
KIMI_CFG = config("moonshotai__Kimi-K3")
NAN_CFG = config("Nanbeige__Nanbeige4.2-3B")

try:
    from sglang.srt import runtime_context as rc
    HAVE_ONE_RANK, ONE_RANK_WHY = hasattr(rc, "_parallel_fields") and hasattr(
        rc.get_context(), "override_server_args"), \
        "this SGLang has no runtime_context override helpers"
except Exception as e:
    HAVE_ONE_RANK, ONE_RANK_WHY = False, f"{type(e).__name__}: {e}"


@contextlib.contextmanager
def one_rank():
    grp = types.SimpleNamespace(
        world_size=1, rank_in_group=0, all_reduce=lambda x: x, ranks=[0], first_rank=0,
        last_rank=0, is_first_rank=True, is_last_rank=True, cpu_group=None,
        device_group=types.SimpleNamespace(rank=lambda: 0, size=lambda: 1))
    kw = dict(tp_size=1, tp_rank=0, attn_tp_rank=0, attn_tp_size=1, attn_dp_rank=0,
              attn_dp_size=1, attn_cp_rank=0, attn_cp_size=1, moe_tp_rank=0, moe_tp_size=1,
              moe_ep_rank=0, moe_ep_size=1, moe_dp_rank=0, moe_dp_size=1, launch_world_rank=0)
    fields = rc._parallel_fields()
    kw.update({f: grp for f in fields if f.endswith("_group")})
    ops = types.ModuleType("vllm._custom_ops")
    ops.rotary_embedding = lambda *a, **k: None
    stub = {}
    try:
        import vllm._custom_ops  # noqa: F401
    except Exception:
        v = types.ModuleType("vllm")
        v._custom_ops = ops
        stub = {"vllm": v, "vllm._custom_ops": ops}
    with rc.get_context().override_server_args(tp_size=1, pp_size=1), \
            rc.get_parallel().override(**{k: v for k, v in kw.items() if k in fields}), \
            mock.patch.dict(sys.modules, stub), \
            mock.patch.object(torch.cuda, "Stream", lambda *a, **k: None), \
            torch.device("meta"):
        yield


def steer(model, cfg, path):
    run = types.SimpleNamespace(model=model, is_draft_worker=False, tp_rank=0, tp_size=1,
                                pp_rank=0, pp_size=1,
                                model_config=types.SimpleNamespace(hf_config=cfg,
                                                                   dtype=torch.bfloat16))
    with mock.patch.dict(os.environ, {"WEIGHTLESS_STEER_PATH": path}, clear=False):
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_") and
                  k != "WEIGHTLESS_STEER_PATH"]:
            os.environ.pop(k)
        with contextlib.redirect_stderr(io.StringIO()):
            return install_steering(run, source="env")


def kinds(layers):
    out = {}
    for layer in layers:
        out[type(layer).__name__] = out.get(type(layer).__name__, 0) + 1
    return out


@unittest.skipUnless(HAVE_ONE_RANK, ONE_RANK_WHY)
class MetaBuildLooped(unittest.TestCase):
    @unittest.skipUnless(NAN_CFG and glpfiles.have(NAN_FILE),
                         "set WEIGHTLESS_TEST_CONFIG_DIR (Nanbeige__Nanbeige4.2-3B) and "
                         "WEIGHTLESS_TEST_GLP_DIR (GLP-44)")
    def test_nanbeige(self):
        from sglang.srt.configs import NanbeigeConfig
        from sglang.srt.models.nanbeige import NanbeigeForCausalLM
        cfg = NanbeigeConfig(**NAN_CFG)
        with one_rank():
            model = NanbeigeForCausalLM(config=cfg)
        self.assertEqual(kinds(model.model.layers), {"NanbeigeDecoderLayer": 22})
        rec = steer(model, cfg, NAN_FILE)
        self.assertEqual((rec["row"], rec["install"], rec["exec_id"]),
                         ("NanbeigeForCausalLM", "hook", "looped"))
        self.assertEqual(rec["local_layer_ids"], list(range(44)))
        self.assertEqual(rec["physical_layers"], list(range(22)))
        self.assertEqual((rec["width"], rec["alpha"], rec["model_hint"]), (3072, 2.0, "nanbeige"))
        self.assertEqual(sum(len(l._forward_hooks) for l in model.model.layers), 22)

    @unittest.skipUnless(NEM_CFG and glpfiles.have(NEM_FILE),
                         "set WEIGHTLESS_TEST_CONFIG_DIR (nvidia__NVIDIA-Nemotron-3.5-Lightning-"
                         "30B-A3B-NVFP4) and WEIGHTLESS_TEST_GLP_DIR (GLP-51)")
    def test_nemotron(self):
        from sglang.srt.configs import NemotronHConfig
        from sglang.srt.models import nemotron_h as NH
        cfg = NemotronHConfig(**NEM_CFG)
        for cls in (NH.NemotronHForCausalLM, NH.NemotronHPuzzleForCausalLM):
            with self.subTest(cls=cls.__name__):
                with one_rank():
                    model = cls(config=cfg)
                bb = model.model
                self.assertEqual(kinds(bb.layers), {"NemotronHMambaDecoderLayer": 23,
                                                    "NemotronHMoEDecoderLayer": 23,
                                                    "NemotronHAttentionDecoderLayer": 6})
                rec = steer(model, cfg, NEM_FILE)
                self.assertEqual((rec["row"], rec["install"]), (cls.__name__, "forward_wrap"))
                self.assertEqual(rec["local_layer_ids"], list(range(1, 52)))
                self.assertEqual((rec["width"], rec["alpha"], rec["model_hint"]),
                                 (2688, 1.0, "nemotron_h"))
                self.assertEqual([i for i, l in enumerate(bb.layers) if "forward" in vars(l)],
                                 list(range(1, 52)))

    @unittest.skipUnless(KIMI_CFG and glpfiles.have(KIMI_FILE),
                         "set WEIGHTLESS_TEST_CONFIG_DIR (moonshotai__Kimi-K3) and "
                         "WEIGHTLESS_TEST_GLP_DIR (GLP-92)")
    def test_kimi_text_model(self):
        from sglang.srt.configs.kimi_linear import KimiLinearConfig
        from sglang.srt.models.kimi_k3 import KimiK3LinearForCausalLM
        text = dict(KIMI_CFG["text_config"])
        text.pop("quantization_config", None)
        cfg = KimiLinearConfig(**text)
        with one_rank():
            model = KimiK3LinearForCausalLM(config=cfg)
        bb = model.model
        self.assertEqual(kinds(bb.layers), {"KimiK3DecoderLayer": 93})
        rec = steer(model, cfg, KIMI_FILE)
        self.assertEqual((rec["row"], rec["install"]), ("KimiK3LinearForCausalLM", "hook"))
        self.assertEqual(rec["local_layer_ids"], list(range(1, 93)))
        self.assertEqual((rec["width"], rec["alpha"], rec["model_hint"]),
                         (7168, 1.0, "kimi_linear"))
        full = [i for i in range(1, 93) if i + 1 in text["linear_attn_config"]["full_attn_layers"]]
        self.assertEqual(rec["full_attn_layers"], full)
        self.assertEqual(len(full), 24)


# ---------------------------------------------------------------- DeepSeek-V4-Flash

DEPTH, WIDTH = 43, 4096
STEERED = list(range(10, 39))  # GLP-29: layers 10-38

CONFIGS = {
    # config folder: (published -ffn file, alpha_default)
    "deepseek-ai__DeepSeek-V4-Flash-0731": (
        "DeepSeek-V4-Flash-0731-abliterated-cyber-GLP-29-L10-38-a4.gguf", 6.0),
    "deepseek-ai__DeepSeek-V4-Flash-Vision-Exp": (
        "DeepSeek-V4-Flash-Vision-Exp-abliterated-cyber-GLP-29-L10-38-a1.0-ffn.gguf", 1.0),
}


def _dsv4_import_or_why():
    try:
        from sglang.srt import runtime_context  # noqa: F401
        from sglang.srt.server_args import ServerArgs  # noqa: F401
        from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config  # noqa: F401
        import sglang.srt.models.deepseek_v4 as D
        if not callable(getattr(D, "is_cross_layer_mhc_fusion_enabled", None)):
            return "models/deepseek_v4.py has no is_cross_layer_mhc_fusion_enabled"
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


DSV4_WHY = _dsv4_import_or_why()


def _dsv4_config(cfg_json):
    """DeepSeekV4Config's field defaults, the published config.json on top."""
    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
    out = {}
    for f in dataclasses.fields(DeepSeekV4Config):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            out[f.name] = f.default_factory()
        else:
            out[f.name] = None
    out.update({k: v for k, v in cfg_json.items() if v is not None and k != "quantization_config"})
    for f in dataclasses.fields(DeepSeekV4Config):  # typed containers left unset
        if out.get(f.name) is None:
            if "list" in str(f.type).lower():
                out[f.name] = []
            elif "dict" in str(f.type).lower():
                out[f.name] = {}
    return types.SimpleNamespace(**out)


@unittest.skipUnless(CONFIG_DIR, "set WEIGHTLESS_TEST_CONFIG_DIR to a folder of <org>__<repo>/config.json")
@unittest.skipIf(DSV4_WHY, f"SGLang's DeepSeek-V4 model or runtime context is not importable here ({DSV4_WHY})")
class MetaBuildDsv4(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_context, reset_context
        from sglang.srt.server_args import ServerArgs
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p = mock.patch.dict(os.environ, {}, clear=False)
        p.start()
        self.addCleanup(p.stop)
        for k in [k for k in os.environ if k.startswith("WEIGHTLESS_STEER_")]:
            os.environ.pop(k)
        get_context().set_server_args(ServerArgs(model_path="unused", device="cpu"))
        self.addCleanup(reset_context)

    @contextlib.contextmanager
    def meta(self, fused):
        import sglang.srt.models.deepseek_v4 as D
        with contextlib.ExitStack() as st:
            for m in list(sys.modules.values()):
                if getattr(m, "__name__", "").startswith("sglang"):
                    for n in ("get_rope", "get_rope_wrapper"):
                        if callable(getattr(m, n, None)):
                            st.enter_context(mock.patch.object(m, n, _Rope))
            st.enter_context(mock.patch.object(D, "is_cross_layer_mhc_fusion_enabled",
                                               lambda: fused))
            st.enter_context(mock.patch.object(torch.cuda, "current_device", lambda: "meta"))
            st.enter_context(mock.patch.object(torch.cuda, "Stream", lambda *a, **k: None))
            st.enter_context(torch.device("meta"))
            yield

    def build(self, cfg, fused):
        import sglang.srt.models.deepseek_v4 as D
        from sglang.srt.runtime_context import get_parallel
        ov = {}
        for _ in range(60):  # give each parallel field a constructor reads as one rank
            try:
                with get_parallel().override(**ov), self.meta(fused):
                    return D.DeepseekV4ForCausalLM(cfg)
            except RuntimeError as e:
                m = re.search(r"parallel name '(\w+)' has not been written", str(e))
                if not m:
                    raise
                n = m.group(1)
                ov[n] = _OneRank() if n.endswith("group") else 0 if n.endswith("rank") else 1
        raise AssertionError(f"too many parallel fields: {sorted(ov)}")

    def glp(self, fname, alpha):
        path = glpfiles.find_glp(fname)
        if path is None:  # a synthetic file of the published one's shape
            m = good_meta(layers=STEERED, alpha=str(alpha))
            m["glp.hook_point"] = m["glp.derived_at"] = "ffn_out_pre_residual"
            m["controlvector.model_hint"] = "deepseek_v4"
            path = os.path.join(self.tmp.name, fname)
            rng = np.random.default_rng(0)
            write_gguf(path, m, {f"direction.{i}": (rng.standard_normal(WIDTH).astype(np.float32), 0)
                                 for i in STEERED})
        return path

    def check(self, folder):
        path = os.path.join(CONFIG_DIR, folder, "config.json")
        if not os.path.isfile(path):
            self.skipTest(f"{folder}/config.json is not in {CONFIG_DIR}")
        with open(path) as f:
            cfg_json = json.load(f)
        self.assertEqual((cfg_json["architectures"], cfg_json["model_type"]),
                         (["DeepseekV4ForCausalLM"], "deepseek_v4"))
        cfg = _dsv4_config(cfg_json)
        fname, alpha = CONFIGS[folder]
        os.environ["WEIGHTLESS_STEER_PATH"] = self.glp(fname, alpha)
        for fused in (True, False):
            with self.subTest(folder=folder, fused=fused):
                model = self.build(cfg, fused)
                bb = model.model
                self.assertEqual(type(bb).__name__, "DeepseekV4Model")
                self.assertEqual(len(bb.layers), DEPTH)
                self.assertEqual((bb.start_layer, bb.end_layer), (0, DEPTH))
                self.assertEqual({type(layer).__name__ for layer in bb.layers},
                                 {"DeepseekV4DecoderLayer"})
                self.assertEqual({layer.hidden_size for layer in bb.layers}, {WIDTH})
                # what the real constructor sets, and what the preflight reads
                self.assertIs(bool(bb.hc_pre_from_prev_sublayer), False)
                self.assertIs(bool(bb.use_fused_mhc_post_pre), fused)
                self.assertEqual({bool(layer.use_fused_mhc_post_pre) for layer in bb.layers},
                                 {fused})
                run = types.SimpleNamespace(
                    model=model, is_draft_worker=False, tp_rank=0, tp_size=1, pp_rank=0,
                    pp_size=1, server_args=types.SimpleNamespace(enable_two_batch_overlap=False),
                    model_config=types.SimpleNamespace(hf_config=cfg, dtype=torch.bfloat16))
                if not fused:
                    with self.assertRaisesRegex(RuntimeError, "unfused mHC mode"):
                        with redirect_stderr(io.StringIO()):
                            install_steering(run, source="env")
                    self.assertFalse(hasattr(bb, "_steer_stack"))
                    continue
                with redirect_stderr(io.StringIO()):
                    rec = install_steering(run, source="env")
                self.assertEqual((rec["row"], rec["width"], rec["num_layers"], rec["alpha"],
                                  rec["hook_point"]),
                                 ("DeepseekV4ForCausalLM", WIDTH, DEPTH, alpha,
                                  "ffn_out_pre_residual"))
                self.assertEqual(rec["local_layer_ids"], STEERED)
                self.assertEqual(tuple(bb._steer_stack.shape), (DEPTH, 1, WIDTH))
                for undo in run._weightless_steer_handles:
                    undo()
                if run._weightless_steer_undo_check:
                    run._weightless_steer_undo_check()
        print(f"  DeepseekV4ForCausalLM from {folder}/config.json: {DEPTH} x DeepseekV4DecoderLayer "
              f"on meta; fused mode installs {fname} on layers 10..38 at {WIDTH}, alpha {alpha}; "
              f"unfused mode refused")

    def test_dsv4_0731(self):
        self.check("deepseek-ai__DeepSeek-V4-Flash-0731")

    def test_dsv4_vision_exp(self):
        self.check("deepseek-ai__DeepSeek-V4-Flash-Vision-Exp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
