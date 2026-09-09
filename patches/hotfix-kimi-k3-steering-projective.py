#!/usr/bin/env python3
# weightless lane copy of refusal-research/experiments/20260903-kimi-k3-glp/
# staging/patch_kimi_k3.py (validated over 16 boots, capture + 4 eval arms on
# 2x H200:8 PP=2xTP=8). Keep byte-compatible with the experiment copy: the
# DSPARK_PROBE_* envs arm the capture probe only; serving uses
# WEIGHTLESS_STEER_PATH / WEIGHTLESS_STEER_ALPHA. Structure test:
# tests/structure/test-k3-steering-structure.py
"""Fail-closed runtime patch: CUDA-graph-safe activation probe AND weightless
projective steering for vLLM's kimi_k3 NVIDIA model inside the day-0 container
(vllm/vllm-openai:kimi-k3), TP=16 across 2 clustered H200:8 nodes.

Modelled on patch_hy4.py (experiments/20260902-hy4-preview-glp), redesigned
per refusal-research/experiments/20260903-kimi-k3-glp/DIRECTIVES.md: NO eager
anywhere in the K3 lane. The probe is a preallocated GPU buffer written with
pure tensor ops inside KimiLinearModel.forward (graph-capturable: index_copy_
into a slot selected by a device tensor, scaled by a gate tensor); the patched
gpu_model_runner arms slot/gate per step and flushes the slot to disk AFTER
the step, outside the graph. No Python I/O and no .cpu() copies mid-forward.

Tap point (STATE.md §4, verified against the installed source): in the
KimiLinearModel.forward layer loop the post-layer accumulated stream is
prefix_sum + hidden_states, plain [T, 7168] (attn_res carries the side
stream in `residual`; there is no HC widening). Steering projects the same
materialized stream, one 7168-wide direction per layer, alpha as a tensor
buffer (compile-safe hotfix discipline, same as patch_hy4.py).

Two files are patched, both fail-closed:
  1. vllm/models/kimi_k3/nvidia/model.py     (buffer probe + steering)
  2. vllm/v1/worker/gpu_model_runner.py      (per-step gate + post-step flush)

Env vars (probe):
  DSPARK_PROBE_LAYERS      comma list of layer ids, e.g. "0,1,...,92"
  DSPARK_PROBE_DUMP_DIR    directory for probe_%06d.pt files
  DSPARK_PROBE_MIN_TOKENS  skip steps shorter than this (default 4)
  DSPARK_PROBE_MAX_TOKENS  skip steps longer than this (default 4096;
                           filters the memory-profile dummy run)

Env vars (steering):
  WEIGHTLESS_STEER_PATH    .pt file: {layer: tensor(7168)} or
                           {"per_layer": {...}}; fail-closed (engine boot
                           raises) if set but missing, malformed, wrong
                           width, or matches no layer
  WEIGHTLESS_STEER_ALPHA   float, default 1.0; alpha=0 with PATH set is the
                           no-op arm (hooks installed, projection x 0)
  WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids

Capture gating (runner, outside the graph): a step is captured iff exactly
one request is scheduled, all its tokens are scheduled in this step
(singleton prefill, no chunking), and the token count is inside
[MIN, MAX]. Decode steps (T=1), batched prefills and mixed steps are never
captured. Dump k <-> request k holds by FCFS with a strictly serial driver
and is verified afterwards via n_tokens in each dump (REFUSAL.md 3ak).

Dump format per captured prefill (TP rank 0 only):
  {"act_mean": [n_layers, 7168] fp32 mean over token positions,
   "act_last": [n_layers, 7168] fp32 last token position,
   "layers": sorted layer ids, "n_tokens": int}

Exit codes: 0 patched (or already patched), 1 anchor mismatch (image drifted
from the extracted reference (staging/vsrc, staging/vsrc2); do NOT proceed).
"""

import os as _os
import py_compile
import sys

# Overridable for dry-runs against copies outside the container.
MODEL_FILE = _os.environ.get(
    "DSPARK_K3_MODEL_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "kimi_k3/nvidia/model.py")
RUNNER_FILE = _os.environ.get(
    "DSPARK_K3_RUNNER_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/"
    "gpu_model_runner.py")
MLA_FILE = _os.environ.get(
    "DSPARK_K3_MLA_PY",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
    "layers/attention/mla_attention.py")
MARKER = "dspark k3 probe"


def _replace_once(src: str, anchor: str, replacement: str, name: str,
                  path: str) -> str:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(
            f"PATCH FAILED: anchor {name!r} found {n} times (expected 1) in "
            f"{path}. The image has drifted from the extracted reference; "
            "refusing to guess. Diff and update this patch."
        )
    return src.replace(anchor, replacement)


def patch_model(path: str) -> None:
    src = open(path).read()
    if MARKER in src:
        print(f"already patched: {path}")
        return

    # 1. imports -------------------------------------------------------------
    src = _replace_once(
        src,
        "import math\nfrom collections.abc import Iterable",
        "import math\nimport os\nfrom collections.abc import Iterable",
        "imports", path)

    # 2. module-level registry + flush helper --------------------------------
    src = _replace_once(
        src,
        'logger = init_logger(__name__)',
        'logger = init_logger(__name__)\n'
        '\n'
        '\n'
        '# ---- dspark k3 probe: live KimiLinearModel instances, keyed by\n'
        '# worker process. The patched gpu_model_runner looks the probe up\n'
        '# here to arm the per-step gate and to flush after the step.\n'
        'DSPARK_K3_REGISTRY: list = []\n'
        '\n'
        '\n'
        'def _dspark_k3_flush(owner, n_tokens: int) -> None:\n'
        '    """Dump the probe work slot to disk. Called by the patched\n'
        '    gpu_model_runner AFTER the step, outside any CUDA graph -- the\n'
        '    .cpu() copy and torch.save are illegal mid-forward under\n'
        '    capture (DIRECTIVES.md). Under PP=2 each stage\'s TP-rank-0\n'
        '    flushes its own layer range, so the filename carries the PP\n'
        '    rank (the volume is shared between nodes)."""\n'
        '    buf = owner._probe_buf[1].detach().float().cpu()\n'
        '    os.makedirs(owner._probe_dump_dir, exist_ok=True)\n'
        '    _pp = get_pp_group().rank_in_group\n'
        '    torch.save(\n'
        '        {\n'
        '            "act_mean": buf[:, 0].contiguous(),  # [L, hidden]\n'
        '            "act_last": buf[:, 1].contiguous(),  # [L, hidden]\n'
        '            "layers": list(owner._probe_layers),\n'
        '            "n_tokens": int(n_tokens),\n'
        '        },\n'
        '        os.path.join(owner._probe_dump_dir,\n'
        '                     "probe_%d_%06d.pt" % (_pp, owner._probe_seq)),\n'
        '    )\n'
        '    owner._probe_seq += 1',
        "registry", path)

    # 3. probe + steering state in KimiLinearModel.__init__ -------------------
    src = _replace_once(
        src,
        "        world_size = get_tensor_model_parallel_world_size()\n"
        "        assert config.num_attention_heads % world_size == 0, (\n"
        "            \"num_attention_heads must be divisible by world_size\"\n"
        "        )\n",
        "        world_size = get_tensor_model_parallel_world_size()\n"
        "        assert config.num_attention_heads % world_size == 0, (\n"
        "            \"num_attention_heads must be divisible by world_size\"\n"
        "        )\n"
        "\n"
        "        # ---- dspark k3 probe (capture lane) --------------------\n"
        "        # CUDA-graph-safe buffer probe (DIRECTIVES.md): pure\n"
        "        # tensor ops inside forward write per-layer prefill stats\n"
        "        # into a preallocated buffer; the patched gpu_model_runner\n"
        "        # arms slot/gate per step and flushes AFTER the step,\n"
        "        # outside the graph. Inert unless DSPARK_PROBE_LAYERS set.\n"
        "        _ds_spec = os.environ.get(\"DSPARK_PROBE_LAYERS\", \"\")."
        "strip()\n"
        "        _ds_ids = []\n"
        "        if _ds_spec:\n"
        "            for _tok in _ds_spec.replace(\" \", \"\").split(\",\"):\n"
        "                if _tok:\n"
        "                    _lid = int(_tok)\n"
        "                    if 0 <= _lid < config.num_hidden_layers:\n"
        "                        _ds_ids.append(_lid)\n"
        "        # PP: each stage probes only its local layers (no-op at\n"
        "        # PP=1). The runner flushes per stage with a PP-tagged\n"
        "        # filename; the driver/derive steps merge by layer id.\n"
        "        _ds_ids = [i for i in _ds_ids\n"
        "                   if self.start_layer <= i < self.end_layer]\n"
        "        self._probe_layers = sorted(set(_ds_ids))\n"
        "        self._probe_layer_set = frozenset(self._probe_layers)\n"
        "        self._probe_dump_dir = os.environ.get(\n"
        "            \"DSPARK_PROBE_DUMP_DIR\", \"\").strip()\n"
        "        self._probe_min_tokens = int(\n"
        "            os.environ.get(\"DSPARK_PROBE_MIN_TOKENS\", \"4\") or 4)\n"
        "        self._probe_max_tokens = int(\n"
        "            os.environ.get(\"DSPARK_PROBE_MAX_TOKENS\", \"4096\") or "
        "4096)\n"
        "        self._probe_seq = 0\n"
        "        if self._probe_layer_set:\n"
        "            if self.use_sequence_parallel:\n"
        "                raise RuntimeError(\n"
        "                    \"dspark k3 probe: sequence-parallel streams \"\n"
        "                    \"are sharded; probe supports plain TP only\")\n"
        "            # slot 0 = trash (decode replays / ungated steps write\n"
        "            # zeros here), slot 1 = work slot flushed by the runner.\n"
        "            self.register_buffer(\n"
        "                \"_probe_buf\",\n"
        "                torch.zeros(2, len(self._probe_layers), 2,\n"
        "                            config.hidden_size, dtype=torch.float32),\n"
        "                persistent=False,\n"
        "            )\n"
        "            self.register_buffer(\n"
        "                \"_probe_gate\", torch.zeros((), dtype=torch.float32),\n"
        "                persistent=False)\n"
        "            self.register_buffer(\n"
        "                \"_probe_slot\", torch.zeros(1, dtype=torch.long),\n"
        "                persistent=False)\n"
        "            self._probe_views = [\n"
        "                self._probe_buf[:, i]\n"
        "                for i in range(len(self._probe_layers))]\n"
        "            self._probe_pos = {\n"
        "                lid: i for i, lid in enumerate(self._probe_layers)}\n"
        "            DSPARK_K3_REGISTRY.append(self)\n"
        "            logger.info(\n"
        "                \"dspark k3 probe active: %d layers -> %s \"\n"
        "                \"(buffer probe, CUDA-graph-safe)\",\n"
        "                len(self._probe_layers),\n"
        "                self._probe_dump_dir or \"(no dump dir!)\",\n"
        "            )\n"
        "\n"
        "        # ---- weightless projective steering (eval lane) --------\n"
        "        # h <- h - alpha * (h . d) d on the post-layer stream\n"
        "        # prefix_sum + hidden_states [T, hidden] -- the same point\n"
        "        # the probe captures. Inert unless WEIGHTLESS_STEER_PATH is\n"
        "        # set. Dense zero-padded stack indexed by layer id (zero\n"
        "        # rows are a numeric no-op) and alpha as a tensor buffer:\n"
        "        # the compile-safe hotfix discipline.\n"
        "        self._steer_alpha_val = float(\n"
        "            os.environ.get(\"WEIGHTLESS_STEER_ALPHA\", \"1.0\") or 1.0)\n"
        "        _ds_dtype = vllm_config.model_config.dtype\n"
        "        self.register_buffer(\n"
        "            \"_steer_stack\",\n"
        "            torch.zeros(config.num_hidden_layers, config.hidden_size,\n"
        "                        dtype=_ds_dtype),\n"
        "            persistent=False,\n"
        "        )\n"
        "        self.register_buffer(\n"
        "            \"_steer_alpha\",\n"
        "            torch.zeros((), dtype=_ds_dtype),\n"
        "            persistent=False,\n"
        "        )\n"
        "        self._steer_layers = ()\n"
        "        _steer_path = os.environ.get(\"WEIGHTLESS_STEER_PATH\", \"\")."
        "strip()\n"
        "        if _steer_path:\n"
        "            try:\n"
        "                _raw = torch.load(_steer_path, map_location=\"cpu\",\n"
        "                                  weights_only=True)\n"
        "            except Exception:\n"
        "                _raw = torch.load(_steer_path, map_location=\"cpu\",\n"
        "                                  weights_only=False)\n"
        "            if isinstance(_raw, dict) and isinstance(\n"
        "                    _raw.get(\"per_layer\"), dict):\n"
        "                _raw = _raw[\"per_layer\"]\n"
        "            if not isinstance(_raw, dict):\n"
        "                raise RuntimeError(\n"
        "                    f\"WEIGHTLESS_STEER_PATH={_steer_path}: expected \"\n"
        "                    \"{layer: tensor} or {'per_layer': {...}}\")\n"
        "            _want = os.environ.get(\n"
        "                \"WEIGHTLESS_STEER_LAYERS\", \"\").strip()\n"
        "            _selected = (\n"
        "                {int(t) for t in _want.replace(\" \", \"\").split(\",\")}\n"
        "                if _want else None)\n"
        "            _dirs = {}\n"
        "            for _k, _v in _raw.items():\n"
        "                _lid = int(_k)\n"
        "                if not (0 <= _lid < config.num_hidden_layers):\n"
        "                    continue\n"
        "                if _selected is not None and _lid not in _selected:\n"
        "                    continue\n"
        "                _v = _v.detach().to(torch.float32).reshape(-1)\n"
        "                if _v.numel() != config.hidden_size:\n"
        "                    raise RuntimeError(\n"
        "                        f\"steering vector layer {_lid} width \"\n"
        "                        f\"{_v.numel()} != {config.hidden_size} \"\n"
        "                        \"(hidden_size)\")\n"
        "                _dirs[_lid] = _v / (_v.norm() + 1e-9)\n"
        "            if not _dirs:\n"
        "                raise RuntimeError(\n"
        "                    f\"WEIGHTLESS_STEER_PATH={_steer_path} matched no \"\n"
        "                    \"layers; refusing to run unsteered\")\n"
        "            for _lid, _v in _dirs.items():\n"
        "                self._steer_stack[_lid] = _v.to(_ds_dtype)\n"
        "            self._steer_alpha.fill_(self._steer_alpha_val)\n"
        "            self._steer_layers = tuple(sorted(_dirs))\n"
        "            logger.info(\n"
        "                \"weightless steering active: alpha=%.3f layers=%d..%d \"\n"
        "                \"(%d) width=%d\",\n"
        "                self._steer_alpha_val, self._steer_layers[0],\n"
        "                self._steer_layers[-1], len(self._steer_layers),\n"
        "                config.hidden_size)\n",
        "init", path)

    # 4. probe write + steering in the layer loop -----------------------------
    src = _replace_once(
        src,
        "        for layer_idx, layer in enumerate(\n"
        "            self.layers[self.start_layer : self.end_layer],\n"
        "            start=self.start_layer,\n"
        "        ):\n"
        "            hidden_states, prefix_sum, residual = layer(\n"
        "                positions=positions,\n"
        "                hidden_states=hidden_states,\n"
        "                prefix_sum=prefix_sum,\n"
        "                residual=residual,\n"
        "            )\n",
        "        for layer_idx, layer in enumerate(\n"
        "            self.layers[self.start_layer : self.end_layer],\n"
        "            start=self.start_layer,\n"
        "        ):\n"
        "            hidden_states, prefix_sum, residual = layer(\n"
        "                positions=positions,\n"
        "                hidden_states=hidden_states,\n"
        "                prefix_sum=prefix_sum,\n"
        "                residual=residual,\n"
        "            )\n"
        "            # ---- dspark k3 probe + weightless steering -----------\n"
        "            # Post-layer stream with attn_res on is\n"
        "            # prefix_sum + hidden_states (plain [T, hidden]); the\n"
        "            # attn_res side stream rides in `residual`. Probe\n"
        "            # records the PRE-steer stream when both are on. All\n"
        "            # ops below are pure tensor ops on preallocated\n"
        "            # buffers: CUDA-graph-capturable (DIRECTIVES.md).\n"
        "            if (\n"
        "                self._probe_layer_set\n"
        "                and layer_idx in self._probe_layer_set\n"
        "            ):\n"
        "                _ds_x = (\n"
        "                    (prefix_sum + hidden_states)\n"
        "                    if prefix_sum is not None\n"
        "                    else (hidden_states + residual)\n"
        "                )\n"
        "                _ds_stats = torch.stack(\n"
        "                    (_ds_x.float().mean(dim=0), _ds_x.float()[-1]))\n"
        "                self._probe_views[self._probe_pos[layer_idx]]."
        "index_copy_(\n"
        "                    0, self._probe_slot,\n"
        "                    (_ds_stats * self._probe_gate).unsqueeze(0),\n"
        "                )\n"
        "            if layer_idx in self._steer_layers:\n"
        "                _ds_h = (\n"
        "                    (prefix_sum + hidden_states)\n"
        "                    if prefix_sum is not None\n"
        "                    else (hidden_states + residual)\n"
        "                )\n"
        "                _ds_d = self._steer_stack[layer_idx]\n"
        "                _ds_h = _ds_h - self._steer_alpha * (\n"
        "                    _ds_h @ _ds_d).unsqueeze(-1) * _ds_d\n"
        "                hidden_states = (\n"
        "                    (_ds_h - prefix_sum)\n"
        "                    if prefix_sum is not None\n"
        "                    else (_ds_h - residual)\n"
        "                )\n",
        "capture", path)

    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"patched OK: {path}")


def patch_runner(path: str) -> None:
    src = open(path).read()
    if MARKER in src:
        print(f"already patched: {path}")
        return

    # 1. per-step gate, right before the model call ---------------------------
    src = _replace_once(
        src,
        "            model_output = self._model_forward(",
        "            # ---- dspark k3 probe: arm the capture gate ---------\n"
        "            # Runs OUTSIDE the CUDA graph (execute_model context).\n"
        "            # A step is captured iff it is a singleton prefill:\n"
        "            # exactly one request, all its tokens scheduled now,\n"
        "            # token count inside the probe window. Decode steps\n"
        "            # (T=1), batched or mixed steps are never captured.\n"
        "            _ds_owner = None\n"
        "            _ds_cap = False\n"
        "            try:\n"
        "                from vllm.models.kimi_k3.nvidia import (\n"
        "                    model as _ds_k3m,\n"
        "                )\n"
        "                if _ds_k3m.DSPARK_K3_REGISTRY:\n"
        "                    _ds_owner = _ds_k3m.DSPARK_K3_REGISTRY[0]\n"
        "            except Exception:\n"
        "                _ds_owner = None\n"
        "            if _ds_owner is not None and _ds_owner._probe_layer_set:\n"
        "                _ds_t = int(num_tokens_unpadded)\n"
        "                _ds_cap = (\n"
        "                    num_reqs == 1\n"
        "                    and int(max_num_scheduled_tokens) == _ds_t\n"
        "                    and _ds_owner._probe_min_tokens\n"
        "                    <= _ds_t\n"
        "                    <= _ds_owner._probe_max_tokens\n"
        "                )\n"
        "                _ds_owner._probe_gate.fill_(1.0 if _ds_cap else 0.0)\n"
        "                _ds_owner._probe_slot.fill_(1 if _ds_cap else 0)\n"
        "\n"
        "            model_output = self._model_forward(",
        "gate", path)

    # 2. post-step flush, right after the forward context closes --------------
    src = _replace_once(
        src,
        '        with record_function_or_nullcontext('
        '"gpu_model_runner: postprocess"):',
        "        # ---- dspark k3 probe: flush the capture slot ------------\n"
        "        # After the step, outside the graph: .cpu() + torch.save\n"
        "        # are legal here (and only here).\n"
        "        if (\n"
        "            _ds_cap\n"
        "            and _ds_owner is not None\n"
        "            and _ds_owner._probe_dump_dir\n"
        "            and get_tp_group().rank_in_group == 0\n"
        "        ):\n"
        "            try:\n"
        "                from vllm.models.kimi_k3.nvidia import (\n"
        "                    model as _ds_k3m,\n"
        "                )\n"
        "                _ds_k3m._dspark_k3_flush(\n"
        "                    _ds_owner, int(num_tokens_unpadded))\n"
        "            except Exception as exc:  # never take the server down\n"
        "                logger.warning(\n"
        "                    \"dspark k3 probe flush failed: %s\", exc)\n"
        "\n"
        '        with record_function_or_nullcontext('
        '"gpu_model_runner: postprocess"):',
        "flush", path)

    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"patched OK: {path}")


def patch_mla(path: str) -> None:
    """Third patch target (boot #14 root cause): the MLA layer lazily
    resolves the impl's dcp_world_size sentinel (-1) from the DCP group on
    the first real forward. Under PP=2 the fork's DCP group reports
    world_size 0 (CP is disabled: decode_context_parallel_size=1), and the
    FA3 kernel rejects cp_world_size<=0 ("cp_world_size must be
    positive... Use 1 if CP is not enabled"). Resolve from the config
    instead of the group."""
    src = open(path).read()
    if MARKER in src:
        print(f"already patched: {path}")
        return
    src = _replace_once(
        src,
        "        if self.impl.dcp_world_size == -1:\n"
        "            self.impl.dcp_world_size = get_dcp_group().world_size\n",
        "        if self.impl.dcp_world_size == -1:\n"
        "            # dspark k3 probe: resolve DCP from the config, not\n"
        "            # the group -- the fork's DCP group reports world_size\n"
        "            # 0 under PP=2 (CP disabled), which the FA3 kernel\n"
        "            # rejects. Config is the source of truth for whether\n"
        "            # CP is enabled.\n"
        "            from vllm.config import get_current_vllm_config\n"
        "            _ds_dcp = (get_current_vllm_config().parallel_config\n"
        "                       .decode_context_parallel_size)\n"
        "            if _ds_dcp and _ds_dcp > 1:\n"
        "                self.impl.dcp_world_size = (\n"
        "                    get_dcp_group().world_size)\n"
        "                self.impl.dcp_rank = get_dcp_group().rank_in_group\n"
        "            else:\n"
        "                self.impl.dcp_world_size = 1\n"
        "                self.impl.dcp_rank = 0\n",
        "dcp-fix", path)
    # The kimi_k3 fork's own MLA wrapper (models/kimi_k3/nvidia/mla.py)
    # calls impl.forward_mqa DIRECTLY, bypassing MLAAttention.forward_impl
    # where the lazy resolution above lives (boot #15: same kernel error
    # with the forward_impl patch in place). Resolve the sentinel at
    # construction instead, config-gated so real DCP keeps working.
    src = _replace_once(
        src,
        "        self.dcp_world_size: int = -1\n",
        "        self.dcp_world_size: int = -1\n"
        "        # dspark k3 probe: the -1 sentinel is resolved lazily in\n"
        "        # MLAAttention.forward_impl, but the kimi_k3 fork's MLA\n"
        "        # wrapper bypasses that path. Resolve at construction when\n"
        "        # CP is disabled (config decode_context_parallel_size <=\n"
        "        # 1): the FA3 kernel requires a positive cp_world_size.\n"
        "        from vllm.config import (\n"
        "            get_current_vllm_config_or_none as _ds_gcc)\n"
        "        _ds_cfg = _ds_gcc()\n"
        "        if _ds_cfg is not None and (\n"
        "                _ds_cfg.parallel_config\n"
        "                .decode_context_parallel_size or 1) <= 1:\n"
        "            self.dcp_world_size = 1\n"
        "            self.dcp_rank = 0\n",
        "dcp-init", path)
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    print(f"patched OK: {path}")


def main() -> None:
    patch_model(MODEL_FILE)
    patch_runner(RUNNER_FILE)
    patch_mla(MLA_FILE)
    with open("/tmp/dspark_k3_patch.ok", "w") as f:
        f.write(f"patched {MODEL_FILE} + {RUNNER_FILE} + {MLA_FILE}\n")


if __name__ == "__main__":
    sys.exit(main())
