#!/usr/bin/env python3
"""Fail-closed runtime patch: dspark activation probe + weightless projective
steering for GLM-5.3 (arch glm_moe_dsa) on vLLM v0.28.0's deepseek_v2.py.

Vendored verbatim from refusal-research/experiments/20260829-glm53-flagship/
patch_glm53.py (the proven 8xH100 Modal lane) for the weightless cloud
serving lane (modal/cloud_serve.py). Probe stays inert unless
WEIGHTLESS_PROBE_* / DSPARK_PROBE_* is set; serving sets neither.

Same lane design as
experiments/20260826-flash-next-vllm-capture/patch_qwen4_exp.py, adapted to
the deepseek_v2 DECOMPOSED convention (no hyper-connection widening here):

  * The residual stream between layers is plain [T, hidden] = [T, 6144]
    carried as a (hidden_states, residual) pair; the add is fused into the
    next layer's input_layernorm. The post-layer stream an HF hook sees is
    hidden_states + residual (proven by the file's own aux_hidden_state
    code: `aux_hidden_state = hidden_states + residual`).
  * Steering: h_sum <- h_sum - alpha * (h_sum . d) d, written back into
    hidden_states with residual untouched (the hotfix pattern from
    hotfix-qwen38-steering-projective.py).
  * Capture stores the pre-steer stream when both are active (in practice
    capture and eval are separate runs).

EAGER-ONLY: this lane runs enforce_eager=True. DeepseekV2Model is
@support_torch_compile; the Python branches, GPU->CPU copy and torch.save
below are only safe because nothing is traced or graphed. A stream-capture
guard refuses to dump under CUDA graphs. Steering still uses a dense
zero-padded stack + tensor alpha so the code is not a trap if ever traced.

Env vars (probe):
  DSPARK_PROBE_LAYERS / DSPARK_PROBE_DUMP_DIR / DSPARK_PROBE_MIN_TOKENS /
  DSPARK_PROBE_MAX_TOKENS -- same semantics as the qwen4_exp lane.

Env vars (steering):
  WEIGHTLESS_STEER_PATH    .pt: {layer: tensor} or {"per_layer": {...}};
                           fail-closed (boot raises) if set but unusable
  WEIGHTLESS_STEER_ALPHA   float, default 1.0
  WEIGHTLESS_STEER_LAYERS  optional comma list restricting layer ids

Dump format per prefill forward (TP rank 0 only):
  {"act_mean": [n_layers, 6144] fp32, "act_last": [n_layers, 6144] fp32,
   "layers": sorted layer ids, "n_tokens": int}

Exit codes: 0 patched (or already patched), 1 anchor mismatch.
"""

import importlib
import py_compile
import sys

MARKER = "dspark glm_moe_dsa probe"


def _find_model_file() -> str:
    m = importlib.import_module("vllm.model_executor.models.deepseek_v2")
    return m.__file__


def _replace_once(src: str, anchor: str, replacement: str, name: str) -> str:
    n = src.count(anchor)
    if n != 1:
        raise SystemExit(
            f"PATCH FAILED: anchor {name!r} found {n} times (expected 1). "
            "The image's deepseek_v2.py has drifted from v0.28.0; refusing "
            "to guess. Diff the installed file against "
            "upstream_src/deepseek_v2_v0280.py and update this patch."
        )
    return src.replace(anchor, replacement)


def main() -> None:
    path = _find_model_file()
    src = open(path).read()

    if MARKER in src:
        print(f"already patched: {path}")
        return

    # 1. module imports (after the stock logger line) -------------------------
    src = _replace_once(
        src,
        "logger = init_logger(__name__)\n",
        "logger = init_logger(__name__)\n"
        "import os as _dspark_os\n"
        "from vllm.distributed import (\n"
        "    get_tensor_model_parallel_rank as _dspark_tp_rank,\n"
        ")\n",
        "imports",
    )

    # 2. probe + steering state in DeepseekV2Model.__init__ -------------------
    src = _replace_once(
        src,
        "        self.aux_hidden_state_layers = tuple[int, ...]()\n",
        "        self.aux_hidden_state_layers = tuple[int, ...]()\n"
        "\n"
        "        # ---- dspark glm_moe_dsa probe (capture-only; eager) --------\n"
        "        # Inert unless DSPARK_PROBE_LAYERS is set.\n"
        "        _probe_spec = _dspark_os.environ.get(\n"
        "            \"DSPARK_PROBE_LAYERS\", \"\").strip()\n"
        "        _probe_ids = []\n"
        "        if _probe_spec:\n"
        "            for _tok in _probe_spec.replace(\" \", \"\").split(\",\"):\n"
        "                if _tok:\n"
        "                    _lid = int(_tok)\n"
        "                    if 0 <= _lid < config.num_hidden_layers:\n"
        "                        _probe_ids.append(_lid)\n"
        "        self._probe_layer_set = frozenset(_probe_ids)\n"
        "        self._probe_dump_dir = _dspark_os.environ.get(\n"
        "            \"DSPARK_PROBE_DUMP_DIR\", \"\").strip()\n"
        "        self._probe_min_tokens = int(\n"
        "            _dspark_os.environ.get(\"DSPARK_PROBE_MIN_TOKENS\", \"4\")\n"
        "            or 4)\n"
        "        self._probe_max_tokens = int(\n"
        "            _dspark_os.environ.get(\"DSPARK_PROBE_MAX_TOKENS\", \"1024\")\n"
        "            or 1024)\n"
        "        self._probe_seq = 0\n"
        "        if self._probe_layer_set:\n"
        "            logger.info(\n"
        "                \"DSpark glm_moe_dsa probe active: %d layers -> %s \"\n"
        "                \"(EAGER ONLY: no torch.compile / CUDA graphs)\",\n"
        "                len(self._probe_layer_set),\n"
        "                self._probe_dump_dir or \"(no dump dir!)\",\n"
        "            )\n"
        "\n"
        "        # ---- weightless projective steering (eval lane) ------------\n"
        "        # h <- h - alpha * (h . d) d on the post-layer stream\n"
        "        # (hidden_states + residual). Inert unless WEIGHTLESS_STEER_PATH\n"
        "        # is set. Dense zero-padded stack + tensor alpha (compile-safe\n"
        "        # discipline from the dspark-deploy hotfixes) even though this\n"
        "        # lane runs eager.\n"
        "        self._steer_alpha_val = float(\n"
        "            _dspark_os.environ.get(\"WEIGHTLESS_STEER_ALPHA\", \"1.0\")\n"
        "            or 1.0)\n"
        "        _steer_dtype = vllm_config.model_config.dtype\n"
        "        self.register_buffer(\n"
        "            \"_steer_stack\",\n"
        "            torch.zeros(config.num_hidden_layers, config.hidden_size,\n"
        "                        dtype=_steer_dtype),\n"
        "            persistent=False,\n"
        "        )\n"
        "        self.register_buffer(\n"
        "            \"_steer_alpha\",\n"
        "            torch.zeros((), dtype=_steer_dtype),\n"
        "            persistent=False,\n"
        "        )\n"
        "        self._steer_layers = ()\n"
        "        _steer_path = _dspark_os.environ.get(\n"
        "            \"WEIGHTLESS_STEER_PATH\", \"\").strip()\n"
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
        "            _want = _dspark_os.environ.get(\n"
        "                \"WEIGHTLESS_STEER_LAYERS\", \"\").strip()\n"
        "            _selected = (\n"
        "                {int(t) for t in _want.replace(\" \", \"\").split(\",\")\n"
        "                 if t}\n"
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
        "                        f\"{_v.numel()} != {config.hidden_size}\")\n"
        "                _dirs[_lid] = _v / (_v.norm() + 1e-9)\n"
        "            if not _dirs:\n"
        "                raise RuntimeError(\n"
        "                    f\"WEIGHTLESS_STEER_PATH={_steer_path} matched no \"\n"
        "                    \"layers; refusing to run unsteered\")\n"
        "            for _lid, _v in _dirs.items():\n"
        "                self._steer_stack[_lid] = _v.to(_steer_dtype)\n"
        "            self._steer_alpha.fill_(self._steer_alpha_val)\n"
        "            self._steer_layers = tuple(sorted(_dirs))\n"
        "            logger.info(\n"
        "                \"weightless steering active: alpha=%.3f layers=%d..%d \"\n"
        "                \"(%d) width=%d\",\n"
        "                self._steer_alpha_val, self._steer_layers[0],\n"
        "                self._steer_layers[-1], len(self._steer_layers),\n"
        "                config.hidden_size)\n",
        "init",
    )

    # 3. probe store init before the layer loop -------------------------------
    src = _replace_once(
        src,
        "        aux_hidden_states = []\n",
        "        aux_hidden_states = []\n"
        "        _dspark_probe_store = {}\n",
        "loop-init",
    )

    # 4. per-layer capture + steering after the layer call ---------------------
    src = _replace_once(
        src,
        "            hidden_states, residual = layer(\n"
        "                positions, hidden_states, residual, llama_4_scaling\n"
        "            )\n",
        "            hidden_states, residual = layer(\n"
        "                positions, hidden_states, residual, llama_4_scaling\n"
        "            )\n"
        "            # ---- dspark probe + weightless steering ----------------\n"
        "            # Decomposed convention: the post-layer stream is\n"
        "            # hidden_states + residual (the add is otherwise fused into\n"
        "            # the next layer's input_layernorm; the file's own\n"
        "            # aux_hidden_state code does the same sum). Steering writes\n"
        "            # the steered sum back into hidden_states, residual\n"
        "            # untouched, so the next layer sees h+r-alpha*(h.d)d.\n"
        "            _ds_probe = (\n"
        "                bool(self._probe_layer_set)\n"
        "                and idx in self._probe_layer_set\n"
        "            )\n"
        "            _ds_steer = idx in self._steer_layers\n"
        "            if _ds_probe or _ds_steer:\n"
        "                _ds_post = hidden_states + residual\n"
        "                if _ds_probe:\n"
        "                    _dspark_probe_store[idx] = _ds_post\n"
        "                if _ds_steer:\n"
        "                    _ds_d = self._steer_stack[idx]\n"
        "                    _ds_post = _ds_post - self._steer_alpha * (\n"
        "                        (_ds_post @ _ds_d).unsqueeze(-1) * _ds_d)\n"
        "                    hidden_states = _ds_post - residual\n",
        "capture",
    )

    # 5. dump after the final norm (last PP rank) ------------------------------
    src = _replace_once(
        src,
        "        hidden_states, _ = self.norm(hidden_states, residual)\n",
        "        hidden_states, _ = self.norm(hidden_states, residual)\n"
        "\n"
        "        # ---- dspark probe: dump prefill activations ----------------\n"
        "        # Eager-only lane; the stream-capture guard refuses to dump\n"
        "        # under CUDA graphs if that ever stops being true.\n"
        "        if (\n"
        "            _dspark_probe_store\n"
        "            and self._probe_dump_dir\n"
        "            and _dspark_tp_rank() == 0\n"
        "        ):\n"
        "            _probe_t = hidden_states.shape[0]\n"
        "            _capturing = (\n"
        "                torch.cuda.is_available()\n"
        "                and torch.cuda.is_current_stream_capturing()\n"
        "            )\n"
        "            if not _capturing and (\n"
        "                self._probe_min_tokens <= _probe_t <= self._probe_max_tokens\n"
        "            ):\n"
        "                try:\n"
        "                    _layers = sorted(_dspark_probe_store)\n"
        "                    _mat = torch.stack(\n"
        "                        [_dspark_probe_store[l] for l in _layers])\n"
        "                    _dspark_os.makedirs(self._probe_dump_dir,\n"
        "                                        exist_ok=True)\n"
        "                    torch.save(\n"
        "                        {\n"
        "                            \"act_mean\": _mat.float().mean(dim=1).cpu(),\n"
        "                            \"act_last\": _mat[:, -1, :].float().cpu(),\n"
        "                            \"layers\": _layers,\n"
        "                            \"n_tokens\": int(_probe_t),\n"
        "                        },\n"
        "                        _dspark_os.path.join(\n"
        "                            self._probe_dump_dir,\n"
        "                            \"probe_%06d.pt\" % self._probe_seq,\n"
        "                        ),\n"
        "                    )\n"
        "                    self._probe_seq += 1\n"
        "                except Exception as exc:  # never take the model down\n"
        "                    logger.warning(\n"
        "                        \"DSpark glm_moe_dsa probe dump failed: %s\", exc)\n",
        "dump",
    )

    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    with open("/tmp/dspark_probe_patch.ok", "w") as f:
        f.write(f"patched {path}\n")
    print(f"patched OK: {path}")


if __name__ == "__main__":
    main()
