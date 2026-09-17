#!/usr/bin/env python3
"""Fail-closed source-patch installer for exact vLLM v0.27.0."""

import os
from pathlib import Path
from typing import Any

EXPECTED_VLLM_VERSION = "0.27.0"
XARGS_MARK = "# [steering-hotfix] per-request Weightless alpha via vllm_xargs"
OPENAI_PROTOCOL_PATCH_REVISION_MARK = (
    "_WEIGHTLESS_OPENAI_PROTOCOL_PATCH_REVISION = 2"
)
REQUEST_SCHEMA_MARK = "_WEIGHTLESS_RUNTIME_SCHEMA_VERSION = 1"
REQUEST_PATCH_REVISION_MARK = "_WEIGHTLESS_RUNTIME_PATCH_REVISION = 4"
REQUEST_CONFIG_MARK = "class WeightlessRequestConfig:"
TELEMETRY_MARK = "# [weightless-telemetry] directional scalar telemetry v1"
MILESTONE2_MARK = "# [weightless-controls] layer and token schedules v1"
RUNTIME_MODULE_NAME = "_weightless_runtime_v027.py"
_CORE_PACKAGE_FILES = ("__init__.py", "controls.py", "telemetry.py")


def _core_source_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "weightless_runtime"
        if all((candidate / name).is_file() for name in _CORE_PACKAGE_FILES):
            return candidate
    raise FileNotFoundError(
        "weightless_runtime core package not found beside the integration bundle"
    )


def install_runtime_module(root: Path) -> tuple[bool, str]:
    runtime_source = Path(__file__).with_name("runtime.py")
    if not runtime_source.is_file():
        return False, f"[steering-hotfix] runtime adapter missing: {runtime_source}"

    core_source = _core_source_dir()
    core_target = root.parent / "weightless_runtime"
    core_target.mkdir(parents=True, exist_ok=True)
    changed = False
    for name in _CORE_PACKAGE_FILES:
        source_path = core_source / name
        target_path = core_target / name
        contents = source_path.read_text()
        if not target_path.is_file() or target_path.read_text() != contents:
            target_path.write_text(contents)
            changed = True

    target = root / RUNTIME_MODULE_NAME
    contents = runtime_source.read_text()
    if not target.is_file() or target.read_text() != contents:
        target.write_text(contents)
        changed = True

    state = "installed" if changed else "already current"
    return True, (
        f"[steering-hotfix] runtime adapter and core package {state} at "
        f"{target} and {core_target}"
    )


LEGACY_REQUEST_IMPORTS = "import enum\nimport math\nimport os\nimport time\n"
MILESTONE1_REQUEST_IMPORTS = (
    "import enum\nimport hashlib\nimport math\nimport os\nimport time\n"
)
REQUEST_IMPORTS = (
    "import enum\nimport hashlib\nimport json\nimport math\nimport os\nimport time\n"
)

LEGACY_REQUEST_STATE_ASSIGNMENT = (
    "        self.weightless_alpha_override = "
    "_weightless_alpha_override_from_sampling_params(sampling_params)\n"
    "        self.cache_salt = _weightless_cache_salt(cache_salt, sampling_params)\n"
)

REQUEST_STATE_ASSIGNMENT = (
    "        self.weightless_config = _weightless_request_config(sampling_params)\n"
    "        self.weightless_alpha_override = self.weightless_config.alpha_override\n"
    "        self.cache_salt = "
    "_weightless_cache_salt(cache_salt, self.weightless_config)\n"
)

OPENAI_PROTOCOL_HELPERS = '''
# [steering-hotfix] per-request Weightless alpha via vllm_xargs
_WEIGHTLESS_OPENAI_PROTOCOL_PATCH_REVISION = 2

from vllm._weightless_runtime_v027 import prepare_weightless_openai_payload


def _validate_weightless_vllm_xargs(data):
    prepare_weightless_openai_payload(data)
'''

REQUEST_HELPERS = '''
# [steering-hotfix] per-request Weightless alpha via vllm_xargs
_WEIGHTLESS_RUNTIME_SCHEMA_VERSION = 1
_WEIGHTLESS_RUNTIME_PATCH_REVISION = 4
_WEIGHTLESS_MUTATION_SCHEMA_VERSION = 2
_WEIGHTLESS_XARG_ALPHA_KEY = "weightless_alpha"
_WEIGHTLESS_STEER_IDENTITIES = {}

from vllm._weightless_runtime_v027 import (
    WeightlessEffectiveControl,
    WeightlessResolvedXArgs,
    WeightlessServerLineage,
    WeightlessTraceConfig,
    effective_weightless_control,
    static_weightless_layers,
    weightless_config_from_sampling_params,
)


@dataclass(frozen=True, slots=True)
class WeightlessRequestConfig:
    alpha_override: float | None
    effective_alpha: float
    control: WeightlessEffectiveControl
    mutation_fingerprint: str | None
    trace: WeightlessTraceConfig | None
    checkpoint_id: str | None
    parent_branch_id: str | None
    branch_id: str | None
    history_fingerprint: str | None
    continuation_fingerprint: str | None


def _weightless_alpha_override_from_sampling_params(sampling_params):
    return weightless_config_from_sampling_params(sampling_params).alpha_override


def _weightless_effective_alpha(sampling_params):
    request_alpha = _weightless_alpha_override_from_sampling_params(sampling_params)
    if request_alpha is not None:
        return request_alpha
    try:
        return float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
    except Exception:
        return 1.0


def _weightless_effective_control(resolved):
    try:
        default_alpha = float(os.environ.get("WEIGHTLESS_STEER_ALPHA", "1.0") or 1.0)
    except Exception:
        default_alpha = 1.0
    return effective_weightless_control(
        resolved,
        default_alpha=default_alpha,
        default_layers=static_weightless_layers(),
    )


def _weightless_steer_identity(steer_path):
    if not steer_path:
        return "none"
    cached = _WEIGHTLESS_STEER_IDENTITIES.get(steer_path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(steer_path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    identity = f"sha256:{digest.hexdigest()}"
    _WEIGHTLESS_STEER_IDENTITIES[steer_path] = identity
    return identity


def _weightless_mutation_fingerprint(steer_identity, control):
    payload = {
        "first_sample_policy": "decode_on_final_prefill_token",
        "base_alpha": format(control.base_alpha, ".17g"),
        "prefill_alpha": format(control.prefill_alpha, ".17g"),
        "decode_alpha": format(control.decode_alpha, ".17g"),
        "layers": "loaded" if control.layers is None else list(control.layers),
        "schedule": [
            {
                "start": segment.start,
                "end": segment.end,
                "start_alpha": format(segment.start_alpha, ".17g"),
                "end_alpha": format(segment.end_alpha, ".17g"),
            }
            for segment in control.schedule
        ],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    return (
        f"weightless:v{_WEIGHTLESS_MUTATION_SCHEMA_VERSION}:"
        f"{steer_identity}:controls=sha256:{digest}"
    )


def _weightless_request_config(sampling_params):
    steer_path = (os.environ.get("WEIGHTLESS_STEER_PATH") or "").strip()
    resolved = weightless_config_from_sampling_params(
        sampling_params, assign_trace_id=True
    )
    alpha_override = resolved.alpha_override
    control = _weightless_effective_control(resolved)
    effective_alpha = control.base_alpha
    fingerprint = None
    if steer_path or alpha_override is not None or resolved.has_advanced_controls:
        steer_identity = _weightless_steer_identity(steer_path)
        fingerprint = _weightless_mutation_fingerprint(
            steer_identity, control
        )
    lineage = (
        resolved.trace.lineage
        if resolved.trace is not None
        else WeightlessServerLineage()
    )
    return WeightlessRequestConfig(
        alpha_override=alpha_override,
        effective_alpha=effective_alpha,
        control=control,
        mutation_fingerprint=fingerprint,
        trace=resolved.trace,
        checkpoint_id=lineage.checkpoint_id,
        parent_branch_id=lineage.parent_branch_id,
        branch_id=lineage.branch_id,
        history_fingerprint=lineage.history_fingerprint,
        continuation_fingerprint=lineage.continuation_fingerprint,
    )


def _weightless_cache_salt(cache_salt, config):
    fingerprint = config.mutation_fingerprint
    if fingerprint is None:
        return cache_salt
    return f"{cache_salt}|{fingerprint}" if cache_salt else fingerprint
'''

RUNNER_HELPERS = '''
# [steering-hotfix] per-request Weightless alpha via vllm_xargs
from vllm._weightless_runtime_v027 import (
    WeightlessControlRunner,
    WeightlessTraceRunner,
    make_weightless_trace_runner,
    weightless_config_from_sampling_params,
)


def _weightless_alpha_override_from_sampling_params(sampling_params):
    return weightless_config_from_sampling_params(sampling_params).alpha_override


def _weightless_steer_target_model(model):
    modules = model.modules() if hasattr(model, "modules") else (model,)
    for module in modules:
        if hasattr(module, "_set_weightless_alpha_rows"):
            return module
    return None
'''


def _find_vllm_root(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name == "vllm":
            return parent
    return None


def _installed_vllm_version(root: Path) -> str | None:
    versions = set()
    for metadata_path in root.parent.glob("vllm-*.dist-info/METADATA"):
        for line in metadata_path.read_text().splitlines():
            if line.startswith("Version: "):
                versions.add(line.removeprefix("Version: ").strip())
                break
    if len(versions) > 1:
        return ",".join(sorted(versions))
    return next(iter(versions), None)


def _replace_one_of(
    src: str, label: str, variants: tuple[str, ...], replacement: str
) -> tuple[str, bool]:
    for old in variants:
        if old in src:
            assert src.count(old) == 1, f"anchor {label!r} not unique"
            return src.replace(old, replacement, 1), True
    return src, False


def _replace_injected_runner_helpers(src: str) -> tuple[str, bool]:
    helper_start = src.find(XARGS_MARK)
    target_start = src.find("def _weightless_steer_target_model", helper_start)
    helper_end_anchor = "    return None\n"
    helper_end = src.find(helper_end_anchor, target_start)
    if helper_start < 0 or target_start < 0 or helper_end < 0:
        return src, False
    helper_end += len(helper_end_anchor)
    return (
        src[:helper_start]
        + RUNNER_HELPERS.strip("\n")
        + "\n"
        + src[helper_end:],
        True,
    )


def _patch_openai_protocol_file(path: Path) -> tuple[bool, str]:
    src = path.read_text()
    if XARGS_MARK in src:
        if (
            "from vllm._weightless_runtime_v027 import" in src
            and OPENAI_PROTOCOL_PATCH_REVISION_MARK in src
        ):
            return True, f"[steering-hotfix] already applied xargs patch to {path}"
        helper_start = src.find(XARGS_MARK)
        class_start = src.find("class OpenAIBaseModel(BaseModel):\n", helper_start)
        if class_start < 0:
            return False, (
                "[steering-hotfix] xargs patch: legacy OpenAI class anchor "
                f"missing in {path}"
            )
        src = (
            src[:helper_start]
            + OPENAI_PROTOCOL_HELPERS.strip("\n")
            + "\n\n"
            + src[class_start:]
        )
        path.write_text(src)
        return True, f"[steering-hotfix] upgraded xargs patch in {path}"

    src, ok = _replace_one_of(
        src,
        "OpenAI protocol imports",
        ("import json\nimport time\n",),
        "import json\nimport math\nimport time\n",
    )
    if not ok:
        return False, f"[steering-hotfix] xargs patch: imports anchor missing in {path}"

    src, ok = _replace_one_of(
        src,
        "OpenAI base model class",
        ("class OpenAIBaseModel(BaseModel):\n",),
        OPENAI_PROTOCOL_HELPERS + "\nclass OpenAIBaseModel(BaseModel):\n",
    )
    if not ok:
        return False, f"[steering-hotfix] xargs patch: class anchor missing in {path}"

    src, ok = _replace_one_of(
        src,
        "OpenAI base model validator",
        (
            "    def __log_extra_fields__(cls, data, handler):\n"
            "        result = handler(data)\n",
        ),
        (
            "    def __log_extra_fields__(cls, data, handler):\n"
            "        _validate_weightless_vllm_xargs(data)\n"
            "        result = handler(data)\n"
        ),
    )
    if not ok:
        return False, f"[steering-hotfix] xargs patch: validator anchor missing in {path}"

    path.write_text(src)
    return True, f"[steering-hotfix] applied xargs patch to {path}"


def _patch_request_file(path: Path) -> tuple[bool, str]:
    src = path.read_text()
    if XARGS_MARK in src:
        if (
            REQUEST_SCHEMA_MARK not in src
            or REQUEST_PATCH_REVISION_MARK not in src
            or REQUEST_CONFIG_MARK not in src
            or REQUEST_STATE_ASSIGNMENT not in src
            or REQUEST_IMPORTS not in src
        ):
            if REQUEST_IMPORTS not in src:
                src, ok = _replace_one_of(
                    src,
                    "legacy request imports",
                    (LEGACY_REQUEST_IMPORTS, MILESTONE1_REQUEST_IMPORTS),
                    REQUEST_IMPORTS,
                )
                if not ok:
                    return False, (
                        "[steering-hotfix] xargs patch: legacy request imports "
                        f"anchor missing in {path}"
                    )
            helper_start = src.find(XARGS_MARK)
            class_start = src.find("class Request:\n", helper_start)
            if class_start < 0:
                return False, (
                    "[steering-hotfix] xargs patch: legacy request class "
                    f"anchor missing in {path}"
                )
            src = (
                src[:helper_start]
                + REQUEST_HELPERS.strip("\n")
                + "\n\n"
                + src[class_start:]
            )
            if REQUEST_STATE_ASSIGNMENT not in src:
                src, ok = _replace_one_of(
                    src,
                    "legacy request state",
                    (LEGACY_REQUEST_STATE_ASSIGNMENT,),
                    REQUEST_STATE_ASSIGNMENT,
                )
                if not ok:
                    return False, (
                        "[steering-hotfix] xargs patch: legacy request state "
                        f"anchor missing in {path}"
                    )
            path.write_text(src)
            return True, f"[steering-hotfix] upgraded xargs patch in {path}"
        return True, f"[steering-hotfix] already applied xargs patch to {path}"

    src, ok = _replace_one_of(
        src,
        "request imports",
        ("import enum\nimport time\n",),
        REQUEST_IMPORTS,
    )
    if not ok:
        return False, f"[steering-hotfix] xargs patch: imports anchor missing in {path}"

    src, ok = _replace_one_of(
        src,
        "request class anchor",
        ("class Request:\n",),
        REQUEST_HELPERS + "\nclass Request:\n",
    )
    if not ok:
        return False, f"[steering-hotfix] xargs patch: class anchor missing in {path}"

    src, ok = _replace_one_of(
        src,
        "request cache_salt",
        (
            "        self.cache_salt: str | None = cache_salt\n",
            "        self.cache_salt: Optional[str] = cache_salt\n",
            "        self.cache_salt = cache_salt\n",
        ),
        REQUEST_STATE_ASSIGNMENT,
    )
    if not ok:
        return False, f"[steering-hotfix] xargs patch: cache_salt anchor missing in {path}"

    path.write_text(src)
    return True, f"[steering-hotfix] applied xargs patch to {path}"


RUNNER_V2_METHOD = '''

    def _set_weightless_alpha_rows(self, input_batch: InputBatch) -> None:
        steer_model = _weightless_steer_target_model(self.get_model())
        if steer_model is None:
            return
        self._weightless_control_runtime.prepare_v2(
            steer_model,
            input_batch,
            self._weightless_control_by_req_id,
        )
        trace_runtime = self._weightless_trace_runtime
        if trace_runtime is not None and hasattr(
            steer_model, "_set_weightless_trace_rows"
        ):
            trace_runtime.prepare_v2(
                steer_model,
                input_batch,
                self._weightless_trace_by_req_id,
            )
'''


RUNNER_V1_METHOD = '''

    def _set_weightless_alpha_rows(
        self,
        req_ids: list[str],
        num_scheduled_tokens_np: np.ndarray,
        num_tokens_after_padding: int,
        ubatch_slices,
    ) -> None:
        steer_model = _weightless_steer_target_model(self.get_model())
        if steer_model is None:
            return
        self._weightless_control_runtime.prepare_v1(
            steer_model,
            req_ids,
            num_scheduled_tokens_np,
            num_tokens_after_padding,
            self.requests,
            ubatch_slices,
        )
        trace_runtime = self._weightless_trace_runtime
        if trace_runtime is not None and hasattr(
            steer_model, "_set_weightless_trace_rows"
        ):
            trace_runtime.prepare_v1(
                steer_model,
                req_ids,
                num_scheduled_tokens_np,
                self.requests,
                ubatch_slices,
            )
'''


def _upgrade_runner_v2_telemetry(src: str) -> tuple[str, bool, str]:
    src, ok = _replace_injected_runner_helpers(src)
    if not ok:
        return src, False, "legacy runner helpers"

    method_start = src.find("    def _set_weightless_alpha_rows(")
    method_end = src.find("    def update_max_model_len(", method_start)
    if method_start < 0 or method_end < 0:
        return src, False, "legacy v2 alpha method"
    src = src[:method_start] + RUNNER_V2_METHOD.strip("\n") + "\n\n" + src[method_end:]

    legacy_state = (
        "        self._weightless_alpha_override_by_req_id: "
        "dict[str, float | None] = {}\n"
    )
    telemetry_state = (
        legacy_state
        + "        self._weightless_control_by_req_id = {}\n"
        + "        self._weightless_control_runtime = WeightlessControlRunner(\n"
        + "            max_num_tokens=self.max_num_tokens,\n"
        + "            max_num_reqs=self.max_num_reqs,\n"
        + "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
        + "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
        + "        )\n"
        + "        self._weightless_trace_by_req_id = {}\n"
        + "        self._weightless_trace_runtime = make_weightless_trace_runner(\n"
        + "            max_num_tokens=self.max_num_tokens,\n"
        + "            max_num_reqs=self.max_num_reqs,\n"
        + "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
        + "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
        + "        )\n"
        + f"        {TELEMETRY_MARK}\n"
        + f"        {MILESTONE2_MARK}\n"
    )
    src, ok = _replace_one_of(src, "legacy v2 runner state", (legacy_state,), telemetry_state)
    if not ok:
        return src, False, "legacy v2 runner state"

    legacy_remove = (
        "        self._weightless_alpha_override_by_req_id.pop(req_id, None)\n"
        "        if req_idx is None:\n"
    )
    telemetry_remove = (
        "        self._weightless_alpha_override_by_req_id.pop(req_id, None)\n"
        "        self._weightless_control_by_req_id.pop(req_id, None)\n"
        "        self._weightless_trace_by_req_id.pop(req_id, None)\n"
        "        if req_idx is None:\n"
    )
    src, ok = _replace_one_of(
        src, "legacy v2 remove request", (legacy_remove,), telemetry_remove
    )
    if not ok:
        return src, False, "legacy v2 remove request"

    legacy_add = (
        "            self._weightless_alpha_override_by_req_id[req_id] = (\n"
        "                _weightless_alpha_override_from_sampling_params(sampling_params)\n"
        "            )\n"
    )
    telemetry_add = (
        "            weightless_config = weightless_config_from_sampling_params(\n"
        "                sampling_params\n"
        "            )\n"
        "            self._weightless_alpha_override_by_req_id[req_id] = (\n"
        "                weightless_config.alpha_override\n"
        "            )\n"
        "            self._weightless_control_by_req_id[req_id] = weightless_config\n"
        "            self._weightless_trace_by_req_id[req_id] = (\n"
        "                weightless_config.trace\n"
        "            )\n"
    )
    src, ok = _replace_one_of(src, "legacy v2 add request", (legacy_add,), telemetry_add)
    if not ok:
        return src, False, "legacy v2 add request"

    collect_anchor = (
        "        if self.is_last_pp_rank:\n"
        "            if self.use_aux_hidden_state_outputs:\n"
    )
    collect_block = (
        "        trace_runtime = self._weightless_trace_runtime\n"
        "        if trace_runtime is not None:\n"
        "            steer_model = _weightless_steer_target_model(self.get_model())\n"
        "            if steer_model is not None and hasattr(\n"
        "                steer_model, \"_weightless_trace_values\"\n"
        "            ):\n"
        "                trace_runtime.collect(steer_model)\n"
        "\n"
        + collect_anchor
    )
    src, ok = _replace_one_of(
        src, "legacy v2 trace collection", (collect_anchor,), collect_block
    )
    if not ok:
        return src, False, "legacy v2 trace collection"
    return src, True, ""


def _upgrade_runner_v1_telemetry(src: str) -> tuple[str, bool, str]:
    src, ok = _replace_injected_runner_helpers(src)
    if not ok:
        return src, False, "legacy runner helpers"

    method_start = src.find("    def _set_weightless_alpha_rows(")
    method_end = src.find("    def _prepare_inputs(", method_start)
    if method_start < 0 or method_end < 0:
        return src, False, "legacy v1 alpha method"
    src = src[:method_start] + RUNNER_V1_METHOD.strip("\n") + "\n\n" + src[method_end:]

    legacy_state = (
        "        self._weightless_alpha_rows_gpu = torch.empty(\n"
        "            (max(1, self.parallel_config.num_ubatches), self.max_num_tokens),\n"
        "            device=self.device, dtype=self.dtype\n"
        "        )\n"
    )
    telemetry_state = (
        legacy_state
        + "        self._weightless_control_runtime = WeightlessControlRunner(\n"
        + "            max_num_tokens=self.max_num_tokens,\n"
        + "            max_num_reqs=self.max_num_reqs,\n"
        + "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
        + "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
        + "        )\n"
        + "        self._weightless_trace_runtime = make_weightless_trace_runner(\n"
        + "            max_num_tokens=self.max_num_tokens,\n"
        + "            max_num_reqs=self.max_num_reqs,\n"
        + "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
        + "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
        + "        )\n"
        + f"        {TELEMETRY_MARK}\n"
        + f"        {MILESTONE2_MARK}\n"
    )
    src, ok = _replace_one_of(src, "legacy v1 runner state", (legacy_state,), telemetry_state)
    if not ok:
        return src, False, "legacy v1 runner state"

    collect_anchor = (
        "        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n"
    )
    collect_block = (
        "        trace_runtime = self._weightless_trace_runtime\n"
        "        if trace_runtime is not None:\n"
        "            steer_model = _weightless_steer_target_model(self.get_model())\n"
        "            if steer_model is not None and hasattr(\n"
        "                steer_model, \"_weightless_trace_values\"\n"
        "            ):\n"
        "                trace_runtime.collect(steer_model)\n"
        "\n"
        + collect_anchor
    )
    src, ok = _replace_one_of(
        src, "legacy v1 trace collection", (collect_anchor,), collect_block
    )
    if not ok:
        return src, False, "legacy v1 trace collection"
    return src, True, ""


def _upgrade_runner_v2_controls(src: str) -> tuple[str, bool, str]:
    src, ok = _replace_injected_runner_helpers(src)
    if not ok:
        return src, False, "milestone 1 runner helpers"

    method_start = src.find("    def _set_weightless_alpha_rows(")
    method_end = src.find("    def update_max_model_len(", method_start)
    if method_start < 0 or method_end < 0:
        return src, False, "milestone 1 v2 alpha method"
    src = src[:method_start] + RUNNER_V2_METHOD.strip("\n") + "\n\n" + src[method_end:]

    state_anchor = (
        "        self._weightless_alpha_override_by_req_id: "
        "dict[str, float | None] = {}\n"
    )
    state_replacement = (
        state_anchor
        + "        self._weightless_control_by_req_id = {}\n"
        + "        self._weightless_control_runtime = WeightlessControlRunner(\n"
        + "            max_num_tokens=self.max_num_tokens,\n"
        + "            max_num_reqs=self.max_num_reqs,\n"
        + "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
        + "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
        + "        )\n"
    )
    if "        self._weightless_control_runtime = WeightlessControlRunner(\n" not in src:
        src, ok = _replace_one_of(
            src, "milestone 1 v2 runner state", (state_anchor,), state_replacement
        )
        if not ok:
            return src, False, "milestone 1 v2 runner state"

    remove_anchor = (
        "        self._weightless_alpha_override_by_req_id.pop(req_id, None)\n"
        "        self._weightless_trace_by_req_id.pop(req_id, None)\n"
    )
    remove_replacement = (
        "        self._weightless_alpha_override_by_req_id.pop(req_id, None)\n"
        "        self._weightless_control_by_req_id.pop(req_id, None)\n"
        "        self._weightless_trace_by_req_id.pop(req_id, None)\n"
    )
    if "        self._weightless_control_by_req_id.pop(req_id, None)\n" not in src:
        src, ok = _replace_one_of(
            src, "milestone 1 v2 remove request", (remove_anchor,), remove_replacement
        )
        if not ok:
            return src, False, "milestone 1 v2 remove request"

    add_anchor = (
        "            self._weightless_alpha_override_by_req_id[req_id] = (\n"
        "                weightless_config.alpha_override\n"
        "            )\n"
        "            self._weightless_trace_by_req_id[req_id] = (\n"
    )
    add_replacement = (
        "            self._weightless_alpha_override_by_req_id[req_id] = (\n"
        "                weightless_config.alpha_override\n"
        "            )\n"
        "            self._weightless_control_by_req_id[req_id] = weightless_config\n"
        "            self._weightless_trace_by_req_id[req_id] = (\n"
    )
    control_assignment = (
        "            self._weightless_control_by_req_id[req_id] = weightless_config\n"
    )
    if control_assignment not in src:
        src, ok = _replace_one_of(
            src, "milestone 1 v2 add request", (add_anchor,), add_replacement
        )
        if not ok:
            return src, False, "milestone 1 v2 add request"

    marker_anchor = f"        {TELEMETRY_MARK}\n"
    src, ok = _replace_one_of(
        src,
        "milestone 1 v2 marker",
        (marker_anchor,),
        marker_anchor + f"        {MILESTONE2_MARK}\n",
    )
    if not ok:
        return src, False, "milestone 1 v2 marker"
    return src, True, ""


def _upgrade_runner_v1_controls(src: str) -> tuple[str, bool, str]:
    src, ok = _replace_injected_runner_helpers(src)
    if not ok:
        return src, False, "milestone 1 runner helpers"

    method_start = src.find("    def _set_weightless_alpha_rows(")
    method_end = src.find("    def _prepare_inputs(", method_start)
    if method_start < 0 or method_end < 0:
        return src, False, "milestone 1 v1 alpha method"
    src = src[:method_start] + RUNNER_V1_METHOD.strip("\n") + "\n\n" + src[method_end:]

    state_anchor = (
        "        self._weightless_trace_runtime = make_weightless_trace_runner(\n"
    )
    state_replacement = (
        "        self._weightless_control_runtime = WeightlessControlRunner(\n"
        + "            max_num_tokens=self.max_num_tokens,\n"
        + "            max_num_reqs=self.max_num_reqs,\n"
        + "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
        + "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
        + "        )\n"
        + state_anchor
    )
    if "        self._weightless_control_runtime = WeightlessControlRunner(\n" not in src:
        src, ok = _replace_one_of(
            src, "milestone 1 v1 runner state", (state_anchor,), state_replacement
        )
        if not ok:
            return src, False, "milestone 1 v1 runner state"

    marker_anchor = f"        {TELEMETRY_MARK}\n"
    src, ok = _replace_one_of(
        src,
        "milestone 1 v1 marker",
        (marker_anchor,),
        marker_anchor + f"        {MILESTONE2_MARK}\n",
    )
    if not ok:
        return src, False, "milestone 1 v1 marker"
    return src, True, ""


def _patch_runner_v2_file(path: Path) -> tuple[bool, str]:
    src = path.read_text()
    if XARGS_MARK in src:
        if MILESTONE2_MARK in src:
            return True, f"[steering-hotfix] already applied xargs patch to {path}"
        if TELEMETRY_MARK in src:
            src, ok, label = _upgrade_runner_v2_controls(src)
            upgrade = "Milestone 2 controls"
        else:
            src, ok, label = _upgrade_runner_v2_telemetry(src)
            upgrade = "telemetry and Milestone 2 controls"
        if not ok:
            return False, (
                f"[steering-hotfix] runtime upgrade anchor missing ({label}) in {path}"
            )
        path.write_text(src)
        return True, f"[steering-hotfix] upgraded {upgrade} in {path}"

    runner_helpers = RUNNER_HELPERS.strip("\n")
    src, ok = _replace_one_of(
        src,
        "runner v2 logger anchor",
        ("logger = init_logger(__name__)\n",),
        "logger = init_logger(__name__)\n\n" + runner_helpers + "\n",
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: logger anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v2 alpha buffers",
        (
            "        self.max_num_reqs = self.scheduler_config.max_num_seqs\n",
        ),
        (
            "        self.max_num_reqs = self.scheduler_config.max_num_seqs\n"
            "        self._weightless_alpha_rows_cpu = torch.empty(\n"
            "            (max(1, self.parallel_config.num_ubatches), self.max_num_tokens),\n"
            "            dtype=torch.float32, pin_memory=True\n"
            "        )\n"
            "        self._weightless_alpha_rows_gpu = torch.empty(\n"
            "            (max(1, self.parallel_config.num_ubatches), self.max_num_tokens),\n"
            "            device=self.device, dtype=self.dtype\n"
            "        )\n"
            "        self._weightless_alpha_override_by_req_id: "
            "dict[str, float | None] = {}\n"
            "        self._weightless_control_by_req_id = {}\n"
            "        self._weightless_control_runtime = WeightlessControlRunner(\n"
            "            max_num_tokens=self.max_num_tokens,\n"
            "            max_num_reqs=self.max_num_reqs,\n"
            "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
            "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
            "        )\n"
            "        self._weightless_trace_by_req_id = {}\n"
            "        self._weightless_trace_runtime = make_weightless_trace_runner(\n"
            "            max_num_tokens=self.max_num_tokens,\n"
            "            max_num_reqs=self.max_num_reqs,\n"
            "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
            "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
            "        )\n"
            f"        {TELEMETRY_MARK}\n"
            f"        {MILESTONE2_MARK}\n"
        ),
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: alpha buffer anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v2 remove request",
        (
            "        req_idx = self.req_states.remove_request(req_id)\n"
            "        if req_idx is None:\n"
            "            return False\n",
        ),
        (
            "        req_idx = self.req_states.remove_request(req_id)\n"
            "        self._weightless_alpha_override_by_req_id.pop(req_id, None)\n"
            "        self._weightless_control_by_req_id.pop(req_id, None)\n"
            "        self._weightless_trace_by_req_id.pop(req_id, None)\n"
            "        if req_idx is None:\n"
            "            return False\n"
        ),
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: remove-request anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v2 add request",
        (
            "            sampling_params = new_req_data.sampling_params\n",
        ),
        (
            "            sampling_params = new_req_data.sampling_params\n"
            "            weightless_config = weightless_config_from_sampling_params(\n"
            "                sampling_params\n"
            "            )\n"
            "            self._weightless_alpha_override_by_req_id[req_id] = (\n"
            "                weightless_config.alpha_override\n"
            "            )\n"
            "            self._weightless_control_by_req_id[req_id] = weightless_config\n"
            "            self._weightless_trace_by_req_id[req_id] = (\n"
            "                weightless_config.trace\n"
            "            )\n"
        ),
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: add-request anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v2 method anchor",
        ("    def update_max_model_len(self, max_model_len: int) -> None:\n",),
        RUNNER_V2_METHOD + "\n    def update_max_model_len(self, max_model_len: int) -> None:\n",
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: method anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v2 execute-model anchor",
        (
            "                block_tables = None\n"
            "                slot_mappings = None\n"
            "\n"
            "        attn_metadata = None\n",
        ),
        (
            "                block_tables = None\n"
            "                slot_mappings = None\n"
            "\n"
            "        self._set_weightless_alpha_rows(input_batch)\n"
            "        attn_metadata = None\n"
        ),
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: execute-model anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v2 trace collection",
        (
            "        if self.is_last_pp_rank:\n"
            "            if self.use_aux_hidden_state_outputs:\n",
        ),
        (
            "        trace_runtime = self._weightless_trace_runtime\n"
            "        if trace_runtime is not None:\n"
            "            steer_model = _weightless_steer_target_model(self.get_model())\n"
            "            if steer_model is not None and hasattr(\n"
            "                steer_model, \"_weightless_trace_values\"\n"
            "            ):\n"
            "                trace_runtime.collect(steer_model)\n"
            "\n"
            "        if self.is_last_pp_rank:\n"
            "            if self.use_aux_hidden_state_outputs:\n"
        ),
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: trace collection anchor missing in {path}",
        )

    path.write_text(src)
    return True, f"[steering-hotfix] applied xargs patch to {path}"


def _patch_runner_v1_file(path: Path) -> tuple[bool, str]:
    src = path.read_text()
    if XARGS_MARK in src:
        if MILESTONE2_MARK in src:
            return True, f"[steering-hotfix] already applied xargs patch to {path}"
        if TELEMETRY_MARK in src:
            src, ok, label = _upgrade_runner_v1_controls(src)
            upgrade = "Milestone 2 controls"
        else:
            src, ok, label = _upgrade_runner_v1_telemetry(src)
            upgrade = "telemetry and Milestone 2 controls"
        if not ok:
            return False, (
                f"[steering-hotfix] runtime upgrade anchor missing ({label}) in {path}"
            )
        path.write_text(src)
        return True, f"[steering-hotfix] upgraded {upgrade} in {path}"

    runner_helpers = RUNNER_HELPERS.strip("\n")
    src, ok = _replace_one_of(
        src,
        "runner v1 logger anchor",
        ("logger = init_logger(__name__)\n",),
        "logger = init_logger(__name__)\n\n" + runner_helpers + "\n",
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: logger anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v1 alpha buffers",
        (
            "        # Request states.\n"
            "        self.requests: dict[str, CachedRequestState] = {}\n",
        ),
        (
            "        # Request states.\n"
            "        self.requests: dict[str, CachedRequestState] = {}\n"
            "        self._weightless_alpha_full_cpu = torch.empty(\n"
            "            self.max_num_tokens, dtype=torch.float32, pin_memory=True\n"
            "        )\n"
            "        self._weightless_alpha_rows_cpu = torch.empty(\n"
            "            (max(1, self.parallel_config.num_ubatches), self.max_num_tokens),\n"
            "            dtype=torch.float32, pin_memory=True\n"
            "        )\n"
            "        self._weightless_alpha_rows_gpu = torch.empty(\n"
            "            (max(1, self.parallel_config.num_ubatches), self.max_num_tokens),\n"
            "            device=self.device, dtype=self.dtype\n"
            "        )\n"
            "        self._weightless_control_runtime = WeightlessControlRunner(\n"
            "            max_num_tokens=self.max_num_tokens,\n"
            "            max_num_reqs=self.max_num_reqs,\n"
            "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
            "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
            "        )\n"
            "        self._weightless_trace_runtime = make_weightless_trace_runner(\n"
            "            max_num_tokens=self.max_num_tokens,\n"
            "            max_num_reqs=self.max_num_reqs,\n"
            "            num_ubatches=max(1, self.parallel_config.num_ubatches),\n"
            "            num_layers=self.model_config.get_total_num_hidden_layers(),\n"
            "        )\n"
            f"        {TELEMETRY_MARK}\n"
            f"        {MILESTONE2_MARK}\n"
        ),
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: alpha buffer anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v1 method anchor",
        ("    def _prepare_inputs(\n",),
        RUNNER_V1_METHOD + "\n    def _prepare_inputs(\n",
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: method anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v1 execute-model anchor",
        (
            "        # Set cudagraph mode to none if calc_kv_scales is true.\n",
        ),
        (
            "        self._set_weightless_alpha_rows(\n"
            "            req_ids, num_scheduled_tokens_np, num_tokens_padded,\n"
            "            ubatch_slices_padded,\n"
            "        )\n"
            "\n"
            "        # Set cudagraph mode to none if calc_kv_scales is true.\n"
        ),
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: execute-model anchor missing in {path}",
        )

    src, ok = _replace_one_of(
        src,
        "runner v1 trace collection",
        (
            "        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n",
        ),
        (
            "        trace_runtime = self._weightless_trace_runtime\n"
            "        if trace_runtime is not None:\n"
            "            steer_model = _weightless_steer_target_model(self.get_model())\n"
            "            if steer_model is not None and hasattr(\n"
            "                steer_model, \"_weightless_trace_values\"\n"
            "            ):\n"
            "                trace_runtime.collect(steer_model)\n"
            "\n"
            "        with record_function_or_nullcontext(\"gpu_model_runner: postprocess\"):\n"
        ),
    )
    if not ok:
        return (
            False,
            f"[steering-hotfix] xargs patch: trace collection anchor missing in {path}",
        )

    path.write_text(src)
    return True, f"[steering-hotfix] applied xargs patch to {path}"


def _apply_runtime_xargs_patches(model_path: Path) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    root = _find_vllm_root(model_path)
    if root is None:
        msgs.append(
            "[steering-hotfix] xargs patch: could not infer vllm root from "
            f"{model_path}; skipping runtime xargs patch"
        )
        return True, msgs

    installed_version = _installed_vllm_version(root)
    if installed_version is not None and installed_version != EXPECTED_VLLM_VERSION:
        return (
            False,
            [
                "[steering-hotfix] xargs patch requires vLLM "
                f"{EXPECTED_VLLM_VERSION}; found {installed_version} at {root}"
            ],
        )

    request_path = root / "v1/request.py"
    openai_protocol_path = root / "entrypoints/openai/engine/protocol.py"
    for expected_path in (openai_protocol_path, request_path):
        if expected_path.is_file():
            continue
        return (
            False,
            [
                "[steering-hotfix] xargs patch: expected runtime file not found: "
                f"{expected_path}"
            ],
        )

    ok, msg = install_runtime_module(root)
    msgs.append(msg)
    if not ok:
        return False, msgs

    for path, patch_fn in (
        (openai_protocol_path, _patch_openai_protocol_file),
        (request_path, _patch_request_file),
    ):
        ok, msg = patch_fn(path)
        msgs.append(msg)
        if not ok:
            return False, msgs

    patched_any_runner = False
    for path, patch_fn in (
        (root / "v1/worker/gpu/model_runner.py", _patch_runner_v2_file),
        (root / "v1/worker/gpu_model_runner.py", _patch_runner_v1_file),
    ):
        if not path.is_file():
            msgs.append(f"[steering-hotfix] xargs patch: runner file missing, skipping {path}")
            continue
        ok, msg = patch_fn(path)
        msgs.append(msg)
        if not ok:
            return False, msgs
        patched_any_runner = True

    if not patched_any_runner:
        return (
            False,
            msgs
            + [
                "[steering-hotfix] xargs patch: no supported GPU runner file "
                f"found under {root / 'v1/worker'}"
            ],
        )

    return True, msgs
