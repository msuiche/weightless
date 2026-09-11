#!/usr/bin/env python3
"""Engine-neutral request controls and schedule semantics for Weightless."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import secrets
from typing import Any

TRACE_SCHEMA_VERSION = 1
TRACE_METRICS = (
    "coefficient",
    "post_coefficient",
    "normalized_coefficient",
    "residual_norm",
    "directional_energy",
    "effective_alpha",
    "delta_norm",
)
TRACE_METRIC_INDEX = {name: index for index, name in enumerate(TRACE_METRICS)}
_INTERNAL_TRACE_ID_KEY = "__weightless_trace_id"
_INTERNAL_LINEAGE_KEY = "__weightless_server_lineage"
_TRACE_ROOT_FIELDS = frozenset({"version", "controls", "trace"})
_CONTROL_FIELDS = frozenset(
    {"alpha", "prefill_alpha", "decode_alpha", "layers", "schedule"}
)
_SCHEDULE_SEGMENT_FIELDS = frozenset(
    {"start", "end", "alpha", "start_alpha", "end_alpha"}
)
_TRACE_CONFIG_FIELDS = frozenset(
    {"metrics", "layers", "token_stride", "max_records"}
)

@dataclass(frozen=True, slots=True)
class WeightlessServerLineage:
    checkpoint_id: str | None = None
    parent_branch_id: str | None = None
    branch_id: str | None = None
    history_fingerprint: str | None = None
    continuation_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class WeightlessTraceConfig:
    trace_id: str
    metrics: tuple[str, ...]
    layers: tuple[int, ...]
    token_stride: int
    max_records: int
    lineage: WeightlessServerLineage


@dataclass(frozen=True, slots=True)
class WeightlessScheduleSegment:
    start: int
    end: int
    start_alpha: float
    end_alpha: float

    @property
    def is_constant(self) -> bool:
        return self.start_alpha == self.end_alpha


@dataclass(frozen=True, slots=True)
class WeightlessResolvedXArgs:
    alpha_override: float | None = None
    trace: WeightlessTraceConfig | None = None
    prefill_alpha_override: float | None = None
    decode_alpha_override: float | None = None
    layers_override: tuple[int, ...] | None = None
    schedule: tuple[WeightlessScheduleSegment, ...] = ()

    @property
    def has_advanced_controls(self) -> bool:
        return (
            self.prefill_alpha_override is not None
            or self.decode_alpha_override is not None
            or self.layers_override is not None
            or bool(self.schedule)
        )


@dataclass(frozen=True, slots=True)
class WeightlessEffectiveControl:
    base_alpha: float
    prefill_alpha: float
    decode_alpha: float
    layers: tuple[int, ...] | None
    schedule: tuple[WeightlessScheduleSegment, ...]


def _finite_float(raw: Any, field: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{field} must be a finite int/float scalar.")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite, got {raw!r}.")
    return value


def _positive_int(raw: Any, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return raw


def _non_negative_int(raw: Any, field: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return raw


def _policy_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}.")
    return value


def _enabled(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _parse_layer_token(raw: str, field: str) -> list[int]:
    token = raw.strip()
    if not token:
        raise ValueError(f"{field} contains an empty layer token.")
    if "-" not in token:
        try:
            return [_non_negative_int(int(token), field)]
        except ValueError as exc:
            raise ValueError(f"{field} contains invalid layer {token!r}.") from exc
    start_raw, end_raw = token.split("-", 1)
    try:
        start = _non_negative_int(int(start_raw), field)
        end = _non_negative_int(int(end_raw), field)
    except ValueError as exc:
        raise ValueError(f"{field} contains invalid range {token!r}.") from exc
    if end < start:
        raise ValueError(f"{field} contains descending range {token!r}.")
    return list(range(start, end + 1))


def _parse_control_layers(raw: Any, field: str) -> tuple[int, ...]:
    if isinstance(raw, str):
        layers = [
            layer
            for token in raw.split(",")
            for layer in _parse_layer_token(token, field)
        ]
    elif isinstance(raw, list):
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in raw):
            raise ValueError(f"{field} entries must be non-negative integers.")
        layers = list(raw)
    else:
        raise ValueError(f"{field} must be an integer list or compact range string.")
    if any(layer < 0 for layer in layers):
        raise ValueError(f"{field} entries must be non-negative integers.")
    if len(layers) != len(set(layers)):
        raise ValueError(f"{field} must not contain duplicate layers.")
    max_layers = _policy_int("WEIGHTLESS_CONTROL_MAX_LAYERS", 128)
    if len(layers) > max_layers:
        raise ValueError(
            f"{field} exceeds WEIGHTLESS_CONTROL_MAX_LAYERS={max_layers}."
        )
    max_layer_id = _policy_int("WEIGHTLESS_CONTROL_MAX_LAYER_ID", 1023)
    if layers and max(layers) > max_layer_id:
        raise ValueError(f"{field} contains a layer above {max_layer_id}.")
    return tuple(sorted(layers))


def _validate_control_layers_enabled(layers: tuple[int, ...] | None) -> None:
    configured_layers_raw = (
        os.environ.get("WEIGHTLESS_STEER_LAYERS") or ""
    ).strip()
    if layers is None or not configured_layers_raw:
        return
    configured_layers = _parse_control_layers(
        configured_layers_raw,
        "WEIGHTLESS_STEER_LAYERS",
    )
    unavailable = tuple(layer for layer in layers if layer not in configured_layers)
    if unavailable:
        raise ValueError(
            "Weightless intervention layers are not enabled by the server: "
            + ", ".join(str(layer) for layer in unavailable)
        )


def _parse_schedule(raw: Any) -> tuple[WeightlessScheduleSegment, ...]:
    field = "vllm_xargs.weightless.controls.schedule"
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list.")
    max_segments = _policy_int("WEIGHTLESS_CONTROL_MAX_SEGMENTS", 8)
    if len(raw) > max_segments:
        raise ValueError(f"{field} exceeds server limit {max_segments}.")
    max_position = _policy_int("WEIGHTLESS_CONTROL_MAX_POSITION", 1048576)
    segments: list[WeightlessScheduleSegment] = []
    for index, item in enumerate(raw):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{item_field} must be an object.")
        unknown = set(item).difference(_SCHEDULE_SEGMENT_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown {item_field} field(s): " + ", ".join(sorted(unknown))
            )
        start = _non_negative_int(item.get("start"), f"{item_field}.start")
        end = _non_negative_int(item.get("end"), f"{item_field}.end")
        if end <= start:
            raise ValueError(f"{item_field}.end must be greater than start.")
        if end > max_position:
            raise ValueError(f"{item_field}.end exceeds server limit {max_position}.")
        has_constant = "alpha" in item
        has_ramp = "start_alpha" in item or "end_alpha" in item
        if has_constant == has_ramp:
            raise ValueError(
                f"{item_field} must define either alpha or both "
                "start_alpha/end_alpha."
            )
        if has_constant:
            start_alpha = end_alpha = _finite_float(
                item["alpha"], f"{item_field}.alpha"
            )
        else:
            if "start_alpha" not in item or "end_alpha" not in item:
                raise ValueError(
                    f"{item_field} must define both start_alpha and end_alpha."
                )
            start_alpha = _finite_float(
                item["start_alpha"], f"{item_field}.start_alpha"
            )
            end_alpha = _finite_float(
                item["end_alpha"], f"{item_field}.end_alpha"
            )
            if end - start == 1 and start_alpha != end_alpha:
                raise ValueError(
                    f"{item_field} cannot ramp across a one-token segment."
                )
        segments.append(
            WeightlessScheduleSegment(start, end, start_alpha, end_alpha)
        )
    segments.sort(key=lambda segment: (segment.start, segment.end))
    for previous, current in zip(segments, segments[1:]):
        if current.start < previous.end:
            raise ValueError(f"{field} segments must not overlap.")
    return tuple(segments)


def _parse_trace_config(
    raw: Any,
    *,
    trace_id: str,
    lineage: WeightlessServerLineage,
) -> WeightlessTraceConfig:
    if not isinstance(raw, dict):
        raise ValueError("vllm_xargs.weightless.trace must be an object.")
    unknown = set(raw).difference(_TRACE_CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            "unknown vllm_xargs.weightless.trace field(s): "
            + ", ".join(sorted(unknown))
        )

    metrics_raw = raw.get("metrics", list(TRACE_METRICS))
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise ValueError(
            "vllm_xargs.weightless.trace.metrics must be a non-empty list."
        )
    if any(not isinstance(metric, str) for metric in metrics_raw):
        raise ValueError(
            "vllm_xargs.weightless.trace.metrics entries must be strings."
        )
    unknown_metrics = set(metrics_raw).difference(TRACE_METRICS)
    if unknown_metrics:
        raise ValueError(
            "unsupported Weightless trace metric(s): "
            + ", ".join(sorted(unknown_metrics))
        )
    metrics = tuple(metric for metric in TRACE_METRICS if metric in metrics_raw)

    layers_raw = raw.get("layers")
    if not isinstance(layers_raw, list) or not layers_raw:
        raise ValueError(
            "vllm_xargs.weightless.trace.layers must be a non-empty list."
        )
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
        for layer in layers_raw
    ):
        raise ValueError(
            "vllm_xargs.weightless.trace.layers entries must be non-negative integers."
        )
    if len(set(layers_raw)) != len(layers_raw):
        raise ValueError(
            "vllm_xargs.weightless.trace.layers must not contain duplicates."
        )
    max_layers = _policy_int("WEIGHTLESS_TRACE_MAX_LAYERS", 8)
    if len(layers_raw) > max_layers:
        raise ValueError(
            "vllm_xargs.weightless.trace.layers exceeds server limit "
            f"WEIGHTLESS_TRACE_MAX_LAYERS={max_layers}."
        )
    max_layer_id = _policy_int("WEIGHTLESS_TRACE_MAX_LAYER_ID", 1023)
    if max(layers_raw) > max_layer_id:
        raise ValueError(
            "vllm_xargs.weightless.trace.layers contains a layer above server "
            f"limit {max_layer_id}."
        )

    token_stride = _positive_int(
        raw.get("token_stride", 1),
        "vllm_xargs.weightless.trace.token_stride",
    )
    max_stride = _policy_int("WEIGHTLESS_TRACE_MAX_TOKEN_STRIDE", 65536)
    if token_stride > max_stride:
        raise ValueError(
            "vllm_xargs.weightless.trace.token_stride exceeds server limit "
            f"{max_stride}."
        )

    server_max_records = _policy_int("WEIGHTLESS_TRACE_MAX_RECORDS", 4096)
    max_records = _positive_int(
        raw.get("max_records", server_max_records),
        "vllm_xargs.weightless.trace.max_records",
    )
    if max_records > server_max_records:
        raise ValueError(
            "vllm_xargs.weightless.trace.max_records exceeds server limit "
            f"WEIGHTLESS_TRACE_MAX_RECORDS={server_max_records}."
        )

    return WeightlessTraceConfig(
        trace_id=trace_id,
        metrics=metrics,
        layers=tuple(layers_raw),
        token_stride=token_stride,
        max_records=max_records,
        lineage=lineage,
    )


def resolve_weightless_vllm_xargs(
    vllm_xargs: Any,
    *,
    assign_trace_id: bool = False,
) -> WeightlessResolvedXArgs:
    if not isinstance(vllm_xargs, dict):
        return WeightlessResolvedXArgs(None, None)

    legacy_alpha = None
    if "weightless_alpha" in vllm_xargs:
        legacy_alpha = _finite_float(
            vllm_xargs["weightless_alpha"],
            "vllm_xargs.weightless_alpha",
        )

    legacy_layers = None
    if "weightless_layers" in vllm_xargs:
        legacy_layers = _parse_control_layers(
            vllm_xargs["weightless_layers"],
            "vllm_xargs.weightless_layers",
        )

    structured = vllm_xargs.get("weightless")
    if structured is None:
        _validate_control_layers_enabled(legacy_layers)
        resolved = WeightlessResolvedXArgs(
            alpha_override=legacy_alpha,
            layers_override=legacy_layers,
        )
        if resolved.has_advanced_controls and not _enabled(
            "WEIGHTLESS_ENABLE_MILESTONE_2"
        ):
            raise ValueError(
                "Weightless Milestone 2 controls are disabled; the server must "
                "set WEIGHTLESS_ENABLE_MILESTONE_2=1."
            )
        return resolved
    if isinstance(structured, str):
        try:
            structured = json.loads(structured)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "vllm_xargs.weightless must be an object."
            ) from exc
    if not isinstance(structured, dict):
        raise ValueError("vllm_xargs.weightless must be an object.")
    unknown = set(structured).difference(_TRACE_ROOT_FIELDS)
    if unknown:
        raise ValueError(
            "unknown vllm_xargs.weightless field(s): "
            + ", ".join(sorted(unknown))
        )
    version = structured.get("version")
    if isinstance(version, bool) or version != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"vllm_xargs.weightless.version must be {TRACE_SCHEMA_VERSION}."
        )

    controls = structured.get("controls", {})
    if not isinstance(controls, dict):
        raise ValueError("vllm_xargs.weightless.controls must be an object.")
    unknown_controls = set(controls).difference(_CONTROL_FIELDS)
    if unknown_controls:
        raise ValueError(
            "unknown vllm_xargs.weightless.controls field(s): "
            + ", ".join(sorted(unknown_controls))
        )
    structured_alpha = None
    if "alpha" in controls:
        structured_alpha = _finite_float(
            controls["alpha"],
            "vllm_xargs.weightless.controls.alpha",
        )
    if (
        legacy_alpha is not None
        and structured_alpha is not None
        and legacy_alpha != structured_alpha
    ):
        raise ValueError(
            "weightless_alpha conflicts with vllm_xargs.weightless.controls.alpha."
        )
    alpha_override = (
        structured_alpha if structured_alpha is not None else legacy_alpha
    )

    prefill_alpha = None
    if "prefill_alpha" in controls:
        prefill_alpha = _finite_float(
            controls["prefill_alpha"],
            "vllm_xargs.weightless.controls.prefill_alpha",
        )
    decode_alpha = None
    if "decode_alpha" in controls:
        decode_alpha = _finite_float(
            controls["decode_alpha"],
            "vllm_xargs.weightless.controls.decode_alpha",
        )
    structured_layers = None
    if "layers" in controls:
        structured_layers = _parse_control_layers(
            controls["layers"],
            "vllm_xargs.weightless.controls.layers",
        )
    if (
        legacy_layers is not None
        and structured_layers is not None
        and legacy_layers != structured_layers
    ):
        raise ValueError(
            "weightless_layers conflicts with "
            "vllm_xargs.weightless.controls.layers."
        )
    layers_override = (
        structured_layers if structured_layers is not None else legacy_layers
    )
    _validate_control_layers_enabled(layers_override)
    schedule = _parse_schedule(controls.get("schedule", []))
    resolved_fields = {
        "alpha_override": alpha_override,
        "prefill_alpha_override": prefill_alpha,
        "decode_alpha_override": decode_alpha,
        "layers_override": layers_override,
        "schedule": schedule,
    }
    advanced = WeightlessResolvedXArgs(**resolved_fields).has_advanced_controls
    if advanced and not _enabled("WEIGHTLESS_ENABLE_MILESTONE_2"):
        raise ValueError(
            "Weightless Milestone 2 controls are disabled; the server must set "
            "WEIGHTLESS_ENABLE_MILESTONE_2=1."
        )

    trace_raw = structured.get("trace")
    if trace_raw is None:
        return WeightlessResolvedXArgs(**resolved_fields)
    trace_dir = (os.environ.get("WEIGHTLESS_TRACE_DIR") or "").strip()
    if not trace_dir:
        raise ValueError(
            "Weightless tracing is disabled; the server must set "
            "WEIGHTLESS_TRACE_DIR."
        )

    lineage_raw = vllm_xargs.get(_INTERNAL_LINEAGE_KEY)
    if lineage_raw is None:
        lineage = WeightlessServerLineage()
    elif isinstance(lineage_raw, WeightlessServerLineage):
        lineage = lineage_raw
    else:
        raise ValueError("Weightless lineage is server-owned and cannot be supplied.")

    if assign_trace_id:
        trace_id = "wt-" + secrets.token_urlsafe(18)
        vllm_xargs[_INTERNAL_TRACE_ID_KEY] = trace_id
    else:
        trace_id_raw = vllm_xargs.get(_INTERNAL_TRACE_ID_KEY)
        trace_id = trace_id_raw if isinstance(trace_id_raw, str) else "validation"
    return WeightlessResolvedXArgs(
        trace=_parse_trace_config(trace_raw, trace_id=trace_id, lineage=lineage),
        **resolved_fields,
    )


def prepare_weightless_openai_payload(data: Any) -> None:
    if not isinstance(data, dict):
        return
    vllm_xargs = data.get("vllm_xargs")
    resolve_weightless_vllm_xargs(vllm_xargs)
    if not isinstance(vllm_xargs, dict):
        return
    structured = vllm_xargs.get("weightless")
    if not isinstance(structured, dict):
        return
    normalized_xargs = dict(vllm_xargs)
    normalized_xargs["weightless"] = json.dumps(
        structured,
        separators=(",", ":"),
        sort_keys=True,
    )
    data["vllm_xargs"] = normalized_xargs


def validate_weightless_vllm_xargs(data: Any) -> None:
    if not isinstance(data, dict):
        return
    resolve_weightless_vllm_xargs(data.get("vllm_xargs"))


def weightless_config_from_sampling_params(
    sampling_params: Any,
    *,
    assign_trace_id: bool = False,
) -> WeightlessResolvedXArgs:
    if sampling_params is None:
        return WeightlessResolvedXArgs(None, None)
    extra_args = getattr(sampling_params, "extra_args", None)
    return resolve_weightless_vllm_xargs(
        extra_args,
        assign_trace_id=assign_trace_id,
    )


def static_weightless_layers() -> tuple[int, ...] | None:
    raw = (os.environ.get("WEIGHTLESS_STEER_LAYERS") or "").strip()
    if not raw:
        return None
    return _parse_control_layers(raw, "WEIGHTLESS_STEER_LAYERS")


def effective_weightless_control(
    resolved: WeightlessResolvedXArgs,
    *,
    default_alpha: float,
    default_layers: tuple[int, ...] | None,
) -> WeightlessEffectiveControl:
    base_alpha = (
        default_alpha
        if resolved.alpha_override is None
        else resolved.alpha_override
    )
    prefill_alpha = (
        base_alpha
        if resolved.prefill_alpha_override is None
        else resolved.prefill_alpha_override
    )
    decode_alpha = (
        base_alpha
        if resolved.decode_alpha_override is None
        else resolved.decode_alpha_override
    )
    layers = (
        default_layers
        if resolved.layers_override is None
        else resolved.layers_override
    )
    return WeightlessEffectiveControl(
        base_alpha=base_alpha,
        prefill_alpha=prefill_alpha,
        decode_alpha=decode_alpha,
        layers=layers,
        schedule=resolved.schedule,
    )


def weightless_alpha_at(
    control: WeightlessEffectiveControl,
    *,
    token_ordinal: int,
    prompt_length: int,
) -> float:
    decision_ordinal = max(0, prompt_length - 1)
    if token_ordinal < decision_ordinal:
        return control.prefill_alpha
    generation_position = token_ordinal - decision_ordinal
    for segment in control.schedule:
        if segment.start <= generation_position < segment.end:
            if segment.is_constant:
                return segment.start_alpha
            fraction = (generation_position - segment.start) / (
                segment.end - segment.start - 1
            )
            return segment.start_alpha + fraction * (
                segment.end_alpha - segment.start_alpha
            )
    return control.decode_alpha


def _fill_weightless_alpha_slice(
    destination: Any,
    *,
    control: WeightlessEffectiveControl,
    start_ordinal: int,
    prompt_length: int,
) -> None:
    import torch

    token_count = int(destination.shape[0])
    if token_count == 0:
        return
    decision_ordinal = max(0, prompt_length - 1)
    end_ordinal = start_ordinal + token_count
    destination.fill_(control.decode_alpha)
    prefill_end = min(end_ordinal, decision_ordinal)
    if prefill_end > start_ordinal:
        destination[: prefill_end - start_ordinal].fill_(control.prefill_alpha)
    for segment in control.schedule:
        segment_start = decision_ordinal + segment.start
        segment_end = decision_ordinal + segment.end
        overlap_start = max(start_ordinal, segment_start)
        overlap_end = min(end_ordinal, segment_end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - start_ordinal
        local_end = overlap_end - start_ordinal
        if segment.is_constant:
            destination[local_start:local_end].fill_(segment.start_alpha)
            continue
        positions = torch.arange(
            overlap_start - segment_start,
            overlap_end - segment_start,
            dtype=destination.dtype,
            device=destination.device,
        )
        values = segment.start_alpha + positions * (
            (segment.end_alpha - segment.start_alpha)
            / (segment.end - segment.start - 1)
        )
        destination[local_start:local_end].copy_(values)
