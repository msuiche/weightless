#!/usr/bin/env python3
"""Runtime adapter for exact vLLM v0.27.0.

This module contains only vLLM batching, device-buffer, and asynchronous trace
transport integration. Request policy and scalar formulas live in the
engine-neutral :mod:`weightless_runtime` package.
"""

import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

from weightless_runtime.controls import (
    TRACE_METRICS,
    TRACE_METRIC_INDEX,
    TRACE_SCHEMA_VERSION,
    _INTERNAL_LINEAGE_KEY,
    _INTERNAL_TRACE_ID_KEY,
    WeightlessEffectiveControl,
    WeightlessResolvedXArgs,
    WeightlessScheduleSegment,
    WeightlessServerLineage,
    WeightlessTraceConfig,
    _fill_weightless_alpha_slice,
    _policy_int,
    effective_weightless_control,
    prepare_weightless_openai_payload,
    resolve_weightless_vllm_xargs,
    static_weightless_layers,
    validate_weightless_vllm_xargs,
    weightless_alpha_at,
    weightless_config_from_sampling_params,
)
from weightless_runtime.telemetry import directional_scalar_metrics

EXPECTED_VLLM_VERSION = "0.27.0"
RUNTIME_MODULE_NAME = "_weightless_runtime_v027.py"

class WeightlessControlRunner:
    """Build bounded per-token alpha rows and per-request layer-mask banks."""

    def __init__(
        self,
        *,
        max_num_tokens: int,
        max_num_reqs: int,
        num_ubatches: int,
        num_layers: int,
    ) -> None:
        import torch

        self.max_num_tokens = max_num_tokens
        self.max_num_reqs = max_num_reqs
        self.num_ubatches = max(1, num_ubatches)
        self.num_layers = num_layers
        pin_memory = torch.cuda.is_available()
        self.alpha_full = torch.zeros(
            max_num_tokens, dtype=torch.float32, pin_memory=pin_memory
        )
        self.alpha_rows = torch.zeros(
            (self.num_ubatches, max_num_tokens),
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
        self.slot_full = torch.zeros(
            max_num_tokens, dtype=torch.int32, pin_memory=pin_memory
        )
        self.slot_rows = torch.zeros(
            (self.num_ubatches, max_num_tokens),
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.layer_bank = torch.zeros(
            (self.num_ubatches, max_num_reqs + 1, num_layers),
            dtype=torch.bool,
            pin_memory=pin_memory,
        )
        self._device_key: tuple[Any, Any] | None = None
        self._alpha_gpu = None
        self._slot_gpu = None
        self._bank_gpu = None

    def _ensure_device(self, model: Any) -> None:
        import torch

        model_rows = model._steer_alpha_rows
        key = (model_rows.device, model_rows.dtype)
        if key == self._device_key:
            return
        self._device_key = key
        self._alpha_gpu = torch.empty(
            self.alpha_rows.shape,
            device=model_rows.device,
            dtype=model_rows.dtype,
        )
        self._slot_gpu = torch.empty(
            self.slot_rows.shape,
            device=model_rows.device,
            dtype=torch.int32,
        )
        self._bank_gpu = torch.empty(
            self.layer_bank.shape,
            device=model_rows.device,
            dtype=torch.bool,
        )

    @staticmethod
    def _loaded_layers(model: Any) -> tuple[int, ...] | None:
        layers = getattr(model, "_weightless_steer_layer_ids", None)
        return None if layers is None else tuple(int(layer) for layer in layers)

    def _effective(
        self,
        model: Any,
        resolved: WeightlessResolvedXArgs,
    ) -> WeightlessEffectiveControl:
        default_alpha = float(getattr(model, "_steer_alpha_val", 1.0))
        loaded_layers = self._loaded_layers(model)
        default_layers = loaded_layers
        if default_layers is None:
            default_layers = static_weightless_layers()
        control = effective_weightless_control(
            resolved,
            default_alpha=default_alpha,
            default_layers=default_layers,
        )
        out_of_range = tuple(
            layer for layer in (control.layers or ()) if layer >= self.num_layers
        )
        if out_of_range:
            raise ValueError(
                "Weightless intervention layers exceed the model depth: "
                + ", ".join(str(layer) for layer in out_of_range)
            )
        if (
            resolved.layers_override is not None
            and loaded_layers is not None
            and not set(resolved.layers_override).issubset(loaded_layers)
        ):
            unavailable = sorted(set(resolved.layers_override).difference(loaded_layers))
            raise ValueError(
                "Weightless intervention layers are not loaded: "
                + ", ".join(str(layer) for layer in unavailable)
            )
        return control

    def _reset(self) -> None:
        self.alpha_full.zero_()
        self.alpha_rows.zero_()
        self.slot_full.zero_()
        self.slot_rows.zero_()
        self.layer_bank.zero_()

    def _add_request(
        self,
        *,
        model: Any,
        resolved: WeightlessResolvedXArgs,
        slot: int,
        global_start: int,
        token_count: int,
        start_ordinal: int,
        prompt_length: int,
    ) -> None:
        control = self._effective(model, resolved)
        token_slice = slice(global_start, global_start + token_count)
        _fill_weightless_alpha_slice(
            self.alpha_full[token_slice],
            control=control,
            start_ordinal=start_ordinal,
            prompt_length=prompt_length,
        )
        self.slot_full[token_slice].fill_(slot)
        if control.layers:
            self.layer_bank[:, slot, list(control.layers)] = True

    def _copy_to_model(self, model: Any) -> None:
        self._ensure_device(model)
        self._alpha_gpu.copy_(self.alpha_rows, non_blocking=True)
        if hasattr(model, "_set_weightless_control_rows"):
            self._slot_gpu.copy_(self.slot_rows, non_blocking=True)
            self._bank_gpu.copy_(self.layer_bank, non_blocking=True)
            model._set_weightless_control_rows(
                self._alpha_gpu,
                self._slot_gpu,
                self._bank_gpu,
            )
        else:
            if bool(self.slot_rows.any()) and any(
                resolved.layers_override is not None
                for resolved in getattr(self, "_current_controls", ())
            ):
                raise ValueError(
                    "this model lane does not support per-request layer masks"
                )
            model._set_weightless_alpha_rows(self._alpha_gpu)

    def prepare_v2(
        self,
        model: Any,
        input_batch: Any,
        control_by_req_id: dict[str, WeightlessResolvedXArgs],
    ) -> None:
        self._reset()
        self._current_controls = []
        cursor = 0
        for slot, (req_id, token_count_raw) in enumerate(
            zip(input_batch.req_ids, input_batch.num_scheduled_tokens), start=1
        ):
            token_count = int(token_count_raw)
            if slot > self.max_num_reqs or cursor + token_count > self.max_num_tokens:
                raise ValueError("Weightless control row capacity exceeded")
            resolved = control_by_req_id.get(req_id, WeightlessResolvedXArgs())
            self._current_controls.append(resolved)
            self._add_request(
                model=model,
                resolved=resolved,
                slot=slot,
                global_start=cursor,
                token_count=token_count,
                start_ordinal=int(input_batch.num_computed_tokens_np[slot - 1]),
                prompt_length=int(input_batch.prefill_len_np[slot - 1]),
            )
            cursor += token_count
        self.alpha_rows[0, :cursor].copy_(self.alpha_full[:cursor])
        self.slot_rows[0, :cursor].copy_(self.slot_full[:cursor])
        self._copy_to_model(model)

    def prepare_v1(
        self,
        model: Any,
        req_ids: list[str],
        num_scheduled_tokens: Any,
        num_tokens_after_padding: int,
        requests: dict[str, Any],
        ubatch_slices: Any,
    ) -> None:
        self._reset()
        self._current_controls = []
        cursor = 0
        for slot, (req_id, token_count_raw) in enumerate(
            zip(req_ids, num_scheduled_tokens), start=1
        ):
            token_count = int(token_count_raw)
            if slot > self.max_num_reqs or cursor + token_count > self.max_num_tokens:
                raise ValueError("Weightless control row capacity exceeded")
            request = requests.get(req_id)
            resolved = weightless_config_from_sampling_params(
                request.sampling_params if request is not None else None
            )
            self._current_controls.append(resolved)
            self._add_request(
                model=model,
                resolved=resolved,
                slot=slot,
                global_start=cursor,
                token_count=token_count,
                start_ordinal=(0 if request is None else int(request.num_computed_tokens)),
                prompt_length=(0 if request is None else int(request.num_prompt_tokens)),
            )
            cursor += token_count
        num_tokens = int(num_tokens_after_padding)
        if num_tokens > self.max_num_tokens:
            raise ValueError("Weightless padded row capacity exceeded")
        if ubatch_slices is None:
            self.alpha_rows[0, :num_tokens].copy_(self.alpha_full[:num_tokens])
            self.slot_rows[0, :num_tokens].copy_(self.slot_full[:num_tokens])
        else:
            for ubatch_id, ubatch_slice in enumerate(ubatch_slices):
                token_slice = ubatch_slice.token_slice
                width = token_slice.stop - token_slice.start
                self.alpha_rows[ubatch_id, :width].copy_(self.alpha_full[token_slice])
                self.slot_rows[ubatch_id, :width].copy_(self.slot_full[token_slice])
        self._copy_to_model(model)

class WeightlessTraceRunner:
    """Runner-side row mapping and bounded asynchronous JSONL trace sink."""

    def __init__(
        self,
        *,
        max_num_tokens: int,
        max_num_reqs: int,
        num_ubatches: int,
        num_layers: int,
    ) -> None:
        import torch

        self.max_num_tokens = max_num_tokens
        self.max_num_reqs = max_num_reqs
        self.num_ubatches = max(1, num_ubatches)
        self.num_layers = num_layers
        self.trace_dir = Path(os.environ["WEIGHTLESS_TRACE_DIR"])
        self.trace_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.trace_dir.chmod(0o700)
        self.slot_rows = torch.zeros(
            (self.num_ubatches, max_num_tokens),
            dtype=torch.int32,
            pin_memory=torch.cuda.is_available(),
        )
        self.layer_bank = torch.zeros(
            (self.num_ubatches, max_num_reqs + 1, num_layers),
            dtype=torch.bool,
            pin_memory=torch.cuda.is_available(),
        )
        self._current_plans: list[dict[str, Any]] = []
        self._banks: list[Any] = []
        self._device_banks: list[Any] = []
        self._index_banks: list[Any] = []
        self._events: list[Any] = []
        self._free_banks: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._work: queue.Queue[Any] = queue.Queue(maxsize=2)
        self._summary: dict[str, dict[str, Any]] = {}
        self._summary_lock = threading.Lock()
        self._dropped: dict[str, int] = {}
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="weightless-trace-writer",
            daemon=True,
        )
        self._writer.start()

    @staticmethod
    def _trace_config(sampling_params: Any) -> WeightlessTraceConfig | None:
        return weightless_config_from_sampling_params(sampling_params).trace

    def _reset_rows(self) -> None:
        self.slot_rows.zero_()
        self.layer_bank.zero_()
        self._current_plans = []

    def _add_request_plan(
        self,
        *,
        req_id: str,
        config: WeightlessTraceConfig,
        slot: int,
        global_start: int,
        token_count: int,
        num_computed_tokens: int,
        prompt_len: int,
        ubatch_slices: Any = None,
        available_layers: frozenset[int],
    ) -> None:
        requested_layers = tuple(
            layer for layer in config.layers if layer < self.num_layers
        )
        layers = tuple(layer for layer in requested_layers if layer in available_layers)
        unavailable_layers = tuple(
            layer for layer in config.layers if layer not in layers
        )
        for ubatch_id in range(self.num_ubatches):
            if layers:
                self.layer_bank[ubatch_id, slot, list(layers)] = True

        rows = []
        for offset in range(token_count):
            ordinal = num_computed_tokens + offset
            if ordinal % config.token_stride:
                continue
            global_row = global_start + offset
            if ubatch_slices is None:
                ubatch_id = 0
                local_row = global_row
            else:
                ubatch_id = -1
                local_row = -1
                for candidate_id, candidate_slice in enumerate(ubatch_slices):
                    token_slice = candidate_slice.token_slice
                    if token_slice.start <= global_row < token_slice.stop:
                        ubatch_id = candidate_id
                        local_row = global_row - token_slice.start
                        break
                if ubatch_id < 0:
                    continue
            self.slot_rows[ubatch_id, local_row] = slot
            phase = "prompt" if ordinal < prompt_len else "decode"
            status = "committed"
            if phase == "decode" and offset > 0:
                status = "candidate"
            rows.append((ubatch_id, local_row, ordinal, phase, status))

        self._current_plans.append(
            {
                "request_id": req_id,
                "config": config,
                "layers": layers,
                "unavailable_layers": unavailable_layers,
                "rows": tuple(rows),
            }
        )

    def prepare_v2(
        self,
        model: Any,
        input_batch: Any,
        trace_by_req_id: dict[str, WeightlessTraceConfig | None],
    ) -> None:
        self._reset_rows()
        available = frozenset(
            getattr(model, "_weightless_trace_rank1_layer_ids", ())
        )
        cursor = 0
        slot = 0
        for req_index, (req_id, token_count_raw) in enumerate(
            zip(input_batch.req_ids, input_batch.num_scheduled_tokens)
        ):
            token_count = int(token_count_raw)
            config = trace_by_req_id.get(req_id)
            if config is not None:
                slot += 1
                self._add_request_plan(
                    req_id=req_id,
                    config=config,
                    slot=slot,
                    global_start=cursor,
                    token_count=token_count,
                    num_computed_tokens=int(input_batch.num_computed_tokens_np[req_index]),
                    prompt_len=int(input_batch.prefill_len_np[req_index]),
                    available_layers=available,
                )
            cursor += token_count
        model._set_weightless_trace_rows(self.slot_rows, self.layer_bank)

    def prepare_v1(
        self,
        model: Any,
        req_ids: list[str],
        num_scheduled_tokens: Any,
        requests: dict[str, Any],
        ubatch_slices: Any,
    ) -> None:
        self._reset_rows()
        available = frozenset(
            getattr(model, "_weightless_trace_rank1_layer_ids", ())
        )
        cursor = 0
        slot = 0
        for req_id, token_count_raw in zip(req_ids, num_scheduled_tokens):
            token_count = int(token_count_raw)
            req_state = requests.get(req_id)
            config = self._trace_config(
                req_state.sampling_params if req_state is not None else None
            )
            if config is not None:
                slot += 1
                self._add_request_plan(
                    req_id=req_id,
                    config=config,
                    slot=slot,
                    global_start=cursor,
                    token_count=token_count,
                    num_computed_tokens=int(req_state.num_computed_tokens),
                    prompt_len=int(req_state.num_prompt_tokens),
                    ubatch_slices=ubatch_slices,
                    available_layers=available,
                )
            cursor += token_count
        model._set_weightless_trace_rows(self.slot_rows, self.layer_bank)

    def _ensure_banks(self, device_values: Any) -> None:
        if self._banks:
            return
        import torch

        pin_memory = device_values.device.type == "cuda"
        max_layers = min(
            self.num_layers,
            _policy_int("WEIGHTLESS_TRACE_MAX_LAYERS", 8),
        )
        self._sample_capacity = self.max_num_tokens * max_layers
        for bank_id in range(2):
            self._banks.append(
                torch.empty(
                    (self._sample_capacity, len(TRACE_METRICS)),
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=pin_memory,
                )
            )
            if pin_memory:
                self._device_banks.append(
                    torch.empty(
                        (self._sample_capacity, len(TRACE_METRICS)),
                        dtype=torch.float32,
                        device=device_values.device,
                    )
                )
                self._index_banks.append(
                    torch.empty(
                        self._sample_capacity,
                        dtype=torch.int64,
                        device=device_values.device,
                    )
                )
                self._events.append(torch.cuda.Event(blocking=True))
            self._free_banks.put(bank_id)

    def collect(self, model: Any) -> None:
        if not self._current_plans:
            return
        device_values = model._weightless_trace_values
        self._ensure_banks(device_values)
        sample_locations = [
            (ubatch_id, layer, local_row)
            for plan in self._current_plans
            for ubatch_id, local_row, _, _, _ in plan["rows"]
            for layer in plan["layers"]
        ]
        sample_count = len(sample_locations)
        if sample_count == 0:
            try:
                self._work.put_nowait(
                    (None, None, tuple(self._current_plans), 0)
                )
            except queue.Full:
                pass
            return
        if sample_count > self._sample_capacity:
            raise RuntimeError(
                "Weightless trace sample capacity exceeded: "
                f"{sample_count} > {self._sample_capacity}"
            )
        try:
            bank_id = self._free_banks.get_nowait()
        except queue.Empty:
            with self._summary_lock:
                for plan in self._current_plans:
                    trace_id = plan["config"].trace_id
                    dropped = len(plan["rows"]) * len(plan["layers"])
                    self._dropped[trace_id] = self._dropped.get(trace_id, 0) + dropped
            return

        bank = self._banks[bank_id]
        import torch

        layer_stride = int(device_values.shape[2])
        ubatch_stride = int(device_values.shape[1]) * layer_stride
        flat_indices = [
            ubatch_id * ubatch_stride + layer * layer_stride + local_row
            for ubatch_id, layer, local_row in sample_locations
        ]
        flat_values = device_values.view(-1, len(TRACE_METRICS))
        if device_values.device.type == "cuda":
            index_bank = self._index_banks[bank_id][:sample_count]
            index_bank.copy_(torch.tensor(flat_indices), non_blocking=True)
            device_bank = self._device_banks[bank_id][:sample_count]
            torch.index_select(flat_values, 0, index_bank, out=device_bank)
            bank[:sample_count].copy_(device_bank, non_blocking=True)
            event = self._events[bank_id]
            event.record(torch.cuda.current_stream(device_values.device))
        else:
            selected_values = flat_values.index_select(
                0, torch.tensor(flat_indices)
            )
            bank[:sample_count].copy_(selected_values)
            event = None
        item = (bank_id, event, tuple(self._current_plans), sample_count)
        try:
            self._work.put_nowait(item)
        except queue.Full:
            self._free_banks.put(bank_id)
            with self._summary_lock:
                for plan in self._current_plans:
                    trace_id = plan["config"].trace_id
                    dropped = len(plan["rows"]) * len(plan["layers"])
                    self._dropped[trace_id] = self._dropped.get(trace_id, 0) + dropped

    def _writer_loop(self) -> None:
        while True:
            bank_id, event, plans, sample_count = self._work.get()
            try:
                if event is not None:
                    event.synchronize()
                values = (
                    None
                    if bank_id is None
                    else self._banks[bank_id][:sample_count]
                )
                self._write_plans(values, plans)
            finally:
                if bank_id is not None:
                    self._free_banks.put(bank_id)
                self._work.task_done()

    @staticmethod
    def _lineage_dict(lineage: WeightlessServerLineage) -> dict[str, str | None]:
        return {
            "checkpoint_id": lineage.checkpoint_id,
            "parent_branch_id": lineage.parent_branch_id,
            "branch_id": lineage.branch_id,
            "history_fingerprint": lineage.history_fingerprint,
            "continuation_fingerprint": lineage.continuation_fingerprint,
        }

    def _summary_for(self, plan: dict[str, Any]) -> dict[str, Any]:
        config = plan["config"]
        summary = self._summary.get(config.trace_id)
        if summary is None:
            now = time.time()
            summary = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "trace_id": config.trace_id,
                "request_id": plan["request_id"],
                "requested_layers": list(config.layers),
                "observed_layers": [],
                "unavailable_layers": list(plan["unavailable_layers"]),
                "metrics": list(config.metrics),
                "token_stride": config.token_stride,
                "max_records": config.max_records,
                "record_count": 0,
                "token_count": 0,
                "metric_aggregates": {
                    metric: {
                        "count": 0,
                        "min": None,
                        "max": None,
                        "mean": None,
                    }
                    for metric in config.metrics
                },
                "truncated": False,
                "dropped_records": 0,
                "created_at": now,
                "updated_at": now,
                **self._lineage_dict(config.lineage),
            }
            self._summary[config.trace_id] = summary
        return summary

    def _write_plans(self, values: Any, plans: tuple[dict[str, Any], ...]) -> None:
        with self._summary_lock:
            value_cursor = 0
            for plan in plans:
                config = plan["config"]
                summary = self._summary_for(plan)
                dropped = self._dropped.pop(config.trace_id, 0)
                summary["dropped_records"] += dropped
                trace_path = self.trace_dir / f"{config.trace_id}.jsonl"
                token_ordinals: set[int] = set()
                records = []
                remaining = max(0, config.max_records - summary["record_count"])
                for ubatch_id, local_row, ordinal, phase, status in plan["rows"]:
                    for layer in plan["layers"]:
                        metric_row = values[value_cursor]
                        value_cursor += 1
                        if remaining == 0:
                            summary["truncated"] = True
                            continue
                        metric_values = {
                            metric: float(
                                metric_row[TRACE_METRIC_INDEX[metric]].item()
                            )
                            for metric in config.metrics
                        }
                        if any(not math.isfinite(value) for value in metric_values.values()):
                            summary["dropped_records"] += 1
                            continue
                        for metric, value in metric_values.items():
                            aggregate = summary["metric_aggregates"][metric]
                            count = aggregate["count"] + 1
                            previous_mean = aggregate["mean"]
                            aggregate["count"] = count
                            aggregate["min"] = (
                                value
                                if aggregate["min"] is None
                                else min(aggregate["min"], value)
                            )
                            aggregate["max"] = (
                                value
                                if aggregate["max"] is None
                                else max(aggregate["max"], value)
                            )
                            aggregate["mean"] = (
                                value
                                if previous_mean is None
                                else previous_mean + (value - previous_mean) / count
                            )
                        records.append(
                            {
                                "schema_version": TRACE_SCHEMA_VERSION,
                                "trace_id": config.trace_id,
                                "request_id": plan["request_id"],
                                "layer": layer,
                                "token_ordinal": ordinal,
                                "phase": phase,
                                "commit_status": status,
                                "metrics": metric_values,
                                **self._lineage_dict(config.lineage),
                            }
                        )
                        token_ordinals.add(ordinal)
                        remaining -= 1
                if records:
                    with trace_path.open("a", encoding="utf-8") as trace_file:
                        for record in records:
                            trace_file.write(json.dumps(record, separators=(",", ":")))
                            trace_file.write("\n")
                    trace_path.chmod(0o600)
                summary["record_count"] += len(records)
                summary["token_count"] += len(token_ordinals)
                summary["observed_layers"] = sorted(
                    set(summary["observed_layers"]).union(plan["layers"])
                )
                summary["updated_at"] = time.time()
                if summary["record_count"] >= config.max_records:
                    summary["truncated"] = True
                self._write_summary(summary)

    def _write_summary(self, summary: dict[str, Any]) -> None:
        path = self.trace_dir / f"{summary['trace_id']}.summary.json"
        temp = path.with_suffix(".summary.json.tmp")
        temp.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.chmod(0o600)
        temp.replace(path)


def make_weightless_trace_runner(**kwargs: Any) -> WeightlessTraceRunner | None:
    if not (os.environ.get("WEIGHTLESS_TRACE_DIR") or "").strip():
        return None
    return WeightlessTraceRunner(**kwargs)
