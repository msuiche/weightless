#!/usr/bin/env python3
"""CPU tests for Weightless Milestone 1 telemetry on vLLM v0.27.0."""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from patches.vllm.v0_27_0 import runtime


class ModelStub:
    def __init__(self, num_ubatches: int, num_layers: int, max_tokens: int):
        self._weightless_trace_rank1_layer_ids = (1, 3)
        self._weightless_trace_values = torch.full(
            (num_ubatches, num_layers, max_tokens, len(runtime.TRACE_METRICS)),
            float("nan"),
            dtype=torch.float32,
        )
        self.slot_rows = None
        self.layer_bank = None

    def _set_weightless_trace_rows(self, slot_rows, layer_bank):
        self.slot_rows = slot_rows.clone()
        self.layer_bank = layer_bank.clone()


class NoReadModel:
    @property
    def _weightless_trace_values(self):
        raise AssertionError("disabled trace collection touched model output")


class UBatchSlice:
    def __init__(self, start: int, stop: int):
        self.token_slice = slice(start, stop)


def sampling_params(vllm_xargs):
    return SimpleNamespace(extra_args=vllm_xargs)


def trace_request(*, layers=(1, 3), stride=1, max_records=32, alpha=None):
    controls = {} if alpha is None else {"alpha": alpha}
    return {
        "weightless": {
            "version": 1,
            "controls": controls,
            "trace": {
                "metrics": list(runtime.TRACE_METRICS),
                "layers": list(layers),
                "token_stride": stride,
                "max_records": max_records,
            },
        }
    }


def main() -> int:
    failures = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal failures
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {label}"
            + (f" — {detail}" if detail and not ok else "")
        )
        failures += 0 if ok else 1

    old_env = {
        name: os.environ.get(name)
        for name in (
            "WEIGHTLESS_TRACE_DIR",
            "WEIGHTLESS_TRACE_MAX_LAYERS",
            "WEIGHTLESS_TRACE_MAX_RECORDS",
        )
    }
    try:
        os.environ.pop("WEIGHTLESS_TRACE_DIR", None)
        try:
            runtime.validate_weightless_vllm_xargs(
                {"vllm_xargs": trace_request()}
            )
        except ValueError as exc:
            disabled_rejected = "tracing is disabled" in str(exc)
        else:
            disabled_rejected = False
        check(disabled_rejected, "trace requests fail while server tracing is disabled")

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ["WEIGHTLESS_TRACE_DIR"] = temp_dir
            os.environ["WEIGHTLESS_TRACE_MAX_LAYERS"] = "3"
            os.environ["WEIGHTLESS_TRACE_MAX_RECORDS"] = "64"

            valid = trace_request(alpha=0.5)
            try:
                runtime.validate_weightless_vllm_xargs({"vllm_xargs": valid})
            except ValueError as exc:
                valid_schema = False
                valid_detail = str(exc)
            else:
                valid_schema = True
                valid_detail = ""
            check(valid_schema, "versioned structured trace request is accepted", valid_detail)

            invalid_requests = (
                {"weightless": {"version": 2, "trace": {}}},
                {
                    "weightless": {
                        "version": 1,
                        "lineage": {},
                        "trace": {"layers": [1]},
                    }
                },
                {
                    "weightless": {
                        "version": 1,
                        "trace": {"layers": [1], "metrics": ["hidden_state"]},
                    }
                },
                {
                    "weightless": {
                        "version": 1,
                        "trace": {"layers": [1, 2, 3, 4]},
                    }
                },
                {
                    "weightless": {
                        "version": 1,
                        "trace": {"layers": [1], "max_records": 65},
                    }
                },
                {
                    "weightless_alpha": 0.5,
                    "weightless": {
                        "version": 1,
                        "controls": {"alpha": 1.0},
                    },
                },
            )
            for index, invalid in enumerate(invalid_requests):
                try:
                    runtime.validate_weightless_vllm_xargs(
                        {"vllm_xargs": invalid}
                    )
                except ValueError:
                    rejected = True
                else:
                    rejected = False
                check(rejected, f"invalid structured request {index + 1} is rejected")

            equal_controls = trace_request(alpha=0.5)
            equal_controls["weightless_alpha"] = 0.5
            resolved = runtime.resolve_weightless_vllm_xargs(equal_controls)
            check(
                resolved.alpha_override == 0.5,
                "equal legacy and structured alpha canonicalize together",
            )

            for coefficient, residual_norm, alpha in (
                (2.0, 4.0, 0.0),
                (-3.0, 5.0, 0.5),
                (1.25, 2.5, 1.0),
                (0.75, 3.0, 2.5),
            ):
                metrics = runtime.directional_scalar_metrics(
                    coefficient, residual_norm, alpha
                )
                reference = np.array(
                    [
                        coefficient,
                        (1 - alpha) * coefficient,
                        coefficient / residual_norm,
                        residual_norm,
                        (coefficient / residual_norm) ** 2,
                        alpha,
                        abs(alpha * coefficient),
                    ],
                    dtype=np.float64,
                )
                actual = np.array(
                    [metrics[name] for name in runtime.TRACE_METRICS],
                    dtype=np.float64,
                )
                check(
                    np.allclose(actual, reference, rtol=0, atol=1e-14),
                    f"directional formulas match reference at alpha={alpha}",
                )
                check(
                    math.isclose(
                        metrics["post_coefficient"],
                        (1 - alpha) * coefficient,
                    ),
                    f"project post-coefficient is analytic at alpha={alpha}",
                )

            lineage = runtime.WeightlessServerLineage(
                checkpoint_id="cp-test",
                parent_branch_id="branch-parent",
                branch_id="branch-child",
                history_fingerprint="history-test",
                continuation_fingerprint="continuation-test",
            )
            lineage_args = trace_request(layers=(1,))
            lineage_args[runtime._INTERNAL_LINEAGE_KEY] = lineage
            lineage_args[runtime._INTERNAL_TRACE_ID_KEY] = "client-forged"
            resolved = runtime.resolve_weightless_vllm_xargs(
                lineage_args, assign_trace_id=True
            )
            check(
                resolved.trace is not None
                and resolved.trace.trace_id != "client-forged"
                and resolved.trace.lineage == lineage,
                "server lineage round-trips and trace IDs cannot be forged",
            )

            manager = runtime.WeightlessTraceRunner(
                max_num_tokens=8,
                max_num_reqs=3,
                num_ubatches=2,
                num_layers=4,
            )
            model = ModelStub(2, 4, 8)
            trace_args = trace_request(stride=2, max_records=8)
            trace_config = runtime.resolve_weightless_vllm_xargs(
                trace_args, assign_trace_id=True
            ).trace
            assert trace_config is not None
            input_batch = SimpleNamespace(
                req_ids=["traced", "plain"],
                num_scheduled_tokens=np.array([3, 2], dtype=np.int32),
                num_computed_tokens_np=np.array([0, 7], dtype=np.int32),
                prefill_len_np=np.array([2, 4], dtype=np.int32),
            )
            manager.prepare_v2(
                model,
                input_batch,
                {"traced": trace_config, "plain": None},
            )
            check(
                torch.equal(
                    model.slot_rows[0, :5],
                    torch.tensor([1, 0, 1, 0, 0], dtype=torch.int32),
                ),
                "V2 trace slots follow request rows and token stride",
            )
            check(
                bool(model.layer_bank[0, 1, 1])
                and bool(model.layer_bank[0, 1, 3])
                and not bool(model.layer_bank[0, 0].any()),
                "V2 layer bank isolates the traced request slot",
            )

            trace_args_v1 = trace_request(layers=(1,), max_records=8)
            trace_config_v1 = runtime.resolve_weightless_vllm_xargs(
                trace_args_v1, assign_trace_id=True
            ).trace
            assert trace_config_v1 is not None
            requests = {
                "traced-v1": SimpleNamespace(
                    sampling_params=sampling_params(trace_args_v1),
                    num_computed_tokens=5,
                    num_prompt_tokens=6,
                ),
                "plain-v1": SimpleNamespace(
                    sampling_params=sampling_params({}),
                    num_computed_tokens=9,
                    num_prompt_tokens=4,
                ),
            }
            manager.prepare_v1(
                model,
                ["traced-v1", "plain-v1"],
                np.array([3, 2], dtype=np.int32),
                requests,
                [UBatchSlice(0, 2), UBatchSlice(2, 5)],
            )
            check(
                torch.equal(
                    model.slot_rows[:, :3],
                    torch.tensor(
                        [[1, 1, 0], [1, 0, 0]], dtype=torch.int32
                    ),
                ),
                "V1 trace slots preserve DBO-local token rows",
            )

            config_v1 = manager._current_plans[0]["config"]
            for plan in manager._current_plans:
                for ubatch_id, local_row, _, _, _ in plan["rows"]:
                    for layer in plan["layers"]:
                        values = runtime.directional_scalar_metrics(2.0, 4.0, 0.5)
                        for metric, value in values.items():
                            model._weightless_trace_values[
                                ubatch_id,
                                layer,
                                local_row,
                                runtime.TRACE_METRIC_INDEX[metric],
                            ] = value
            manager.collect(model)
            manager._work.join()
            check(
                manager._banks[0].numel()
                < model._weightless_trace_values.numel(),
                "device-to-host staging is compact rather than full-buffer",
            )
            trace_path = pathlib.Path(temp_dir) / f"{config_v1.trace_id}.jsonl"
            summary_path = (
                pathlib.Path(temp_dir) / f"{config_v1.trace_id}.summary.json"
            )
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            summary = json.loads(summary_path.read_text())
            check(
                len(records) == 3
                and [record["token_ordinal"] for record in records] == [5, 6, 7],
                "sidecar writes monotonic token ordinals",
            )
            check(
                records[0]["phase"] == "prompt"
                and records[1]["phase"] == "decode"
                and records[2]["commit_status"] == "candidate",
                "sidecar distinguishes prompt, decode, and speculative candidates",
            )
            check(
                summary["record_count"] == 3
                and summary["observed_layers"] == [1]
                and summary["checkpoint_id"] is None
                and summary["metric_aggregates"]["coefficient"]
                == {"count": 3, "min": 2.0, "max": 2.0, "mean": 2.0},
                "trace summary records bounds, layers, and empty ordinary lineage",
            )
            check(
                (trace_path.stat().st_mode & 0o777) == 0o600
                and (summary_path.stat().st_mode & 0o777) == 0o600,
                "trace artifacts are owner-only",
            )

            manager._current_plans = []
            try:
                manager.collect(NoReadModel())
            except AssertionError:
                no_copy = False
            else:
                no_copy = True
            check(no_copy, "disabled request path performs no device-to-host read")
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    print()
    print(
        "weightless telemetry v0.27: "
        + ("all checks passed" if not failures else f"{failures} failure(s)")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
