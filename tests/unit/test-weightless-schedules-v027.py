#!/usr/bin/env python3
"""CPU tests for Weightless Milestone 2 control resolution and row mapping."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNTIME = REPO / "patches/vllm/v0_27_0/runtime.py"


def load_runtime():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("weightless_runtime_v027", RUNTIME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelStub:
    def __init__(self, *, rows: int = 2, width: int = 12, layers=(10, 30, 58)):
        self._steer_alpha_val = 0.5
        self._steer_alpha_rows = torch.empty((rows, width), dtype=torch.float32)
        self._weightless_steer_layer_ids = tuple(layers)
        self.alpha_rows = None
        self.slot_rows = None
        self.layer_bank = None

    def _set_weightless_control_rows(self, alpha_rows, slot_rows, layer_bank):
        self.alpha_rows = alpha_rows.clone()
        self.slot_rows = slot_rows.clone()
        self.layer_bank = layer_bank.clone()


def structured(controls):
    return {"weightless": {"version": 1, "controls": controls}}


def main() -> int:
    runtime = load_runtime()
    failures = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal failures
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {label}"
            + (f" — {detail}" if detail and not ok else "")
        )
        failures += 0 if ok else 1

    previous_gate = os.environ.get("WEIGHTLESS_ENABLE_MILESTONE_2")
    original_static_layers = os.environ.get("WEIGHTLESS_STEER_LAYERS")
    os.environ.pop("WEIGHTLESS_ENABLE_MILESTONE_2", None)
    os.environ.pop("WEIGHTLESS_STEER_LAYERS", None)
    try:
        legacy = runtime.resolve_weightless_vllm_xargs({"weightless_alpha": 0.25})
        check(
            legacy.alpha_override == 0.25 and not legacy.has_advanced_controls,
            "legacy alpha remains available without the Milestone 2 gate",
        )
        try:
            runtime.resolve_weightless_vllm_xargs({"weightless_layers": [10]})
        except ValueError as exc:
            gated = "WEIGHTLESS_ENABLE_MILESTONE_2=1" in str(exc)
        else:
            gated = False
        check(gated, "advanced controls fail closed when the server gate is off")

        os.environ["WEIGHTLESS_ENABLE_MILESTONE_2"] = "1"
        parsed = runtime.resolve_weightless_vllm_xargs(
            structured(
                {
                    "alpha": 0.5,
                    "prefill_alpha": 0.1,
                    "decode_alpha": 0.8,
                    "layers": "10,30-31,58",
                    "schedule": [
                        {"start": 2, "end": 5, "start_alpha": 0.2, "end_alpha": 1.0},
                        {"start": 0, "end": 2, "alpha": 0.0},
                    ],
                }
            )
        )
        check(parsed.layers_override == (10, 30, 31, 58), "layer ranges canonicalize")
        check(
            tuple((segment.start, segment.end) for segment in parsed.schedule)
            == ((0, 2), (2, 5)),
            "piecewise segments canonicalize by generation position",
        )

        control = runtime.effective_weightless_control(
            parsed,
            default_alpha=1.0,
            default_layers=(10, 30, 58),
        )
        values = [
            runtime.weightless_alpha_at(control, token_ordinal=ordinal, prompt_length=4)
            for ordinal in range(8)
        ]
        check(
            np.allclose(values, [0.1, 0.1, 0.1, 0.0, 0.0, 0.2, 0.6, 1.0]),
            "prefill, first-sample, pulse, and inclusive ramp values resolve",
            repr(values),
        )

        malformed = (
            [{"start": 0, "end": 2, "alpha": 0.0}, {"start": 1, "end": 3, "alpha": 1.0}],
            [{"start": 0, "end": 0, "alpha": 1.0}],
            [{"start": 0, "end": 1, "start_alpha": 0.0, "end_alpha": 1.0}],
        )
        for index, schedule in enumerate(malformed):
            try:
                runtime.resolve_weightless_vllm_xargs(
                    structured({"schedule": schedule})
                )
            except ValueError:
                rejected = True
            else:
                rejected = False
            check(rejected, f"invalid schedule form {index + 1} is rejected")

        parsed_a = runtime.resolve_weightless_vllm_xargs(
            structured(
                {
                    "prefill_alpha": 0.1,
                    "decode_alpha": 0.8,
                    "layers": [10, 58],
                    "schedule": [
                        {"start": 0, "end": 2, "alpha": 0.0},
                        {"start": 2, "end": 5, "start_alpha": 0.2, "end_alpha": 1.0},
                    ],
                }
            )
        )
        parsed_b = runtime.resolve_weightless_vllm_xargs(
            structured({"decode_alpha": 1.5, "layers": [30]})
        )
        model = ModelStub()
        runner = runtime.WeightlessControlRunner(
            max_num_tokens=12,
            max_num_reqs=4,
            num_ubatches=2,
            num_layers=64,
        )
        batch = SimpleNamespace(
            req_ids=["a", "b"],
            num_scheduled_tokens=np.array([6, 4], dtype=np.int32),
            num_computed_tokens_np=np.array([0, 1], dtype=np.int32),
            prefill_len_np=np.array([4, 2], dtype=np.int32),
        )
        runner.prepare_v2(model, batch, {"a": parsed_a, "b": parsed_b})
        check(
            torch.allclose(
                model.alpha_rows[0],
                torch.tensor(
                    [0.1, 0.1, 0.1, 0.0, 0.0, 0.2, 1.5, 1.5, 1.5, 1.5, 0.0, 0.0]
                ),
            ),
            "v2 mixed schedules map to token rows and zero padding",
        )
        check(
            torch.equal(model.slot_rows[0], torch.tensor([1] * 6 + [2] * 4 + [0] * 2)),
            "v2 token rows route through bounded request slots",
        )
        check(
            bool(model.layer_bank[0, 1, 10])
            and bool(model.layer_bank[0, 1, 58])
            and not bool(model.layer_bank[0, 1, 30])
            and bool(model.layer_bank[0, 2, 30])
            and not bool(model.layer_bank[0, 2, 10]),
            "distinct requests receive isolated layer masks",
        )

        request_a = SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args=structured({"decode_alpha": 0.25, "layers": [10]})),
            num_computed_tokens=3,
            num_prompt_tokens=4,
        )
        request_b = SimpleNamespace(
            sampling_params=SimpleNamespace(extra_args=structured({"decode_alpha": 2.0, "layers": [58]})),
            num_computed_tokens=9,
            num_prompt_tokens=4,
        )
        slices = [
            SimpleNamespace(token_slice=slice(0, 3)),
            SimpleNamespace(token_slice=slice(3, 8)),
        ]
        runner.prepare_v1(
            model,
            ["a", "b"],
            np.array([3, 2], dtype=np.int32),
            8,
            {"a": request_a, "b": request_b},
            slices,
        )
        check(
            torch.equal(model.alpha_rows[0, :3], torch.tensor([0.25] * 3))
            and torch.equal(model.alpha_rows[1, :5], torch.tensor([2.0, 2.0, 0.0, 0.0, 0.0])),
            "v1 DBO slices preserve decode positions and zero graph padding",
        )
        check(
            torch.equal(model.slot_rows[0, :3], torch.tensor([1, 1, 1]))
            and torch.equal(model.slot_rows[1, :5], torch.tensor([2, 2, 0, 0, 0])),
            "v1 DBO slices preserve mask-slot routing",
        )

        runner.prepare_v2(
            model,
            SimpleNamespace(
                req_ids=["b"],
                num_scheduled_tokens=np.array([1], dtype=np.int32),
                num_computed_tokens_np=np.array([10], dtype=np.int32),
                prefill_len_np=np.array([4], dtype=np.int32),
            ),
            {"b": parsed_b},
        )
        check(
            model.slot_rows[0, 0].item() == 1
            and model.slot_rows[0, 1:].count_nonzero().item() == 0
            and bool(model.layer_bank[0, 1, 30])
            and not bool(model.layer_bank[0, 1, 10]),
            "request removal and slot reuse clear stale control state",
        )

        unavailable = runtime.resolve_weightless_vllm_xargs(
            structured({"layers": [31]})
        )
        try:
            runner.prepare_v2(
                model,
                SimpleNamespace(
                    req_ids=["bad"],
                    num_scheduled_tokens=np.array([1], dtype=np.int32),
                    num_computed_tokens_np=np.array([0], dtype=np.int32),
                    prefill_len_np=np.array([1], dtype=np.int32),
                ),
                {"bad": unavailable},
            )
        except ValueError as exc:
            rejected = "not loaded" in str(exc)
        else:
            rejected = False
        check(rejected, "request masks cannot select unloaded intervention layers")

        previous_static_layers = os.environ.get("WEIGHTLESS_STEER_LAYERS")
        os.environ["WEIGHTLESS_STEER_LAYERS"] = "10,30,58"
        try:
            try:
                runtime.resolve_weightless_vllm_xargs(
                    structured({"layers": [31]})
                )
            except ValueError as exc:
                early_rejection = "not enabled by the server" in str(exc)
            else:
                early_rejection = False
        finally:
            if previous_static_layers is None:
                os.environ.pop("WEIGHTLESS_STEER_LAYERS", None)
            else:
                os.environ["WEIGHTLESS_STEER_LAYERS"] = previous_static_layers
        check(early_rejection, "configured layer bounds reject before engine execution")
    finally:
        if previous_gate is None:
            os.environ.pop("WEIGHTLESS_ENABLE_MILESTONE_2", None)
        else:
            os.environ["WEIGHTLESS_ENABLE_MILESTONE_2"] = previous_gate
        if original_static_layers is None:
            os.environ.pop("WEIGHTLESS_STEER_LAYERS", None)
        else:
            os.environ["WEIGHTLESS_STEER_LAYERS"] = original_static_layers

    print()
    print(
        "weightless schedules v0.27: "
        + ("all checks passed" if not failures else f"{failures} failure(s)")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
