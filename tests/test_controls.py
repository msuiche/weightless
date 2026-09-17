"""Tests for weightless_runtime.controls -- the engine-neutral request policy.

Stdlib only: the policy layer imports no torch and no vLLM, so these run
anywhere. The torch-dependent row mapping is covered by
vllm-plugin/tests/test_control_plane.py.

Focus is the server alpha policy window. Alpha is the one request
dimension that used to be unbounded while layers, schedule segments,
token positions and trace records all had caps, and it is the dimension
the README documents as model-specific and quality-destroying out of range.
"""
import os
import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weightless_runtime.controls import (  # noqa: E402
    resolve_weightless_vllm_xargs,
)

ALPHA_ENV = ("WEIGHTLESS_CONTROL_MIN_ALPHA", "WEIGHTLESS_CONTROL_MAX_ALPHA")


def structured(**controls):
    return {"weightless": {"version": 1, "controls": controls}}


class AlphaPolicyWindow(unittest.TestCase):
    def setUp(self):
        # Advanced controls are gated; the window applies to plain alpha too.
        patch = mock.patch.dict(
            os.environ, {"WEIGHTLESS_ENABLE_MILESTONE_2": "1"}
        )
        patch.start()
        self.addCleanup(patch.stop)
        for name in ALPHA_ENV:
            os.environ.pop(name, None)

    def test_default_window_accepts_the_shipped_range(self):
        # 0.0 is the null arm, 4.0 the highest alpha any lane runs.
        for value in (0.0, 1.0, 2.0, 4.0):
            with self.subTest(value=value):
                resolved = resolve_weightless_vllm_xargs(
                    {"weightless_alpha": value}
                )
                self.assertEqual(resolved.alpha_override, value)

    def test_alpha_above_the_window_is_refused(self):
        with self.assertRaisesRegex(ValueError, "alpha policy"):
            resolve_weightless_vllm_xargs({"weightless_alpha": 50.0})

    def test_negative_alpha_is_refused_by_default(self):
        # A negative alpha amplifies the refusal direction instead of
        # projecting it out -- the opposite of what the lane is calibrated
        # for, and not something a client should reach by default.
        with self.assertRaisesRegex(ValueError, "alpha policy"):
            resolve_weightless_vllm_xargs({"weightless_alpha": -1.0})

    def test_every_alpha_field_is_bounded(self):
        cases = {
            "alpha": structured(alpha=99.0),
            "prefill_alpha": structured(prefill_alpha=99.0),
            "decode_alpha": structured(decode_alpha=99.0),
            "segment alpha": structured(
                schedule=[{"start": 0, "end": 2, "alpha": 99.0}]
            ),
            "segment start_alpha": structured(
                schedule=[{"start": 0, "end": 3, "start_alpha": 99.0,
                           "end_alpha": 1.0}]
            ),
            "segment end_alpha": structured(
                schedule=[{"start": 0, "end": 3, "start_alpha": 1.0,
                           "end_alpha": 99.0}]
            ),
        }
        for label, payload in cases.items():
            with self.subTest(field=label):
                with self.assertRaisesRegex(ValueError, "alpha policy"):
                    resolve_weightless_vllm_xargs(payload)

    def test_legacy_alpha_is_bounded_too(self):
        with self.assertRaisesRegex(ValueError, "alpha policy"):
            resolve_weightless_vllm_xargs({"weightless_alpha": 9.0})

    def test_server_can_widen_the_window(self):
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_CONTROL_MAX_ALPHA": "10"}):
            resolved = resolve_weightless_vllm_xargs(
                {"weightless_alpha": 9.0}
            )
        self.assertEqual(resolved.alpha_override, 9.0)

    def test_server_can_lower_the_floor(self):
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_CONTROL_MIN_ALPHA": "-2"}):
            resolved = resolve_weightless_vllm_xargs(
                {"weightless_alpha": -1.5}
            )
        self.assertEqual(resolved.alpha_override, -1.5)

    def test_server_can_pin_the_window_shut(self):
        """A lane that wants its calibrated alpha and nothing else."""
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_CONTROL_MIN_ALPHA": "1",
                              "WEIGHTLESS_CONTROL_MAX_ALPHA": "1"}):
            self.assertEqual(
                resolve_weightless_vllm_xargs(
                    {"weightless_alpha": 1.0}).alpha_override,
                1.0,
            )
            with self.assertRaisesRegex(ValueError, "alpha policy"):
                resolve_weightless_vllm_xargs({"weightless_alpha": 1.5})

    def test_inverted_window_is_refused(self):
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_CONTROL_MIN_ALPHA": "3",
                              "WEIGHTLESS_CONTROL_MAX_ALPHA": "1"}):
            with self.assertRaisesRegex(ValueError, "must be >="):
                resolve_weightless_vllm_xargs({"weightless_alpha": 2.0})

    def test_non_numeric_policy_env_is_refused(self):
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_CONTROL_MAX_ALPHA": "loads"}):
            with self.assertRaisesRegex(ValueError, "finite float"):
                resolve_weightless_vllm_xargs({"weightless_alpha": 1.0})

    def test_non_finite_alpha_still_refused_ahead_of_the_window(self):
        for value in (float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    resolve_weightless_vllm_xargs({"weightless_alpha": value})

    def test_no_alpha_means_no_policy_check(self):
        """A request that does not steer is not subject to the window."""
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_CONTROL_MAX_ALPHA": "0"}):
            self.assertIsNone(
                resolve_weightless_vllm_xargs({}).alpha_override
            )


class UnrelatedValidationStillHolds(unittest.TestCase):
    """The window must not have loosened anything else."""

    def setUp(self):
        patch = mock.patch.dict(
            os.environ, {"WEIGHTLESS_ENABLE_MILESTONE_2": "1"}
        )
        patch.start()
        self.addCleanup(patch.stop)

    def test_overlapping_segments_still_refused(self):
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            resolve_weightless_vllm_xargs(structured(schedule=[
                {"start": 0, "end": 4, "alpha": 1.0},
                {"start": 2, "end": 6, "alpha": 1.0},
            ]))

    def test_unknown_control_field_still_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown"):
            resolve_weightless_vllm_xargs(structured(alpha=1.0, nope=1))

    def test_bool_is_not_a_float(self):
        with self.assertRaisesRegex(ValueError, "finite int/float"):
            resolve_weightless_vllm_xargs({"weightless_alpha": True})

    def test_milestone_gate_still_required(self):
        with mock.patch.dict(os.environ,
                             {"WEIGHTLESS_ENABLE_MILESTONE_2": ""}):
            with self.assertRaisesRegex(ValueError, "Milestone 2"):
                resolve_weightless_vllm_xargs(structured(decode_alpha=1.0))


if __name__ == "__main__":
    unittest.main()
