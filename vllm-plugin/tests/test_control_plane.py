"""Per-request control plane + the SteeringCore rows it drives.

CPU only: torch and numpy, no vLLM, no GPU. The apply checks compare the
steered output against a hand-computed projection so a wrong row mapping
shows up as a numeric difference, not just a shape that happens to fit.
"""
import os
import pathlib
import sys
import unittest

import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))          # vllm-plugin/
sys.path.insert(0, str(_HERE.parents[2]))          # repo root, weightless_runtime/

from weightless_runtime.controls import (  # noqa: E402
    WeightlessResolvedXArgs,
    WeightlessScheduleSegment,
)
from weightless_steer.control_plane import (  # noqa: E402
    ScheduledRequest,
    WeightlessControlPlane,
)
from weightless_steer.core import SteeringCore  # noqa: E402

HIDDEN = 4
NUM_LAYERS = 6
MAX_TOKENS = 32
MAX_REQS = 4


def unit(*values):
    v = torch.tensor(values, dtype=torch.float32)
    return v / v.norm()


DIRS = {
    1: unit(1.0, 0.0, 0.0, 0.0),
    3: unit(0.0, 1.0, 0.0, 0.0),
    4: unit(0.0, 0.0, 1.0, 0.0),
}


class Owner(torch.nn.Module):
    """Stands in for the steered inner model: just carries the buffers."""


def build_core(*, per_request=True, alpha=2.0, dirs=None):
    core = SteeringCore(
        DIRS if dirs is None else dirs, alpha, "residual_stream_post_layer",
        NUM_LAYERS, HIDDEN,
        max_num_tokens=MAX_TOKENS if per_request else None,
        max_num_reqs=MAX_REQS if per_request else None,
    )
    owner = Owner()
    core.register_buffers(owner, torch.float32)
    return core, owner


def project(h, layer, alpha):
    """Hand-computed h - alpha*(h.d)d for one layer."""
    if layer not in DIRS:
        return h.clone()
    d = DIRS[layer]
    return h - alpha * (h @ d).unsqueeze(-1) * d


def plane(**kw):
    kw.setdefault("max_num_tokens", MAX_TOKENS)
    kw.setdefault("max_num_reqs", MAX_REQS)
    kw.setdefault("num_layers", NUM_LAYERS)
    kw.setdefault("default_alpha", 2.0)
    kw.setdefault("loaded_layers", tuple(sorted(DIRS)))
    return WeightlessControlPlane(**kw)


def req(req_id="r0", resolved=None, token_count=1, start_ordinal=0,
        prompt_length=1):
    return ScheduledRequest(
        req_id=req_id,
        resolved=resolved or WeightlessResolvedXArgs(),
        token_count=token_count,
        start_ordinal=start_ordinal,
        prompt_length=prompt_length,
    )


class DefaultsReproduceScalarLane(unittest.TestCase):
    """The property that keeps this change safe to deploy."""

    def test_unwritten_buffers_steer_at_the_server_alpha(self):
        core, owner = build_core(alpha=2.0)
        h = torch.randn(5, HIDDEN)
        # Nothing ever called set_control_rows.
        for layer in range(NUM_LAYERS):
            torch.testing.assert_close(
                core.apply(layer, h.clone()), project(h, layer, 2.0)
            )

    def test_per_request_off_matches_per_request_on_with_defaults(self):
        scalar, _ = build_core(per_request=False, alpha=1.5)
        rows, owner = build_core(per_request=True, alpha=1.5)
        p = plane(default_alpha=1.5)
        p.build([req(token_count=4, prompt_length=4)])
        p.install(_ModelShim(rows))
        h = torch.randn(4, HIDDEN)
        for layer in range(NUM_LAYERS):
            torch.testing.assert_close(
                scalar.apply(layer, h.clone()), rows.apply(layer, h.clone())
            )

    def test_padded_tail_rows_keep_the_server_alpha(self):
        core, owner = build_core(alpha=2.0)
        p = plane()
        p.build([req(token_count=2, prompt_length=2)])
        p.install(_ModelShim(core))
        # A captured graph may run more rows than the step scheduled.
        h = torch.randn(8, HIDDEN)
        torch.testing.assert_close(core.apply(1, h.clone()),
                                   project(h, 1, 2.0))


class _ModelShim:
    """Gives a bare core the mixin's control-plane method name."""

    def __init__(self, core):
        self._core = core

    def set_weightless_control_rows(self, a, s, b):
        self._core.set_control_rows(a, s, b)


class PerRequestAlpha(unittest.TestCase):
    def test_two_requests_get_their_own_alpha(self):
        core, owner = build_core(alpha=2.0)
        p = plane()
        written = p.build([
            req("a", WeightlessResolvedXArgs(alpha_override=0.0),
                token_count=2, prompt_length=2),
            req("b", WeightlessResolvedXArgs(alpha_override=3.0),
                token_count=2, prompt_length=2),
        ])
        self.assertEqual(written, 4)
        p.install(_ModelShim(core))

        h = torch.randn(4, HIDDEN)
        got = core.apply(1, h.clone())
        # rows 0-1 unsteered (alpha 0), rows 2-3 at alpha 3.
        torch.testing.assert_close(got[:2], h[:2])
        torch.testing.assert_close(got[2:], project(h, 1, 3.0)[2:])

    def test_prefill_and_decode_alpha_split(self):
        core, owner = build_core(alpha=2.0)
        p = plane()
        # prompt_length 3 => ordinals 0,1 are prefill; 2 onward decode.
        p.build([req("a", WeightlessResolvedXArgs(
            prefill_alpha_override=1.0, decode_alpha_override=4.0),
            token_count=4, start_ordinal=0, prompt_length=3)])
        p.install(_ModelShim(core))
        self.assertEqual(
            [round(v, 3) for v in p.alpha_rows[:4].tolist()],
            [1.0, 1.0, 4.0, 4.0],
        )

    def test_constant_schedule_segment(self):
        p = plane()
        p.build([req("a", WeightlessResolvedXArgs(
            decode_alpha_override=1.0,
            schedule=(WeightlessScheduleSegment(0, 2, 3.0, 3.0),)),
            token_count=4, start_ordinal=0, prompt_length=1)])
        # prompt_length 1 => decision ordinal 0, generation position == ordinal
        self.assertEqual([round(v, 3) for v in p.alpha_rows[:4].tolist()],
                         [3.0, 3.0, 1.0, 1.0])

    def test_ramped_schedule_segment_interpolates(self):
        p = plane()
        p.build([req("a", WeightlessResolvedXArgs(
            decode_alpha_override=0.0,
            schedule=(WeightlessScheduleSegment(0, 3, 0.0, 2.0),)),
            token_count=4, start_ordinal=0, prompt_length=1)])
        self.assertEqual([round(v, 3) for v in p.alpha_rows[:4].tolist()],
                         [0.0, 1.0, 2.0, 0.0])

    def test_schedule_resumes_mid_sequence(self):
        """A decode step starting at ordinal 2 must land on the right rung."""
        p = plane()
        p.build([req("a", WeightlessResolvedXArgs(
            decode_alpha_override=0.0,
            schedule=(WeightlessScheduleSegment(0, 3, 0.0, 2.0),)),
            token_count=1, start_ordinal=2, prompt_length=1)])
        self.assertEqual(round(p.alpha_rows[0].item(), 3), 2.0)


class PerRequestLayerMask(unittest.TestCase):
    def test_mask_restricts_which_layers_steer(self):
        core, owner = build_core(alpha=2.0)
        p = plane()
        p.build([req("a", WeightlessResolvedXArgs(layers_override=(3,)),
                     token_count=2, prompt_length=2)])
        p.install(_ModelShim(core))
        h = torch.randn(2, HIDDEN)
        # layer 3 is masked in: steered.
        torch.testing.assert_close(core.apply(3, h.clone()),
                                   project(h, 3, 2.0))
        # layer 1 carries a direction but is masked out: untouched.
        torch.testing.assert_close(core.apply(1, h.clone()), h)

    def test_mask_is_per_request_not_per_batch(self):
        core, owner = build_core(alpha=2.0)
        p = plane()
        p.build([
            req("a", WeightlessResolvedXArgs(layers_override=(1,)),
                token_count=1, prompt_length=1),
            req("b", WeightlessResolvedXArgs(layers_override=(3,)),
                token_count=1, prompt_length=1),
        ])
        p.install(_ModelShim(core))
        h = torch.randn(2, HIDDEN)
        got = core.apply(1, h.clone())
        torch.testing.assert_close(got[:1], project(h, 1, 2.0)[:1])  # a on
        torch.testing.assert_close(got[1:], h[1:])                   # b off

    def test_unloaded_layer_is_refused(self):
        p = plane()
        with self.assertRaisesRegex(ValueError, "not loaded"):
            p.build([req("a", WeightlessResolvedXArgs(layers_override=(2,)),
                         token_count=1, prompt_length=1)])

    def test_layer_beyond_model_depth_is_refused(self):
        p = plane(loaded_layers=None)
        with self.assertRaisesRegex(ValueError, "outside this model's depth"):
            p.build([req("a", WeightlessResolvedXArgs(layers_override=(99,)),
                         token_count=1, prompt_length=1)])


class SlotHygiene(unittest.TestCase):
    def test_a_freed_slot_does_not_leak_its_old_plan(self):
        p = plane()
        p.build([req("a", WeightlessResolvedXArgs(layers_override=(1,)),
                     token_count=1, prompt_length=1)])
        self.assertEqual(p.layer_bank[:, 1].tolist(),
                         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        # Next step: the request is gone, slot 1 belongs to a plain request.
        p.build([req("b", token_count=1, prompt_length=1)])
        self.assertEqual(p.layer_bank[:, 1].tolist(), [1.0] * NUM_LAYERS)
        self.assertEqual(round(p.alpha_rows[0].item(), 3), 2.0)

    def test_slot_zero_stays_fully_gated_on(self):
        p = plane()
        p.build([req("a", WeightlessResolvedXArgs(layers_override=(1,)),
                     token_count=1, prompt_length=1)])
        self.assertEqual(p.layer_bank[:, 0].tolist(), [1.0] * NUM_LAYERS)


class Capacity(unittest.TestCase):
    def test_too_many_requests(self):
        p = plane()
        with self.assertRaisesRegex(ValueError, "capacity is 4"):
            p.build([req(f"r{i}", token_count=1, prompt_length=1)
                     for i in range(MAX_REQS + 1)])

    def test_too_many_tokens(self):
        p = plane()
        with self.assertRaisesRegex(ValueError, "capacity is 32"):
            p.build([req("a", token_count=MAX_TOKENS + 1, prompt_length=1)])


class ScalarCoreRefusesControlPlans(unittest.TestCase):
    def test_set_control_rows_on_a_scalar_core_raises(self):
        core, owner = build_core(per_request=False)
        with self.assertRaisesRegex(RuntimeError, "without per-request"):
            core.set_control_rows(torch.zeros(4))

    def test_oversized_plan_is_refused(self):
        core, owner = build_core()
        with self.assertRaisesRegex(ValueError, "exceeds buffer"):
            core.set_control_rows(torch.zeros(MAX_TOKENS + 1))


class StaleStateCannotSurviveAStep(unittest.TestCase):
    """A shorter step must not inherit the previous step's plan."""

    def test_short_alpha_write_resets_the_tail(self):
        core, owner = build_core(alpha=2.0)
        # Step 1: eight tokens, all unsteered.
        core.set_control_rows(torch.zeros(8))
        self.assertEqual(owner._steer_alpha_rows[:8].tolist(), [0.0] * 8)
        # Step 2: only two tokens scheduled.
        core.set_control_rows(torch.full((2,), 3.0))
        self.assertEqual(owner._steer_alpha_rows[:2].tolist(), [3.0, 3.0])
        # Rows 2-7 must be back at the server alpha, not still 0.0 from
        # step 1 -- those rows would otherwise serve UNSTEERED.
        self.assertEqual(owner._steer_alpha_rows[2:8].tolist(), [2.0] * 6)

    def test_short_gate_write_resets_the_tail(self):
        core, owner = build_core()
        bank = torch.zeros(NUM_LAYERS, 3)
        core.set_control_rows(layer_bank=bank)
        core.set_control_rows(layer_bank=torch.ones(NUM_LAYERS, 1))
        self.assertEqual(owner._steer_layer_bank[:, 2].tolist(),
                         [1.0] * NUM_LAYERS)

    def test_short_slot_write_resets_the_tail(self):
        core, owner = build_core()
        core.set_control_rows(slot_rows=torch.full((8,), 3,
                                                   dtype=torch.long))
        core.set_control_rows(slot_rows=torch.full((2,), 1,
                                                   dtype=torch.long))
        self.assertEqual(owner._steer_slot_rows[2:8].tolist(), [0] * 6)


class RowMappingNeedsAFlattenedBatch(unittest.TestCase):
    def test_three_dim_stream_is_refused(self):
        """[batch, seq, hidden] would index rows by BATCH, not by token."""
        core, _ = build_core()
        with self.assertRaisesRegex(RuntimeError, "flattened"):
            core.apply(1, torch.randn(2, 3, HIDDEN))

    def test_scalar_lane_still_accepts_extra_leading_dims(self):
        """The unchanged path keeps its `...` einsum generality."""
        core, _ = build_core(per_request=False, alpha=2.0)
        h = torch.randn(2, 3, HIDDEN)
        torch.testing.assert_close(core.apply(1, h.clone()),
                                   project(h, 1, 2.0))


class GeometryIsBothOrNeither(unittest.TestCase):
    def test_half_specified_geometry_is_refused(self):
        for kw in ({"max_num_tokens": 8}, {"max_num_reqs": 2}):
            with self.subTest(**kw):
                with self.assertRaisesRegex(ValueError, "both-or-neither"):
                    SteeringCore(DIRS, 1.0, "residual_stream_post_layer",
                                 NUM_LAYERS, HIDDEN, **kw)


class GateAgreesWithTheRequestParser(unittest.TestCase):
    """Server-side registration and request-side acceptance, one test."""

    def test_same_truthiness_for_every_spelling(self):
        import os
        from unittest import mock

        from weightless_runtime.controls import _enabled
        from weightless_steer.archs.base import (
            _PER_REQUEST_ENV,
            _per_request_enabled,
        )

        for raw in ("1", "true", "TRUE", "yes", "on", "0", "false", "", "no",
                    " on ", "maybe"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {_PER_REQUEST_ENV: raw}):
                    self.assertEqual(_per_request_enabled(),
                                     _enabled(_PER_REQUEST_ENV))


class VectorisedFillMatchesTheScalarReference(unittest.TestCase):
    """The two alpha implementations must not drift apart.

    weightless_alpha_at() is the readable per-token reference;
    _fill_weightless_alpha_slice() is the vectorised fill build() actually
    uses on the hot path. A disagreement would give a request the wrong
    alpha schedule in production, and no example-based test would
    necessarily catch it -- so compare them over randomised schedules,
    fixed seed for reproducibility.
    """

    def test_agree_over_random_schedules(self):
        import random

        from weightless_runtime.controls import (
            WeightlessEffectiveControl,
            _fill_weightless_alpha_slice,
            weightless_alpha_at,
        )

        rng = random.Random(20260917)
        worst = 0.0
        compared = 0
        for _ in range(500):
            prompt_length = rng.randint(1, 12)
            segments, cursor = [], 0
            for _ in range(rng.randint(0, 3)):
                start = cursor + rng.randint(0, 3)
                end = start + rng.randint(1, 5)
                if rng.random() < 0.5:
                    alpha = round(rng.uniform(0, 4), 3)
                    segment = WeightlessScheduleSegment(start, end, alpha,
                                                        alpha)
                else:
                    if end - start == 1:      # a ramp needs two tokens
                        end += 1
                    segment = WeightlessScheduleSegment(
                        start, end, round(rng.uniform(0, 4), 3),
                        round(rng.uniform(0, 4), 3))
                segments.append(segment)
                cursor = end
            control = WeightlessEffectiveControl(
                base_alpha=round(rng.uniform(0, 4), 3),
                prefill_alpha=round(rng.uniform(0, 4), 3),
                decode_alpha=round(rng.uniform(0, 4), 3),
                layers=None,
                schedule=tuple(segments),
            )
            start_ordinal = rng.randint(0, 20)
            count = rng.randint(1, 16)

            filled = torch.zeros(count, dtype=torch.float32)
            _fill_weightless_alpha_slice(
                filled, control=control, start_ordinal=start_ordinal,
                prompt_length=prompt_length,
            )
            for offset in range(count):
                reference = weightless_alpha_at(
                    control,
                    token_ordinal=start_ordinal + offset,
                    prompt_length=prompt_length,
                )
                worst = max(worst, abs(reference - filled[offset].item()))
                compared += 1

        self.assertGreater(compared, 2000, "the sweep did not cover much")
        self.assertLess(worst, 1e-5,
                        f"vectorised fill drifted from the reference by "
                        f"{worst:g} over {compared} token positions")


if __name__ == "__main__":
    unittest.main()
