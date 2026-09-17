"""Tests for weightless_runtime.telemetry -- directional scalar formulas.

These are checked against a real projection rather than restated, so an
algebra slip shows up: build a vector and a unit direction, apply
h' = h - alpha*(h.d)d by hand, and confirm each reported metric matches
what the projection actually did.

Stdlib only -- the formulas take floats, so no torch is needed here.
"""
import math
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from weightless_runtime.telemetry import directional_scalar_metrics  # noqa: E402


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    return [x / n for x in a]


def project(h, d, alpha):
    """h - alpha*(h.d)d, the apply the serving lanes implement."""
    coefficient = dot(h, d)
    return [x - alpha * coefficient * di for x, di in zip(h, d)]


class FormulasMatchTheProjection(unittest.TestCase):
    H = [3.0, -1.0, 2.0, 0.5]
    D = unit([1.0, 1.0, 0.0, 0.0])

    def metrics(self, alpha):
        return directional_scalar_metrics(
            dot(self.H, self.D), norm(self.H), alpha
        )

    def test_coefficient_is_the_pre_apply_projection(self):
        self.assertAlmostEqual(self.metrics(1.0)["coefficient"],
                               dot(self.H, self.D))

    def test_post_coefficient_is_what_survives_the_apply(self):
        for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
            with self.subTest(alpha=alpha):
                steered = project(self.H, self.D, alpha)
                self.assertAlmostEqual(
                    self.metrics(alpha)["post_coefficient"],
                    dot(steered, self.D),
                )

    def test_alpha_one_removes_the_component_entirely(self):
        self.assertAlmostEqual(self.metrics(1.0)["post_coefficient"], 0.0)

    def test_alpha_zero_is_a_no_op(self):
        m = self.metrics(0.0)
        self.assertAlmostEqual(m["post_coefficient"], m["coefficient"])
        self.assertAlmostEqual(m["delta_norm"], 0.0)

    def test_delta_norm_is_the_distance_the_apply_moved_h(self):
        for alpha in (0.0, 0.5, 2.0, 4.0):
            with self.subTest(alpha=alpha):
                steered = project(self.H, self.D, alpha)
                moved = norm([a - b for a, b in zip(steered, self.H)])
                self.assertAlmostEqual(self.metrics(alpha)["delta_norm"],
                                       moved)

    def test_alpha_two_reflects_the_component(self):
        """a=2 flips the component's sign, keeping its magnitude."""
        m = self.metrics(2.0)
        self.assertAlmostEqual(m["post_coefficient"], -m["coefficient"])

    def test_normalized_coefficient_is_scale_free(self):
        """Doubling h leaves the normalized coefficient unchanged."""
        big = [2 * x for x in self.H]
        a = directional_scalar_metrics(dot(self.H, self.D), norm(self.H), 1.0)
        b = directional_scalar_metrics(dot(big, self.D), norm(big), 1.0)
        self.assertAlmostEqual(a["normalized_coefficient"],
                               b["normalized_coefficient"])

    def test_directional_energy_is_the_squared_normalized_coefficient(self):
        m = self.metrics(1.0)
        self.assertAlmostEqual(m["directional_energy"],
                               m["normalized_coefficient"] ** 2)

    def test_energy_is_the_fraction_of_h_lying_along_d(self):
        """cos^2 of the angle between h and d -- so it sits in [0, 1]."""
        m = self.metrics(1.0)
        cos = dot(self.H, self.D) / norm(self.H)
        self.assertAlmostEqual(m["directional_energy"], cos * cos)
        self.assertGreaterEqual(m["directional_energy"], 0.0)
        self.assertLessEqual(m["directional_energy"], 1.0)

    def test_h_parallel_to_d_is_all_energy(self):
        h = [2 * x for x in self.D]
        m = directional_scalar_metrics(dot(h, self.D), norm(h), 1.0)
        self.assertAlmostEqual(m["directional_energy"], 1.0)

    def test_h_orthogonal_to_d_is_no_energy_and_no_movement(self):
        h = [0.0, 0.0, 5.0, 0.0]
        m = directional_scalar_metrics(dot(h, self.D), norm(h), 4.0)
        self.assertAlmostEqual(m["directional_energy"], 0.0)
        self.assertAlmostEqual(m["delta_norm"], 0.0)

    def test_zero_residual_norm_does_not_divide_by_zero(self):
        m = directional_scalar_metrics(0.0, 0.0, 1.0)
        self.assertTrue(math.isfinite(m["normalized_coefficient"]))
        self.assertTrue(math.isfinite(m["directional_energy"]))

    def test_reported_alpha_is_the_one_applied(self):
        self.assertEqual(self.metrics(2.5)["effective_alpha"], 2.5)


class MetricNamesMatchTheTraceSchema(unittest.TestCase):
    def test_every_advertised_metric_is_produced(self):
        from weightless_runtime.controls import TRACE_METRICS

        produced = directional_scalar_metrics(1.0, 2.0, 1.0)
        self.assertEqual(set(produced), set(TRACE_METRICS))


if __name__ == "__main__":
    unittest.main()
