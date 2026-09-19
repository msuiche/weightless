"""Robustness contracts for scripts/dash.py against malformed metrics payloads.

Regression tests for the crash/misreport paths hit when a /metrics endpoint
(proxied, partial, or non-vLLM) returns lines that are regex-shaped but not
usable: one bad sample must never blank the whole dashboard, kill it with a
traceback, or let a lookalike label (not_finished_reason, other_position)
masquerade as the real one.
"""
import contextlib
import importlib.util
import io
from collections import deque
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dash = load("dash", ROOT / "scripts" / "dash.py")


def render_for(metrics, prev=None):
    return dash.render("http://lane", metrics, prev, 1.0, deque(), deque(), 0.0,
                       dash.palette(False))


class ParseMetricsTests(unittest.TestCase):
    def test_malformed_values_are_skipped_not_fatal(self):
        text = (
            "# HELP vllm:num_requests_running Number running\n"
            "vllm:num_requests_running 3\n"
            "vllm:broken_exponent 1e\n"
            "vllm:broken_dots 1.2.3\n"
            "vllm:broken_sign +\n"
            "not a metric line\n"
            'vllm:request_success_total{finished_reason="stop"} 5\n'
            "vllm:negative -2.5e+1\n"
        )
        m = dash.parse_metrics(text)
        self.assertEqual(m["vllm:num_requests_running"], 3.0)
        self.assertEqual(m['vllm:request_success_total|finished_reason="stop"'], 5.0)
        self.assertEqual(m["vllm:negative"], -25.0)
        for bad in ("vllm:broken_exponent", "vllm:broken_dots", "vllm:broken_sign"):
            self.assertNotIn(bad, m)

    def test_labels_and_missing_labels_both_kept(self):
        m = dash.parse_metrics('a{b="c"} 1\na 2\n')
        self.assertEqual(m["a|b=\"c\""], 1.0)
        self.assertEqual(m["a"], 2.0)

    def test_nonfinite_values_are_skipped(self):
        text = (
            "vllm:num_requests_running 3\n"
            "vllm:overflow 1e999\n"
            "vllm:neg_overflow -1e999\n"
            "vllm:underflow 1e-999\n"
        )
        m = dash.parse_metrics(text)
        self.assertEqual(m["vllm:num_requests_running"], 3.0)
        self.assertEqual(m["vllm:underflow"], 0.0)  # underflows to finite zero, kept
        for bad in ("vllm:overflow", "vllm:neg_overflow"):
            self.assertNotIn(bad, m)


class RenderTests(unittest.TestCase):
    def test_empty_finished_reason_does_not_crash(self):
        out = render_for({
            "vllm:num_requests_running": 1.0,
            'vllm:request_success_total|finished_reason=""': 2.0,
            'vllm:request_success_total|finished_reason="stop"': 3.0,
        })
        self.assertIn("stop 3", out)
        self.assertIn("done 3", out)  # empty reason is skipped, not counted

    def test_non_integer_spec_position_does_not_crash(self):
        out = render_for({
            "vllm:spec_decode_num_draft_tokens_total": 10.0,
            "vllm:spec_decode_num_accepted_tokens_total": 8.0,
            'vllm:spec_decode_num_accepted_tokens_per_pos_total|position="abc"': 5.0,
            'vllm:spec_decode_num_accepted_tokens_per_pos_total|position="1"': 4.0,
        })
        self.assertIn("accept 80%", out)
        self.assertIn("per-pos 100", out)  # only the valid position renders

    def test_lookalike_labels_do_not_masquerade(self):
        out = render_for({
            'vllm:request_success_total|finished_reason="stop"': 3.0,
            'vllm:request_success_total|not_finished_reason="length"': 7.0,
            "vllm:spec_decode_num_draft_tokens_total": 10.0,
            "vllm:spec_decode_num_accepted_tokens_total": 8.0,
            'vllm:spec_decode_num_accepted_tokens_per_pos_total|position="1"': 4.0,
            'vllm:spec_decode_num_accepted_tokens_per_pos_total|other_position="2"': 9.0,
        })
        self.assertIn("done 3", out)  # not_finished_reason is not counted
        self.assertNotIn("length", out)
        self.assertIn("per-pos 100\n", out)  # other_position adds no "225" entry

    def test_label_after_comma_still_matches(self):
        out = render_for({
            'vllm:request_success_total|worker="0",finished_reason="stop"': 4.0,
            "vllm:spec_decode_num_draft_tokens_total": 10.0,
            "vllm:spec_decode_num_accepted_tokens_total": 8.0,
            'vllm:spec_decode_num_accepted_tokens_per_pos_total|lane="a",position="1"': 4.0,
        })
        self.assertIn("stop 4", out)
        self.assertIn("per-pos 100\n", out)

    def test_well_formed_payload_renders_unchanged(self):
        out = render_for({
            "vllm:num_requests_running": 2.0,
            "vllm:num_requests_waiting": 0.0,
            "vllm:kv_cache_usage_perc": 0.5,
            "vllm:prefix_cache_hits_total": 3.0,
            "vllm:prefix_cache_queries_total": 4.0,
            'vllm:request_success_total|finished_reason="length"': 1.0,
            'vllm:request_success_total|finished_reason="stop"': 6.0,
            "vllm:time_to_first_token_seconds_count": 7.0,
            "vllm:time_to_first_token_seconds_sum": 14.0,
        })
        self.assertIn("running 2", out)
        self.assertIn("50% used", out)
        self.assertIn("length 1, stop 6", out)
        self.assertIn("2.0s lifetime", out)


class SnapshotRobustnessTests(unittest.TestCase):
    def test_once_snapshot_tolerates_malformed_lines(self):
        payload = (
            "vllm:num_requests_running 2\n"
            "vllm:broken 1e\n"
            'vllm:request_success_total{finished_reason=""} 1\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="x"} 9\n'
        )
        with patch.object(dash, "fetch", return_value=payload), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(dash.main(["--once", "--no-color"]), 0)
        self.assertIn("running 2", output.getvalue())
        self.assertNotIn("\x1b", output.getvalue())

    def test_once_snapshot_tolerates_nonfinite_values(self):
        payload = (
            "vllm:num_requests_running 1e999\n"
            "vllm:num_requests_waiting 0\n"
            "vllm:generation_tokens_total -1e999\n"
        )
        with patch.object(dash, "fetch", return_value=payload), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(dash.main(["--once", "--no-color"]), 0)
        self.assertIn("running 0", output.getvalue())  # inf sample dropped, no int() crash


if __name__ == "__main__":
    unittest.main()
