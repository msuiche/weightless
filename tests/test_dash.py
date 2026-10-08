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


class FormatTests(unittest.TestCase):
    def test_human_scales(self):
        self.assertEqual(dash.human(81_384), "81.4k")
        self.assertEqual(dash.human(16_964_612), "17.0M")
        self.assertEqual(dash.human(108_584), "108.6k")
        self.assertEqual(dash.human(604), "604")

    def test_fmt_rate_keeps_one_decimal_under_1k(self):
        self.assertEqual(dash.fmt_rate(12.1), "12.1")
        self.assertEqual(dash.fmt_rate(3_023), "3.0k")

    def test_clip_counts_visible_chars_not_ansi(self):
        line = "\033[38;5;170mabcdef\033[0m"
        self.assertEqual(dash.clip(line, 4), "\033[38;5;170mabcd\033[0m")
        self.assertIs(dash.clip(line, None), line)  # falsy width: no-op
        self.assertEqual(dash.clip("short", 80), "short")

    def test_fmt_dur(self):
        self.assertEqual(dash.fmt_dur(65), "1m")
        self.assertEqual(dash.fmt_dur(3_720), "1h02m")


class RateWindowTests(unittest.TestCase):
    def test_stable_rate_over_window(self):
        w = dash.RateWindow(span=30.0)
        # Jittery per-interval deltas (200, 20, 200) still average out.
        for t, v in ((0.0, 0.0), (2.0, 200.0), (4.0, 220.0), (6.0, 420.0)):
            w.add(t, v)
        self.assertAlmostEqual(w.rate(), 70.0)  # 420 tok over 6s

    def test_single_sample_is_zero(self):
        w = dash.RateWindow()
        w.add(1.0, 50.0)
        self.assertEqual(w.rate(), 0.0)

    def test_counter_reset_clears_history(self):
        w = dash.RateWindow()
        for t, v in ((0.0, 1_000.0), (2.0, 2_000.0)):
            w.add(t, v)
        w.add(4.0, 10.0)  # server restarted: counter went backwards
        w.add(6.0, 110.0)
        self.assertAlmostEqual(w.rate(), 50.0)  # only post-reset samples count

    def test_old_samples_expire(self):
        w = dash.RateWindow(span=10.0)
        for t, v in ((0.0, 0.0), (100.0, 100.0), (102.0, 300.0)):
            w.add(t, v)
        self.assertAlmostEqual(w.rate(), 100.0)  # t=0 sample is outside the span


class RenderContextTests(unittest.TestCase):
    def test_info_line_renders_model_ctx_version(self):
        out = dash.render("http://lane", {"vllm:num_requests_running": 1.0}, None, 1.0,
                          deque(), deque(), 0.0, dash.palette(False),
                          info={"model": "nvidia/GLM-5.3-Flash-NVFP4", "ctx": 1_048_576,
                                "version": "0.29.0"})
        self.assertIn("nvidia/GLM-5.3-Flash-NVFP4", out)
        self.assertIn("ctx 1.0M", out)
        self.assertIn("vllm 0.29.0", out)

    def test_per_request_decode_shown_when_running(self):
        prev = {"vllm:generation_tokens_total": 0.0, "vllm:num_requests_running": 2.0}
        out = dash.render("http://lane",
                          {"vllm:generation_tokens_total": 100.0, "vllm:num_requests_running": 2.0},
                          prev, 1.0, deque(), deque(), 0.0, dash.palette(False))
        self.assertIn("decode 50.0 tok/s/req", out)

    def test_width_clipping_prevents_wraps(self):
        out = dash.render("http://a-very-long-lane-hostname.example:8888",
                          {"vllm:num_requests_running": 1.0}, None, 1.0,
                          deque(), deque(), 0.0, dash.palette(True),
                          width=60)
        for line in out.splitlines():
            self.assertLessEqual(len(dash.ANSI.sub("", line)), 60)
        self.assertTrue(out.endswith("\033[0m") or "\033[0m" in out)


if __name__ == "__main__":
    unittest.main()
