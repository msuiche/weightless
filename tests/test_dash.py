"""Contracts for scripts/dash.py: the numbers it derives from a vLLM
/metrics payload and its robustness against malformed ones.

One bad sample must never blank the whole dashboard, kill it with a
traceback, or let a lookalike label (not_finished_reason, other_position)
masquerade as the real one. Rates must be summed across the `engine` label,
headline prefill must be computed tokens (not cache hits), and anything the
lane does not expose renders as "—", never as a confident 0.
"""
import contextlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dash = load("dash", ROOT / "scripts" / "dash.py")


def lane_for(metrics, prev=None, dt=1.0):
    """A Lane that saw `prev` at t=0 and `metrics` at t=dt (or only `metrics`)."""
    lane = dash.Lane()
    if prev is not None:
        lane.observe(prev, 0.0)
        lane.observe(metrics, dt)
    else:
        lane.observe(metrics, 0.0)
    return lane


def render_for(metrics, prev=None, color=False, **kw):
    lane = lane_for(metrics, prev)
    return dash.render("http://lane", lane.snapshot(), dash.palette(color), **kw)


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


class LookupTests(unittest.TestCase):
    def test_g_sums_across_engines(self):
        m = {'vllm:prompt_tokens_total|model_name="m",engine="0"': 100.0,
             'vllm:prompt_tokens_total|model_name="m",engine="1"': 50.0,
             'vllm:prompt_tokens_total_other|engine="0"': 999.0}
        self.assertEqual(dash.g(m, "vllm:prompt_tokens_total"), 150.0)  # suffix lookalike excluded

    def test_gmean_for_ratios(self):
        m = {'vllm:kv_cache_usage_perc|engine="0"': 0.2, 'vllm:kv_cache_usage_perc|engine="1"': 0.6}
        self.assertAlmostEqual(dash.gmean(m, "vllm:kv_cache_usage_perc"), 0.4)

    def test_pick_prefers_present_name(self):
        m = {"vllm:gpu_cache_usage_perc": 0.5}
        self.assertEqual(dash.pick(m, *dash.KV_NAMES), "vllm:gpu_cache_usage_perc")
        self.assertEqual(dash.pick({}, *dash.KV_NAMES), "vllm:kv_cache_usage_perc")

    def test_buckets_summed_and_parsed(self):
        m = {'vllm:ttft_bucket|le="0.5",engine="0"': 1.0, 'vllm:ttft_bucket|le="0.5",engine="1"': 2.0,
             'vllm:ttft_bucket|le="+Inf",engine="0"': 4.0, 'vllm:ttft_bucket|le="bad"': 9.0}
        b = dash.buckets(m, "vllm:ttft")
        self.assertEqual(b, {0.5: 3.0, math.inf: 4.0})


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

    def test_set_span_shrinks_history(self):
        w = dash.RateWindow(span=100.0)
        for t, v in ((0.0, 0.0), (50.0, 50.0), (60.0, 160.0)):
            w.add(t, v)
        w.set_span(10.0)
        self.assertAlmostEqual(w.rate(), 11.0)  # only the last 10s remain


class HistWindowTests(unittest.TestCase):
    def test_quantiles_interpolate_within_bucket(self):
        h = dash.HistWindow(span=60.0)
        h.add(0.0, {0.1: 0.0, 0.5: 0.0, 1.0: 0.0, math.inf: 0.0})
        # 10 requests: 2 under 0.1s, 6 under 0.5s, all under 1.0s.
        h.add(2.0, {0.1: 2.0, 0.5: 6.0, 1.0: 10.0, math.inf: 10.0})
        self.assertAlmostEqual(h.quantile(0.5), 0.4)   # 3 of 4 into [0.1, 0.5]
        self.assertAlmostEqual(h.quantile(0.9), 0.875)  # 3 of 4 into [0.5, 1.0]

    def test_empty_window_is_unknown(self):
        h = dash.HistWindow()
        self.assertIsNone(h.quantile(0.5))
        h.add(0.0, {math.inf: 5.0})
        h.add(1.0, {math.inf: 5.0})  # no new requests
        self.assertIsNone(h.quantile(0.5))

    def test_overflow_bucket_returns_last_edge(self):
        h = dash.HistWindow()
        h.add(0.0, {1.0: 0.0, math.inf: 0.0})
        h.add(1.0, {1.0: 0.0, math.inf: 4.0})  # everything slower than 1s
        self.assertEqual(h.quantile(0.5), 1.0)

    def test_counter_reset_clears(self):
        h = dash.HistWindow()
        h.add(0.0, {math.inf: 100.0})
        h.add(1.0, {math.inf: 2.0})  # restart
        self.assertIsNone(h.quantile(0.5))


class LaneTests(unittest.TestCase):
    def test_headline_prefill_is_computed_tokens(self):
        prev = {"vllm:prompt_tokens_total": 0.0, "vllm:prefix_cache_queries_total": 0.0,
                "vllm:prefix_cache_hits_total": 0.0}
        m = {"vllm:prompt_tokens_total": 1_000.0, "vllm:prefix_cache_queries_total": 1_000.0,
             "vllm:prefix_cache_hits_total": 990.0}
        s = lane_for(m, prev).snapshot()
        self.assertAlmostEqual(s["prefill"], 10.0)            # 1000 - 990 tok/s of real compute
        self.assertAlmostEqual(s["prefill_admitted"], 1_000.0)
        self.assertAlmostEqual(s["cached_pct"], 99.0)

    def test_prefill_falls_back_to_prompt_tokens_without_cache_counters(self):
        s = lane_for({"vllm:prompt_tokens_total": 300.0}, {"vllm:prompt_tokens_total": 0.0}).snapshot()
        self.assertAlmostEqual(s["prefill"], 300.0)
        self.assertIsNone(s["prefill_admitted"])
        self.assertIsNone(s["cached_pct"])

    def test_instant_and_windowed_rates_differ(self):
        lane = dash.Lane(span=30.0)
        lane.observe({"vllm:generation_tokens_total": 0.0}, 0.0)
        lane.observe({"vllm:generation_tokens_total": 200.0}, 2.0)
        lane.observe({"vllm:generation_tokens_total": 220.0}, 4.0)
        s = lane.snapshot()
        self.assertAlmostEqual(s["decode"], 55.0)      # 220 over 4s
        self.assertAlmostEqual(s["decode_now"], 10.0)  # 20 over the last 2s
        self.assertAlmostEqual(s["decode_peak"], 100.0)
        self.assertEqual(list(lane.hist_dec), [100.0, 10.0])

    def test_restart_resets_session_baseline(self):
        lane = dash.Lane()
        lane.observe({"vllm:prompt_tokens_total": 1_000.0, "vllm:generation_tokens_total": 500.0}, 0.0)
        lane.observe({"vllm:prompt_tokens_total": 10.0, "vllm:generation_tokens_total": 5.0}, 2.0)
        lane.observe({"vllm:prompt_tokens_total": 110.0, "vllm:generation_tokens_total": 55.0}, 4.0)
        s = lane.snapshot()
        self.assertEqual(s["session_prompt"], 100.0)
        self.assertEqual(s["session_gen"], 50.0)
        self.assertAlmostEqual(s["avg_decode"], 25.0)

    def test_unknown_metrics_are_none_not_zero(self):
        s = lane_for({"vllm:num_requests_running": 1.0}).snapshot()
        for key in ("kv_pct", "ttft_p50", "ttft_life", "queue_s", "steps_per_s", "preempt_total",
                    "stream_per_req", "busy", "accept_pct", "cached_pct", "req_prompt_tok",
                    "ms_per_step"):
            self.assertIsNone(s[key], key)
        self.assertFalse(s["warm"])

    def test_engine_steps_and_preemptions(self):
        prev = {"vllm:iteration_tokens_total_sum": 0.0, "vllm:iteration_tokens_total_count": 0.0,
                "vllm:num_preemptions_total": 3.0}
        m = {"vllm:iteration_tokens_total_sum": 400.0, "vllm:iteration_tokens_total_count": 20.0,
             "vllm:num_preemptions_total": 5.0}
        s = lane_for(m, prev).snapshot()
        self.assertAlmostEqual(s["steps_per_s"], 20.0)
        self.assertAlmostEqual(s["tok_per_step"], 20.0)
        self.assertEqual(s["preempt_total"], 5)
        self.assertAlmostEqual(s["preempt_rate"], 2.0)

    def test_spec_decode_acceptance_length_and_per_pos(self):
        prev = {"vllm:spec_decode_num_draft_tokens_total": 0.0,
                "vllm:spec_decode_num_accepted_tokens_total": 0.0,
                "vllm:spec_decode_num_drafts_total": 0.0}
        m = {"vllm:spec_decode_num_draft_tokens_total": 100.0,
             "vllm:spec_decode_num_accepted_tokens_total": 60.0,
             "vllm:spec_decode_num_drafts_total": 40.0,
             'vllm:spec_decode_num_accepted_tokens_per_pos_total|engine="0",position="0"': 20.0,
             'vllm:spec_decode_num_accepted_tokens_per_pos_total|engine="1",position="0"': 10.0,
             'vllm:spec_decode_num_accepted_tokens_per_pos_total|engine="0",position="1"': 20.0}
        s = lane_for(m, prev).snapshot()
        self.assertAlmostEqual(s["accept_pct"], 60.0)
        self.assertAlmostEqual(s["accept_len"], 2.5)             # 1 + 60/40, vLLM's convention
        self.assertEqual(s["per_pos_pct"], [75.0, 50.0])          # summed over engines, normalised by drafts

    def test_ms_per_step_from_decode_seconds_and_drafts(self):
        prev = {"vllm:request_decode_time_seconds_sum": 0.0,
                "vllm:spec_decode_num_drafts_total": 0.0}
        m = {"vllm:request_decode_time_seconds_sum": 2.0,
             "vllm:spec_decode_num_drafts_total": 200.0}
        s = lane_for(m, prev).snapshot()
        self.assertAlmostEqual(s["ms_per_step"], 10.0)  # 2s decode ÷ 200 steps

    def test_ms_per_step_none_without_drafts(self):
        prev = {"vllm:request_decode_time_seconds_sum": 0.0}
        m = {"vllm:request_decode_time_seconds_sum": 2.0}
        self.assertIsNone(lane_for(m, prev).snapshot()["ms_per_step"])

    def test_kv_capacity_from_cache_config(self):
        m = {"vllm:kv_cache_usage_perc": 0.5,
             'vllm:cache_config_info|block_size="16",num_gpu_blocks="1000",engine=""': 1.0}
        s = lane_for(m).snapshot()
        self.assertEqual(s["kv_capacity_tok"], 16_000)
        self.assertAlmostEqual(s["kv_pct"], 50.0)
        bad = {'vllm:cache_config_info|block_size="16",num_gpu_blocks="None"': 1.0}
        self.assertIsNone(lane_for(bad).snapshot()["kv_capacity_tok"])

    def test_itl_name_fallback(self):
        prev = {"vllm:generation_tokens_total": 0.0, "vllm:time_per_output_token_seconds_sum": 0.0}
        m = {"vllm:generation_tokens_total": 100.0, "vllm:time_per_output_token_seconds_sum": 2.0}
        self.assertAlmostEqual(lane_for(m, prev).snapshot()["stream_per_req"], 50.0)

    def test_busy_fraction(self):
        lane = dash.Lane()
        for t, running in ((0.0, 1.0), (2.0, 0.0), (4.0, 0.0), (6.0, 2.0)):
            lane.observe({"vllm:num_requests_running": running}, t)
        self.assertAlmostEqual(lane.snapshot()["busy"], 2 / 6)


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

    def test_well_formed_payload_renders(self):
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
        self.assertIn("lifetime 2.0s", out)
        self.assertIn("lifetime 75%", out)  # prefix-cache hit rate since boot
        self.assertIn("rates need a second sample", out)

    def test_missing_metrics_render_as_dash(self):
        out = render_for({"vllm:num_requests_running": 1.0})
        self.assertIn("kv cache   — used", out)
        self.assertIn("p50 —  p90 —", out)
        self.assertNotIn("lifecycle", out)
        self.assertNotIn("spec dec", out)
        self.assertNotIn("engine ", out)

    def test_queued_hint_depends_on_kv_pressure(self):
        full = render_for({"vllm:num_requests_waiting": 2.0, "vllm:kv_cache_usage_perc": 0.95})
        self.assertIn("QUEUED (KV full)", full)
        seqs = render_for({"vllm:num_requests_waiting": 2.0, "vllm:kv_cache_usage_perc": 0.3})
        self.assertIn("QUEUED (raise MAX_NUM_SEQS)", seqs)

    def test_info_line_renders_model_ctx_version(self):
        out = render_for({"vllm:num_requests_running": 1.0},
                         info={"model": "nvidia/GLM-5.3-Flash-NVFP4", "ctx": 1_048_576,
                               "version": "0.29.0"})
        self.assertIn("nvidia/GLM-5.3-Flash-NVFP4", out)
        self.assertIn("ctx 1M", out)
        self.assertIn("vllm 0.29.0", out)

    def test_error_and_paused_banners(self):
        out = render_for({"vllm:num_requests_running": 1.0}, error="cannot read x: boom", paused=True)
        self.assertIn("error  cannot read x: boom", out)
        self.assertIn("PAUSED", out)

    def test_per_request_decode_shown_when_running(self):
        prev = {"vllm:generation_tokens_total": 0.0, "vllm:num_requests_running": 2.0}
        out = render_for({"vllm:generation_tokens_total": 100.0, "vllm:num_requests_running": 2.0}, prev)
        self.assertIn("decode 50.0 tok/s/req", out)

    def test_width_clipping_prevents_wraps(self):
        out = render_for({"vllm:num_requests_running": 1.0}, color=True, width=60)
        for line in out.splitlines():
            self.assertLessEqual(len(dash.ANSI.sub("", line)), 60)
        self.assertIn("\033[0m", out)

    def test_sparkline_fills_the_terminal_width(self):
        lane = dash.Lane()
        for t in range(80):
            lane.observe({"vllm:generation_tokens_total": float(t * 10)}, float(t))
        s = lane.snapshot()
        s["_hist_dec"] = lane.hist_dec
        out = dash.render("http://lane", s, dash.palette(False), width=120)
        decode = next(line for line in out.splitlines() if line.startswith("  decode"))
        self.assertGreater(decode.count("█"), 30)
        self.assertLessEqual(len(decode), 120)

    def test_token_totals_with_session_deltas(self):
        prev = {"vllm:prompt_tokens_total": 1_000_000.0, "vllm:generation_tokens_total": 2_000.0}
        m = {"vllm:prompt_tokens_total": 1_500_000.0, "vllm:generation_tokens_total": 5_000.0}
        out = render_for(m, prev)
        self.assertIn("prompt 1.5M", out)
        self.assertIn("gen 5.0k", out)
        self.assertIn("+500.0k prompt", out)
        self.assertIn("+3.0k gen", out)

    def test_prefill_shows_computed_and_admitted(self):
        prev = {"vllm:prompt_tokens_total": 0.0, "vllm:prefix_cache_queries_total": 0.0,
                "vllm:prefix_cache_hits_total": 0.0}
        m = {"vllm:prompt_tokens_total": 1_000.0, "vllm:prefix_cache_queries_total": 1_000.0,
             "vllm:prefix_cache_hits_total": 990.0}
        out = render_for(m, prev)
        self.assertIn("prefill    10.0 tok/s", out)   # real compute is the headline
        self.assertIn("admitted 1.0k · cached 99%", out)

    def test_prefill_split_hidden_without_cache_metrics(self):
        self.assertNotIn("admitted", render_for({"vllm:num_requests_running": 1.0}))

    def test_decode_shows_stream_rate_from_itl(self):
        prev = {"vllm:generation_tokens_total": 0.0, "vllm:inter_token_latency_seconds_sum": 0.0}
        m = {"vllm:generation_tokens_total": 100.0, "vllm:inter_token_latency_seconds_sum": 2.0}
        self.assertIn("stream 50.0/req", render_for(m, prev))  # 100 tok in 2s decode time

    def test_stream_hidden_without_itl_metrics(self):
        self.assertNotIn("stream", render_for({"vllm:num_requests_running": 1.0}))

    def test_lifecycle_line_shows_phase_means(self):
        names = ["queue", "prefill", "decode"]
        prev, m = {}, {}
        for name in names:
            prev[f"vllm:request_{name}_time_seconds_sum"] = 0.0
            prev[f"vllm:request_{name}_time_seconds_count"] = 0.0
        m["vllm:request_queue_time_seconds_sum"] = 20.0    # 2.0s per request
        m["vllm:request_prefill_time_seconds_sum"] = 10.0  # 1.0s
        m["vllm:request_decode_time_seconds_sum"] = 80.0   # 8.0s
        for name in names:
            m[f"vllm:request_{name}_time_seconds_count"] = 10.0
        self.assertIn("queue 2.0s → prefill 1.0s → decode 8.0s", render_for(m, prev))

    def test_ttft_percentiles_from_buckets(self):
        prev = {'vllm:time_to_first_token_seconds_bucket|le="0.1"': 0.0,
                'vllm:time_to_first_token_seconds_bucket|le="0.5"': 0.0,
                'vllm:time_to_first_token_seconds_bucket|le="+Inf"': 0.0}
        m = {'vllm:time_to_first_token_seconds_bucket|le="0.1"': 5.0,
             'vllm:time_to_first_token_seconds_bucket|le="0.5"': 9.0,
             'vllm:time_to_first_token_seconds_bucket|le="+Inf"': 10.0}
        out = render_for(m, prev)
        self.assertIn("p50 100ms", out)
        self.assertIn("p90 500ms", out)

    def test_engine_line(self):
        prev = {"vllm:iteration_tokens_total_sum": 0.0, "vllm:iteration_tokens_total_count": 0.0,
                "vllm:num_preemptions_total": 0.0, "vllm:num_requests_running": 1.0}
        m = {"vllm:iteration_tokens_total_sum": 400.0, "vllm:iteration_tokens_total_count": 20.0,
             "vllm:num_preemptions_total": 1.0, "vllm:num_requests_running": 1.0}
        out = render_for(m, prev)
        self.assertIn("steps 20.0/s   tok/step 20.0   busy 100%   preempt 1 (+1.00/s)", out)


class SnapshotTests(unittest.TestCase):
    def run_once(self, *args, payloads):
        calls = iter(payloads)
        with patch.object(dash, "fetch", side_effect=lambda *a, **k: next(calls)), \
             patch.object(dash, "lane_info", return_value={}), \
             patch.object(dash.time, "sleep") as sleep, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            rc = dash.main(["--once", "--no-color", *args])
        return rc, output.getvalue(), sleep

    def test_once_takes_two_samples_so_rates_are_real(self):
        rc, out, sleep = self.run_once("--interval", "0.5", payloads=[
            "vllm:generation_tokens_total 0\nvllm:num_requests_running 2\n",
            "vllm:generation_tokens_total 100\nvllm:num_requests_running 2\n"])
        self.assertEqual(rc, 0)
        sleep.assert_called_once_with(0.5)
        self.assertIn("running 2", out)
        self.assertNotIn("rates need a second sample", out)
        self.assertRegex(out, r"decode\s+\d")  # a real, non-zero decode rate
        self.assertNotIn("\x1b", out)

    def test_json_snapshot(self):
        rc, out, _ = self.run_once("--json", payloads=[
            "vllm:generation_tokens_total 0\nvllm:kv_cache_usage_perc 0.25\n",
            "vllm:generation_tokens_total 100\nvllm:kv_cache_usage_perc 0.25\n"])
        self.assertEqual(rc, 0)
        data = json.loads(out)
        self.assertEqual(data["target"], "http://spark-4687.local:8888")
        self.assertAlmostEqual(data["kv_pct"], 25.0)
        self.assertGreater(data["decode"], 0)
        self.assertIsNone(data["ttft_p50"])
        self.assertFalse(any(k.startswith("_") for k in data))

    def test_once_snapshot_tolerates_malformed_lines(self):
        payload = (
            "vllm:num_requests_running 2\n"
            "vllm:broken 1e\n"
            'vllm:request_success_total{finished_reason=""} 1\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="x"} 9\n'
        )
        rc, out, _ = self.run_once(payloads=[payload, payload])
        self.assertEqual(rc, 0)
        self.assertIn("running 2", out)

    def test_once_snapshot_tolerates_nonfinite_values(self):
        payload = (
            "vllm:num_requests_running 1e999\n"
            "vllm:num_requests_waiting 0\n"
            "vllm:generation_tokens_total -1e999\n"
        )
        rc, out, _ = self.run_once(payloads=[payload, payload])
        self.assertEqual(rc, 0)
        self.assertIn("running 0", out)  # inf sample dropped, no int() crash

    def test_once_fails_fast_on_first_bad_sample(self):
        with patch.object(dash, "fetch", side_effect=OSError("down")), \
             patch.object(dash.time, "sleep") as sleep, \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(dash.main(["--once"]), 1)
        sleep.assert_not_called()
        self.assertIn("down", errors.getvalue())


class FormatTests(unittest.TestCase):
    def test_human_scales(self):
        self.assertEqual(dash.human(81_384), "81.4k")
        self.assertEqual(dash.human(16_964_612), "17.0M")
        self.assertEqual(dash.human(108_584), "108.6k")
        self.assertEqual(dash.human(604), "604")
        self.assertEqual(dash.human(999_999), "1.0M")  # never "1000.0k"

    def test_fmt_rate_keeps_one_decimal_under_1k(self):
        self.assertEqual(dash.fmt_rate(12.1), "12.1")
        self.assertEqual(dash.fmt_rate(3_023), "3.0k")
        self.assertEqual(dash.fmt_rate(None), "—")

    def test_fmt_secs(self):
        self.assertEqual(dash.fmt_secs(0.0421), "42ms")
        self.assertEqual(dash.fmt_secs(2.04), "2.0s")
        self.assertEqual(dash.fmt_secs(None), "—")

    def test_fmt_ctx(self):
        self.assertEqual(dash.fmt_ctx(131_072), "128k")
        self.assertEqual(dash.fmt_ctx(1_048_576), "1M")
        self.assertEqual(dash.fmt_ctx(200_000), "200.0k")

    def test_clip_counts_visible_chars_not_ansi(self):
        line = "\033[38;5;170mabcdef\033[0m"
        self.assertEqual(dash.clip(line, 4), "\033[38;5;170mabcd\033[0m")
        self.assertIs(dash.clip(line, None), line)  # falsy width: no-op
        self.assertEqual(dash.clip("short", 80), "short")

    def test_fmt_dur(self):
        self.assertEqual(dash.fmt_dur(65), "1m")
        self.assertEqual(dash.fmt_dur(3_720), "1h02m")


class SpecStepRenderTests(unittest.TestCase):
    def test_step_ms_shown_in_spec_row(self):
        prev = {"vllm:request_decode_time_seconds_sum": 0.0,
                "vllm:spec_decode_num_drafts_total": 0.0,
                "vllm:spec_decode_num_draft_tokens_total": 0.0,
                "vllm:spec_decode_num_accepted_tokens_total": 0.0}
        m = {"vllm:request_decode_time_seconds_sum": 2.0,
             "vllm:spec_decode_num_drafts_total": 200.0,
             "vllm:spec_decode_num_draft_tokens_total": 400.0,
             "vllm:spec_decode_num_accepted_tokens_total": 200.0}
        out = render_for(m, prev)
        self.assertIn("step 10.0ms", out)   # 2s decode ÷ 200 steps


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {"WEIGHTLESS_DASH_BASELINES":
                                      os.path.join(self.tmp.name, "baselines.json")})
        env.start()
        self.addCleanup(env.stop)
        self.info = {"model": "fake/model", "ctx": 8192, "version": "0.0.0-fake"}

    def test_save_and_load_roundtrip(self):
        dash.save_baseline(self.info, {"source": "t", "decode_tps": 50.0, "prefill_tps": None})
        base = dash.load_baseline(self.info)
        self.assertEqual(base["decode_tps"], 50.0)
        self.assertNotIn("prefill_tps", base)           # Nones are dropped
        self.assertEqual(base["key"], "fake/model|8192|0.0.0-fake")

    def test_key_requires_a_model_id(self):
        self.assertIsNone(dash.load_baseline({}))
        with self.assertRaises(RuntimeError):
            dash.save_baseline({}, {"decode_tps": 1.0})

    def test_nothing_to_bless_is_an_error(self):
        with self.assertRaises(RuntimeError):
            dash.save_baseline(self.info, {"decode_tps": None})

    def test_corrupt_store_reads_as_empty(self):
        with open(os.environ["WEIGHTLESS_DASH_BASELINES"], "w") as f:
            f.write("{not json")
        self.assertEqual(dash.load_baselines(), {})
        self.assertIsNone(dash.load_baseline(self.info))

    def test_render_shows_ratio_against_baseline(self):
        prev = {"vllm:generation_tokens_total": 0.0}
        m = {"vllm:generation_tokens_total": 150.0}
        out = render_for(m, prev, baseline={"decode_tps": 200.0})
        self.assertIn("base 0.75×", out)

    def test_render_without_baseline_shows_no_ratio(self):
        prev = {"vllm:generation_tokens_total": 0.0}
        out = render_for({"vllm:generation_tokens_total": 150.0}, prev)
        self.assertNotIn("×", out)

    def test_once_save_baseline(self):
        payloads = iter(["vllm:generation_tokens_total 0\n",
                         "vllm:generation_tokens_total 100\n"])
        with patch.object(dash, "fetch", side_effect=lambda *a, **k: next(payloads)), \
             patch.object(dash, "lane_info", return_value=dict(self.info)), \
             patch.object(dash.time, "sleep"), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            rc = dash.main(["--once", "--save-baseline", "--interval", "0.5"])
        self.assertEqual(rc, 0)
        self.assertIn("baseline saved", errors.getvalue())
        self.assertGreater(dash.load_baseline(self.info)["decode_tps"], 0)


class FakeLane:
    """A minimal vLLM lookalike for bench tests: /metrics, /v1/models,
    /version and /v1/chat/completions whose request counters move
    synchronously with each completion, so trials isolate exactly."""

    COUNTERS = ("prompt", "gen", "finished", "ttft", "queue", "prefill",
                "decode", "e2e", "draft_tok", "acc_tok", "drafts")

    def __init__(self, running=0.0, contaminate=False):
        self.running = running
        self.contaminate = contaminate   # pretend another client shares the lane
        self.lock = threading.Lock()
        self.c = {name: 0.0 for name in self.COUNTERS}

    def metrics_text(self):
        with self.lock:
            c = dict(self.c)
        hist = "".join(f"vllm:request_{name}_time_seconds_sum {c[key]}\n"
                       f"vllm:request_{name}_time_seconds_count {c['finished']}\n"
                       for name, key in (("queue", "queue"), ("prefill", "prefill"),
                                         ("decode", "decode")))
        return (
            f"vllm:num_requests_running {self.running}\n"
            "vllm:num_requests_waiting 0\n"
            f"vllm:prompt_tokens_total {c['prompt']}\n"
            f"vllm:generation_tokens_total {c['gen']}\n"
            f"vllm:request_generation_tokens_count {c['finished']}\n"
            f"vllm:request_generation_tokens_sum {c['gen']}\n"
            f'vllm:request_success_total{{finished_reason="stop"}} {c["finished"]}\n'
            f"vllm:time_to_first_token_seconds_sum {c['ttft']}\n"
            f"vllm:time_to_first_token_seconds_count {c['finished']}\n"
            f"{hist}"
            f"vllm:e2e_request_latency_seconds_sum {c['e2e']}\n"
            f"vllm:e2e_request_latency_seconds_count {c['finished']}\n"
            f"vllm:spec_decode_num_draft_tokens_total {c['draft_tok']}\n"
            f"vllm:spec_decode_num_accepted_tokens_total {c['acc_tok']}\n"
            f"vllm:spec_decode_num_drafts_total {c['drafts']}\n"
        )

    def complete(self, body):
        max_tokens = int(body.get("max_tokens") or 16)
        prompt = max(1, sum(len(m.get("content", ""))
                            for m in body.get("messages", [])) // 4)
        with self.lock:
            self.c["prompt"] += prompt
            self.c["gen"] += max_tokens
            self.c["finished"] += 2 if self.contaminate else 1
            self.c["ttft"] += 0.05
            self.c["queue"] += 0.01
            self.c["prefill"] += prompt / 100_000.0
            self.c["decode"] += max_tokens / 100.0
            self.c["e2e"] += 0.06 + prompt / 100_000.0 + max_tokens / 100.0
            self.c["draft_tok"] += max_tokens * 2
            self.c["acc_tok"] += max_tokens
            self.c["drafts"] += max_tokens / 2
        return {"prompt_tokens": prompt, "completion_tokens": max_tokens,
                "total_tokens": prompt + max_tokens}

    def __enter__(self):
        lane = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body, ctype="application/json", code=200):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/metrics":
                    self._send(lane.metrics_text(), "text/plain")
                elif self.path == "/v1/models":
                    self._send(json.dumps({"data": [{"id": "fake/model",
                                                     "max_model_len": 8192}]}))
                elif self.path == "/version":
                    self._send(json.dumps({"version": "0.0.0-fake"}))
                else:
                    self._send("{}", code=404)

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self._send("{}", code=404)
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                usage = lane.complete(body)
                self._send(json.dumps({
                    "choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant", "content": "ok"}}],
                    "usage": usage}))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class BenchTests(unittest.TestCase):
    """--bench against a fake lane: the run must isolate every trial in the
    request counters, and any violation — busy lane, other traffic, an
    existing report — must fail loudly, never measure anyway."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {"WEIGHTLESS_DASH_BASELINES":
                                      os.path.join(self.tmp.name, "baselines.json")})
        env.start()
        self.addCleanup(env.stop)
        self.info = {"model": "fake/model", "ctx": 8192, "version": "0.0.0-fake"}

    def run_bench(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
             contextlib.redirect_stderr(io.StringIO()) as err:
            rc = dash.main(list(args))
        return rc, out.getvalue(), err.getvalue()

    def test_complete_run_isolates_every_trial(self):
        out_path = os.path.join(self.tmp.name, "run1.json")
        with FakeLane() as lane:
            rc, _, _ = self.run_bench(lane.url, "--bench", "--wave", "3", "--out", out_path)
        self.assertEqual(rc, 0)
        with open(out_path) as f:
            report = json.load(f)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["warmup"]["label"], "excluded_warmup")
        self.assertEqual([r["label"] for r in report["cases"]],
                         ["short", "medium", "long", "prefill"])
        for row in report["cases"]:
            self.assertAlmostEqual(row["decode_tps"], 100.0)
            self.assertAlmostEqual(row["accept_pct"], 50.0)
            self.assertAlmostEqual(row["ms_per_step"], 20.0)
        self.assertEqual(report["wave"]["n"], 3)
        self.assertGreater(report["wave"]["aggregate_tps"], 0)
        self.assertEqual(report["summary"]["decode_tps"], 100.0)
        self.assertEqual(report["script_sha256"], dash.script_sha256())
        self.assertNotIn("vs_baseline", report)

    def test_save_baseline_then_ratio_in_later_report(self):
        out1 = os.path.join(self.tmp.name, "run1.json")
        out2 = os.path.join(self.tmp.name, "run2.json")
        with FakeLane() as lane:
            rc1, _, _ = self.run_bench(lane.url, "--bench", "--wave", "2",
                                       "--out", out1, "--save-baseline")
            rc2, _, _ = self.run_bench(lane.url, "--bench", "--wave", "2", "--out", out2)
        self.assertEqual((rc1, rc2), (0, 0))
        base = dash.load_baseline(self.info)
        self.assertEqual(base["source"], "bench")
        self.assertEqual(base["decode_tps"], 100.0)
        with open(out2) as f:
            report = json.load(f)
        self.assertEqual(report["vs_baseline"]["decode_tps"], 1.0)

    def test_refuses_to_overwrite_a_report(self):
        out_path = os.path.join(self.tmp.name, "run1.json")
        with open(out_path, "w") as f:
            f.write("{}")
        with FakeLane() as lane:
            rc, _, err = self.run_bench(lane.url, "--bench", "--out", out_path)
        self.assertEqual(rc, 2)
        self.assertIn("overwrite", err)

    def test_busy_lane_fails_loudly(self):
        out_path = os.path.join(self.tmp.name, "run.json")
        with FakeLane(running=5.0) as lane:
            rc, _, err = self.run_bench(lane.url, "--bench", "--idle-timeout", "0.2",
                                        "--out", out_path)
        self.assertEqual(rc, 2)
        self.assertIn("lane busy", err)
        with open(out_path) as f:
            self.assertEqual(json.load(f)["status"], "failed")

    def test_other_traffic_breaks_isolation_and_fails(self):
        out_path = os.path.join(self.tmp.name, "run.json")
        with FakeLane(contaminate=True) as lane:
            rc, _, err = self.run_bench(lane.url, "--bench", "--out", out_path)
        self.assertEqual(rc, 2)
        self.assertIn("isolate the trial", err)
        with open(out_path) as f:
            report = json.load(f)
        self.assertEqual(report["status"], "failed")
        self.assertIn("isolate the trial", report["error"])

    def test_out_without_bench_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                dash.main(["--out", "x.json"])


if __name__ == "__main__":
    unittest.main()
