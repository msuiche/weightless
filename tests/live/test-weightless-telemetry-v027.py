#!/usr/bin/env python3
"""Live Milestone 1 telemetry checks against a Weightless vLLM server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import pathlib
import statistics
import time
import urllib.error
import urllib.request


def post_json(url: str, body: dict, timeout: float) -> tuple[int, dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            return response.status, payload, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read())
        return exc.code, payload, time.perf_counter() - started


def response_text(payload: dict) -> tuple[str, str]:
    message = payload["choices"][0]["message"]
    return message.get("content") or "", message.get("reasoning_content") or ""


def trace_xargs(alpha: float, layers: list[int], max_records: int = 4096) -> dict:
    return {
        "weightless": {
            "version": 1,
            "controls": {"alpha": alpha},
            "trace": {
                "metrics": [
                    "coefficient",
                    "post_coefficient",
                    "normalized_coefficient",
                    "residual_norm",
                    "directional_energy",
                    "effective_alpha",
                    "delta_norm",
                ],
                "layers": layers,
                "token_stride": 1,
                "max_records": max_records,
            },
        }
    }


def request_body(
    model: str,
    prompt: str,
    *,
    alpha: float,
    max_tokens: int,
    layers: list[int] | None,
    max_records: int = 4096,
) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if layers is None:
        body["vllm_xargs"] = {"weightless_alpha": alpha}
    else:
        body["vllm_xargs"] = trace_xargs(alpha, layers, max_records)
    return body


def wait_for_summaries(
    trace_dir: pathlib.Path,
    previous: set[pathlib.Path],
    count: int,
    timeout: float,
) -> list[pathlib.Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = set(trace_dir.glob("wt-*.summary.json"))
        created = sorted(current - previous)
        if len(created) >= count:
            return created
        time.sleep(0.1)
    return sorted(set(trace_dir.glob("wt-*.summary.json")) - previous)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-weightless")
    parser.add_argument("--trace-dir", required=True, type=pathlib.Path)
    parser.add_argument("--layers", default="10,30,58")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--performance-runs", type=int, default=3)
    parser.add_argument("--performance-max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    layers = [int(value) for value in args.layers.split(",") if value]
    if not layers:
        parser.error("--layers must contain at least one layer")
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    api_base = args.base_url.rstrip("/") + "/v1"
    url = api_base + "/chat/completions"
    failures: list[str] = []
    results: dict = {"layers": layers, "checks": {}}

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        results["checks"][label] = {"passed": ok, "detail": detail}
        if not ok:
            failures.append(label)

    health_status, _, _ = post_json(
        url,
        request_body(
            args.model,
            "Reply with exactly: READY",
            alpha=0.0,
            max_tokens=8,
            layers=None,
        ),
        args.timeout,
    )
    check(health_status == 200, "server accepts baseline request", str(health_status))

    before = set(args.trace_dir.glob("wt-*.summary.json"))
    prompt = "Briefly explain why deterministic tests are useful."
    baseline_status, baseline_payload, _ = post_json(
        url,
        request_body(
            args.model,
            prompt,
            alpha=0.5,
            max_tokens=args.max_tokens,
            layers=None,
        ),
        args.timeout,
    )
    traced_status, traced_payload, _ = post_json(
        url,
        request_body(
            args.model,
            prompt,
            alpha=0.5,
            max_tokens=args.max_tokens,
            layers=layers,
        ),
        args.timeout,
    )
    parity = (
        baseline_status == traced_status == 200
        and response_text(baseline_payload) == response_text(traced_payload)
    )
    parity_detail = f"baseline={baseline_status}, traced={traced_status}"
    if traced_status != 200:
        parity_detail += f", error={traced_payload.get('error')}"
    check(parity, "trace/no-trace output parity", parity_detail)

    parity_summaries = (
        wait_for_summaries(args.trace_dir, before, 1, args.timeout)
        if traced_status == 200
        else []
    )
    endpoint_before = set(args.trace_dir.glob("wt-*.summary.json"))
    completion_status, completion_payload, _ = post_json(
        api_base + "/completions",
        {
            "model": args.model,
            "prompt": "Return the word ready.",
            "temperature": 0,
            "seed": 0,
            "max_tokens": 8,
            "vllm_xargs": trace_xargs(0.5, layers, 128),
        },
        args.timeout,
    )
    responses_status, responses_payload, _ = post_json(
        api_base + "/responses",
        {
            "model": args.model,
            "input": "Return the word ready.",
            "temperature": 0,
            "max_output_tokens": 8,
            "vllm_xargs": trace_xargs(0.5, layers, 128),
        },
        args.timeout,
    )
    endpoint_count = sum(
        status == 200 for status in (completion_status, responses_status)
    )
    endpoint_paths = (
        wait_for_summaries(
            args.trace_dir,
            endpoint_before,
            endpoint_count,
            args.timeout,
        )
        if endpoint_count
        else []
    )
    check(
        completion_status == responses_status == 200
        and len(endpoint_paths) == 2,
        "completion and responses endpoints produce traces",
        f"completion={completion_status}, responses={responses_status}, "
        f"sidecars={len(endpoint_paths)}",
    )
    concurrent_before = set(args.trace_dir.glob("wt-*.summary.json"))
    bodies = [
        request_body(
            args.model,
            f"Return the integer {index} and no other text.",
            alpha=(0.0, 0.5, 1.0, 2.5)[index % 4],
            max_tokens=8,
            layers=layers,
        )
        for index in range(args.concurrency)
    ]
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        concurrent_results = list(
            executor.map(lambda body: post_json(url, body, args.timeout), bodies)
        )
    check(
        all(status == 200 for status, _, _ in concurrent_results),
        "mixed concurrent trace requests complete",
    )

    completed_traces = sum(
        status == 200 for status, _, _ in concurrent_results
    )
    concurrent_paths = (
        wait_for_summaries(
            args.trace_dir,
            concurrent_before,
            completed_traces,
            args.timeout,
        )
        if completed_traces
        else []
    )
    summaries = parity_summaries + endpoint_paths + concurrent_paths
    expected_new = args.concurrency + 3
    check(
        len(summaries) == expected_new,
        "one isolated summary is produced per traced request",
        f"expected {expected_new}, found {len(summaries)}",
    )
    check(
        len(concurrent_paths) == args.concurrency,
        "concurrent traces use distinct server trace IDs",
    )
    response_request_ids = {
        payload["id"]
        for status, payload in (
            (traced_status, traced_payload),
            (completion_status, completion_payload),
            (responses_status, responses_payload),
            *((status, payload) for status, payload, _ in concurrent_results),
        )
        if status == 200
    }
    summary_request_ids = {
        json.loads(path.read_text())["request_id"] for path in summaries
    }
    def correlates(summary_id: str, response_id: str) -> bool:
        return (
            summary_id == response_id
            or summary_id.startswith(response_id + "_")
            or summary_id.startswith(response_id + "-")
        )

    check(
        len(summary_request_ids) == len(response_request_ids)
        and all(
            sum(correlates(summary_id, response_id) for summary_id in summary_request_ids)
            == 1
            for response_id in response_request_ids
        )
        and all(
            sum(correlates(summary_id, response_id) for response_id in response_request_ids)
            == 1
            for summary_id in summary_request_ids
        ),
        "trace summaries correlate only to their response request IDs",
    )

    all_records_valid = True
    all_isolated = True
    all_monotonic = True
    all_lineage_empty = True
    all_aggregates_valid = True
    all_formulas_valid = True
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        trace_path = summary_path.with_name(
            summary_path.name.removesuffix(".summary.json") + ".jsonl"
        )
        records = [json.loads(line) for line in trace_path.read_text().splitlines()]
        all_records_valid &= bool(records)
        all_records_valid &= all(
            all(math.isfinite(value) for value in record["metrics"].values())
            for record in records
        )
        for record in records:
            metrics = record["metrics"]
            coefficient = metrics["coefficient"]
            residual_norm = metrics["residual_norm"]
            normalized = coefficient / max(residual_norm, 1e-12)
            alpha = metrics["effective_alpha"]
            all_formulas_valid &= math.isclose(
                metrics["post_coefficient"],
                (1.0 - alpha) * coefficient,
                rel_tol=2e-3,
                abs_tol=2e-4,
            )
            all_formulas_valid &= math.isclose(
                metrics["normalized_coefficient"],
                normalized,
                rel_tol=2e-3,
                abs_tol=2e-4,
            )
            all_formulas_valid &= math.isclose(
                metrics["directional_energy"],
                normalized * normalized,
                rel_tol=3e-3,
                abs_tol=2e-4,
            )
            all_formulas_valid &= math.isclose(
                metrics["delta_norm"],
                abs(alpha * coefficient),
                rel_tol=2e-3,
                abs_tol=2e-4,
            )
        all_isolated &= all(
            record["trace_id"] == summary["trace_id"]
            and record["request_id"] == summary["request_id"]
            for record in records
        )
        ordinals = [record["token_ordinal"] for record in records]
        all_monotonic &= ordinals == sorted(ordinals)
        all_lineage_empty &= all(
            summary[field] is None
            for field in (
                "checkpoint_id",
                "parent_branch_id",
                "branch_id",
                "history_fingerprint",
                "continuation_fingerprint",
            )
        )
        all_aggregates_valid &= all(
            aggregate["count"] > 0
            and all(
                math.isfinite(aggregate[field])
                for field in ("min", "max", "mean")
            )
            for aggregate in summary["metric_aggregates"].values()
        )
    check(all_records_valid, "trace records contain finite scalar metrics")
    check(all_formulas_valid, "GPU trace metrics satisfy project formulas")
    check(all_isolated, "trace records cannot cross request/trace ownership")
    check(all_monotonic, "trace token ordinals are monotonic")
    check(all_lineage_empty, "ordinary traces serialize empty branch lineage")
    check(all_aggregates_valid, "trace summaries contain finite metric aggregates")
    trace_files = [
        path
        for summary_path in summaries
        for path in (
            summary_path,
            summary_path.with_name(
                summary_path.name.removesuffix(".summary.json") + ".jsonl"
            ),
        )
    ]
    check(
        (args.trace_dir.stat().st_mode & 0o777) == 0o700
        and all((path.stat().st_mode & 0o777) == 0o600 for path in trace_files),
        "trace directory and files are owner-only",
    )

    bounded_before = set(args.trace_dir.glob("wt-*.summary.json"))
    bounded_status, _, _ = post_json(
        url,
        request_body(
            args.model,
            "Count from one to five.",
            alpha=1.0,
            max_tokens=16,
            layers=layers,
            max_records=2,
        ),
        args.timeout,
    )
    bounded_paths = (
        wait_for_summaries(args.trace_dir, bounded_before, 1, args.timeout)
        if bounded_status == 200
        else []
    )
    bounded_summary = json.loads(bounded_paths[0].read_text()) if bounded_paths else {}
    check(
        bounded_status == 200
        and bounded_summary.get("record_count") == 2
        and bounded_summary.get("truncated") is True,
        "record budget truncates explicitly",
    )

    invalid = trace_xargs(1.0, layers + [100, 101, 102, 103, 104, 105])
    invalid_status, _, _ = post_json(
        url,
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "vllm_xargs": invalid,
        },
        args.timeout,
    )
    check(invalid_status == 400, "oversized layer allowlist returns HTTP 400")

    performance_prompt = (
        "Activation telemetry measures a low-dimensional projection without "
        "capturing full hidden states. Summarize this statement in two sentences."
    )
    baseline_latencies = []
    trace_latencies = []
    for _ in range(args.performance_runs):
        status, _, latency = post_json(
            url,
            request_body(
                args.model,
                performance_prompt,
                alpha=1.0,
                max_tokens=args.performance_max_tokens,
                layers=None,
            ),
            args.timeout,
        )
        if status == 200:
            baseline_latencies.append(latency)
        status, _, latency = post_json(
            url,
            request_body(
                args.model,
                performance_prompt,
                alpha=1.0,
                max_tokens=args.performance_max_tokens,
                layers=layers,
            ),
            args.timeout,
        )
        if status == 200:
            trace_latencies.append(latency)
    if baseline_latencies and trace_latencies:
        baseline_median = statistics.median(baseline_latencies)
        trace_median = statistics.median(trace_latencies)
        overhead = trace_median / baseline_median - 1.0
        results["performance"] = {
            "baseline_median_seconds": baseline_median,
            "trace_median_seconds": trace_median,
            "relative_overhead": overhead,
        }
        check(
            overhead <= 0.10,
            "three-layer trace latency stays within 10% budget",
            f"overhead={overhead:.2%}",
        )
    else:
        check(False, "three-layer trace latency stays within 10% budget")

    results["passed"] = not failures
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print()
    print(
        "weightless telemetry live: "
        + ("all checks passed" if not failures else f"{len(failures)} failure(s)")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
