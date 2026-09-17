#!/usr/bin/env python3
"""Send one traced request and inspect its Weightless telemetry sidecars."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import stat
import sys
import time
import urllib.error
import urllib.request
from typing import Any


METRICS = (
    "coefficient",
    "post_coefficient",
    "normalized_coefficient",
    "residual_norm",
    "directional_energy",
    "effective_alpha",
    "delta_norm",
)


def parse_layers(raw: str) -> list[int]:
    layers: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError(f"descending layer range: {part}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    if not layers:
        raise ValueError("at least one layer is required")
    if any(layer < 0 for layer in layers):
        raise ValueError("layers must be non-negative")
    if len(layers) != len(set(layers)):
        raise ValueError("layers must not contain duplicates")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise one Weightless telemetry request, validate its scalar "
            "formulas, and print the resulting sidecar summary."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", help="Defaults to the first model from /v1/models")
    parser.add_argument(
        "--trace-dir",
        type=pathlib.Path,
        default=pathlib.Path("/var/lib/weightless/traces"),
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--trace-layers",
        "--layers",
        dest="trace_layers",
        default="10,30,58",
        help=(
            "Layers to observe, as comma-separated IDs or inclusive ranges; "
            "--layers remains a compatibility alias"
        ),
    )
    parser.add_argument("--token-stride", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--show-records", type=int, default=12)
    parser.add_argument(
        "--prompt",
        default="Explain directional activation telemetry in two short sentences.",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="Do not send the deterministic untraced comparison request",
    )
    parser.add_argument("--output", type=pathlib.Path)
    arguments = parser.parse_args()
    try:
        arguments.trace_layers = parse_layers(arguments.trace_layers)
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(arguments.alpha):
        parser.error("--alpha must be finite")
    if arguments.token_stride <= 0:
        parser.error("--token-stride must be positive")
    if arguments.max_records <= 0:
        parser.error("--max-records must be positive")
    if arguments.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if arguments.show_records < 0:
        parser.error("--show-records must be non-negative")
    return arguments


def json_request(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[int, dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
            return response.status, payload, time.perf_counter() - started
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw.decode(errors="replace")}
        return exc.code, payload, time.perf_counter() - started


def discover_model(base_url: str, timeout: float) -> str:
    status, payload, _ = json_request(
        f"{base_url}/models",
        timeout=timeout,
    )
    if status != 200 or not payload.get("data"):
        raise RuntimeError(f"model discovery failed with HTTP {status}: {payload}")
    return str(payload["data"][0]["id"])


def request_body(
    model: str,
    prompt: str,
    *,
    alpha: float,
    max_tokens: int,
    layers: list[int] | None,
    token_stride: int,
    max_records: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
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
        body["vllm_xargs"] = {
            "weightless": {
                "version": 1,
                "controls": {"alpha": alpha},
                "trace": {
                    "layers": layers,
                    "metrics": list(METRICS),
                    "token_stride": token_stride,
                    "max_records": max_records,
                },
            }
        }
    return body


def generated_text(payload: dict[str, Any]) -> str:
    message = payload["choices"][0]["message"]
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    return reasoning + content


def request_ids_correlate(summary_id: str, response_id: str) -> bool:
    return (
        summary_id == response_id
        or summary_id.startswith(response_id + "_")
        or summary_id.startswith(response_id + "-")
    )


def wait_for_summary(
    trace_dir: pathlib.Path,
    previous: set[pathlib.Path],
    response_id: str,
    timeout: float,
) -> tuple[pathlib.Path, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    matched_path: pathlib.Path | None = None
    matched_summary: dict[str, Any] | None = None
    stable_signature: tuple[int, float, int] | None = None
    stable_since: float | None = None
    while time.monotonic() < deadline:
        for path in set(trace_dir.glob("wt-*.summary.json")) - previous:
            try:
                summary = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if request_ids_correlate(str(summary.get("request_id")), response_id):
                matched_path = path
                matched_summary = summary
                signature = (
                    int(summary.get("record_count", 0)),
                    float(summary.get("updated_at", 0.0)),
                    int(summary.get("dropped_records", 0)),
                )
                if signature != stable_signature:
                    stable_signature = signature
                    stable_since = time.monotonic()
                elif stable_since is not None and time.monotonic() - stable_since >= 0.5:
                    return path, summary
                break
        time.sleep(0.1)
    if matched_path is not None and matched_summary is not None:
        return matched_path, matched_summary
    raise TimeoutError(f"no trace summary appeared for response {response_id}")


def trace_path_for(summary_path: pathlib.Path) -> pathlib.Path:
    return summary_path.with_name(
        summary_path.name.removesuffix(".summary.json") + ".jsonl"
    )


def read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def close(actual: float, expected: float, *, rel: float, absolute: float) -> bool:
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=absolute)


def validate_records(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    expected_alpha: float,
) -> list[str]:
    errors: list[str] = []
    previous_ordinal = -1
    for index, record in enumerate(records):
        prefix = f"record {index}"
        if record.get("trace_id") != summary.get("trace_id"):
            errors.append(f"{prefix}: trace ownership mismatch")
        if record.get("request_id") != summary.get("request_id"):
            errors.append(f"{prefix}: request ownership mismatch")
        ordinal = int(record["token_ordinal"])
        if ordinal < previous_ordinal:
            errors.append(f"{prefix}: token ordinals are not monotonic")
        previous_ordinal = ordinal
        metrics = record["metrics"]
        if any(not math.isfinite(float(value)) for value in metrics.values()):
            errors.append(f"{prefix}: non-finite metric")
            continue
        coefficient = float(metrics["coefficient"])
        alpha = float(metrics["effective_alpha"])
        residual_norm = float(metrics["residual_norm"])
        normalized = coefficient / max(residual_norm, 1e-12)
        if not close(alpha, expected_alpha, rel=2e-3, absolute=2e-4):
            errors.append(f"{prefix}: effective alpha {alpha} != {expected_alpha}")
        if not close(
            float(metrics["post_coefficient"]),
            (1.0 - alpha) * coefficient,
            rel=2e-3,
            absolute=2e-4,
        ):
            errors.append(f"{prefix}: post-coefficient formula mismatch")
        if not close(
            float(metrics["normalized_coefficient"]),
            normalized,
            rel=2e-3,
            absolute=2e-4,
        ):
            errors.append(f"{prefix}: normalized-coefficient formula mismatch")
        if not close(
            float(metrics["directional_energy"]),
            normalized * normalized,
            rel=3e-3,
            absolute=2e-4,
        ):
            errors.append(f"{prefix}: directional-energy formula mismatch")
        if not close(
            float(metrics["delta_norm"]),
            abs(alpha * coefficient),
            rel=2e-3,
            absolute=2e-4,
        ):
            errors.append(f"{prefix}: delta-norm formula mismatch")
    return errors


def mode(path: pathlib.Path) -> str:
    return oct(stat.S_IMODE(path.stat().st_mode))


def format_float(value: Any) -> str:
    return f"{float(value): .6g}"


def print_aggregates(summary: dict[str, Any]) -> None:
    print("\nMetric aggregates")
    print(f"  {'metric':<24} {'count':>7} {'minimum':>14} {'mean':>14} {'maximum':>14}")
    for metric in METRICS:
        aggregate = summary["metric_aggregates"].get(metric)
        if not aggregate or aggregate["count"] == 0:
            continue
        print(
            f"  {metric:<24} {aggregate['count']:>7} "
            f"{format_float(aggregate['min']):>14} "
            f"{format_float(aggregate['mean']):>14} "
            f"{format_float(aggregate['max']):>14}"
        )


def print_records(records: list[dict[str, Any]], limit: int) -> None:
    if not records or limit == 0:
        return
    print(f"\nFirst {min(limit, len(records))} records")
    print(
        f"  {'token':>5} {'phase':<6} {'status':<9} {'layer':>5} "
        f"{'coefficient':>13} {'normalized':>13} {'alpha':>9} {'delta':>13}"
    )
    for record in records[:limit]:
        metrics = record["metrics"]
        print(
            f"  {record['token_ordinal']:>5} {record['phase']:<6} "
            f"{record['commit_status']:<9} {record['layer']:>5} "
            f"{format_float(metrics['coefficient']):>13} "
            f"{format_float(metrics['normalized_coefficient']):>13} "
            f"{format_float(metrics['effective_alpha']):>9} "
            f"{format_float(metrics['delta_norm']):>13}"
        )


def main() -> int:
    arguments = parse_args()
    base_url = arguments.base_url.rstrip("/")
    try:
        model = arguments.model or discover_model(base_url, arguments.timeout)
        if not arguments.trace_dir.is_dir():
            raise RuntimeError(f"trace directory does not exist: {arguments.trace_dir}")
        previous = set(arguments.trace_dir.glob("wt-*.summary.json"))
        endpoint = f"{base_url}/chat/completions"
        baseline_payload: dict[str, Any] | None = None
        baseline_latency: float | None = None
        if not arguments.skip_parity:
            baseline_status, baseline_payload, baseline_latency = json_request(
                endpoint,
                method="POST",
                body=request_body(
                    model,
                    arguments.prompt,
                    alpha=arguments.alpha,
                    max_tokens=arguments.max_tokens,
                    layers=None,
                    token_stride=arguments.token_stride,
                    max_records=arguments.max_records,
                ),
                timeout=arguments.timeout,
            )
            if baseline_status != 200:
                raise RuntimeError(
                    f"untraced request failed with HTTP {baseline_status}: {baseline_payload}"
                )
        traced_status, traced_payload, traced_latency = json_request(
            endpoint,
            method="POST",
            body=request_body(
                model,
                arguments.prompt,
                alpha=arguments.alpha,
                max_tokens=arguments.max_tokens,
                layers=arguments.trace_layers,
                token_stride=arguments.token_stride,
                max_records=arguments.max_records,
            ),
            timeout=arguments.timeout,
        )
        if traced_status != 200:
            raise RuntimeError(
                f"traced request failed with HTTP {traced_status}: {traced_payload}"
            )
        response_id = str(traced_payload["id"])
        summary_path, summary = wait_for_summary(
            arguments.trace_dir,
            previous,
            response_id,
            arguments.timeout,
        )
        trace_path = trace_path_for(summary_path)
        records = read_records(trace_path)
    except (OSError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    errors = validate_records(records, summary, arguments.alpha)
    parity = (
        None
        if baseline_payload is None
        else generated_text(baseline_payload) == generated_text(traced_payload)
    )
    if parity is False:
        errors.append("traced output differs from the deterministic untraced request")
    if not records:
        errors.append("trace contains no records")
    if summary.get("record_count") != len(records):
        errors.append(
            f"summary record_count={summary.get('record_count')} but JSONL has {len(records)}"
        )
    if (arguments.trace_dir.stat().st_mode & 0o777) != 0o700:
        errors.append(f"trace directory mode is {mode(arguments.trace_dir)}, expected 0o700")
    for path in (summary_path, trace_path):
        if not path.exists():
            errors.append(f"expected sidecar does not exist: {path}")
        elif (path.stat().st_mode & 0o777) != 0o600:
            errors.append(f"{path.name} mode is {mode(path)}, expected 0o600")

    print("Weightless telemetry demo")
    print(f"  model:              {model}")
    print(f"  response ID:        {response_id}")
    print(f"  alpha:              {arguments.alpha}")
    print(f"  requested trace layers: {arguments.trace_layers}")
    print(f"  observed layers:    {summary['observed_layers']}")
    print(f"  unavailable layers: {summary['unavailable_layers']}")
    print(f"  records/tokens:     {summary['record_count']}/{summary['token_count']}")
    print(f"  truncated/dropped:  {summary['truncated']}/{summary['dropped_records']}")
    print(f"  traced latency:     {traced_latency:.3f}s")
    if baseline_latency is not None:
        print(f"  untraced latency:   {baseline_latency:.3f}s")
        print(f"  output parity:      {'PASS' if parity else 'FAIL'}")
    print(f"  formula checks:     {'PASS' if not errors else 'FAIL'}")
    print(f"  trace directory:    {arguments.trace_dir} ({mode(arguments.trace_dir)})")
    print(f"  summary:            {summary_path} ({mode(summary_path)})")
    trace_mode = mode(trace_path) if trace_path.exists() else "missing"
    print(f"  records:            {trace_path} ({trace_mode})")
    print_aggregates(summary)
    print_records(records, arguments.show_records)
    print("\nGenerated output")
    print(generated_text(traced_payload).strip())

    report = {
        "passed": not errors,
        "model": model,
        "response_id": response_id,
        "alpha": arguments.alpha,
        "trace_layers": arguments.trace_layers,
        "output_parity": parity,
        "baseline_latency_seconds": baseline_latency,
        "traced_latency_seconds": traced_latency,
        "summary_path": str(summary_path),
        "trace_path": str(trace_path),
        "summary": summary,
        "validation_errors": errors,
    }
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nReport written to {arguments.output}")
    if errors:
        print("\nValidation failures", file=sys.stderr)
        for error in errors[:20]:
            print(f"  - {error}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  - ... and {len(errors) - 20} more", file=sys.stderr)
        return 1
    print("\nAll telemetry checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
