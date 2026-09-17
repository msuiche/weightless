#!/usr/bin/env python3
"""Exercise Weightless per-request alpha isolation through a live vLLM API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

MISSING = object()
VALID_ALPHAS = (MISSING, -1.0, 0.0, 0.5, 1.0, 2.5)
INVALID_ALPHAS = (
    ("string", "1.0"),
    ("boolean", True),
    ("null", None),
    ("array", [0.5]),
    ("object", {"value": 0.5}),
    ("nan", math.nan),
    ("infinity", math.inf),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model")
    parser.add_argument("--default-alpha", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--check-prefix-cache", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def discover_model(session: requests.Session, base_url: str, timeout: float) -> str:
    response = session.get(f"{base_url}/models", timeout=timeout)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def make_payload(model: str, max_tokens: int, alpha=MISSING, stream=False) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Write twenty short numbered facts about ocean currents.",
            }
        ],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if alpha is not MISSING:
        payload["vllm_xargs"] = {"weightless_alpha": alpha}
    return payload


def response_summary(response: requests.Response, elapsed: float) -> dict:
    summary = {"status": response.status_code, "elapsed_seconds": round(elapsed, 4)}
    try:
        body = response.json()
    except ValueError:
        summary["body"] = response.text[:500]
        return summary
    if response.ok:
        message = body["choices"][0]["message"]
        generated = (message.get("reasoning_content") or "") + (
            message.get("content") or ""
        )
        summary["output_sha256"] = hashlib.sha256(generated.encode()).hexdigest()
        summary["completion_tokens"] = (body.get("usage") or {}).get(
            "completion_tokens", 0
        )
    else:
        summary["error"] = body.get("error", body)
    return summary


def send_request(
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    alpha=MISSING,
) -> dict:
    payload = make_payload(model, max_tokens, alpha)
    started = time.monotonic()
    if isinstance(alpha, float) and not math.isfinite(alpha):
        response = requests.post(
            f"{base_url}/chat/completions",
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    else:
        response = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            timeout=timeout,
        )
    return response_summary(response, time.monotonic() - started)


def cancel_stream(
    base_url: str, model: str, timeout: float, alpha: float
) -> tuple[int, bool]:
    response = requests.post(
        f"{base_url}/chat/completions",
        json=make_payload(model, 256, alpha, stream=True),
        timeout=timeout,
        stream=True,
    )
    received_data = False
    try:
        for line in response.iter_lines():
            if line.startswith(b"data:"):
                received_data = True
                break
    finally:
        response.close()
    return response.status_code, received_data


def metric_total(metrics_text: str, name: str) -> float | None:
    values = []
    for line in metrics_text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            values.append(float(line.rsplit(" ", 1)[1]))
    return sum(values) if values else None


def prefix_cache_totals(base_url: str, timeout: float) -> tuple[float, float]:
    server_url = base_url.removesuffix("/v1")
    response = requests.get(f"{server_url}/metrics", timeout=timeout)
    response.raise_for_status()
    queries = metric_total(response.text, "vllm:prefix_cache_queries_total")
    hits = metric_total(response.text, "vllm:prefix_cache_hits_total")
    if queries is None or hits is None:
        raise RuntimeError("prefix-cache counters are unavailable")
    return queries, hits


def send_cache_probe(
    base_url: str, model: str, timeout: float, alpha: float
) -> dict:
    payload = make_payload(model, 1, alpha)
    payload["messages"][0]["content"] = (
        "Weightless prefix-cache domain probe. "
        + "ocean current circulation marker " * 600
    )
    started = time.monotonic()
    response = requests.post(
        f"{base_url}/chat/completions", json=payload, timeout=timeout
    )
    return response_summary(response, time.monotonic() - started)


def main() -> int:
    arguments = parse_args()
    base_url = arguments.base_url.rstrip("/")
    session = requests.Session()
    model = arguments.model or discover_model(session, base_url, arguments.timeout)
    report = {"base_url": base_url, "model": model, "checks": []}
    failures = 0

    def check(passed: bool, name: str, detail) -> None:
        nonlocal failures
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        report["checks"].append({"name": name, "passed": passed, "detail": detail})
        failures += 0 if passed else 1

    valid_results = {}
    for alpha in VALID_ALPHAS:
        label = "omitted" if alpha is MISSING else str(alpha)
        result = send_request(
            base_url, model, arguments.max_tokens, arguments.timeout, alpha
        )
        valid_results[label] = result
        check(result["status"] == 200, f"valid alpha {label} is accepted", result)

    omitted = valid_results["omitted"]
    explicit_default = valid_results[str(arguments.default_alpha)]
    check(
        omitted.get("output_sha256") == explicit_default.get("output_sha256"),
        "omitted alpha matches the explicit server default",
        {"omitted": omitted, "explicit_default": explicit_default},
    )

    for label, alpha in INVALID_ALPHAS:
        result = send_request(
            base_url, model, arguments.max_tokens, arguments.timeout, alpha
        )
        check(result["status"] == 400, f"invalid {label} alpha returns HTTP 400", result)

    completion_payload = {
        "model": model,
        "prompt": "Return the word ready.",
        "temperature": 0,
        "max_tokens": 8,
        "vllm_xargs": {"weightless_alpha": 0.5},
    }
    completion_response = requests.post(
        f"{base_url}/completions",
        json=completion_payload,
        timeout=arguments.timeout,
    )
    check(
        completion_response.status_code == 200,
        "completions endpoint accepts per-request alpha",
        {"status": completion_response.status_code},
    )

    responses_payload = {
        "model": model,
        "input": "Return the word ready.",
        "temperature": 0,
        "max_output_tokens": 8,
        "vllm_xargs": {"weightless_alpha": 0.5},
    }
    responses_response = requests.post(
        f"{base_url}/responses",
        json=responses_payload,
        timeout=arguments.timeout,
    )
    check(
        responses_response.status_code == 200,
        "responses endpoint accepts per-request alpha",
        {"status": responses_response.status_code},
    )

    mixed_alphas = tuple(
        (0.0, 0.5, 1.0, -1.0, 2.5)[index % 5]
        for index in range(arguments.concurrency)
    )
    mixed_results = []
    with ThreadPoolExecutor(max_workers=arguments.concurrency) as executor:
        futures = {
            executor.submit(
                send_request,
                base_url,
                model,
                arguments.max_tokens,
                arguments.timeout,
                alpha,
            ): alpha
            for alpha in mixed_alphas
        }
        for future in as_completed(futures):
            mixed_results.append({"alpha": futures[future], **future.result()})
    check(
        all(
            result["status"] == 200 and result.get("completion_tokens", 0) >= 10
            for result in mixed_results
        ),
        "mixed concurrent alphas survive at least ten decode tokens",
        mixed_results,
    )

    cancel_status, received_data = cancel_stream(
        base_url, model, arguments.timeout, 2.5
    )
    check(
        cancel_status == 200 and received_data,
        "streaming request can be cancelled after execution starts",
        {"status": cancel_status, "received_data": received_data},
    )

    after_cancel = send_request(
        base_url, model, arguments.max_tokens, arguments.timeout, 0.0
    )
    check(
        after_cancel["status"] == 200
        and after_cancel.get("output_sha256")
        == valid_results["0.0"].get("output_sha256"),
        "cancellation and batch removal leave no stale alpha state",
        {"before": valid_results["0.0"], "after": after_cancel},
    )

    if arguments.check_prefix_cache:
        before = prefix_cache_totals(base_url, arguments.timeout)
        alpha_zero_first = send_cache_probe(
            base_url, model, arguments.timeout, 0.0
        )
        after_zero_first = prefix_cache_totals(base_url, arguments.timeout)
        alpha_zero_second = send_cache_probe(
            base_url, model, arguments.timeout, 0.0
        )
        after_zero_second = prefix_cache_totals(base_url, arguments.timeout)
        alpha_one_first = send_cache_probe(base_url, model, arguments.timeout, 1.0)
        after_one_first = prefix_cache_totals(base_url, arguments.timeout)
        alpha_one_second = send_cache_probe(base_url, model, arguments.timeout, 1.0)
        after_one_second = prefix_cache_totals(base_url, arguments.timeout)
        cache_detail = {
            "before": before,
            "after_zero_first": after_zero_first,
            "after_zero_second": after_zero_second,
            "after_one_first": after_one_first,
            "after_one_second": after_one_second,
            "statuses": [
                alpha_zero_first["status"],
                alpha_zero_second["status"],
                alpha_one_first["status"],
                alpha_one_second["status"],
            ],
        }
        check(
            all(status == 200 for status in cache_detail["statuses"])
            and after_zero_second[1] > after_zero_first[1]
            and after_one_first[1] == after_zero_second[1]
            and after_one_second[1] > after_one_first[1],
            "prefix cache hits only within the same effective alpha domain",
            cache_detail,
        )

    report["failures"] = failures
    if arguments.output:
        Path(arguments.output).write_text(json.dumps(report, indent=2) + "\n")
    print()
    print(f"weightless xargs live: {'all checks passed' if failures == 0 else f'{failures} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
