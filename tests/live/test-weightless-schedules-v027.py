#!/usr/bin/env python3
"""Live H100 checks for Weightless Milestone 2 schedules and layer masks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import pathlib
import time
import urllib.error
import urllib.request


def post_json(url: str, body: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def wait_for_summary(
    trace_dir: pathlib.Path,
    previous: set[pathlib.Path],
    timeout: float,
) -> pathlib.Path | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        created = set(trace_dir.glob("wt-*.summary.json")) - previous
        if created:
            return max(created, key=lambda path: path.stat().st_mtime_ns)
        time.sleep(0.1)
    return None


def wait_for_summaries(
    trace_dir: pathlib.Path,
    previous: set[pathlib.Path],
    count: int,
    timeout: float,
) -> list[pathlib.Path]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        created = sorted(set(trace_dir.glob("wt-*.summary.json")) - previous)
        if len(created) >= count:
            return created
        time.sleep(0.1)
    return sorted(set(trace_dir.glob("wt-*.summary.json")) - previous)


def metric_total(metrics_text: str, name: str) -> float | None:
    values = []
    for line in metrics_text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            values.append(float(line.rsplit(" ", 1)[1]))
    return sum(values) if values else None


def prefix_cache_totals(base_url: str, timeout: float) -> tuple[float, float]:
    with urllib.request.urlopen(
        base_url.rstrip("/") + "/metrics", timeout=timeout
    ) as response:
        text = response.read().decode()
    queries = metric_total(text, "vllm:prefix_cache_queries_total")
    hits = metric_total(text, "vllm:prefix_cache_hits_total")
    if queries is None or hits is None:
        raise RuntimeError("prefix-cache counters are unavailable")
    return queries, hits


def generated_text(payload: dict) -> str:
    message = payload["choices"][0]["message"]
    return (message.get("reasoning_content") or "") + (message.get("content") or "")


def request_body(model: str, prompt: str, max_tokens: int, xargs: dict) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "vllm_xargs": xargs,
    }


def expected_alpha(
    token_ordinal: int,
    decision_ordinal: int,
    *,
    prefill_alpha: float,
    decode_alpha: float,
) -> float:
    if token_ordinal < decision_ordinal:
        return prefill_alpha
    position = token_ordinal - decision_ordinal
    if 0 <= position < 2:
        return 0.0
    if 2 <= position < 5:
        return 0.2 + ((position - 2) / 2) * 0.8
    return decode_alpha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen-weightless")
    parser.add_argument("--trace-dir", required=True, type=pathlib.Path)
    parser.add_argument("--layers", default="10,30,58")
    parser.add_argument("--active-layers", default="10,58")
    parser.add_argument("--max-tokens", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--check-prefix-cache", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    trace_layers = [int(value) for value in args.layers.split(",") if value]
    active_layers = [int(value) for value in args.active_layers.split(",") if value]
    if not trace_layers or not active_layers:
        parser.error("--layers and --active-layers must not be empty")
    excluded_layers = set(trace_layers).difference(active_layers)
    if not excluded_layers:
        parser.error("--layers must include at least one layer outside --active-layers")

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    prompt = "List six practical reasons to use deterministic software tests."
    legacy_status, legacy_payload = post_json(
        url,
        request_body(
            args.model,
            prompt,
            args.max_tokens,
            {"weightless_alpha": 0.5},
        ),
        args.timeout,
    )
    structured_status, structured_payload = post_json(
        url,
        request_body(
            args.model,
            prompt,
            args.max_tokens,
            {"weightless": {"version": 1, "controls": {"alpha": 0.5}}},
        ),
        args.timeout,
    )
    check(
        legacy_status == structured_status == 200
        and generated_text(legacy_payload) == generated_text(structured_payload),
        "legacy and structured alpha-only requests remain identical",
        f"legacy={legacy_status}, structured={structured_status}",
    )

    before = set(args.trace_dir.glob("wt-*.summary.json"))
    controls = {
        "prefill_alpha": 0.1,
        "decode_alpha": 0.8,
        "layers": active_layers,
        "schedule": [
            {"start": 0, "end": 2, "alpha": 0.0},
            {"start": 2, "end": 5, "start_alpha": 0.2, "end_alpha": 1.0},
        ],
    }
    trace = {
        "layers": trace_layers,
        "metrics": ["effective_alpha", "coefficient", "post_coefficient"],
        "token_stride": 1,
        "max_records": 4096,
    }
    status, payload = post_json(
        url,
        request_body(
            args.model,
            prompt,
            args.max_tokens,
            {"weightless": {"version": 1, "controls": controls, "trace": trace}},
        ),
        args.timeout,
    )
    check(status == 200, "scheduled layer-mask request completes", str(payload.get("error", "")))

    summary_path = wait_for_summary(args.trace_dir, before, args.timeout) if status == 200 else None
    check(summary_path is not None, "scheduled request writes a trace summary")
    records = []
    if summary_path is not None:
        records_path = summary_path.with_name(summary_path.name.replace(".summary.json", ".jsonl"))
        if records_path.is_file():
            records = [json.loads(line) for line in records_path.read_text().splitlines()]
    check(bool(records), "scheduled request writes telemetry records")

    if records:
        prompt_ordinals = [
            record["token_ordinal"] for record in records if record["phase"] == "prompt"
        ]
        decision_ordinal = max(prompt_ordinals)
        mismatches = []
        for record in records:
            observed = record["metrics"]["effective_alpha"]
            if record["layer"] in excluded_layers:
                expected = 0.0
            else:
                expected = expected_alpha(
                    record["token_ordinal"],
                    decision_ordinal,
                    prefill_alpha=0.1,
                    decode_alpha=0.8,
                )
            if abs(observed - expected) > 5e-3:
                mismatches.append(
                    (record["token_ordinal"], record["layer"], observed, expected)
                )
        check(
            not mismatches,
            "telemetry confirms phase, pulse, ramp, and layer-mask alphas",
            repr(mismatches[:8]),
        )
        first_sample = [
            record
            for record in records
            if record["token_ordinal"] == decision_ordinal
            and record["layer"] in active_layers
        ]
        check(
            bool(first_sample)
            and all(record["metrics"]["effective_alpha"] == 0.0 for record in first_sample),
            "final prefill row uses decode position zero for the first sample",
        )

    concurrent_before = set(args.trace_dir.glob("wt-*.summary.json"))
    concurrent_bodies = []
    for index in range(args.concurrency):
        layer = active_layers[index % len(active_layers)]
        alpha = (0.25, 0.75)[index % 2]
        concurrent_bodies.append(
            request_body(
                args.model,
                f"Return the integer {index} and then five short words.",
                12,
                {
                    "weightless": {
                        "version": 1,
                        "controls": {"alpha": alpha, "layers": [layer]},
                        "trace": {
                            "layers": active_layers,
                            "metrics": ["effective_alpha"],
                            "max_records": 512,
                        },
                    }
                },
            )
        )
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        concurrent_results = list(
            executor.map(
                lambda body: post_json(url, body, args.timeout),
                concurrent_bodies,
            )
        )
    check(
        all(status == 200 for status, _ in concurrent_results),
        "mixed concurrent layer masks complete",
    )
    concurrent_summaries = wait_for_summaries(
        args.trace_dir,
        concurrent_before,
        sum(status == 200 for status, _ in concurrent_results),
        args.timeout,
    )
    isolated = len(concurrent_summaries) == args.concurrency
    for summary in concurrent_summaries:
        records_path = summary.with_name(summary.name.replace(".summary.json", ".jsonl"))
        trace_records = [
            json.loads(line) for line in records_path.read_text().splitlines()
        ]
        nonzero_layers = {
            record["layer"]
            for record in trace_records
            if abs(record["metrics"]["effective_alpha"]) > 1e-6
        }
        nonzero_values = {
            round(record["metrics"]["effective_alpha"], 3)
            for record in trace_records
            if abs(record["metrics"]["effective_alpha"]) > 1e-6
        }
        isolated &= len(nonzero_layers) == 1 and nonzero_values in ({0.25}, {0.75})
    check(isolated, "concurrent trace rows isolate each request's layer mask")

    invalid_status, _ = post_json(
        url,
        request_body(
            args.model,
            "Return READY.",
            4,
            {
                "weightless": {
                    "version": 1,
                    "controls": {
                        "schedule": [
                            {"start": 0, "end": 3, "alpha": 0.0},
                            {"start": 2, "end": 4, "alpha": 1.0},
                        ]
                    },
                }
            },
        ),
        args.timeout,
    )
    check(invalid_status == 400, "overlapping schedule returns HTTP 400", str(invalid_status))

    invalid_layer_status, _ = post_json(
        url,
        request_body(
            args.model,
            "Return READY.",
            4,
            {
                "weightless": {
                    "version": 1,
                    "controls": {"layers": [max(trace_layers) + 100]},
                }
            },
        ),
        args.timeout,
    )
    check(
        invalid_layer_status == 400,
        "server-disabled intervention layer returns HTTP 400",
        str(invalid_layer_status),
    )

    if args.check_prefix_cache:
        marker = str(time.time_ns())
        cache_prompt = ("Milestone two cache identity " + marker + ". ") * 700
        cache_a = {
            "weightless": {
                "version": 1,
                "controls": {
                    "layers": [active_layers[-1], active_layers[0]],
                    "schedule": [
                        {"start": 4, "end": 8, "alpha": 1.0},
                        {"start": 0, "end": 2, "alpha": 0.0},
                    ],
                },
            }
        }
        cache_equivalent = {
            "weightless": {
                "version": 1,
                "controls": {
                    "layers": f"{active_layers[0]},{active_layers[-1]}",
                    "schedule": [
                        {"start": 0, "end": 2, "alpha": 0.0},
                        {"start": 4, "end": 8, "alpha": 1.0},
                    ],
                },
            }
        }
        cache_distinct = {
            "weightless": {
                "version": 1,
                "controls": {
                    "layers": [active_layers[0]],
                    "schedule": [{"start": 0, "end": 2, "alpha": 0.0}],
                },
            }
        }

        def cache_probe(xargs: dict) -> int:
            status, _ = post_json(
                url,
                request_body(args.model, cache_prompt, 1, xargs),
                args.timeout,
            )
            return status

        before_cache = prefix_cache_totals(args.base_url, args.timeout)
        statuses = [cache_probe(cache_a)]
        after_first = prefix_cache_totals(args.base_url, args.timeout)
        statuses.append(cache_probe(cache_a))
        after_second = prefix_cache_totals(args.base_url, args.timeout)
        statuses.append(cache_probe(cache_equivalent))
        after_equivalent = prefix_cache_totals(args.base_url, args.timeout)
        statuses.append(cache_probe(cache_distinct))
        after_distinct = prefix_cache_totals(args.base_url, args.timeout)
        check(
            all(status == 200 for status in statuses)
            and after_second[1] > after_first[1]
            and after_equivalent[1] > after_second[1]
            and after_distinct[1] == after_equivalent[1],
            "canonical schedules hit and distinct controls miss prefix cache",
            repr(
                {
                    "before": before_cache,
                    "first": after_first,
                    "second": after_second,
                    "equivalent": after_equivalent,
                    "distinct": after_distinct,
                }
            ),
        )

    print()
    print(
        "weightless schedules live: "
        + ("all checks passed" if not failures else f"{len(failures)} failure(s)")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
