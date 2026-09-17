#!/usr/bin/env python3
"""Compare alpha-zero responses with a stored unsteered evaluation."""

import argparse
import difflib
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def post_json(url: str, payload: dict, timeout: float) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen-weightless")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--omit-alpha", action="store_true")
    parser.add_argument("--capture-current", type=Path)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.capture_only and args.capture_current is None:
        parser.error("--capture-only requires --capture-current")
    return args


def main() -> int:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    protocol = baseline.get("protocol", {})
    rows = baseline.get("results", [])[: args.count]
    report = {
        "baseline": str(args.baseline),
        "base_url": args.base_url,
        "model": args.model,
        "comparison_mode": "capture-only" if args.capture_only else "exact",
        "protocol": {
            "enable_thinking": bool(protocol.get("enable_thinking", False)),
            "temperature": protocol.get("temperature", 0),
            "top_p": 1,
            "seed": 0,
            "max_tokens": protocol.get("max_tokens", 1024),
            "weightless_alpha": "omitted" if args.omit_alpha else 0.0,
        },
        "results": [],
    }
    captured_rows = []
    failures = 0
    for row in rows:
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": row["prompt"]}],
            "temperature": report["protocol"]["temperature"],
            "top_p": report["protocol"]["top_p"],
            "seed": report["protocol"]["seed"],
            "max_tokens": report["protocol"]["max_tokens"],
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": report["protocol"]["enable_thinking"]
            },
        }
        if not args.omit_alpha:
            payload["vllm_xargs"] = {"weightless_alpha": 0.0}
        started = time.monotonic()
        status, response = post_json(
            f"{args.base_url.rstrip('/')}/chat/completions",
            payload,
            args.timeout,
        )
        elapsed = round(time.monotonic() - started, 4)
        current_response = ""
        current_reasoning = ""
        usage = {}
        if status == 200:
            message = response["choices"][0]["message"]
            current_response = message.get("content") or ""
            current_reasoning = message.get("reasoning_content") or ""
            usage = response.get("usage", {})
        expected_response = row.get("response") or ""
        expected_reasoning = row.get("reasoning") or ""
        response_matches = status == 200 and current_response == expected_response
        reasoning_matches = status == 200 and current_reasoning == expected_reasoning
        passed = status == 200 if args.capture_only else (
            response_matches and reasoning_matches
        )
        failures += not passed
        result = {
            "i": row.get("i"),
            "status": status,
            "passed": passed,
            "elapsed_seconds": elapsed,
            "response_matches": response_matches,
            "reasoning_matches": reasoning_matches,
            "response_similarity": round(
                difflib.SequenceMatcher(
                    None, expected_response, current_response, autojunk=False
                ).ratio(),
                6,
            ),
            "expected_response_length": len(expected_response),
            "current_response_length": len(current_response),
            "expected_response_sha256": digest(expected_response),
            "current_response_sha256": digest(current_response),
            "expected_reasoning_sha256": digest(expected_reasoning),
            "current_reasoning_sha256": digest(current_reasoning),
            "usage": usage,
        }
        report["results"].append(result)
        captured_rows.append(
            {
                "i": row.get("i"),
                "prompt": row["prompt"],
                "response": current_response,
                "reasoning": current_reasoning,
                "_finish_reason": (
                    response.get("choices", [{}])[0].get("finish_reason")
                    if status == 200
                    else None
                ),
                "_usage": usage,
            }
        )
        label = "CAPTURE" if args.capture_only and passed else (
            "PASS" if passed else "FAIL"
        )
        print(
            f"  [{label}] item {row.get('i')}: status={status} "
            f"response={response_matches} reasoning={reasoning_matches} "
            f"elapsed={elapsed:.4f}s"
        )

    report["failures"] = failures
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.capture_current:
        args.capture_current.parent.mkdir(parents=True, exist_ok=True)
        args.capture_current.write_text(
            json.dumps(
                {
                    "protocol": report["protocol"],
                    "results": captured_rows,
                },
                indent=2,
            )
            + "\n"
        )
    print()
    print(
        "alpha-zero baseline: "
        + ("all checks passed" if not failures else f"{failures} failure(s)")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
