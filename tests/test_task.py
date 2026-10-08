"""Aggregation contracts for scripts/task.py (omp session token/time usage).

The journal is append-only and read while omp is still writing, so the tail
line can be partial JSON: it must be skipped, not fatal. Records without a
usage block (user messages, tool results) contribute nothing.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


task = load("task", ROOT / "scripts" / "task.py")


def journal(tmpdir, records):
    path = Path(tmpdir) / "s.jsonl"
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return str(path)


def assistant(ts, **usage):
    return {"timestamp": ts, "message": {"role": "assistant", "usage": usage}}


class ScanTests(unittest.TestCase):
    def test_sums_usage_and_tracks_time(self):
        with tempfile.TemporaryDirectory() as d:
            path = journal(d, [
                assistant("2026-10-08T16:00:00Z", input=100, output=10,
                          cacheRead=500, cacheWrite=0, totalTokens=610),
                {"timestamp": "2026-10-08T16:01:00Z", "message": {"role": "user"}},
                assistant("2026-10-08T16:02:00Z", input=200, output=20,
                          cacheRead=600, cacheWrite=50, totalTokens=870),
            ])
            s = task.scan(path)
        self.assertEqual(s["calls"], 2)
        self.assertEqual(s["input"], 300)
        self.assertEqual(s["output"], 30)
        self.assertEqual(s["cache_read"], 1100)
        self.assertEqual(s["cache_write"], 50)
        self.assertEqual(s["ctx"], 870)  # last reported context size
        self.assertEqual((s["last"] - s["first"]).total_seconds(), 120)

    def test_partial_tail_line_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.jsonl"
            with open(path, "w") as fh:
                fh.write(json.dumps(assistant("2026-10-08T16:00:00Z", input=5, output=1)) + "\n")
                fh.write('{"type":"message","id":"truncated')
            s = task.scan(str(path))
        self.assertEqual(s["calls"], 1)
        self.assertEqual(s["input"], 5)

    def test_no_usage_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            path = journal(d, [{"timestamp": "2026-10-08T16:00:00Z",
                                "message": {"role": "user"}}])
            self.assertIsNone(task.scan(path))

    def test_missing_file_returns_none(self):
        self.assertIsNone(task.scan("/nonexistent/session.jsonl"))


if __name__ == "__main__":
    unittest.main()
