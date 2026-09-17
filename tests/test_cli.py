"""Offline CLI contracts: no deployment or remote endpoint required."""
import contextlib
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cli = load("cli", ROOT / "weightless.py")
setup = load("cli_setup", ROOT / "setup.py")
dash = load("cli_dash", ROOT / "scripts/dash.py")


class CommandTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "weightless.py"), *args],
                              input="", capture_output=True, text=True, timeout=10)

    def test_help_is_read_only_for_setup_serve_and_test(self):
        for args in [(), ("--help",), ("setup", "--help"), ("serve", "--help"),
                     ("test", "--help"), ("help", "serve"), ("help", "dash"),
                     ("help", "validate"), ("help", "inspect"), ("help", "export"),
                     ("help", "bake")]:
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("commands:" if args in ((), ("--help",)) else "usage:",
                              result.stdout.lower())
                self.assertNotIn("Traceback", result.stderr)

    def test_invalid_arguments_do_not_enter_workflow(self):
        for args in [("setup", "--bogus"), ("serve", "0", "--skip-wai"),
                     ("serve", "--unknown"), ("test", "--unknown"), ("setup",)]:
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertNotIn("== weightless", result.stdout)

    def test_typo_suggests_command(self):
        result = self.invoke("serbe")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Did you mean 'serve'", result.stderr)

    def test_dispatch_preserves_arguments(self):
        with patch.object(cli.os, "execvp") as execute:
            execute.side_effect = SystemExit(0)
            with self.assertRaises(SystemExit):
                cli.main(["inspect", "a file.gguf", "--json"])
        self.assertEqual(execute.call_args.args[1][-2:], ["a file.gguf", "--json"])

    def test_missing_executable_has_actionable_error(self):
        with patch.object(cli.os, "execvp", side_effect=FileNotFoundError("bash missing")), \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(cli.main(["test"]), 127)
        self.assertIn("cannot start test", errors.getvalue())

    def test_serve_parses_names_and_flags(self):
        with patch.object(setup, "quick_serve", return_value=0) as serve, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.main(["serve", "DeepSeek", "Flash", "--skip-wait"]), 0)
        self.assertEqual(serve.call_args.args[1], "DeepSeek Flash")
        self.assertEqual(serve.call_args.kwargs, {"skip_assets": False, "skip_wait": True})

    def test_confirmation_reprompts_on_typo(self):
        with patch("builtins.input", side_effect=["yesterday", "no"]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(setup.CliIO().confirm("Deploy?"))
        self.assertIn("enter yes or no", output.getvalue())

    def test_tui_failure_does_not_restart_workflow(self):
        if setup.curses is None:
            self.skipTest("curses unavailable")
        def wrapper(callback):
            return callback(Mock())
        with patch.object(setup.sys.stdin, "isatty", return_value=True), \
             patch.object(setup.sys.stdout, "isatty", return_value=True), \
             patch.object(setup.curses, "wrapper", side_effect=wrapper), \
             patch.object(setup, "_tui_main", side_effect=setup.curses.error("lost terminal")), \
             patch.object(setup, "run") as run, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(setup.main([]), 1)
        run.assert_not_called()

    def test_plain_mode_and_tui_initialization_fallback_run_once(self):
        if setup.curses is None:
            self.skipTest("curses unavailable")
        for argv in (["--plain"], []):
            with self.subTest(argv=argv), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()), \
                 patch.object(setup.sys.stdin, "isatty", return_value=True), \
                 patch.object(setup.sys.stdout, "isatty", return_value=True), \
                 patch.object(setup.curses, "wrapper", side_effect=setup.curses.error("no terminal")) as wrapper, \
                 patch.object(setup, "splash_cli"), patch.object(setup, "completion"), \
                 patch.object(setup, "run", return_value=0) as run:
                self.assertEqual(setup.main(argv), 0)
                run.assert_called_once()
                self.assertEqual(wrapper.call_count, 0 if argv else 1)

    def test_ambiguous_external_lane_never_deploys(self):
        stacks = [{"name": "external alpha", "match": "alpha", "boot": "start"},
                  {"name": "external beta", "match": "beta", "boot": "start"}]
        with patch.object(setup, "EXTERNAL_STACKS", stacks), \
             patch.object(setup, "read_lane_env", return_value={}), \
             patch.object(setup, "default_base", return_value="http://localhost:8888/v1"), \
             patch.object(setup, "quick_serve_external") as deploy:
            self.assertEqual(setup.quick_serve(Mock(), "external"), 2)
        deploy.assert_not_called()


class DashboardTests(unittest.TestCase):
    def test_real_http_snapshot_and_http_failure(self):
        paths = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                self.send_response(503 if self.path == "/offline/metrics" else 200)
                self.end_headers()
                self.wfile.write(b"vllm:num_requests_running 3\n")

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            for suffix, code in [("/v1", 0), ("/metrics", 0), ("/offline", 1)]:
                with self.subTest(suffix=suffix):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "weightless.py"), "dash", base + suffix, "--once"],
                        capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, code, result.stderr)
                    self.assertNotIn("\x1b", result.stdout + result.stderr)
                    if code == 0:
                        self.assertIn("running 3", result.stdout)
                    else:
                        self.assertIn("503", result.stderr)
            self.assertEqual(paths, ["/metrics", "/metrics", "/offline/metrics"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_invalid_options_fail_before_network(self):
        for args in [["--interval", value] for value in ("0", "-1", "nan", "inf")] + [
                ["--timeout", "0"], ["localhost:8888"], ["http://localhost:bad"],
                ["http://localhost?token=abc"]]:
            with self.subTest(args=args), patch.object(dash, "fetch") as fetch, \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                dash.main(args)
            self.assertEqual(error.exception.code, 2)
            fetch.assert_not_called()

    def test_snapshot_normalizes_endpoint_and_passes_timeout(self):
        with patch.object(dash, "fetch", return_value="vllm:num_requests_running 2") as fetch, \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(dash.main(["http://localhost:8888/v1/", "--once", "--timeout", "3"]), 0)
        fetch.assert_called_once_with("http://localhost:8888", timeout=3)
        self.assertIn("running 2", output.getvalue())
        self.assertNotIn("\x1b", output.getvalue())

    def test_non_metrics_response_fails_snapshot(self):
        with patch.object(dash, "fetch", return_value="<html>Not metrics</html>"), \
             contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(dash.main(["--once"]), 1)
        self.assertIn("no vLLM metrics", errors.getvalue())
        self.assertNotIn("\x1b", errors.getvalue())

    def test_stream_redirected_output_has_no_escape_codes(self):
        with patch.object(dash, "fetch", return_value="vllm:num_requests_running 1"), \
             patch.object(dash.time, "sleep", side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(KeyboardInterrupt):
                dash.main([])
        self.assertIn("running 1", output.getvalue())
        self.assertNotIn("\x1b", output.getvalue())

    def test_no_color_supports_empty_environment_value(self):
        with patch.dict(os.environ, {"NO_COLOR": ""}), \
             patch.object(setup.sys.stdout, "isatty", return_value=True):
            self.assertFalse(setup.CliIO().color)


if __name__ == "__main__":
    unittest.main()
