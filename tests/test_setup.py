"""Offline regressions for client setup and the serving status router."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load("wizard", ROOT / "setup.py")
router = load("router", ROOT / "scripts/dspark-router.py")


class SetupTests(unittest.TestCase):
    def test_endpoint_smoke_parses_formatted_json_and_exact_model_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            curl = Path(tmp) / "curl"
            curl.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_MODEL_RESPONSE"\n')
            curl.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.environ["PATH"],
                       WEIGHTLESS_MODEL="inkling-small-nvfp4")
            for model, expected in [("inkling-small-nvfp4", 0), ("inkling-small-nvfp4-other", 1)]:
                env["TEST_MODEL_RESPONSE"] = json.dumps({"data": [{"id": model}]}, indent=2)
                result = subprocess.run(["bash", str(ROOT / "tests/01-endpoint.sh")],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_hermes_uses_selected_model_context(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "HERMES_CONFIG", tmp + "/config.yaml"):
            path = Path(setup.HERMES_CONFIG)
            path.write_text('model:\n  default: old\n  extra: keep\nagent:\n  max_turns: 12\n')
            setup.install_hermes("localhost", "glm53-flash")
            text = path.read_text()
            self.assertIn("context_length: 131072", text)
            self.assertIn("max_tokens: 16384", text)
            self.assertIn("extra: keep", text)
            self.assertIn("max_turns: 12", text)

    def test_live_context_takes_precedence_over_template(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "HERMES_CONFIG", tmp + "/config.yaml"):
            setup.install_hermes("localhost", "glm53-flash", context_length=262144)
            self.assertIn("context_length: 262144", Path(setup.HERMES_CONFIG).read_text())

    def test_inkling_profile_enables_native_tools(self):
        profile = setup.model_profile("inkling-small-nvfp4")
        self.assertTrue(profile["supports_tools"])
        self.assertEqual(profile["context_length"], 65536)

    def test_omp_uses_live_window_without_changing_other_models(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "OMP_MODELS", tmp + "/models.yml"):
            setup.install_provider("node.local", model="glm53-flash", context_length=262144)
            text = Path(setup.OMP_MODELS).read_text()
            self.assertIn("contextWindow: 262144", text.split("- id: glm53-flash")[1].split("- id:")[0])
            self.assertIn("contextWindow: 65536", text.split("- id: inkling-small-nvfp4")[1].split("- id:")[0])

    def test_clients_keep_verified_custom_endpoint_and_fit_small_window(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(setup, "OMP_MODELS", tmp + "/models.yml"), \
             patch.object(setup, "HERMES_CONFIG", tmp + "/hermes.yaml"):
            base = "https://node.example:8443/custom/v1"
            setup.install_provider("node.example", "glm53-flash", 8192, base_url=base)
            setup.install_hermes("node.example", "glm53-flash", 8192, base_url=base)
            omp = Path(setup.OMP_MODELS).read_text().split("- id: glm53-flash")[1].split("- id:")[0]
            hermes = Path(setup.HERMES_CONFIG).read_text()
            self.assertIn(base, omp)
            self.assertIn(base, hermes)
            self.assertIn("maxTokens: 2048", omp)
            self.assertIn("max_tokens: 2048", hermes)

    def test_inkling_without_router_uses_configured_engine_port(self):
        dialog = Mock()
        dialog.text.side_effect = lambda prompt, default: default
        dialog.confirm.return_value = False
        env = "MASTER_ADDR=node.local\nSERVED_MODEL_NAME=inkling-small-nvfp4\nVLLM_PORT=9082\n"
        from unittest.mock import mock_open
        with patch("builtins.open", mock_open(read_data=env)), \
             patch.object(setup, "probe_models", side_effect=[(None, "refused"), (["inkling-small-nvfp4"], None)]) as probe, \
             patch.object(setup, "probe_generation", return_value=("pong", None)), \
             patch.object(setup, "served_context", return_value=65536):
            self.assertEqual(setup.tests_chain(dialog, "node.local", 5), 0)
        self.assertEqual(probe.call_args.args[0], "http://node.local:9082/v1")

    def test_diagnosis_uses_same_normalized_url_for_discovery_and_generation(self):
        for base, expected in [("localhost:8000", "http://localhost:8000/v1"),
                               ("https://host/proxy/v1/", "https://host/proxy/v1")]:
            with self.subTest(base=base), \
                 patch.object(setup.socket, "gethostbyname", return_value="127.0.0.1"), \
                 patch.object(setup.socket, "create_connection"), \
                 patch.object(setup, "probe_models", return_value=(["inkling-small-nvfp4"], None)) as models, \
                 patch.object(setup, "probe_generation", return_value=("pong", None)) as generation:
                self.assertEqual(setup.diagnose_chain(Mock(), base), 0)
            models.assert_called_once_with(expected)
            generation.assert_called_once_with(expected, "inkling-small-nvfp4")

    def test_server_context_is_read_by_model_id(self):
        from io import BytesIO
        response = BytesIO(json.dumps({"data": [{"id": "other", "max_model_len": 8192},
                                                {"id": "glm53-flash", "max_model_len": 262144}]}).encode())
        with patch.object(setup.urllib.request, "urlopen", return_value=response):
            self.assertEqual(setup.served_context("http://localhost/v1", "glm53-flash"), 262144)

    def test_inkling_env_matches_client_window(self):
        lane = setup.LANES[5]
        text = (ROOT / lane["example"]).read_text()
        self.assertIn("MAX_MODEL_LEN=65536", text)
        self.assertIn("GPU_MEMORY_UTILIZATION=0.78", text)
        self.assertFalse(lane["steering_supported"])

    def test_legacy_provider_is_not_required_by_omp_smoke(self):
        text = (ROOT / "tests/04-omp-headless.sh").read_text()
        self.assertNotIn('"^  dspark:"', text)
        self.assertIn("weightless/", text)

    def test_generation_probe_rejects_empty_output(self):
        from io import BytesIO
        response = BytesIO(json.dumps({"choices": [{"message": {"content": ""}}]}).encode())
        with patch.object(setup.urllib.request, "urlopen", return_value=response):
            content, error = setup.probe_generation("http://localhost/v1", "inkling-small-nvfp4")
        self.assertIsNone(content)
        self.assertIsNotNone(error)

    def test_generation_allows_reasoning_budget_and_explains_truncation(self):
        def response(request, timeout):
            budget = json.loads(request.data)["max_tokens"]
            choice = {"message": {"content": "pong" if budget >= 512 else ""},
                      "finish_reason": "stop" if budget >= 512 else "length"}
            return io.BytesIO(json.dumps({"choices": [choice]}).encode())
        with patch.object(setup.urllib.request, "urlopen", side_effect=response):
            self.assertEqual(setup.probe_generation("http://localhost/v1", "thinking-model"), ("pong", None))
        truncated = io.BytesIO(b'{"choices":[{"message":{"content":""},"finish_reason":"length"}]}')
        with patch.object(setup.urllib.request, "urlopen", return_value=truncated):
            self.assertIn("output budget", setup.probe_generation("http://localhost/v1", "thinking-model")[1])

    def test_failed_generation_does_not_change_client_configs(self):
        io = Mock()
        io.text.return_value = "http://localhost:8000/v1"
        with patch.object(setup, "probe_models", return_value=(["inkling-small-nvfp4"], None)), \
             patch.object(setup, "probe_generation", return_value=(None, "timed out")), \
             patch.object(setup, "install_provider") as omp, patch.object(setup, "install_hermes") as hermes:
            self.assertEqual(setup.tests_chain(io), 1)
        omp.assert_not_called()
        hermes.assert_not_called()

    def test_full_client_chain_uses_live_context(self):
        io = Mock()
        io.text.return_value = "http://localhost:8000/v1"
        io.confirm.side_effect = [True, True, False]
        with patch.object(setup, "probe_models", return_value=(["glm53-flash"], None)), \
             patch.object(setup, "probe_generation", return_value=("pong", None)), \
             patch.object(setup, "served_context", return_value=262144), \
             patch.object(setup, "omp_roles_chain"), \
             patch.object(setup, "install_provider") as omp, patch.object(setup, "install_hermes") as hermes:
            self.assertEqual(setup.tests_chain(io), 0)
        omp.assert_called_once_with("localhost", model="glm53-flash", context_length=262144, base_url="http://localhost:8000/v1")
        hermes.assert_called_once_with("localhost", "glm53-flash", context_length=262144, base_url="http://localhost:8000/v1")

    def test_demo_does_not_write_or_contact_servers(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(setup, "DEMO", True), \
             patch.object(setup, "HERMES_CONFIG", tmp + "/hermes.yaml"), \
             patch.object(setup, "OMP_MODELS", tmp + "/models.yml"), \
             patch.object(setup.urllib.request, "urlopen") as request:
            setup.install_provider("example", model="glm53-flash", context_length=262144)
            setup.install_hermes("example", "glm53-flash", context_length=262144)
            setup.probe_generation("http://example/v1", "glm53-flash")
            setup.served_context("http://example/v1", "glm53-flash")
            self.assertEqual(list(Path(tmp).iterdir()), [])
        request.assert_not_called()


class RouterTests(unittest.TestCase):
    def test_inkling_tools_reach_upstream_unchanged(self):
        payload = {"model": "inkling-small-nvfp4", "messages": [{"role": "user", "content": "Use lookup"}],
                   "tools": [{"type": "function", "function": {"name": "lookup"}}], "tool_choice": "auto"}
        raw = json.dumps(payload).encode()
        handler = object.__new__(router.Handler)
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}
        handler.rfile = io.BytesIO(raw)
        handler.wfile = io.BytesIO()
        connection = Mock()
        response = Mock(status=200)
        response.getheader.return_value = "application/json"
        response.getheaders.return_value = [("Content-Type", "application/json")]
        response.read1.side_effect = [b'{}', b'']
        connection.getresponse.return_value = response
        with patch.object(router.http.client, "HTTPConnection", return_value=connection), \
             patch.object(handler, "send_response"), patch.object(handler, "send_header"), patch.object(handler, "end_headers"):
            handler.do_POST()
        self.assertEqual(json.loads(connection.request.call_args.kwargs["body"]), payload)
        connection.close.assert_called_once()

    def test_unhealthy_engine_is_excluded_and_connection_closed(self):
        connection = Mock()
        connection.getresponse.return_value.status = 503
        with patch.object(router, "ROUTES", {"inkling-small-nvfp4": 8082}), \
             patch.object(router.http.client, "HTTPConnection", return_value=connection):
            self.assertEqual(router.upstream_models(), [])
        connection.request.assert_called_once_with("GET", "/health")
        connection.close.assert_called_once()

    def test_healthy_engine_preserves_live_context_metadata(self):
        connection = Mock()
        health = Mock(status=200)
        response = Mock(status=200)
        models = [{"id": "inkling-small-nvfp4", "max_model_len": 131072}]
        response.read.return_value = json.dumps({"data": models}).encode()
        connection.getresponse.side_effect = [health, response]
        with patch.object(router, "ROUTES", {"inkling-small-nvfp4": 8082}), \
             patch.object(router.http.client, "HTTPConnection", return_value=connection):
            self.assertEqual(router.upstream_models(), models)
        connection.close.assert_called_once()

    def test_no_models_advertised_when_all_lanes_are_down(self):
        handler = object.__new__(router.Handler)
        handler.path = "/v1/models"
        with patch.object(router, "upstream_models", return_value=[]), patch.object(handler, "_send_json") as send:
            handler.do_GET()
        self.assertEqual(send.call_args.args, (200, {"object": "list", "data": []}))

    def test_health_fails_when_no_engine_is_ready(self):
        handler = object.__new__(router.Handler)
        handler.path = "/health"
        with patch.object(router, "upstream_models", return_value=[]), patch.object(handler, "_send_json") as send:
            handler.do_GET()
        self.assertEqual(send.call_args.args[0], 503)


if __name__ == "__main__":
    unittest.main()
