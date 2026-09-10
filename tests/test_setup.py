"""Offline regressions for client setup and the serving status router."""
import importlib.util
import io
import json
import os
import shlex
import shutil
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
    def test_chat_smoke_rejects_reasoning_leaked_into_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            curl = Path(tmp) / "curl"
            curl.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_CHAT_RESPONSE"\n')
            curl.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.environ["PATH"])
            for content, expected in [("pong", 0), ("We need to answer pong. Let me think.", 1), ("", 1)]:
                env["TEST_CHAT_RESPONSE"] = json.dumps({"choices": [{"message": {"content": content}}]})
                result = subprocess.run(["bash", str(ROOT / "tests/smoke/02-chat.sh")], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)

    def test_endpoint_smoke_parses_formatted_json_and_exact_model_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            curl = Path(tmp) / "curl"
            curl.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_MODEL_RESPONSE"\n')
            curl.chmod(0o755)
            env = dict(os.environ, PATH=tmp + os.pathsep + os.environ["PATH"],
                       WEIGHTLESS_MODEL="inkling-small-nvfp4")
            for model, expected in [("inkling-small-nvfp4", 0), ("inkling-small-nvfp4-other", 1)]:
                env["TEST_MODEL_RESPONSE"] = json.dumps({"data": [{"id": model}]}, indent=2)
                result = subprocess.run(["bash", str(ROOT / "tests/smoke/01-endpoint.sh")],
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
        self.assertTrue(lane["steering_supported"])

    def test_inkling_deploy_stages_combined_patch_on_both_nodes_before_boot(self):
        steps = setup.deploy_commands(5, {"user": "tester", "head-ip": "head", "worker-ip": "worker"})
        transfers = [args for _, args in steps if args[0] == "scp"]
        self.assertTrue(any(any("inkling-model-gb10-steered.py" in arg for arg in args) for args in transfers))
        worker = next(i for i, (desc, _) in enumerate(steps) if desc == "sync Inkling patches to worker")
        self.assertLess(worker, len(steps) - 1)
        self.assertIn("tester@worker", steps[worker][1][-1])
        self.assertIn("inkling-model-gb10-steered.py", steps[worker][1][-1])

    def test_legacy_provider_is_not_required_by_omp_smoke(self):
        text = (ROOT / "tests/smoke/04-omp-headless.sh").read_text()
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


class AssetAndParkingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        Path(self.tmp.name, "tests/smoke").mkdir(parents=True)
        shutil.copyfile(ROOT / "tests/smoke/models.yml", Path(self.tmp.name, "tests/smoke/models.yml"))
        for lane in setup.LANES:
            if "example" not in lane:  # pure cloud lanes (e.g. Kimi K3) ship no env example
                continue
            dest = Path(self.tmp.name) / lane["example"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / lane["example"], dest)
        self.here = patch.object(setup, "HERE", self.tmp.name)
        self.here.start()
        self.addCleanup(self.here.stop)
        self.demo = patch.object(setup, "DEMO", False)
        self.demo.start()
        self.addCleanup(self.demo.stop)
        self.values = {"user": "tester", "head-ip": "192.0.2.1", "worker-ip": "192.0.2.2",
                       "worker2-ip": "192.0.2.3", "worker3-ip": "192.0.2.4",
                       "hf-cache": "/home/tester/.cache/huggingface"}

    def test_lane_metadata_and_asset_paths_match_recipes(self):
        for idx, lane in enumerate(setup.LANES):
            with self.subTest(lane=idx):
                if lane.get("cloud"):
                    # Cloud lanes hold assets on Modal volumes, not the rig.
                    self.assertEqual(lane["cloud"], "modal")
                    self.assertEqual(setup.asset_commands(idx, self.values,
                                                          "head.local"), [])
                    self.assertTrue(os.path.exists(
                        ROOT / lane["modal_app"]))
                    self.assertTrue(setup.park_other_lanes(
                        None, idx, self.values, "head.local"))
                    continue
                env = setup.lane_env(idx, self.values)
                self.assertEqual(lane["docker_image"], env[lane["image_key"]])
                if env.get("MODEL"):
                    self.assertEqual(lane["model_repo"], env["MODEL"])
                plan = setup.asset_commands(idx, self.values, "head.local")
                commands = "\n".join(shlex.join(argv) for _, argv in plan)
                self.assertIn(lane["model_repo"], commands)
                # Stock lanes ship no GLP vector (e.g. museglimmer) — no
                # steering downloads to assert on those.
                if lane.get("vector_repo"):
                    self.assertIn(lane["vector_repo"], commands)
                    self.assertIn(os.path.basename(env[lane["steer_key"]]), commands)
                self.assertIn("/home/tester/.cache/huggingface", commands)
                nodes = lane.get("nodes", 1)
                self.assertEqual(sum(desc.startswith("download GLP") for desc, _ in plan),
                                 nodes if lane.get("vector_repo") else 0)
                self.assertEqual(sum(desc.startswith("rsync model cache") for desc, _ in plan), nodes - 1)
                pulls = [argv for desc, argv in plan if desc.startswith("pull Docker")]
                self.assertEqual(len(pulls), 0 if lane.get("local_image") else nodes)
                for desc, argv in plan:
                    self.assertIn("tester@head.local", argv)
                    if desc.startswith("rsync model cache"):
                        self.assertIn("models--" + lane["model_repo"].replace("/", "--"), argv[-1])
                        self.assertIn("--progress", argv[-1])
                # The remote shell programs must parse without being executed.
                for _, argv in plan:
                    result = subprocess.run(["sh", "-n", "-c", argv[-1]], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_dsv4_pins_revision_and_qwen_materializes_mount(self):
        plan = setup.asset_commands(0, self.values, "head.local")
        self.assertIn("--revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062", plan[0][1][-1])
        qwen = "\n".join(argv[-1] for _, argv in setup.asset_commands(1, self.values, "head.local"))
        self.assertIn("unsloth/Qwen3.8-27B-NVFP4", qwen)
        self.assertIn("/home/tester/models-local-qwen38/Qwen3.8-27B-NVFP4", qwen)
        self.assertIn("/home/tester/models-local-qwen38/cvec", qwen)
        self.assertIn("rsync -aL", qwen)
        self.assertEqual(setup.validate_lane_env(setup.LANES[1], ""), ([], []))

    def test_multinode_env_rejects_mdns_fabric_addresses(self):
        # 2026-09-09: .local in MASTER_ADDR/VLLM_HOST_IP crash-looped three
        # DSV4 boots — host-networked containers cannot resolve mDNS names.
        self.assertEqual(setup.LANES[0]["steer_hook"], "ffn_out_pre_residual")
        env = ("MASTER_ADDR=spark-4687.local\nVLLM_HOST_IP=spark-4687.local\n"
               "WORKER_VLLM_HOST_IP=spark-5bc3.local\n")
        errors, _ = setup.validate_lane_env(setup.LANES[0], env)
        self.assertEqual(len(errors), 3)
        self.assertIn("fabric IP", errors[0])
        ok_env = ("MASTER_ADDR=192.168.100.1\nVLLM_HOST_IP=192.168.100.1\n"
                  "WORKER_VLLM_HOST_IP=192.168.100.2\n")
        self.assertEqual(setup.validate_lane_env(setup.LANES[0], ok_env), ([], []))
        # single-node lanes are exempt (no cross-node rendezvous)
        self.assertEqual(setup.validate_lane_env(setup.LANES[1], env), ([], []))

    def test_nemotron35_single_node_lane(self):
        lane = setup.LANES[7]
        self.assertEqual(lane["steer_key"], "WEIGHTLESS_GLP")
        env = setup.lane_env(7, self.values)
        self.assertEqual(env["NEMOTRON_IMAGE"], "vllm/vllm-openai:v0.28.0")
        self.assertEqual(env["SPECULATIVE_MODE"], "none")  # MTP is not steered
        self.assertIn("/home/tester/.cache/huggingface/", env["WEIGHTLESS_GLP"])
        plan = setup.asset_commands(7, self.values, "head.local")
        commands = "\n".join(shlex.join(argv) for _, argv in plan)
        self.assertIn("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", commands)
        self.assertIn("--revision cc84af2fe71647d87f4486c064f320e1e7535243", commands)
        self.assertIn("Nemotron-3.5-Lightning-30B-A3B-abliterated-GLP-51-L1-51-a1.0.gguf",
                      commands)
        self.assertNotIn("rsync model cache", commands)  # single node
        deploy = setup.deploy_commands(7, self.values, "head.local")
        text = "\n".join(shlex.join(argv) for _, argv in deploy)
        self.assertIn("nemotron35-glp/recipe/nemotron35", text)
        self.assertIn("nemotron35-glp/patches", text)
        self.assertIn("serve-nemotron35.sh", deploy[-1][1][-1])
        self.assertEqual(setup.CONTAINER_GREP[7], "nemotron35")
        self.assertIn((7, "nemotron35"),
                      setup.current_lanes("nemotron35\ninkling-sm121\n"))

    def test_museglimmer_single_node_lane(self):
        lane = setup.LANES[9]
        self.assertFalse(lane["steering_supported"])  # no GLP vector exists yet
        env = setup.lane_env(9, self.values)
        self.assertEqual(env["MUSE_IMAGE"], "vllm/vllm-openai:v0.28.0")
        self.assertEqual(env["SPECULATIVE_MODE"], "dflash")
        plan = setup.asset_commands(9, self.values, "head.local")
        commands = "\n".join(shlex.join(argv) for _, argv in plan)
        self.assertIn("nvidia/Muse-Glimmer-30B-NVFP4", commands)
        self.assertIn("--revision f45fad5689e9a4d937f7e872fbec20c4e8a74154", commands)
        # The DFlash drafter is prefetched alongside the target weights.
        self.assertIn("meta-models/Muse-Glimmer-30B-assistant", commands)
        self.assertIn("--revision e8192f3a8f617f74be2ce220360c89ef4789f39f",
                      commands)
        self.assertNotIn("rsync model cache", commands)  # single node
        deploy = setup.deploy_commands(9, self.values, "head.local")
        text = "\n".join(shlex.join(argv) for _, argv in deploy)
        self.assertIn("museglimmer/recipe/museglimmer", text)
        self.assertNotIn("hotfix", text)  # stock vLLM serves the arch natively
        self.assertIn("serve-museglimmer.sh", deploy[-1][1][-1])
        self.assertEqual(setup.CONTAINER_GREP[9], "museglimmer")
        self.assertIn((9, "museglimmer"),
                      setup.current_lanes("museglimmer\ninkling-sm121\n"))

    def test_saved_env_overrides_cache_workers_image_and_disabled_steering(self):
        path = Path(self.tmp.name) / setup.LANES[6]["target"]
        path.write_text('HF_CACHE="/srv/hf cache" # custom mount\nWORKER_HF_CACHE=/srv/worker-hf\n'
                        'WORKER_HOST=198.51.100.8\nMODEL=RedHatAI/GLM-5.3-Flash-NVFP4\n'
                        'GLM53_IMAGE=custom/image:pin\nWEIGHTLESS_STEER_PATH=\n')
        plan = setup.asset_commands(6, self.values, "head.local")
        commands = "\n".join(argv[-1] for _, argv in plan)
        self.assertIn("/srv/hf cache", commands)
        self.assertIn("/srv/worker-hf", commands)
        self.assertIn("tester@198.51.100.8", commands)
        self.assertIn("docker pull custom/image:pin", commands)
        self.assertNotIn("download GLP", str(plan))

    def test_qwen_lora_is_downloaded_and_packaged_in_its_mount(self):
        env = setup.lane_env(1, self.values)
        env["STEER_MODE"] = "lora"
        with patch.object(setup, "lane_env", return_value=env):
            plan = setup.asset_commands(1, self.values, "head.local")
        self.assertIn("download Qwen LoRA adapter", plan[-2][0])
        self.assertIn(setup.LANES[1]["vector_repo"], plan[-2][1][-1])
        command = plan[-1][1][-1]
        self.assertIn("/home/tester/models-local-qwen38/lora/qwen-abliterated/adapter_model.safetensors", command)
        self.assertIn('"target_modules": ["mlp.down_proj"]', command)
        self.assertEqual(subprocess.run(["sh", "-n", "-c", command]).returncode, 0)

    def test_token_env_file_precedence_and_missing_token_blocks_assets(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env-token"}), \
             patch.object(setup.os.path, "expanduser", return_value=self.tmp.name + "/token"):
            Path(self.tmp.name, "token").write_text("file-token\n")
            self.assertEqual(setup.hf_token(), "env-token")
            with patch.dict(os.environ, {"HF_TOKEN": ""}):
                self.assertEqual(setup.hf_token(), "file-token")
        dialog = Mock()
        with patch.object(setup, "hf_token", return_value=""), \
             patch.object(setup.subprocess, "run") as run:
            self.assertFalse(setup.prepare_assets(dialog, 6, self.values, "head.local"))
        run.assert_not_called()
        self.assertIn("HF_TOKEN", dialog.err.call_args.args[0])

    def test_download_token_uses_stdin_only_and_failure_stops_plan(self):
        dialog = Mock()
        dialog.confirm.return_value = True
        with patch.object(setup, "hf_token", return_value="secret-token"), \
             patch.object(setup.subprocess, "run", return_value=Mock(returncode=1)) as run:
            self.assertFalse(setup.prepare_assets(dialog, 6, self.values, "head.local"))
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs["input"], "secret-token\n")
        self.assertNotIn("secret-token", str(run.call_args.args))
        self.assertNotIn("secret-token", str(dialog.mock_calls))

    def test_download_shell_preserves_token_and_paths_without_executing_them(self):
        bindir = Path(self.tmp.name, ".cache/weightless-hf/bin")
        bindir.mkdir(parents=True)
        hf = bindir / "hf"
        hf.write_text('#!/bin/sh\nprintf "%s\\n" "$HF_TOKEN" "$@"\n')
        hf.chmod(0o755)
        command = setup.asset_commands(6, self.values, "head.local")[0][1][-1]
        token = "token with spaces; $(false) `false`"
        result = subprocess.run(["sh", "-c", command], input=token + "\n", text=True,
                                capture_output=True, env=dict(os.environ, HOME=self.tmp.name))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines()[0], token)
        self.assertIn("RedHatAI/GLM-5.3-Flash-NVFP4", result.stdout)

    def test_detection_disambiguates_lanes_and_parking_only_targets_other_lanes(self):
        running = setup.current_lanes("qwen38\nqwen38fn\nvllm-glm53tp2\nglm53\nnginx\n")
        self.assertEqual(running, [(1, "qwen38"), (2, "qwen38fn"), (6, "vllm-glm53tp2"), (3, "glm53")])
        self.assertEqual(setup.park_commands(6, [(6, "vllm-glm53tp2")], self.values), [])
        self.assertEqual(setup.park_commands(6, [], self.values), [])
        cmds = setup.park_commands(6, [(2, "qwen38fn")], self.values, "head.local")
        self.assertEqual(cmds[0][1][-1], "docker rm -f qwen38fn")

    def test_detection_failure_is_not_treated_as_idle(self):
        with patch.object(setup.subprocess, "run", return_value=Mock(returncode=255)), \
             patch.object(setup.subprocess, "call") as call:
            self.assertFalse(setup.park_other_lanes(Mock(), 6, self.values, "head.local"))
        call.assert_not_called()

    def test_switch_from_four_nodes_parks_all_old_ranks_only_after_confirmation(self):
        for accepted in (False, True):
            with self.subTest(accepted=accepted):
                dialog = Mock()
                dialog.confirm.return_value = accepted
                with patch.object(setup, "detect_current_lanes", return_value=[(3, "glm53")]) as detect, \
                     patch.object(setup.subprocess, "call", return_value=0) as call:
                    self.assertEqual(setup.park_other_lanes(dialog, 6, self.values, "head.local"), accepted)
                self.assertEqual(detect.call_count, 4)
                self.assertEqual(call.call_count, 4 if accepted else 0)
                self.assertIs(dialog.confirm.call_args.args[1], False)
                self.assertIn("relaunch later", str(dialog.info.mock_calls))
                self.assertIn("start-glm53-flash-dspark.sh", str(dialog.info.mock_calls))

    def test_park_failure_blocks_boot(self):
        dialog = Mock()
        dialog.confirm.return_value = True
        with patch.object(setup, "detect_current_lanes", return_value=[(5, "inkling-sm121")]), \
             patch.object(setup.subprocess, "call", return_value=1) as call:
            self.assertFalse(setup.park_other_lanes(dialog, 6, self.values, "head.local"))
        call.assert_called_once()

    def test_surviving_worker_discovers_the_rest_of_the_old_lane(self):
        dialog = Mock()
        dialog.confirm.return_value = True
        def detect(values, host, worker=None):
            return [(3, "glm53")] if worker else []
        with patch.object(setup, "detect_current_lanes", side_effect=detect) as probe, \
             patch.object(setup.subprocess, "call", return_value=0) as call:
            self.assertTrue(setup.park_other_lanes(dialog, 6, self.values, "head.local"))
        self.assertEqual(probe.call_count, 4)
        self.assertEqual(call.call_count, 3)

    def test_lane_chain_prepares_assets_and_parks_before_boot(self):
        for assets_ok, park_ok in [(False, True), (True, False), (True, True)]:
            with self.subTest(assets_ok=assets_ok, park_ok=park_ok):
                dialog = Mock()
                dialog.text.side_effect = lambda prompt, default="": "tester" if "user" in prompt else default
                dialog.confirm.side_effect = lambda prompt, default=True: not prompt.startswith("Enable refusal")
                events = []
                def assets(*args):
                    events.append("assets")
                    return assets_ok
                def park(*args):
                    events.append("park")
                    return park_ok
                def execute(argv):
                    events.append("boot" if "bash start-glm53-flash-tp2.sh" in argv[-1] else "sync")
                    return 0
                with patch.object(setup, "read_lane_env", return_value={}), \
                     patch.object(setup, "pick_host", side_effect=["192.0.2.1", "192.0.2.2", "head.local"]), \
                     patch.object(setup, "remote_preflight", return_value=False), \
                     patch.object(setup, "prepare_assets", side_effect=assets), \
                     patch.object(setup, "park_other_lanes", side_effect=park), \
                     patch.object(setup.subprocess, "call", side_effect=execute), \
                     patch.object(setup, "tests_chain", return_value=0):
                    self.assertEqual(setup.lane_chain(dialog, 6), 0 if assets_ok and park_ok else 1)
                self.assertEqual(events[0], "assets")
                if assets_ok and park_ok:
                    self.assertEqual(events[-2:], ["park", "boot"])
                else:
                    self.assertNotIn("boot", events)

    def test_hub_alias_supports_existing_caches_and_repeated_staging(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as cache:
                name = "models--thinkingmachines--Inkling-Small-NVFP4"
                model = Path(cache, name)
                model.mkdir()
                (model / "config.json").write_text("new config")
                alias = Path(cache, "hub", name)
                if existing:
                    alias.mkdir(parents=True)
                    (alias / "config.json").write_text("old")
                env = setup.lane_env(5, self.values)
                env["HF_CACHE"] = cache
                with patch.object(setup, "lane_env", return_value=env):
                    plan = setup.asset_commands(5, self.values, "head.local")
                command = next(argv[-1] for desc, argv in plan if desc == "expose HF hub cache on head")
                for _ in range(2):
                    result = subprocess.run(["sh", "-c", command], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual((alias / "config.json").read_text(), "new config")

    def test_demo_full_lane_chain_emits_no_network_commands_or_writes(self):
        dialog = Mock()
        dialog.confirm.return_value = True
        dialog.text.side_effect = lambda prompt, default="": default or "tester"
        dialog.menu.return_value = 0
        before = sorted(str(p) for p in Path(self.tmp.name).rglob("*"))
        with patch.object(setup, "DEMO", True), \
             patch.object(setup.subprocess, "run") as run, \
             patch.object(setup.subprocess, "call") as call, \
             patch.object(setup.subprocess, "Popen") as popen, \
             patch.object(setup.urllib.request, "urlopen") as urlopen:
            for idx in range(len(setup.LANES)):
                self.assertEqual(setup.asset_commands(idx, self.values), [])
                self.assertEqual(setup.deploy_commands(idx, self.values), [])
                self.assertEqual(setup.park_commands(idx, [(3, "glm53")], self.values), [])
                self.assertEqual(setup.detect_current_lanes(self.values, "head.local"), [])
                self.assertTrue(setup.park_other_lanes(dialog, idx, self.values, "head.local"))
                self.assertEqual(setup.lane_chain(dialog, idx), 0)
            setup.remote_preflight(dialog, 6, self.values, "head.local")
            setup.remote_diagnose(dialog, "head.local")
            setup.diagnose_chain(dialog, "http://head.local/v1")
        for mock in (run, call, popen, urlopen):
            mock.assert_not_called()
        self.assertEqual(before, sorted(str(p) for p in Path(self.tmp.name).rglob("*")))
        self.assertNotIn("$ ssh", str(dialog.mock_calls))
        self.assertNotIn("hf download", str(dialog.mock_calls))


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
