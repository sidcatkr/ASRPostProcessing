import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.model_server import (
    _PROCESSES,
    _ServerRuntimeOptions,
    _default_command,
    _gpu_memory_utilization_for_spec,
    _models_payload_contains,
    _models_payload_model_ids,
    _runtime_options_for_spec,
    _server_specs,
    _start_process,
    ensure_model_servers,
    stop_model_servers,
)


class ModelServerTest(unittest.TestCase):
    def test_disabled_auto_start_does_nothing(self):
        config = ExperimentConfig(auto_start_model_servers=False, asr_backend="vllm_chat", post_backend="vllm_openai")
        self.assertEqual(ensure_model_servers(config), [])

    def test_server_specs_for_default_cuda_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(
                auto_start_model_servers=True,
                asr_backend="vllm_chat",
                post_backend="vllm_openai",
                server_log_dir=str(Path(tmp) / "logs"),
            )
            specs = _server_specs(config)
            self.assertEqual([spec.name for spec in specs], ["asr", "post"])
            self.assertEqual(specs[0].port, 18000)
            self.assertEqual(specs[1].port, 18001)
            self.assertIn("Qwen/Qwen3-ASR-1.7B", _default_command(specs[0]))
            asr_command = _default_command(specs[0])
            post_command = _default_command(specs[1])
            self.assertIn("python", Path(asr_command[0]).name)
            self.assertEqual(asr_command[1:3], ["-m", "asrpostprocessing.qwen_asr_serve_compat"])
            self.assertIn("--gpu-memory-utilization", asr_command)
            self.assertIn("--max-model-len", asr_command)
            self.assertIn("65536", asr_command)
            self.assertIn("--attention-backend", asr_command)
            self.assertIn("TRITON_ATTN", asr_command)
            self.assertIn("--enforce-eager", asr_command)
            self.assertIn("vllm", post_command)
            self.assertIn("--dtype", post_command)
            self.assertIn("2048", post_command)
            self.assertIn("--language-model-only", post_command)
            self.assertIn("--quantization", post_command)
            self.assertIn("bitsandbytes", post_command)
            self.assertIn("--enforce-eager", post_command)
            self.assertIn("--attention-backend", post_command)
            self.assertIn("TRITON_ATTN", post_command)
            self.assertIn("--max-num-seqs", post_command)

    def test_custom_command_templates_can_use_active_python_and_vllm_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(
                auto_start_model_servers=True,
                asr_backend="vllm_chat",
                post_backend="vllm_openai",
                server_log_dir=str(Path(tmp) / "logs"),
                asr_server_command="{python} -m asrpostprocessing.qwen_asr_serve_compat {model}",
                post_server_command="{vllm} serve {model}",
            )
            specs = _server_specs(config)

            with patch("asrpostprocessing.model_server._endpoint_ready", return_value=False), patch(
                "asrpostprocessing.model_server._tcp_port_open", return_value=False
            ), patch("asrpostprocessing.model_server._gpu_memory_utilization_for_spec", return_value="0.9"), patch(
                "asrpostprocessing.model_server.subprocess.Popen"
            ) as popen, patch("asrpostprocessing.model_server.shutil.which", return_value=None):
                process = Mock()
                process.pid = 123
                popen.return_value = process
                for spec in specs:
                    _PROCESSES.pop(f"{spec.name}:{spec.base_url}", None)
                    try:
                        _start_process(spec)
                    finally:
                        _PROCESSES.pop(f"{spec.name}:{spec.base_url}", None)

            commands = [call.args[0] for call in popen.call_args_list]
            envs = [call.kwargs["env"] for call in popen.call_args_list]
            self.assertIn("asrpostprocessing.qwen_asr_serve_compat", commands[0])
            self.assertIn(Path(sys.executable).name, commands[0])
            self.assertIn("vllm", commands[1])
            self.assertNotIn("{python}", commands[0])
            self.assertNotIn("{vllm}", commands[1])
            self.assertEqual(envs[0]["PATH"].split(os.pathsep)[0], str(Path(sys.executable).parent))
            self.assertEqual(envs[1]["PATH"].split(os.pathsep)[0], str(Path(sys.executable).parent))

    def test_ready_endpoints_are_not_started(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="vllm_openai")
        with patch("asrpostprocessing.model_server._endpoint_ready", return_value=True), patch(
            "asrpostprocessing.model_server._start_process"
        ) as start_process:
            statuses = ensure_model_servers(config)
        self.assertEqual([item.status for item in statuses], ["ready", "ready"])
        start_process.assert_not_called()

    def test_server_specs_expand_pipeline_lanes(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            asr_backend="vllm_chat",
            post_backend="vllm_openai",
            pipeline_lanes=[
                {
                    "name": "lane_a",
                    "asr_base_url": "http://127.0.0.1:18000/v1",
                    "post_base_url": "http://127.0.0.1:18001/v1",
                    "asr_server_gpu": "0",
                    "post_server_gpu": "1",
                },
                {
                    "name": "lane_b",
                    "asr_base_url": "http://127.0.0.1:18002/v1",
                    "post_base_url": "http://127.0.0.1:18003/v1",
                    "asr_server_gpu": "2",
                    "post_server_gpu": "3",
                },
            ],
        )
        specs = _server_specs(config)
        self.assertEqual([spec.name for spec in specs], ["asr_lane_a", "post_lane_a", "asr_lane_b", "post_lane_b"])
        self.assertEqual([spec.stage for spec in specs], ["asr", "post", "asr", "post"])
        self.assertEqual([spec.gpu for spec in specs], ["0", "1", "2", "3"])
        self.assertEqual(specs[2].port, 18002)

    def test_stage_replicas_create_asr_and_post_specs_on_each_gpu(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            model_residency="stage_replicas",
            asr_backend="vllm_chat",
            post_backend="vllm_openai",
            stage_server_base_urls=[
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                "http://127.0.0.1:18002/v1",
                "http://127.0.0.1:18003/v1",
            ],
            stage_server_gpus=["0", "1", "2", "3"],
        )

        specs = _server_specs(config)

        self.assertEqual([spec.name for spec in specs], [
            "asr_stage_0",
            "post_stage_0",
            "asr_stage_1",
            "post_stage_1",
            "asr_stage_2",
            "post_stage_2",
            "asr_stage_3",
            "post_stage_3",
        ])
        self.assertEqual([spec.stage for spec in specs], ["asr", "post"] * 4)
        self.assertEqual([spec.gpu for spec in specs], ["0", "0", "1", "1", "2", "2", "3", "3"])
        self.assertEqual([spec.port for spec in specs if spec.stage == "asr"], [18000, 18001, 18002, 18003])

    def test_models_payload_must_match_expected_model(self):
        payload = {"data": [{"id": "Qwen/Qwen3-ASR-1.7B"}]}

        self.assertTrue(_models_payload_contains(payload, "Qwen/Qwen3-ASR-1.7B"))
        self.assertFalse(_models_payload_contains(payload, "Qwen/Qwen3.5-9B"))
        self.assertEqual(_models_payload_model_ids(payload), ["Qwen/Qwen3-ASR-1.7B"])

    def test_adaptive_gpu_memory_utilization_uses_free_vram_without_overclaiming(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            asr_backend="vllm_chat",
            post_backend="mock",
            server_gpu_memory_utilization="auto",
            server_gpu_memory_utilization_max=0.9,
            server_gpu_memory_reserved_mb=256,
        )
        spec = _server_specs(config)[0]
        completed = Mock(returncode=0, stdout="24570, 20300\n")
        with patch("asrpostprocessing.model_server.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "asrpostprocessing.model_server.subprocess.run", return_value=completed
        ):
            utilization = float(_gpu_memory_utilization_for_spec(spec))
        self.assertAlmostEqual(utilization, (20300 - 256) / 24570, places=4)

    def test_adaptive_gpu_memory_utilization_uses_cap_when_gpu_is_empty(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            asr_backend="vllm_chat",
            post_backend="mock",
            server_gpu_memory_utilization="auto",
            server_gpu_memory_utilization_max=0.9,
            server_gpu_memory_reserved_mb=256,
        )
        spec = _server_specs(config)[0]
        completed = Mock(returncode=0, stdout="24570, 24500\n")
        with patch("asrpostprocessing.model_server.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "asrpostprocessing.model_server.subprocess.run", return_value=completed
        ):
            self.assertEqual(_gpu_memory_utilization_for_spec(spec), "0.9")

    def test_adaptive_runtime_reduces_template_capacity_for_busy_gpu(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            asr_backend="mock",
            post_backend="vllm_openai",
            post_server_command=(
                "{vllm} serve {model} --gpu-memory-utilization {gpu_memory_utilization} "
                "--max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 16384"
            ),
            server_gpu_memory_utilization="auto",
            server_gpu_memory_utilization_max=0.9,
            server_gpu_memory_reserved_mb=256,
        )
        spec = _server_specs(config)[0]
        completed = Mock(returncode=0, stdout="24570, 20300\n")
        with patch("asrpostprocessing.model_server.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "asrpostprocessing.model_server.subprocess.run", return_value=completed
        ), patch("asrpostprocessing.model_server.subprocess.Popen") as popen:
            runtime = _runtime_options_for_spec(spec)
            process = Mock()
            process.pid = 123
            popen.return_value = process
            _start_process(spec, runtime)

        self.assertAlmostEqual(float(runtime.gpu_memory_utilization), (20300 - 256) / 24570, places=4)
        self.assertLess(runtime.max_model_len, 8192)
        self.assertLess(runtime.max_num_seqs, 8)
        self.assertLess(runtime.max_num_batched_tokens, 16384)
        command = popen.call_args.args[0]
        self.assertIn("--max-model-len 7168", command)
        self.assertIn("--max-num-seqs 7", command)
        self.assertIn("--max-num-batched-tokens 14848", command)

    def test_adaptive_runtime_keeps_full_capacity_for_nearly_empty_gpu(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            asr_backend="mock",
            post_backend="vllm_openai",
            post_server_command=(
                "{vllm} serve {model} --gpu-memory-utilization {gpu_memory_utilization} "
                "--max-model-len 8192 --max-num-seqs 16 --max-num-batched-tokens 32768"
            ),
            server_gpu_memory_utilization="auto",
            server_gpu_memory_utilization_max=0.99,
            server_gpu_memory_reserved_mb=0,
        )
        spec = _server_specs(config)[0]
        completed = Mock(returncode=0, stdout="24570, 24077\n")
        with patch("asrpostprocessing.model_server.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "asrpostprocessing.model_server.subprocess.run", return_value=completed
        ), patch("asrpostprocessing.model_server.subprocess.Popen") as popen:
            runtime = _runtime_options_for_spec(spec)
            process = Mock()
            process.pid = 123
            popen.return_value = process
            _start_process(spec, runtime)

        self.assertAlmostEqual(float(runtime.gpu_memory_utilization), 24077 / 24570, places=4)
        self.assertEqual(runtime.max_model_len, 8192)
        self.assertEqual(runtime.max_num_seqs, 16)
        self.assertEqual(runtime.max_num_batched_tokens, 32768)
        command = popen.call_args.args[0]
        self.assertIn("--max-model-len 8192", command)
        self.assertIn("--max-num-seqs 16", command)
        self.assertIn("--max-num-batched-tokens 32768", command)

    def test_open_non_model_port_fails_fast(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="mock")
        with patch("asrpostprocessing.model_server._endpoint_ready", return_value=False), patch(
            "asrpostprocessing.model_server._tcp_port_open", return_value=True
        ), patch(
            "asrpostprocessing.model_server._fetch_models_payload",
            return_value=(None, "connection accepted but no model endpoint responded"),
        ), patch("asrpostprocessing.model_server._start_process") as start_process:
            with self.assertRaisesRegex(RuntimeError, "already open"):
                ensure_model_servers(config)
        start_process.assert_not_called()

    def test_open_wrong_model_port_reports_served_model(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="mock", post_backend="vllm_openai")
        payload = {"data": [{"id": "Qwen/Qwen3-ASR-1.7B"}]}
        with patch("asrpostprocessing.model_server._endpoint_ready", return_value=False), patch(
            "asrpostprocessing.model_server._tcp_port_open", return_value=True
        ), patch("asrpostprocessing.model_server._fetch_models_payload", return_value=(payload, "")), patch(
            "asrpostprocessing.model_server._start_process"
        ) as start_process:
            with self.assertRaisesRegex(RuntimeError, "Qwen/Qwen3.5-9B.*Qwen/Qwen3-ASR-1.7B"):
                ensure_model_servers(config, names=["post"])
        start_process.assert_not_called()

    def test_can_prepare_only_one_stage_server(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="vllm_openai")
        with patch("asrpostprocessing.model_server._endpoint_ready", return_value=True), patch(
            "asrpostprocessing.model_server._start_process"
        ) as start_process:
            statuses = ensure_model_servers(config, names=["asr"])
        self.assertEqual([item.name for item in statuses], ["asr"])
        start_process.assert_not_called()

    def test_adaptive_start_retries_with_smaller_capacity(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="mock")
        spec = _server_specs(config)[0]
        first_process = Mock()
        first_process.pid = 123
        first_process.poll.return_value = 1
        second_process = Mock()
        second_process.pid = 456
        second_process.poll.return_value = None
        first_runtime = _ServerRuntimeOptions("0.8", 32768, 1, 2048, "first profile")
        second_runtime = _ServerRuntimeOptions("0.6", 24576, 1, 2048, "second profile")
        messages = []
        key = f"{spec.name}:{spec.base_url}"
        try:
            with patch("asrpostprocessing.model_server._endpoint_ready", return_value=False), patch(
                "asrpostprocessing.model_server._tcp_port_open", return_value=False
            ), patch(
                "asrpostprocessing.model_server._runtime_options_for_spec",
                side_effect=[first_runtime, second_runtime],
            ), patch(
                "asrpostprocessing.model_server._start_process",
                side_effect=[first_process, second_process],
            ) as start_process, patch(
                "asrpostprocessing.model_server._wait_until_ready",
                side_effect=[
                    RuntimeError("asr model server exited before becoming ready. Engine core initialization failed"),
                    None,
                ],
            ), patch(
                "asrpostprocessing.model_server._wait_until_port_closed", return_value=True
            ):
                statuses = ensure_model_servers(config, status_callback=messages.append, names=["asr"])
        finally:
            _PROCESSES.pop(key, None)

        self.assertEqual(statuses[0].status, "started")
        self.assertEqual(statuses[0].pid, 456)
        self.assertEqual(start_process.call_args_list[0].args[1], first_runtime)
        self.assertEqual(start_process.call_args_list[1].args[1], second_runtime)
        self.assertTrue(any("retrying on GPU" in message for message in messages))

    def test_stop_model_servers_terminates_managed_process(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="mock")
        spec = _server_specs(config)[0]
        process = Mock()
        process.pid = 123
        process.poll.return_value = None
        process.wait.return_value = 0
        key = f"{spec.name}:{spec.base_url}"
        _PROCESSES[key] = process
        try:
            with patch("asrpostprocessing.model_server.os.getpgid", side_effect=ProcessLookupError), patch(
                "asrpostprocessing.model_server._tcp_port_open", return_value=False
            ):
                statuses = stop_model_servers(config, names=["asr"])
        finally:
            _PROCESSES.pop(key, None)
        self.assertEqual(statuses[0].status, "stopped")
        process.terminate.assert_called_once()

    def test_stop_model_servers_waits_for_port_release(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="mock")
        spec = _server_specs(config)[0]
        process = Mock()
        process.pid = 123
        process.poll.return_value = None
        process.wait.return_value = 0
        key = f"{spec.name}:{spec.base_url}"
        _PROCESSES[key] = process
        messages = []
        try:
            with patch("asrpostprocessing.model_server.os.getpgid", side_effect=ProcessLookupError), patch(
                "asrpostprocessing.model_server._tcp_port_open", side_effect=[True, False]
            ), patch("asrpostprocessing.model_server.time.sleep"):
                statuses = stop_model_servers(config, status_callback=messages.append, names=["asr"])
        finally:
            _PROCESSES.pop(key, None)
        self.assertEqual(statuses[0].status, "stopped")
        self.assertIn("port released", statuses[0].detail)
        self.assertTrue(any("Waiting for asr port" in message for message in messages))

    def test_stage_replicas_reclaims_stale_endpoint_before_starting_expected_stage(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            asr_backend="mock",
            post_backend="vllm_openai",
            model_residency="stage_replicas",
            stage_server_base_urls=["http://127.0.0.1:18000/v1"],
            stage_server_gpus=["0"],
        )
        spec = _server_specs(config)[0]
        process = Mock()
        process.pid = 456
        process.poll.return_value = None
        key = f"{spec.name}:{spec.base_url}"
        try:
            with patch("asrpostprocessing.model_server._endpoint_ready", return_value=False), patch(
                "asrpostprocessing.model_server._tcp_port_open", side_effect=[True, False, False]
            ), patch(
                "asrpostprocessing.model_server._reclaim_unmanaged_stage_endpoint", return_value=True
            ) as reclaim, patch(
                "asrpostprocessing.model_server._start_process", return_value=process
            ) as start_process, patch(
                "asrpostprocessing.model_server._wait_until_ready", return_value=None
            ):
                statuses = ensure_model_servers(config, names=["post_stage_0"])
        finally:
            _PROCESSES.pop(key, None)

        reclaim.assert_called_once()
        start_process.assert_called_once()
        self.assertEqual(statuses[0].name, "post_stage_0")
        self.assertEqual(statuses[0].status, "started")

    def test_stage_replicas_stop_reclaims_unmanaged_server_after_app_restart(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            asr_backend="vllm_chat",
            post_backend="mock",
            model_residency="stage_replicas",
            stage_server_base_urls=["http://127.0.0.1:18000/v1"],
            stage_server_gpus=["0"],
        )
        messages = []
        with patch("asrpostprocessing.model_server._tcp_port_open", side_effect=[True, False]), patch(
            "asrpostprocessing.model_server._safe_stage_server_pids", return_value=[123]
        ), patch("asrpostprocessing.model_server._signal_pid_group") as signal_pid:
            statuses = stop_model_servers(config, status_callback=messages.append, names=["asr_stage_0"])

        self.assertEqual(statuses[0].status, "stopped_unmanaged")
        signal_pid.assert_called_once_with(123, signal.SIGTERM)
        self.assertTrue(any("Stopping unmanaged asr_stage_0" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
