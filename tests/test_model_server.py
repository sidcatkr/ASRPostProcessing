import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.model_server import (
    _PROCESSES,
    _default_command,
    _gpu_memory_utilization_for_spec,
    _server_specs,
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
            self.assertIn("32768", asr_command)
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

    def test_open_non_model_port_fails_fast(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="mock")
        with patch("asrpostprocessing.model_server._endpoint_ready", return_value=False), patch(
            "asrpostprocessing.model_server._tcp_port_open", return_value=True
        ), patch("asrpostprocessing.model_server._start_process") as start_process:
            with self.assertRaisesRegex(RuntimeError, "already open"):
                ensure_model_servers(config)
        start_process.assert_not_called()

    def test_can_prepare_only_one_stage_server(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="vllm_openai")
        with patch("asrpostprocessing.model_server._endpoint_ready", return_value=True), patch(
            "asrpostprocessing.model_server._start_process"
        ) as start_process:
            statuses = ensure_model_servers(config, names=["asr"])
        self.assertEqual([item.name for item in statuses], ["asr"])
        start_process.assert_not_called()

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
            statuses = stop_model_servers(config, names=["asr"])
        finally:
            _PROCESSES.pop(key, None)
        self.assertEqual(statuses[0].status, "stopped")
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
