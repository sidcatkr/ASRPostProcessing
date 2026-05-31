import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.model_server import _PROCESSES, _default_command, _server_specs, ensure_model_servers, stop_model_servers


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
            self.assertIn("--max-num-seqs", post_command)

    def test_ready_endpoints_are_not_started(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="vllm_openai")
        with patch("asrpostprocessing.model_server._endpoint_ready", return_value=True), patch(
            "asrpostprocessing.model_server._start_process"
        ) as start_process:
            statuses = ensure_model_servers(config)
        self.assertEqual([item.status for item in statuses], ["ready", "ready"])
        start_process.assert_not_called()

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
