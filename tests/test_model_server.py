import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.model_server import _default_command, _server_specs, ensure_model_servers


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
            self.assertEqual(specs[0].port, 8000)
            self.assertEqual(specs[1].port, 8001)
            self.assertIn("Qwen/Qwen3-ASR-1.7B", _default_command(specs[0]))
            asr_command = _default_command(specs[0])
            post_command = _default_command(specs[1])
            self.assertIn("--dtype", asr_command)
            self.assertIn("float16", asr_command)
            self.assertIn("--language-model-only", post_command)
            self.assertIn("8192", post_command)

    def test_ready_endpoints_are_not_started(self):
        config = ExperimentConfig(auto_start_model_servers=True, asr_backend="vllm_chat", post_backend="vllm_openai")
        with patch("asrpostprocessing.model_server._endpoint_ready", return_value=True), patch(
            "asrpostprocessing.model_server._start_process"
        ) as start_process:
            statuses = ensure_model_servers(config)
        self.assertEqual([item.status for item in statuses], ["ready", "ready"])
        start_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
