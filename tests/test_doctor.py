import unittest

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.doctor import run_doctor


class DoctorTest(unittest.TestCase):
    def test_mock_config_does_not_require_nvidia(self):
        config = ExperimentConfig(asr_backend="mock", post_backend="mock", rag_embedding_backend="lexical")
        checks = run_doctor(config)
        names = [check.name for check in checks]
        self.assertNotIn("nvidia-smi", names)

    def test_cuda_config_requires_nvidia(self):
        config = ExperimentConfig(asr_backend="vllm_chat", post_backend="vllm_openai")
        checks = run_doctor(config)
        names = [check.name for check in checks]
        self.assertIn("nvidia-smi", names)
        nvidia_check = next(check for check in checks if check.name == "nvidia-smi")
        self.assertIn(nvidia_check.status, {"ok", "fail"})

    def test_auto_start_vllm_checks_ninja_for_flashinfer_jit(self):
        config = ExperimentConfig(
            auto_start_model_servers=True,
            asr_backend="vllm_chat",
            post_backend="vllm_openai",
        )
        checks = run_doctor(config)
        names = [check.name for check in checks]
        self.assertIn("ninja", names)

    def test_preprocess_doctor_checks_ffmpeg_without_command_template(self):
        config = ExperimentConfig(
            asr_backend="mock",
            post_backend="mock",
            enable_noise_reduction=True,
            noise_reduction_model="RNNoise",
        )
        checks = run_doctor(config)
        ffmpeg_check = next(check for check in checks if check.name == "preprocess:ffmpeg")
        self.assertIn(ffmpeg_check.status, {"ok", "warn"})
        self.assertNotIn("command", ffmpeg_check.detail.lower())


if __name__ == "__main__":
    unittest.main()
