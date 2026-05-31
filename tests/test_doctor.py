import unittest

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.doctor import has_failures, run_doctor


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
        self.assertTrue(has_failures(checks))


if __name__ == "__main__":
    unittest.main()
