import subprocess
import unittest
from unittest.mock import patch

from asrpostprocessing.gpu_status import query_gpu_status


class GPUStatusTest(unittest.TestCase):
    def test_missing_nvidia_smi_is_reported(self):
        with patch("asrpostprocessing.gpu_status.shutil.which", return_value=None):
            status = query_gpu_status()
        self.assertFalse(status["available"])
        self.assertEqual(status["gpus"], [])
        self.assertIn("nvidia-smi", status["error"])

    def test_gpu_and_process_output_are_parsed(self):
        def fake_run(command, capture_output, text, timeout):
            query = command[1]
            if query.startswith("--query-gpu"):
                return subprocess.CompletedProcess(command, 0, "0, Quadro RTX 5000, 16384, 2048, 14336, 12, 55, 97.5, 230.0, P2\n", "")
            return subprocess.CompletedProcess(command, 0, "1234, python, 2048\n", "")

        with patch("asrpostprocessing.gpu_status.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "asrpostprocessing.gpu_status.subprocess.run", side_effect=fake_run
        ):
            status = query_gpu_status()
        self.assertTrue(status["available"])
        self.assertEqual(status["gpus"][0]["memory_used_mb"], 2048)
        self.assertEqual(status["gpus"][0]["memory_used_percent"], 12.5)
        self.assertEqual(status["gpus"][0]["power_draw_w"], 97.5)
        self.assertEqual(status["gpus"][0]["performance_state"], "P2")
        self.assertEqual(status["processes"][0]["pid"], 1234)
        self.assertEqual(status["processes"][0]["used_memory_mb"], 2048)

    def test_gpu_query_falls_back_when_power_fields_are_unavailable(self):
        def fake_run(command, capture_output, text, timeout):
            query = command[1]
            if query.startswith("--query-gpu") and "power.draw" in query:
                return subprocess.CompletedProcess(command, 1, "", "unsupported field")
            if query.startswith("--query-gpu"):
                return subprocess.CompletedProcess(command, 0, "0, Quadro RTX 5000, 16384, 2048, 14336, 12, 55\n", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        with patch("asrpostprocessing.gpu_status.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "asrpostprocessing.gpu_status.subprocess.run", side_effect=fake_run
        ):
            status = query_gpu_status()
        self.assertTrue(status["available"])
        self.assertEqual(status["gpus"][0]["gpu_utilization_percent"], 12)
        self.assertIsNone(status["gpus"][0]["power_draw_w"])
        self.assertIn("extended GPU query failed", status["warnings"][0])


if __name__ == "__main__":
    unittest.main()
