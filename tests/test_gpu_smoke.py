import os
import unittest

from asrpostprocessing.config import load_config
from asrpostprocessing.pipeline import PipelineRunner

try:
    import pytest  # type: ignore

    pytestmark = pytest.mark.gpu
except Exception:
    pytestmark = ()


@unittest.skipUnless(os.environ.get("ASRPP_GPU_SMOKE_AUDIO"), "set ASRPP_GPU_SMOKE_AUDIO to run GPU smoke test")
class GPUSmokeTest(unittest.TestCase):
    def test_cuda_server_end_to_end(self):
        config = load_config(os.environ.get("ASRPP_GPU_SMOKE_CONFIG", "configs/cuda.yaml"))
        output = PipelineRunner(config).run(
            audio_path=os.environ["ASRPP_GPU_SMOKE_AUDIO"],
            reference_text=os.environ.get("ASRPP_GPU_SMOKE_REFERENCE"),
            run_id="gpu-smoke",
        )
        self.assertTrue(output.raw.text.strip())
        self.assertTrue(output.correction.corrected_text.strip())


if __name__ == "__main__":
    unittest.main()
