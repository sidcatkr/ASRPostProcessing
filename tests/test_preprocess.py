import tempfile
import unittest
from pathlib import Path

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.preprocess import preprocess_audio


class PreprocessTest(unittest.TestCase):
    def test_external_rnnoise_command_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            source.write_bytes(b"fake wav for command adapter")
            config = ExperimentConfig(
                enable_preprocess=True,
                preprocess_model="rnnoise",
                rnnoise_command="/bin/cp {input} {output}",
                output_dir=str(Path(tmp) / "outputs"),
            )
            result = preprocess_audio(str(source), config)
            self.assertTrue(result.applied)
            self.assertTrue(Path(result.audio_path).exists())

    def test_missing_external_command_falls_back_with_warning(self):
        config = ExperimentConfig(enable_preprocess=True, preprocess_model="BS-RoFormer")
        result = preprocess_audio("input.wav", config)
        self.assertFalse(result.applied)
        self.assertTrue(result.warnings)


if __name__ == "__main__":
    unittest.main()
