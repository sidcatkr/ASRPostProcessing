import tempfile
import unittest
import wave
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
            self.assertEqual(result.steps[0]["step"], "noise_reduction")

    def test_missing_external_command_falls_back_with_warning(self):
        config = ExperimentConfig(enable_noise_reduction=True, noise_reduction_model="BS-RoFormer")
        result = preprocess_audio("input.wav", config)
        self.assertFalse(result.applied)
        self.assertTrue(result.warnings)

    def test_noise_and_volume_are_independent_ordered_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [1000, -1000, 1000, -1000])
            config = ExperimentConfig(
                enable_noise_reduction=True,
                noise_reduction_model="RNNoise",
                noise_reduction_strength=0.5,
                rnnoise_command="/bin/cp {input} {output}",
                enable_volume_normalization=True,
                volume_normalization_strength=1.0,
                volume_target_dbfs=-12,
                output_dir=str(Path(tmp) / "outputs"),
            )
            result = preprocess_audio(str(source), config)
            self.assertTrue(result.applied)
            self.assertEqual([step["step"] for step in result.steps], ["noise_reduction", "volume_normalization"])
            self.assertEqual(result.steps[1]["metadata"]["target_dbfs"], -12.0)
            self.assertIn("gain_factor", result.steps[1]["metadata"])

    def test_volume_normalization_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [1000, -1000, 1000, -1000])
            config = ExperimentConfig(
                enable_volume_normalization=True,
                volume_normalization_strength=1.0,
                volume_target_dbfs=-20,
                output_dir=str(Path(tmp) / "outputs"),
            )
            result = preprocess_audio(str(source), config)
            self.assertTrue(result.applied)
            self.assertTrue(Path(result.audio_path).exists())
            self.assertEqual(result.steps[0]["step"], "volume_normalization")
            self.assertGreater(result.steps[0]["metadata"]["target_rms"], 0)


def _write_pcm16_wav(path: Path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


if __name__ == "__main__":
    unittest.main()
