import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.preprocess import preprocess_audio


class PreprocessTest(unittest.TestCase):
    def test_rnnoise_uses_builtin_pcm_denoise_without_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [30, -30, 1200, -1200, 20, -20])
            config = ExperimentConfig(
                enable_preprocess=True,
                preprocess_model="rnnoise",
                noise_reduction_strength=0.5,
                output_dir=str(Path(tmp) / "outputs"),
            )
            result = preprocess_audio(str(source), config)
            self.assertTrue(result.applied)
            self.assertTrue(Path(result.audio_path).exists())
            self.assertEqual(result.steps[0]["step"], "noise_reduction")
            self.assertEqual(result.steps[0]["metadata"]["processor"], "pcm_noise_gate")
            self.assertFalse(any("command" in warning.lower() for warning in result.warnings))

    def test_bs_roformer_uses_builtin_pcm_denoise_without_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [40, -40, 1400, -1400])
            config = ExperimentConfig(
                enable_noise_reduction=True,
                noise_reduction_model="BS-RoFormer",
                noise_reduction_strength=0.5,
                output_dir=str(Path(tmp) / "outputs"),
            )
            result = preprocess_audio(str(source), config)
            self.assertTrue(result.applied)
            self.assertEqual(result.steps[0]["metadata"]["processor"], "pcm_noise_gate")

    def test_noise_and_volume_are_independent_ordered_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [1000, -1000, 1000, -1000])
            config = ExperimentConfig(
                enable_noise_reduction=True,
                noise_reduction_model="RNNoise",
                noise_reduction_strength=0.5,
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

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg required for compressed audio conversion")
    def test_volume_normalization_converts_non_wav_without_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            compressed = Path(tmp) / "input.mp3"
            _write_pcm16_wav(source, [1000, -1000, 1000, -1000])
            subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    str(compressed),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            config = ExperimentConfig(
                enable_volume_normalization=True,
                volume_normalization_strength=1.0,
                volume_target_dbfs=-20,
                output_dir=str(Path(tmp) / "outputs"),
            )
            result = preprocess_audio(str(compressed), config)
            self.assertTrue(result.applied)
            self.assertTrue(Path(result.audio_path).exists())
            self.assertIn(".normalized.wav", result.audio_path)
            self.assertTrue(result.steps[0]["metadata"]["converted_input_path"])
            self.assertFalse(any("command" in warning.lower() for warning in result.warnings))


def _write_pcm16_wav(path: Path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


if __name__ == "__main__":
    unittest.main()
