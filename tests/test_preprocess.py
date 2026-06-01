import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.preprocess import ffmpeg_executable, preprocess_audio


class PreprocessTest(unittest.TestCase):
    @unittest.skipUnless(ffmpeg_executable(), "ffmpeg required for RNNoise preview denoise")
    def test_rnnoise_uses_browser_safe_ffmpeg_denoise_without_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [30, -30, 1200, -1200, 20, -20, 900, -900] * 2000)
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
            self.assertEqual(result.steps[0]["metadata"]["processor"], "ffmpeg_afftdn")
            self.assertEqual(result.steps[0]["metadata"]["output_format"], "wav_pcm_s16le")
            info = _read_wav_info(Path(result.audio_path))
            self.assertEqual(info["sample_width"], 2)
            self.assertGreater(info["frames"], 0)
            self.assertGreater(info["duration_seconds"], 0.9)
            self.assertFalse(any("command" in warning.lower() for warning in result.warnings))

    @unittest.skipUnless(ffmpeg_executable(), "ffmpeg required for BS-RoFormer preview denoise")
    def test_bs_roformer_uses_browser_safe_ffmpeg_denoise_without_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [40, -40, 1400, -1400] * 4000)
            config = ExperimentConfig(
                enable_noise_reduction=True,
                noise_reduction_model="BS-RoFormer",
                noise_reduction_strength=0.5,
                output_dir=str(Path(tmp) / "outputs"),
            )
            result = preprocess_audio(str(source), config)
            self.assertTrue(result.applied)
            self.assertEqual(result.steps[0]["metadata"]["processor"], "ffmpeg_afftdn")
            self.assertEqual(_read_wav_info(Path(result.audio_path))["sample_width"], 2)

    @unittest.skipUnless(ffmpeg_executable(), "ffmpeg required for RNNoise preview denoise")
    def test_noise_and_volume_are_independent_ordered_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [1000, -1000, 1000, -1000] * 4000)
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

    def test_volume_normalization_peak_limits_gain_to_avoid_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, ([100, -100] * 1000) + [30000])
            config = ExperimentConfig(
                enable_volume_normalization=True,
                volume_normalization_strength=1.0,
                volume_target_dbfs=-20,
                output_dir=str(Path(tmp) / "outputs"),
            )
            result = preprocess_audio(str(source), config)

            metadata = result.steps[0]["metadata"]
            self.assertTrue(result.applied)
            self.assertTrue(metadata["peak_limited"])
            self.assertLess(metadata["gain_factor"], metadata["requested_gain_factor"])
            self.assertEqual(metadata["clipped_samples"], 0)
            self.assertLessEqual(_read_wav_peak(Path(result.audio_path)), metadata["peak_limit"])

    @unittest.skipUnless(ffmpeg_executable(), "ffmpeg required for compressed audio conversion")
    def test_volume_normalization_converts_non_wav_without_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            compressed = Path(tmp) / "input.mp3"
            _write_pcm16_wav(source, [1000, -1000, 1000, -1000])
            subprocess.run(
                [
                    ffmpeg_executable() or "ffmpeg",
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
            self.assertIn(".normalized.", result.audio_path)
            self.assertTrue(result.steps[0]["metadata"]["converted_input_path"])
            self.assertFalse(any("command" in warning.lower() for warning in result.warnings))

    @unittest.skipUnless(ffmpeg_executable(), "ffmpeg required for collision-free denoise output")
    def test_noise_reduction_outputs_unique_browser_safe_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.wav"
            _write_pcm16_wav(source, [40, -40, 1400, -1400] * 4000)
            config = ExperimentConfig(
                enable_noise_reduction=True,
                noise_reduction_model="RNNoise",
                noise_reduction_strength=0.5,
                output_dir=str(Path(tmp) / "outputs"),
            )
            first = preprocess_audio(str(source), config)
            second = preprocess_audio(str(source), config)
            self.assertTrue(first.applied)
            self.assertTrue(second.applied)
            self.assertNotEqual(first.audio_path, second.audio_path)
            self.assertEqual(_read_wav_info(Path(first.audio_path))["sample_width"], 2)
            self.assertEqual(_read_wav_info(Path(second.audio_path))["sample_width"], 2)


def _write_pcm16_wav(path: Path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


def _read_wav_info(path: Path):
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "sample_rate": rate,
            "frames": frames,
            "duration_seconds": frames / float(rate) if rate else 0.0,
        }


def _read_wav_peak(path: Path):
    with wave.open(str(path), "rb") as handle:
        frames = handle.readframes(handle.getnframes())
    if not frames:
        return 0
    samples = [int.from_bytes(frames[index : index + 2], "little", signed=True) for index in range(0, len(frames), 2)]
    return max(abs(sample) for sample in samples)


if __name__ == "__main__":
    unittest.main()
