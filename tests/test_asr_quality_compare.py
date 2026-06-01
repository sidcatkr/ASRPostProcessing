import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asrpostprocessing.asr_quality_compare import run_asr_quality_compare
from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.preprocess import PreprocessResult
from asrpostprocessing.schemas import TranscriptResult, TranscriptSegment


class _FakeASRAdapter:
    def transcribe(self, audio_path, config, keyword_instruction=""):
        text = f"{config.asr_chunking_strategy} {config.asr_chunk_seconds:g}"
        chunked = config.asr_chunk_seconds < 120
        metadata = {
            "backend": "fake",
            "chunked": chunked,
            "chunking_strategy": config.asr_chunking_strategy,
            "chunk_seconds": config.asr_chunk_seconds,
            "context_chars": config.asr_context_chars,
        }
        segments = []
        if chunked:
            metadata["chunks"] = [
                {
                    "index": 0,
                    "start_s": 0.0,
                    "end_s": config.asr_chunk_seconds,
                    "method": config.asr_chunking_strategy,
                    "previous_context_chars": 0,
                }
            ]
            segments.append(TranscriptSegment(text=text, start_s=0.0, end_s=config.asr_chunk_seconds))
        return TranscriptResult(language="ko", text=text, segments=segments, metadata=metadata)


class ASRQualityCompareTest(unittest.TestCase):
    def test_compare_writes_rows_for_chunk_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.wav"
            audio.write_bytes(b"audio")
            output = Path(tmp) / "compare.json"
            config = ExperimentConfig(output_dir=str(Path(tmp) / "outputs"), asr_backend="mock")
            with patch("asrpostprocessing.asr_quality_compare.preprocess_audio") as preprocess, patch(
                "asrpostprocessing.asr_quality_compare.build_asr_adapter", return_value=_FakeASRAdapter()
            ):
                preprocess.return_value = PreprocessResult(audio_path=str(audio), applied=False)
                result_path = run_asr_quality_compare(
                    str(audio),
                    config,
                    output_path=str(output),
                    chunk_seconds=[30.0, 120.0],
                    strategies=["fixed"],
                    preprocess_mode="none",
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["rows"]), 2)
            self.assertEqual([row["chunk_seconds"] for row in payload["rows"]], [30.0, 120.0])
            self.assertEqual(payload["rows"][0]["condition"], "asr_none_fixed_30s")
            self.assertTrue(any("120s" in item for item in payload["rows"][0]["asr_quality"]["action_items"]))

    def test_compare_accepts_multiple_sample_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "audio.wav"
            audio.write_bytes(b"audio")
            output = Path(tmp) / "compare.json"
            config = ExperimentConfig(output_dir=str(Path(tmp) / "outputs"), asr_backend="mock")
            with patch("asrpostprocessing.asr_quality_compare._sample_audio", return_value=audio) as sample_audio, patch(
                "asrpostprocessing.asr_quality_compare.preprocess_audio"
            ) as preprocess, patch("asrpostprocessing.asr_quality_compare.build_asr_adapter", return_value=_FakeASRAdapter()):
                preprocess.return_value = PreprocessResult(audio_path=str(audio), applied=False)
                result_path = run_asr_quality_compare(
                    str(audio),
                    config,
                    output_path=str(output),
                    chunk_seconds=[120.0],
                    strategies=["fixed"],
                    preprocess_mode="none",
                    sample_seconds=10.0,
                    sample_start_s=[0.0, 30.0],
                )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["sample_starts_s"], [0.0, 30.0])
            self.assertEqual([row["sample_start_s"] for row in payload["rows"]], [0.0, 30.0])
            self.assertEqual(sample_audio.call_count, 2)


if __name__ == "__main__":
    unittest.main()
