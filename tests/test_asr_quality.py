import unittest

from asrpostprocessing.asr_quality import build_asr_quality_report
from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.schemas import TranscriptResult, TranscriptSegment


class ASRQualityReportTest(unittest.TestCase):
    def test_report_flags_clipping_and_short_chunks(self):
        raw = TranscriptResult(
            language="ko",
            text="first chunk\nsecond chunk",
            segments=[
                TranscriptSegment(text="first chunk", start_s=0.0, end_s=30.0, metadata={"previous_context_chars": 0}),
                TranscriptSegment(text="second chunk", start_s=30.0, end_s=60.0, metadata={"previous_context_chars": 11}),
            ],
            metadata={
                "backend": "vllm_chat",
                "chunked": True,
                "chunking_strategy": "fixed",
                "chunk_seconds": 30.0,
                "context_chars": 240,
                "chunks": [
                    {"index": 0, "start_s": 0.0, "end_s": 30.0, "method": "fixed", "previous_context_chars": 0},
                    {"index": 1, "start_s": 30.0, "end_s": 60.0, "method": "fixed", "previous_context_chars": 11},
                ],
            },
        )
        preprocess = {
            "applied": True,
            "audio_path": "preprocessed.wav",
            "steps": [
                {
                    "step": "volume_normalization",
                    "applied": True,
                    "metadata": {"clipped_samples": 12, "gain_factor": 2.0, "duration_seconds": 60.0},
                }
            ],
        }

        report = build_asr_quality_report(raw, preprocess, ExperimentConfig(asr_chunk_seconds=30.0))

        self.assertEqual(report["chunking"]["chunk_count"], 2)
        self.assertEqual(report["chunks"][1]["previous_context_chars"], 11)
        self.assertTrue(any("clipped" in warning for warning in report["warnings"]))
        self.assertTrue(any("short" in warning for warning in report["warnings"]))
        self.assertTrue(any("120s" in action for action in report["action_items"]))

    def test_report_keeps_single_chunk_preview(self):
        raw = TranscriptResult(language="ko", text="single transcript text", metadata={"backend": "vllm_chat"})

        report = build_asr_quality_report(raw, {"applied": False, "audio_path": "input.wav"}, ExperimentConfig())

        self.assertEqual(report["chunking"]["chunk_count"], 1)
        self.assertEqual(report["chunks"][0]["method"], "single")
        self.assertIn("Reference-free", report["note"])

    def test_report_recommends_inspection_for_empty_chunks(self):
        raw = TranscriptResult(
            language="ko",
            text="",
            segments=[TranscriptSegment(text="", start_s=0.0, end_s=30.0)],
            metadata={"backend": "vllm_chat", "chunked": True, "chunk_seconds": 120.0},
        )

        report = build_asr_quality_report(raw, {"applied": False, "audio_path": "input.wav"}, ExperimentConfig())

        self.assertTrue(any("empty text" in warning for warning in report["warnings"]))
        self.assertTrue(any("language drift" in action for action in report["action_items"]))


if __name__ == "__main__":
    unittest.main()
