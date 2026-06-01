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

        self.assertTrue(any("empty transcript" in warning for warning in report["warnings"]))
        self.assertTrue(any("empty text" in warning for warning in report["warnings"]))
        self.assertTrue(any("language drift" in action for action in report["action_items"]))

    def test_report_flags_keyword_near_miss_terms(self):
        raw = TranscriptResult(language="ko", text="여러분이 이제 서면 연구를 찾아보고 읽어볼 거니까")
        config = ExperimentConfig(keywords=["선행 연구"])

        report = build_asr_quality_report(raw, {"applied": False, "audio_path": "input.wav"}, config)

        self.assertEqual(report["keyword_near_misses"][0]["before"], "서면 연구를")
        self.assertEqual(report["keyword_near_misses"][0]["after"], "선행 연구를")
        self.assertTrue(any("keyword near-miss" in warning for warning in report["warnings"]))

    def test_report_surfaces_filtered_language_drift(self):
        raw = TranscriptResult(
            language="ko",
            text="쉬다 와요. 다음 곡 잡자.",
            segments=[
                TranscriptSegment(
                    text="쉬다 와요. 다음 곡 잡자.",
                    start_s=3300.0,
                    end_s=3360.0,
                    metadata={"asr_metadata": {"parsed": {"filtered_reason": "inline_cjk_drift_removed"}}},
                )
            ],
            metadata={"backend": "vllm_chat", "chunked": True, "chunk_seconds": 120.0},
        )

        report = build_asr_quality_report(raw, {"applied": False, "audio_path": "input.wav"}, ExperimentConfig())

        self.assertEqual(report["language_drift"]["filtered_reasons"], ["inline_cjk_drift_removed"])
        self.assertEqual(report["chunks"][0]["filtered_reason"], "inline_cjk_drift_removed")
        self.assertTrue(any("language drift" in warning for warning in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
