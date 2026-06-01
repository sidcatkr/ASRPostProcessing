import unittest

from asrpostprocessing.asr_quality import build_asr_quality_report, build_correction_quality_report
from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.schemas import CorrectionResult, Edit, TranscriptResult, TranscriptSegment


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
        raw = TranscriptResult(language="ko", text="오늘은 모표 용어를 설명합니다.")
        config = ExperimentConfig(keywords=["목표 용어"])

        report = build_asr_quality_report(raw, {"applied": False, "audio_path": "input.wav"}, config)

        self.assertEqual(report["keyword_near_misses"][0]["before"], "모표 용어를")
        self.assertEqual(report["keyword_near_misses"][0]["after"], "목표 용어를")
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

    def test_report_flags_raw_text_artifacts_without_metadata(self):
        raw = TranscriptResult(language="ko", text="앞 문장 language None<asr_text> 如果测试新闻，然后示例文本。 다음 문장")

        report = build_asr_quality_report(raw, {"applied": False, "audio_path": "input.wav"}, ExperimentConfig())

        self.assertEqual(report["text_artifacts"]["asr_text_tag_count"], 1)
        self.assertEqual(report["text_artifacts"]["language_label_count"], 1)
        self.assertTrue(report["text_artifacts"]["non_korean_cjk_drift_candidate"])
        self.assertTrue(any("artifact marker" in warning for warning in report["warnings"]))

    def test_report_flags_near_duplicate_phrase_variants_without_keywords(self):
        raw = TranscriptResult(language="ko", text="오늘은 모표 용어를 설명합니다. 다음에는 목표 용어를 설명합니다.")

        report = build_asr_quality_report(raw, {"applied": False, "audio_path": "input.wav"}, ExperimentConfig())

        self.assertGreaterEqual(len(report["phrase_instability"]), 1)
        phrases = {item["text"] for item in report["phrase_instability"][0]["phrases"]}
        self.assertTrue(any(phrase.startswith("모표 용어를") for phrase in phrases))
        self.assertTrue(any(phrase.startswith("목표 용어를") for phrase in phrases))
        self.assertFalse(any("near-duplicate phrase" in warning for warning in report["warnings"]))

    def test_report_ignores_common_suffix_phrase_variants(self):
        raw = TranscriptResult(language="ko", text="오늘은 우리가 이제 시작합니다. 내일은 우리는 이제 정리합니다.")

        report = build_asr_quality_report(raw, {"applied": False, "audio_path": "input.wav"}, ExperimentConfig())

        self.assertEqual(report["phrase_instability"], [])

    def test_correction_quality_counts_resolved_keyword_near_misses(self):
        raw = TranscriptResult(language="ko", text="오늘은 모표 용어를 설명합니다.")
        correction = CorrectionResult(
            corrected_text="오늘은 목표 용어를 설명합니다.",
            edits=[
                Edit(
                    before="모표 용어를",
                    after="목표 용어를",
                    reason="Keyword-guided ASR near-miss correction.",
                    confidence=0.82,
                )
            ],
            risk="low",
        )
        config = ExperimentConfig(keywords=["목표 용어"])

        report = build_correction_quality_report(raw, correction, config)

        self.assertEqual(report["keyword_near_misses"]["raw_count"], 1)
        self.assertEqual(report["keyword_near_misses"]["corrected_count"], 0)
        self.assertEqual(report["keyword_near_misses"]["resolved_count"], 1)
        self.assertEqual(report["edits"]["keyword_near_miss_count"], 1)
        self.assertTrue(any("fewer keyword" in item for item in report["improvements"]))

    def test_correction_quality_flags_artifacts_and_fallbacks(self):
        raw = TranscriptResult(language="ko", text="원문")
        correction = CorrectionResult(
            corrected_text="정리된 문장 language None<asr_text> 測試文本內容",
            risk="high",
            metadata={
                "chunks": [
                    {
                        "risk": "high",
                        "metadata": {
                            "fallback": "raw_transcript_after_postprocess_error",
                            "post_backend": "mock",
                            "postprocess_error": "post backend timeout",
                        },
                    }
                ]
            },
        )

        report = build_correction_quality_report(raw, correction, ExperimentConfig())

        self.assertEqual(report["postprocess"]["fallback_chunk_count"], 1)
        self.assertEqual(report["postprocess"]["postprocess_error_count"], 1)
        self.assertTrue(report["artifacts"]["corrected"]["has_asr_artifact_markers"])
        self.assertTrue(report["artifacts"]["corrected"]["non_korean_cjk_drift_candidate"])
        self.assertTrue(any("fallback" in warning for warning in report["warnings"]))
        self.assertTrue(any("artifact" in warning for warning in report["warnings"]))

    def test_correction_quality_counts_resolved_phrase_instability(self):
        raw = TranscriptResult(language="ko", text="오늘은 모표 용어를 설명합니다. 다음에는 목표 용어를 설명합니다.")
        correction = CorrectionResult(corrected_text="오늘은 목표 용어를 설명합니다. 다음에는 목표 용어를 설명합니다.")

        report = build_correction_quality_report(raw, correction, ExperimentConfig())

        self.assertGreaterEqual(report["phrase_instability"]["raw_count"], 1)
        self.assertEqual(report["phrase_instability"]["corrected_count"], 0)
        self.assertTrue(any("near-duplicate phrase" in item for item in report["improvements"]))


if __name__ == "__main__":
    unittest.main()
