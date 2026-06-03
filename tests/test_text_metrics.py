import unittest

from asrpostprocessing.metrics import evaluate_transcripts
from asrpostprocessing.text import cer, normalize_text, transcript_error_breakdown, wer_eojeol


class TextMetricsTest(unittest.TestCase):
    def test_normalize_mixed_korean_english(self):
        self.assertEqual(normalize_text(" Alpha   Term 로  테스트 "), "alpha term 로 테스트")
        self.assertEqual(normalize_text("A B C", remove_spaces=True), "abc")
        self.assertEqual(normalize_text("A, B? C!", remove_spaces=True, remove_symbols=True), "abc")

    def test_cer_and_wer(self):
        self.assertEqual(cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(cer("abc", "axc"), 1 / 3)
        self.assertAlmostEqual(wer_eojeol("Alpha Term 실행", "Alpha 실행"), 1 / 3)

    def test_metrics_ignore_spacing_linebreak_and_punctuation_only_differences(self):
        reference = "해든 마을 주민센터,\nOmar Lee?"
        hypothesis = "해든마을   주민센터 OmarLee!"
        self.assertGreater(cer(reference, hypothesis, remove_spaces=True), 0.0)
        self.assertEqual(cer(reference, hypothesis, remove_spaces=True, remove_symbols=True), 0.0)
        self.assertEqual(wer_eojeol(reference, hypothesis), 0.0)

    def test_spacing_insensitive_wer_still_penalizes_content_changes(self):
        self.assertGreater(wer_eojeol("해든마을 주민센터", "해든마을 시민센터"), 0.0)

    def test_error_breakdown_uses_metric_normalization(self):
        spacing_only = transcript_error_breakdown("해든 마을 주민센터,\nOmar Lee?", "해든마을 주민센터 OmarLee!")
        self.assertEqual(spacing_only["cer"]["errors"], 0)
        self.assertEqual(spacing_only["wer"]["errors"], 0)

        changed = transcript_error_breakdown("해든마을 주민센터", "해든마을 시민센터")
        self.assertGreater(changed["cer"]["errors"], 0)
        self.assertGreater(changed["wer"]["errors"], 0)
        self.assertIn("segments", changed["cer"])
        self.assertGreater(changed["cer"]["segments"][0]["error_share"], 0)

    def test_evaluate_delta_positive_when_corrected_improves(self):
        metrics = evaluate_transcripts("AlphaTerm으로 테스트 작성", "알파텀으로 예시 작성", "AlphaTerm으로 테스트 작성")
        self.assertGreater(metrics.delta_cer, 0)
        self.assertEqual(metrics.cer_normalized_no_space, 0.0)
        self.assertEqual(metrics.wer_eojeol, 0.0)


if __name__ == "__main__":
    unittest.main()
