import unittest

from asrpostprocessing.metrics import evaluate_transcripts
from asrpostprocessing.text import cer, normalize_text, wer_eojeol


class TextMetricsTest(unittest.TestCase):
    def test_normalize_mixed_korean_english(self):
        self.assertEqual(normalize_text(" Claude   Code 로  포물 "), "claude code 로 포물")
        self.assertEqual(normalize_text("A B C", remove_spaces=True), "abc")

    def test_cer_and_wer(self):
        self.assertEqual(cer("abc", "abc"), 0.0)
        self.assertAlmostEqual(cer("abc", "axc"), 1 / 3)
        self.assertAlmostEqual(wer_eojeol("Claude Code 실행", "Claude 실행"), 1 / 3)

    def test_evaluate_delta_positive_when_corrected_improves(self):
        metrics = evaluate_transcripts("Claude Code로 for문 작성", "클러드 코드로 포물 작성", "Claude Code로 for문 작성")
        self.assertGreater(metrics.delta_cer, 0)
        self.assertEqual(metrics.cer_normalized_no_space, 0.0)


if __name__ == "__main__":
    unittest.main()
