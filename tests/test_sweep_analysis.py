import unittest

from asrpostprocessing.sweep import analyze_sweep


class SweepAnalysisTest(unittest.TestCase):
    def test_zero_cer_is_best_not_falsy_default(self):
        rows = [
            {
                "audio": "a.wav",
                "condition": "A_raw_asr",
                "cer_normalized_no_space": 0.7,
                "semantic_similarity": 1.0,
                "latency_ms": 1.0,
            },
            {
                "audio": "a.wav",
                "condition": "C_llm_only",
                "cer_normalized_no_space": 0.0,
                "semantic_similarity": 0.5,
                "latency_ms": 2.0,
            },
        ]
        analysis = analyze_sweep(rows)
        self.assertEqual(analysis["best_by_cer"]["condition"], "C_llm_only")

    def test_over_bias_compares_zero_weight(self):
        rows = [
            {
                "audio": "a.wav",
                "condition": "E_keyword_bias_llm",
                "keyword_bias_weight": 0.0,
                "rag_strength": 0.0,
                "postprocess_strength": 0.5,
                "cer_normalized_no_space": 0.1,
                "semantic_similarity": 0.9,
                "latency_ms": 1.0,
            },
            {
                "audio": "a.wav",
                "condition": "E_keyword_bias_llm",
                "keyword_bias_weight": 1.0,
                "rag_strength": 0.0,
                "postprocess_strength": 0.5,
                "cer_normalized_no_space": 0.2,
                "semantic_similarity": 0.9,
                "latency_ms": 1.0,
            },
        ]
        analysis = analyze_sweep(rows)
        self.assertEqual(analysis["over_bias_cases"][0]["over_bias_reason"], "worse_than_zero_weight")


if __name__ == "__main__":
    unittest.main()
