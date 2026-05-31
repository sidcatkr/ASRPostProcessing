import unittest

from asrpostprocessing.sweep import _condition_grid, analyze_sweep


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

    def test_notes_conditions_are_generated(self):
        rows = list(_condition_grid([0.0, 0.5], [0.0, 0.5], [0.25], [0.5], [0.5]))
        conditions = {row[0] for row in rows}
        self.assertEqual(
            conditions,
            {
                "A_raw_asr",
                "B1_noise_reduction_raw_asr",
                "B2_volume_normalization_raw_asr",
                "B3_noise_volume_raw_asr",
                "C_llm_only",
                "D_rag_llm",
                "E_keyword_bias_llm",
                "F_keyword_bias_rag_llm",
                "G_search_rag_llm",
            },
        )


if __name__ == "__main__":
    unittest.main()
