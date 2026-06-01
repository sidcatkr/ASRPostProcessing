import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from asrpostprocessing.sweep import _condition_grid, analyze_sweep, shard_manifest


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

    def test_worse_than_raw_is_separate_from_keyword_overbias(self):
        rows = [
            {
                "audio": "a.wav",
                "condition": "A_raw_asr",
                "cer_normalized_no_space": 0.1,
                "semantic_similarity": 1.0,
                "latency_ms": 1.0,
            },
            {
                "audio": "a.wav",
                "condition": "B1_noise_reduction_raw_asr",
                "noise_reduction_strength": 0.5,
                "cer_normalized_no_space": 0.2,
                "semantic_similarity": 1.0,
                "latency_ms": 1.0,
            },
        ]
        analysis = analyze_sweep(rows)
        self.assertEqual(analysis["worse_than_raw_cases"][0]["worse_than_raw_reason"], "cer_above_raw_asr")
        self.assertEqual(analysis["over_preprocess_cases"][0]["over_preprocess_reason"], "worse_than_raw_asr")
        self.assertEqual(analysis["over_bias_cases"], [])

    def test_manifest_shard_writes_round_robin_shards(self):
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.csv"
            manifest.write_text(
                "audio,reference_text,subset\n"
                "a.wav,a,clean\n"
                "b.wav,b,noisy\n"
                "c.wav,c,technical\n",
                encoding="utf-8",
            )
            paths = shard_manifest(str(manifest), 2, str(Path(tmpdir) / "shards"))
            self.assertEqual([path.name for path in paths], ["shard_0.csv", "shard_1.csv"])
            self.assertIn("a.wav", paths[0].read_text(encoding="utf-8"))
            self.assertIn("c.wav", paths[0].read_text(encoding="utf-8"))
            self.assertIn("b.wav", paths[1].read_text(encoding="utf-8"))

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
