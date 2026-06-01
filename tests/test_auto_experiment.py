import tempfile
import unittest
import csv
from pathlib import Path

from asrpostprocessing.auto_experiment import run_auto_experiment
from asrpostprocessing.config import ExperimentConfig, load_config
from asrpostprocessing.experiment_matrix import generate_auto_conditions


class AutoExperimentTest(unittest.TestCase):
    def test_full_valid_matrix_generates_40_conditions(self):
        conditions = generate_auto_conditions(
            include_keyword_bias=True,
            include_noise_reduction=True,
            include_volume_normalization=True,
            include_llm_postprocess=True,
            include_rag=True,
            include_search=True,
            mode="full_valid",
        )
        self.assertEqual(len(conditions), 40)
        self.assertEqual(len({condition.condition_id for condition in conditions}), 40)
        self.assertTrue(any(condition.condition_id == "baseline" for condition in conditions))
        self.assertTrue(any(condition.condition_id == "keyword__noise__volume__llm__rag__search" for condition in conditions))
        self.assertFalse(any((condition.enable_rag or condition.enable_search) and not condition.enable_llm_postprocess for condition in conditions))
        self.assertEqual(len({condition.asr_group_key for condition in conditions}), 8)

    def test_core_ablation_is_smaller_valid_subset(self):
        conditions = generate_auto_conditions(mode="core_ablation")
        self.assertLess(len(conditions), 40)
        self.assertTrue(any(condition.condition_id == "llm__rag__search" for condition in conditions))

    def test_full_strength_sweep_expands_strength_axes_and_cache_groups(self):
        conditions = generate_auto_conditions(
            include_keyword_bias=True,
            include_noise_reduction=True,
            include_volume_normalization=True,
            include_llm_postprocess=True,
            include_rag=True,
            include_search=True,
            mode="full_strength_sweep",
        )

        self.assertGreater(len(conditions), 40)
        self.assertTrue(any("__kw0p25" in condition.condition_id for condition in conditions))
        full_condition = next(
            condition
            for condition in conditions
            if condition.enable_keyword_bias
            and condition.enable_noise_reduction
            and condition.enable_volume_normalization
            and condition.enable_llm_postprocess
            and condition.enable_rag
            and condition.enable_search
        )
        self.assertIsNotNone(full_condition.keyword_bias_weight)
        self.assertIsNotNone(full_condition.noise_reduction_strength)
        self.assertIsNotNone(full_condition.volume_normalization_strength)
        self.assertIsNotNone(full_condition.postprocess_strength)
        self.assertIsNotNone(full_condition.rag_strength)
        self.assertIsNotNone(full_condition.rag_top_k)
        self.assertIsNotNone(full_condition.search_strength)
        self.assertIn("kw=", full_condition.asr_group_key)
        self.assertGreater(len({condition.asr_group_key for condition in conditions}), 8)

    def test_full_strength_sweep_accepts_custom_grids(self):
        conditions = generate_auto_conditions(
            include_keyword_bias=True,
            include_noise_reduction=False,
            include_volume_normalization=False,
            include_llm_postprocess=False,
            include_rag=False,
            include_search=False,
            mode="full_strength_sweep",
            keyword_strengths=[0.4, 0.8],
        )

        self.assertEqual(len(conditions), 3)
        self.assertEqual(
            {condition.keyword_bias_weight for condition in conditions if condition.enable_keyword_bias},
            {0.4, 0.8},
        )

    def test_full_strength_sweep_accepts_custom_rag_top_k_grid(self):
        conditions = generate_auto_conditions(
            include_keyword_bias=True,
            include_noise_reduction=False,
            include_volume_normalization=False,
            include_llm_postprocess=True,
            include_rag=True,
            include_search=False,
            mode="full_strength_sweep",
            keyword_strengths=[0.4],
            postprocess_strengths=[0.5],
            rag_strengths=[0.25],
            rag_top_ks=[3, 7],
        )

        rag_conditions = [condition for condition in conditions if condition.enable_rag]
        self.assertEqual({condition.rag_top_k for condition in rag_conditions}, {3, 7})
        self.assertTrue(any("__topk7" in condition.condition_id for condition in rag_conditions))

    def test_l4x4_config_loads_pipeline_lanes(self):
        config = load_config("configs/l4x4.yaml")
        self.assertEqual(config.model_residency, "parallel")
        self.assertEqual(config.postprocess_parallelism, 8)
        self.assertEqual(config.auto_experiment_parallelism, 8)
        self.assertTrue(config.auto_experiment_saturate_lanes)
        self.assertEqual(len(config.pipeline_lanes), 2)
        self.assertEqual(config.pipeline_lanes[1]["asr_base_url"], "http://127.0.0.1:18002/v1")
        self.assertTrue(config.asr_cache_enabled)

    def test_auto_experiment_runs_mock_core_with_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                enable_keyword_bias=True,
                enable_noise_reduction=False,
                enable_volume_normalization=False,
                enable_llm_postprocess=True,
                enable_rag=False,
                enable_search=False,
                auto_experiment_parallelism=2,
                asr_cache_enabled=True,
                preprocess_cache_enabled=True,
            )
            report = run_auto_experiment(
                str(audio),
                config,
                reference_text="테스트 전사 문장입니다.",
                mode="core_ablation",
            )
            self.assertTrue(Path(report["summary_csv"]).exists())
            self.assertTrue(Path(report["analysis_json"]).exists())
            self.assertGreaterEqual(report["condition_count"], 3)
            self.assertEqual(report["analysis"]["num_failed_rows"], 0)
            self.assertIn("effect_summary", report["analysis"])
            with Path(report["summary_csv"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertIn("delta_cer_vs_baseline", rows[0])
            self.assertIn("asr_latency_ms", rows[0])
            self.assertIn("vllm_total_tokens", rows[0])
            self.assertIn("preprocess_cache_hit", rows[0])

    def test_auto_experiment_can_expand_model_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                enable_keyword_bias=False,
                enable_noise_reduction=False,
                enable_volume_normalization=False,
                enable_llm_postprocess=True,
                enable_rag=False,
                enable_search=False,
                auto_experiment_include_models=True,
                auto_experiment_asr_models=["asr-a", "asr-b"],
                auto_experiment_post_models=["post-a", "post-b"],
                auto_experiment_parallelism=4,
                asr_cache_enabled=True,
                preprocess_cache_enabled=True,
            )
            report = run_auto_experiment(
                str(audio),
                config,
                reference_text="테스트 전사 문장입니다.",
                mode="full_valid",
            )
            self.assertEqual(report["condition_count"], 2)
            self.assertEqual(report["case_count"], 6)
            self.assertEqual(report["analysis"]["num_failed_rows"], 0)
            models = {(row["asr_model"], row["post_model"]) for row in report["rows"] if row["llm_postprocess_enabled"]}
            self.assertEqual(models, {("asr-a", "post-a"), ("asr-a", "post-b"), ("asr-b", "post-a"), ("asr-b", "post-b")})

    def test_auto_experiment_strength_case_applies_override_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                enable_keyword_bias=True,
                enable_noise_reduction=False,
                enable_volume_normalization=False,
                enable_llm_postprocess=False,
                auto_experiment_parallelism=2,
                auto_experiment_keyword_weights=[0.4, 0.8],
                asr_cache_enabled=True,
                preprocess_cache_enabled=True,
            )

            report = run_auto_experiment(
                str(audio),
                config,
                reference_text="테스트 전사 문장입니다.",
                mode="full_strength_sweep",
            )

            keyword_rows = [row for row in report["rows"] if row["keyword_bias_enabled"]]
            self.assertEqual({float(row["keyword_bias_weight"]) for row in keyword_rows}, {0.4, 0.8})
            baseline = next(row for row in report["rows"] if row["condition_id"] == "baseline")
            self.assertEqual(float(baseline["keyword_bias_weight"]), 0.0)

    def test_auto_experiment_strength_case_applies_rag_top_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                enable_keyword_bias=False,
                enable_noise_reduction=False,
                enable_volume_normalization=False,
                enable_llm_postprocess=True,
                enable_rag=True,
                enable_search=False,
                auto_experiment_parallelism=2,
                auto_experiment_postprocess_strengths=[0.5],
                auto_experiment_rag_strengths=[0.25],
                auto_experiment_rag_top_ks=[3, 7],
                asr_cache_enabled=True,
                preprocess_cache_enabled=True,
            )

            report = run_auto_experiment(
                str(audio),
                config,
                reference_text="테스트 전사 문장입니다.",
                mode="full_strength_sweep",
            )

            rag_rows = [row for row in report["rows"] if row["rag_enabled"]]
            self.assertEqual({int(row["rag_top_k"]) for row in rag_rows}, {3, 7})
            with Path(report["summary_csv"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("rag_top_k", rows[0])


if __name__ == "__main__":
    unittest.main()
