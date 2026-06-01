import tempfile
import unittest
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

    def test_l4x4_config_loads_pipeline_lanes(self):
        config = load_config("configs/l4x4.yaml")
        self.assertEqual(config.model_residency, "parallel")
        self.assertEqual(config.postprocess_parallelism, 8)
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


if __name__ == "__main__":
    unittest.main()
