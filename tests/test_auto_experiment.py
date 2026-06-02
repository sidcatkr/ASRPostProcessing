import tempfile
import threading
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

from asrpostprocessing.auto_experiment import run_auto_experiment
from asrpostprocessing.config import ExperimentConfig, load_config
from asrpostprocessing.experiment_matrix import generate_auto_conditions
from asrpostprocessing.schemas import SearchResult


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

    def test_auto_experiment_model_axes_apply_only_to_relevant_conditions(self):
        conditions = generate_auto_conditions(
            include_keyword_bias=False,
            include_noise_reduction=True,
            include_volume_normalization=False,
            include_llm_postprocess=True,
            include_rag=True,
            include_search=False,
            mode="full_valid",
            noise_models=["afftdn", "deepfilternet2"],
            rag_embedding_models=["intfloat/multilingual-e5-base", "BAAI/bge-m3"],
        )

        self.assertEqual(len(conditions), 12)
        noise_conditions = [condition for condition in conditions if condition.enable_noise_reduction]
        rag_conditions = [condition for condition in conditions if condition.enable_rag]
        self.assertEqual({condition.noise_reduction_model for condition in noise_conditions}, {"afftdn", "deepfilternet2"})
        self.assertEqual(
            {condition.rag_embedding_model for condition in rag_conditions},
            {"intfloat/multilingual-e5-base", "BAAI/bge-m3"},
        )
        self.assertTrue(any("__nmodel_afftdn" in condition.condition_id for condition in noise_conditions))
        self.assertTrue(any("__emb_baai_bge_m3" in condition.condition_id for condition in rag_conditions))
        self.assertGreater(len({condition.asr_group_key for condition in conditions}), 2)

    def test_l4x4_config_loads_pipeline_lanes(self):
        config = load_config("configs/l4x4.yaml")
        self.assertEqual(config.model_residency, "stage_replicas")
        self.assertEqual(config.postprocess_parallelism, 8)
        self.assertEqual(config.auto_experiment_parallelism, 8)
        self.assertTrue(config.auto_experiment_saturate_lanes)
        self.assertEqual(len(config.pipeline_lanes), 2)
        self.assertEqual(config.pipeline_lanes[1]["asr_base_url"], "http://127.0.0.1:18002/v1")
        self.assertEqual(config.pipeline_lanes[1]["preprocess_gpu"], "3")
        self.assertEqual(len(config.stage_server_base_urls), 4)
        self.assertEqual(config.stage_server_gpus, ["0", "1", "2", "3"])
        self.assertEqual(config.preprocess_gpus, ["0", "1", "2", "3"])
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
                enable_volume_normalization=True,
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
            self.assertIn("audit", report)
            self.assertTrue(report["audit"]["strict_valid"])
            self.assertEqual(report["audit"]["failed_count"], 0)
            self.assertEqual(report["audit"]["row_count"], report["case_count"])
            self.assertEqual(report["analysis"]["audit"]["verdict"], "valid")
            with Path(report["summary_csv"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            self.assertIn("delta_cer_vs_baseline", rows[0])
            self.assertIn("asr_latency_ms", rows[0])
            self.assertIn("vllm_total_tokens", rows[0])
            self.assertIn("preprocess_cache_hit", rows[0])
            self.assertIn("asr_base_url", rows[0])
            self.assertIn("post_base_url", rows[0])
            self.assertIn("preprocess_gpu", rows[0])
            self.assertIn("model_residency", rows[0])
            self.assertIn("planned_asr_cache_group_key", rows[0])

    def test_auto_experiment_starts_ready_conditions_before_all_asr_groups_finish(self):
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
                auto_experiment_saturate_lanes=True,
                asr_cache_enabled=True,
                preprocess_cache_enabled=True,
                pipeline_lanes=[
                    {
                        "asr_base_url": "http://lane-a/v1",
                        "post_base_url": "http://lane-a-post/v1",
                        "preprocess_gpu": "1",
                    },
                    {
                        "asr_base_url": "http://lane-b/v1",
                        "post_base_url": "http://lane-b-post/v1",
                        "preprocess_gpu": "3",
                    },
                ],
            )
            events = []
            preprocess_gpus = []
            event_lock = threading.Lock()
            first_prime_done = threading.Event()
            allow_second_prime_done = threading.Event()
            first_group = {"key": ""}

            def fake_prime(audio_path, base_config, case, index, reference_text, rag_inline_text, status_callback):
                key = case.condition.asr_group_key
                with event_lock:
                    is_first = not first_group["key"]
                    if is_first:
                        first_group["key"] = key
                    events.append(("prime_start", key))
                if is_first:
                    with event_lock:
                        events.append(("prime_done", key))
                    first_prime_done.set()
                    return
                first_prime_done.wait(1.0)
                allow_second_prime_done.wait(1.0)
                with event_lock:
                    events.append(("prime_done", key))

            def fake_run_condition(audio_path, config, case, reference_text, rag_inline_text):
                with event_lock:
                    events.append(("condition", case.condition.asr_group_key))
                    preprocess_gpus.append(config.preprocess_gpu)
                allow_second_prime_done.set()
                return {
                    "case_id": case.case_id,
                    "condition_id": case.condition.condition_id,
                    "label": case.condition.label,
                    "group": case.condition.group,
                    "asr_model": case.asr_model,
                    "post_model": case.post_model,
                    "llm_postprocess_enabled": case.condition.enable_llm_postprocess,
                    "cer_normalized_no_space": 0.0,
                    "wer_eojeol": 0.0,
                    "error": "",
                }

            with patch("asrpostprocessing.auto_experiment._prime_one_asr_group", side_effect=fake_prime), patch(
                "asrpostprocessing.auto_experiment._run_condition", side_effect=fake_run_condition
            ):
                report = run_auto_experiment(
                    str(audio),
                    config,
                    reference_text="테스트 전사 문장입니다.",
                    mode="full_valid",
                )

            self.assertEqual(report["analysis"]["num_failed_rows"], 0)
            first_key = first_group["key"]
            second_prime_done_index = next(
                index
                for index, event in enumerate(events)
                if event[0] == "prime_done" and event[1] != first_key
            )
            first_condition_index = next(index for index, event in enumerate(events) if event[0] == "condition")
            self.assertLess(first_condition_index, second_prime_done_index)
            self.assertIn("1", preprocess_gpus)
            self.assertIn("3", preprocess_gpus)

    def test_stage_replicas_route_cases_across_all_stage_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                model_residency="stage_replicas",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                enable_keyword_bias=True,
                enable_noise_reduction=False,
                enable_volume_normalization=True,
                enable_llm_postprocess=False,
                auto_experiment_parallelism=4,
                auto_experiment_saturate_lanes=True,
                asr_cache_enabled=True,
                preprocess_cache_enabled=True,
                stage_server_base_urls=[
                    "http://stage-0/v1",
                    "http://stage-1/v1",
                    "http://stage-2/v1",
                    "http://stage-3/v1",
                ],
                stage_server_gpus=["0", "1", "2", "3"],
                preprocess_gpus=["0", "1", "2", "3"],
            )
            seen = []

            def fake_run_condition(audio_path, condition_config, case, reference_text, rag_inline_text):
                seen.append((condition_config.asr_base_url, condition_config.post_base_url, condition_config.preprocess_gpu))
                return {
                    "case_id": case.case_id,
                    "condition_id": case.condition.condition_id,
                    "label": case.condition.label,
                    "group": case.condition.group,
                    "asr_model": case.asr_model,
                    "post_model": case.post_model,
                    "asr_base_url": condition_config.asr_base_url,
                    "post_base_url": condition_config.post_base_url,
                    "preprocess_gpu": condition_config.preprocess_gpu,
                    "model_residency": condition_config.model_residency,
                    "asr_cache_key": case.condition.asr_group_key,
                    "llm_postprocess_enabled": case.condition.enable_llm_postprocess,
                    "cer_normalized_no_space": 0.0,
                    "wer_eojeol": 0.0,
                    "error": "",
                }

            with patch("asrpostprocessing.auto_experiment._prime_one_asr_group"), patch(
                "asrpostprocessing.auto_experiment._run_condition", side_effect=fake_run_condition
            ):
                report = run_auto_experiment(str(audio), config, reference_text="테스트 전사 문장입니다.", mode="full_valid")

            self.assertEqual({item[0] for item in seen}, {"http://stage-0/v1", "http://stage-1/v1", "http://stage-2/v1", "http://stage-3/v1"})
            self.assertEqual({item[0] for item in seen}, {item[1] for item in seen})
            self.assertEqual({item[2] for item in seen}, {"0", "1", "2", "3"})
            self.assertEqual(
                set(report["audit"]["observed_asr_base_urls"]),
                {"http://stage-0/v1", "http://stage-1/v1", "http://stage-2/v1", "http://stage-3/v1"},
            )
            self.assertEqual(set(report["audit"]["observed_preprocess_gpus"]), {"0", "1", "2", "3"})

    def test_stage_replicas_auto_experiment_reloads_all_gpu_stage_servers(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="vllm_chat",
                post_backend="vllm_openai",
                model_residency="stage_replicas",
                auto_start_model_servers=True,
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                enable_keyword_bias=False,
                enable_noise_reduction=False,
                enable_volume_normalization=False,
                enable_llm_postprocess=True,
                enable_rag=False,
                enable_search=False,
                stage_server_base_urls=[
                    "http://stage-0/v1",
                    "http://stage-1/v1",
                    "http://stage-2/v1",
                    "http://stage-3/v1",
                ],
                stage_server_gpus=["0", "1", "2", "3"],
                preprocess_gpus=["0", "1", "2", "3"],
            )
            events = []

            def fake_ensure(config, status_callback=None, names=None):
                events.append(("ensure", tuple(names or ()), config.auto_start_model_servers))
                return []

            def fake_stop(config, status_callback=None, names=None):
                events.append(("stop", tuple(names or ()), config.auto_start_model_servers))
                return []

            def fake_prime(audio_path, base_config, cases, reference_text, rag_inline_text, status_callback):
                events.append(("prime", len(cases), base_config.auto_start_model_servers))

            def fake_run(audio_path, base_config, indexed_cases, reference_text, rag_inline_text, status_callback):
                events.append(("run", len(indexed_cases), base_config.auto_start_model_servers))
                return [
                    {
                        "case_id": case.case_id,
                        "condition_id": case.condition.condition_id,
                        "label": case.condition.label,
                        "group": case.condition.group,
                        "asr_model": case.asr_model,
                        "post_model": case.post_model,
                        "llm_postprocess_enabled": case.condition.enable_llm_postprocess,
                        "cer_normalized_no_space": 0.0,
                        "wer_eojeol": 0.0,
                        "error": "",
                    }
                    for _, case in indexed_cases
                ]

            with patch("asrpostprocessing.auto_experiment.ensure_model_servers", side_effect=fake_ensure), patch(
                "asrpostprocessing.auto_experiment.stop_model_servers", side_effect=fake_stop
            ), patch("asrpostprocessing.auto_experiment._prime_asr_groups", side_effect=fake_prime), patch(
                "asrpostprocessing.auto_experiment._run_conditions_parallel", side_effect=fake_run
            ):
                run_auto_experiment(str(audio), config, reference_text="테스트 전사 문장입니다.", mode="full_valid")

            self.assertEqual(
                events,
                [
                    ("ensure", ("asr_stage_0",), True),
                    ("ensure", ("asr_stage_1",), True),
                    ("ensure", ("asr_stage_2",), True),
                    ("ensure", ("asr_stage_3",), True),
                    ("prime", 2, False),
                    ("stop", ("asr_stage_0", "asr_stage_1", "asr_stage_2", "asr_stage_3"), True),
                    ("ensure", ("post_stage_0",), True),
                    ("ensure", ("post_stage_1",), True),
                    ("ensure", ("post_stage_2",), True),
                    ("ensure", ("post_stage_3",), True),
                    ("run", 2, False),
                    ("stop", ("post_stage_0", "post_stage_1", "post_stage_2", "post_stage_3"), True),
                ],
            )

    def test_stage_replicas_auto_experiment_skips_failed_post_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="vllm_chat",
                post_backend="vllm_openai",
                model_residency="stage_replicas",
                auto_start_model_servers=True,
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                enable_keyword_bias=False,
                enable_noise_reduction=False,
                enable_volume_normalization=False,
                enable_llm_postprocess=True,
                enable_rag=False,
                enable_search=False,
                stage_server_base_urls=[
                    "http://stage-0/v1",
                    "http://stage-1/v1",
                    "http://stage-2/v1",
                    "http://stage-3/v1",
                ],
                stage_server_gpus=["0", "1", "2", "3"],
                preprocess_gpus=["0", "1", "2", "3"],
            )
            events = []
            run_stage_urls = []
            run_preprocess_gpus = []

            def fake_ensure(config, status_callback=None, names=None):
                names_tuple = tuple(names or ())
                events.append(("ensure", names_tuple))
                if names_tuple == ("post_stage_0",):
                    raise RuntimeError("GPU 0 has insufficient free VRAM")
                return []

            def fake_stop(config, status_callback=None, names=None):
                events.append(("stop", tuple(names or ())))
                return []

            def fake_prime(audio_path, base_config, cases, reference_text, rag_inline_text, status_callback):
                events.append(("prime", tuple(base_config.stage_server_base_urls), tuple(base_config.preprocess_gpus)))

            def fake_run(audio_path, base_config, indexed_cases, reference_text, rag_inline_text, status_callback):
                run_stage_urls.extend(base_config.stage_server_base_urls)
                run_preprocess_gpus.extend(base_config.preprocess_gpus)
                return [
                    {
                        "case_id": case.case_id,
                        "condition_id": case.condition.condition_id,
                        "label": case.condition.label,
                        "group": case.condition.group,
                        "asr_model": case.asr_model,
                        "post_model": case.post_model,
                        "asr_base_url": base_config.stage_server_base_urls[0],
                        "post_base_url": base_config.stage_server_base_urls[0],
                        "preprocess_gpu": base_config.preprocess_gpus[0],
                        "model_residency": base_config.model_residency,
                        "planned_asr_cache_group_key": case.condition.asr_group_key,
                        "asr_cache_key": case.condition.asr_group_key,
                        "llm_postprocess_enabled": case.condition.enable_llm_postprocess,
                        "cer_normalized_no_space": 0.0,
                        "wer_eojeol": 0.0,
                        "error": "",
                    }
                    for _, case in indexed_cases
                ]

            with patch("asrpostprocessing.auto_experiment.ensure_model_servers", side_effect=fake_ensure), patch(
                "asrpostprocessing.auto_experiment.stop_model_servers", side_effect=fake_stop
            ), patch("asrpostprocessing.auto_experiment._prime_asr_groups", side_effect=fake_prime), patch(
                "asrpostprocessing.auto_experiment._run_conditions_parallel", side_effect=fake_run
            ):
                report = run_auto_experiment(str(audio), config, reference_text="테스트 전사 문장입니다.", mode="full_valid")

            self.assertIn(("ensure", ("post_stage_0",)), events)
            self.assertIn(("stop", ("post_stage_0",)), events)
            self.assertEqual(run_stage_urls, ["http://stage-1/v1", "http://stage-2/v1", "http://stage-3/v1"])
            self.assertEqual(run_preprocess_gpus, ["1", "2", "3"])
            self.assertEqual(report["analysis"]["num_failed_rows"], 0)

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

    def test_auto_experiment_applies_rag_embedding_model_axis(self):
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
                auto_experiment_include_models=True,
                auto_experiment_rag_embedding_models=["intfloat/multilingual-e5-base", "BAAI/bge-m3"],
                auto_experiment_postprocess_strengths=[0.5],
                auto_experiment_rag_strengths=[0.25],
                auto_experiment_rag_top_ks=[3],
                auto_experiment_parallelism=2,
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
            self.assertEqual(
                {row["rag_embedding_model"] for row in rag_rows},
                {"intfloat/multilingual-e5-base", "BAAI/bge-m3"},
            )
            with Path(report["summary_csv"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("rag_embedding_model", rows[0])

    def test_auto_experiment_rows_record_rag_and_search_runtime_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                mock_transcript="AlphaTerm 관련 설명입니다.",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                enable_keyword_bias=False,
                enable_noise_reduction=False,
                enable_volume_normalization=False,
                enable_llm_postprocess=True,
                enable_rag=True,
                rag_inline_text="AlphaTerm은 프로젝트 핵심 용어입니다.",
                rag_strength=1.0,
                rag_top_k=2,
                enable_search=True,
                search_provider="endpoint",
                search_endpoint="https://search.example.test",
                search_strength=0.8,
                auto_experiment_parallelism=2,
                asr_cache_enabled=True,
                preprocess_cache_enabled=True,
            )
            search_result = SearchResult(
                query="AlphaTerm",
                title="AlphaTerm reference",
                url="https://example.test/alpha",
                snippet="AlphaTerm search context",
                source="endpoint",
            )
            with patch("asrpostprocessing.pipeline.CachedSearchProvider.search", return_value=[search_result]):
                report = run_auto_experiment(
                    str(audio),
                    config,
                    reference_text="AlphaTerm 관련 설명입니다.",
                    mode="core_ablation",
                )

            rag_rows = [row for row in report["rows"] if row["rag_enabled"]]
            search_rows = [row for row in report["rows"] if row["search_enabled"]]
            self.assertTrue(rag_rows)
            self.assertTrue(search_rows)
            self.assertTrue(all(int(row["rag_context_count"]) >= 1 for row in rag_rows))
            self.assertTrue(all(int(row["search_result_count"]) == 1 for row in search_rows))
            self.assertIn("best_methods", report["analysis"])
            with Path(report["summary_csv"]).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("rag_context_count", rows[0])
            self.assertIn("rag_used_context_count", rows[0])
            self.assertIn("search_result_count", rows[0])


if __name__ == "__main__":
    unittest.main()
