import os
import json
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.correction_parser import parse_correction_response
from asrpostprocessing.model_server import ModelServerStatus
from asrpostprocessing.pipeline import PipelineRunner, _preprocess_status
from asrpostprocessing.ui import (
    NOISE_REDUCTION_MODEL_CHOICES,
    _apply_runtime_saturation,
    _auto_experiment_diff_html,
    _canonical_noise_reduction_model,
    preview_preprocessed_audio_from_ui,
    run_from_ui,
    run_from_ui_stream,
)


class ParserPipelineUiTest(unittest.TestCase):
    def test_noise_reduction_dropdown_accepts_config_canonical_values(self):
        choice_values = {value for _, value in NOISE_REDUCTION_MODEL_CHOICES}
        self.assertIn(_canonical_noise_reduction_model("deepfilternet2"), choice_values)
        self.assertIn(_canonical_noise_reduction_model("DeepFilterNet2"), choice_values)
        self.assertIn(_canonical_noise_reduction_model("DeepFilterNet2-PF"), choice_values)
        self.assertIn(_canonical_noise_reduction_model("RNNoise"), choice_values)

    def test_parse_correction_json(self):
        payload = '{"corrected_text":"교정된 문장","edits":[{"before":"원문","after":"교정","reason":"term","confidence":0.8}],"risk":"low","used_context_ids":["ctx1"]}'
        result = parse_correction_response(payload, "원문 문장")
        self.assertEqual(result.corrected_text, "교정된 문장")
        self.assertEqual(result.edits[0].before, "원문")
        self.assertEqual(result.used_context_ids, ["ctx1"])

    def test_parse_failure_keeps_original(self):
        result = parse_correction_response("not json", "원문")
        self.assertEqual(result.corrected_text, "원문")
        self.assertEqual(result.risk, "high")

    def test_auto_experiment_html_lists_each_case_cer_wer_readably(self):
        report = {
            "summary_csv": "/tmp/auto_experiment_summary.csv",
            "analysis": {"best_by_cer": {"case_id": "case-b", "cer_normalized_no_space": 0.1, "wer_eojeol": 0.2}},
            "audit": {
                "verdict": "valid",
                "row_count": 2,
                "expected_case_count": 2,
                "failed_count": 0,
                "cer_wer_row_count": 2,
                "baseline_cer_normalized_no_space": 0.3,
                "best_cer_improvement_vs_baseline": 0.2,
                "best_wer_improvement_vs_baseline": 0.2,
                "observed_asr_cache_group_count": 2,
                "expected_asr_cache_group_count": 2,
                "peak_gpu_utilization_percent": 98.5,
                "peak_vram_mb": 20480,
                "observed_asr_base_urls": ["http://127.0.0.1:18000/v1"],
                "observed_post_base_urls": ["http://127.0.0.1:18001/v1"],
                "observed_preprocess_gpus": ["0", "1"],
                "conclusion": "Strictly comparable run completed.",
                "gates": {
                    "reference_provided": True,
                    "all_expected_cases_finished": True,
                    "condition_coverage_complete": True,
                    "no_failed_cases": True,
                    "baseline_present": True,
                    "cer_wer_available_for_all_rows": True,
                    "asr_cache_groups_observed": True,
                },
            },
            "rows": [
                {
                    "case_id": "case-a",
                    "condition_id": "baseline",
                    "label": "Baseline",
                    "cer_normalized_no_space": 0.3,
                    "wer_eojeol": 0.4,
                    "delta_cer_vs_baseline": 0.0,
                    "delta_wer_vs_baseline": 0.0,
                    "asr_base_url": "http://127.0.0.1:18000/v1",
                    "post_base_url": "http://127.0.0.1:18001/v1",
                    "preprocess_gpu": "0",
                    "peak_gpu_utilization_percent": 80,
                    "asr_cache_hit": False,
                    "risk": "unchanged",
                },
                {
                    "case_id": "case-b",
                    "condition_id": "keyword__llm",
                    "label": "Keyword + LLM",
                    "keyword_bias_enabled": True,
                    "llm_postprocess_enabled": True,
                    "cer_normalized_no_space": 0.1,
                    "wer_eojeol": 0.2,
                    "delta_cer_vs_baseline": 0.2,
                    "delta_wer_vs_baseline": 0.2,
                    "asr_base_url": "http://127.0.0.1:18000/v1",
                    "post_base_url": "http://127.0.0.1:18001/v1",
                    "preprocess_gpu": "1",
                    "peak_gpu_utilization_percent": 98.5,
                    "asr_cache_hit": True,
                    "risk": "low",
                },
            ],
        }

        html = _auto_experiment_diff_html(report, "기준 문장")

        self.assertIn("Auto Experiment CER/WER by condition", html)
        self.assertIn("Keyword + LLM", html)
        self.assertIn("Baseline", html)
        self.assertIn("0.1000", html)
        self.assertIn("0.2000", html)
        self.assertIn("+0.2000", html)
        self.assertIn("Strict Experiment Audit", html)
        self.assertIn("PASS", html)
        self.assertIn("ASR URL", html)
        self.assertIn("Post URL", html)
        self.assertIn("PRE GPU", html)
        self.assertIn("Peak GPU", html)
        self.assertIn("ASR Cache", html)
        self.assertIn("http://127.0.0.1:18000/v1", html)
        self.assertIn("http://127.0.0.1:18001/v1", html)
        self.assertIn("98.5000%", html)
        self.assertLess(html.index("case-b"), html.index("case-a"))

    def test_auto_experiment_blank_model_inputs_use_configured_model_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            captured = {}
            base_config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                asr_model="base-asr",
                post_model="base-post",
                auto_experiment_asr_models=["strong-asr-a", "strong-asr-b"],
                auto_experiment_post_models=["strong-post-a", "strong-post-b"],
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
            )

            def fake_run_auto_experiment(*, audio_path, base_config, reference_text, rag_inline_text, mode, status_callback=None):
                captured["audio_path"] = audio_path
                captured["config"] = base_config
                captured["reference_text"] = reference_text
                captured["mode"] = mode
                return {
                    "run_id": "auto-test",
                    "mode": mode,
                    "condition_count": 0,
                    "summary_csv": str(Path(tmp) / "summary.csv"),
                    "analysis_json": str(Path(tmp) / "analysis.json"),
                    "output_dir": str(Path(tmp) / "outputs"),
                    "analysis": {},
                    "rows": [],
                }

            with patch("asrpostprocessing.ui.run_auto_experiment", side_effect=fake_run_auto_experiment):
                *_unused, status, _preprocessed_audio, _html, _gpu = run_from_ui(
                    audio_path=str(audio),
                    large_audio_file=None,
                    reference_text="기준 문장입니다.",
                    reference_file=None,
                    enable_keyword_bias=False,
                    keyword_bias_weight=0.0,
                    keywords="",
                    enable_noise_reduction=False,
                    noise_reduction_model="none",
                    noise_reduction_strength=0.0,
                    enable_volume_normalization=False,
                    volume_normalization_strength=0.0,
                    volume_target_dbfs=-20.0,
                    enable_llm=True,
                    postprocess_strength=0.5,
                    enable_rag=False,
                    rag_strength=0.0,
                    rag_top_k=5,
                    rag_text="",
                    rag_files=None,
                    enable_search=False,
                    search_strength=0.0,
                    search_provider="duckduckgo",
                    search_endpoint="",
                    asr_model="",
                    post_model="",
                    asr_base_url="",
                    post_base_url="",
                    auto_start_model_servers=False,
                    server_start_timeout_s=60,
                    server_log_dir=str(Path(tmp) / "server_logs"),
                    asr_server_gpu="0",
                    post_server_gpu="1",
                    asr_server_host="127.0.0.1",
                    post_server_host="127.0.0.1",
                    asr_server_command="",
                    post_server_command="",
                    asr_backend="mock",
                    post_backend="mock",
                    auto_experiment_mode=True,
                    auto_experiment_coverage="full_valid",
                    auto_experiment_include_models=True,
                    auto_experiment_asr_models="",
                    auto_experiment_post_models="",
                    base_config_state=base_config.to_dict(),
                )

            config = captured["config"]
            self.assertEqual(config.asr_model, "base-asr")
            self.assertEqual(config.post_model, "base-post")
            self.assertEqual(config.auto_experiment_asr_models, ["strong-asr-a", "strong-asr-b", "base-asr"])
            self.assertEqual(config.auto_experiment_post_models, ["strong-post-a", "strong-post-b", "base-post"])
            self.assertEqual(captured["mode"], "full_valid")
            self.assertIn("Auto Experiment ID: auto-test", status)

    def test_pipeline_mock_end_to_end_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"not a real wav but mock backend does not inspect it")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
            )
            output = PipelineRunner(config).run(str(audio), reference_text="테스트 전사 문장입니다.", run_id="test-run")
            self.assertEqual(output.correction.corrected_text, "테스트 전사 문장입니다.")
            self.assertTrue((Path(output.output_dir) / "result.json").exists())
            self.assertTrue((Path(output.output_dir) / "raw_transcript.txt").exists())
            self.assertTrue((Path(output.output_dir) / "corrected_transcript.txt").exists())
            self.assertTrue((Path(output.output_dir) / "diff.html").exists())
            self.assertEqual((Path(output.output_dir) / "raw_transcript.txt").read_text(encoding="utf-8"), output.raw.text)
            self.assertEqual(
                (Path(output.output_dir) / "corrected_transcript.txt").read_text(encoding="utf-8"),
                output.correction.corrected_text,
            )
            self.assertTrue((Path(output.output_dir) / "asr_quality.json").exists())
            self.assertTrue((Path(output.output_dir) / "correction_quality.json").exists())
            self.assertTrue((Path(output.output_dir) / "metrics.json").exists())
            self.assertTrue((Path(output.output_dir) / "preprocess.json").exists())
            self.assertTrue((Path(output.output_dir) / "vllm_metrics.json").exists())
            result_payload = json.loads((Path(output.output_dir) / "result.json").read_text(encoding="utf-8"))
            self.assertIn("asr_quality", result_payload)
            self.assertIn("correction_quality", result_payload)
            self.assertIn("artifacts", result_payload)
            self.assertIn("vllm_metrics", result_payload)
            self.assertEqual(result_payload["artifacts"]["raw_transcript"], output.artifacts["raw_transcript"])
            self.assertIn("raw_transcript", output.artifacts)
            self.assertIn("correction_quality", output.artifacts)
            self.assertIn("vllm_metrics", output.artifacts)
            self.assertFalse(output.vllm_metrics["available"])
            self.assertEqual(output.asr_quality["backend"], "mock")
            self.assertEqual(output.correction_quality["risk"], output.correction.risk)
            run_dir = Path(tmp) / "runs" / "test-run"
            metrics_tsv = run_dir / "metrics.tsv"
            self.assertTrue(metrics_tsv.exists())
            metrics_text = metrics_tsv.read_text(encoding="utf-8")
            self.assertIn("cer_normalized_no_space", metrics_text)
            self.assertIn("wer_eojeol", metrics_text)
            self.assertTrue(any(path.name.startswith("events.out.tfevents") for path in run_dir.iterdir()))

    def test_pipeline_emits_progress_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            events = []
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
            )
            PipelineRunner(config, status_callback=events.append).run(
                str(audio),
                reference_text="테스트 전사 문장입니다.",
                run_id="progress-test",
            )
        event_text = "\n".join(events)
        self.assertIn("Checking model server readiness", event_text)
        self.assertIn("Sending audio to ASR backend mock", event_text)
        self.assertIn("Post-processing chunk 1/1", event_text)
        self.assertIn("Run progress-test complete", event_text)

    def test_mock_asr_cache_tracks_sidecar_transcript_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"same audio bytes")
            sidecar = audio.with_suffix(".txt")
            sidecar.write_text("첫 번째 전사", encoding="utf-8")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                asr_cache_enabled=True,
                cache_dir=str(Path(tmp) / "cache"),
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
            )

            first = PipelineRunner(config).run(str(audio), reference_text="첫 번째 전사", run_id="first")
            sidecar.write_text("두 번째 전사", encoding="utf-8")
            second = PipelineRunner(config).run(str(audio), reference_text="두 번째 전사", run_id="second")

        self.assertEqual(first.raw.text, "첫 번째 전사")
        self.assertEqual(second.raw.text, "두 번째 전사")

    def test_stage_replicas_post_endpoint_pool_starts_from_assigned_case_endpoint(self):
        config = ExperimentConfig(
            model_residency="stage_replicas",
            post_base_url="http://stage-3/v1",
            stage_server_base_urls=[
                "http://stage-0/v1",
                "http://stage-1/v1",
                "http://stage-2/v1",
                "http://stage-3/v1",
            ],
        )

        self.assertEqual(
            PipelineRunner(config)._post_endpoint_pool(),
            [
                "http://stage-3/v1",
                "http://stage-0/v1",
                "http://stage-1/v1",
                "http://stage-2/v1",
            ],
        )

    def test_pipeline_falls_back_when_postprocess_chunk_fails(self):
        class FailingPostprocessor:
            def correct(self, chunk_text, config, contexts, search_results):
                raise RuntimeError("post backend timeout")

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                mock_transcript="모표 용어를 설명합니다.",
                keywords=["목표 용어"],
                postprocess_strength=0.5,
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
            )
            with patch("asrpostprocessing.pipeline.build_postprocess_adapter", return_value=FailingPostprocessor()):
                output = PipelineRunner(config).run(str(audio), run_id="fallback-test")

        self.assertEqual(output.correction.corrected_text, "목표 용어를 설명합니다.")
        self.assertEqual(output.correction.risk, "high")
        self.assertEqual(output.correction.edits[0].before, "모표 용어를")
        chunk_metadata = output.correction.metadata["chunks"][0]["metadata"]
        self.assertEqual(chunk_metadata["fallback"], "raw_transcript_after_postprocess_error")
        self.assertIn("post backend timeout", chunk_metadata["postprocess_error"])
        self.assertEqual(output.correction_quality["postprocess"]["fallback_chunk_count"], 1)
        self.assertEqual(output.correction_quality["keyword_near_misses"]["corrected_count"], 0)

    def test_preprocess_status_surfaces_applied_warnings(self):
        status = _preprocess_status(
            {
                "applied": True,
                "steps": [{"step": "volume_normalization"}],
                "warnings": ["Volume normalization gain was peak-limited to avoid clipping."],
            }
        )

        self.assertIn("Preprocessing complete", status)
        self.assertIn("peak-limited", status)

    def test_sequential_model_residency_prepares_and_releases_each_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                auto_start_model_servers=True,
                model_residency="sequential",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
            )
            calls = []

            def fake_ensure(_config, status_callback=None, names=None):
                name = list(names or ["all"])[0]
                calls.append(("ensure", name))
                return [ModelServerStatus(name=name, base_url=f"http://{name}", status="ready", detail="test")]

            def fake_stop(_config, status_callback=None, names=None):
                name = list(names or ["all"])[0]
                calls.append(("stop", name))
                return [ModelServerStatus(name=name, base_url=f"http://{name}", status="stopped", detail="test")]

            with patch("asrpostprocessing.pipeline.ensure_model_servers", side_effect=fake_ensure), patch(
                "asrpostprocessing.pipeline.stop_model_servers", side_effect=fake_stop
            ):
                output = PipelineRunner(config).run(str(audio), reference_text="테스트 전사 문장입니다.")
            self.assertEqual(calls, [("ensure", "asr"), ("stop", "asr"), ("ensure", "post"), ("stop", "post")])
            self.assertEqual([item["status"] for item in output.server_statuses], ["ready", "stopped", "ready", "stopped"])

    def test_ui_event_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            base_config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                cache_dir=str(Path(tmp) / "cache"),
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                upload_cache_dir=str(Path(tmp) / "upload_cache"),
            )
            raw, corrected, diff, metrics, edits, preprocess, servers, status, preprocessed_audio, preprocessed_audio_html, gpu_status = run_from_ui(
                str(audio),
                None,
                "테스트 전사 문장입니다.",
                None,
                True,
                0.5,
                "",
                False,
                "none",
                0.0,
                False,
                0.0,
                -20.0,
                True,
                0.5,
                False,
                0.0,
                5,
                "",
                None,
                False,
                0.0,
                "duckduckgo",
                "",
                "Qwen/Qwen3-ASR-1.7B",
                "Qwen/Qwen3.5-9B",
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                False,
                60,
                str(Path(tmp) / "server_logs"),
                "0",
                "1",
                "127.0.0.1",
                "127.0.0.1",
                "",
                "",
                "mock",
                "mock",
                base_config_state=base_config.to_dict(),
            )
            self.assertEqual(raw, "테스트 전사 문장입니다.")
            self.assertEqual(corrected, "테스트 전사 문장입니다.")
            self.assertIn("Run ID:", status)
            self.assertIn("cer_normalized_no_space", metrics)
            self.assertIsInstance(edits, list)
            self.assertFalse(preprocess["applied"])
            self.assertNotEqual(preprocessed_audio, str(audio))
            self.assertTrue(Path(str(preprocessed_audio)).exists())
            self.assertEqual(preprocessed_audio_html, "")
            self.assertEqual(servers, [])
            self.assertIn("diff", diff.lower())
            self.assertIn("Audio upload cache", status)
            self.assertIn("available", gpu_status)

    def test_ui_preserves_configured_pipeline_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            base_config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                asr_base_urls=["http://127.0.0.1:18000/v1", "http://127.0.0.1:18002/v1"],
                post_base_urls=["http://127.0.0.1:18001/v1", "http://127.0.0.1:18003/v1"],
                pipeline_lanes=[
                    {
                        "name": "lane_a",
                        "asr_base_url": "http://127.0.0.1:18000/v1",
                        "post_base_url": "http://127.0.0.1:18001/v1",
                        "asr_server_gpu": "0",
                        "post_server_gpu": "1",
                    },
                    {
                        "name": "lane_b",
                        "asr_base_url": "http://127.0.0.1:18002/v1",
                        "post_base_url": "http://127.0.0.1:18003/v1",
                        "asr_server_gpu": "2",
                        "post_server_gpu": "3",
                    },
                ],
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
            )
            raw, corrected, diff, metrics, edits, preprocess, servers, status, preprocessed_audio, preprocessed_audio_html, gpu_status = run_from_ui(
                str(audio),
                None,
                "테스트 전사 문장입니다.",
                None,
                True,
                0.5,
                "",
                False,
                "none",
                0.0,
                False,
                0.0,
                -20.0,
                True,
                0.5,
                False,
                0.0,
                5,
                "",
                None,
                False,
                0.0,
                "duckduckgo",
                "",
                "Qwen/Qwen3-ASR-1.7B",
                "Qwen/Qwen3.5-9B",
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                False,
                60,
                str(Path(tmp) / "server_logs"),
                "0",
                "1",
                "127.0.0.1",
                "127.0.0.1",
                "",
                "",
                "mock",
                "mock",
                base_config_state=base_config.to_dict(),
            )
            self.assertEqual(raw, "테스트 전사 문장입니다.")
            self.assertIn("Pipeline lanes:", status)
            self.assertIn("lane_b", status)
            self.assertIn("GPU 2", status)
            self.assertIn("http://127.0.0.1:18003/v1", status)

    def test_ui_stream_reports_progress_before_final_result(self):
        final_output = (
            "raw",
            "corrected",
            "<div class='diff'></div>",
            {"latency_ms": 50.0},
            [],
            {},
            [],
            "Run ID: stream-test",
            None,
            "",
            {"available": True},
        )

        def fake_run_from_ui(*_args, status_callback=None):
            if status_callback:
                status_callback("Sending audio to ASR backend mock.")
            time.sleep(0.05)
            return final_output

        gpu_status = {
            "available": True,
            "gpus": [
                {
                    "index": 0,
                    "memory_used_mb": 11023,
                    "memory_total_mb": 16384,
                    "gpu_utilization_percent": 0,
                    "temperature_c": 22,
                }
            ],
            "processes": [{"pid": 40289, "process_name": "VLLM::EngineCore", "used_memory_mb": 10944}],
        }
        with patch("asrpostprocessing.ui.run_from_ui", side_effect=fake_run_from_ui), patch(
            "asrpostprocessing.ui.query_gpu_status", return_value=gpu_status
        ), patch("asrpostprocessing.ui.RUN_STATUS_POLL_INTERVAL_S", 0.01):
            outputs = list(run_from_ui_stream("mock-arg"))

        self.assertGreaterEqual(len(outputs), 2)
        progress_text = "\n".join(output[7] for output in outputs[:-1])
        self.assertIn("Run in progress", progress_text)
        self.assertIn("Sending audio to ASR backend mock", progress_text)
        self.assertIn("GPU0: util 0%", progress_text)
        self.assertIn("VLLM::EngineCore", progress_text)
        self.assertEqual(outputs[-1], final_output)

    def test_ui_vllm_failure_has_actionable_hint(self):
        raw, corrected, diff, metrics, edits, preprocess, servers, status, preprocessed_audio, preprocessed_audio_html, gpu_status = run_from_ui(
            "missing.wav",
            None,
            "",
            None,
            False,
            0.0,
            "",
            False,
            "none",
            0.0,
            False,
            0.0,
            -20.0,
            True,
            0.5,
            False,
            0.0,
            5,
            "",
            None,
            False,
            0.0,
            "duckduckgo",
            "",
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3.5-9B",
            "http://127.0.0.1:1/v1",
            "http://127.0.0.1:1/v1",
            False,
            60,
            "outputs/model_servers",
            "0",
            "1",
            "127.0.0.1",
            "127.0.0.1",
            "",
            "",
            "vllm_chat",
            "vllm_openai",
        )
        self.assertEqual(raw, "")
        self.assertEqual(preprocess, {})
        self.assertIsNone(preprocessed_audio)
        self.assertEqual(preprocessed_audio_html, "")
        self.assertEqual(servers, [])
        self.assertIn("Run failed:", status)
        self.assertIn("For UI-only testing", status)
        self.assertIn("available", gpu_status)

    def test_preprocessed_audio_preview_returns_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = Path.cwd()
            try:
                os.chdir(tmp)
                audio = Path(tmp) / "sample.wav"
                _write_pcm16_wav(audio, [1000, -1000, 1000, -1000])
                preview_path, preview_html, preprocess, status = preview_preprocessed_audio_from_ui(
                    str(audio),
                    None,
                    False,
                    "none",
                    0.0,
                    True,
                    1.0,
                    -20.0,
                )
                self.assertIsNotNone(preview_path)
                self.assertNotEqual(preview_path, str(audio))
                self.assertTrue(Path(preview_path).exists())
                self.assertTrue(preprocess["applied"])
                self.assertIn("<audio controls", preview_html)
                self.assertIn("Duration:", preview_html)
                self.assertIn("Preprocessed audio ready", status)
            finally:
                os.chdir(current)

    def test_preprocessed_audio_preview_applies_afftdn_without_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            current = Path.cwd()
            try:
                os.chdir(tmp)
                audio = Path(tmp) / "sample.wav"
                _write_pcm16_wav(audio, [40, -40, 1400, -1400] * 4000)
                preview_path, preview_html, preprocess, status = preview_preprocessed_audio_from_ui(
                    str(audio),
                    None,
                    True,
                    "afftdn",
                    0.5,
                    False,
                    0.0,
                    -20.0,
                )
                self.assertIsNotNone(preview_path)
                self.assertTrue(Path(preview_path).exists())
                self.assertTrue(preprocess["applied"])
                self.assertEqual(preprocess["steps"][0]["metadata"]["processor"], "ffmpeg_afftdn")
                self.assertGreater(preprocess["steps"][0]["metadata"]["duration_seconds"], 0.9)
                self.assertIn("<audio controls", preview_html)
                self.assertIn("Duration: 0:01", preview_html)
                self.assertIn("Preprocessed audio ready", status)
                self.assertNotIn("No preprocessing", status)
                self.assertNotIn("command", status.lower())
                second_preview_path, second_preview_html, second_preprocess, _second_status = preview_preprocessed_audio_from_ui(
                    str(audio),
                    None,
                    True,
                    "afftdn",
                    0.5,
                    False,
                    0.0,
                    -20.0,
                )
                self.assertTrue(second_preprocess["applied"])
                self.assertIn("<audio controls", second_preview_html)
                self.assertNotEqual(preview_path, second_preview_path)
            finally:
                os.chdir(current)

    def test_ui_accepts_large_audio_file_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "hour-long.mp3"
            audio.write_bytes(b"mock mp3")
            audio.with_suffix(".txt").write_text("긴 오디오 테스트 문장입니다.", encoding="utf-8")
            base_config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                cache_dir=str(Path(tmp) / "cache"),
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                upload_cache_dir=str(Path(tmp) / "upload_cache"),
            )
            raw, corrected, diff, metrics, edits, preprocess, servers, status, preprocessed_audio, preprocessed_audio_html, gpu_status = run_from_ui(
                None,
                {"name": str(audio)},
                "긴 오디오 테스트 문장입니다.",
                None,
                True,
                0.5,
                "",
                False,
                "none",
                0.0,
                False,
                0.0,
                -20.0,
                True,
                0.5,
                False,
                0.0,
                5,
                "",
                None,
                False,
                0.0,
                "duckduckgo",
                "",
                "Qwen/Qwen3-ASR-1.7B",
                "Qwen/Qwen3.5-9B",
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                False,
                60,
                str(Path(tmp) / "server_logs"),
                "0",
                "1",
                "127.0.0.1",
                "127.0.0.1",
                "",
                "",
                "mock",
                "mock",
                base_config_state=base_config.to_dict(),
            )
            self.assertEqual(raw, "긴 오디오 테스트 문장입니다.")
            self.assertEqual(corrected, "긴 오디오 테스트 문장입니다.")
            self.assertNotEqual(preprocessed_audio, str(audio))
            self.assertTrue(Path(str(preprocessed_audio)).exists())
            self.assertEqual(preprocessed_audio_html, "")
            self.assertFalse(preprocess["applied"])
            self.assertIn("Run ID:", status)
            self.assertIn("Audio upload cache", status)
            self.assertIn("available", gpu_status)

    def test_ui_metrics_warn_when_reference_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            base_config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                cache_dir=str(Path(tmp) / "cache"),
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                upload_cache_dir=str(Path(tmp) / "upload_cache"),
            )
            raw, corrected, diff, metrics, _edits, _preprocess, _servers, status, _audio, _html, _gpu = run_from_ui(
                str(audio),
                None,
                "",
                None,
                False,
                0.0,
                "",
                False,
                "none",
                0.0,
                False,
                0.0,
                -20.0,
                True,
                0.5,
                False,
                0.0,
                5,
                "",
                None,
                False,
                0.0,
                "duckduckgo",
                "",
                "Qwen/Qwen3-ASR-1.7B",
                "Qwen/Qwen3.5-9B",
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                False,
                60,
                str(Path(tmp) / "server_logs"),
                "0",
                "1",
                "127.0.0.1",
                "127.0.0.1",
                "",
                "",
                "mock",
                "mock",
                base_config_state=base_config.to_dict(),
            )

        self.assertEqual(raw, corrected)
        self.assertIn("Raw -&gt; Corrected", diff)
        self.assertIsNone(metrics["cer_normalized_no_space"])
        self.assertFalse(metrics["reference_provided"])
        self.assertIn("reference", metrics["reference_required"].lower())
        self.assertIn("Run ID:", status)

    def test_ui_reads_sidecar_reference_for_cached_large_audio_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "hour-long.mp3"
            audio.write_bytes(b"mock mp3 payload")
            audio.with_suffix(".txt").write_text("긴 오디오 테스트 문장입니다.", encoding="utf-8")
            base_config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                cache_dir=str(Path(tmp) / "cache"),
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                upload_cache_dir=str(Path(tmp) / "upload_cache"),
            )
            raw, corrected, diff, metrics, _edits, _preprocess, _servers, status, preprocessed_audio, _html, _gpu = run_from_ui(
                None,
                {"name": str(audio)},
                "",
                None,
                False,
                0.0,
                "",
                False,
                "none",
                0.0,
                False,
                0.0,
                -20.0,
                True,
                0.5,
                False,
                0.0,
                5,
                "",
                None,
                False,
                0.0,
                "duckduckgo",
                "",
                "Qwen/Qwen3-ASR-1.7B",
                "Qwen/Qwen3.5-9B",
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                False,
                60,
                str(Path(tmp) / "server_logs"),
                "0",
                "1",
                "127.0.0.1",
                "127.0.0.1",
                "",
                "",
                "mock",
                "mock",
                base_config_state=base_config.to_dict(),
            )

            self.assertEqual(raw, "긴 오디오 테스트 문장입니다.")
            self.assertEqual(corrected, "긴 오디오 테스트 문장입니다.")
            self.assertNotEqual(preprocessed_audio, str(audio))
            self.assertTrue(metrics["reference_provided"])
            self.assertEqual(metrics["cer_normalized_no_space"], 0.0)
            self.assertEqual(metrics["wer_eojeol"], 0.0)
            self.assertIn("Reference -&gt; Corrected", diff)
            self.assertIn("Audio upload cache", status)

    def test_ui_caches_large_audio_file_input_by_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "hour-long.mp3"
            audio.write_bytes(b"mock mp3 payload")
            audio.with_suffix(".txt").write_text("긴 오디오 테스트 문장입니다.", encoding="utf-8")
            base_config = ExperimentConfig(
                asr_backend="mock",
                post_backend="mock",
                output_dir=str(Path(tmp) / "outputs"),
                runs_dir=str(Path(tmp) / "runs"),
                upload_cache_dir=str(Path(tmp) / "upload_cache"),
            )

            first = run_from_ui(
                None,
                {"name": str(audio)},
                "긴 오디오 테스트 문장입니다.",
                None,
                True,
                0.5,
                "",
                False,
                "none",
                0.0,
                False,
                0.0,
                -20.0,
                True,
                0.5,
                False,
                0.0,
                5,
                "",
                None,
                False,
                0.0,
                "duckduckgo",
                "",
                "Qwen/Qwen3-ASR-1.7B",
                "Qwen/Qwen3.5-9B",
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                False,
                60,
                str(Path(tmp) / "server_logs"),
                "0",
                "1",
                "127.0.0.1",
                "127.0.0.1",
                "",
                "",
                "mock",
                "mock",
                base_config_state=base_config.to_dict(),
            )
            second = run_from_ui(
                None,
                {"name": str(audio)},
                "긴 오디오 테스트 문장입니다.",
                None,
                True,
                0.5,
                "",
                False,
                "none",
                0.0,
                False,
                0.0,
                -20.0,
                True,
                0.5,
                False,
                0.0,
                5,
                "",
                None,
                False,
                0.0,
                "duckduckgo",
                "",
                "Qwen/Qwen3-ASR-1.7B",
                "Qwen/Qwen3.5-9B",
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                False,
                60,
                str(Path(tmp) / "server_logs"),
                "0",
                "1",
                "127.0.0.1",
                "127.0.0.1",
                "",
                "",
                "mock",
                "mock",
                base_config_state=base_config.to_dict(),
            )

            first_cached_audio = first[8]
            second_cached_audio = second[8]
            self.assertNotEqual(first_cached_audio, str(audio))
            self.assertEqual(first_cached_audio, second_cached_audio)
            self.assertTrue(Path(str(first_cached_audio)).exists())
            self.assertIn("Audio upload cache stored", first[7])
            self.assertIn("Audio upload cache hit", second[7])

    def test_runtime_saturation_raises_workers_to_lane_count_without_hardcoded_gpu_ids(self):
        config = ExperimentConfig(
            model_residency="stage_replicas",
            stage_server_base_urls=[
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18001/v1",
                "http://127.0.0.1:18002/v1",
                "http://127.0.0.1:18003/v1",
            ],
            preprocess_gpus=["0", "1", "2", "3"],
            asr_context_chars=0,
            asr_chunk_parallelism=1,
            postprocess_parallelism=1,
            auto_experiment_parallelism=1,
            auto_experiment_saturate_lanes=True,
        )

        messages = _apply_runtime_saturation(config)

        self.assertEqual(config.asr_chunk_parallelism, 4)
        self.assertEqual(config.postprocess_parallelism, 4)
        self.assertEqual(config.auto_experiment_parallelism, 4)
        self.assertTrue(any("ASR chunk workers 1 -> 4" in message for message in messages))


def _write_pcm16_wav(path: Path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


if __name__ == "__main__":
    unittest.main()
