import os
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
from asrpostprocessing.ui import preview_preprocessed_audio_from_ui, run_from_ui, run_from_ui_stream


class ParserPipelineUiTest(unittest.TestCase):
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
            self.assertTrue((Path(output.output_dir) / "metrics.json").exists())
            self.assertTrue((Path(output.output_dir) / "preprocess.json").exists())
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
            )
            self.assertEqual(raw, "테스트 전사 문장입니다.")
            self.assertEqual(corrected, "테스트 전사 문장입니다.")
            self.assertIn("Run ID:", status)
            self.assertIn("cer_normalized_no_space", metrics)
            self.assertIsInstance(edits, list)
            self.assertFalse(preprocess["applied"])
            self.assertEqual(preprocessed_audio, str(audio))
            self.assertEqual(preprocessed_audio_html, "")
            self.assertEqual(servers, [])
            self.assertIn("diff", diff.lower())
            self.assertIn("available", gpu_status)

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

    def test_preprocessed_audio_preview_applies_noise_without_command(self):
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
                    "RNNoise",
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
                    "RNNoise",
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
            )
        self.assertEqual(raw, "긴 오디오 테스트 문장입니다.")
        self.assertEqual(corrected, "긴 오디오 테스트 문장입니다.")
        self.assertEqual(preprocessed_audio, str(audio))
        self.assertEqual(preprocessed_audio_html, "")
        self.assertFalse(preprocess["applied"])
        self.assertIn("Run ID:", status)
        self.assertIn("available", gpu_status)


def _write_pcm16_wav(path: Path, samples):
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))


if __name__ == "__main__":
    unittest.main()
