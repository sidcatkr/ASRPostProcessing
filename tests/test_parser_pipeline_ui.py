import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.correction_parser import parse_correction_response
from asrpostprocessing.model_server import ModelServerStatus
from asrpostprocessing.pipeline import PipelineRunner
from asrpostprocessing.ui import run_from_ui


class ParserPipelineUiTest(unittest.TestCase):
    def test_parse_correction_json(self):
        payload = '{"corrected_text":"Boolean","edits":[{"before":"불련","after":"Boolean","reason":"term","confidence":0.8}],"risk":"low","used_context_ids":["ctx1"]}'
        result = parse_correction_response(payload, "불련")
        self.assertEqual(result.corrected_text, "Boolean")
        self.assertEqual(result.edits[0].before, "불련")
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
            output = PipelineRunner(config).run(str(audio), reference_text="Claude Code로 for문 작성 보조", run_id="test-run")
            self.assertIn("Claude Code", output.correction.corrected_text)
            self.assertTrue((Path(output.output_dir) / "result.json").exists())
            self.assertTrue((Path(output.output_dir) / "metrics.json").exists())
            self.assertTrue((Path(output.output_dir) / "preprocess.json").exists())
            self.assertTrue((Path(tmp) / "runs" / "test-run" / "metrics.tsv").exists())

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
                output = PipelineRunner(config).run(str(audio), reference_text="Claude Code로 for문 작성 보조")
            self.assertEqual(calls, [("ensure", "asr"), ("stop", "asr"), ("ensure", "post"), ("stop", "post")])
            self.assertEqual([item["status"] for item in output.server_statuses], ["ready", "stopped", "ready", "stopped"])

    def test_ui_event_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"mock")
            raw, corrected, diff, metrics, edits, preprocess, servers, status = run_from_ui(
                str(audio),
                "Claude Code로 for문 작성 보조",
                None,
                True,
                0.5,
                "Claude Code, for문",
                False,
                "none",
                0.0,
                False,
                0.0,
                -20.0,
                "",
                "",
                "",
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
            self.assertIn("클러드 코드", raw)
            self.assertIn("Claude Code", corrected)
            self.assertIn("Run ID:", status)
            self.assertIn("cer_normalized_no_space", metrics)
            self.assertIsInstance(edits, list)
            self.assertFalse(preprocess["applied"])
            self.assertEqual(servers, [])
            self.assertIn("diff", diff.lower())

    def test_ui_vllm_failure_has_actionable_hint(self):
        raw, corrected, diff, metrics, edits, preprocess, servers, status = run_from_ui(
            "missing.wav",
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
            "",
            "",
            "",
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
        self.assertEqual(servers, [])
        self.assertIn("Run failed:", status)
        self.assertIn("For UI-only testing", status)


if __name__ == "__main__":
    unittest.main()
