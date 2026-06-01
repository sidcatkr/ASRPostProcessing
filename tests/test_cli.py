import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asrpostprocessing.cli import main


class _FakeOutput:
    def to_dict(self):
        return {"ok": True}


class CliTest(unittest.TestCase):
    def test_run_stdout_remains_json_when_backend_logs_to_stdout(self):
        def fake_run(**_kwargs):
            print("backend warning")
            return _FakeOutput()

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("asrpostprocessing.cli.PipelineRunner") as runner_cls:
            runner_cls.return_value.run.side_effect = fake_run
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(["run", "--audio", "audio.wav", "--asr-backend", "mock", "--post-backend", "mock"])

        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {"ok": True})
        self.assertIn("backend warning", stderr.getvalue())

    def test_asr_quality_stdout_remains_path_when_backend_logs_to_stdout(self):
        def fake_compare(**_kwargs):
            print("backend warning")
            self.assertEqual(_kwargs["sample_start_s"], [0.0, 30.0])
            return Path("compare.json")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("asrpostprocessing.cli.run_asr_quality_compare", side_effect=fake_compare):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(
                    [
                        "asr-quality",
                        "--audio",
                        "audio.wav",
                        "--asr-backend",
                        "mock",
                        "--sample-seconds",
                        "10",
                        "--sample-start-s",
                        "0",
                        "--sample-start-s",
                        "30",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip(), "compare.json")
        self.assertIn("backend warning", stderr.getvalue())

    def test_transcript_quality_writes_report_for_existing_text_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.txt"
            corrected = Path(tmp) / "corrected.txt"
            output = Path(tmp) / "quality.json"
            raw.write_text("앞 문장 language None<asr_text> 如果测试新闻，然后示例文本。 다음 문장", encoding="utf-8")
            corrected.write_text("앞 문장 다음 문장", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(
                    [
                        "transcript-quality",
                        "--raw",
                        str(raw),
                        "--corrected",
                        str(corrected),
                        "--output",
                        str(output),
                    ]
                )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue().strip(), str(output))
            self.assertEqual(payload["asr_quality"]["text_artifacts"]["asr_text_tag_count"], 1)
            self.assertTrue(payload["asr_quality"]["text_artifacts"]["non_korean_cjk_drift_candidate"])
            self.assertEqual(payload["correction_quality"]["artifacts"]["corrected"]["han_char_count"], 0)

    def test_transcript_quality_prints_json_without_output_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.txt"
            raw.write_text("원문", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["transcript-quality", "--raw", str(raw)])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["files"]["raw"], str(raw))
            self.assertIn("asr_quality", payload)


if __name__ == "__main__":
    unittest.main()
