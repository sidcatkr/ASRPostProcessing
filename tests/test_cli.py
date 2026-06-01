import contextlib
import io
import json
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
            return Path("compare.json")

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("asrpostprocessing.cli.run_asr_quality_compare", side_effect=fake_compare):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(["asr-quality", "--audio", "audio.wav", "--asr-backend", "mock"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip(), "compare.json")
        self.assertIn("backend warning", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
