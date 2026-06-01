import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from asrpostprocessing.adapters.qwen_asr import QwenASRPackageAdapter
from asrpostprocessing.adapters.vllm import ASRAudioChunk
from asrpostprocessing.config import ExperimentConfig


class _FakeASRResult:
    def __init__(self, text: str, language: str = "Korean"):
        self.text = text
        self.language = language
        self.time_stamps = []


class _FakeQwenModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        return [_FakeASRResult(f"chunk {len(self.calls)} text")]


class QwenASRPackageAdapterTest(unittest.TestCase):
    def test_package_backend_chunks_audio_and_passes_rolling_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "chunk0.wav"
            second = Path(tmp) / "chunk1.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            chunks = [
                ASRAudioChunk(path=first, index=0, start_s=0.0, end_s=30.0, method="fixed"),
                ASRAudioChunk(path=second, index=1, start_s=30.0, end_s=60.0, method="fixed"),
            ]
            model = _FakeQwenModel()
            config = ExperimentConfig(asr_backend="qwen_asr_vllm", asr_chunking_strategy="fixed")
            with patch("asrpostprocessing.adapters.qwen_asr._get_model", return_value=model), patch(
                "asrpostprocessing.adapters.qwen_asr._asr_audio_chunks", return_value=chunks
            ):
                result = QwenASRPackageAdapter("vllm").transcribe(str(Path(tmp) / "source.wav"), config)

        self.assertEqual(result.text, "chunk 1 text\nchunk 2 text")
        self.assertTrue(result.metadata["chunked"])
        self.assertEqual(result.metadata["backend"], "qwen_asr_vllm")
        self.assertEqual(result.metadata["context_chars"], 240)
        self.assertEqual(result.metadata["chunks"][0]["previous_context_chars"], 0)
        self.assertGreater(result.metadata["chunks"][1]["previous_context_chars"], 0)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[1].metadata["previous_context_chars"], len("chunk 1 text"))
        self.assertEqual(len(model.calls), 2)
        self.assertIn("language", model.calls[0])
        self.assertIn("context", model.calls[0])
        self.assertNotIn("Previous transcript context", model.calls[0]["context"])
        self.assertIn("Previous transcript context", model.calls[1]["context"])
        self.assertIn("chunk 1 text", model.calls[1]["context"])


if __name__ == "__main__":
    unittest.main()
