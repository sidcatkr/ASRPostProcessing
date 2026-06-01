import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from asrpostprocessing.adapters.vllm import (
    ASRAudioChunk,
    VLLMChatASRAdapter,
    VLLMOpenAIPostProcessAdapter,
    _asr_request_timeout_s,
    _duration_from_ffmpeg_output,
    _post_chat,
)
from asrpostprocessing.config import ExperimentConfig


class VLLMAdapterTest(unittest.TestCase):
    def test_postprocess_request_caps_tokens_and_disables_vllm_thinking(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"corrected_text":"교정된 문장",'
                            '"edits":[{"before":"원문","after":"교정","confidence":0.9}],'
                            '"risk":"low","used_context_ids":[]}'
                        )
                    }
                }
            ]
        }
        config = ExperimentConfig(post_backend="vllm_openai", post_base_url="http://127.0.0.1:18001/v1")
        with patch("requests.post", return_value=response) as post:
            result = VLLMOpenAIPostProcessAdapter().correct("원문 문장", config, [], [])

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("Do not include reasoning", payload["messages"][1]["content"])
        self.assertEqual(result.corrected_text, "교정된 문장")

    def test_asr_request_splits_long_audio_into_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "chunk0.wav"
            second = Path(tmp) / "chunk1.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            chunks = [
                ASRAudioChunk(path=first, index=0, start_s=0.0, end_s=30.0),
                ASRAudioChunk(path=second, index=1, start_s=30.0, end_s=60.0),
            ]
            payloads = []

            def fake_post(_base_url, payload, _timeout_s, _service_name):
                payloads.append(payload)
                return {"choices": [{"message": {"content": f"chunk {len(payloads)} text"}}]}

            config = ExperimentConfig(asr_backend="vllm_chat", asr_base_url="http://127.0.0.1:18000/v1")
            with patch("asrpostprocessing.adapters.vllm._audio_duration_seconds", return_value=75.0), patch(
                "asrpostprocessing.adapters.vllm._split_audio_for_asr", return_value=chunks
            ), patch("asrpostprocessing.adapters.vllm._post_chat", side_effect=fake_post):
                result = VLLMChatASRAdapter().transcribe(str(Path(tmp) / "source.wav"), config)

        self.assertEqual(result.text, "chunk 1 text\nchunk 2 text")
        self.assertTrue(result.metadata["chunked"])
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[1].start_s, 30.0)
        self.assertEqual(len(payloads), 2)
        self.assertTrue(payloads[0]["messages"][0]["content"][1]["audio_url"]["url"].startswith("data:audio/"))
        self.assertIn(";base64", payloads[0]["messages"][0]["content"][1]["audio_url"]["url"])

    def test_asr_uses_dedicated_timeout(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "transcript"}}]}
        config = ExperimentConfig(request_timeout_s=120.0, asr_request_timeout_s=300.0)
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"audio")
            with patch("requests.post", return_value=response) as post:
                result = VLLMChatASRAdapter()._transcribe_one(str(audio), config)

        self.assertEqual(result.text, "transcript")
        self.assertEqual(post.call_args.kwargs["timeout"], 300.0)
        self.assertEqual(_asr_request_timeout_s(config), 300.0)

    def test_post_chat_includes_response_body_for_bad_request(self):
        response = Mock()
        response.text = '{"error":"Input length exceeds model context"}'
        response.raise_for_status.side_effect = RuntimeError("400 Client Error")
        config = ExperimentConfig(asr_base_url="http://127.0.0.1:18000/v1")
        with patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Input length exceeds model context"):
                _post_chat(config.asr_base_url, {"messages": []}, config.request_timeout_s, "ASR")

    def test_post_chat_gives_actionable_timeout_hint(self):
        config = ExperimentConfig(asr_base_url="http://127.0.0.1:18000/v1")
        with patch("requests.post", side_effect=requests.exceptions.ReadTimeout("read timeout=120.0")):
            with self.assertRaisesRegex(RuntimeError, "asr_chunk_seconds"):
                _post_chat(config.asr_base_url, {"messages": []}, config.request_timeout_s, "ASR")

    def test_ffmpeg_duration_output_is_parsed(self):
        output = "Duration: 01:02:03.45, start: 0.000000, bitrate: 192 kb/s"
        self.assertEqual(_duration_from_ffmpeg_output(output), 3723.45)

    def test_default_asr_chunk_and_timeout_are_conservative_for_rtx5000(self):
        config = ExperimentConfig()
        self.assertEqual(config.asr_chunk_seconds, 15.0)
        self.assertEqual(config.asr_request_timeout_s, 300.0)


if __name__ == "__main__":
    unittest.main()
