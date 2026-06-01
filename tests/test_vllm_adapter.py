import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from asrpostprocessing.adapters.vllm import (
    ASRAudioChunk,
    VLLMChatASRAdapter,
    VLLMOpenAIPostProcessAdapter,
    _asr_audio_chunks,
    _asr_instruction,
    _asr_request_timeout_s,
    _duration_from_ffmpeg_output,
    _filter_asr_language_drift,
    _parse_asr_text,
    _post_chat,
    _rolling_asr_context,
    _silence_aware_chunk_specs,
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
        config = ExperimentConfig(
            post_backend="vllm_openai",
            post_base_url="http://127.0.0.1:18001/v1",
            keywords=["선행 연구"],
        )
        with patch("requests.post", return_value=response) as post:
            result = VLLMOpenAIPostProcessAdapter().correct("원문 문장", config, [], [])

        payload = post.call_args.kwargs["json"]
        prompt = payload["messages"][1]["content"]
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("Do not include reasoning", prompt)
        self.assertIn("Keyword correction guidance", prompt)
        self.assertIn("서론 연구", prompt)
        self.assertEqual(result.corrected_text, "교정된 문장")

    def test_postprocess_applies_high_strength_keyword_near_miss(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"corrected_text":"서론 연구를 찾아보고 읽어볼 거니까",'
                            '"edits":[],"risk":"unchanged","used_context_ids":[]}'
                        )
                    }
                }
            ]
        }
        config = ExperimentConfig(
            post_backend="vllm_openai",
            post_base_url="http://127.0.0.1:18001/v1",
            keywords=["선행 연구"],
            postprocess_strength=0.9,
        )
        with patch("requests.post", return_value=response):
            result = VLLMOpenAIPostProcessAdapter().correct("서론 연구를 찾아보고 읽어볼 거니까", config, [], [])

        self.assertEqual(result.corrected_text, "선행 연구를 찾아보고 읽어볼 거니까")
        self.assertEqual(result.edits[-1].before, "서론 연구를")
        self.assertEqual(result.edits[-1].after, "선행 연구를")
        self.assertIn("keyword_near_miss_corrections", result.metadata)

    def test_postprocess_applies_keyword_near_miss_variants(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"corrected_text":"서면 연구를 찾아보고 사학 연구를 읽어볼 거니까",'
                            '"edits":[],"risk":"unchanged","used_context_ids":[]}'
                        )
                    }
                }
            ]
        }
        config = ExperimentConfig(
            post_backend="vllm_openai",
            post_base_url="http://127.0.0.1:18001/v1",
            keywords=["선행 연구"],
            postprocess_strength=0.9,
        )
        with patch("requests.post", return_value=response):
            result = VLLMOpenAIPostProcessAdapter().correct("서면 연구를 찾아보고 사학 연구를 읽어볼 거니까", config, [], [])

        self.assertEqual(result.corrected_text, "선행 연구를 찾아보고 선행 연구를 읽어볼 거니까")
        self.assertEqual([edit.before for edit in result.edits[-2:]], ["서면 연구를", "사학 연구를"])

    def test_postprocess_keeps_keyword_near_miss_disabled_at_balanced_strength(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"corrected_text":"서론 연구를 찾아보고 읽어볼 거니까",'
                            '"edits":[],"risk":"unchanged","used_context_ids":[]}'
                        )
                    }
                }
            ]
        }
        config = ExperimentConfig(
            post_backend="vllm_openai",
            post_base_url="http://127.0.0.1:18001/v1",
            keywords=["선행 연구"],
            postprocess_strength=0.5,
        )
        with patch("requests.post", return_value=response):
            result = VLLMOpenAIPostProcessAdapter().correct("서론 연구를 찾아보고 읽어볼 거니까", config, [], [])

        self.assertEqual(result.corrected_text, "서론 연구를 찾아보고 읽어볼 거니까")

    def test_postprocess_keyword_near_miss_rejects_unrelated_context_words(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"corrected_text":"청중 그날 청중이 누구냐",'
                            '"edits":[],"risk":"unchanged","used_context_ids":[]}'
                        )
                    }
                }
            ]
        }
        config = ExperimentConfig(
            post_backend="vllm_openai",
            post_base_url="http://127.0.0.1:18001/v1",
            keywords=["청중 분석"],
            postprocess_strength=0.9,
        )
        with patch("requests.post", return_value=response):
            result = VLLMOpenAIPostProcessAdapter().correct("청중 그날 청중이 누구냐", config, [], [])

        self.assertEqual(result.corrected_text, "청중 그날 청중이 누구냐")
        self.assertNotIn("keyword_near_miss_corrections", result.metadata)

    def test_asr_request_splits_long_audio_into_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "chunk0.wav"
            second = Path(tmp) / "chunk1.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            chunks = [
                ASRAudioChunk(path=first, index=0, start_s=0.0, end_s=30.0, method="fixed"),
                ASRAudioChunk(path=second, index=1, start_s=30.0, end_s=60.0, method="fixed"),
            ]
            payloads = []

            def fake_post(_base_url, payload, _timeout_s, _service_name):
                payloads.append(payload)
                return {"choices": [{"message": {"content": f"chunk {len(payloads)} text"}}]}

            config = ExperimentConfig(
                asr_backend="vllm_chat",
                asr_base_url="http://127.0.0.1:18000/v1",
                asr_chunking_strategy="fixed",
                asr_chunk_seconds=30.0,
            )
            with patch("asrpostprocessing.adapters.vllm._audio_duration_seconds", return_value=75.0), patch(
                "asrpostprocessing.adapters.vllm._split_audio_for_asr", return_value=chunks
            ), patch("asrpostprocessing.adapters.vllm._post_chat", side_effect=fake_post):
                result = VLLMChatASRAdapter().transcribe(str(Path(tmp) / "source.wav"), config)

        self.assertEqual(result.text, "chunk 1 text\nchunk 2 text")
        self.assertTrue(result.metadata["chunked"])
        self.assertEqual(result.metadata["chunking_strategy"], "fixed")
        self.assertEqual(result.metadata["chunks"][0]["method"], "fixed")
        self.assertEqual(result.metadata["context_chars"], 240)
        self.assertEqual(result.metadata["chunks"][0]["previous_context_chars"], 0)
        self.assertGreater(result.metadata["chunks"][1]["previous_context_chars"], 0)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[1].start_s, 30.0)
        self.assertEqual(result.segments[1].metadata["previous_context_chars"], len("chunk 1 text"))
        self.assertEqual(len(payloads), 2)
        self.assertTrue(payloads[0]["messages"][0]["content"][1]["audio_url"]["url"].startswith("data:audio/"))
        self.assertIn(";base64", payloads[0]["messages"][0]["content"][1]["audio_url"]["url"])
        first_instruction = payloads[0]["messages"][0]["content"][0]["text"]
        second_instruction = payloads[1]["messages"][0]["content"][0]["text"]
        self.assertNotIn("Previous transcript context", first_instruction)
        self.assertIn("Previous transcript context", second_instruction)
        self.assertIn("chunk 1 text", second_instruction)

    def test_asr_request_skips_empty_qwen_artifact_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "chunk0.wav"
            second = Path(tmp) / "chunk1.wav"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            chunks = [
                ASRAudioChunk(path=first, index=0, start_s=0.0, end_s=30.0, method="fixed"),
                ASRAudioChunk(path=second, index=1, start_s=30.0, end_s=60.0, method="fixed"),
            ]
            responses = iter(["강의 전사", "language None<asr_text>"])

            def fake_post(_base_url, _payload, _timeout_s, _service_name):
                return {"choices": [{"message": {"content": next(responses)}}]}

            config = ExperimentConfig(asr_backend="vllm_chat", asr_chunking_strategy="fixed", asr_chunk_seconds=30.0)
            with patch("asrpostprocessing.adapters.vllm._audio_duration_seconds", return_value=75.0), patch(
                "asrpostprocessing.adapters.vllm._split_audio_for_asr", return_value=chunks
            ), patch("asrpostprocessing.adapters.vllm._post_chat", side_effect=fake_post):
                result = VLLMChatASRAdapter().transcribe(str(Path(tmp) / "source.wav"), config)

        self.assertEqual(result.text, "강의 전사")
        self.assertEqual(result.segments[1].text, "")
        self.assertEqual(result.metadata["chunks"][1]["text_chars"], 0)

    def test_parse_asr_text_treats_empty_qwen_marker_as_empty(self):
        parsed = _parse_asr_text("language None<asr_text>")

        self.assertEqual(parsed["text"], "")
        self.assertIn(parsed.get("language"), (None, ""))

    def test_parse_asr_text_preserves_text_around_midstream_none_marker(self):
        fake_qwen_asr = types.SimpleNamespace(
            parse_asr_output=lambda _text: {"language": "Korean", "text": "다음 곡 잡자."}
        )

        with patch.dict("sys.modules", {"qwen_asr": fake_qwen_asr}):
            parsed = _parse_asr_text("쉬다 와요. language None<asr_text> 假如我查新闻，然后卡特总统。 다음 곡 잡자.")
        text, reason = _filter_asr_language_drift(parsed["text"], "ko")

        self.assertEqual(text, "쉬다 와요. 다음 곡 잡자.")
        self.assertEqual(reason, "inline_cjk_drift_removed")
        self.assertTrue(parsed["raw_marker_mix_preserved"])

    def test_korean_asr_filters_cjk_language_drift_chunk(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "假如我查新闻，然后卡特总统。"}}]}
        config = ExperimentConfig(language="ko")
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"audio")
            with patch("requests.post", return_value=response):
                result = VLLMChatASRAdapter()._transcribe_one(str(audio), config)

        self.assertEqual(result.text, "")
        self.assertEqual(result.metadata["parsed"]["filtered_reason"], "non_korean_cjk_drift")

    def test_korean_asr_removes_inline_cjk_drift_before_postprocess(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "쉬다 와요. language None<asr_text> 假如我查新闻，然后卡特总统。 다음 곡 잡자."
                    }
                }
            ]
        }
        config = ExperimentConfig(language="ko")
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"audio")
            with patch("requests.post", return_value=response):
                result = VLLMChatASRAdapter()._transcribe_one(str(audio), config)

        self.assertEqual(result.text, "쉬다 와요. 다음 곡 잡자.")
        self.assertEqual(result.metadata["parsed"]["filtered_reason"], "inline_cjk_drift_removed")

    def test_korean_asr_keeps_short_hanja_terms(self):
        text, reason = _filter_asr_language_drift("공자 曰 다음 문장", "ko")

        self.assertEqual(text, "공자 曰 다음 문장")
        self.assertEqual(reason, "")

    def test_asr_instruction_discourages_language_drift_and_artifacts(self):
        instruction = _asr_instruction("", "")

        self.assertIn("Transcribe only the Korean speech", instruction)
        self.assertIn("Do not translate", instruction)
        self.assertIn("empty transcript", instruction)

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

    def test_default_asr_chunking_prefers_longer_context_for_quality(self):
        config = ExperimentConfig()
        self.assertEqual(config.asr_chunking_strategy, "silence")
        self.assertEqual(config.asr_chunk_seconds, 120.0)
        self.assertEqual(config.asr_chunk_padding_seconds, 0.5)
        self.assertEqual(config.asr_request_timeout_s, 300.0)
        self.assertEqual(config.asr_context_chars, 240)

    def test_rolling_asr_context_uses_recent_suffix(self):
        context = _rolling_asr_context(["abcdef", "ghijkl"], 6)

        self.assertEqual(context, "ghijkl")

    def test_none_chunking_strategy_keeps_long_audio_whole(self):
        config = ExperimentConfig(asr_chunking_strategy="none")
        with patch("asrpostprocessing.adapters.vllm._audio_duration_seconds", return_value=600.0), patch(
            "asrpostprocessing.adapters.vllm._split_audio_for_asr_silence"
        ) as silence_split, patch("asrpostprocessing.adapters.vllm._split_audio_for_asr") as fixed_split:
            chunks = _asr_audio_chunks("long.wav", config)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].method, "none")
        self.assertEqual(chunks[0].end_s, 600.0)
        silence_split.assert_not_called()
        fixed_split.assert_not_called()

    def test_silence_chunking_falls_back_to_fixed_segments(self):
        fixed_chunks = [
            ASRAudioChunk(path=Path("chunk0.wav"), index=0, start_s=0.0, end_s=30.0, method="fixed"),
            ASRAudioChunk(path=Path("chunk1.wav"), index=1, start_s=30.0, end_s=60.0, method="fixed"),
        ]
        config = ExperimentConfig(asr_chunking_strategy="silence", asr_chunk_seconds=30.0)
        with patch("asrpostprocessing.adapters.vllm._audio_duration_seconds", return_value=75.0), patch(
            "asrpostprocessing.adapters.vllm._split_audio_for_asr_silence", return_value=[]
        ) as silence_split, patch("asrpostprocessing.adapters.vllm._split_audio_for_asr", return_value=fixed_chunks) as fixed_split:
            chunks = _asr_audio_chunks("long.wav", config)

        self.assertEqual(chunks, fixed_chunks)
        silence_split.assert_called_once()
        fixed_split.assert_called_once()

    def test_silence_aware_chunk_specs_split_on_silence_and_pad_without_overlap(self):
        specs = _silence_aware_chunk_specs([(20.0, 21.0)], duration=50.0, chunk_seconds=30.0, padding_seconds=0.5)

        self.assertEqual(len(specs), 2)
        self.assertAlmostEqual(specs[0].start_s, 0.0)
        self.assertAlmostEqual(specs[0].end_s, 20.5)
        self.assertAlmostEqual(specs[1].start_s, 20.5)
        self.assertAlmostEqual(specs[1].end_s, 50.0)
        self.assertEqual(specs[0].speech_end_s, 20.0)
        self.assertEqual(specs[1].speech_start_s, 21.0)

    def test_silence_aware_chunk_specs_require_detected_silence(self):
        self.assertEqual(_silence_aware_chunk_specs([], duration=50.0, chunk_seconds=30.0, padding_seconds=0.5), [])


if __name__ == "__main__":
    unittest.main()
