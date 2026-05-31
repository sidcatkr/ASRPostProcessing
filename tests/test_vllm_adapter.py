import unittest
from unittest.mock import Mock, patch

from asrpostprocessing.adapters.vllm import VLLMOpenAIPostProcessAdapter
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
                            '{"corrected_text":"클라우드 코드",'
                            '"edits":[{"before":"클러드","after":"클라우드","confidence":0.9}],'
                            '"risk":"low","used_context_ids":[]}'
                        )
                    }
                }
            ]
        }
        config = ExperimentConfig(post_backend="vllm_openai", post_base_url="http://127.0.0.1:18001/v1")
        with patch("requests.post", return_value=response) as post:
            result = VLLMOpenAIPostProcessAdapter().correct("클러드 코드", config, [], [])

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 512)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("Do not include reasoning", payload["messages"][1]["content"])
        self.assertEqual(result.corrected_text, "클라우드 코드")


if __name__ == "__main__":
    unittest.main()
