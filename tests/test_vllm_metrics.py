import unittest

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.vllm_metrics import (
    diff_vllm_metrics,
    parse_prometheus_metrics,
    summarize_vllm_counters,
    vllm_metrics_endpoint_pool,
)


class VLLMMetricsTest(unittest.TestCase):
    def test_parse_and_summarize_prometheus_counters(self):
        text = """
        # HELP vllm:num_preemptions_total Number of preemptions.
        vllm:num_preemptions_total{model_name="asr"} 2
        vllm:prompt_tokens_total{model_name="asr"} 10
        vllm:generation_tokens_total{model_name="asr"} 7
        vllm:request_success_total{model_name="asr"} 3
        vllm:e2e_request_latency_seconds_bucket{le="1.0"} 99
        vllm:prompt_tokens_total{model_name="post"} 5
        """

        metrics = parse_prometheus_metrics(text)
        counters = summarize_vllm_counters(metrics)

        self.assertNotIn("vllm:e2e_request_latency_seconds_bucket", metrics)
        self.assertEqual(counters["preemption_count"], 2.0)
        self.assertEqual(counters["prompt_tokens"], 15.0)
        self.assertEqual(counters["generation_tokens"], 7.0)
        self.assertEqual(counters["request_success_count"], 3.0)

    def test_diff_vllm_metrics_sums_available_endpoint_deltas(self):
        before = {
            "timestamp": 1.0,
            "endpoints": {
                "http://127.0.0.1:18000/v1": {
                    "available": True,
                    "counters": {
                        "preemption_count": 1,
                        "prompt_tokens": 10,
                        "generation_tokens": 20,
                        "request_success_count": 2,
                    },
                },
                "http://127.0.0.1:18001/v1": {
                    "available": False,
                    "error": "connection refused",
                    "counters": {},
                },
            },
        }
        after = {
            "timestamp": 2.0,
            "endpoints": {
                "http://127.0.0.1:18000/v1": {
                    "available": True,
                    "counters": {
                        "preemption_count": 4,
                        "prompt_tokens": 16,
                        "generation_tokens": 35,
                        "request_success_count": 5,
                    },
                },
                "http://127.0.0.1:18001/v1": {
                    "available": False,
                    "error": "connection refused",
                    "counters": {},
                },
            },
        }

        result = diff_vllm_metrics(before, after)

        self.assertTrue(result["available"])
        self.assertEqual(result["delta"]["preemption_count"], 3.0)
        self.assertEqual(result["delta"]["prompt_tokens"], 6.0)
        self.assertEqual(result["delta"]["generation_tokens"], 15.0)
        self.assertEqual(result["delta"]["total_tokens"], 21.0)
        self.assertEqual(result["delta"]["request_success_count"], 3.0)
        self.assertEqual(result["endpoint_deltas"]["http://127.0.0.1:18001/v1"]["available"], False)

    def test_endpoint_pool_uses_lanes_lists_and_primary_urls_without_duplicates(self):
        config = ExperimentConfig(
            asr_backend="vllm_chat",
            post_backend="vllm_openai",
            asr_base_url="http://127.0.0.1:18000/v1",
            post_base_url="http://127.0.0.1:18001/v1",
            asr_base_urls=["http://127.0.0.1:18002/v1", "http://127.0.0.1:18000/v1"],
            post_base_urls=["http://127.0.0.1:18003/v1"],
            pipeline_lanes=[
                {
                    "name": "lane_a",
                    "asr_model": "Qwen/Qwen3-ASR-1.7B",
                    "post_model": "Qwen/Qwen3.5-9B",
                    "asr_base_url": "http://127.0.0.1:18000/v1",
                    "post_base_url": "http://127.0.0.1:18001/v1",
                },
                {
                    "name": "lane_b",
                    "asr_model": "Qwen/Qwen3-ASR-1.7B",
                    "post_model": "Qwen/Qwen3.5-9B",
                    "asr_base_url": "http://127.0.0.1:18002/v1",
                    "post_base_url": "http://127.0.0.1:18003/v1",
                },
            ],
        )

        self.assertEqual(
            vllm_metrics_endpoint_pool(config),
            [
                "http://127.0.0.1:18000/v1",
                "http://127.0.0.1:18002/v1",
                "http://127.0.0.1:18001/v1",
                "http://127.0.0.1:18003/v1",
            ],
        )


if __name__ == "__main__":
    unittest.main()
