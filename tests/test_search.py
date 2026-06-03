import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.search import CachedSearchProvider, clear_search_memory_cache


class SearchTest(unittest.TestCase):
    def tearDown(self):
        clear_search_memory_cache()

    def test_duckduckgo_provider_caches_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(enable_search=True, search_provider="duckduckgo", search_cache_dir=tmp)
            provider = CachedSearchProvider(config)
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "Heading": "ExampleTerm",
                "AbstractText": "ExampleTerm is a cached search result.",
                "AbstractURL": "https://example.test/example-term",
                "RelatedTopics": [],
            }
            with patch("requests.get", return_value=response) as get:
                first = provider.search("ExampleTerm")
                second = provider.search("ExampleTerm")
            self.assertEqual(len(first), 1)
            self.assertEqual(first[0].source, "duckduckgo")
            self.assertEqual(second[0].snippet, first[0].snippet)
            self.assertEqual(get.call_count, 1)
            self.assertTrue(list(Path(tmp).glob("*.json")))

    def test_invalid_cache_is_ignored_and_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(enable_search=True, search_provider="duckduckgo", search_cache_dir=tmp)
            provider = CachedSearchProvider(config)
            cache_path = provider._cache_path("ExampleTerm")
            cache_path.write_text('{"query":"ExampleTerm","results":[]}}', encoding="utf-8")
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "Heading": "ExampleTerm",
                "AbstractText": "Recovered cache result.",
                "AbstractURL": "https://example.test/example-term",
                "RelatedTopics": [],
            }
            with patch("requests.get", return_value=response) as get:
                result = provider.search("ExampleTerm")

            self.assertEqual(get.call_count, 1)
            self.assertEqual(result[0].snippet, "Recovered cache result.")
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["results"][0]["snippet"], "Recovered cache result.")

    def test_memory_cache_avoids_reloading_same_disk_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ExperimentConfig(enable_search=True, search_provider="duckduckgo", search_cache_dir=tmp)
            provider = CachedSearchProvider(config)
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "Heading": "ExampleTerm",
                "AbstractText": "Memory cached search result.",
                "AbstractURL": "https://example.test/example-term",
                "RelatedTopics": [],
            }
            with patch("requests.get", return_value=response):
                provider.search("ExampleTerm")
            with patch("asrpostprocessing.search.read_json") as read_json_mock, patch("requests.get") as get:
                result = provider.search("ExampleTerm")
            self.assertEqual(result[0].snippet, "Memory cached search result.")
            read_json_mock.assert_not_called()
            get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
