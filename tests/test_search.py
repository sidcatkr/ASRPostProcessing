import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.search import CachedSearchProvider


class SearchTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
