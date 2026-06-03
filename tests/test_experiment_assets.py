import unittest
from pathlib import Path


ASSET_ROOT = Path("experiment_assets/audio_references")


class ExperimentAssetsTest(unittest.TestCase):
    def test_keyword_bias_does_not_include_wrong_english_proper_noun_transliterations(self):
        keyword_files = list((ASSET_ROOT / "keywords").glob("*.txt")) + [Path("experiment_assets/general_keyword_bias_terms.txt")]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in keyword_files)

        self.assertIn("Asteria-9", combined)
        self.assertIn("The Quiet Harbor", combined)
        self.assertNotIn("아스테리아", combined)
        self.assertNotIn("더 콰이어트", combined)

    def test_ten_minute_reference_asset_set_is_available(self):
        reference = ASSET_ROOT / "references" / "ten_minute_mixed.txt"
        rag = ASSET_ROOT / "rag" / "ten_minute_mixed.md"
        keywords = ASSET_ROOT / "keywords" / "ten_minute_mixed.txt"
        manifest = ASSET_ROOT / "manifest.md"

        self.assertGreater(len(reference.read_text(encoding="utf-8")), 6000)
        self.assertIn("Asteria-9", reference.read_text(encoding="utf-8"))
        self.assertIn("Asteria-9", rag.read_text(encoding="utf-8"))
        self.assertIn("Asteria-9", keywords.read_text(encoding="utf-8"))
        self.assertIn("ten_minute_mixed", manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
