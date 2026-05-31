import unittest

from asrpostprocessing.chunking import chunk_text
from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.keyword_bias import build_keyword_bias_instruction, quantize_keyword_weight
from asrpostprocessing.rag import build_rag_index


class ChunkingKeywordRagTest(unittest.TestCase):
    def test_chunk_overlap(self):
        text = "가" * 150 + ". " + "나" * 150 + ". " + "다" * 150
        chunks = chunk_text(text, max_chars=180, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertLess(chunks[1].start_char, chunks[0].end_char)
        self.assertLessEqual(chunks[0].end_char - chunks[0].start_char, 180)

    def test_keyword_weight_quantization_and_prompt_growth(self):
        self.assertEqual(quantize_keyword_weight(0.62), 0.5)
        weak = build_keyword_bias_instruction(["Claude Code"], 0.25)
        strong = build_keyword_bias_instruction(["Claude Code"], 1.0)
        self.assertIn("Claude Code", weak)
        self.assertGreater(len(strong), len(weak))

    def test_rag_retrieval_filters_contexts(self):
        config = ExperimentConfig(enable_rag=True, rag_inline_text="Claude Code는 AI coding agent입니다.\n\n무관한 날씨 문서입니다.", rag_top_k=2)
        index = build_rag_index(config)
        contexts = index.retrieve("클러드 코드 Claude Code", top_k=2, strength=0.5)
        self.assertTrue(contexts)
        self.assertIn("Claude Code", contexts[0].text)


if __name__ == "__main__":
    unittest.main()
