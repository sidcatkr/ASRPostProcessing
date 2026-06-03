import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from asrpostprocessing.chunking import chunk_text
from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.keyword_bias import build_keyword_bias_instruction, quantize_keyword_weight
from asrpostprocessing.rag import SUPPORTED_RAG_EXTENSIONS, build_rag_index, clear_rag_index_cache, load_rag_documents


class ChunkingKeywordRagTest(unittest.TestCase):
    def tearDown(self):
        clear_rag_index_cache()

    def test_chunk_overlap(self):
        text = "가" * 150 + ". " + "나" * 150 + ". " + "다" * 150
        chunks = chunk_text(text, max_chars=180, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertLess(chunks[1].start_char, chunks[0].end_char)
        self.assertLessEqual(chunks[0].end_char - chunks[0].start_char, 180)

    def test_keyword_weight_quantization_and_prompt_growth(self):
        self.assertEqual(quantize_keyword_weight(0.62), 0.5)
        weak = build_keyword_bias_instruction(["AlphaTerm"], 0.25)
        strong = build_keyword_bias_instruction(["AlphaTerm"], 1.0)
        self.assertIn("AlphaTerm", weak)
        self.assertGreater(len(strong), len(weak))

    def test_rag_retrieval_filters_contexts(self):
        config = ExperimentConfig(enable_rag=True, rag_inline_text="AlphaTerm은 프로젝트 용어입니다.\n\n무관한 날씨 문서입니다.", rag_top_k=2)
        index = build_rag_index(config)
        contexts = index.retrieve("알파텀 AlphaTerm", top_k=2, strength=0.5)
        self.assertTrue(contexts)
        self.assertIn("AlphaTerm", contexts[0].text)

    def test_faiss_rag_index_is_cached_for_same_documents(self):
        class FakeFaissIndex:
            builds = 0
            retrieve_calls = 0

            def __init__(self, documents, model_name):
                FakeFaissIndex.builds += 1
                self.documents = documents
                self.model_name = model_name

            def retrieve(self, query, top_k=5, strength=0.5):
                FakeFaissIndex.retrieve_calls += 1
                return []

        config = ExperimentConfig(
            enable_rag=True,
            rag_embedding_backend="faiss",
            rag_embedding_model="fake-embedding",
            rag_inline_text="AlphaTerm은 프로젝트 용어입니다.",
        )
        with patch("asrpostprocessing.rag.FaissRAGIndex", FakeFaissIndex):
            first = build_rag_index(config)
            second = build_rag_index(config)
        self.assertIs(first, second)
        self.assertEqual(FakeFaissIndex.builds, 1)

    def test_lexical_rag_retrieval_is_cached_per_query(self):
        config = ExperimentConfig(
            enable_rag=True,
            rag_embedding_backend="lexical",
            rag_inline_text="AlphaTerm은 프로젝트 용어입니다.\n\n무관한 날씨 문서입니다.",
            rag_top_k=2,
        )
        index = build_rag_index(config)
        with patch("asrpostprocessing.rag._term_counts", wraps=__import__("asrpostprocessing.rag", fromlist=["_term_counts"])._term_counts) as counts:
            first = index.retrieve("AlphaTerm", top_k=2, strength=0.5)
            second = index.retrieve("AlphaTerm", top_k=2, strength=0.5)
        self.assertEqual([context.context_id for context in first], [context.context_id for context in second])
        self.assertEqual(counts.call_count, 1)

    def test_faiss_rag_index_cache_changes_when_documents_change(self):
        class FakeFaissIndex:
            builds = 0

            def __init__(self, documents, model_name):
                FakeFaissIndex.builds += 1
                self.documents = documents
                self.model_name = model_name

        first_config = ExperimentConfig(
            enable_rag=True,
            rag_embedding_backend="faiss",
            rag_embedding_model="fake-embedding",
            rag_inline_text="AlphaTerm은 프로젝트 용어입니다.",
        )
        second_config = ExperimentConfig(
            enable_rag=True,
            rag_embedding_backend="faiss",
            rag_embedding_model="fake-embedding",
            rag_inline_text="BetaTerm은 다른 프로젝트 용어입니다.",
        )
        with patch("asrpostprocessing.rag.FaissRAGIndex", FakeFaissIndex):
            first = build_rag_index(first_config)
            second = build_rag_index(second_config)
        self.assertIsNot(first, second)
        self.assertEqual(FakeFaissIndex.builds, 2)

    def test_rag_loads_text_file_formats(self):
        self.assertIn(".txt", SUPPORTED_RAG_EXTENSIONS)
        self.assertIn(".md", SUPPORTED_RAG_EXTENSIONS)
        self.assertIn(".pdf", SUPPORTED_RAG_EXTENSIONS)

    def test_rag_reads_uploaded_text_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "terms.md"
            path.write_text("AlphaTerm은 프로젝트 용어입니다.", encoding="utf-8")
            documents = load_rag_documents([str(path)])
        self.assertEqual(len(documents), 1)
        self.assertIn("AlphaTerm", documents[0].text)
        self.assertEqual(documents[0].source, str(path))

    def test_rag_document_loading_is_cached_until_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "terms.md"
            path.write_text("AlphaTerm은 프로젝트 용어입니다.", encoding="utf-8")
            first = load_rag_documents([str(path)])
            with patch("asrpostprocessing.rag.read_rag_file") as read_file:
                second = load_rag_documents([str(path)])
            self.assertEqual([doc.doc_id for doc in first], [doc.doc_id for doc in second])
            read_file.assert_not_called()
            path.write_text("BetaTerm은 다른 용어입니다.", encoding="utf-8")
            changed = load_rag_documents([str(path)])
        self.assertNotEqual(first[0].doc_id, changed[0].doc_id)

    def test_rag_reads_pdf_with_pypdf(self):
        class FakePage:
            def extract_text(self):
                return "PDF AlphaTerm context"

        class FakePdfReader:
            def __init__(self, path):
                self.path = path
                self.pages = [FakePage()]

        module = ModuleType("pypdf")
        module.PdfReader = FakePdfReader
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "guide.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            with patch.dict("sys.modules", {"pypdf": module}):
                documents = load_rag_documents([str(path)])
        self.assertEqual(len(documents), 1)
        self.assertIn("[page 1]", documents[0].text)
        self.assertIn("PDF AlphaTerm context", documents[0].text)


if __name__ == "__main__":
    unittest.main()
