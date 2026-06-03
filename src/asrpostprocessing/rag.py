from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import ExperimentConfig, clamp01
from .schemas import RAGContext


TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-.]+|[가-힣]+")
SUPPORTED_RAG_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".pdf"}
_RAG_INDEX_CACHE_LOCK = threading.RLock()
_RAG_INDEX_CACHE: Dict[Tuple[str, str, str], object] = {}
_RAG_DOCUMENT_CACHE: Dict[Tuple[Tuple[Tuple[str, bool, int, int], ...], str, Tuple[str, ...]], List[RAGDocument]] = {}


@dataclass
class RAGDocument:
    doc_id: str
    text: str
    source: str


class LexicalRAGIndex:
    def __init__(self, documents: Iterable[RAGDocument]):
        self.documents = list(documents)
        self._vectors = [_term_counts(doc.text) for doc in self.documents]
        self._retrieve_lock = threading.RLock()
        self._retrieve_cache: Dict[Tuple[str, int, float], List[RAGContext]] = {}

    def retrieve(self, query: str, top_k: int = 5, strength: float = 0.5) -> List[RAGContext]:
        cache_key = _retrieve_cache_key(query, top_k, strength)
        with self._retrieve_lock:
            cached = self._retrieve_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        query_vector = _term_counts(query)
        if not query_vector:
            return []
        threshold = 0.02 + 0.18 * (1.0 - clamp01(strength))
        scored = []
        for document, vector in zip(self.documents, self._vectors):
            score = _cosine(query_vector, vector)
            if score >= threshold:
                scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        contexts = [
            RAGContext(context_id=document.doc_id, text=document.text, score=score, source=document.source)
            for score, document in scored[:top_k]
        ]
        with self._retrieve_lock:
            self._retrieve_cache[cache_key] = list(contexts)
        return contexts


class FaissRAGIndex:
    def __init__(self, documents: List[RAGDocument], model_name: str):
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self._lock = threading.RLock()
        self._retrieve_cache: Dict[Tuple[str, int, float], List[RAGContext]] = {}
        texts = [document.text for document in documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5, strength: float = 0.5) -> List[RAGContext]:
        if not self.documents:
            return []
        cache_key = _retrieve_cache_key(query, top_k, strength)
        with self._lock:
            cached = self._retrieve_cache.get(cache_key)
            if cached is not None:
                return list(cached)
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            threshold = 0.15 + 0.35 * (1.0 - clamp01(strength))
            contexts: List[RAGContext] = []
            for score, index in zip(scores[0], indices[0]):
                if index < 0 or float(score) < threshold:
                    continue
                document = self.documents[int(index)]
                contexts.append(RAGContext(context_id=document.doc_id, text=document.text, score=float(score), source=document.source))
            self._retrieve_cache[cache_key] = list(contexts)
            return contexts


def build_rag_index(config: ExperimentConfig, extra_texts: Optional[Iterable[str]] = None):
    documents = load_rag_documents(config.rag_files, config.rag_inline_text, extra_texts=extra_texts)
    cache_key = _rag_index_cache_key(config, documents)
    with _RAG_INDEX_CACHE_LOCK:
        cached = _RAG_INDEX_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if config.rag_embedding_backend == "faiss":
            faiss_index = _try_build_faiss_index(documents, config)
            index = faiss_index if faiss_index is not None else LexicalRAGIndex(documents)
        else:
            index = LexicalRAGIndex(documents)
        _RAG_INDEX_CACHE[cache_key] = index
        return index


def clear_rag_index_cache() -> None:
    with _RAG_INDEX_CACHE_LOCK:
        _RAG_INDEX_CACHE.clear()
        _RAG_DOCUMENT_CACHE.clear()


def load_rag_documents(paths: Iterable[str], inline_text: str = "", extra_texts: Optional[Iterable[str]] = None) -> List[RAGDocument]:
    path_list = [str(path) for path in (paths or [])]
    extra_list = [text for text in (extra_texts or []) if text]
    cache_key = _rag_document_cache_key(path_list, inline_text, extra_list)
    with _RAG_INDEX_CACHE_LOCK:
        cached = _RAG_DOCUMENT_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)
    documents: List[RAGDocument] = []
    for path in path_list:
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_file():
            continue
        text = read_rag_file(path_obj)
        documents.extend(_split_document(text, source=str(path_obj)))
    if inline_text:
        documents.extend(_split_document(inline_text, source="inline"))
    for index, text in enumerate(extra_list):
        if text:
            documents.extend(_split_document(text, source=f"extra:{index}"))
    with _RAG_INDEX_CACHE_LOCK:
        _RAG_DOCUMENT_CACHE[cache_key] = list(documents)
    return documents


def read_rag_file(path: str | Path) -> str:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix not in SUPPORTED_RAG_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_RAG_EXTENSIONS))
        raise ValueError(
            f"Unsupported RAG file format '{suffix or '(none)'}' for {path_obj}. Supported formats: {supported}"
        )
    if suffix == ".pdf":
        return _read_pdf(path_obj)
    return path_obj.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF RAG files require pypdf. Install the project with `pip install -e '.[rag]'`.") from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append(f"[page {page_number}]\n{text}")
    return "\n\n".join(pages)


def _split_document(text: str, source: str, max_chars: int = 900) -> List[RAGDocument]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    chunks: List[RAGDocument] = []
    for paragraph in paragraphs or [text.strip()]:
        if not paragraph:
            continue
        start = 0
        while start < len(paragraph):
            chunk = paragraph[start : start + max_chars].strip()
            if chunk:
                digest = hashlib.sha1(f"{source}:{start}:{chunk}".encode("utf-8")).hexdigest()[:12]
                chunks.append(RAGDocument(doc_id=digest, text=chunk, source=source))
            start += max_chars
    return chunks


def _term_counts(text: str) -> dict:
    counts = {}
    for token in TOKEN_RE.findall((text or "").lower()):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _cosine(left: dict, right: dict) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _try_build_faiss_index(documents: List[RAGDocument], config: ExperimentConfig):
    if not documents:
        return LexicalRAGIndex([])
    try:
        return FaissRAGIndex(documents, config.rag_embedding_model)
    except Exception:
        return None


def _rag_index_cache_key(config: ExperimentConfig, documents: List[RAGDocument]) -> Tuple[str, str, str]:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.doc_id.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(document.source.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(document.text.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return (config.rag_embedding_backend, config.rag_embedding_model, digest.hexdigest())


def _retrieve_cache_key(query: str, top_k: int, strength: float) -> Tuple[str, int, float]:
    return (query or "", max(1, int(top_k)), round(clamp01(strength), 4))


def _rag_document_cache_key(paths: List[str], inline_text: str, extra_texts: List[str]) -> Tuple[Tuple[Tuple[str, bool, int, int], ...], str, Tuple[str, ...]]:
    path_parts = []
    for path in paths:
        path_obj = Path(path)
        try:
            stat = path_obj.stat()
            path_parts.append((str(path_obj), path_obj.is_file(), int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            path_parts.append((str(path_obj), False, 0, 0))
    return (tuple(path_parts), inline_text or "", tuple(extra_texts))
