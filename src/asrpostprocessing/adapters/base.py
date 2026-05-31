from __future__ import annotations

from typing import Iterable, Protocol

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.schemas import CorrectionResult, RAGContext, SearchResult, TranscriptResult


class ASRAdapter(Protocol):
    def transcribe(self, audio_path: str, config: ExperimentConfig, keyword_instruction: str = "") -> TranscriptResult:
        ...


class PostProcessAdapter(Protocol):
    def correct(
        self,
        chunk_text: str,
        config: ExperimentConfig,
        contexts: Iterable[RAGContext],
        search_results: Iterable[SearchResult],
    ) -> CorrectionResult:
        ...
