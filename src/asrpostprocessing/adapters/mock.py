from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.schemas import CorrectionResult, Edit, RAGContext, SearchResult, TranscriptResult


class MockASRAdapter:
    def transcribe(self, audio_path: str, config: ExperimentConfig, keyword_instruction: str = "") -> TranscriptResult:
        path = Path(audio_path)
        sidecar = path.with_suffix(".txt")
        if path.suffix.lower() == ".txt" and path.exists():
            text = path.read_text(encoding="utf-8").strip()
        elif sidecar.exists():
            text = sidecar.read_text(encoding="utf-8").strip()
        else:
            text = config.mock_transcript
        return TranscriptResult(
            language=config.language,
            text=text,
            metadata={"backend": "mock", "audio_path": audio_path, "keyword_instruction": keyword_instruction},
        )


class MockPostProcessAdapter:
    REPLACEMENTS: List[Tuple[str, str, str]] = []

    def correct(
        self,
        chunk_text: str,
        config: ExperimentConfig,
        contexts: Iterable[RAGContext],
        search_results: Iterable[SearchResult],
    ) -> CorrectionResult:
        corrected = chunk_text
        edits: List[Edit] = []
        context_text = "\n".join(context.text for context in contexts)
        allowed_terms = set(keyword.lower() for keyword in config.keywords)
        allowed_blob = " ".join(allowed_terms) + "\n" + context_text.lower()
        for before, after, reason in self.REPLACEMENTS:
            if before not in corrected:
                continue
            if config.postprocess_strength < 0.25 and after.lower() not in allowed_blob:
                continue
            corrected = corrected.replace(before, after)
            edits.append(Edit(before=before, after=after, reason=reason, confidence=0.9))
        return CorrectionResult(
            corrected_text=corrected,
            edits=edits,
            risk="low" if edits else "unchanged",
            used_context_ids=[context.context_id for context in contexts],
            metadata={"backend": "mock", "search_results": [result.to_dict() for result in search_results]},
        )
