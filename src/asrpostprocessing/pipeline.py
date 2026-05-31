from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters import build_asr_adapter, build_postprocess_adapter
from .chunking import chunk_segments, chunk_text
from .config import ExperimentConfig
from .keyword_bias import build_keyword_bias_instruction
from .logging import RunLogger, make_run_id
from .metrics import evaluate_transcripts
from .preprocess import preprocess_audio
from .rag import build_rag_index
from .schemas import CorrectionResult, Edit, MetricsResult, RAGContext, SearchResult, TranscriptResult
from .search import CachedSearchProvider
from .text import make_diff_html, merge_overlapping_texts


@dataclass
class PipelineOutput:
    run_id: str
    raw: TranscriptResult
    correction: CorrectionResult
    metrics: MetricsResult
    diff_html: str
    output_dir: str
    artifacts: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "raw": self.raw.to_dict(),
            "correction": self.correction.to_dict(),
            "metrics": self.metrics.to_dict(),
            "output_dir": self.output_dir,
            "artifacts": self.artifacts,
        }


class PipelineRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(
        self,
        audio_path: str,
        reference_text: Optional[str] = None,
        run_id: Optional[str] = None,
        rag_inline_text: str = "",
    ) -> PipelineOutput:
        started = time.time()
        run_id = run_id or make_run_id(self.config.run_name or "run")
        if rag_inline_text:
            self.config.rag_inline_text = rag_inline_text

        preprocess_result = preprocess_audio(audio_path, self.config)
        keyword_instruction = ""
        if self.config.enable_keyword_bias:
            keyword_instruction = build_keyword_bias_instruction(self.config.keywords, self.config.keyword_bias_weight)

        asr = build_asr_adapter(self.config)
        raw = asr.transcribe(preprocess_result.audio_path, self.config, keyword_instruction=keyword_instruction)

        correction = self._postprocess(raw)
        latency_ms = (time.time() - started) * 1000.0
        metrics = evaluate_transcripts(reference_text, raw.text, correction.corrected_text, latency_ms=latency_ms)
        diff_html = make_diff_html(raw.text, correction.corrected_text)

        logger = RunLogger(self.config, run_id)
        artifacts = {
            "result": str(logger.write_json("result.json", self._result_payload(raw, correction, metrics, preprocess_result))),
            "metrics": str(logger.write_json("metrics.json", metrics.to_dict())),
            "edits": str(logger.write_edits(correction.edits)),
            "config": str(logger.write_config()),
            "tensorboard_fallback": str(logger.write_tensorboard_metrics(metrics)),
        }
        return PipelineOutput(
            run_id=run_id,
            raw=raw,
            correction=correction,
            metrics=metrics,
            diff_html=diff_html,
            output_dir=str(logger.output_dir),
            artifacts=artifacts,
        )

    def _postprocess(self, raw: TranscriptResult) -> CorrectionResult:
        if not self.config.enable_llm_postprocess:
            return CorrectionResult(corrected_text=raw.text, risk="unchanged", metadata={"reason": "postprocess_disabled"})

        chunks = chunk_segments(raw.segments) if raw.segments else chunk_text(raw.text, self.config.chunk_max_chars, self.config.chunk_overlap)
        rag_index = build_rag_index(self.config) if self.config.enable_rag else None
        search_provider = CachedSearchProvider(self.config)
        postprocessor = build_postprocess_adapter(self.config)

        corrected_chunks: List[str] = []
        all_edits: List[Edit] = []
        all_context_ids: List[str] = []
        chunk_metadata: List[Dict[str, Any]] = []
        for chunk in chunks:
            contexts: List[RAGContext] = []
            if rag_index is not None:
                contexts = rag_index.retrieve(chunk.text, top_k=self.config.rag_top_k, strength=self.config.rag_strength)
            query = self._search_query(chunk.text)
            search_results: List[SearchResult] = search_provider.search(query) if self.config.enable_search else []
            result = postprocessor.correct(chunk.text, self.config, contexts, search_results)
            corrected_chunks.append(result.corrected_text)
            all_edits.extend(result.edits)
            all_context_ids.extend(result.used_context_ids)
            chunk_metadata.append({"chunk": chunk.__dict__, "risk": result.risk, "metadata": result.metadata})

        corrected_text = merge_overlapping_texts(corrected_chunks, max_overlap=self.config.chunk_overlap + 40)
        risk = _combine_risk([item["risk"] for item in chunk_metadata])
        return CorrectionResult(
            corrected_text=corrected_text,
            edits=all_edits,
            risk=risk,
            used_context_ids=sorted(set(all_context_ids)),
            metadata={"chunks": chunk_metadata},
        )

    def _search_query(self, chunk_text: str) -> str:
        keywords = " ".join(self.config.keywords[:8])
        return " ".join(part for part in [keywords, chunk_text[:300]] if part).strip()

    def _result_payload(self, raw: TranscriptResult, correction: CorrectionResult, metrics: MetricsResult, preprocess_result) -> Dict[str, Any]:
        return {
            "raw": raw.to_dict(),
            "correction": correction.to_dict(),
            "metrics": metrics.to_dict(),
            "preprocess": {
                "audio_path": preprocess_result.audio_path,
                "applied": preprocess_result.applied,
                "warnings": preprocess_result.warnings,
                "metadata": preprocess_result.metadata,
            },
        }


def read_reference(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8").strip()


def _combine_risk(risks: List[str]) -> str:
    if "high" in risks:
        return "high"
    if "medium" in risks:
        return "medium"
    if "low" in risks:
        return "low"
    if risks and all(risk == "unchanged" for risk in risks):
        return "unchanged"
    return "unknown"
