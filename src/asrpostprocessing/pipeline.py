from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapters import build_asr_adapter, build_postprocess_adapter
from .chunking import chunk_segments, chunk_text
from .config import ExperimentConfig, normalize_model_residency
from .keyword_bias import build_keyword_bias_instruction
from .logging import RunLogger, make_run_id
from .metrics import evaluate_transcripts
from .model_server import ensure_model_servers, stop_model_servers
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
    server_statuses: List[Dict[str, Any]]
    preprocess: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "raw": self.raw.to_dict(),
            "correction": self.correction.to_dict(),
            "metrics": self.metrics.to_dict(),
            "output_dir": self.output_dir,
            "artifacts": self.artifacts,
            "server_statuses": self.server_statuses,
            "preprocess": self.preprocess,
        }


class PipelineRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.model_residency = normalize_model_residency(self.config.model_residency)

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

        server_statuses = self._initial_server_statuses()
        preprocess_result = preprocess_audio(audio_path, self.config)
        keyword_instruction = ""
        if self.config.enable_keyword_bias:
            keyword_instruction = build_keyword_bias_instruction(self.config.keywords, self.config.keyword_bias_weight)

        if self._sequential_model_residency():
            server_statuses.extend(status.to_dict() for status in ensure_model_servers(self.config, names=["asr"]))
        try:
            asr = build_asr_adapter(self.config)
            raw = asr.transcribe(preprocess_result.audio_path, self.config, keyword_instruction=keyword_instruction)
        finally:
            if self._sequential_model_residency():
                server_statuses.extend(self._release_stage_model("asr"))

        correction = self._postprocess(raw, server_statuses)
        latency_ms = (time.time() - started) * 1000.0
        metrics = evaluate_transcripts(reference_text, raw.text, correction.corrected_text, latency_ms=latency_ms)
        diff_html = make_diff_html(raw.text, correction.corrected_text)

        logger = RunLogger(self.config, run_id)
        artifacts = {
            "result": str(logger.write_json("result.json", self._result_payload(raw, correction, metrics, preprocess_result, server_statuses))),
            "preprocess": str(logger.write_json("preprocess.json", preprocess_result.to_dict())),
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
            server_statuses=server_statuses,
            preprocess=preprocess_result.to_dict(),
        )

    def _postprocess(self, raw: TranscriptResult, server_statuses: List[Dict[str, Any]]) -> CorrectionResult:
        if not self.config.enable_llm_postprocess:
            return CorrectionResult(corrected_text=raw.text, risk="unchanged", metadata={"reason": "postprocess_disabled"})

        chunks = chunk_segments(raw.segments) if raw.segments else chunk_text(raw.text, self.config.chunk_max_chars, self.config.chunk_overlap)
        rag_index = build_rag_index(self.config) if self.config.enable_rag else None
        search_provider = CachedSearchProvider(self.config)
        if self._sequential_model_residency():
            server_statuses.extend(status.to_dict() for status in ensure_model_servers(self.config, names=["post"]))
        try:
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
        finally:
            if self._sequential_model_residency():
                server_statuses.extend(self._release_stage_model("post"))

        corrected_text = merge_overlapping_texts(corrected_chunks, max_overlap=self.config.chunk_overlap + 40)
        risk = _combine_risk([item["risk"] for item in chunk_metadata])
        return CorrectionResult(
            corrected_text=corrected_text,
            edits=all_edits,
            risk=risk,
            used_context_ids=sorted(set(all_context_ids)),
            metadata={"chunks": chunk_metadata},
        )

    def _initial_server_statuses(self) -> List[Dict[str, Any]]:
        if self._sequential_model_residency():
            return []
        return [status.to_dict() for status in ensure_model_servers(self.config)]

    def _sequential_model_residency(self) -> bool:
        return self.config.model_residency == "sequential"

    def _release_stage_model(self, name: str) -> List[Dict[str, Any]]:
        statuses: List[Dict[str, Any]] = []
        if self.config.auto_start_model_servers:
            statuses.extend(status.to_dict() for status in stop_model_servers(self.config, names=[name]))
        if name == "asr":
            self._clear_direct_asr_cache()
        return statuses

    def _clear_direct_asr_cache(self) -> None:
        if not (self.config.asr_backend or "").startswith("qwen_asr"):
            return
        try:
            from .adapters.qwen_asr import clear_model_cache

            clear_model_cache()
        except Exception:
            pass

    def _search_query(self, chunk_text: str) -> str:
        keywords = " ".join(self.config.keywords[:8])
        return " ".join(part for part in [keywords, chunk_text[:300]] if part).strip()

    def _result_payload(
        self,
        raw: TranscriptResult,
        correction: CorrectionResult,
        metrics: MetricsResult,
        preprocess_result,
        server_statuses: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "raw": raw.to_dict(),
            "correction": correction.to_dict(),
            "metrics": metrics.to_dict(),
            "server_statuses": server_statuses,
            "preprocess": preprocess_result.to_dict(),
            "config": self.config.to_dict(),
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
