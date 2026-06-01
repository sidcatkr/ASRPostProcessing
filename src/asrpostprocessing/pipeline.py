from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .adapters import build_asr_adapter, build_postprocess_adapter
from .asr_quality import build_asr_quality_report, build_correction_quality_report
from .cache import cache_json_path, file_sha256, read_json, stable_json_hash, transcript_from_dict, write_json_atomic
from .chunking import chunk_segments, chunk_text
from .config import ExperimentConfig, normalize_model_residency
from .keyword_correction import apply_keyword_near_miss_corrections
from .keyword_bias import build_keyword_bias_instruction
from .logging import RunLogger, make_run_id
from .metrics import evaluate_transcripts
from .model_server import ensure_model_servers, stop_model_servers
from .preprocess import preprocess_audio
from .rag import build_rag_index
from .schemas import CorrectionResult, Edit, MetricsResult, RAGContext, SearchResult, TranscriptResult
from .search import CachedSearchProvider
from .text import make_diff_html, merge_overlapping_texts

StatusCallback = Callable[[str], None]


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
    asr_quality: Dict[str, Any]
    correction_quality: Dict[str, Any]

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
            "asr_quality": self.asr_quality,
            "correction_quality": self.correction_quality,
        }


class PipelineRunner:
    def __init__(self, config: ExperimentConfig, status_callback: Optional[StatusCallback] = None):
        self.config = config
        self.config.model_residency = normalize_model_residency(self.config.model_residency)
        self.status_callback = status_callback

    def run(
        self,
        audio_path: str,
        reference_text: Optional[str] = None,
        run_id: Optional[str] = None,
        rag_inline_text: str = "",
    ) -> PipelineOutput:
        started = time.time()
        run_id = run_id or make_run_id(self.config.run_name or "run")
        self._emit(f"Run {run_id} started.")
        if rag_inline_text:
            self.config.rag_inline_text = rag_inline_text

        self._emit("Checking model server readiness.")
        server_statuses = self._initial_server_statuses()
        self._emit("Preprocessing audio.")
        preprocess_result = preprocess_audio(audio_path, self.config)
        self._emit(_preprocess_status(preprocess_result.to_dict()))
        keyword_instruction = ""
        if self.config.enable_keyword_bias:
            self._emit("Building ASR keyword bias instruction.")
            keyword_instruction = build_keyword_bias_instruction(self.config.keywords, self.config.keyword_bias_weight)

        if self._sequential_model_residency():
            self._emit("Starting ASR model server.")
            server_statuses.extend(status.to_dict() for status in ensure_model_servers(self.config, status_callback=self._emit, names=["asr"]))
        try:
            self._emit(f"Sending audio to ASR backend {self.config.asr_backend}.")
            raw = self._transcribe_with_cache(preprocess_result.audio_path, keyword_instruction)
            self._emit(f"ASR complete: {len(raw.text)} transcript characters.")
        finally:
            if self._sequential_model_residency():
                server_statuses.extend(self._release_stage_model("asr"))

        correction = self._postprocess(raw, server_statuses)
        latency_ms = (time.time() - started) * 1000.0
        self._emit("Evaluating transcript metrics.")
        metrics = evaluate_transcripts(reference_text, raw.text, correction.corrected_text, latency_ms=latency_ms)
        diff_html = make_diff_html(raw.text, correction.corrected_text)
        asr_quality = build_asr_quality_report(raw, preprocess_result.to_dict(), self.config)
        correction_quality = build_correction_quality_report(raw, correction, self.config)

        self._emit("Writing run artifacts.")
        logger = RunLogger(self.config, run_id)
        artifacts = {
            "result": str(logger.output_dir / "result.json"),
            "raw_transcript": str(logger.write_text("raw_transcript.txt", raw.text)),
            "corrected_transcript": str(logger.write_text("corrected_transcript.txt", correction.corrected_text)),
            "diff_html": str(logger.write_text("diff.html", diff_html)),
            "asr_quality": str(logger.write_json("asr_quality.json", asr_quality)),
            "correction_quality": str(logger.write_json("correction_quality.json", correction_quality)),
            "preprocess": str(logger.write_json("preprocess.json", preprocess_result.to_dict())),
            "metrics": str(logger.write_json("metrics.json", metrics.to_dict())),
            "edits": str(logger.write_edits(correction.edits)),
            "config": str(logger.write_config()),
            "tensorboard_fallback": str(logger.write_tensorboard_metrics(metrics)),
        }
        logger.write_json(
            "result.json",
            self._result_payload(raw, correction, metrics, preprocess_result, server_statuses, asr_quality, correction_quality, artifacts),
        )
        output = PipelineOutput(
            run_id=run_id,
            raw=raw,
            correction=correction,
            metrics=metrics,
            diff_html=diff_html,
            output_dir=str(logger.output_dir),
            artifacts=artifacts,
            server_statuses=server_statuses,
            preprocess=preprocess_result.to_dict(),
            asr_quality=asr_quality,
            correction_quality=correction_quality,
        )
        self._emit(f"Run {run_id} complete in {latency_ms / 1000.0:.1f}s.")
        return output

    def _postprocess(self, raw: TranscriptResult, server_statuses: List[Dict[str, Any]]) -> CorrectionResult:
        if not self.config.enable_llm_postprocess:
            self._emit("LLM post-processing is disabled.")
            return CorrectionResult(corrected_text=raw.text, risk="unchanged", metadata={"reason": "postprocess_disabled"})

        chunks = chunk_segments(raw.segments) if raw.segments else chunk_text(raw.text, self.config.chunk_max_chars, self.config.chunk_overlap)
        self._emit(f"Preparing LLM post-processing for {len(chunks)} chunk(s).")
        rag_index = build_rag_index(self.config) if self.config.enable_rag else None
        if self._sequential_model_residency():
            self._emit("Starting post-processing model server.")
            server_statuses.extend(status.to_dict() for status in ensure_model_servers(self.config, status_callback=self._emit, names=["post"]))
        try:
            corrected_chunks, all_edits, all_context_ids, chunk_metadata = self._postprocess_chunks(chunks, rag_index)
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

    def _transcribe_with_cache(self, audio_path: str, keyword_instruction: str) -> TranscriptResult:
        cache_path = self._asr_cache_path(audio_path, keyword_instruction) if self.config.asr_cache_enabled else None
        if cache_path is not None:
            cached = read_json(cache_path)
            if cached and isinstance(cached.get("transcript"), dict):
                raw = transcript_from_dict(cached["transcript"])
                raw.metadata.setdefault("asr_cache", {})
                raw.metadata["asr_cache"].update({"hit": True, "path": str(cache_path), "key": cached.get("key")})
                self._emit(f"ASR cache hit: {cache_path}")
                return raw
        asr = build_asr_adapter(self.config)
        raw = asr.transcribe(audio_path, self.config, keyword_instruction=keyword_instruction)
        if cache_path is not None:
            key = cache_path.stem
            write_json_atomic(
                cache_path,
                {
                    "key": key,
                    "created_at": time.time(),
                    "transcript": raw.to_dict(),
                },
            )
            raw.metadata.setdefault("asr_cache", {})
            raw.metadata["asr_cache"].update({"hit": False, "path": str(cache_path), "key": key})
            self._emit(f"ASR cache stored: {cache_path}")
        return raw

    def _asr_cache_path(self, audio_path: str, keyword_instruction: str) -> Path:
        payload = {
            "audio_sha256": file_sha256(audio_path),
            "asr_model": self.config.asr_model,
            "asr_backend": self.config.asr_backend,
            "language": self.config.language,
            "asr_chunking_strategy": self.config.asr_chunking_strategy,
            "asr_chunk_seconds": float(self.config.asr_chunk_seconds),
            "asr_chunk_padding_seconds": float(self.config.asr_chunk_padding_seconds),
            "asr_silence_threshold_db": float(self.config.asr_silence_threshold_db),
            "asr_min_silence_seconds": float(self.config.asr_min_silence_seconds),
            "asr_context_chars": int(self.config.asr_context_chars),
            "keyword_bias_enabled": bool(self.config.enable_keyword_bias),
            "keyword_bias_weight": float(self.config.keyword_bias_weight),
            "keywords": sorted(self.config.keywords),
            "keyword_instruction": keyword_instruction,
            "prompt_version": "vllm_asr_instruction_2026_06_01",
        }
        return cache_json_path(self.config.cache_dir, "asr", stable_json_hash(payload))

    def _postprocess_chunks(self, chunks, rag_index):
        if self.config.postprocess_parallelism <= 1 or len(chunks) <= 1:
            ordered = [self._postprocess_one_chunk(chunk, len(chunks), rag_index) for chunk in chunks]
        else:
            self._emit(
                f"Post-processing chunks with parallelism={self.config.postprocess_parallelism} "
                f"across {len(self._post_endpoint_pool())} endpoint(s)."
            )
            ordered = [None] * len(chunks)
            with ThreadPoolExecutor(max_workers=self.config.postprocess_parallelism) as executor:
                future_to_index = {
                    executor.submit(self._postprocess_one_chunk, chunk, len(chunks), rag_index): chunk.index for chunk in chunks
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    ordered[index] = future.result()
            ordered = [item for item in ordered if item is not None]
        corrected_chunks: List[str] = []
        all_edits: List[Edit] = []
        all_context_ids: List[str] = []
        chunk_metadata: List[Dict[str, Any]] = []
        for _, chunk, result in sorted(ordered, key=lambda item: item[0]):
            corrected_chunks.append(result.corrected_text)
            all_edits.extend(result.edits)
            all_context_ids.extend(result.used_context_ids)
            chunk_metadata.append({"chunk": chunk.__dict__, "risk": result.risk, "metadata": result.metadata})
        return corrected_chunks, all_edits, all_context_ids, chunk_metadata

    def _postprocess_one_chunk(self, chunk, total_chunks: int, rag_index):
        self._emit(f"Post-processing chunk {chunk.index + 1}/{total_chunks}.")
        contexts: List[RAGContext] = []
        if rag_index is not None:
            self._emit(f"Retrieving RAG context for chunk {chunk.index + 1}/{total_chunks}.")
            contexts = rag_index.retrieve(chunk.text, top_k=self.config.rag_top_k, strength=self.config.rag_strength)
        query = self._search_query(chunk.text)
        search_results: List[SearchResult] = []
        chunk_config = self._config_for_post_chunk(chunk.index)
        if chunk_config.enable_search:
            self._emit(f"Searching external context for chunk {chunk.index + 1}/{total_chunks}.")
            search_results = CachedSearchProvider(chunk_config).search(query)
        self._emit(
            f"Sending chunk {chunk.index + 1}/{total_chunks} to post-processing backend "
            f"{chunk_config.post_backend} at {chunk_config.post_base_url}."
        )
        try:
            postprocessor = build_postprocess_adapter(chunk_config)
            result = postprocessor.correct(chunk.text, chunk_config, contexts, search_results)
            self._emit(f"Post-processing chunk {chunk.index + 1}/{total_chunks} complete.")
        except Exception as exc:
            self._emit(f"Post-processing chunk {chunk.index + 1}/{total_chunks} failed; using deterministic fallback.")
            result = _fallback_postprocess_result(chunk.text, chunk_config, exc)
        result.metadata.setdefault("post_base_url", chunk_config.post_base_url)
        return chunk.index, chunk, result

    def _config_for_post_chunk(self, chunk_index: int) -> ExperimentConfig:
        endpoints = self._post_endpoint_pool()
        if not endpoints:
            return self.config
        config = copy.deepcopy(self.config)
        config.post_base_url = endpoints[chunk_index % len(endpoints)]
        return config

    def _post_endpoint_pool(self) -> List[str]:
        lanes = getattr(self.config, "pipeline_lanes", []) or []
        lane_candidates = [
            lane
            for lane in lanes
            if isinstance(lane, dict)
            and str(lane.get("post_base_url") or "").strip()
            and (not lane.get("post_model") or str(lane.get("post_model")) == self.config.post_model)
        ]
        endpoints = [str(lane.get("post_base_url")).strip() for lane in lane_candidates]
        endpoints.extend(str(item).strip() for item in (getattr(self.config, "post_base_urls", []) or []) if str(item).strip())
        if not endpoints:
            endpoints = [self.config.post_base_url]
        deduped: List[str] = []
        for endpoint in endpoints:
            if endpoint not in deduped:
                deduped.append(endpoint)
        return deduped

    def _initial_server_statuses(self) -> List[Dict[str, Any]]:
        if self._sequential_model_residency():
            return []
        return [status.to_dict() for status in ensure_model_servers(self.config, status_callback=self._emit)]

    def _sequential_model_residency(self) -> bool:
        return self.config.model_residency == "sequential"

    def _release_stage_model(self, name: str) -> List[Dict[str, Any]]:
        statuses: List[Dict[str, Any]] = []
        if self.config.auto_start_model_servers:
            statuses.extend(status.to_dict() for status in stop_model_servers(self.config, status_callback=self._emit, names=[name]))
        if name == "asr":
            self._clear_direct_asr_cache()
        return statuses

    def _emit(self, message: str) -> None:
        if not self.status_callback:
            return
        try:
            self.status_callback(message)
        except Exception:
            pass

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
        asr_quality: Dict[str, Any],
        correction_quality: Dict[str, Any],
        artifacts: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "raw": raw.to_dict(),
            "correction": correction.to_dict(),
            "metrics": metrics.to_dict(),
            "artifacts": artifacts,
            "server_statuses": server_statuses,
            "preprocess": preprocess_result.to_dict(),
            "asr_quality": asr_quality,
            "correction_quality": correction_quality,
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


def _fallback_postprocess_result(chunk_text: str, config: ExperimentConfig, exc: Exception) -> CorrectionResult:
    result = CorrectionResult(
        corrected_text=chunk_text,
        risk="high",
        metadata={
            "fallback": "raw_transcript_after_postprocess_error",
            "postprocess_error": str(exc),
            "post_backend": config.post_backend,
        },
    )
    return apply_keyword_near_miss_corrections(result, config)


def _preprocess_status(preprocess: Dict[str, Any]) -> str:
    if preprocess.get("applied"):
        steps = preprocess.get("steps") or []
        labels = [str(step.get("step") or step.get("name") or "preprocess") for step in steps if isinstance(step, dict)]
        detail = ", ".join(labels) if labels else "selected preprocessing"
        message = f"Preprocessing complete: {detail}."
        warnings = preprocess.get("warnings") or []
        if warnings:
            message += f" Warning: {warnings[0]}"
        return message
    warnings = preprocess.get("warnings") or []
    if warnings:
        return f"Preprocessing skipped with warning: {warnings[0]}"
    return "Preprocessing skipped: using input audio."
