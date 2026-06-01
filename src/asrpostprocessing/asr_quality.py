from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import ExperimentConfig
from .schemas import TranscriptResult


def build_asr_quality_report(raw: TranscriptResult, preprocess: Dict[str, Any], config: ExperimentConfig) -> Dict[str, Any]:
    chunk_reports = _chunk_reports(raw)
    warnings: List[str] = []
    action_items: List[str] = []

    preprocess_warnings = [str(warning) for warning in preprocess.get("warnings") or []]
    warnings.extend(f"preprocess: {warning}" for warning in preprocess_warnings)
    for step in preprocess.get("steps") or []:
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        step_name = str(step.get("step") or step.get("model") or "preprocess")
        clipped = int(metadata.get("clipped_samples") or 0)
        if clipped > 0:
            warnings.append(f"{step_name}: {clipped} clipped sample(s) were introduced before ASR.")
            action_items.append("Re-run without volume normalization or with peak-limited normalization.")
        if metadata.get("peak_limited"):
            warnings.append(f"{step_name}: gain was peak-limited to avoid clipping.")

    metadata = raw.metadata or {}
    chunk_seconds = _as_float(metadata.get("chunk_seconds")) or float(getattr(config, "asr_chunk_seconds", 120.0) or 120.0)
    if metadata.get("chunked") and chunk_seconds < 60.0:
        warnings.append(f"ASR audio chunks are short ({chunk_seconds:g}s); short chunks can lose lecture context.")
        action_items.append("Compare with 120s ASR chunks on the same audio.")
    if not chunk_reports:
        warnings.append("ASR did not produce segment/chunk diagnostics.")
    if any(chunk["text_chars"] == 0 for chunk in chunk_reports):
        warnings.append("At least one ASR chunk produced empty text.")
        action_items.append("Inspect empty ASR chunks for low-volume speech, silence, noise, or language drift.")

    if not action_items:
        action_items.append("If quality is still poor, compare no-preprocess, fixed 120s, and silence-aware 120s ASR runs.")

    return {
        "backend": metadata.get("backend") or getattr(config, "asr_backend", ""),
        "language": raw.language,
        "text_chars": len(raw.text or ""),
        "chunking": {
            "chunked": bool(metadata.get("chunked")),
            "strategy": metadata.get("chunking_strategy") or getattr(config, "asr_chunking_strategy", ""),
            "chunk_seconds": chunk_seconds,
            "context_chars": metadata.get("context_chars") if "context_chars" in metadata else getattr(config, "asr_context_chars", 0),
            "chunk_count": len(chunk_reports),
        },
        "preprocess": _preprocess_summary(preprocess),
        "chunks": chunk_reports,
        "warnings": _dedupe(warnings),
        "action_items": _dedupe(action_items),
        "note": "Reference-free ASR quality report; CER/WER still require reference text.",
    }


def _chunk_reports(raw: TranscriptResult) -> List[Dict[str, Any]]:
    chunks = []
    metadata_chunks = raw.metadata.get("chunks") if isinstance(raw.metadata, dict) else None
    if isinstance(metadata_chunks, list) and metadata_chunks:
        for index, item in enumerate(metadata_chunks):
            if not isinstance(item, dict):
                continue
            segment_text = raw.segments[index].text if index < len(raw.segments) else ""
            start_s = _as_float(item.get("start_s"))
            end_s = _as_float(item.get("end_s"))
            chunks.append(_chunk_report(index, start_s, end_s, segment_text, item))
        return chunks

    if raw.segments:
        for index, segment in enumerate(raw.segments):
            chunks.append(_chunk_report(index, segment.start_s, segment.end_s, segment.text, segment.metadata))
        return chunks

    if raw.text:
        chunks.append(_chunk_report(0, None, None, raw.text, {"method": "single"}))
    return chunks


def _chunk_report(index: int, start_s: Optional[float], end_s: Optional[float], text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    duration_s = (end_s - start_s) if start_s is not None and end_s is not None and end_s >= start_s else None
    text_chars = len(text or "")
    report = {
        "index": index,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": duration_s,
        "method": metadata.get("method") or metadata.get("chunk_method") or "unknown",
        "text_chars": text_chars,
        "chars_per_second": (text_chars / duration_s) if duration_s else None,
        "previous_context_chars": int(metadata.get("previous_context_chars") or 0),
        "text_preview": _preview(text),
    }
    if "audio_path" in metadata:
        report["audio_path"] = metadata["audio_path"]
    return report


def _preprocess_summary(preprocess: Dict[str, Any]) -> Dict[str, Any]:
    steps = []
    for step in preprocess.get("steps") or []:
        if not isinstance(step, dict):
            continue
        metadata = step.get("metadata") if isinstance(step.get("metadata"), dict) else {}
        steps.append(
            {
                "step": step.get("step"),
                "model": step.get("model"),
                "applied": bool(step.get("applied")),
                "warnings": step.get("warnings") or [],
                "clipped_samples": int(metadata.get("clipped_samples") or 0),
                "peak_limited": bool(metadata.get("peak_limited")),
                "gain_factor": metadata.get("gain_factor"),
                "duration_seconds": metadata.get("duration_seconds"),
            }
        )
    return {
        "applied": bool(preprocess.get("applied")),
        "audio_path": preprocess.get("audio_path"),
        "warnings": preprocess.get("warnings") or [],
        "steps": steps,
    }


def _preview(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
