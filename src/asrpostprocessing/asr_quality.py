from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .config import ExperimentConfig
from .keyword_bias import normalize_keywords
from .keyword_correction import keyword_near_miss_replacements
from .schemas import CorrectionResult, TranscriptResult

_ASR_TEXT_TAG_RE = re.compile(r"</?\s*asr_text\s*>", flags=re.IGNORECASE)
_LANGUAGE_LABEL_RE = re.compile(
    r"(?:^|\n|\s)language\s+[A-Za-z_-]{1,32}(?=\s*<\s*asr_text\s*>|\s*$|\n)",
    flags=re.IGNORECASE,
)
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_PUNCT_RE = re.compile(r"[\u3000-\u303f\uff00-\uffef，。！？、：；]")


def build_asr_quality_report(raw: TranscriptResult, preprocess: Dict[str, Any], config: ExperimentConfig) -> Dict[str, Any]:
    chunk_reports = _chunk_reports(raw)
    keyword_near_misses = _keyword_near_misses(raw.text or "", config)
    language_drift_reasons = _language_drift_reasons(raw)
    text_artifacts = _text_artifact_summary(raw.text or "")
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
    if not (raw.text or "").strip():
        warnings.append("ASR produced an empty transcript.")
        action_items.append("Inspect whether the audio segment is silence, low-volume speech, noise, or filtered language drift.")
    if metadata.get("chunked") and chunk_seconds < 60.0:
        warnings.append(f"ASR audio chunks are short ({chunk_seconds:g}s); short chunks can lose lecture context.")
        action_items.append("Compare with 120s ASR chunks on the same audio.")
    if not chunk_reports:
        warnings.append("ASR did not produce segment/chunk diagnostics.")
    if any(chunk["text_chars"] == 0 for chunk in chunk_reports):
        warnings.append("At least one ASR chunk produced empty text.")
        action_items.append("Inspect empty ASR chunks for low-volume speech, silence, noise, or language drift.")
    if language_drift_reasons:
        warnings.append("ASR language drift artifact(s) were filtered before post-processing.")
        action_items.append("Inspect filtered ASR chunks if transcript context is missing near those timestamps.")
    if _has_artifact_risk(text_artifacts):
        warnings.append("ASR transcript contains artifact marker or non-Korean CJK drift candidate(s).")
        action_items.append("Inspect ASR parsing, language cleanup, and the affected raw transcript span before post-processing.")
    if keyword_near_misses:
        warnings.append("ASR contains keyword near-miss candidate(s).")
        action_items.append("Enable keyword-guided post-processing or inspect the listed near-miss terms.")

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
        "language_drift": {"filtered_reasons": language_drift_reasons},
        "text_artifacts": text_artifacts,
        "keyword_near_misses": keyword_near_misses,
        "chunks": chunk_reports,
        "warnings": _dedupe(warnings),
        "action_items": _dedupe(action_items),
        "note": "Reference-free ASR quality report; CER/WER still require reference text.",
    }


def build_correction_quality_report(
    raw: TranscriptResult,
    correction: CorrectionResult,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    raw_text = raw.text or ""
    corrected_text = correction.corrected_text or ""
    raw_keyword_near_misses = _keyword_near_misses(raw_text, config)
    corrected_keyword_near_misses = _keyword_near_misses(corrected_text, config)
    raw_artifacts = _text_artifact_summary(raw_text)
    corrected_artifacts = _text_artifact_summary(corrected_text)
    fallback = _fallback_summary(correction)
    warnings: List[str] = []
    action_items: List[str] = []
    improvements: List[str] = []

    if raw_keyword_near_misses and len(corrected_keyword_near_misses) < len(raw_keyword_near_misses):
        improvements.append("Corrected transcript has fewer keyword near-miss candidate(s) than raw transcript.")
    if _artifact_score(corrected_artifacts) < _artifact_score(raw_artifacts):
        improvements.append("Corrected transcript has fewer ASR artifact marker(s) than raw transcript.")

    if corrected_keyword_near_misses:
        warnings.append("Corrected transcript still contains keyword near-miss candidate(s).")
        action_items.append("Inspect keyword list, post-processing edits, and local context for missed supported corrections.")
    if _has_artifact_risk(corrected_artifacts):
        warnings.append("Corrected transcript still contains ASR artifact or non-Korean CJK drift candidate(s).")
        action_items.append("Inspect ASR cleanup and rerun the affected chunk if transcript context is missing.")
    if fallback["fallback_chunk_count"] > 0:
        warnings.append("Post-processing fallback was used for at least one chunk.")
        action_items.append("Check post-processing backend health, request timeout, and text chunk size.")
    if correction.risk in {"medium", "high"}:
        warnings.append(f"Post-processing returned {correction.risk} risk.")

    if not action_items:
        action_items.append("Use reference text with CER/WER evaluation for final quality judgment.")

    return {
        "risk": correction.risk,
        "text_chars": {
            "raw": len(raw_text),
            "corrected": len(corrected_text),
            "delta": len(corrected_text) - len(raw_text),
        },
        "edits": _edit_summary(correction),
        "keyword_near_misses": {
            "raw_count": len(raw_keyword_near_misses),
            "corrected_count": len(corrected_keyword_near_misses),
            "resolved_count": max(0, len(raw_keyword_near_misses) - len(corrected_keyword_near_misses)),
            "raw": raw_keyword_near_misses,
            "corrected": corrected_keyword_near_misses,
        },
        "artifacts": {
            "raw": raw_artifacts,
            "corrected": corrected_artifacts,
        },
        "postprocess": fallback,
        "improvements": improvements,
        "warnings": _dedupe(warnings),
        "action_items": _dedupe(action_items),
        "note": "Reference-free corrected-output quality report; CER/WER still require reference text.",
    }


def _keyword_near_misses(text: str, config: ExperimentConfig) -> List[Dict[str, Any]]:
    keywords = [keyword for keyword in normalize_keywords(getattr(config, "keywords", [])) if keyword]
    if not text or not keywords:
        return []
    results = []
    for start, end, before, after in keyword_near_miss_replacements(text, keywords):
        results.append({"before": before, "after": after, "start_char": start, "end_char": end})
    return results


def _edit_summary(correction: CorrectionResult) -> Dict[str, Any]:
    keyword_edit_count = 0
    for edit in correction.edits or []:
        if "keyword" in (edit.reason or "").lower() and "near-miss" in (edit.reason or "").lower():
            keyword_edit_count += 1
    return {
        "count": len(correction.edits or []),
        "keyword_near_miss_count": keyword_edit_count,
    }


def _fallback_summary(correction: CorrectionResult) -> Dict[str, Any]:
    chunks = []
    for index, item in enumerate(_correction_chunk_metadata(correction)):
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        fallback = str(metadata.get("fallback") or "").strip()
        error = str(metadata.get("postprocess_error") or "").strip()
        if not fallback and not error:
            continue
        chunks.append(
            {
                "index": int(item.get("index", index)),
                "fallback": fallback,
                "post_backend": metadata.get("post_backend"),
                "postprocess_error": _preview(error, limit=300) if error else "",
            }
        )

    metadata = correction.metadata if isinstance(correction.metadata, dict) else {}
    if not chunks and (metadata.get("fallback") or metadata.get("postprocess_error")):
        error = str(metadata.get("postprocess_error") or "").strip()
        chunks.append(
            {
                "index": 0,
                "fallback": str(metadata.get("fallback") or "").strip(),
                "post_backend": metadata.get("post_backend"),
                "postprocess_error": _preview(error, limit=300) if error else "",
            }
        )

    return {
        "chunk_count": len(_correction_chunk_metadata(correction)),
        "fallback_chunk_count": len(chunks),
        "postprocess_error_count": sum(1 for chunk in chunks if chunk.get("postprocess_error")),
        "fallback_chunks": chunks,
    }


def _correction_chunk_metadata(correction: CorrectionResult) -> List[Dict[str, Any]]:
    metadata = correction.metadata if isinstance(correction.metadata, dict) else {}
    chunks = metadata.get("chunks")
    if not isinstance(chunks, list):
        return []
    result = []
    for index, item in enumerate(chunks):
        if not isinstance(item, dict):
            continue
        normalized = dict(item)
        normalized.setdefault("index", index)
        result.append(normalized)
    return result


def _text_artifact_summary(text: str) -> Dict[str, Any]:
    asr_text_tag_count = len(_ASR_TEXT_TAG_RE.findall(text or ""))
    language_label_count = len(_LANGUAGE_LABEL_RE.findall(text or ""))
    han_chars = _HAN_RE.findall(text or "")
    cjk_punctuation_count = len(_CJK_PUNCT_RE.findall(text or ""))
    return {
        "asr_text_tag_count": asr_text_tag_count,
        "language_label_count": language_label_count,
        "han_char_count": len(han_chars),
        "cjk_punctuation_count": cjk_punctuation_count,
        "non_korean_cjk_drift_candidate": len(han_chars) >= 4,
        "has_asr_artifact_markers": asr_text_tag_count > 0 or language_label_count > 0,
        "han_preview": "".join(han_chars[:24]),
    }


def _artifact_score(summary: Dict[str, Any]) -> int:
    return (
        int(summary.get("asr_text_tag_count") or 0)
        + int(summary.get("language_label_count") or 0)
        + int(summary.get("han_char_count") or 0)
        + int(summary.get("cjk_punctuation_count") or 0)
    )


def _has_artifact_risk(summary: Dict[str, Any]) -> bool:
    return bool(summary.get("has_asr_artifact_markers") or summary.get("non_korean_cjk_drift_candidate"))


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
    filtered_reason = _filtered_reason_from_metadata(metadata)
    if filtered_reason:
        report["filtered_reason"] = filtered_reason
    return report


def _language_drift_reasons(raw: TranscriptResult) -> List[str]:
    reasons = []
    metadata = raw.metadata if isinstance(raw.metadata, dict) else {}
    reason = _filtered_reason_from_metadata(metadata)
    if reason:
        reasons.append(reason)
    for segment in raw.segments or []:
        segment_metadata = segment.metadata if isinstance(segment.metadata, dict) else {}
        reason = _filtered_reason_from_metadata(segment_metadata)
        if reason:
            reasons.append(reason)
    return _dedupe(reasons)


def _filtered_reason_from_metadata(metadata: Dict[str, Any]) -> str:
    if not isinstance(metadata, dict):
        return ""
    direct = str(metadata.get("filtered_asr_text_reason") or "").strip()
    if direct:
        return direct
    parsed = metadata.get("parsed") if isinstance(metadata.get("parsed"), dict) else {}
    parsed_reason = str(parsed.get("filtered_reason") or "").strip()
    if parsed_reason:
        return parsed_reason
    asr_metadata = metadata.get("asr_metadata") if isinstance(metadata.get("asr_metadata"), dict) else {}
    return _filtered_reason_from_metadata(asr_metadata) if asr_metadata else ""


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
