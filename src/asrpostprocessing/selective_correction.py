from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from .config import ExperimentConfig
from .schemas import CorrectionResult, Edit


@dataclass(frozen=True)
class VerifiedEdit:
    edit: Edit
    start: int
    end: int


def verify_and_apply_correction(original_text: str, candidate: CorrectionResult, config: ExperimentConfig) -> CorrectionResult:
    """Apply only high-confidence, non-overlapping edit proposals to the raw text.

    The LLM may still return a full ``corrected_text`` for backward compatibility, but this
    verifier treats it as an edit proposal source. The raw transcript is an explicit keep
    candidate, and edits are accepted only when their confidence and span checks pass.
    """
    if not bool(getattr(config, "enable_selective_correction", True)):
        candidate.metadata.setdefault("selective_correction", {"enabled": False})
        return candidate

    threshold = _confidence_threshold(config)
    max_ratio = _max_edit_ratio(config)
    max_edits = _max_edits_for_text(original_text, config)
    verified: List[VerifiedEdit] = []
    rejected: List[dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []

    for edit in candidate.edits:
        accepted, start, end, reason = _verify_edit(edit, original_text, threshold, occupied)
        if not accepted:
            rejected.append({**edit.to_dict(), "reject_reason": reason})
            continue
        verified.append(VerifiedEdit(edit=edit, start=start, end=end))
        occupied.append((start, end))

    verified.sort(key=lambda item: item.start)
    if len(verified) > max_edits:
        rejected.extend({**item.edit.to_dict(), "reject_reason": "max_edits_per_text_exceeded"} for item in verified[max_edits:])
        verified = verified[:max_edits]

    changed_chars = sum(max(item.end - item.start, len(item.edit.after)) for item in verified)
    ratio = changed_chars / max(1, len(original_text))
    if ratio > max_ratio:
        rejected.extend({**item.edit.to_dict(), "reject_reason": "max_edit_ratio_exceeded"} for item in verified)
        verified = []
        ratio = 0.0

    corrected = original_text
    applied_edits: List[Edit] = []
    for item in reversed(verified):
        corrected = corrected[: item.start] + item.edit.after + corrected[item.end :]
        applied_edits.append(
            Edit(
                before=item.edit.before,
                after=item.edit.after,
                reason=item.edit.reason,
                confidence=item.edit.confidence,
                start_char=item.start,
                end_char=item.end,
            )
        )
    applied_edits.reverse()

    metadata = dict(candidate.metadata)
    metadata["selective_correction"] = {
        "enabled": True,
        "minimum_confidence": threshold,
        "max_edit_ratio": max_ratio,
        "max_edits": max_edits,
        "applied_count": len(applied_edits),
        "rejected_count": len(rejected),
        "rejected_edits": rejected,
        "candidate_text_ignored": candidate.corrected_text != corrected,
    }
    return CorrectionResult(
        corrected_text=corrected,
        edits=applied_edits,
        risk="low" if applied_edits else "unchanged",
        used_context_ids=list(candidate.used_context_ids),
        metadata=metadata,
    )


def _verify_edit(edit: Edit, text: str, threshold: float, occupied: List[Tuple[int, int]]) -> Tuple[bool, int, int, str]:
    if not edit.before or not edit.after or edit.before == edit.after:
        return False, -1, -1, "empty_or_noop"
    if float(edit.confidence or 0.0) < threshold:
        return False, -1, -1, "confidence_below_threshold"
    start = edit.start_char
    end = edit.end_char
    if start is not None and end is not None and 0 <= start < end <= len(text) and text[start:end] == edit.before:
        span = (start, end)
    else:
        found = text.find(edit.before)
        if found < 0:
            return False, -1, -1, "before_span_not_found"
        span = (found, found + len(edit.before))
    if any(not (span[1] <= used_start or span[0] >= used_end) for used_start, used_end in occupied):
        return False, -1, -1, "overlapping_edit"
    return True, span[0], span[1], ""


def _confidence_threshold(config: ExperimentConfig) -> float:
    explicit = getattr(config, "selective_min_confidence", None)
    if explicit is not None:
        try:
            return max(0.0, min(1.0, float(explicit)))
        except (TypeError, ValueError):
            pass
    strength = max(0.0, min(1.0, float(getattr(config, "postprocess_strength", 0.5) or 0.5)))
    if strength < 0.35:
        return 0.95
    if strength < 0.7:
        return 0.90
    return 0.80


def _max_edit_ratio(config: ExperimentConfig) -> float:
    try:
        return max(0.0, min(1.0, float(getattr(config, "selective_max_edit_ratio", 0.20))))
    except (TypeError, ValueError):
        return 0.20


def _max_edits_for_text(text: str, config: ExperimentConfig) -> int:
    try:
        per_100 = max(1, int(getattr(config, "selective_max_edits_per_100_chars", 2)))
    except (TypeError, ValueError):
        per_100 = 2
    return max(1, int((max(1, len(text)) / 100.0) * per_100 + 0.999))
