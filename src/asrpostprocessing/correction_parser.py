from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .schemas import CorrectionResult, Edit


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def parse_correction_response(response_text: str, original_text: str) -> CorrectionResult:
    payload_text = _extract_json(response_text or "")
    if not payload_text:
        return CorrectionResult(
            corrected_text=original_text,
            risk="high",
            metadata={"parse_error": "no_json_payload", "raw_response": response_text},
        )
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return CorrectionResult(
            corrected_text=original_text,
            risk="high",
            metadata={"parse_error": str(exc), "raw_response": response_text},
        )
    return correction_from_payload(payload, original_text)


def correction_from_payload(payload: Dict[str, Any], original_text: str) -> CorrectionResult:
    corrected = payload.get("corrected_text")
    if not isinstance(corrected, str) or not corrected.strip():
        corrected = original_text
    edits: List[Edit] = []
    for item in payload.get("edits", []) or []:
        if not isinstance(item, dict):
            continue
        before = str(item.get("before", ""))
        after = str(item.get("after", ""))
        if not before and not after:
            continue
        edits.append(
            Edit(
                before=before,
                after=after,
                reason=str(item.get("reason", "")),
                confidence=_float_or_zero(item.get("confidence")),
                start_char=_optional_int(item.get("start_char")),
                end_char=_optional_int(item.get("end_char")),
            )
        )
    risk = str(payload.get("risk", "unknown"))
    used_context_ids = [str(item) for item in payload.get("used_context_ids", []) or []]
    metadata = {key: value for key, value in payload.items() if key not in {"corrected_text", "edits", "risk", "used_context_ids"}}
    return CorrectionResult(corrected_text=corrected, edits=edits, risk=risk, used_context_ids=used_context_ids, metadata=metadata)


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1)
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        return text[first : last + 1]
    return ""


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_int(value: Any):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
