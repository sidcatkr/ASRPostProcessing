from __future__ import annotations

from typing import Optional

from .schemas import MetricsResult
from .text import cer, character_f1, wer_eojeol


def evaluate_transcripts(
    reference: Optional[str],
    raw_text: str,
    corrected_text: str,
    latency_ms: Optional[float] = None,
) -> MetricsResult:
    if not reference:
        return MetricsResult(latency_ms=latency_ms)

    raw_cer_no_space = cer(reference, raw_text, remove_spaces=True, remove_symbols=True)
    corrected_cer_no_space = cer(reference, corrected_text, remove_spaces=True, remove_symbols=True)
    raw_cer_strict = cer(reference, raw_text, remove_spaces=False)
    corrected_cer_strict = cer(reference, corrected_text, remove_spaces=False)
    raw_wer = wer_eojeol(reference, raw_text)
    corrected_wer = wer_eojeol(reference, corrected_text)
    return MetricsResult(
        cer_normalized_no_space=corrected_cer_no_space,
        cer_strict=corrected_cer_strict,
        wer_eojeol=corrected_wer,
        raw_cer_normalized_no_space=raw_cer_no_space,
        raw_cer_strict=raw_cer_strict,
        raw_wer_eojeol=raw_wer,
        delta_cer=raw_cer_no_space - corrected_cer_no_space,
        delta_wer=raw_wer - corrected_wer,
        semantic_similarity=character_f1(raw_text, corrected_text),
        latency_ms=latency_ms,
    )
