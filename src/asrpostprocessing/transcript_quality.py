from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .asr_quality import build_asr_quality_report, build_correction_quality_report
from .config import ExperimentConfig
from .schemas import CorrectionResult, TranscriptResult


def build_transcript_quality_report(
    raw_path: str,
    config: ExperimentConfig,
    corrected_path: Optional[str] = None,
) -> Dict[str, Any]:
    raw_file = Path(raw_path)
    raw_text = raw_file.read_text(encoding="utf-8")
    raw = TranscriptResult(
        language=config.language,
        text=raw_text,
        metadata={"backend": "file", "source_path": str(raw_file)},
    )
    payload: Dict[str, Any] = {
        "files": {
            "raw": str(raw_file),
            "corrected": str(Path(corrected_path)) if corrected_path else "",
        },
        "asr_quality": build_asr_quality_report(
            raw,
            {"applied": False, "audio_path": str(raw_file), "warnings": [], "steps": []},
            config,
        ),
        "note": "Transcript-file quality report; CER/WER still require reference text.",
    }
    if corrected_path:
        corrected_file = Path(corrected_path)
        corrected_text = corrected_file.read_text(encoding="utf-8")
        correction = CorrectionResult(
            corrected_text=corrected_text,
            risk="unknown",
            metadata={"source_path": str(corrected_file)},
        )
        payload["correction_quality"] = build_correction_quality_report(raw, correction, config)
    return payload
