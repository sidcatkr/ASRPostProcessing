from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TranscriptSegment:
    text: str
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptResult:
    language: str
    text: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
            "metadata": self.metadata,
        }


@dataclass
class Edit:
    before: str
    after: str
    reason: str
    confidence: float = 0.0
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectionResult:
    corrected_text: str
    edits: List[Edit] = field(default_factory=list)
    risk: str = "unknown"
    used_context_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "corrected_text": self.corrected_text,
            "edits": [edit.to_dict() for edit in self.edits],
            "risk": self.risk,
            "used_context_ids": self.used_context_ids,
            "metadata": self.metadata,
        }


@dataclass
class MetricsResult:
    cer_normalized_no_space: Optional[float] = None
    cer_strict: Optional[float] = None
    wer_eojeol: Optional[float] = None
    raw_cer_normalized_no_space: Optional[float] = None
    raw_cer_strict: Optional[float] = None
    raw_wer_eojeol: Optional[float] = None
    delta_cer: Optional[float] = None
    delta_wer: Optional[float] = None
    semantic_similarity: Optional[float] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    query: str
    title: str
    url: str
    snippet: str
    source: str = "cache"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGContext:
    context_id: str
    text: str
    score: float
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
