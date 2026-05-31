from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .schemas import TranscriptSegment


@dataclass
class TextChunk:
    index: int
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？\n])\s+")


def chunk_text(text: str, max_chars: int = 700, overlap: int = 80) -> List[TextChunk]:
    text = text or ""
    if len(text) <= max_chars:
        return [TextChunk(index=0, text=text, start_char=0, end_char=len(text))]

    chunks: List[TextChunk] = []
    start = 0
    index = 0
    while start < len(text):
        window_end = min(len(text), start + max_chars)
        end = _best_boundary(text, start, window_end)
        if end <= start:
            end = window_end
        chunks.append(TextChunk(index=index, text=text[start:end], start_char=start, end_char=end))
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start
        index += 1
    return chunks


def chunk_segments(segments: List[TranscriptSegment], min_seconds: float = 30.0, max_seconds: float = 90.0) -> List[TextChunk]:
    if not segments:
        return []
    chunks: List[TextChunk] = []
    current_texts: List[str] = []
    start_s = segments[0].start_s
    end_s = segments[0].end_s
    for segment in segments:
        if not current_texts:
            start_s = segment.start_s
        current_texts.append(segment.text)
        end_s = segment.end_s
        duration = 0.0
        if start_s is not None and end_s is not None:
            duration = end_s - start_s
        if duration >= max_seconds or (duration >= min_seconds and segment.text.endswith((".", "?", "!"))):
            chunks.append(
                TextChunk(
                    index=len(chunks),
                    text=" ".join(current_texts),
                    start_char=0,
                    end_char=0,
                    metadata={"start_s": start_s, "end_s": end_s},
                )
            )
            current_texts = []
    if current_texts:
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=" ".join(current_texts),
                start_char=0,
                end_char=0,
                metadata={"start_s": start_s, "end_s": end_s},
            )
        )
    return chunks


def _best_boundary(text: str, start: int, window_end: int) -> int:
    window = text[start:window_end]
    minimum = max(1, int(len(window) * 0.45))
    candidates = [match.end() for match in SENTENCE_BOUNDARY_RE.finditer(window) if match.end() >= minimum]
    if candidates:
        return start + candidates[-1]
    for marker in [".", "?", "!", "\n", "。", "！", "？"]:
        pos = window.rfind(marker)
        if pos >= minimum:
            return start + pos + 1
    pos = window.rfind(" ")
    if pos >= minimum:
        return start + pos
    return window_end
