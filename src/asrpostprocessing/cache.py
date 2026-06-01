from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import TranscriptResult, TranscriptSegment


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cache_json_path(cache_dir: str | Path, namespace: str, key: str) -> Path:
    path = Path(cache_dir) / namespace / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def transcript_from_dict(payload: Dict[str, Any]) -> TranscriptResult:
    segments = [
        TranscriptSegment(
            text=str(item.get("text", "")),
            start_s=item.get("start_s"),
            end_s=item.get("end_s"),
            confidence=item.get("confidence"),
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
        )
        for item in payload.get("segments", [])
        if isinstance(item, dict)
    ]
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return TranscriptResult(
        language=str(payload.get("language") or "ko"),
        text=str(payload.get("text") or ""),
        segments=segments,
        metadata=metadata,
    )
