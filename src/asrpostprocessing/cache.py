from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from .schemas import TranscriptResult, TranscriptSegment


@dataclass(frozen=True)
class CachedFile:
    source_path: str
    cached_path: str
    sha256: str
    size_bytes: int
    cache_hit: bool
    link_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def cache_file_by_sha256(source_path: str | Path, cache_dir: str | Path, namespace: str = "files") -> CachedFile:
    source = Path(source_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(str(source))
    digest = file_sha256(source)
    size_bytes = source.stat().st_size
    suffix = _safe_suffix(source.suffix)
    target_dir = Path(cache_dir) / namespace / digest[:2]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{digest}{suffix}"
    if target.exists() and target.stat().st_size == size_bytes:
        return CachedFile(
            source_path=str(source),
            cached_path=str(target),
            sha256=digest,
            size_bytes=size_bytes,
            cache_hit=True,
            link_type="existing",
        )
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
    if tmp.exists():
        tmp.unlink()
    link_type = "copy"
    try:
        os.link(source, tmp)
        link_type = "hardlink"
    except OSError:
        shutil.copy2(source, tmp)
    try:
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return CachedFile(
        source_path=str(source),
        cached_path=str(target),
        sha256=digest,
        size_bytes=size_bytes,
        cache_hit=False,
        link_type=link_type,
    )


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


def _safe_suffix(value: str) -> str:
    suffix = (value or "").strip().lower()
    if not suffix.startswith("."):
        return ".bin"
    safe = "." + "".join(char for char in suffix[1:] if char.isalnum())
    return safe if len(safe) > 1 else ".bin"


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
