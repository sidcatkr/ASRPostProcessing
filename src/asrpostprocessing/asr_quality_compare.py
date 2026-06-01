from __future__ import annotations

import copy
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .adapters import build_asr_adapter
from .asr_quality import build_asr_quality_report
from .config import ExperimentConfig
from .keyword_bias import build_keyword_bias_instruction
from .model_server import ensure_model_servers
from .preprocess import ffmpeg_executable, preprocess_audio


DEFAULT_COMPARE_CHUNK_SECONDS = [30.0, 60.0, 120.0]
DEFAULT_COMPARE_STRATEGIES = ["fixed"]


def run_asr_quality_compare(
    audio_path: str,
    base_config: ExperimentConfig,
    output_path: Optional[str] = None,
    chunk_seconds: Optional[Iterable[float]] = None,
    strategies: Optional[Iterable[str]] = None,
    preprocess_mode: str = "both",
    sample_seconds: Optional[float] = None,
    sample_start_s: float | Iterable[float] = 0.0,
) -> Path:
    output = Path(output_path) if output_path else Path(base_config.output_dir) / "asr_quality_compare.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    modes = _preprocess_modes(preprocess_mode)
    sample_starts_s = _sample_starts(sample_start_s, sample_seconds)
    for start_s in sample_starts_s:
        source_audio = _sample_audio(audio_path, base_config, sample_seconds, start_s)
        for mode in modes:
            for strategy in list(strategies or DEFAULT_COMPARE_STRATEGIES):
                for seconds in list(chunk_seconds or DEFAULT_COMPARE_CHUNK_SECONDS):
                    config = copy.deepcopy(base_config)
                    _apply_asr_compare_condition(config, mode, strategy, seconds)
                    server_statuses = []
                    if config.auto_start_model_servers:
                        server_statuses = [status.to_dict() for status in ensure_model_servers(config, names=["asr"])]
                    started = time.time()
                    preprocess_result = preprocess_audio(str(source_audio), config)
                    keyword_instruction = ""
                    if config.enable_keyword_bias:
                        keyword_instruction = build_keyword_bias_instruction(config.keywords, config.keyword_bias_weight)
                    raw = build_asr_adapter(config).transcribe(preprocess_result.audio_path, config, keyword_instruction=keyword_instruction)
                    elapsed_s = time.time() - started
                    quality = build_asr_quality_report(raw, preprocess_result.to_dict(), config)
                    rows.append(
                        {
                            "condition": _condition_name(mode, strategy, seconds),
                            "audio": str(source_audio),
                            "sample_start_s": float(start_s),
                            "sample_seconds": sample_seconds,
                            "preprocess_mode": mode,
                            "strategy": strategy,
                            "chunk_seconds": float(seconds),
                            "elapsed_s": elapsed_s,
                            "text_chars": len(raw.text or ""),
                            "text_preview": _preview(raw.text),
                            "asr_quality": quality,
                            "server_statuses": server_statuses,
                        }
                    )
    payload = {
        "audio": str(audio_path),
        "sample_audio": rows[0]["audio"] if len(rows) == 1 else "",
        "sample_start_s": sample_starts_s[0] if len(sample_starts_s) == 1 else None,
        "sample_starts_s": sample_starts_s,
        "sample_seconds": sample_seconds,
        "summary": _scan_summary(rows),
        "rows": rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _apply_asr_compare_condition(config: ExperimentConfig, mode: str, strategy: str, seconds: float) -> None:
    config.enable_llm_postprocess = False
    config.asr_chunking_strategy = strategy
    config.asr_chunk_seconds = float(seconds)
    if mode == "none":
        config.enable_preprocess = False
        config.preprocess_model = "none"
        config.enable_noise_reduction = False
        config.enable_volume_normalization = False
        config.noise_reduction_strength = 0.0
        config.volume_normalization_strength = 0.0


def _preprocess_modes(mode: str) -> List[str]:
    normalized = (mode or "both").strip().lower()
    if normalized == "both":
        return ["none", "configured"]
    if normalized in {"none", "configured"}:
        return [normalized]
    raise ValueError("preprocess_mode must be one of: none, configured, both")


def _sample_starts(sample_start_s: float | Iterable[float], sample_seconds: Optional[float]) -> List[float]:
    if sample_seconds is None:
        return [0.0]
    if isinstance(sample_start_s, (str, bytes)):
        values: Iterable[Any] = [sample_start_s]
    else:
        try:
            values = list(sample_start_s)  # type: ignore[arg-type]
        except TypeError:
            values = [sample_start_s]
    starts = []
    for value in values:
        try:
            start = max(0.0, float(value))
        except (TypeError, ValueError):
            continue
        if start not in starts:
            starts.append(start)
    return starts or [0.0]


def _sample_audio(audio_path: str, config: ExperimentConfig, sample_seconds: Optional[float], sample_start_s: float) -> Path:
    input_path = Path(audio_path)
    if sample_seconds is None:
        return input_path
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for --sample-seconds ASR quality comparison.")
    output_dir = Path(config.output_dir) / "asr_quality_compare"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in input_path.stem).strip("._") or "audio"
    output_path = output_dir / f"{safe_stem}.sample.{sample_start_s:g}-{sample_seconds:g}s.wav"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(sample_start_s)):.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{max(0.05, float(sample_seconds)):.3f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output_path


def _condition_name(mode: str, strategy: str, seconds: float) -> str:
    return f"asr_{mode}_{strategy}_{seconds:g}s"


def _scan_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    warning_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    risky_rows = []
    empty_count = 0
    near_miss_count = 0
    for row in rows:
        quality = row.get("asr_quality") if isinstance(row.get("asr_quality"), dict) else {}
        warnings = [str(item) for item in quality.get("warnings") or []]
        actions = [str(item) for item in quality.get("action_items") or []]
        near_misses = quality.get("keyword_near_misses") or []
        flags = []
        if int(row.get("text_chars") or 0) == 0:
            empty_count += 1
            flags.append("empty_transcript")
        if near_misses:
            near_miss_count += 1
            flags.append("keyword_near_miss")
        if warnings:
            flags.append("warnings")
            warning_counts.update(warnings)
        if actions:
            action_counts.update(actions)
        if flags:
            risky_rows.append(
                {
                    "condition": row.get("condition"),
                    "sample_start_s": row.get("sample_start_s"),
                    "sample_seconds": row.get("sample_seconds"),
                    "text_chars": row.get("text_chars"),
                    "flags": flags,
                    "warnings": warnings,
                    "keyword_near_misses": near_misses,
                    "text_preview": row.get("text_preview", ""),
                }
            )
    return {
        "row_count": len(rows),
        "sample_start_count": len({row.get("sample_start_s") for row in rows if row.get("sample_start_s") is not None}),
        "empty_transcript_rows": empty_count,
        "keyword_near_miss_rows": near_miss_count,
        "warning_rows": sum(1 for row in rows if (row.get("asr_quality") or {}).get("warnings")),
        "warning_counts": dict(warning_counts),
        "action_item_counts": dict(action_counts),
        "risky_rows": risky_rows,
    }


def _preview(text: str, limit: int = 500) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."
