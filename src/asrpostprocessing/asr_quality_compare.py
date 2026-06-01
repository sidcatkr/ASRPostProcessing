from __future__ import annotations

import copy
import json
import subprocess
import time
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
    sample_start_s: float = 0.0,
) -> Path:
    source_audio = _sample_audio(audio_path, base_config, sample_seconds, sample_start_s)
    output = Path(output_path) if output_path else Path(base_config.output_dir) / "asr_quality_compare.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    modes = _preprocess_modes(preprocess_mode)
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
        "sample_audio": str(source_audio),
        "sample_start_s": float(sample_start_s),
        "sample_seconds": sample_seconds,
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


def _preview(text: str, limit: int = 500) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."
