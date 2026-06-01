from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentConfig:
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    post_model: str = "Qwen/Qwen3.5-9B"
    asr_backend: str = "vllm_chat"
    post_backend: str = "vllm_openai"
    asr_base_url: str = "http://127.0.0.1:18000/v1"
    post_base_url: str = "http://127.0.0.1:18001/v1"
    request_timeout_s: float = 120.0
    asr_request_timeout_s: float = 300.0
    asr_chunking_strategy: str = "silence"
    asr_chunk_seconds: float = 30.0
    asr_chunk_padding_seconds: float = 0.5
    asr_silence_threshold_db: float = -35.0
    asr_min_silence_seconds: float = 0.6
    asr_context_chars: int = 240
    language: str = "ko"
    auto_start_model_servers: bool = False
    model_residency: str = "parallel"
    server_start_timeout_s: float = 600.0
    server_shutdown_timeout_s: float = 30.0
    server_log_dir: str = "outputs/model_servers"
    asr_server_gpu: str = "0"
    post_server_gpu: str = "1"
    asr_server_host: str = "0.0.0.0"
    post_server_host: str = "0.0.0.0"
    asr_server_command: str = ""
    post_server_command: str = ""

    enable_preprocess: bool = False
    preprocess_model: str = "none"
    preprocess_strength: float = 0.0
    enable_noise_reduction: bool = False
    noise_reduction_model: str = "none"
    noise_reduction_strength: float = 0.0
    enable_volume_normalization: bool = False
    volume_normalization_strength: float = 0.0
    volume_target_dbfs: float = -20.0

    enable_keyword_bias: bool = False
    keyword_bias_weight: float = 0.0
    keywords: List[str] = field(default_factory=list)

    enable_llm_postprocess: bool = True
    postprocess_strength: float = 0.5

    enable_rag: bool = False
    rag_strength: float = 0.0
    rag_top_k: int = 5
    rag_files: List[str] = field(default_factory=list)
    rag_inline_text: str = ""
    rag_embedding_backend: str = "lexical"
    rag_embedding_model: str = "intfloat/multilingual-e5-base"

    enable_search: bool = False
    search_strength: float = 0.0
    search_provider: str = "duckduckgo"
    search_cache_dir: str = "outputs/search_cache"
    search_endpoint: str = ""

    output_dir: str = "outputs"
    runs_dir: str = "runs"
    tensorboard_port: int = 6006
    chunk_max_chars: int = 700
    chunk_overlap: int = 80
    run_name: str = ""
    mock_transcript: str = "테스트 전사 문장입니다."

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "ExperimentConfig":
        known = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in mapping.items() if key in known}
        config = cls(**filtered)
        config.preprocess_strength = clamp01(config.preprocess_strength)
        config.noise_reduction_strength = clamp01(config.noise_reduction_strength)
        config.volume_normalization_strength = clamp01(config.volume_normalization_strength)
        config.keyword_bias_weight = clamp01(config.keyword_bias_weight)
        config.postprocess_strength = clamp01(config.postprocess_strength)
        config.rag_strength = clamp01(config.rag_strength)
        config.search_strength = clamp01(config.search_strength)
        config.model_residency = normalize_model_residency(config.model_residency)
        config.rag_top_k = max(1, int(config.rag_top_k))
        config.chunk_max_chars = max(120, int(config.chunk_max_chars))
        config.chunk_overlap = max(0, min(int(config.chunk_overlap), config.chunk_max_chars // 2))
        config.asr_request_timeout_s = max(30.0, float(config.asr_request_timeout_s))
        config.asr_chunking_strategy = normalize_asr_chunking_strategy(config.asr_chunking_strategy)
        config.asr_chunk_seconds = max(5.0, float(config.asr_chunk_seconds))
        config.asr_chunk_padding_seconds = max(0.0, min(5.0, float(config.asr_chunk_padding_seconds)))
        config.asr_silence_threshold_db = max(-80.0, min(-10.0, float(config.asr_silence_threshold_db)))
        config.asr_min_silence_seconds = max(0.1, min(5.0, float(config.asr_min_silence_seconds)))
        config.asr_context_chars = max(0, min(2000, int(config.asr_context_chars)))
        config.server_start_timeout_s = max(1.0, float(config.server_start_timeout_s))
        config.server_shutdown_timeout_s = max(1.0, float(config.server_shutdown_timeout_s))
        return config


def clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, number))


def normalize_model_residency(value: Any) -> str:
    normalized = str(value or "parallel").strip().lower().replace("-", "_")
    aliases = {
        "all": "parallel",
        "all_at_once": "parallel",
        "fast": "parallel",
        "parallel": "parallel",
        "resident": "parallel",
        "single": "sequential",
        "single_model": "sequential",
        "low_vram": "sequential",
        "low_memory": "sequential",
        "one_at_a_time": "sequential",
        "sequential": "sequential",
    }
    return aliases.get(normalized, "parallel")


def normalize_asr_chunking_strategy(value: Any) -> str:
    normalized = str(value or "silence").strip().lower().replace("-", "_")
    aliases = {
        "auto": "silence",
        "vad": "silence",
        "silence": "silence",
        "silence_aware": "silence",
        "speech": "silence",
        "fixed": "fixed",
        "segment": "fixed",
        "segments": "fixed",
        "none": "none",
        "no": "none",
        "off": "none",
        "disabled": "none",
        "disable": "none",
        "false": "none",
    }
    return aliases.get(normalized, "silence")


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    data: Dict[str, Any] = {}
    if path:
        config_path = Path(path)
        raw = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            data.update(json.loads(raw))
        else:
            data.update(_load_yaml_or_minimal(raw))
    if overrides:
        data.update({key: value for key, value in overrides.items() if value is not None})
    env_asr = os.environ.get("ASRPP_ASR_BASE_URL")
    env_post = os.environ.get("ASRPP_POST_BASE_URL")
    if env_asr and "asr_base_url" not in data:
        data["asr_base_url"] = env_asr
    if env_post and "post_base_url" not in data:
        data["post_base_url"] = env_post
    return ExperimentConfig.from_mapping(data)


def dump_resolved_yaml(config: ExperimentConfig) -> str:
    return _dump_simple_yaml(config.to_dict())


def _load_yaml_or_minimal(raw: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw)
        return loaded or {}
    except Exception:
        return _load_minimal_yaml(raw)


def _load_minimal_yaml(raw: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list_key:
            data.setdefault(current_list_key, []).append(_coerce_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = []
            current_list_key = key
        else:
            data[key] = _coerce_scalar(value)
            current_list_key = None
    return data


def _coerce_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return [part.strip() for part in value[1:-1].split(",") if part.strip()]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def _dump_simple_yaml(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key in sorted(data):
        value = data[key]
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, str) else value}")
    return "\n".join(lines) + "\n"
