from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .model_options import AUTO_EXPERIMENT_NOISE_MODELS, AUTO_EXPERIMENT_RAG_EMBEDDING_MODELS

DEFAULT_AUTO_EXPERIMENT_KEYWORD_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_AUTO_EXPERIMENT_STRENGTHS = [0.25, 0.5, 0.75]
DEFAULT_AUTO_EXPERIMENT_RAG_TOP_KS = [3, 5, 8, 12]


@dataclass
class ExperimentConfig:
    asr_model: str = "Qwen/Qwen3-ASR-1.7B"
    post_model: str = "Qwen/Qwen3.5-9B"
    asr_backend: str = "vllm_chat"
    post_backend: str = "vllm_openai"
    asr_base_url: str = "http://127.0.0.1:18000/v1"
    post_base_url: str = "http://127.0.0.1:18001/v1"
    asr_base_urls: List[str] = field(default_factory=list)
    post_base_urls: List[str] = field(default_factory=list)
    request_timeout_s: float = 120.0
    asr_request_timeout_s: float = 300.0
    asr_chunking_strategy: str = "silence"
    asr_chunk_seconds: float = 120.0
    asr_chunk_padding_seconds: float = 0.5
    asr_silence_threshold_db: float = -35.0
    asr_min_silence_seconds: float = 0.6
    asr_context_chars: int = 240
    asr_chunk_parallelism: int = 1
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
    server_gpu_memory_utilization: str = "auto"
    server_gpu_memory_utilization_max: float = 0.90
    server_gpu_memory_reserved_mb: int = 256
    pipeline_lanes: List[Dict[str, Any]] = field(default_factory=list)
    stage_server_base_urls: List[str] = field(default_factory=list)
    stage_server_gpus: List[str] = field(default_factory=list)
    postprocess_parallelism: int = 1
    sweep_parallelism: int = 1
    sweep_saturate_lanes: bool = True
    auto_experiment_parallelism: int = 1
    auto_experiment_saturate_lanes: bool = True
    auto_experiment_include_models: bool = False
    auto_experiment_asr_models: List[str] = field(default_factory=list)
    auto_experiment_post_models: List[str] = field(default_factory=list)
    auto_experiment_noise_models: List[str] = field(default_factory=lambda: list(AUTO_EXPERIMENT_NOISE_MODELS))
    auto_experiment_rag_embedding_models: List[str] = field(
        default_factory=lambda: list(AUTO_EXPERIMENT_RAG_EMBEDDING_MODELS)
    )
    auto_experiment_keyword_weights: List[float] = field(
        default_factory=lambda: list(DEFAULT_AUTO_EXPERIMENT_KEYWORD_WEIGHTS)
    )
    auto_experiment_noise_strengths: List[float] = field(default_factory=lambda: list(DEFAULT_AUTO_EXPERIMENT_STRENGTHS))
    auto_experiment_volume_strengths: List[float] = field(default_factory=lambda: list(DEFAULT_AUTO_EXPERIMENT_STRENGTHS))
    auto_experiment_postprocess_strengths: List[float] = field(default_factory=lambda: list(DEFAULT_AUTO_EXPERIMENT_STRENGTHS))
    auto_experiment_rag_strengths: List[float] = field(default_factory=lambda: list(DEFAULT_AUTO_EXPERIMENT_STRENGTHS))
    auto_experiment_rag_top_ks: List[int] = field(default_factory=lambda: list(DEFAULT_AUTO_EXPERIMENT_RAG_TOP_KS))
    auto_experiment_search_strengths: List[float] = field(default_factory=lambda: list(DEFAULT_AUTO_EXPERIMENT_STRENGTHS))
    asr_cache_enabled: bool = False
    preprocess_cache_enabled: bool = False
    cache_dir: str = "outputs/cache"
    upload_cache_enabled: bool = True
    upload_cache_dir: str = "outputs/upload_cache"

    enable_preprocess: bool = False
    preprocess_model: str = "none"
    preprocess_strength: float = 0.0
    preprocess_gpu: str = ""
    preprocess_gpus: List[str] = field(default_factory=list)
    enable_noise_reduction: bool = False
    noise_reduction_model: str = "none"
    noise_reduction_command: str = ""
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
        config.preprocess_gpu = str(config.preprocess_gpu or "").strip()
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
        config.asr_chunk_parallelism = max(1, min(64, int(config.asr_chunk_parallelism)))
        config.server_start_timeout_s = max(1.0, float(config.server_start_timeout_s))
        config.server_shutdown_timeout_s = max(1.0, float(config.server_shutdown_timeout_s))
        config.server_gpu_memory_utilization = str(config.server_gpu_memory_utilization or "auto").strip() or "auto"
        config.server_gpu_memory_utilization_max = max(
            0.05, min(0.99, float(config.server_gpu_memory_utilization_max))
        )
        config.server_gpu_memory_reserved_mb = max(0, int(config.server_gpu_memory_reserved_mb))
        config.postprocess_parallelism = max(1, min(64, int(config.postprocess_parallelism)))
        config.sweep_parallelism = max(1, min(64, int(config.sweep_parallelism)))
        config.auto_experiment_parallelism = max(1, min(64, int(config.auto_experiment_parallelism)))
        config.pipeline_lanes = normalize_pipeline_lanes(config.pipeline_lanes)
        config.asr_base_urls = normalize_url_list(config.asr_base_urls)
        config.post_base_urls = normalize_url_list(config.post_base_urls)
        config.stage_server_base_urls = normalize_url_list(config.stage_server_base_urls)
        config.stage_server_gpus = normalize_url_list(config.stage_server_gpus)
        config.preprocess_gpus = normalize_url_list(config.preprocess_gpus)
        config.auto_experiment_asr_models = normalize_url_list(config.auto_experiment_asr_models)
        config.auto_experiment_post_models = normalize_url_list(config.auto_experiment_post_models)
        config.auto_experiment_noise_models = normalize_url_list(config.auto_experiment_noise_models)
        config.auto_experiment_rag_embedding_models = normalize_url_list(
            config.auto_experiment_rag_embedding_models
        )
        config.auto_experiment_keyword_weights = normalize_strength_grid(
            config.auto_experiment_keyword_weights,
            DEFAULT_AUTO_EXPERIMENT_KEYWORD_WEIGHTS,
        )
        config.auto_experiment_noise_strengths = normalize_strength_grid(
            config.auto_experiment_noise_strengths,
            DEFAULT_AUTO_EXPERIMENT_STRENGTHS,
        )
        config.auto_experiment_volume_strengths = normalize_strength_grid(
            config.auto_experiment_volume_strengths,
            DEFAULT_AUTO_EXPERIMENT_STRENGTHS,
        )
        config.auto_experiment_postprocess_strengths = normalize_strength_grid(
            config.auto_experiment_postprocess_strengths,
            DEFAULT_AUTO_EXPERIMENT_STRENGTHS,
        )
        config.auto_experiment_rag_strengths = normalize_strength_grid(
            config.auto_experiment_rag_strengths,
            DEFAULT_AUTO_EXPERIMENT_STRENGTHS,
        )
        config.auto_experiment_rag_top_ks = normalize_int_grid(
            config.auto_experiment_rag_top_ks,
            DEFAULT_AUTO_EXPERIMENT_RAG_TOP_KS,
            minimum=1,
        )
        config.auto_experiment_search_strengths = normalize_strength_grid(
            config.auto_experiment_search_strengths,
            DEFAULT_AUTO_EXPERIMENT_STRENGTHS,
        )
        config.upload_cache_enabled = bool(config.upload_cache_enabled)
        config.upload_cache_dir = str(config.upload_cache_dir or "outputs/upload_cache")
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
        "all_gpus_per_stage": "stage_replicas",
        "stage": "stage_replicas",
        "stage_parallel": "stage_replicas",
        "stage_replicas": "stage_replicas",
        "stage_replicas_all_gpus": "stage_replicas",
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


def normalize_url_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_strength_grid(value: Any, fallback: List[float]) -> List[float]:
    if not value:
        return list(fallback)
    if isinstance(value, str):
        value = [part.strip() for part in value.replace("\n", ",").split(",")]
    if not isinstance(value, list):
        return list(fallback)
    strengths: List[float] = []
    for item in value:
        try:
            number = clamp01(item)
        except Exception:
            continue
        if number > 0.0 and number not in strengths:
            strengths.append(number)
    return strengths or list(fallback)


def normalize_int_grid(value: Any, fallback: List[int], minimum: int = 1) -> List[int]:
    if not value:
        return list(fallback)
    if isinstance(value, str):
        value = [part.strip() for part in value.replace("\n", ",").split(",")]
    if not isinstance(value, list):
        return list(fallback)
    numbers: List[int] = []
    for item in value:
        try:
            number = max(minimum, int(item))
        except (TypeError, ValueError):
            continue
        if number not in numbers:
            numbers.append(number)
    return numbers or list(fallback)


def normalize_pipeline_lanes(value: Any) -> List[Dict[str, Any]]:
    if not value:
        return []
    if not isinstance(value, list):
        return []
    lanes: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        lane = {str(key): val for key, val in item.items()}
        lane_name = str(lane.get("name") or f"lane_{index}").strip()
        if not lane_name:
            lane_name = f"lane_{index}"
        lane["name"] = lane_name
        lanes.append(lane)
    return lanes


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
