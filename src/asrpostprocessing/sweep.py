from __future__ import annotations

import copy
import csv
import itertools
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import ExperimentConfig
from .model_server import ensure_model_servers
from .pipeline import PipelineRunner, read_reference

DEFAULT_KEYWORD_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_RAG_STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_POST_STRENGTHS = [0.25, 0.5, 0.75]
DEFAULT_PREPROCESS_STRENGTHS = [0.25, 0.5, 0.75]

CONDITIONS = [
    "A_raw_asr",
    "B1_noise_reduction_raw_asr",
    "B2_volume_normalization_raw_asr",
    "B3_noise_volume_raw_asr",
    "C_llm_only",
    "D_rag_llm",
    "E_keyword_bias_llm",
    "F_keyword_bias_rag_llm",
    "G_search_rag_llm",
]


def run_sweep(
    manifest_path: str,
    base_config: ExperimentConfig,
    keyword_weights: Optional[Iterable[float]] = None,
    rag_strengths: Optional[Iterable[float]] = None,
    post_strengths: Optional[Iterable[float]] = None,
    noise_strengths: Optional[Iterable[float]] = None,
    volume_strengths: Optional[Iterable[float]] = None,
    jobs: Optional[int] = None,
) -> Path:
    rows = _read_manifest(manifest_path)
    summary_dir = Path(base_config.output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "sweep_summary.csv"
    keyword_weights = list(keyword_weights or DEFAULT_KEYWORD_WEIGHTS)
    rag_strengths = list(rag_strengths or DEFAULT_RAG_STRENGTHS)
    post_strengths = list(post_strengths or DEFAULT_POST_STRENGTHS)
    noise_strengths = list(noise_strengths or DEFAULT_PREPROCESS_STRENGTHS)
    volume_strengths = list(volume_strengths or DEFAULT_PREPROCESS_STRENGTHS)
    work_items = _sweep_work_items(
        rows,
        base_config,
        keyword_weights,
        rag_strengths,
        post_strengths,
        noise_strengths,
        volume_strengths,
    )
    max_workers = _sweep_worker_count(base_config, jobs)
    if max_workers > 1 and base_config.auto_start_model_servers and base_config.model_residency == "parallel":
        ensure_model_servers(base_config)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_id",
            "condition",
            "audio",
            "lane_id",
            "asr_endpoint",
            "post_endpoint",
            "post_endpoint_pool",
            "asr_model",
            "post_model",
            "asr_backend",
            "post_backend",
            "preprocess_model",
            "noise_reduction_model",
            "keyword_bias_weight",
            "noise_reduction_strength",
            "volume_normalization_strength",
            "rag_strength",
            "postprocess_strength",
            "model_residency",
            "cer_normalized_no_space",
            "wer_eojeol",
            "delta_cer",
            "delta_wer",
            "semantic_similarity",
            "risk",
            "latency_ms",
            "server_readiness_ms",
            "preprocess_latency_ms",
            "asr_latency_ms",
            "postprocess_latency_ms",
            "queue_wait_ms",
            "audio_duration_s",
            "audio_seconds_per_second",
            "tokens_per_second",
            "post_output_tokens",
            "vllm_preemption_count",
            "vllm_prompt_tokens",
            "vllm_generation_tokens",
            "vllm_total_tokens",
            "vllm_request_success_count",
            "vllm_metrics_available",
            "peak_vram_mb",
            "peak_gpu_utilization_percent",
            "asr_cache_hit",
            "preprocess_cache_hit",
            "output_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        summary_rows: List[Dict[str, object]] = []
        if max_workers <= 1:
            for item in work_items:
                summary_row = _run_sweep_item(item)
                summary_rows.append(summary_row)
                writer.writerow(summary_row)
                handle.flush()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {executor.submit(_run_sweep_item, item): item for item in work_items}
                for future in as_completed(future_to_item):
                    summary_row = future.result()
                    summary_rows.append(summary_row)
                    writer.writerow(summary_row)
                    handle.flush()
    analysis_path = summary_dir / "sweep_analysis.json"
    analysis_path.write_text(json.dumps(analyze_sweep(summary_rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def shard_manifest(manifest_path: str, num_shards: int, out_dir: str, prefix: str = "shard") -> List[Path]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not fieldnames:
        raise ValueError("Manifest must include a header row.")
    for row in rows:
        if not row.get("audio"):
            raise ValueError("Manifest rows must include an audio column.")
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for shard_index in range(num_shards):
        path = output_dir / f"{prefix}_{shard_index}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows[shard_index::num_shards]:
                writer.writerow(row)
        paths.append(path)
    return paths


def _sweep_work_items(
    rows: List[Dict[str, str]],
    base_config: ExperimentConfig,
    keyword_weights: Iterable[float],
    rag_strengths: Iterable[float],
    post_strengths: Iterable[float],
    noise_strengths: Iterable[float],
    volume_strengths: Iterable[float],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row in rows:
        for condition, keyword_weight, rag_strength, post_strength, noise_strength, volume_strength in _condition_grid(
            keyword_weights,
            rag_strengths,
            post_strengths,
            noise_strengths,
            volume_strengths,
        ):
            config = _config_for_condition(
                base_config,
                condition,
                keyword_weight,
                rag_strength,
                post_strength,
                noise_strength,
                volume_strength,
            )
            lane = _lane_for_case(base_config, len(items))
            if lane:
                _apply_lane_to_config(config, lane)
            items.append(
                {
                    "row": row,
                    "config": config,
                    "lane": lane,
                    "condition": condition,
                    "keyword_weight": keyword_weight,
                    "rag_strength": rag_strength,
                    "post_strength": post_strength,
                    "noise_strength": noise_strength,
                    "volume_strength": volume_strength,
                }
            )
    return items


def _run_sweep_item(item: Dict[str, Any]) -> Dict[str, object]:
    row = item["row"]
    config = item["config"]
    reference = row.get("reference_text") or read_reference(row.get("reference"))
    output = PipelineRunner(config).run(
        audio_path=row["audio"],
        reference_text=reference,
        rag_inline_text=row.get("rag_inline_text", ""),
    )
    metrics = output.metrics.to_dict()
    timings = output.timings or {}
    hardware = output.hardware or {}
    vllm_delta = (output.vllm_metrics or {}).get("delta") if isinstance(output.vllm_metrics, dict) else {}
    if not isinstance(vllm_delta, dict):
        vllm_delta = {}
    vllm_total_tokens = _metric_from_mapping(vllm_delta, "total_tokens")
    vllm_prompt_tokens = _metric_from_mapping(vllm_delta, "prompt_tokens")
    vllm_generation_tokens = _metric_from_mapping(vllm_delta, "generation_tokens")
    vllm_request_success_count = _metric_from_mapping(vllm_delta, "request_success_count")
    vllm_preemption_count = _metric_from_mapping(vllm_delta, "preemption_count")
    post_output_tokens = _post_output_tokens(output.correction.metadata)
    postprocess_latency_ms = _metric_from_mapping(timings, "postprocess_latency_ms")
    latency_ms = _metric_from_mapping(timings, "latency_ms") or _metric_from_mapping(metrics, "latency_ms")
    token_count = post_output_tokens if post_output_tokens is not None else vllm_total_tokens
    token_latency_ms = postprocess_latency_ms if post_output_tokens is not None else latency_ms
    return {
        "run_id": output.run_id,
        "condition": item["condition"],
        "audio": row["audio"],
        "lane_id": (item.get("lane") or {}).get("lane_id", ""),
        "asr_endpoint": config.asr_base_url,
        "post_endpoint": config.post_base_url,
        "post_endpoint_pool": ",".join(_post_endpoint_pool(config)),
        "asr_model": config.asr_model,
        "post_model": config.post_model if config.enable_llm_postprocess else "",
        "asr_backend": config.asr_backend,
        "post_backend": config.post_backend if config.enable_llm_postprocess else "",
        "preprocess_model": config.preprocess_model,
        "noise_reduction_model": config.noise_reduction_model if config.enable_noise_reduction else "",
        "keyword_bias_weight": item["keyword_weight"],
        "noise_reduction_strength": item["noise_strength"],
        "volume_normalization_strength": item["volume_strength"],
        "rag_strength": item["rag_strength"],
        "postprocess_strength": item["post_strength"],
        "model_residency": config.model_residency,
        "cer_normalized_no_space": metrics.get("cer_normalized_no_space"),
        "wer_eojeol": metrics.get("wer_eojeol"),
        "delta_cer": metrics.get("delta_cer"),
        "delta_wer": metrics.get("delta_wer"),
        "semantic_similarity": metrics.get("semantic_similarity"),
        "risk": output.correction.risk,
        "latency_ms": metrics.get("latency_ms"),
        "server_readiness_ms": timings.get("server_readiness_ms"),
        "preprocess_latency_ms": timings.get("preprocess_latency_ms"),
        "asr_latency_ms": timings.get("asr_latency_ms"),
        "postprocess_latency_ms": timings.get("postprocess_latency_ms"),
        "queue_wait_ms": "",
        "audio_duration_s": timings.get("audio_duration_s"),
        "audio_seconds_per_second": timings.get("audio_seconds_per_second"),
        "tokens_per_second": (
            token_count / (token_latency_ms / 1000.0)
            if token_count is not None and token_latency_ms and token_latency_ms > 0.0
            else ""
        ),
        "post_output_tokens": post_output_tokens if post_output_tokens is not None else "",
        "vllm_preemption_count": vllm_preemption_count if vllm_preemption_count is not None else "",
        "vllm_prompt_tokens": vllm_prompt_tokens if vllm_prompt_tokens is not None else "",
        "vllm_generation_tokens": vllm_generation_tokens if vllm_generation_tokens is not None else "",
        "vllm_total_tokens": vllm_total_tokens if vllm_total_tokens is not None else "",
        "vllm_request_success_count": vllm_request_success_count if vllm_request_success_count is not None else "",
        "vllm_metrics_available": bool((output.vllm_metrics or {}).get("available"))
        if isinstance(output.vllm_metrics, dict)
        else False,
        "peak_vram_mb": hardware.get("observed_peak_vram_mb"),
        "peak_gpu_utilization_percent": hardware.get("observed_peak_gpu_utilization_percent"),
        "asr_cache_hit": _cache_hit(output.raw.metadata.get("asr_cache")),
        "preprocess_cache_hit": _cache_hit(output.preprocess.get("metadata", {}).get("cache_hit")),
        "output_dir": output.output_dir,
    }


def _sweep_worker_count(config: ExperimentConfig, jobs: Optional[int]) -> int:
    requested = int(jobs) if jobs is not None else int(getattr(config, "sweep_parallelism", 1) or 1)
    requested = max(1, min(64, requested))
    if not bool(getattr(config, "sweep_saturate_lanes", True)):
        return requested
    lane_count = max(
        1,
        len([lane for lane in (config.pipeline_lanes or []) if isinstance(lane, dict)]),
        len(config.asr_base_urls or []),
        len(config.post_base_urls or []),
    )
    return max(requested, lane_count)


def _lane_for_case(config: ExperimentConfig, case_index: int) -> Dict[str, str]:
    lanes = [lane for lane in (config.pipeline_lanes or []) if isinstance(lane, dict)]
    if lanes:
        index = case_index % len(lanes)
        lane = lanes[index]
        return {
            "lane_id": str(lane.get("name") or lane.get("id") or f"lane_{index}"),
            "asr_base_url": str(lane.get("asr_base_url") or ""),
            "post_base_url": str(lane.get("post_base_url") or ""),
        }
    asr_urls = [str(url).strip() for url in (config.asr_base_urls or []) if str(url).strip()]
    post_urls = [str(url).strip() for url in (config.post_base_urls or []) if str(url).strip()]
    endpoint_count = max(len(asr_urls), len(post_urls))
    if endpoint_count <= 0:
        return {}
    index = case_index % endpoint_count
    return {
        "lane_id": f"endpoint_{index}",
        "asr_base_url": asr_urls[index % len(asr_urls)] if asr_urls else "",
        "post_base_url": post_urls[index % len(post_urls)] if post_urls else "",
    }


def _apply_lane_to_config(config: ExperimentConfig, lane: Dict[str, str]) -> None:
    if lane.get("asr_base_url"):
        config.asr_base_url = lane["asr_base_url"]
    if lane.get("post_base_url"):
        config.post_base_url = lane["post_base_url"]


def _post_endpoint_pool(config: ExperimentConfig) -> List[str]:
    endpoints = [
        str(lane.get("post_base_url")).strip()
        for lane in (config.pipeline_lanes or [])
        if isinstance(lane, dict) and str(lane.get("post_base_url") or "").strip()
    ]
    endpoints.extend(str(url).strip() for url in (config.post_base_urls or []) if str(url).strip())
    endpoints.append(config.post_base_url)
    deduped: List[str] = []
    for endpoint in endpoints:
        if endpoint and endpoint not in deduped:
            deduped.append(endpoint)
    return deduped


def _post_output_tokens(metadata: Dict[str, Any]) -> Optional[int]:
    total = 0
    found = False
    for chunk in metadata.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        chunk_metadata = chunk.get("metadata")
        if not isinstance(chunk_metadata, dict):
            continue
        usage = (chunk_metadata.get("raw") or {}).get("usage") if isinstance(chunk_metadata.get("raw"), dict) else None
        if not isinstance(usage, dict):
            continue
        value = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("total_tokens")
        try:
            total += int(value)
            found = True
        except (TypeError, ValueError):
            continue
    return total if found else None


def _metric_from_mapping(values: Dict[str, Any], key: str) -> Optional[float]:
    value = values.get(key)
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cache_hit(value: Any) -> object:
    if isinstance(value, dict):
        return value.get("hit", "")
    if isinstance(value, bool):
        return value
    return ""


def _read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if not row.get("audio"):
            raise ValueError("Manifest rows must include an audio column.")
    return rows


def _condition_grid(
    keyword_weights: Iterable[float],
    rag_strengths: Iterable[float],
    post_strengths: Iterable[float],
    noise_strengths: Iterable[float],
    volume_strengths: Iterable[float],
) -> Iterable[Tuple[str, float, float, float, float, float]]:
    yield ("A_raw_asr", 0.0, 0.0, 0.0, 0.0, 0.0)
    for noise_strength in noise_strengths:
        yield ("B1_noise_reduction_raw_asr", 0.0, 0.0, 0.0, float(noise_strength), 0.0)
    for volume_strength in volume_strengths:
        yield ("B2_volume_normalization_raw_asr", 0.0, 0.0, 0.0, 0.0, float(volume_strength))
    for noise_strength, volume_strength in itertools.product(noise_strengths, volume_strengths):
        yield ("B3_noise_volume_raw_asr", 0.0, 0.0, 0.0, float(noise_strength), float(volume_strength))
    for post_strength in post_strengths:
        yield ("C_llm_only", 0.0, 0.0, float(post_strength), 0.0, 0.0)
    for rag_strength, post_strength in itertools.product(rag_strengths, post_strengths):
        yield ("D_rag_llm", 0.0, float(rag_strength), float(post_strength), 0.0, 0.0)
    for keyword_weight, post_strength in itertools.product(keyword_weights, post_strengths):
        yield ("E_keyword_bias_llm", float(keyword_weight), 0.0, float(post_strength), 0.0, 0.0)
    for keyword_weight, rag_strength, post_strength in itertools.product(keyword_weights, rag_strengths, post_strengths):
        yield ("F_keyword_bias_rag_llm", float(keyword_weight), float(rag_strength), float(post_strength), 0.0, 0.0)
    for rag_strength, post_strength in itertools.product(rag_strengths, post_strengths):
        yield ("G_search_rag_llm", 0.0, float(rag_strength), float(post_strength), 0.0, 0.0)


def _config_for_condition(
    base_config: ExperimentConfig,
    condition: str,
    keyword_weight: float,
    rag_strength: float,
    post_strength: float,
    noise_strength: float,
    volume_strength: float,
) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    config.enable_preprocess = False
    config.enable_noise_reduction = condition in {"B1_noise_reduction_raw_asr", "B3_noise_volume_raw_asr"}
    config.enable_volume_normalization = condition in {"B2_volume_normalization_raw_asr", "B3_noise_volume_raw_asr"}
    config.noise_reduction_model = base_config.noise_reduction_model if base_config.noise_reduction_model != "none" else "rnnoise"
    config.noise_reduction_strength = noise_strength
    config.volume_normalization_strength = volume_strength
    config.enable_llm_postprocess = condition not in {"A_raw_asr", "B1_noise_reduction_raw_asr", "B2_volume_normalization_raw_asr", "B3_noise_volume_raw_asr"}
    config.enable_keyword_bias = condition in {"E_keyword_bias_llm", "F_keyword_bias_rag_llm"} and keyword_weight > 0
    config.enable_rag = condition in {"D_rag_llm", "F_keyword_bias_rag_llm", "G_search_rag_llm"} and rag_strength > 0
    config.enable_search = condition == "G_search_rag_llm"
    config.keyword_bias_weight = keyword_weight
    config.rag_strength = rag_strength
    config.postprocess_strength = post_strength
    return config


def analyze_sweep(rows: List[Dict[str, object]]) -> Dict[str, object]:
    comparable = [row for row in rows if _metric(row, "cer_normalized_no_space") is not None]
    best = _best_by(comparable, "cer_normalized_no_space")
    best_by_wer = _best_by(comparable, "wer_eojeol")
    semantic_safe = [
        row
        for row in comparable
        if _metric_or(row, "semantic_similarity", 1.0) >= 0.85 and str(row.get("risk", "")).lower() != "high"
    ]
    best_semantic_safe = _best_by(semantic_safe, "cer_normalized_no_space")
    best_latency_quality_tradeoff = _best_latency_quality_tradeoff(comparable, best)
    raw_by_audio = {
        str(row["audio"]): _metric(row, "cer_normalized_no_space")
        for row in comparable
        if row.get("condition") == "A_raw_asr"
    }
    zero_weight_by_condition = {
        _zero_key(row): _metric(row, "cer_normalized_no_space")
        for row in comparable
        if _is_zero_weight_row(row)
    }
    lowest_post_by_condition: Dict[Tuple[str, str, float, float], Tuple[float, Optional[float]]] = {}
    for row in comparable:
        key = _post_strength_key(row)
        post_strength = _metric_or(row, "postprocess_strength", 0.0)
        cer_value = _metric(row, "cer_normalized_no_space")
        current = lowest_post_by_condition.get(key)
        if current is None or post_strength < current[0]:
            lowest_post_by_condition[key] = (post_strength, cer_value)
    worse_than_raw_cases = []
    over_keyword_cases = []
    over_rag_cases = []
    over_postprocess_cases = []
    over_preprocess_cases = []
    for row in comparable:
        audio = str(row["audio"])
        raw_cer = raw_by_audio.get(audio)
        cer_value = _metric(row, "cer_normalized_no_space")
        if raw_cer is not None and cer_value is not None and cer_value > raw_cer:
            worse_than_raw_cases.append({**row, "worse_than_raw_reason": "cer_above_raw_asr"})
            if _is_preprocess_row(row):
                over_preprocess_cases.append({**row, "over_preprocess_reason": "worse_than_raw_asr"})
        zero_cer = zero_weight_by_condition.get(_zero_key(row))
        if zero_cer is not None and cer_value is not None and cer_value > zero_cer:
            if _metric_or(row, "keyword_bias_weight", 0.0) > 0.0:
                over_keyword_cases.append({**row, "over_bias_reason": "worse_than_zero_weight"})
        if _metric_or(row, "rag_strength", 0.0) > 0.0 and zero_cer is not None and cer_value is not None and cer_value > zero_cer:
            over_rag_cases.append({**row, "over_rag_reason": "worse_than_rag_zero"})
        post_baseline = lowest_post_by_condition.get(_post_strength_key(row))
        if post_baseline is not None:
            baseline_strength, baseline_cer = post_baseline
            post_strength = _metric_or(row, "postprocess_strength", 0.0)
            if (
                post_strength > baseline_strength
                and baseline_cer is not None
                and cer_value is not None
                and cer_value > baseline_cer
            ):
                over_postprocess_cases.append({**row, "over_postprocess_reason": "worse_than_lowest_post_strength"})
    return {
        "best_by_cer": best,
        "best_by_wer": best_by_wer,
        "best_by_cer_under_semantic_risk_threshold": best_semantic_safe,
        "best_latency_quality_tradeoff": best_latency_quality_tradeoff,
        "worse_than_raw_cases": worse_than_raw_cases,
        "over_bias_cases": over_keyword_cases,
        "over_keyword_cases": over_keyword_cases,
        "over_rag_cases": over_rag_cases,
        "over_postprocess_cases": over_postprocess_cases,
        "over_preprocess_cases": over_preprocess_cases,
        "num_rows": len(rows),
        "num_comparable_rows": len(comparable),
    }


def _metric(row: Dict[str, object], key: str):
    value = row.get(key)
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _best_by(rows: List[Dict[str, object]], metric_key: str):
    return min(
        [row for row in rows if _metric(row, metric_key) is not None],
        key=lambda row: (
            _metric_or(row, metric_key, 999.0),
            _semantic_drift(row),
            _metric_or(row, "latency_ms", 999999.0),
        ),
        default=None,
    )


def _best_latency_quality_tradeoff(rows: List[Dict[str, object]], best_row: Optional[Dict[str, object]]):
    best_cer = _metric(best_row or {}, "cer_normalized_no_space")
    if best_cer is None:
        return None
    tolerance = max(0.005, abs(best_cer) * 0.05)
    near_best = [
        row
        for row in rows
        if _metric_or(row, "cer_normalized_no_space", 999.0) <= best_cer + tolerance
        and _metric(row, "latency_ms") is not None
    ]
    return min(
        near_best,
        key=lambda row: (
            _metric_or(row, "latency_ms", 999999.0),
            _metric_or(row, "cer_normalized_no_space", 999.0),
            _semantic_drift(row),
        ),
        default=best_row,
    )


def _semantic_drift(row: Dict[str, object]) -> float:
    similarity = _metric(row, "semantic_similarity")
    if similarity is None:
        return 1.0
    return 1.0 - similarity


def _metric_or(row: Dict[str, object], key: str, default: float) -> float:
    value = _metric(row, key)
    return default if value is None else value


def _is_zero_weight_row(row: Dict[str, object]) -> bool:
    return (
        _metric_or(row, "keyword_bias_weight", 0.0) == 0.0
        and _metric_or(row, "rag_strength", 0.0) == 0.0
    )


def _zero_key(row: Dict[str, object]) -> Tuple[str, str, float]:
    return (
        str(row.get("audio", "")),
        str(row.get("condition", "")),
        _metric_or(row, "postprocess_strength", 0.0),
    )


def _post_strength_key(row: Dict[str, object]) -> Tuple[str, str, float, float]:
    return (
        str(row.get("audio", "")),
        str(row.get("condition", "")),
        _metric_or(row, "keyword_bias_weight", 0.0),
        _metric_or(row, "rag_strength", 0.0),
    )


def _is_preprocess_row(row: Dict[str, object]) -> bool:
    return (
        _metric_or(row, "noise_reduction_strength", 0.0) > 0.0
        or _metric_or(row, "volume_normalization_strength", 0.0) > 0.0
        or str(row.get("condition", "")).startswith("B")
    )
