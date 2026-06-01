from __future__ import annotations

import copy
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import ExperimentConfig
from .experiment_matrix import ConditionSpec, generate_auto_conditions
from .logging import make_run_id
from .pipeline import PipelineRunner

StatusCallback = Callable[[str], None]


def run_auto_experiment(
    audio_path: str,
    base_config: ExperimentConfig,
    reference_text: Optional[str] = None,
    rag_inline_text: str = "",
    mode: str = "full_valid",
    status_callback: Optional[StatusCallback] = None,
) -> Dict[str, Any]:
    conditions = generate_auto_conditions(
        include_keyword_bias=base_config.enable_keyword_bias,
        include_noise_reduction=base_config.enable_noise_reduction,
        include_volume_normalization=base_config.enable_volume_normalization,
        include_llm_postprocess=base_config.enable_llm_postprocess,
        include_rag=base_config.enable_rag,
        include_search=base_config.enable_search,
        mode=mode,
    )
    run_id = make_run_id("auto-experiment")
    output_dir = Path(base_config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit(status_callback, f"Auto experiment {run_id} generated {len(conditions)} condition(s).")
    _prime_asr_cache(audio_path, base_config, conditions, reference_text, rag_inline_text, status_callback)
    started = time.time()
    rows: List[Dict[str, Any]] = []
    max_workers = max(1, int(base_config.auto_experiment_parallelism or 1))
    _emit(status_callback, f"Running auto experiment with {max_workers} parallel condition worker(s).")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, condition in enumerate(conditions):
            config = _config_for_condition(base_config, condition, index)
            futures[executor.submit(_run_condition, audio_path, config, condition, reference_text, rag_inline_text)] = condition
        for future in as_completed(futures):
            condition = futures[future]
            try:
                row = future.result()
                _emit(status_callback, f"Auto condition complete: {condition.condition_id}.")
            except Exception as exc:
                row = _error_row(condition, exc)
                _emit(status_callback, f"Auto condition failed: {condition.condition_id}: {exc}")
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("condition_id", "")))
    summary_path = output_dir / "auto_experiment_summary.csv"
    analysis = analyze_auto_experiment(rows)
    analysis_path = output_dir / "auto_experiment_analysis.json"
    manifest_path = output_dir / "auto_experiment_conditions.json"
    _write_summary_csv(summary_path, rows)
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps([condition.to_dict() for condition in conditions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "mode": mode,
        "condition_count": len(conditions),
        "elapsed_s": time.time() - started,
        "summary_csv": str(summary_path),
        "analysis_json": str(analysis_path),
        "conditions_json": str(manifest_path),
        "analysis": analysis,
        "rows": rows,
    }


def analyze_auto_experiment(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparable = [row for row in rows if _metric(row, "cer_normalized_no_space") is not None]
    best_by_cer = min(comparable, key=lambda row: _metric(row, "cer_normalized_no_space"), default=None)
    best_by_wer = min(
        [row for row in rows if _metric(row, "wer_eojeol") is not None],
        key=lambda row: _metric(row, "wer_eojeol"),
        default=None,
    )
    baseline = next((row for row in comparable if row.get("condition_id") == "baseline"), None)
    baseline_cer = _metric(baseline or {}, "cer_normalized_no_space")
    worse_than_baseline = []
    if baseline_cer is not None:
        for row in comparable:
            value = _metric(row, "cer_normalized_no_space")
            if value is not None and value > baseline_cer:
                worse_than_baseline.append(row)
    return {
        "best_by_cer": best_by_cer,
        "best_by_wer": best_by_wer,
        "baseline": baseline,
        "worse_than_baseline": worse_than_baseline,
        "num_rows": len(rows),
        "num_comparable_rows": len(comparable),
        "num_failed_rows": len([row for row in rows if row.get("error")]),
    }


def _prime_asr_cache(
    audio_path: str,
    base_config: ExperimentConfig,
    conditions: List[ConditionSpec],
    reference_text: Optional[str],
    rag_inline_text: str,
    status_callback: Optional[StatusCallback],
) -> None:
    if not base_config.asr_cache_enabled:
        return
    grouped: Dict[str, ConditionSpec] = {}
    for condition in conditions:
        grouped.setdefault(condition.asr_group_key, condition)
    _emit(status_callback, f"Priming ASR cache for {len(grouped)} pre/ASR group(s).")
    for index, condition in enumerate(grouped.values()):
        config = _config_for_condition(base_config, condition, index)
        config.enable_llm_postprocess = False
        try:
            PipelineRunner(config, status_callback=status_callback).run(
                audio_path=audio_path,
                reference_text=reference_text,
                rag_inline_text=rag_inline_text,
            )
        except Exception as exc:
            _emit(status_callback, f"ASR cache priming failed for {condition.asr_group_key}: {exc}")


def _run_condition(
    audio_path: str,
    config: ExperimentConfig,
    condition: ConditionSpec,
    reference_text: Optional[str],
    rag_inline_text: str,
) -> Dict[str, Any]:
    output = PipelineRunner(config).run(audio_path=audio_path, reference_text=reference_text, rag_inline_text=rag_inline_text)
    metrics = output.metrics.to_dict()
    asr_cache = output.raw.metadata.get("asr_cache") if isinstance(output.raw.metadata, dict) else {}
    return {
        "condition_id": condition.condition_id,
        "label": condition.label,
        "group": condition.group,
        "run_id": output.run_id,
        "output_dir": output.output_dir,
        "keyword_bias_enabled": condition.enable_keyword_bias,
        "noise_reduction_enabled": condition.enable_noise_reduction,
        "volume_normalization_enabled": condition.enable_volume_normalization,
        "llm_postprocess_enabled": condition.enable_llm_postprocess,
        "rag_enabled": condition.enable_rag,
        "search_enabled": condition.enable_search,
        "keyword_bias_weight": config.keyword_bias_weight,
        "noise_reduction_model": config.noise_reduction_model,
        "noise_reduction_strength": config.noise_reduction_strength,
        "volume_normalization_strength": config.volume_normalization_strength,
        "postprocess_strength": config.postprocess_strength,
        "rag_strength": config.rag_strength,
        "search_strength": config.search_strength,
        "asr_cache_key": asr_cache.get("key") if isinstance(asr_cache, dict) else "",
        "asr_cache_hit": asr_cache.get("hit") if isinstance(asr_cache, dict) else "",
        "cer_normalized_no_space": metrics.get("cer_normalized_no_space"),
        "raw_cer_normalized_no_space": metrics.get("raw_cer_normalized_no_space"),
        "delta_cer": metrics.get("delta_cer"),
        "wer_eojeol": metrics.get("wer_eojeol"),
        "raw_wer_eojeol": metrics.get("raw_wer_eojeol"),
        "delta_wer": metrics.get("delta_wer"),
        "semantic_similarity": metrics.get("semantic_similarity"),
        "latency_ms": metrics.get("latency_ms"),
        "risk": output.correction.risk,
        "error": "",
    }


def _config_for_condition(base_config: ExperimentConfig, condition: ConditionSpec, index: int) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    config.asr_cache_enabled = True
    config.preprocess_cache_enabled = True
    config.enable_keyword_bias = condition.enable_keyword_bias
    config.enable_noise_reduction = condition.enable_noise_reduction
    config.enable_volume_normalization = condition.enable_volume_normalization
    config.enable_llm_postprocess = condition.enable_llm_postprocess
    config.enable_rag = condition.enable_rag
    config.enable_search = condition.enable_search
    if config.enable_keyword_bias and config.keyword_bias_weight <= 0:
        config.keyword_bias_weight = 0.5
    if config.enable_noise_reduction:
        if (config.noise_reduction_model or "none").lower() == "none":
            config.noise_reduction_model = "deepfilternet2"
        if config.noise_reduction_strength <= 0:
            config.noise_reduction_strength = 0.5
    if config.enable_volume_normalization and config.volume_normalization_strength <= 0:
        config.volume_normalization_strength = 1.0
    if config.enable_llm_postprocess and config.postprocess_strength <= 0:
        config.postprocess_strength = 0.5
    if config.enable_rag and config.rag_strength <= 0:
        config.rag_strength = 0.5
    if config.enable_search and config.search_strength <= 0:
        config.search_strength = 0.5
    lanes = [lane for lane in (config.pipeline_lanes or []) if isinstance(lane, dict)]
    if lanes:
        lane = lanes[index % len(lanes)]
        if lane.get("asr_base_url"):
            config.asr_base_url = str(lane["asr_base_url"])
        if lane.get("post_base_url"):
            config.post_base_url = str(lane["post_base_url"])
    return config


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "condition_id",
        "label",
        "group",
        "run_id",
        "output_dir",
        "keyword_bias_enabled",
        "noise_reduction_enabled",
        "volume_normalization_enabled",
        "llm_postprocess_enabled",
        "rag_enabled",
        "search_enabled",
        "keyword_bias_weight",
        "noise_reduction_model",
        "noise_reduction_strength",
        "volume_normalization_strength",
        "postprocess_strength",
        "rag_strength",
        "search_strength",
        "asr_cache_key",
        "asr_cache_hit",
        "cer_normalized_no_space",
        "raw_cer_normalized_no_space",
        "delta_cer",
        "wer_eojeol",
        "raw_wer_eojeol",
        "delta_wer",
        "semantic_similarity",
        "latency_ms",
        "risk",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _error_row(condition: ConditionSpec, exc: Exception) -> Dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "label": condition.label,
        "group": condition.group,
        "keyword_bias_enabled": condition.enable_keyword_bias,
        "noise_reduction_enabled": condition.enable_noise_reduction,
        "volume_normalization_enabled": condition.enable_volume_normalization,
        "llm_postprocess_enabled": condition.enable_llm_postprocess,
        "rag_enabled": condition.enable_rag,
        "search_enabled": condition.enable_search,
        "error": str(exc),
    }


def _metric(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _emit(callback: Optional[StatusCallback], message: str) -> None:
    if callback is not None:
        callback(message)
