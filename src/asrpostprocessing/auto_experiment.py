from __future__ import annotations

import copy
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .cache import stable_json_hash
from .config import ExperimentConfig
from .experiment_matrix import ConditionSpec, generate_auto_conditions
from .logging import make_run_id
from .pipeline import PipelineRunner

StatusCallback = Callable[[str], None]


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    condition: ConditionSpec
    asr_model: str
    post_model: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["condition"] = self.condition.to_dict()
        return payload


def run_auto_experiment(
    audio_path: str,
    base_config: ExperimentConfig,
    reference_text: Optional[str] = None,
    rag_inline_text: str = "",
    mode: str = "full_valid",
    status_callback: Optional[StatusCallback] = None,
) -> Dict[str, Any]:
    preview = preview_auto_experiment(base_config, mode=mode)
    conditions = preview["conditions"]
    cases = preview["cases"]
    run_id = make_run_id("auto-experiment")
    output_dir = Path(base_config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit(status_callback, f"Auto experiment {run_id} generated {len(cases)} case(s) from {len(conditions)} condition(s).")
    _prime_asr_cache(audio_path, base_config, cases, reference_text, rag_inline_text, status_callback)
    started = time.time()
    rows: List[Dict[str, Any]] = []
    max_workers = _effective_worker_count(base_config)
    _emit(status_callback, f"Running auto experiment with {max_workers} parallel condition worker(s).")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, case in enumerate(cases):
            config = _config_for_case(base_config, case, index)
            futures[executor.submit(_run_condition, audio_path, config, case, reference_text, rag_inline_text)] = case
        for future in as_completed(futures):
            case = futures[future]
            try:
                row = future.result()
                _emit(status_callback, f"Auto case complete: {case.case_id}.")
            except Exception as exc:
                row = _error_row(case, exc)
                _emit(status_callback, f"Auto case failed: {case.case_id}: {exc}")
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("case_id") or row.get("condition_id", "")))
    _annotate_baseline_deltas(rows)
    summary_path = output_dir / "auto_experiment_summary.csv"
    analysis = analyze_auto_experiment(rows)
    analysis_path = output_dir / "auto_experiment_analysis.json"
    manifest_path = output_dir / "auto_experiment_conditions.json"
    _write_summary_csv(summary_path, rows)
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps([case.to_dict() for case in cases], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "mode": mode,
        "condition_count": len(conditions),
        "case_count": len(cases),
        "asr_cache_group_count": preview["asr_cache_group_count"],
        "elapsed_s": time.time() - started,
        "summary_csv": str(summary_path),
        "analysis_json": str(analysis_path),
        "conditions_json": str(manifest_path),
        "analysis": analysis,
        "rows": rows,
    }


def preview_auto_experiment(base_config: ExperimentConfig, mode: str = "full_valid") -> Dict[str, Any]:
    conditions = generate_auto_conditions(
        include_keyword_bias=base_config.enable_keyword_bias,
        include_noise_reduction=base_config.enable_noise_reduction,
        include_volume_normalization=base_config.enable_volume_normalization,
        include_llm_postprocess=base_config.enable_llm_postprocess,
        include_rag=base_config.enable_rag,
        include_search=base_config.enable_search,
        mode=mode,
        keyword_strengths=base_config.auto_experiment_keyword_weights,
        noise_strengths=base_config.auto_experiment_noise_strengths,
        volume_strengths=base_config.auto_experiment_volume_strengths,
        postprocess_strengths=base_config.auto_experiment_postprocess_strengths,
        rag_strengths=base_config.auto_experiment_rag_strengths,
        rag_top_ks=base_config.auto_experiment_rag_top_ks,
        search_strengths=base_config.auto_experiment_search_strengths,
    )
    cases = _expand_model_cases(conditions, base_config)
    return {
        "mode": mode,
        "conditions": conditions,
        "cases": cases,
        "condition_count": len(conditions),
        "case_count": len(cases),
        "asr_cache_group_count": _asr_cache_group_count(cases),
        "model_axis_enabled": bool(base_config.auto_experiment_include_models),
        "condition_ids": [condition.condition_id for condition in conditions],
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
        "best_latency_quality_tradeoff": _best_latency_quality_tradeoff(comparable, best_by_cer),
        "baseline": baseline,
        "worse_than_baseline": worse_than_baseline,
        "effect_summary": _auto_effect_summary(rows),
        "num_rows": len(rows),
        "num_comparable_rows": len(comparable),
        "num_failed_rows": len([row for row in rows if row.get("error")]),
    }


def _prime_asr_cache(
    audio_path: str,
    base_config: ExperimentConfig,
    cases: List[ExperimentCase],
    reference_text: Optional[str],
    rag_inline_text: str,
    status_callback: Optional[StatusCallback],
) -> None:
    if not base_config.asr_cache_enabled:
        return
    grouped: Dict[str, ExperimentCase] = {}
    for case in cases:
        grouped.setdefault(f"{case.condition.asr_group_key}|asr={case.asr_model}", case)
    workers = min(len(grouped), _effective_asr_worker_count(base_config))
    _emit(status_callback, f"Priming ASR cache for {len(grouped)} pre/ASR model group(s) with {workers} worker(s).")
    if workers <= 1:
        for index, case in enumerate(grouped.values()):
            _prime_one_asr_group(audio_path, base_config, case, index, reference_text, rag_inline_text, status_callback)
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _prime_one_asr_group,
                audio_path,
                base_config,
                case,
                index,
                reference_text,
                rag_inline_text,
                status_callback,
            ): case
            for index, case in enumerate(grouped.values())
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                future.result()
            except Exception as exc:
                _emit(status_callback, f"ASR cache priming failed for {case.case_id}: {exc}")


def _asr_cache_group_count(cases: List[ExperimentCase]) -> int:
    return len({f"{case.condition.asr_group_key}|asr={case.asr_model}" for case in cases})


def _prime_one_asr_group(
    audio_path: str,
    base_config: ExperimentConfig,
    case: ExperimentCase,
    index: int,
    reference_text: Optional[str],
    rag_inline_text: str,
    status_callback: Optional[StatusCallback],
) -> None:
    config = _config_for_case(base_config, case, index)
    config.enable_llm_postprocess = False
    PipelineRunner(config, status_callback=status_callback).run(
        audio_path=audio_path,
        reference_text=reference_text,
        rag_inline_text=rag_inline_text,
    )


def _run_condition(
    audio_path: str,
    config: ExperimentConfig,
    case: ExperimentCase,
    reference_text: Optional[str],
    rag_inline_text: str,
) -> Dict[str, Any]:
    output = PipelineRunner(config).run(audio_path=audio_path, reference_text=reference_text, rag_inline_text=rag_inline_text)
    metrics = output.metrics.to_dict()
    timings = output.timings or {}
    hardware = output.hardware or {}
    vllm_delta = (output.vllm_metrics or {}).get("delta") if isinstance(output.vllm_metrics, dict) else {}
    if not isinstance(vllm_delta, dict):
        vllm_delta = {}
    post_output_tokens = _post_output_tokens(output.correction.metadata)
    postprocess_latency_ms = _metric(timings, "postprocess_latency_ms")
    latency_ms = _metric(timings, "latency_ms") or _metric(metrics, "latency_ms")
    vllm_total_tokens = _metric(vllm_delta, "total_tokens")
    token_count = post_output_tokens if post_output_tokens is not None else vllm_total_tokens
    token_latency_ms = postprocess_latency_ms if post_output_tokens is not None else latency_ms
    asr_cache = output.raw.metadata.get("asr_cache") if isinstance(output.raw.metadata, dict) else {}
    preprocess_cache = output.preprocess.get("metadata", {}).get("cache_hit") if isinstance(output.preprocess, dict) else ""
    return {
        "case_id": case.case_id,
        "condition_id": case.condition.condition_id,
        "label": case.condition.label,
        "group": case.condition.group,
        "asr_model": config.asr_model,
        "post_model": config.post_model,
        "run_id": output.run_id,
        "output_dir": output.output_dir,
        "keyword_bias_enabled": case.condition.enable_keyword_bias,
        "noise_reduction_enabled": case.condition.enable_noise_reduction,
        "volume_normalization_enabled": case.condition.enable_volume_normalization,
        "llm_postprocess_enabled": case.condition.enable_llm_postprocess,
        "rag_enabled": case.condition.enable_rag,
        "search_enabled": case.condition.enable_search,
        "keyword_bias_weight": config.keyword_bias_weight,
        "noise_reduction_model": config.noise_reduction_model,
        "noise_reduction_strength": config.noise_reduction_strength,
        "volume_normalization_strength": config.volume_normalization_strength,
        "postprocess_strength": config.postprocess_strength,
        "rag_strength": config.rag_strength,
        "rag_top_k": config.rag_top_k if case.condition.enable_rag else "",
        "search_strength": config.search_strength,
        "asr_cache_key": asr_cache.get("key") if isinstance(asr_cache, dict) else "",
        "asr_cache_hit": asr_cache.get("hit") if isinstance(asr_cache, dict) else "",
        "preprocess_cache_hit": preprocess_cache,
        "cer_normalized_no_space": metrics.get("cer_normalized_no_space"),
        "raw_cer_normalized_no_space": metrics.get("raw_cer_normalized_no_space"),
        "delta_cer": metrics.get("delta_cer"),
        "wer_eojeol": metrics.get("wer_eojeol"),
        "raw_wer_eojeol": metrics.get("raw_wer_eojeol"),
        "delta_wer": metrics.get("delta_wer"),
        "semantic_similarity": metrics.get("semantic_similarity"),
        "latency_ms": metrics.get("latency_ms"),
        "server_readiness_ms": timings.get("server_readiness_ms"),
        "preprocess_latency_ms": timings.get("preprocess_latency_ms"),
        "asr_latency_ms": timings.get("asr_latency_ms"),
        "postprocess_latency_ms": timings.get("postprocess_latency_ms"),
        "audio_duration_s": timings.get("audio_duration_s"),
        "audio_seconds_per_second": timings.get("audio_seconds_per_second"),
        "tokens_per_second": (
            token_count / (token_latency_ms / 1000.0)
            if token_count is not None and token_latency_ms and token_latency_ms > 0.0
            else ""
        ),
        "post_output_tokens": post_output_tokens if post_output_tokens is not None else "",
        "vllm_preemption_count": _metric_or_blank(vllm_delta, "preemption_count"),
        "vllm_prompt_tokens": _metric_or_blank(vllm_delta, "prompt_tokens"),
        "vllm_generation_tokens": _metric_or_blank(vllm_delta, "generation_tokens"),
        "vllm_total_tokens": _metric_or_blank(vllm_delta, "total_tokens"),
        "vllm_request_success_count": _metric_or_blank(vllm_delta, "request_success_count"),
        "vllm_metrics_available": bool((output.vllm_metrics or {}).get("available"))
        if isinstance(output.vllm_metrics, dict)
        else False,
        "peak_vram_mb": hardware.get("observed_peak_vram_mb"),
        "peak_gpu_utilization_percent": hardware.get("observed_peak_gpu_utilization_percent"),
        "risk": output.correction.risk,
        "error": "",
    }


def _config_for_case(base_config: ExperimentConfig, case: ExperimentCase, index: int) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    condition = case.condition
    config.asr_model = case.asr_model
    config.post_model = case.post_model
    config.asr_cache_enabled = True
    config.preprocess_cache_enabled = True
    config.enable_keyword_bias = condition.enable_keyword_bias
    config.enable_noise_reduction = condition.enable_noise_reduction
    config.enable_volume_normalization = condition.enable_volume_normalization
    config.enable_llm_postprocess = condition.enable_llm_postprocess
    config.enable_rag = condition.enable_rag
    config.enable_search = condition.enable_search
    if config.enable_keyword_bias and condition.keyword_bias_weight is not None:
        config.keyword_bias_weight = condition.keyword_bias_weight
    elif config.enable_keyword_bias and config.keyword_bias_weight <= 0:
        config.keyword_bias_weight = 0.5
    elif not config.enable_keyword_bias:
        config.keyword_bias_weight = 0.0
    if config.enable_noise_reduction:
        if (config.noise_reduction_model or "none").lower() == "none":
            config.noise_reduction_model = "deepfilternet2"
        if condition.noise_reduction_strength is not None:
            config.noise_reduction_strength = condition.noise_reduction_strength
        elif config.noise_reduction_strength <= 0:
            config.noise_reduction_strength = 0.5
    else:
        config.noise_reduction_strength = 0.0
    if config.enable_volume_normalization and condition.volume_normalization_strength is not None:
        config.volume_normalization_strength = condition.volume_normalization_strength
    elif config.enable_volume_normalization and config.volume_normalization_strength <= 0:
        config.volume_normalization_strength = 1.0
    elif not config.enable_volume_normalization:
        config.volume_normalization_strength = 0.0
    if config.enable_llm_postprocess and condition.postprocess_strength is not None:
        config.postprocess_strength = condition.postprocess_strength
    elif config.enable_llm_postprocess and config.postprocess_strength <= 0:
        config.postprocess_strength = 0.5
    elif not config.enable_llm_postprocess:
        config.postprocess_strength = 0.0
    if config.enable_rag and condition.rag_strength is not None:
        config.rag_strength = condition.rag_strength
    elif config.enable_rag and config.rag_strength <= 0:
        config.rag_strength = 0.5
    elif not config.enable_rag:
        config.rag_strength = 0.0
    if config.enable_rag and condition.rag_top_k is not None:
        config.rag_top_k = condition.rag_top_k
    if config.enable_search and condition.search_strength is not None:
        config.search_strength = condition.search_strength
    elif config.enable_search and config.search_strength <= 0:
        config.search_strength = 0.5
    elif not config.enable_search:
        config.search_strength = 0.0
    lanes = _matching_lanes(config.pipeline_lanes or [], config.asr_model, config.post_model)
    if lanes:
        lane = lanes[index % len(lanes)]
        if lane.get("asr_base_url"):
            config.asr_base_url = str(lane["asr_base_url"])
        if lane.get("post_base_url"):
            config.post_base_url = str(lane["post_base_url"])
    return config


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "condition_id",
        "label",
        "group",
        "asr_model",
        "post_model",
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
        "rag_top_k",
        "search_strength",
        "asr_cache_key",
        "asr_cache_hit",
        "preprocess_cache_hit",
        "cer_normalized_no_space",
        "raw_cer_normalized_no_space",
        "delta_cer",
        "delta_cer_vs_baseline",
        "wer_eojeol",
        "raw_wer_eojeol",
        "delta_wer",
        "delta_wer_vs_baseline",
        "semantic_similarity",
        "latency_ms",
        "server_readiness_ms",
        "preprocess_latency_ms",
        "asr_latency_ms",
        "postprocess_latency_ms",
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
        "risk",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _annotate_baseline_deltas(rows: List[Dict[str, Any]]) -> None:
    baselines: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("condition_id") == "baseline" and not row.get("error"):
            baselines.setdefault(str(row.get("asr_model") or ""), row)
    fallback = next((row for row in rows if row.get("condition_id") == "baseline" and not row.get("error")), None)
    for row in rows:
        baseline = baselines.get(str(row.get("asr_model") or "")) or fallback or {}
        row["delta_cer_vs_baseline"] = _baseline_delta(baseline, row, "cer_normalized_no_space")
        row["delta_wer_vs_baseline"] = _baseline_delta(baseline, row, "wer_eojeol")


def _baseline_delta(baseline: Dict[str, Any], row: Dict[str, Any], key: str) -> float | str:
    baseline_value = _metric(baseline, key)
    value = _metric(row, key)
    if baseline_value is None or value is None:
        return ""
    return baseline_value - value


def _best_latency_quality_tradeoff(
    rows: List[Dict[str, Any]], best_row: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    best_cer = _metric(best_row or {}, "cer_normalized_no_space")
    if best_cer is None:
        return None
    tolerance = max(0.005, abs(best_cer) * 0.05)
    near_best = [
        row
        for row in rows
        if _metric(row, "latency_ms") is not None
        and _metric(row, "cer_normalized_no_space") is not None
        and float(_metric(row, "cer_normalized_no_space") or 999.0) <= best_cer + tolerance
    ]
    return min(
        near_best,
        key=lambda row: (
            _metric(row, "latency_ms") or 999999.0,
            _metric(row, "cer_normalized_no_space") or 999.0,
        ),
        default=best_row,
    )


def _auto_effect_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = {
        "keyword_bias_only": "keyword",
        "noise_reduction_only": "noise",
        "volume_normalization_only": "volume",
        "llm_only": "llm",
        "llm_rag": "llm__rag",
        "llm_search": "llm__search",
        "llm_rag_search": "llm__rag__search",
    }
    by_condition: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        by_condition.setdefault(str(row.get("condition_id") or ""), row)
    return {
        name: _effect_payload(by_condition.get(condition_id))
        for name, condition_id in keys.items()
        if by_condition.get(condition_id) is not None
    }


def _effect_payload(row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "case_id": row.get("case_id"),
        "cer_normalized_no_space": row.get("cer_normalized_no_space"),
        "wer_eojeol": row.get("wer_eojeol"),
        "delta_cer_vs_baseline": row.get("delta_cer_vs_baseline"),
        "delta_wer_vs_baseline": row.get("delta_wer_vs_baseline"),
        "latency_ms": row.get("latency_ms"),
        "risk": row.get("risk"),
    }


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


def _metric_or_blank(values: Dict[str, Any], key: str) -> float | str:
    value = _metric(values, key)
    return value if value is not None else ""


def _error_row(case: ExperimentCase, exc: Exception) -> Dict[str, Any]:
    return {
        "case_id": case.case_id,
        "condition_id": case.condition.condition_id,
        "label": case.condition.label,
        "group": case.condition.group,
        "asr_model": case.asr_model,
        "post_model": case.post_model,
        "keyword_bias_enabled": case.condition.enable_keyword_bias,
        "noise_reduction_enabled": case.condition.enable_noise_reduction,
        "volume_normalization_enabled": case.condition.enable_volume_normalization,
        "llm_postprocess_enabled": case.condition.enable_llm_postprocess,
        "rag_enabled": case.condition.enable_rag,
        "search_enabled": case.condition.enable_search,
        "rag_top_k": case.condition.rag_top_k or "",
        "error": str(exc),
    }


def _expand_model_cases(conditions: List[ConditionSpec], config: ExperimentConfig) -> List[ExperimentCase]:
    asr_models = _model_values(config.auto_experiment_asr_models, config.asr_model)
    post_models = _model_values(config.auto_experiment_post_models, config.post_model)
    if not config.auto_experiment_include_models:
        asr_models = [config.asr_model]
        post_models = [config.post_model]
    cases: List[ExperimentCase] = []
    for condition in conditions:
        for asr_model in asr_models:
            active_post_models = post_models if condition.enable_llm_postprocess else [config.post_model]
            for post_model in active_post_models:
                suffix = stable_json_hash(
                    {
                        "condition": condition.condition_id,
                        "asr_model": asr_model,
                        "post_model": post_model if condition.enable_llm_postprocess else "",
                    }
                )[:8]
                case_id = condition.condition_id
                if config.auto_experiment_include_models:
                    case_id = f"{condition.condition_id}__model_{suffix}"
                cases.append(
                    ExperimentCase(
                        case_id=case_id,
                        condition=condition,
                        asr_model=asr_model,
                        post_model=post_model,
                    )
                )
    return cases


def _model_values(values: List[str], fallback: str) -> List[str]:
    models = [str(value).strip() for value in values if str(value).strip()]
    if not models:
        models = [fallback]
    deduped: List[str] = []
    for model in models:
        if model not in deduped:
            deduped.append(model)
    return deduped


def _effective_worker_count(config: ExperimentConfig) -> int:
    requested = max(1, int(config.auto_experiment_parallelism or 1))
    if not config.auto_experiment_saturate_lanes:
        return requested
    lane_count = max(1, len([lane for lane in (config.pipeline_lanes or []) if isinstance(lane, dict)]))
    chunk_workers = max(1, int(config.postprocess_parallelism or 1))
    return max(requested, lane_count * min(4, chunk_workers))


def _effective_asr_worker_count(config: ExperimentConfig) -> int:
    lane_count = len([lane for lane in (config.pipeline_lanes or []) if isinstance(lane, dict) and lane.get("asr_base_url")])
    if lane_count <= 0:
        lane_count = max(1, len(config.asr_base_urls or []))
    if not config.auto_experiment_saturate_lanes:
        return max(1, min(int(config.auto_experiment_parallelism or 1), lane_count))
    return max(1, lane_count)


def _matching_lanes(lanes: List[Dict[str, Any]], asr_model: str, post_model: str) -> List[Dict[str, Any]]:
    dict_lanes = [lane for lane in lanes if isinstance(lane, dict)]
    if not dict_lanes:
        return []
    matching = [
        lane
        for lane in dict_lanes
        if (not lane.get("asr_model") or str(lane.get("asr_model")) == asr_model)
        and (not lane.get("post_model") or str(lane.get("post_model")) == post_model)
    ]
    return matching or dict_lanes


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
