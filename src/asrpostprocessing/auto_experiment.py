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
    conditions = generate_auto_conditions(
        include_keyword_bias=base_config.enable_keyword_bias,
        include_noise_reduction=base_config.enable_noise_reduction,
        include_volume_normalization=base_config.enable_volume_normalization,
        include_llm_postprocess=base_config.enable_llm_postprocess,
        include_rag=base_config.enable_rag,
        include_search=base_config.enable_search,
        mode=mode,
    )
    cases = _expand_model_cases(conditions, base_config)
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
    asr_cache = output.raw.metadata.get("asr_cache") if isinstance(output.raw.metadata, dict) else {}
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
