from __future__ import annotations

import copy
import csv
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .cache import stable_json_hash
from .config import ExperimentConfig
from .experiment_matrix import ConditionSpec, generate_auto_conditions
from .logging import make_run_id
from .model_server import ensure_model_servers, stop_model_servers
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
    started = time.time()
    if base_config.model_residency == "stage_replicas" and base_config.auto_start_model_servers:
        rows = _run_stage_replicas_auto_experiment(
            audio_path,
            base_config,
            cases,
            reference_text,
            rag_inline_text,
            status_callback,
        )
    elif base_config.asr_cache_enabled:
        rows = _run_conditions_after_asr_cache_ready(
            audio_path,
            base_config,
            cases,
            reference_text,
            rag_inline_text,
            status_callback,
        )
    else:
        rows = _run_conditions_parallel(
            audio_path,
            base_config,
            list(enumerate(cases)),
            reference_text,
            rag_inline_text,
            status_callback,
        )
    rows.sort(key=lambda row: str(row.get("case_id") or row.get("condition_id", "")))
    _annotate_baseline_deltas(rows)
    summary_path = output_dir / "auto_experiment_summary.csv"
    audit = build_experiment_audit(
        rows,
        expected_condition_ids=[condition.condition_id for condition in conditions],
        expected_case_ids=[case.case_id for case in cases],
        expected_asr_cache_group_count=preview["asr_cache_group_count"],
        reference_text=reference_text,
        mode=mode,
    )
    analysis = analyze_auto_experiment(rows, audit=audit)
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
        "audit": audit,
        "elapsed_s": time.time() - started,
        "summary_csv": str(summary_path),
        "analysis_json": str(analysis_path),
        "conditions_json": str(manifest_path),
        "analysis": analysis,
        "rows": rows,
    }


def _run_conditions_parallel(
    audio_path: str,
    base_config: ExperimentConfig,
    indexed_cases: List[Tuple[int, ExperimentCase]],
    reference_text: Optional[str],
    rag_inline_text: str,
    status_callback: Optional[StatusCallback],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    max_workers = _effective_worker_count(base_config)
    _emit(status_callback, f"Running auto experiment with {max_workers} parallel condition worker(s).")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for index, case in indexed_cases:
            config = _config_for_case(base_config, case, index)
            futures[executor.submit(_run_condition, audio_path, config, case, reference_text, rag_inline_text)] = (
                case,
                config,
            )
        for future in as_completed(futures):
            case, config = futures[future]
            rows.append(_condition_row_from_future(future, case, status_callback, config))
    return rows


def _run_conditions_after_asr_cache_ready(
    audio_path: str,
    base_config: ExperimentConfig,
    cases: List[ExperimentCase],
    reference_text: Optional[str],
    rag_inline_text: str,
    status_callback: Optional[StatusCallback],
) -> List[Dict[str, Any]]:
    grouped = _group_indexed_cases_by_asr_cache_key(cases)
    if not grouped:
        return []
    asr_workers = min(len(grouped), _effective_asr_worker_count(base_config))
    condition_workers = _effective_worker_count(base_config)
    _emit(
        status_callback,
        f"Priming ASR cache for {len(grouped)} pre/ASR model group(s) with {asr_workers} ASR worker(s); "
        f"running ready cases with {condition_workers} condition worker(s).",
    )
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=asr_workers) as asr_executor, ThreadPoolExecutor(max_workers=condition_workers) as condition_executor:
        pending_asr = {}
        for group_index, indexed_cases in enumerate(grouped.values()):
            case = indexed_cases[0][1]
            pending_asr[
                asr_executor.submit(
                    _prime_one_asr_group,
                    audio_path,
                    base_config,
                    case,
                    group_index,
                    reference_text,
                    rag_inline_text,
                    status_callback,
                )
            ] = indexed_cases
        pending_conditions = {}
        while pending_asr or pending_conditions:
            done, _ = wait(list(pending_asr) + list(pending_conditions), return_when=FIRST_COMPLETED)
            for future in done:
                if future in pending_asr:
                    indexed_cases = pending_asr.pop(future)
                    case = indexed_cases[0][1]
                    try:
                        future.result()
                        _emit(status_callback, f"ASR cache ready for group {case.condition.asr_group_key}.")
                    except Exception as exc:
                        _emit(status_callback, f"ASR cache priming failed for {case.case_id}: {exc}")
                    for index, condition_case in indexed_cases:
                        config = _config_for_case(base_config, condition_case, index)
                        pending_conditions[
                            condition_executor.submit(
                                _run_condition,
                                audio_path,
                                config,
                                condition_case,
                                reference_text,
                                rag_inline_text,
                            )
                        ] = (condition_case, config)
                    continue
                case, config = pending_conditions.pop(future)
                rows.append(_condition_row_from_future(future, case, status_callback, config))
    return rows


def _run_stage_replicas_auto_experiment(
    audio_path: str,
    base_config: ExperimentConfig,
    cases: List[ExperimentCase],
    reference_text: Optional[str],
    rag_inline_text: str,
    status_callback: Optional[StatusCallback],
) -> List[Dict[str, Any]]:
    managed_config = copy.deepcopy(base_config)
    managed_config.asr_cache_enabled = True
    managed_config.preprocess_cache_enabled = True
    worker_config = copy.deepcopy(managed_config)
    worker_config.auto_start_model_servers = False

    _emit(status_callback, "Starting ASR replicas on scalable stage GPUs.")
    asr_active_names = _start_stage_replicas_scalable(
        managed_config,
        worker_config,
        "asr",
        status_callback,
    )
    try:
        _prime_asr_groups(audio_path, worker_config, cases, reference_text, rag_inline_text, status_callback)
    finally:
        _emit(status_callback, "Stopping ASR replicas before post-processing stage.")
        stop_model_servers(managed_config, status_callback=status_callback, names=asr_active_names)

    post_active_names: List[str] = []
    if _has_postprocess_cases(cases):
        _emit(status_callback, "Starting post-processing replicas on scalable stage GPUs.")
        post_active_names = _start_stage_replicas_scalable(
            managed_config,
            worker_config,
            "post",
            status_callback,
        )
    try:
        return _run_conditions_parallel(
            audio_path,
            worker_config,
            list(enumerate(cases)),
            reference_text,
            rag_inline_text,
            status_callback,
        )
    finally:
        if _has_postprocess_cases(cases):
            _emit(status_callback, "Stopping post-processing replicas.")
            stop_model_servers(managed_config, status_callback=status_callback, names=post_active_names)


def _start_stage_replicas_scalable(
    managed_config: ExperimentConfig,
    worker_config: ExperimentConfig,
    stage: str,
    status_callback: Optional[StatusCallback],
) -> List[str]:
    pairs = _stage_server_pairs_for_config(managed_config)
    if not pairs:
        raise RuntimeError(f"No configured {stage} stage replicas are available.")
    active_pairs: List[Tuple[str, str]] = []
    active_names: List[str] = []
    for index, (base_url, gpu) in enumerate(pairs):
        spec_name = f"{stage}_stage_{index}"
        try:
            ensure_model_servers(managed_config, status_callback=status_callback, names=[spec_name])
        except Exception as exc:
            _emit(
                status_callback,
                f"Skipping {spec_name} on GPU {gpu} at {base_url}; startup failed and scalable mode will use remaining replicas: {exc}",
            )
            try:
                stop_model_servers(managed_config, status_callback=status_callback, names=[spec_name])
            except Exception as cleanup_exc:
                _emit(status_callback, f"Cleanup for failed {spec_name} reported: {cleanup_exc}")
            continue
        active_pairs.append((base_url, gpu))
        active_names.append(spec_name)
    if not active_pairs:
        raise RuntimeError(f"No {stage} stage replicas became ready. Check GPU VRAM usage and model server logs.")
    _restrict_stage_worker_pool(worker_config, active_pairs)
    _emit(
        status_callback,
        f"Scalable {stage} stage active replicas: "
        + ", ".join(f"{base_url} on GPU {gpu}" for base_url, gpu in active_pairs),
    )
    return active_names


def _stage_server_pairs_for_config(config: ExperimentConfig) -> List[Tuple[str, str]]:
    base_urls = [str(url).strip() for url in (getattr(config, "stage_server_base_urls", []) or []) if str(url).strip()]
    gpus = [str(gpu).strip() for gpu in (getattr(config, "stage_server_gpus", []) or []) if str(gpu).strip()]
    count = min(len(base_urls), len(gpus))
    return [(base_urls[index], gpus[index]) for index in range(count)]


def _restrict_stage_worker_pool(config: ExperimentConfig, active_pairs: List[Tuple[str, str]]) -> None:
    config.stage_server_base_urls = [base_url for base_url, _ in active_pairs]
    config.stage_server_gpus = [gpu for _, gpu in active_pairs]
    config.preprocess_gpus = [gpu for _, gpu in active_pairs]
    if active_pairs:
        config.asr_base_url = active_pairs[0][0]
        config.post_base_url = active_pairs[0][0]
        config.preprocess_gpu = active_pairs[0][1]


def _prime_asr_groups(
    audio_path: str,
    base_config: ExperimentConfig,
    cases: List[ExperimentCase],
    reference_text: Optional[str],
    rag_inline_text: str,
    status_callback: Optional[StatusCallback],
) -> None:
    grouped = _group_indexed_cases_by_asr_cache_key(cases)
    if not grouped:
        return
    workers = min(len(grouped), _effective_asr_worker_count(base_config))
    _emit(status_callback, f"Priming ASR cache for {len(grouped)} group(s) with {workers} all-GPU ASR worker(s).")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for group_index, indexed_cases in enumerate(grouped.values()):
            case = indexed_cases[0][1]
            futures[
                executor.submit(
                    _prime_one_asr_group,
                    audio_path,
                    base_config,
                    case,
                    group_index,
                    reference_text,
                    rag_inline_text,
                    status_callback,
                )
            ] = case
        for future in as_completed(futures):
            case = futures[future]
            try:
                future.result()
                _emit(status_callback, f"ASR cache ready for group {case.condition.asr_group_key}.")
            except Exception as exc:
                _emit(status_callback, f"ASR cache priming failed for {case.case_id}: {exc}")


def _has_postprocess_cases(cases: List[ExperimentCase]) -> bool:
    return any(case.condition.enable_llm_postprocess for case in cases)


def _condition_row_from_future(
    future,
    case: ExperimentCase,
    status_callback: Optional[StatusCallback],
    config: Optional[ExperimentConfig] = None,
) -> Dict[str, Any]:
    try:
        row = future.result()
        _emit(status_callback, f"Auto case complete: {case.case_id}.")
    except Exception as exc:
        row = _error_row(case, exc)
        if config is not None:
            row.update(_routing_payload(config))
        _emit(status_callback, f"Auto case failed: {case.case_id}: {exc}")
    return row


def _group_indexed_cases_by_asr_cache_key(cases: List[ExperimentCase]) -> Dict[str, List[Tuple[int, ExperimentCase]]]:
    grouped: Dict[str, List[Tuple[int, ExperimentCase]]] = {}
    for index, case in enumerate(cases):
        grouped.setdefault(_asr_cache_group_key(case), []).append((index, case))
    return grouped


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
        noise_models=base_config.auto_experiment_noise_models
        if base_config.auto_experiment_include_models
        else None,
        noise_strengths=base_config.auto_experiment_noise_strengths,
        volume_strengths=base_config.auto_experiment_volume_strengths,
        postprocess_strengths=base_config.auto_experiment_postprocess_strengths,
        rag_embedding_models=base_config.auto_experiment_rag_embedding_models
        if base_config.auto_experiment_include_models
        else None,
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


def analyze_auto_experiment(rows: List[Dict[str, Any]], audit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    comparable = [row for row in rows if _metric(row, "cer_normalized_no_space") is not None]
    best_by_cer = min(comparable, key=lambda row: _metric(row, "cer_normalized_no_space"), default=None)
    best_by_wer = min(
        [row for row in rows if _metric(row, "wer_eojeol") is not None],
        key=lambda row: _metric(row, "wer_eojeol"),
        default=None,
    )
    best_latency_quality_tradeoff = _best_latency_quality_tradeoff(comparable, best_by_cer)
    baseline = next((row for row in comparable if row.get("condition_id") == "baseline"), None)
    baseline_cer = _metric(baseline or {}, "cer_normalized_no_space")
    worse_than_baseline = []
    if baseline_cer is not None:
        for row in comparable:
            value = _metric(row, "cer_normalized_no_space")
            if value is not None and value > baseline_cer:
                worse_than_baseline.append(row)
    result = {
        "best_by_cer": best_by_cer,
        "best_by_wer": best_by_wer,
        "best_latency_quality_tradeoff": best_latency_quality_tradeoff,
        "best_methods": _best_method_summary(best_by_cer, best_by_wer, best_latency_quality_tradeoff),
        "baseline": baseline,
        "worse_than_baseline": worse_than_baseline,
        "effect_summary": _auto_effect_summary(rows),
        "num_rows": len(rows),
        "num_comparable_rows": len(comparable),
        "num_failed_rows": len([row for row in rows if row.get("error")]),
    }
    if audit is not None:
        result["audit"] = audit
    return result


def build_experiment_audit(
    rows: List[Dict[str, Any]],
    expected_condition_ids: List[str],
    expected_case_ids: List[str],
    expected_asr_cache_group_count: int,
    reference_text: Optional[str],
    mode: str,
) -> Dict[str, Any]:
    failed_rows = [row for row in rows if row.get("error")]
    cer_wer_rows = [
        row
        for row in rows
        if _metric(row, "cer_normalized_no_space") is not None and _metric(row, "wer_eojeol") is not None
    ]
    baseline_rows = [row for row in cer_wer_rows if row.get("condition_id") == "baseline"]
    expected_conditions = {str(condition_id) for condition_id in expected_condition_ids if condition_id}
    expected_cases = {str(case_id) for case_id in expected_case_ids if case_id}
    condition_ids = {str(row.get("condition_id") or "") for row in rows if row.get("condition_id")}
    case_ids = {str(row.get("case_id") or "") for row in rows if row.get("case_id")}
    missing_conditions = sorted(expected_conditions - condition_ids)
    extra_conditions = sorted(condition_ids - expected_conditions)
    missing_cases = sorted(expected_cases - case_ids)
    extra_cases = sorted(case_ids - expected_cases)
    planned_asr_cache_groups = {
        str(row.get("planned_asr_cache_group_key") or row.get("asr_cache_key") or "")
        for row in rows
        if row.get("planned_asr_cache_group_key") or row.get("asr_cache_key")
    }
    actual_asr_cache_groups = {str(row.get("asr_cache_key") or "") for row in rows if row.get("asr_cache_key")}
    observed_asr_urls = sorted({str(row.get("asr_base_url") or "") for row in rows if row.get("asr_base_url")})
    observed_post_urls = sorted({str(row.get("post_base_url") or "") for row in rows if row.get("post_base_url")})
    observed_preprocess_gpus = sorted({str(row.get("preprocess_gpu") or "") for row in rows if row.get("preprocess_gpu")})
    baseline = min(baseline_rows, key=lambda row: _metric(row, "cer_normalized_no_space") or 999999.0, default=None)
    best_cer_row = min(cer_wer_rows, key=lambda row: _metric(row, "cer_normalized_no_space") or 999999.0, default=None)
    best_wer_row = min(cer_wer_rows, key=lambda row: _metric(row, "wer_eojeol") or 999999.0, default=None)
    baseline_cer = _metric(baseline or {}, "cer_normalized_no_space")
    baseline_wer = _metric(baseline or {}, "wer_eojeol")
    best_cer = _metric(best_cer_row or {}, "cer_normalized_no_space")
    best_wer = _metric(best_wer_row or {}, "wer_eojeol")
    best_cer_improvement = _sub_or_none(baseline_cer, best_cer)
    best_wer_improvement = _sub_or_none(baseline_wer, best_wer)
    expected_case_count = len(expected_cases)
    expected_condition_count = len(expected_conditions)
    gates = {
        "reference_provided": bool(reference_text),
        "all_expected_cases_finished": len(rows) == expected_case_count and not missing_cases and not extra_cases,
        "condition_coverage_complete": not missing_conditions and not extra_conditions,
        "no_failed_cases": not failed_rows,
        "baseline_present": bool(baseline_rows),
        "cer_wer_available_for_all_rows": bool(reference_text) and len(cer_wer_rows) == len(rows),
        "asr_cache_groups_observed": len(planned_asr_cache_groups) >= min(expected_asr_cache_group_count, expected_case_count),
    }
    strict_valid = all(gates.values())
    return {
        "mode": mode,
        "strict_valid": strict_valid,
        "gates": gates,
        "verdict": "valid" if strict_valid else "incomplete",
        "row_count": len(rows),
        "expected_case_count": expected_case_count,
        "missing_case_ids": missing_cases,
        "extra_case_ids": extra_cases,
        "failed_count": len(failed_rows),
        "cer_wer_row_count": len(cer_wer_rows),
        "expected_condition_count": expected_condition_count,
        "observed_condition_count": len(condition_ids),
        "missing_condition_ids": missing_conditions,
        "extra_condition_ids": extra_conditions,
        "expected_asr_cache_group_count": expected_asr_cache_group_count,
        "observed_asr_cache_group_count": len(planned_asr_cache_groups),
        "actual_asr_cache_key_count": len(actual_asr_cache_groups),
        "reference_char_count": len(reference_text or ""),
        "baseline_case_id": baseline.get("case_id") if baseline else "",
        "baseline_cer_normalized_no_space": baseline_cer,
        "baseline_wer_eojeol": baseline_wer,
        "best_cer_case_id": best_cer_row.get("case_id") if best_cer_row else "",
        "best_cer_normalized_no_space": best_cer,
        "best_cer_improvement_vs_baseline": best_cer_improvement,
        "best_wer_case_id": best_wer_row.get("case_id") if best_wer_row else "",
        "best_wer_eojeol": best_wer,
        "best_wer_improvement_vs_baseline": best_wer_improvement,
        "observed_asr_base_urls": observed_asr_urls,
        "observed_post_base_urls": observed_post_urls,
        "observed_preprocess_gpus": observed_preprocess_gpus,
        "peak_gpu_utilization_percent": _max_metric(rows, "peak_gpu_utilization_percent"),
        "peak_vram_mb": _max_metric(rows, "peak_vram_mb"),
        "conclusion": _strict_conclusion(strict_valid, best_cer_improvement, best_wer_improvement),
    }


def _asr_cache_group_count(cases: List[ExperimentCase]) -> int:
    return len({_asr_cache_group_key(case) for case in cases})


def _asr_cache_group_key(case: ExperimentCase) -> str:
    return f"{case.condition.asr_group_key}|asr={case.asr_model}"


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
    rag_context_count = _postprocess_metadata_sum(output.correction.metadata, "rag_context_count")
    search_result_count = _postprocess_metadata_sum(output.correction.metadata, "search_result_count")
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
        **_routing_payload(config),
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
        "rag_embedding_backend": config.rag_embedding_backend if case.condition.enable_rag else "",
        "rag_embedding_model": config.rag_embedding_model if case.condition.enable_rag else "",
        "rag_strength": config.rag_strength,
        "rag_top_k": config.rag_top_k if case.condition.enable_rag else "",
        "search_strength": config.search_strength,
        "rag_context_count": rag_context_count,
        "rag_used_context_count": len(output.correction.used_context_ids),
        "search_result_count": search_result_count,
        "planned_asr_cache_group_key": _asr_cache_group_key(case),
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
        if condition.noise_reduction_model:
            config.noise_reduction_model = condition.noise_reduction_model
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
    if config.enable_rag and condition.rag_embedding_model:
        config.rag_embedding_model = condition.rag_embedding_model
    if config.enable_search and condition.search_strength is not None:
        config.search_strength = condition.search_strength
    elif config.enable_search and config.search_strength <= 0:
        config.search_strength = 0.5
    elif not config.enable_search:
        config.search_strength = 0.0
    if config.model_residency == "stage_replicas":
        endpoints = _stage_server_base_urls(config)
        if endpoints:
            endpoint = endpoints[index % len(endpoints)]
            config.asr_base_url = endpoint
            config.post_base_url = endpoint
        preprocess_gpus = _preprocess_gpus(config)
        if preprocess_gpus:
            config.preprocess_gpu = preprocess_gpus[index % len(preprocess_gpus)]
        return config
    lanes = _matching_lanes(config.pipeline_lanes or [], config.asr_model, config.post_model)
    if lanes:
        lane = lanes[index % len(lanes)]
        if lane.get("asr_base_url"):
            config.asr_base_url = str(lane["asr_base_url"])
        if lane.get("post_base_url"):
            config.post_base_url = str(lane["post_base_url"])
        if lane.get("preprocess_gpu"):
            config.preprocess_gpu = str(lane["preprocess_gpu"])
    return config


def _routing_payload(config: ExperimentConfig) -> Dict[str, Any]:
    return {
        "asr_base_url": config.asr_base_url,
        "post_base_url": config.post_base_url,
        "preprocess_gpu": config.preprocess_gpu,
        "model_residency": config.model_residency,
    }


def _write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "case_id",
        "condition_id",
        "label",
        "group",
        "asr_model",
        "post_model",
        "asr_base_url",
        "post_base_url",
        "preprocess_gpu",
        "model_residency",
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
        "rag_embedding_backend",
        "rag_embedding_model",
        "rag_strength",
        "rag_top_k",
        "search_strength",
        "rag_context_count",
        "rag_used_context_count",
        "search_result_count",
        "planned_asr_cache_group_key",
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


def _best_method_summary(
    best_by_cer: Optional[Dict[str, Any]],
    best_by_wer: Optional[Dict[str, Any]],
    best_latency_quality_tradeoff: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates = [
        ("Best CER", "cer_normalized_no_space", best_by_cer),
        ("Best WER", "wer_eojeol", best_by_wer),
        ("Best speed/quality", "latency_ms", best_latency_quality_tradeoff),
    ]
    methods: List[Dict[str, Any]] = []
    for badge, metric_key, row in candidates:
        if not row:
            continue
        methods.append(
            {
                "badge": badge,
                "metric_key": metric_key,
                "case_id": row.get("case_id") or "",
                "condition_id": row.get("condition_id") or "",
                "label": row.get("label") or "",
                "method": _method_label(row),
                "cer_normalized_no_space": row.get("cer_normalized_no_space"),
                "wer_eojeol": row.get("wer_eojeol"),
                "delta_cer_vs_baseline": row.get("delta_cer_vs_baseline"),
                "delta_wer_vs_baseline": row.get("delta_wer_vs_baseline"),
                "latency_ms": row.get("latency_ms"),
                "rag_context_count": row.get("rag_context_count"),
                "rag_used_context_count": row.get("rag_used_context_count"),
                "search_result_count": row.get("search_result_count"),
            }
        )
    return methods


def _method_label(row: Dict[str, Any]) -> str:
    features = []
    feature_map = [
        ("keyword_bias_enabled", "Keyword"),
        ("noise_reduction_enabled", "Noise"),
        ("volume_normalization_enabled", "Volume"),
        ("llm_postprocess_enabled", "LLM"),
        ("rag_enabled", "RAG"),
        ("search_enabled", "Search"),
    ]
    for key, label in feature_map:
        if _truthy(row.get(key)):
            features.append(label)
    return " + ".join(features) if features else "Baseline"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


def _sub_or_none(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def _max_metric(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [_metric(row, key) for row in rows]
    numeric = [value for value in values if value is not None]
    return max(numeric, default=None)


def _strict_conclusion(
    strict_valid: bool,
    best_cer_improvement: Optional[float],
    best_wer_improvement: Optional[float],
) -> str:
    if not strict_valid:
        return "Not enough evidence for a strict conclusion. Check failed or missing audit gates."
    cer_improved = best_cer_improvement is not None and best_cer_improvement > 0.0
    wer_improved = best_wer_improvement is not None and best_wer_improvement > 0.0
    if cer_improved and wer_improved:
        return "Strictly comparable run completed; at least one condition improves both CER and WER over baseline."
    if cer_improved:
        return "Strictly comparable run completed; best condition improves CER, but WER did not improve over baseline."
    if wer_improved:
        return "Strictly comparable run completed; best condition improves WER, but CER did not improve over baseline."
    return "Strictly comparable run completed; no condition improved CER/WER over baseline."


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


def _postprocess_metadata_sum(metadata: Dict[str, Any], key: str) -> int:
    total = 0
    for chunk in metadata.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        chunk_metadata = chunk.get("metadata")
        if not isinstance(chunk_metadata, dict):
            continue
        value = chunk_metadata.get(key)
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


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
        "planned_asr_cache_group_key": _asr_cache_group_key(case),
        "keyword_bias_enabled": case.condition.enable_keyword_bias,
        "noise_reduction_enabled": case.condition.enable_noise_reduction,
        "volume_normalization_enabled": case.condition.enable_volume_normalization,
        "llm_postprocess_enabled": case.condition.enable_llm_postprocess,
        "rag_enabled": case.condition.enable_rag,
        "search_enabled": case.condition.enable_search,
        "noise_reduction_model": case.condition.noise_reduction_model or "",
        "rag_embedding_model": case.condition.rag_embedding_model or "",
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
    lane_count = _available_lane_count(config)
    chunk_workers = max(1, int(config.postprocess_parallelism or 1))
    return max(requested, lane_count * min(4, chunk_workers))


def _effective_asr_worker_count(config: ExperimentConfig) -> int:
    if config.model_residency == "stage_replicas":
        return max(1, _available_lane_count(config))
    lane_count = len([lane for lane in (config.pipeline_lanes or []) if isinstance(lane, dict) and lane.get("asr_base_url")])
    if lane_count <= 0:
        lane_count = max(1, len(config.asr_base_urls or []))
    if not config.auto_experiment_saturate_lanes:
        return max(1, min(int(config.auto_experiment_parallelism or 1), lane_count))
    return max(1, lane_count)


def _available_lane_count(config: ExperimentConfig) -> int:
    if config.model_residency == "stage_replicas":
        return max(1, len(_stage_server_base_urls(config)), len(_preprocess_gpus(config)))
    return max(1, len([lane for lane in (config.pipeline_lanes or []) if isinstance(lane, dict)]))


def _stage_server_base_urls(config: ExperimentConfig) -> List[str]:
    return [
        str(item).strip()
        for item in (getattr(config, "stage_server_base_urls", []) or [])
        if str(item).strip()
    ]


def _preprocess_gpus(config: ExperimentConfig) -> List[str]:
    return [
        str(item).strip()
        for item in (getattr(config, "preprocess_gpus", []) or [])
        if str(item).strip()
    ]


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
