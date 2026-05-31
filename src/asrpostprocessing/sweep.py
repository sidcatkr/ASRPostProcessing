from __future__ import annotations

import copy
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import ExperimentConfig
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
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_id",
            "condition",
            "audio",
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
            "output_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        summary_rows: List[Dict[str, object]] = []
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
                reference = row.get("reference_text") or read_reference(row.get("reference"))
                output = PipelineRunner(config).run(
                    audio_path=row["audio"],
                    reference_text=reference,
                    rag_inline_text=row.get("rag_inline_text", ""),
                )
                metrics = output.metrics.to_dict()
                summary_row = {
                    "run_id": output.run_id,
                    "condition": condition,
                    "audio": row["audio"],
                    "keyword_bias_weight": keyword_weight,
                    "noise_reduction_strength": noise_strength,
                    "volume_normalization_strength": volume_strength,
                    "rag_strength": rag_strength,
                    "postprocess_strength": post_strength,
                    "model_residency": config.model_residency,
                    "cer_normalized_no_space": metrics.get("cer_normalized_no_space"),
                    "wer_eojeol": metrics.get("wer_eojeol"),
                    "delta_cer": metrics.get("delta_cer"),
                    "delta_wer": metrics.get("delta_wer"),
                    "semantic_similarity": metrics.get("semantic_similarity"),
                    "risk": output.correction.risk,
                    "latency_ms": metrics.get("latency_ms"),
                    "output_dir": output.output_dir,
                }
                summary_rows.append(summary_row)
                writer.writerow(summary_row)
                handle.flush()
    analysis_path = summary_dir / "sweep_analysis.json"
    analysis_path.write_text(json.dumps(analyze_sweep(summary_rows), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


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
    best = min(
        comparable,
        key=lambda row: (
            _metric_or(row, "cer_normalized_no_space", 999.0),
            _semantic_drift(row),
            _metric_or(row, "latency_ms", 999999.0),
        ),
        default=None,
    )
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
    over_bias_cases = []
    over_rag_cases = []
    over_postprocess_cases = []
    for row in comparable:
        audio = str(row["audio"])
        raw_cer = raw_by_audio.get(audio)
        cer_value = _metric(row, "cer_normalized_no_space")
        if raw_cer is not None and cer_value is not None and cer_value > raw_cer:
            over_bias_cases.append({**row, "over_bias_reason": "worse_than_raw_asr"})
        zero_cer = zero_weight_by_condition.get(_zero_key(row))
        if zero_cer is not None and cer_value is not None and cer_value > zero_cer:
            over_bias_cases.append({**row, "over_bias_reason": "worse_than_zero_weight"})
        if _metric_or(row, "rag_strength", 0.0) > 0.0 and zero_cer is not None and cer_value is not None and cer_value > zero_cer:
            over_rag_cases.append({**row, "over_rag_reason": "worse_than_rag_zero"})
        if _metric_or(row, "postprocess_strength", 0.0) > 0.25 and zero_cer is not None and cer_value is not None and cer_value > zero_cer:
            over_postprocess_cases.append({**row, "over_postprocess_reason": "worse_than_zero_weight"})
    return {
        "best_by_cer": best,
        "over_bias_cases": over_bias_cases,
        "over_rag_cases": over_rag_cases,
        "over_postprocess_cases": over_postprocess_cases,
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
