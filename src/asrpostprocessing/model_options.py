from __future__ import annotations

from typing import Iterable, List

NOISE_REDUCTION_MODEL_CHOICES = [
    ("None", "none"),
    ("FFmpeg afftdn", "afftdn"),
    ("RNNoise", "rnnoise"),
    ("DeepFilterNet2", "deepfilternet2"),
    ("DeepFilterNet2-PF", "deepfilternet2_pf"),
    ("DeepFilterNet3", "deepfilternet3"),
    ("BS-RoFormer", "bs-roformer"),
]

AUTO_EXPERIMENT_NOISE_MODELS = [
    value for _label, value in NOISE_REDUCTION_MODEL_CHOICES if value != "none"
]

AUTO_EXPERIMENT_RAG_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
]

_NOISE_REDUCTION_MODEL_ALIASES = {
    "": "none",
    "none": "none",
    "off": "none",
    "disabled": "none",
    "afftdn": "afftdn",
    "ffmpeg_afftdn": "afftdn",
    "basic": "afftdn",
    "built_in": "afftdn",
    "denoise": "afftdn",
    "rnnoise": "rnnoise",
    "deepfilternet2": "deepfilternet2",
    "deep_filter_net2": "deepfilternet2",
    "deepfilternet2_pf": "deepfilternet2_pf",
    "deep_filter_net2_pf": "deepfilternet2_pf",
    "deepfilternet3": "deepfilternet3",
    "deep_filter_net3": "deepfilternet3",
    "bs_roformer": "bs-roformer",
    "bsroformer": "bs-roformer",
}


def canonical_noise_reduction_model(value: str) -> str:
    normalized = str(value or "none").strip().lower().replace("-", "_")
    canonical = _NOISE_REDUCTION_MODEL_ALIASES.get(normalized)
    if canonical is not None:
        return canonical
    return normalized.replace("_", "-") if normalized.startswith("bs_") else normalized


def auto_experiment_noise_models(configured_values: Iterable[str], selected_model: str = "") -> List[str]:
    values = _dedupe(
        canonical_noise_reduction_model(value)
        for value in configured_values
        if str(value or "").strip()
    )
    if not values:
        values = list(AUTO_EXPERIMENT_NOISE_MODELS)
    selected = canonical_noise_reduction_model(selected_model)
    if selected and selected != "none" and selected not in values:
        values.append(selected)
    return values


def auto_experiment_rag_embedding_models(configured_values: Iterable[str], selected_model: str = "") -> List[str]:
    values = _dedupe(str(value).strip() for value in configured_values if str(value or "").strip())
    if not values:
        values = list(AUTO_EXPERIMENT_RAG_EMBEDDING_MODELS)
    selected = str(selected_model or "").strip()
    if selected and selected not in values:
        values.append(selected)
    return values


def _dedupe(values: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in deduped:
            deduped.append(text)
    return deduped
