from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    label: str
    group: str
    enable_keyword_bias: bool = False
    enable_noise_reduction: bool = False
    enable_volume_normalization: bool = False
    enable_llm_postprocess: bool = False
    enable_rag: bool = False
    enable_search: bool = False

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def asr_group_key(self) -> str:
        return "|".join(
            [
                f"k={int(self.enable_keyword_bias)}",
                f"n={int(self.enable_noise_reduction)}",
                f"v={int(self.enable_volume_normalization)}",
            ]
        )


def generate_auto_conditions(
    include_keyword_bias: bool = True,
    include_noise_reduction: bool = True,
    include_volume_normalization: bool = True,
    include_llm_postprocess: bool = True,
    include_rag: bool = True,
    include_search: bool = True,
    mode: str = "full_valid",
) -> List[ConditionSpec]:
    mode = (mode or "full_valid").strip().lower().replace("-", "_")
    if mode in {"core", "core_ablation", "ablation"}:
        return _core_ablation_conditions(
            include_keyword_bias,
            include_noise_reduction,
            include_volume_normalization,
            include_llm_postprocess,
            include_rag,
            include_search,
        )
    return _full_valid_conditions(
        include_keyword_bias,
        include_noise_reduction,
        include_volume_normalization,
        include_llm_postprocess,
        include_rag,
        include_search,
    )


def _full_valid_conditions(
    include_keyword_bias: bool,
    include_noise_reduction: bool,
    include_volume_normalization: bool,
    include_llm_postprocess: bool,
    include_rag: bool,
    include_search: bool,
) -> List[ConditionSpec]:
    pre_modes = _pre_modes(include_keyword_bias, include_noise_reduction, include_volume_normalization)
    post_modes = _post_modes(include_llm_postprocess, include_rag, include_search)
    conditions: List[ConditionSpec] = []
    for pre, post in itertools.product(pre_modes, post_modes):
        condition = _condition_from_modes(pre, post)
        conditions.append(condition)
    return conditions


def _core_ablation_conditions(
    include_keyword_bias: bool,
    include_noise_reduction: bool,
    include_volume_normalization: bool,
    include_llm_postprocess: bool,
    include_rag: bool,
    include_search: bool,
) -> List[ConditionSpec]:
    candidates = [
        ({}, {}),
        ({"keyword_bias": True}, {}),
        ({"noise_reduction": True}, {}),
        ({"volume_normalization": True}, {}),
        ({"noise_reduction": True, "volume_normalization": True}, {}),
        ({"keyword_bias": True, "noise_reduction": True, "volume_normalization": True}, {}),
        ({}, {"llm": True}),
        ({}, {"llm": True, "rag": True}),
        ({}, {"llm": True, "search": True}),
        ({}, {"llm": True, "rag": True, "search": True}),
        ({"keyword_bias": True}, {"llm": True}),
        ({"keyword_bias": True}, {"llm": True, "rag": True}),
        ({"keyword_bias": True, "noise_reduction": True, "volume_normalization": True}, {"llm": True}),
        (
            {"keyword_bias": True, "noise_reduction": True, "volume_normalization": True},
            {"llm": True, "rag": True, "search": True},
        ),
    ]
    conditions: List[ConditionSpec] = []
    seen = set()
    for pre, post in candidates:
        if pre.get("keyword_bias") and not include_keyword_bias:
            continue
        if pre.get("noise_reduction") and not include_noise_reduction:
            continue
        if pre.get("volume_normalization") and not include_volume_normalization:
            continue
        if post.get("llm") and not include_llm_postprocess:
            continue
        if post.get("rag") and not include_rag:
            continue
        if post.get("search") and not include_search:
            continue
        condition = _condition_from_modes(pre, post)
        if condition.condition_id not in seen:
            seen.add(condition.condition_id)
            conditions.append(condition)
    return conditions


def _pre_modes(include_keyword_bias: bool, include_noise_reduction: bool, include_volume_normalization: bool):
    axes = []
    if include_keyword_bias:
        axes.append("keyword_bias")
    if include_noise_reduction:
        axes.append("noise_reduction")
    if include_volume_normalization:
        axes.append("volume_normalization")
    modes = [{}]
    for width in range(1, len(axes) + 1):
        for combo in itertools.combinations(axes, width):
            modes.append({axis: True for axis in combo})
    return modes


def _post_modes(include_llm_postprocess: bool, include_rag: bool, include_search: bool):
    modes = [{}]
    if not include_llm_postprocess:
        return modes
    modes.append({"llm": True})
    if include_rag:
        modes.append({"llm": True, "rag": True})
    if include_search:
        modes.append({"llm": True, "search": True})
    if include_rag and include_search:
        modes.append({"llm": True, "rag": True, "search": True})
    return modes


def _condition_from_modes(pre: Dict[str, bool], post: Dict[str, bool]) -> ConditionSpec:
    parts = []
    if pre.get("keyword_bias"):
        parts.append("keyword")
    if pre.get("noise_reduction"):
        parts.append("noise")
    if pre.get("volume_normalization"):
        parts.append("volume")
    if post.get("llm"):
        parts.append("llm")
    if post.get("rag"):
        parts.append("rag")
    if post.get("search"):
        parts.append("search")
    condition_id = "baseline" if not parts else "__".join(parts)
    group = "baseline" if not parts else ("post" if post and not pre else "pre_asr" if pre and not post else "mixed")
    label = " + ".join(_label_for(part) for part in parts) if parts else "All off baseline"
    return ConditionSpec(
        condition_id=condition_id,
        label=label,
        group=group,
        enable_keyword_bias=bool(pre.get("keyword_bias")),
        enable_noise_reduction=bool(pre.get("noise_reduction")),
        enable_volume_normalization=bool(pre.get("volume_normalization")),
        enable_llm_postprocess=bool(post.get("llm")),
        enable_rag=bool(post.get("rag") and post.get("llm")),
        enable_search=bool(post.get("search") and post.get("llm")),
    )


def _label_for(part: str) -> str:
    labels = {
        "keyword": "Keyword Bias",
        "noise": "Noise Reduction",
        "volume": "Volume Normalization",
        "llm": "LLM",
        "rag": "RAG",
        "search": "Search",
    }
    return labels.get(part, part)
