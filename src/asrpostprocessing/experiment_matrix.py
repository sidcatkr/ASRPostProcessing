from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

KEYWORD_STRENGTH_SWEEP = [0.25, 0.5, 0.75, 1.0]
PREPROCESS_STRENGTH_SWEEP = [0.25, 0.5, 0.75]
POST_STRENGTH_SWEEP = [0.25, 0.5, 0.75]


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
    keyword_bias_weight: Optional[float] = None
    noise_reduction_strength: Optional[float] = None
    volume_normalization_strength: Optional[float] = None
    postprocess_strength: Optional[float] = None
    rag_strength: Optional[float] = None
    search_strength: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @property
    def asr_group_key(self) -> str:
        return "|".join(
            [
                f"k={int(self.enable_keyword_bias)}",
                f"kw={_strength_key(self.keyword_bias_weight) if self.enable_keyword_bias else ''}",
                f"n={int(self.enable_noise_reduction)}",
                f"ns={_strength_key(self.noise_reduction_strength) if self.enable_noise_reduction else ''}",
                f"v={int(self.enable_volume_normalization)}",
                f"vs={_strength_key(self.volume_normalization_strength) if self.enable_volume_normalization else ''}",
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
    conditions = _full_valid_conditions(
        include_keyword_bias,
        include_noise_reduction,
        include_volume_normalization,
        include_llm_postprocess,
        include_rag,
        include_search,
    )
    if mode in {"full_strength", "full_strength_sweep", "strength", "strength_sweep"}:
        return _strength_sweep_conditions(conditions)
    return conditions


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


def _strength_sweep_conditions(conditions: List[ConditionSpec]) -> List[ConditionSpec]:
    expanded: List[ConditionSpec] = []
    for condition in conditions:
        for strengths in _strength_grid_for_condition(condition):
            expanded.append(_condition_with_strengths(condition, strengths))
    return expanded


def _strength_grid_for_condition(condition: ConditionSpec):
    axes = []
    if condition.enable_keyword_bias:
        axes.append(("keyword_bias_weight", KEYWORD_STRENGTH_SWEEP))
    if condition.enable_noise_reduction:
        axes.append(("noise_reduction_strength", PREPROCESS_STRENGTH_SWEEP))
    if condition.enable_volume_normalization:
        axes.append(("volume_normalization_strength", PREPROCESS_STRENGTH_SWEEP))
    if condition.enable_llm_postprocess:
        axes.append(("postprocess_strength", POST_STRENGTH_SWEEP))
    if condition.enable_rag:
        axes.append(("rag_strength", POST_STRENGTH_SWEEP))
    if condition.enable_search:
        axes.append(("search_strength", POST_STRENGTH_SWEEP))
    if not axes:
        yield {}
        return
    keys = [axis[0] for axis in axes]
    values = [axis[1] for axis in axes]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _condition_with_strengths(condition: ConditionSpec, strengths: Dict[str, float]) -> ConditionSpec:
    if not strengths:
        return condition
    suffix_parts = [_strength_suffix(key, value) for key, value in strengths.items()]
    suffix = "__" + "__".join(suffix_parts)
    label_suffix = " (" + ", ".join(part.replace("_", " ") for part in suffix_parts) + ")"
    return ConditionSpec(
        condition_id=f"{condition.condition_id}{suffix}",
        label=f"{condition.label}{label_suffix}",
        group=f"{condition.group}_strength",
        enable_keyword_bias=condition.enable_keyword_bias,
        enable_noise_reduction=condition.enable_noise_reduction,
        enable_volume_normalization=condition.enable_volume_normalization,
        enable_llm_postprocess=condition.enable_llm_postprocess,
        enable_rag=condition.enable_rag,
        enable_search=condition.enable_search,
        keyword_bias_weight=strengths.get("keyword_bias_weight"),
        noise_reduction_strength=strengths.get("noise_reduction_strength"),
        volume_normalization_strength=strengths.get("volume_normalization_strength"),
        postprocess_strength=strengths.get("postprocess_strength"),
        rag_strength=strengths.get("rag_strength"),
        search_strength=strengths.get("search_strength"),
    )


def _strength_suffix(key: str, value: float) -> str:
    prefixes = {
        "keyword_bias_weight": "kw",
        "noise_reduction_strength": "noise",
        "volume_normalization_strength": "vol",
        "postprocess_strength": "post",
        "rag_strength": "rag",
        "search_strength": "search",
    }
    return f"{prefixes.get(key, key)}{_strength_key(value)}"


def _strength_key(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}".rstrip("0").rstrip(".").replace(".", "p")


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
