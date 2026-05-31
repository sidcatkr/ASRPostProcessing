from __future__ import annotations

from typing import Iterable, List

from .config import clamp01

WEIGHT_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def quantize_keyword_weight(weight: float) -> float:
    weight = clamp01(weight)
    return min(WEIGHT_LEVELS, key=lambda level: abs(level - weight))


def normalize_keywords(keywords: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for keyword in keywords or []:
        cleaned = " ".join(str(keyword).strip().split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def build_keyword_bias_instruction(keywords: Iterable[str], weight: float) -> str:
    keywords = normalize_keywords(keywords)
    level = quantize_keyword_weight(weight)
    if not keywords or level <= 0:
        return ""
    joined = ", ".join(keywords)
    base = (
        "아래 키워드는 음성에 등장할 수 있습니다. "
        "발음과 문맥상 타당할 때만 사용하고, 들리지 않는 단어를 추가하지 마세요."
    )
    if level == 0.25:
        return f"{base}\n키워드: {joined}"
    if level == 0.5:
        return (
            f"{base}\n키워드 후보: {joined}\n"
            "한글식 발음으로 들린 기술 용어는 위 후보와 맞는 경우 원래 표기로 전사하세요."
        )
    if level == 0.75:
        return (
            f"{base}\n중요 키워드 후보: {joined}\n"
            "ASR 결과에서 유사 발음이 나오면 이 목록을 우선 참고하되, 의미가 바뀌면 원음을 유지하세요."
        )
    repeated = "; ".join(keywords + keywords)
    return (
        f"{base}\n강한 키워드 후보: {joined}\n반복 확인 목록: {repeated}\n"
        "단, 실제 음성 근거 없이 키워드를 삽입하면 안 됩니다."
    )
