from __future__ import annotations

import re
from typing import List, Tuple

from .config import ExperimentConfig
from .keyword_bias import normalize_keywords
from .schemas import CorrectionResult, Edit


def apply_keyword_near_miss_corrections(result: CorrectionResult, config: ExperimentConfig) -> CorrectionResult:
    if float(getattr(config, "postprocess_strength", 0.0) or 0.0) < 0.5:
        return result
    keywords = [keyword for keyword in normalize_keywords(config.keywords) if hangul_count(keyword) >= 2]
    if not keywords or not result.corrected_text:
        return result
    replacements = keyword_near_miss_replacements(result.corrected_text, keywords)
    if not replacements:
        return result
    text = result.corrected_text
    for start, end, _before, after in reversed(replacements):
        text = text[:start] + after + text[end:]
    edits = [
        Edit(
            before=before,
            after=after,
            reason="Keyword-guided ASR near-miss correction.",
            confidence=0.82,
            start_char=start,
            end_char=end,
        )
        for start, end, before, after in replacements
    ]
    result.corrected_text = text
    result.edits.extend(edits)
    result.metadata.setdefault("keyword_near_miss_corrections", [])
    result.metadata["keyword_near_miss_corrections"].extend(edit.to_dict() for edit in edits)
    if result.risk in {"unknown", "unchanged"}:
        result.risk = "low"
    return result


def keyword_near_miss_replacements(text: str, keywords: List[str]) -> List[Tuple[int, int, str, str]]:
    spans = list(re.finditer(r"[A-Za-z0-9_\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+", text))
    replacements: List[Tuple[int, int, str, str]] = []
    used_ranges: List[Tuple[int, int]] = []
    for keyword in keywords:
        keyword_tokens = _word_tokens(keyword)
        if len(keyword_tokens) < 2 or len(keyword_tokens) > 4:
            continue
        keyword_key = _phrase_key(keyword)
        if len(keyword_key) < 4:
            continue
        for index in range(0, len(spans) - len(keyword_tokens) + 1):
            window = spans[index : index + len(keyword_tokens)]
            start, end = window[0].start(), window[-1].end()
            if any(not (end <= used_start or start >= used_end) for used_start, used_end in used_ranges):
                continue
            before = text[start:end]
            if _phrase_key(before) == keyword_key:
                continue
            after = _keyword_near_miss_after(before, keyword, keyword_tokens, keyword_key)
            if after:
                replacements.append((start, end, before, after))
                used_ranges.append((start, end))
    return sorted(replacements, key=lambda item: item[0])


def _keyword_near_miss_after(before: str, keyword: str, keyword_tokens: List[str], keyword_key: str) -> str:
    before_tokens = _word_tokens(before)
    if len(before_tokens) != len(keyword_tokens):
        return ""
    adjusted_tokens = []
    suffixes = []
    for before_token, keyword_token in zip(before_tokens, keyword_tokens):
        adjusted, suffix = _strip_keyword_particle_suffix(before_token, keyword_token)
        adjusted_tokens.append(adjusted)
        suffixes.append(suffix)
    token_pairs = list(zip(adjusted_tokens, keyword_tokens))
    if not any(_phrase_key(left) == _phrase_key(right) for left, right in token_pairs):
        return ""
    non_exact_pairs = [(left, right) for left, right in token_pairs if _phrase_key(left) != _phrase_key(right)]
    if not non_exact_pairs:
        return ""
    if any(_hangul_phonetic_distance_ratio(left, right) > 0.7 for left, right in non_exact_pairs):
        return ""
    before_key = "".join(_phrase_key(token) for token in adjusted_tokens)
    if len(before_key) < 4 or hangul_count(before_key) < 2:
        return ""
    distance = _levenshtein_distance(before_key, keyword_key)
    threshold = max(1, min(3, int(round(len(keyword_key) * 0.5))))
    if not (0 < distance <= threshold):
        return ""
    after_tokens = [keyword_token + suffix for keyword_token, suffix in zip(keyword_tokens, suffixes)]
    return " ".join(after_tokens)


def _strip_keyword_particle_suffix(before_token: str, keyword_token: str) -> Tuple[str, str]:
    before_key = _phrase_key(before_token)
    keyword_key = _phrase_key(keyword_token)
    if before_key.startswith(keyword_key):
        suffix = before_token[len(keyword_token) :]
        if suffix in {"은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "로", "으로", "와", "과"}:
            return keyword_token, suffix
    return before_token, ""


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+", text or "")


def _phrase_key(text: str) -> str:
    return "".join(_word_tokens(text)).lower()


def hangul_count(text: str) -> int:
    return len(re.findall(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", text or ""))


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]


HANGUL_CHO = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
HANGUL_JUNG = [
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]
HANGUL_JONG = [
    "",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]


def _hangul_phonetic_distance_ratio(left: str, right: str) -> float:
    left_key = _hangul_phonetic_key(left)
    right_key = _hangul_phonetic_key(right)
    if not left_key or not right_key:
        return 1.0
    return _levenshtein_distance(left_key, right_key) / max(len(left_key), len(right_key))


def _hangul_phonetic_key(text: str) -> str:
    parts = []
    for char in text or "":
        code = ord(char) - 0xAC00
        if 0 <= code < 11172:
            cho = code // 588
            jung = (code % 588) // 28
            jong = code % 28
            parts.append(HANGUL_CHO[cho])
            parts.append(HANGUL_JUNG[jung])
            if HANGUL_JONG[jong]:
                parts.append(HANGUL_JONG[jong])
        else:
            parts.append(char.lower())
    return "".join(parts)
