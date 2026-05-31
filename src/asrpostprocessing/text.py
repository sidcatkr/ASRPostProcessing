from __future__ import annotations

import re
import unicodedata
from difflib import HtmlDiff
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")


def normalize_text(text: str, remove_spaces: bool = False, lowercase_latin: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if lowercase_latin:
        text = "".join(char.lower() if char.isascii() else char for char in text)
    if remove_spaces:
        text = re.sub(r"\s+", "", text)
    return text


def levenshtein(a: Sequence[T], b: Sequence[T]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[T], hypothesis: Sequence[T]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / float(len(reference))


def cer(reference: str, hypothesis: str, remove_spaces: bool = False) -> float:
    ref = normalize_text(reference, remove_spaces=remove_spaces)
    hyp = normalize_text(hypothesis, remove_spaces=remove_spaces)
    return error_rate(list(ref), list(hyp))


def wer_eojeol(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    return error_rate(ref, hyp)


def character_f1(a: str, b: str) -> float:
    a_norm = normalize_text(a, remove_spaces=True)
    b_norm = normalize_text(b, remove_spaces=True)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    a_counts = _counts(a_norm)
    b_counts = _counts(b_norm)
    overlap = sum(min(a_counts.get(char, 0), b_counts.get(char, 0)) for char in a_counts)
    precision = overlap / float(len(b_norm)) if b_norm else 0.0
    recall = overlap / float(len(a_norm)) if a_norm else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def make_diff_html(reference: str, hypothesis: str) -> str:
    ref_lines = (reference or "").splitlines() or [reference or ""]
    hyp_lines = (hypothesis or "").splitlines() or [hypothesis or ""]
    return HtmlDiff(wrapcolumn=100).make_table(ref_lines, hyp_lines, "Raw", "Corrected", context=True)


def merge_overlapping_texts(texts: Iterable[str], max_overlap: int = 120) -> str:
    merged = ""
    for text in texts:
        if not text:
            continue
        if not merged:
            merged = text
            continue
        overlap = _suffix_prefix_overlap(merged, text, max_overlap=max_overlap)
        merged += text[overlap:]
    return merged


def _suffix_prefix_overlap(left: str, right: str, max_overlap: int) -> int:
    max_len = min(len(left), len(right), max_overlap)
    for size in range(max_len, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _counts(text: str) -> dict:
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
    return counts
