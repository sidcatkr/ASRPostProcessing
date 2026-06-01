from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")
_DIFF_TOKEN_RE = re.compile(r"\s+|[^\s]+")


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


def make_diff_html(reference: str, hypothesis: str, reference_label: str = "Raw", hypothesis_label: str = "Corrected") -> str:
    reference = reference or ""
    hypothesis = hypothesis or ""
    body, stats = _inline_diff_body(reference, hypothesis)
    if not body:
        body = '<span class="asrpp-diff-empty">(empty)</span>'
    no_change = stats["delete"] == 0 and stats["insert"] == 0 and stats["replace"] == 0
    no_change_pill = '<span class="asrpp-diff-pill">No character changes</span>' if no_change else ""
    escaped_reference_label = html.escape(reference_label or "Reference")
    escaped_hypothesis_label = html.escape(hypothesis_label or "Hypothesis")
    return f"""
<div class="asrpp-inline-diff" role="region" aria-label="Inline transcript diff">
  <style>
    .asrpp-inline-diff {{
      box-sizing: border-box;
      width: 100%;
      border: 1px solid #d8dee4;
      border-radius: 8px;
      background: #ffffff;
      color: #1f2328;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }}
    .asrpp-diff-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      border-bottom: 1px solid #d8dee4;
      background: #f6f8fa;
      font-size: 12px;
      line-height: 1.35;
    }}
    .asrpp-diff-pill {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border: 1px solid #d0d7de;
      border-radius: 999px;
      background: #ffffff;
      white-space: nowrap;
    }}
    .asrpp-diff-pill.insert {{ border-color: #8fd19e; background: #dafbe1; }}
    .asrpp-diff-pill.delete {{ border-color: #ffb3ad; background: #ffebe9; }}
    .asrpp-diff-pill.replace {{ border-color: #d4a72c; background: #fff8c5; }}
    .asrpp-diff-text {{
      max-height: 62vh;
      overflow: auto;
      padding: 14px 16px;
      font-size: 15px;
      line-height: 1.75;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .asrpp-diff-delete,
    .asrpp-diff-insert {{
      border-radius: 3px;
      padding: 0 2px;
      text-decoration-thickness: 1px;
      text-underline-offset: 2px;
    }}
    .asrpp-diff-delete {{
      background: #ffebe9;
      color: #82071e;
      text-decoration: line-through;
    }}
    .asrpp-diff-insert {{
      background: #dafbe1;
      color: #116329;
      text-decoration: none;
    }}
    .asrpp-diff-replace {{
      border-radius: 4px;
      background: #fff8c5;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
    }}
    .asrpp-diff-empty {{ color: #57606a; }}
  </style>
  <div class="asrpp-diff-bar">
    <span class="asrpp-diff-pill">{escaped_reference_label} {len(reference):,} chars</span>
    <span class="asrpp-diff-pill">{escaped_hypothesis_label} {len(hypothesis):,} chars</span>
    <span class="asrpp-diff-pill delete">-{stats["delete"]}</span>
    <span class="asrpp-diff-pill insert">+{stats["insert"]}</span>
    <span class="asrpp-diff-pill replace">~{stats["replace"]}</span>
    {no_change_pill}
  </div>
  <div class="asrpp-diff-text">{body}</div>
</div>
""".strip()


def _inline_diff_body(reference: str, hypothesis: str) -> tuple[str, dict]:
    ref_tokens = _diff_tokens(reference)
    hyp_tokens = _diff_tokens(hypothesis)
    matcher = SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    parts: List[str] = []
    stats = {"delete": 0, "insert": 0, "replace": 0}
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        ref_text = "".join(ref_tokens[ref_start:ref_end])
        hyp_text = "".join(hyp_tokens[hyp_start:hyp_end])
        if tag == "equal":
            parts.append(_escape_diff_text(hyp_text))
        elif tag == "delete":
            stats["delete"] += len(ref_text)
            parts.append(_diff_span("del", "asrpp-diff-delete", ref_text))
        elif tag == "insert":
            stats["insert"] += len(hyp_text)
            parts.append(_diff_span("ins", "asrpp-diff-insert", hyp_text))
        elif tag == "replace":
            stats["replace"] += max(len(ref_text), len(hyp_text))
            parts.append(_render_replacement(ref_text, hyp_text))
    return "".join(parts), stats


def _diff_tokens(text: str) -> List[str]:
    return _DIFF_TOKEN_RE.findall(text or "")


def _render_replacement(reference: str, hypothesis: str) -> str:
    if len(reference) + len(hypothesis) <= 240:
        matcher = SequenceMatcher(a=list(reference), b=list(hypothesis), autojunk=False)
        parts: List[str] = []
        for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
            ref_text = reference[ref_start:ref_end]
            hyp_text = hypothesis[hyp_start:hyp_end]
            if tag == "equal":
                parts.append(_escape_diff_text(hyp_text))
            elif tag == "delete":
                parts.append(_diff_span("del", "asrpp-diff-delete", ref_text))
            elif tag == "insert":
                parts.append(_diff_span("ins", "asrpp-diff-insert", hyp_text))
            elif tag == "replace":
                parts.append(_diff_span("del", "asrpp-diff-delete", ref_text))
                parts.append(_diff_span("ins", "asrpp-diff-insert", hyp_text))
        return f'<span class="asrpp-diff-replace">{"".join(parts)}</span>'
    return (
        '<span class="asrpp-diff-replace">'
        f'{_diff_span("del", "asrpp-diff-delete", reference)}'
        f'{_diff_span("ins", "asrpp-diff-insert", hypothesis)}'
        "</span>"
    )


def _diff_span(tag: str, class_name: str, text: str) -> str:
    return f'<{tag} class="{class_name}">{_escape_diff_text(text)}</{tag}>'


def _escape_diff_text(text: str) -> str:
    return html.escape(text, quote=False)


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
