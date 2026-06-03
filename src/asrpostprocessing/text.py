from __future__ import annotations

import html
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
_DIFF_TOKEN_RE = re.compile(r"\s+|[^\s]+")
_ERROR_MONITOR_MAX_ROWS = 12
_DIFF_CHANGE_MAX_ROWS = 10


def normalize_text(
    text: str,
    remove_spaces: bool = False,
    lowercase_latin: bool = True,
    remove_symbols: bool = False,
) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if lowercase_latin:
        text = "".join(char.lower() if char.isascii() else char for char in text)
    if remove_symbols:
        text = "".join(char for char in text if not _is_ignored_metric_symbol(char))
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


def cer(reference: str, hypothesis: str, remove_spaces: bool = False, remove_symbols: bool = False) -> float:
    ref = normalize_text(reference, remove_spaces=remove_spaces, remove_symbols=remove_symbols)
    hyp = normalize_text(hypothesis, remove_spaces=remove_spaces, remove_symbols=remove_symbols)
    return error_rate(list(ref), list(hyp))


def wer_eojeol(reference: str, hypothesis: str) -> float:
    ref = spacing_insensitive_tokens(reference)
    hyp = spacing_insensitive_tokens(hypothesis)
    return error_rate(ref, hyp)


def spacing_insensitive_tokens(text: str) -> List[str]:
    normalized = normalize_text(text, remove_spaces=True, remove_symbols=True)
    tokens: List[str] = []
    buffer: List[str] = []
    buffer_kind = ""

    def flush() -> None:
        nonlocal buffer, buffer_kind
        if buffer:
            tokens.append("".join(buffer))
            buffer = []
            buffer_kind = ""

    for char in normalized:
        kind = _wer_token_kind(char)
        if kind == "hangul":
            flush()
            tokens.append(char)
            continue
        if kind == "alnum":
            if buffer_kind != kind:
                flush()
                buffer_kind = kind
            buffer.append(char)
            continue
        flush()
        tokens.append(char)
    flush()
    return tokens


def _wer_token_kind(char: str) -> str:
    if "\uac00" <= char <= "\ud7a3":
        return "hangul"
    if char.isalnum():
        return "alnum"
    return "symbol"


def _is_ignored_metric_symbol(char: str) -> bool:
    return unicodedata.category(char)[:1] in {"P", "S"}


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


def make_diff_html(
    reference: str,
    hypothesis: str,
    reference_label: str = "Raw",
    hypothesis_label: str = "Corrected",
    show_error_monitor: bool = False,
) -> str:
    return _make_diff_html(
        reference,
        hypothesis,
        reference_label,
        hypothesis_label,
        character_level=False,
        show_error_monitor=show_error_monitor,
    )


def make_character_diff_html(
    reference: str,
    hypothesis: str,
    reference_label: str = "Raw",
    hypothesis_label: str = "Corrected",
) -> str:
    return _make_diff_html(
        reference,
        hypothesis,
        reference_label,
        hypothesis_label,
        character_level=True,
        show_error_monitor=False,
    )


def make_diff_export_document(
    body_html: str,
    title: str = "Transcript Diff Export",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    escaped_title = html.escape(title or "Transcript Diff Export")
    metadata_html = _diff_export_metadata_html(metadata or {})
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      --font: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --monospace-font: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --block-radius: 8px;
      --block-background-fill: #ffffff;
      --background-fill-primary: #f7f7f8;
      --background-fill-secondary: rgba(127, 127, 127, 0.08);
      --input-background-fill: rgba(127, 127, 127, 0.06);
      --border-color-primary: #d0d7de;
      --body-text-color: #1f2328;
      --body-text-color-subdued: #6e7781;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --block-background-fill: #1f2026;
        --background-fill-primary: #0f1117;
        --background-fill-secondary: rgba(255, 255, 255, 0.06);
        --input-background-fill: rgba(255, 255, 255, 0.06);
        --border-color-primary: #3f4652;
        --body-text-color: #f3f4f6;
        --body-text-color-subdued: #b4bbc6;
      }}
    }}
    body {{
      margin: 0;
      background: var(--background-fill-primary);
      color: var(--body-text-color);
      font-family: var(--font);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }}
    .asrpp-export-header {{
      margin-bottom: 16px;
    }}
    .asrpp-export-header h1 {{
      margin: 0 0 8px 0;
      font-size: 22px;
      line-height: 1.25;
    }}
    .asrpp-export-metadata {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0;
      color: var(--body-text-color-subdued);
      font-size: 12px;
    }}
    .asrpp-export-metadata div {{
      border: 1px solid var(--border-color-primary);
      border-radius: 6px;
      padding: 4px 7px;
      background: var(--block-background-fill);
    }}
    .asrpp-diff-section + .asrpp-diff-section {{
      margin-top: 16px;
    }}
    .asrpp-diff-section h3 {{
      color: var(--body-text-color);
    }}
    @media print {{
      main {{ max-width: none; padding: 12mm; }}
      .asrpp-diff-text {{ max-height: none !important; overflow: visible !important; }}
      .asrpp-error-table-wrap {{ max-height: none !important; overflow: visible !important; }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="asrpp-export-header">
      <h1>{escaped_title}</h1>
      {metadata_html}
    </header>
    {body_html or ""}
  </main>
</body>
</html>
"""


def transcript_error_breakdown(reference: str, hypothesis: str) -> Dict[str, Any]:
    return {
        "cer": _sequence_error_summary(
            list(normalize_text(reference, remove_spaces=True, remove_symbols=True)),
            list(normalize_text(hypothesis, remove_spaces=True, remove_symbols=True)),
            "CER",
            "char",
            "",
            context_window=8,
        ),
        "wer": _sequence_error_summary(
            spacing_insensitive_tokens(reference),
            spacing_insensitive_tokens(hypothesis),
            "WER",
            "token",
            " ",
            context_window=6,
        ),
    }


def _diff_export_metadata_html(metadata: Dict[str, Any]) -> str:
    if not metadata:
        return ""
    rows = []
    for key, value in metadata.items():
        if value is None or value == "":
            continue
        rows.append(f"<div><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</div>")
    if not rows:
        return ""
    return '<div class="asrpp-export-metadata">' + "".join(rows) + "</div>"


def _make_diff_html(
    reference: str,
    hypothesis: str,
    reference_label: str,
    hypothesis_label: str,
    character_level: bool,
    show_error_monitor: bool,
) -> str:
    reference = reference or ""
    hypothesis = hypothesis or ""
    body, stats, changes = _inline_diff_body(reference, hypothesis, character_level=character_level)
    if not body:
        body = '<span class="asrpp-diff-empty">(empty)</span>'
    no_change = stats["delete"] == 0 and stats["insert"] == 0 and stats["replace"] == 0
    no_change_pill = '<span class="asrpp-diff-count">No character changes</span>' if no_change else ""
    escaped_reference_label = html.escape(reference_label or "Reference")
    escaped_hypothesis_label = html.escape(hypothesis_label or "Hypothesis")
    error_monitor = _error_monitor_html(reference, hypothesis) if show_error_monitor else ""
    change_summary = _diff_change_summary_html(changes, escaped_reference_label, escaped_hypothesis_label)
    return f"""
<div class="asrpp-inline-diff" role="region" aria-label="Inline transcript diff">
  <style>
    .asrpp-inline-diff {{
      box-sizing: border-box;
      width: 100%;
      border: 1px solid var(--border-color-primary, #d0d7de);
      border-radius: var(--block-radius, 8px);
      background: var(--block-background-fill, #ffffff);
      color: var(--body-text-color, #1f2328);
      font-family: var(--font, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
      overflow: hidden;
    }}
    .asrpp-diff-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 9px 12px;
      border-bottom: 1px solid var(--border-color-primary, #d0d7de);
      background: var(--background-fill-secondary, rgba(127, 127, 127, 0.08));
      color: var(--body-text-color, #1f2328);
      font-size: 12px;
      line-height: 1.35;
    }}
    .asrpp-diff-title {{
      font-weight: 650;
      margin-right: 2px;
    }}
    .asrpp-diff-meta {{
      color: var(--body-text-color-subdued, #6e7781);
      margin-right: 4px;
    }}
    .asrpp-diff-count {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border: 1px solid var(--border-color-primary, #d0d7de);
      border-radius: 6px;
      background: var(--input-background-fill, rgba(127, 127, 127, 0.06));
      color: var(--body-text-color, #1f2328);
      font-weight: 500;
      white-space: nowrap;
    }}
    .asrpp-diff-count.insert {{ border-color: rgba(46, 160, 67, 0.42); }}
    .asrpp-diff-count.delete {{ border-color: rgba(207, 34, 46, 0.42); }}
    .asrpp-diff-count.replace {{ border-color: rgba(154, 103, 0, 0.42); }}
    .asrpp-diff-legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      padding: 8px 12px;
      border-bottom: 1px solid var(--border-color-primary, #d0d7de);
      color: var(--body-text-color-subdued, #6e7781);
      font-size: 12px;
      line-height: 1.45;
    }}
    .asrpp-diff-legend span {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }}
    .asrpp-diff-text {{
      max-height: 62vh;
      overflow: auto;
      padding: 14px 16px;
      background: var(--block-background-fill, #ffffff);
      color: var(--body-text-color, #1f2328);
      font-size: 15px;
      line-height: 1.85;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .asrpp-diff-delete,
    .asrpp-diff-insert {{
      border-radius: 3px;
      padding: 0 3px;
      text-decoration-thickness: 1px;
      text-underline-offset: 2px;
    }}
    .asrpp-diff-delete {{
      background: rgba(207, 34, 46, 0.16);
      color: var(--body-text-color, #1f2328);
      text-decoration: line-through;
    }}
    .asrpp-diff-insert {{
      background: rgba(46, 160, 67, 0.16);
      color: var(--body-text-color, #1f2328);
      text-decoration: none;
    }}
    .asrpp-diff-replace {{
      border-radius: 4px;
      background: rgba(187, 128, 9, 0.13);
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
    }}
    .asrpp-diff-empty {{ color: var(--body-text-color-subdued, #6e7781); }}
    .asrpp-diff-change-summary {{
      border-top: 1px solid var(--border-color-primary, #d0d7de);
      background: var(--background-fill-secondary, rgba(127, 127, 127, 0.05));
      padding: 10px 12px 12px;
    }}
    .asrpp-diff-change-heading {{
      margin: 0 0 8px 0;
      color: var(--body-text-color, #1f2328);
      font-size: 13px;
      font-weight: 650;
    }}
    .asrpp-diff-change-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 7px;
    }}
    .asrpp-diff-change-item {{
      display: grid;
      grid-template-columns: minmax(96px, 142px) 1fr;
      gap: 8px;
      align-items: start;
      border: 1px solid var(--border-color-primary, #d0d7de);
      border-radius: 6px;
      background: var(--block-background-fill, #ffffff);
      padding: 8px 9px;
      font-size: 12px;
      line-height: 1.45;
    }}
    .asrpp-diff-change-kind {{
      font-weight: 650;
      color: var(--body-text-color, #1f2328);
    }}
    .asrpp-diff-change-copy {{
      color: var(--body-text-color, #1f2328);
      overflow-wrap: anywhere;
    }}
    .asrpp-diff-change-copy code {{
      font-family: var(--monospace-font, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace);
      font-size: 12px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .asrpp-diff-change-context {{
      margin-top: 4px;
      color: var(--body-text-color-subdued, #6e7781);
    }}
    .asrpp-diff-change-muted {{
      color: var(--body-text-color-subdued, #6e7781);
    }}
    .asrpp-error-monitor {{
      border-top: 1px solid var(--border-color-primary, #d0d7de);
      background: var(--block-background-fill, #ffffff);
      padding: 12px;
    }}
    .asrpp-error-monitor h4 {{
      margin: 0 0 8px 0;
      color: var(--body-text-color, #1f2328);
      font-size: 13px;
      font-weight: 650;
    }}
    .asrpp-error-note {{
      margin: 0 0 10px 0;
      color: var(--body-text-color-subdued, #6e7781);
      font-size: 12px;
      line-height: 1.45;
    }}
    .asrpp-error-cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
      margin-bottom: 10px;
    }}
    .asrpp-error-card {{
      border: 1px solid var(--border-color-primary, #d0d7de);
      border-radius: 6px;
      background: var(--background-fill-secondary, rgba(127, 127, 127, 0.05));
      padding: 8px 9px;
    }}
    .asrpp-error-card span {{
      display: block;
      color: var(--body-text-color-subdued, #6e7781);
      font-size: 11px;
    }}
    .asrpp-error-card strong {{
      display: block;
      color: var(--body-text-color, #1f2328);
      font-size: 14px;
      font-variant-numeric: tabular-nums;
      margin-top: 2px;
    }}
    .asrpp-error-section {{
      margin-top: 10px;
    }}
    .asrpp-error-section h5 {{
      margin: 0 0 6px 0;
      color: var(--body-text-color, #1f2328);
      font-size: 12px;
      font-weight: 650;
    }}
    .asrpp-error-table-wrap {{
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--border-color-primary, #d0d7de);
      border-radius: 6px;
    }}
    .asrpp-error-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .asrpp-error-table th,
    .asrpp-error-table td {{
      padding: 6px 7px;
      border-bottom: 1px solid var(--border-color-primary, #d0d7de);
      vertical-align: top;
      text-align: left;
    }}
    .asrpp-error-table th {{
      position: sticky;
      top: 0;
      background: var(--background-fill-secondary, rgba(127, 127, 127, 0.08));
      color: var(--body-text-color, #1f2328);
      z-index: 1;
    }}
    .asrpp-error-table td {{
      color: var(--body-text-color, #1f2328);
    }}
    .asrpp-error-muted {{
      color: var(--body-text-color-subdued, #6e7781);
    }}
    .asrpp-error-context {{
      color: var(--body-text-color-subdued, #6e7781);
      overflow-wrap: anywhere;
    }}
    @media (max-width: 640px) {{
      .asrpp-diff-change-item {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
  <div class="asrpp-diff-bar">
    <span class="asrpp-diff-title">{escaped_reference_label} -> {escaped_hypothesis_label}</span>
    <span class="asrpp-diff-meta">{len(reference):,} -> {len(hypothesis):,} chars</span>
    <span class="asrpp-diff-count delete">Deleted {stats["delete"]}</span>
    <span class="asrpp-diff-count insert">Inserted {stats["insert"]}</span>
    <span class="asrpp-diff-count replace">Replaced {stats["replace"]}</span>
    {no_change_pill}
  </div>
  <div class="asrpp-diff-legend" aria-label="Diff legend">
    <span><del class="asrpp-diff-delete">deleted</del> removed from {escaped_reference_label}</span>
    <span><ins class="asrpp-diff-insert">inserted</ins> added in {escaped_hypothesis_label}</span>
    <span><span class="asrpp-diff-replace">replacement</span> old text followed by new text</span>
  </div>
  <div class="asrpp-diff-text">{body}</div>
  {change_summary}
  {error_monitor}
</div>
""".strip()


def _error_monitor_html(reference: str, hypothesis: str) -> str:
    breakdown = transcript_error_breakdown(reference, hypothesis)
    cer_summary = breakdown["cer"]
    wer_summary = breakdown["wer"]
    return f"""
  <div class="asrpp-error-monitor" aria-label="CER and WER error monitor">
    <h4>CER/WER error monitor</h4>
    <p class="asrpp-error-note">CER/WER use one-line normalized text without spacing, line breaks, punctuation, or symbols. Formatting-only differences are not counted here.</p>
    <div class="asrpp-error-cards">
      {_error_summary_card_html(cer_summary)}
      {_error_summary_card_html(wer_summary)}
    </div>
    {_error_segments_section_html(cer_summary)}
    {_error_segments_section_html(wer_summary)}
  </div>
""".rstrip()


def _error_summary_card_html(summary: Dict[str, Any]) -> str:
    return (
        '<div class="asrpp-error-card">'
        f"<span>{html.escape(str(summary['metric']))}</span>"
        f"<strong>{_format_error_percent(summary['error_rate'])}</strong>"
        f"<span>{int(summary['errors'])}/{int(summary['reference_units'])} {html.escape(str(summary['unit_label']))} error units</span>"
        "</div>"
    )


def _error_segments_section_html(summary: Dict[str, Any]) -> str:
    segments = summary.get("segments") or []
    metric = html.escape(str(summary.get("metric") or "Error"))
    if not segments:
        return f'<section class="asrpp-error-section"><h5>{metric} locations</h5><p class="asrpp-error-note">No {metric} errors after metric normalization.</p></section>'
    shown = segments[:_ERROR_MONITOR_MAX_ROWS]
    omitted = len(segments) - len(shown)
    rows = "\n".join(_error_segment_row_html(segment) for segment in shown)
    omitted_note = (
        f'<p class="asrpp-error-note">Showing first {_ERROR_MONITOR_MAX_ROWS} of {len(segments)} error clusters in transcript order.</p>'
        if omitted > 0
        else ""
    )
    return f"""
    <section class="asrpp-error-section">
      <h5>{metric} locations</h5>
      {omitted_note}
      <div class="asrpp-error-table-wrap">
        <table class="asrpp-error-table">
          <thead>
            <tr>
              <th>Ref pos</th>
              <th>Type</th>
              <th>Reference</th>
              <th>Hypothesis</th>
              <th>Context</th>
              <th>Rate part</th>
              <th>Error share</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>
""".rstrip()


def _error_segment_row_html(segment: Dict[str, Any]) -> str:
    return (
        "<tr>"
        f'<td class="asrpp-error-muted">{html.escape(str(segment["position_label"]))}</td>'
        f"<td>{html.escape(str(segment['operation']))}</td>"
        f"<td>{html.escape(str(segment['reference_text']))}</td>"
        f"<td>{html.escape(str(segment['hypothesis_text']))}</td>"
        f'<td class="asrpp-error-context">{html.escape(str(segment["context"]))}</td>'
        f'<td class="asrpp-error-muted">{int(segment["error_count"])}/{int(segment["reference_units"])} = {_format_error_percent(segment["rate_contribution"])}</td>'
        f'<td class="asrpp-error-muted">{_format_error_percent(segment["error_share"])}</td>'
        "</tr>"
    )


def _sequence_error_summary(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    metric: str,
    unit_label: str,
    joiner: str,
    context_window: int,
) -> Dict[str, Any]:
    operations = _levenshtein_alignment(reference, hypothesis)
    errors = sum(1 for operation in operations if operation["tag"] != "equal")
    reference_units = len(reference)
    error_rate_value = error_rate(reference, hypothesis)
    segments = _error_segments(operations, reference, errors, reference_units, unit_label, joiner, context_window)
    return {
        "metric": metric,
        "unit_label": unit_label,
        "errors": errors,
        "reference_units": reference_units,
        "error_rate": error_rate_value,
        "segments": segments,
    }


def _levenshtein_alignment(reference: Sequence[str], hypothesis: Sequence[str]) -> List[Dict[str, Any]]:
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    distances = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    for ref_index in range(ref_len + 1):
        distances[ref_index][0] = ref_index
    for hyp_index in range(hyp_len + 1):
        distances[0][hyp_index] = hyp_index
    for ref_index in range(1, ref_len + 1):
        for hyp_index in range(1, hyp_len + 1):
            replace_cost = 0 if reference[ref_index - 1] == hypothesis[hyp_index - 1] else 1
            distances[ref_index][hyp_index] = min(
                distances[ref_index - 1][hyp_index] + 1,
                distances[ref_index][hyp_index - 1] + 1,
                distances[ref_index - 1][hyp_index - 1] + replace_cost,
            )

    operations: List[Dict[str, Any]] = []
    ref_index = ref_len
    hyp_index = hyp_len
    while ref_index > 0 or hyp_index > 0:
        if ref_index > 0 and hyp_index > 0:
            replace_cost = 0 if reference[ref_index - 1] == hypothesis[hyp_index - 1] else 1
            if distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index - 1] + replace_cost:
                tag = "equal" if replace_cost == 0 else "replace"
                operations.append(
                    {
                        "tag": tag,
                        "ref": reference[ref_index - 1],
                        "hyp": hypothesis[hyp_index - 1],
                        "ref_pos": ref_index - 1,
                        "hyp_pos": hyp_index - 1,
                    }
                )
                ref_index -= 1
                hyp_index -= 1
                continue
        if ref_index > 0 and distances[ref_index][hyp_index] == distances[ref_index - 1][hyp_index] + 1:
            operations.append(
                {
                    "tag": "delete",
                    "ref": reference[ref_index - 1],
                    "hyp": None,
                    "ref_pos": ref_index - 1,
                    "hyp_pos": hyp_index,
                }
            )
            ref_index -= 1
            continue
        operations.append(
            {
                "tag": "insert",
                "ref": None,
                "hyp": hypothesis[hyp_index - 1],
                "ref_pos": ref_index,
                "hyp_pos": hyp_index - 1,
            }
        )
        hyp_index -= 1
    operations.reverse()
    return operations


def _error_segments(
    operations: List[Dict[str, Any]],
    reference: Sequence[str],
    total_errors: int,
    reference_units: int,
    unit_label: str,
    joiner: str,
    context_window: int,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for operation in operations:
        if operation["tag"] == "equal":
            if current is not None:
                segments.append(_finalize_error_segment(current, reference, total_errors, reference_units, unit_label, joiner, context_window))
                current = None
            continue
        if current is None:
            current = {
                "start_ref": int(operation["ref_pos"]),
                "end_ref": int(operation["ref_pos"]),
                "reference": [],
                "hypothesis": [],
                "counts": {"replace": 0, "delete": 0, "insert": 0},
            }
        current["start_ref"] = min(int(current["start_ref"]), int(operation["ref_pos"]))
        consumed_ref = operation["ref"] is not None
        end_ref = int(operation["ref_pos"]) + (1 if consumed_ref else 0)
        current["end_ref"] = max(int(current["end_ref"]), end_ref)
        if operation["ref"] is not None:
            current["reference"].append(str(operation["ref"]))
        if operation["hyp"] is not None:
            current["hypothesis"].append(str(operation["hyp"]))
        current["counts"][operation["tag"]] += 1
    if current is not None:
        segments.append(_finalize_error_segment(current, reference, total_errors, reference_units, unit_label, joiner, context_window))
    return segments


def _finalize_error_segment(
    segment: Dict[str, Any],
    reference: Sequence[str],
    total_errors: int,
    reference_units: int,
    unit_label: str,
    joiner: str,
    context_window: int,
) -> Dict[str, Any]:
    counts = segment["counts"]
    error_count = sum(int(value) for value in counts.values())
    operation_types = [key for key, value in counts.items() if value]
    start_ref = int(segment["start_ref"])
    end_ref = int(segment["end_ref"])
    context_start = max(0, start_ref - context_window)
    context_end = min(len(reference), max(end_ref, start_ref) + context_window)
    before = _join_error_units(reference[context_start:start_ref], joiner)
    after = _join_error_units(reference[end_ref:context_end], joiner)
    position_end = max(start_ref + 1, end_ref)
    return {
        "position_label": f"{start_ref + 1}-{position_end} {unit_label}",
        "operation": operation_types[0] if len(operation_types) == 1 else "mixed",
        "reference_text": _join_error_units(segment["reference"], joiner) or "(empty)",
        "hypothesis_text": _join_error_units(segment["hypothesis"], joiner) or "(empty)",
        "context": f"{before} -> {after}".strip(),
        "error_count": error_count,
        "reference_units": reference_units,
        "rate_contribution": _safe_ratio(error_count, reference_units),
        "error_share": _safe_ratio(error_count, total_errors),
    }


def _join_error_units(values: Sequence[str], joiner: str) -> str:
    return joiner.join(str(value) for value in values)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


def _format_error_percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def _inline_diff_body(reference: str, hypothesis: str, character_level: bool = False) -> tuple[str, dict, List[Dict[str, str]]]:
    ref_tokens = list(reference) if character_level else _diff_tokens(reference)
    hyp_tokens = list(hypothesis) if character_level else _diff_tokens(hypothesis)
    matcher = SequenceMatcher(a=ref_tokens, b=hyp_tokens, autojunk=not character_level)
    parts: List[str] = []
    stats = {"delete": 0, "insert": 0, "replace": 0}
    changes: List[Dict[str, str]] = []
    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        ref_text = "".join(ref_tokens[ref_start:ref_end])
        hyp_text = "".join(hyp_tokens[hyp_start:hyp_end])
        if tag == "equal":
            parts.append(_escape_diff_text(hyp_text))
        elif tag == "delete":
            stats["delete"] += len(ref_text)
            parts.append(_diff_span("del", "asrpp-diff-delete", ref_text))
            changes.append(
                {
                    "kind": "delete",
                    "reference_text": ref_text,
                    "hypothesis_text": "",
                    "context": _diff_context(ref_tokens, ref_start, ref_end),
                }
            )
        elif tag == "insert":
            stats["insert"] += len(hyp_text)
            parts.append(_diff_span("ins", "asrpp-diff-insert", hyp_text))
            changes.append(
                {
                    "kind": "insert",
                    "reference_text": "",
                    "hypothesis_text": hyp_text,
                    "context": _diff_context(hyp_tokens, hyp_start, hyp_end),
                }
            )
        elif tag == "replace":
            stats["replace"] += max(len(ref_text), len(hyp_text))
            parts.append(_render_replacement(ref_text, hyp_text))
            changes.append(
                {
                    "kind": "replace",
                    "reference_text": ref_text,
                    "hypothesis_text": hyp_text,
                    "context": _diff_context(ref_tokens, ref_start, ref_end),
                }
            )
    return "".join(parts), stats, changes


def _diff_change_summary_html(changes: List[Dict[str, str]], reference_label: str, hypothesis_label: str) -> str:
    if not changes:
        return """
  <div class="asrpp-diff-change-summary" aria-label="Diff change details">
    <h4 class="asrpp-diff-change-heading">Change details</h4>
    <p class="asrpp-diff-change-muted">No deletion, insertion, or replacement was detected.</p>
  </div>
""".rstrip()
    shown = changes[:_DIFF_CHANGE_MAX_ROWS]
    omitted = len(changes) - len(shown)
    rows = "\n".join(_diff_change_item_html(change, reference_label, hypothesis_label) for change in shown)
    omitted_html = (
        f'<p class="asrpp-diff-change-muted">Showing first {_DIFF_CHANGE_MAX_ROWS} of {len(changes)} changes.</p>'
        if omitted > 0
        else ""
    )
    return f"""
  <div class="asrpp-diff-change-summary" aria-label="Diff change details">
    <h4 class="asrpp-diff-change-heading">Change details</h4>
    <ol class="asrpp-diff-change-list">
      {rows}
    </ol>
    {omitted_html}
  </div>
""".rstrip()


def _diff_change_item_html(change: Dict[str, str], reference_label: str, hypothesis_label: str) -> str:
    kind = change.get("kind") or "replace"
    if kind == "delete":
        label = "Deletion"
        explanation = f"Removed from {reference_label}"
    elif kind == "insert":
        label = "Insertion"
        explanation = f"Added in {hypothesis_label}"
    else:
        label = "Replacement"
        explanation = f"Changed {reference_label} text into {hypothesis_label} text"
    reference_text = _compact_diff_snippet(change.get("reference_text") or "")
    hypothesis_text = _compact_diff_snippet(change.get("hypothesis_text") or "")
    context = _compact_diff_snippet(change.get("context") or "")
    return (
        f'<li class="asrpp-diff-change-item {html.escape(kind)}">'
        f'<div class="asrpp-diff-change-kind">{html.escape(label)}<br>'
        f'<span class="asrpp-diff-change-muted">{explanation}</span></div>'
        '<div class="asrpp-diff-change-copy">'
        f'<div><span class="asrpp-diff-change-muted">{reference_label}:</span> <code>{html.escape(reference_text)}</code></div>'
        f'<div><span class="asrpp-diff-change-muted">{hypothesis_label}:</span> <code>{html.escape(hypothesis_text)}</code></div>'
        f'<div class="asrpp-diff-change-context">Context: {html.escape(context)}</div>'
        "</div>"
        "</li>"
    )


def _diff_context(tokens: Sequence[str], start: int, end: int, window: int = 4) -> str:
    context_start = max(0, start - window)
    context_end = min(len(tokens), end + window)
    before = "".join(tokens[context_start:start])
    selected = "".join(tokens[start:end])
    after = "".join(tokens[end:context_end])
    parts = []
    if before:
        parts.append(before)
    if selected:
        parts.append(f"[{selected}]")
    if after:
        parts.append(after)
    return "".join(parts)


def _compact_diff_snippet(text: str, max_chars: int = 180) -> str:
    compacted = re.sub(r"\s+", " ", text or "").strip()
    if not compacted:
        return "(empty)"
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 1].rstrip() + "..."


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
