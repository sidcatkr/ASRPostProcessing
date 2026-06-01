from __future__ import annotations

import base64
import json
import mimetypes
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from uuid import uuid4

from asrpostprocessing.config import ExperimentConfig, normalize_asr_chunking_strategy
from asrpostprocessing.correction_parser import parse_correction_response
from asrpostprocessing.keyword_bias import normalize_keywords
from asrpostprocessing.preprocess import ffmpeg_executable
from asrpostprocessing.schemas import CorrectionResult, Edit, RAGContext, SearchResult, TranscriptResult, TranscriptSegment

ASR_CHUNK_OUTPUT_DIR = Path("outputs/asr_chunks")
ASR_CHUNK_THRESHOLD_RATIO = 1.1


@dataclass(frozen=True)
class ASRAudioChunk:
    path: Path
    index: int
    start_s: float
    end_s: Optional[float]
    method: str = "single"
    speech_start_s: Optional[float] = None
    speech_end_s: Optional[float] = None


@dataclass(frozen=True)
class AudioChunkSpec:
    start_s: float
    end_s: float
    method: str = "silence"
    speech_start_s: Optional[float] = None
    speech_end_s: Optional[float] = None


class VLLMChatASRAdapter:
    def transcribe(self, audio_path: str, config: ExperimentConfig, keyword_instruction: str = "") -> TranscriptResult:
        chunks = _asr_audio_chunks(audio_path, config)
        if len(chunks) > 1 or (chunks and chunks[0].path != Path(audio_path)):
            return self._transcribe_chunks(chunks, config, keyword_instruction)
        return self._transcribe_one(audio_path, config, keyword_instruction)

    def _transcribe_one(
        self,
        audio_path: str,
        config: ExperimentConfig,
        keyword_instruction: str = "",
        previous_context: str = "",
    ) -> TranscriptResult:
        payload = {
            "model": config.asr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _asr_instruction(keyword_instruction, previous_context)},
                        {"type": "audio_url", "audio_url": {"url": _data_url(audio_path)}},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        data = _post_chat(config.asr_base_url, payload, _asr_request_timeout_s(config), "ASR")
        text = _extract_message_text(data)
        parsed = _parse_asr_text(text)
        transcript_text = parsed["text"] if "text" in parsed else _clean_asr_transcript_text(text)
        filtered_text, filtered_reason = _filter_asr_language_drift(transcript_text, config.language)
        if filtered_reason:
            parsed["text"] = filtered_text
            parsed["filtered_reason"] = filtered_reason
            transcript_text = filtered_text
        return TranscriptResult(
            language=parsed.get("language") or config.language,
            text=transcript_text,
            metadata={"backend": "vllm_chat", "raw": data, "parsed": parsed},
        )

    def _transcribe_chunks(
        self,
        chunks: List[ASRAudioChunk],
        config: ExperimentConfig,
        keyword_instruction: str,
    ) -> TranscriptResult:
        texts: List[str] = []
        segments: List[TranscriptSegment] = []
        chunk_metadata = []
        language = config.language
        context_chars = max(0, int(getattr(config, "asr_context_chars", 240) or 0))
        for chunk in chunks:
            previous_context = _rolling_asr_context(texts, context_chars)
            result = self._transcribe_one(
                str(chunk.path),
                config,
                keyword_instruction=keyword_instruction,
                previous_context=previous_context,
            )
            text = result.text.strip()
            if text:
                texts.append(text)
            language = result.language or language
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_s=chunk.start_s,
                    end_s=chunk.end_s,
                    metadata={
                        "chunk_index": chunk.index,
                        "audio_path": str(chunk.path),
                        "chunk_method": chunk.method,
                        "speech_start_s": chunk.speech_start_s,
                        "speech_end_s": chunk.speech_end_s,
                        "previous_context_chars": len(previous_context),
                        "asr_metadata": result.metadata,
                    },
                )
            )
            chunk_metadata.append(
                {
                    "index": chunk.index,
                    "audio_path": str(chunk.path),
                    "start_s": chunk.start_s,
                    "end_s": chunk.end_s,
                    "method": chunk.method,
                    "speech_start_s": chunk.speech_start_s,
                    "speech_end_s": chunk.speech_end_s,
                    "previous_context_chars": len(previous_context),
                    "text_chars": len(text),
                }
            )
        return TranscriptResult(
            language=language,
            text="\n".join(texts).strip(),
            segments=segments,
            metadata={
                "backend": "vllm_chat",
                "chunked": True,
                "chunking_strategy": normalize_asr_chunking_strategy(getattr(config, "asr_chunking_strategy", "silence")),
                "chunk_seconds": float(getattr(config, "asr_chunk_seconds", 120.0) or 120.0),
                "chunk_padding_seconds": float(getattr(config, "asr_chunk_padding_seconds", 0.5) or 0.0),
                "context_chars": context_chars,
                "chunks": chunk_metadata,
            },
        )


class VLLMOpenAIPostProcessAdapter:
    def correct(
        self,
        chunk_text: str,
        config: ExperimentConfig,
        contexts: Iterable[RAGContext],
        search_results: Iterable[SearchResult],
    ) -> CorrectionResult:
        contexts = list(contexts)
        search_results = list(search_results)
        prompt = _postprocess_prompt(chunk_text, config, contexts, search_results)
        payload = {
            "model": config.post_model,
            "messages": [
                {"role": "system", "content": "You correct Korean ASR transcripts. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": _temperature_for_strength(config.postprocess_strength),
            "max_tokens": 512,
        }
        if (config.post_backend or "").lower() in {"vllm", "vllm_openai"}:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        data = _post_chat(config.post_base_url, payload, config.request_timeout_s, "post-processing LLM")
        response_text = _extract_message_text(data)
        result = parse_correction_response(response_text, chunk_text)
        result = _apply_keyword_near_miss_corrections(result, config)
        result.metadata.setdefault("backend", "vllm_openai")
        result.metadata.setdefault("raw", data)
        return result


def _asr_instruction(keyword_instruction: str, previous_context: str = "") -> str:
    base = (
        "Transcribe only the Korean speech in the audio faithfully. "
        "Preserve spoken meaning and do not invent content. "
        "Do not translate, switch languages, add language labels, or emit XML-like tags. "
        "If the current audio contains only silence, noise, music, or unintelligible speech, return an empty transcript."
    )
    parts = [base]
    if previous_context:
        parts.append(
            "Previous transcript context for continuity only. "
            "Do not repeat it unless it is spoken again in the current audio:\n"
            f"{previous_context}"
        )
    if keyword_instruction:
        parts.append(keyword_instruction)
    return "\n\n".join(parts)


def _rolling_asr_context(texts: List[str], max_chars: int) -> str:
    if max_chars <= 0 or not texts:
        return ""
    context = " ".join(" ".join(text.split()) for text in texts if text.strip()).strip()
    if len(context) <= max_chars:
        return context
    return context[-max_chars:].lstrip()


def _postprocess_prompt(
    chunk_text: str,
    config: ExperimentConfig,
    contexts: List[RAGContext],
    search_results: List[SearchResult],
) -> str:
    context_block = "\n\n".join(f"[{ctx.context_id}] {ctx.text}" for ctx in contexts) or "(none)"
    search_block = "\n".join(f"- {item.title}: {item.snippet} ({item.url})" for item in search_results) or "(none)"
    keywords = ", ".join(config.keywords) or "(none)"
    keyword_guidance = _keyword_correction_guidance(config.keywords)
    strength_policy = _postprocess_strength_policy(config.postprocess_strength)
    return f"""
Raw transcript chunk:
{chunk_text}

Keywords:
{keywords}

Keyword correction guidance:
{keyword_guidance}

Retrieved context:
{context_block}

Search results:
{search_block}

Correction strength:
{strength_policy}

Rules:
- Do not include reasoning or analysis text.
- Preserve the original meaning.
- Correct only clear ASR errors such as Korean phonetic renderings of technical terms.
- If uncertain, keep the original phrase.
- Do not translate or expand foreign-language fragments, ASR tags, or model artifacts into plausible Korean.
- Do not add facts that are not supported by the audio transcript or context.
- Return compact JSON with keys: corrected_text, edits, risk, used_context_ids.
- Each edit item must include before, after, reason, confidence.
""".strip()


def _keyword_correction_guidance(keywords: List[str]) -> str:
    if not keywords:
        return "(none)"
    return (
        "Treat the keyword list as correction candidates, not content to insert. "
        "When the raw transcript contains a close Korean ASR near-miss for a listed keyword and the surrounding sentence supports it, "
        "prefer the listed keyword. Use only the terms provided in the keyword list and local transcript context; "
        "do not rely on domain-specific examples or fixed replacement pairs. "
        "If the match is not close or the context does not support it, keep the raw phrase."
    )


def _apply_keyword_near_miss_corrections(result: CorrectionResult, config: ExperimentConfig) -> CorrectionResult:
    if float(getattr(config, "postprocess_strength", 0.0) or 0.0) < 0.5:
        return result
    keywords = [keyword for keyword in normalize_keywords(config.keywords) if _hangul_count(keyword) >= 2]
    if not keywords or not result.corrected_text:
        return result
    replacements = _keyword_near_miss_replacements(result.corrected_text, keywords)
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


def _keyword_near_miss_replacements(text: str, keywords: List[str]) -> List[Tuple[int, int, str, str]]:
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
    if len(before_key) < 4 or _hangul_count(before_key) < 2:
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


def _hangul_count(text: str) -> int:
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


def _temperature_for_strength(strength: float) -> float:
    strength = max(0.0, min(1.0, float(strength)))
    return 0.1 + 0.2 * strength


def _postprocess_strength_policy(strength: float) -> str:
    strength = max(0.0, min(1.0, float(strength)))
    if strength < 0.35:
        label = "conservative"
        rule = "Only fix unambiguous ASR errors. Keep wording, spacing, and uncertain terms unchanged."
    elif strength < 0.7:
        label = "balanced"
        rule = "Fix likely domain-term ASR errors when keywords, RAG context, or local context support the edit."
    else:
        label = "aggressive-but-faithful"
        rule = "Use keywords and RAG context actively for phonetic term recovery, but never add unsupported content."
    return f"{label} ({strength:.2f}): {rule}"


def _post_chat(base_url: str, payload: dict, timeout_s: float, service_name: str) -> dict:
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise RuntimeError("The requests package is required for vLLM/OpenAI-compatible backends.") from exc
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        response = requests.post(url, json=payload, timeout=timeout_s)
        try:
            response.raise_for_status()
        except Exception as exc:
            detail = _response_error_detail(response)
            raise RuntimeError(
                f"{service_name} vLLM OpenAI-compatible endpoint request failed at {url}. "
                f"Start the model server or switch the backend to mock for UI testing. "
                f"Original error: {exc}{detail}"
            ) from exc
        return response.json()
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            f"{service_name} vLLM OpenAI-compatible endpoint request failed at {url}. "
            f"Start the model server or switch the backend to mock for UI testing. "
            f"Original error: {exc}. "
            "The request reached the server but did not finish before the client timeout; "
            "for ASR, use silence/fixed asr_chunking_strategy with shorter asr_chunk_seconds "
            "or increase asr_request_timeout_s."
        ) from exc
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"{service_name} vLLM OpenAI-compatible endpoint request failed at {url}. "
            f"Start the model server or switch the backend to mock for UI testing. Original error: {exc}"
        ) from exc


def _response_error_detail(response) -> str:  # type: ignore[no-untyped-def]
    try:
        text = (response.text or "").strip()
    except Exception:
        text = ""
    if not text:
        return ""
    return f". Response body: {text[:1000]}"


def _extract_message_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_asr_text(text: str) -> dict:
    raw_text = (text or "").strip()
    try:
        from qwen_asr import parse_asr_output  # type: ignore

        parsed = parse_asr_output(raw_text)
        if isinstance(parsed, dict):
            return _clean_parsed_asr(parsed, raw_text)
        if isinstance(parsed, tuple) and len(parsed) >= 2:
            return _clean_parsed_asr({"language": parsed[0], "text": parsed[1]}, raw_text)
        if isinstance(parsed, str):
            return {"text": _clean_asr_transcript_text(parsed)}
    except Exception:
        pass
    return {"text": _clean_asr_transcript_text(raw_text)}


def _clean_parsed_asr(parsed: dict, raw_text: str) -> dict:
    cleaned = dict(parsed)
    raw_cleaned_text = _clean_asr_transcript_text(raw_text)
    language = str(cleaned.get("language") or "").strip()
    if "text" in cleaned:
        parsed_text = _clean_asr_transcript_text(str(cleaned.get("text") or ""))
    else:
        parsed_text = raw_cleaned_text
    if language.lower() == "none":
        cleaned["language"] = ""
    if _has_transcript_before_asr_marker(raw_text) and raw_cleaned_text and raw_cleaned_text != parsed_text:
        cleaned["text"] = raw_cleaned_text
        cleaned["raw_marker_mix_preserved"] = True
        return cleaned
    cleaned["text"] = parsed_text
    return cleaned


def _has_transcript_before_asr_marker(raw_text: str) -> bool:
    match = re.search(
        r"(?:^|\s)language\s+(?:none|korean|english|chinese|[a-z_-]+)?\s*<\s*asr_text\s*>|<\s*asr_text\s*>",
        raw_text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    prefix = _clean_asr_transcript_text((raw_text or "")[: match.start()])
    return bool(_HANGUL_RE.search(prefix))


def _clean_asr_transcript_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"(?:^|\s)language\s+(?:none|korean|english|chinese|[a-z_-]+)?\s*<\s*asr_text\s*>\s*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"</?\s*asr_text\s*>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"(?:^|\s)language\s+(?:none|korean|english|chinese|[a-z_-]+)(?=\s|$)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[ \t\r\f\v]+", " ", cleaned).strip()


def _filter_asr_language_drift(text: str, language: str) -> Tuple[str, str]:
    if not _is_korean_target_language(language):
        return text, ""
    original_compact = "".join((text or "").split())
    original_hangul_count = len(_HANGUL_RE.findall(original_compact))
    original_han_count = len(_HAN_RE.findall(original_compact))
    if original_hangul_count == 0 and original_han_count >= 4:
        return "", "non_korean_cjk_drift"
    filtered_text, removed_inline_cjk = _remove_inline_cjk_drift(text)
    compact = "".join((filtered_text or "").split())
    if not compact:
        return filtered_text, "non_korean_cjk_drift" if removed_inline_cjk else ""
    hangul_count = len(_HANGUL_RE.findall(compact))
    han_count = len(_HAN_RE.findall(compact))
    if hangul_count == 0 and han_count >= 4:
        return "", "non_korean_cjk_drift"
    if removed_inline_cjk:
        return filtered_text, "inline_cjk_drift_removed"
    return text, ""


_HANGUL_RE = re.compile(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_DRIFT_RUN_RE = re.compile(r"[\u4e00-\u9fff][\u4e00-\u9fff\u3000-\u303f\uff00-\uffef·，。！？、：；“”‘’《》「」『』（）()…,.?!\s-]*")
_CJK_DRIFT_PUNCT_RE = re.compile(r"[\u3000-\u303f\uff00-\uffef，。！？、：；]")


def _remove_inline_cjk_drift(text: str) -> Tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        block = match.group(0)
        han_count = len(_HAN_RE.findall(block))
        has_cjk_punctuation = bool(_CJK_DRIFT_PUNCT_RE.search(block))
        if han_count >= 4 and (has_cjk_punctuation or han_count >= 6):
            changed = True
            return " "
        return block

    cleaned = _CJK_DRIFT_RUN_RE.sub(replace, text or "")
    if changed:
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        cleaned = re.sub(r"\n\s+", "\n", cleaned)
    return cleaned.strip(), changed


def _is_korean_target_language(language: str) -> bool:
    return (language or "").strip().lower() in {"ko", "kr", "kor", "korean", "한국어"}


def _data_url(audio_path: str) -> str:
    path = Path(audio_path)
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _asr_request_timeout_s(config: ExperimentConfig) -> float:
    return max(30.0, float(getattr(config, "asr_request_timeout_s", config.request_timeout_s) or config.request_timeout_s))


def _asr_audio_chunks(audio_path: str, config: ExperimentConfig) -> List[ASRAudioChunk]:
    path = Path(audio_path)
    duration = _audio_duration_seconds(path)
    chunk_seconds = max(5.0, float(getattr(config, "asr_chunk_seconds", 120.0) or 120.0))
    strategy = normalize_asr_chunking_strategy(getattr(config, "asr_chunking_strategy", "silence"))
    if strategy == "none":
        return [ASRAudioChunk(path=path, index=0, start_s=0.0, end_s=duration, method="none")]
    if duration is None or duration <= chunk_seconds * ASR_CHUNK_THRESHOLD_RATIO:
        return [ASRAudioChunk(path=path, index=0, start_s=0.0, end_s=duration)]
    if strategy == "silence":
        chunks = _split_audio_for_asr_silence(path, config, duration, chunk_seconds)
        if chunks:
            return chunks
    chunks = _split_audio_for_asr(path, chunk_seconds, duration)
    if chunks:
        return chunks
    raise RuntimeError(
        f"ASR input is {duration:.1f}s, which is too long for one vLLM ASR request, "
        "but ffmpeg could not create audio chunks."
    )


def _split_audio_for_asr_silence(
    input_path: Path,
    config: ExperimentConfig,
    duration: float,
    chunk_seconds: float,
) -> List[ASRAudioChunk]:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return []
    threshold_db = max(-80.0, min(-10.0, float(getattr(config, "asr_silence_threshold_db", -35.0) or -35.0)))
    min_silence_s = max(0.1, min(5.0, float(getattr(config, "asr_min_silence_seconds", 0.6) or 0.6)))
    padding_s = max(0.0, min(5.0, float(getattr(config, "asr_chunk_padding_seconds", 0.5) or 0.0)))
    silences = _detect_silences(ffmpeg, input_path, threshold_db, min_silence_s, duration)
    specs = _silence_aware_chunk_specs(silences, duration, chunk_seconds, padding_s)
    if not specs:
        return []
    return _write_audio_chunk_specs(ffmpeg, input_path, specs, "silence")


def _detect_silences(
    ffmpeg: str,
    input_path: Path,
    threshold_db: float,
    min_silence_s: float,
    duration: float,
) -> List[Tuple[float, float]]:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(input_path),
        "-af",
        f"silencedetect=noise={threshold_db:.1f}dB:d={min_silence_s:.3f}",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(30.0, min(600.0, duration * 0.25 + 30.0)),
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    output = (result.stderr or "") + "\n" + (result.stdout or "")
    silences: List[Tuple[float, float]] = []
    pending_start: Optional[float] = None
    for match in re.finditer(r"silence_(start|end):\s*([0-9]+(?:\.[0-9]+)?)", output):
        value = float(match.group(2))
        if match.group(1) == "start":
            pending_start = value
            continue
        start = pending_start if pending_start is not None else max(0.0, value - min_silence_s)
        if value > start:
            silences.append((start, value))
        pending_start = None
    if pending_start is not None and duration > pending_start:
        silences.append((pending_start, duration))
    return _merge_silence_intervals(silences, duration)


def _merge_silence_intervals(silences: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    normalized: List[Tuple[float, float]] = []
    for start_s, end_s in silences:
        start_s = max(0.0, min(duration, start_s))
        end_s = max(0.0, min(duration, end_s))
        if end_s > start_s:
            normalized.append((start_s, end_s))
    if not normalized:
        return []
    normalized.sort()
    merged = [normalized[0]]
    for start_s, end_s in normalized[1:]:
        previous_start, previous_end = merged[-1]
        if start_s <= previous_end + 0.05:
            merged[-1] = (previous_start, max(previous_end, end_s))
        else:
            merged.append((start_s, end_s))
    return merged


def _silence_aware_chunk_specs(
    silences: List[Tuple[float, float]],
    duration: float,
    chunk_seconds: float,
    padding_seconds: float,
) -> List[AudioChunkSpec]:
    if not silences:
        return []
    speech_intervals = _silences_to_speech_intervals(silences, duration)
    if not speech_intervals:
        return []
    specs = _merge_speech_intervals(speech_intervals, chunk_seconds)
    return _apply_chunk_padding(specs, duration, padding_seconds)


def _silences_to_speech_intervals(silences: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    cursor = 0.0
    for silence_start, silence_end in silences:
        if silence_start - cursor > 0.05:
            intervals.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if duration - cursor > 0.05:
        intervals.append((cursor, duration))
    return intervals


def _merge_speech_intervals(intervals: List[Tuple[float, float]], chunk_seconds: float) -> List[AudioChunkSpec]:
    specs: List[AudioChunkSpec] = []
    pending: Optional[AudioChunkSpec] = None

    def flush_pending() -> None:
        nonlocal pending
        if pending is not None:
            specs.append(pending)
            pending = None

    for start_s, end_s in intervals:
        if end_s <= start_s:
            continue
        if end_s - start_s > chunk_seconds:
            flush_pending()
            specs.extend(_split_long_speech_interval(start_s, end_s, chunk_seconds))
            continue
        if pending is None:
            pending = AudioChunkSpec(
                start_s=start_s,
                end_s=end_s,
                method="silence",
                speech_start_s=start_s,
                speech_end_s=end_s,
            )
            continue
        if end_s - pending.start_s <= chunk_seconds:
            pending = AudioChunkSpec(
                start_s=pending.start_s,
                end_s=end_s,
                method="silence",
                speech_start_s=pending.speech_start_s,
                speech_end_s=end_s,
            )
        else:
            flush_pending()
            pending = AudioChunkSpec(
                start_s=start_s,
                end_s=end_s,
                method="silence",
                speech_start_s=start_s,
                speech_end_s=end_s,
            )
    flush_pending()
    return specs


def _split_long_speech_interval(start_s: float, end_s: float, chunk_seconds: float) -> List[AudioChunkSpec]:
    specs: List[AudioChunkSpec] = []
    cursor = start_s
    while cursor < end_s:
        next_end = min(end_s, cursor + chunk_seconds)
        if next_end - cursor > 0.05:
            specs.append(
                AudioChunkSpec(
                    start_s=cursor,
                    end_s=next_end,
                    method="silence_long_speech",
                    speech_start_s=cursor,
                    speech_end_s=next_end,
                )
            )
        cursor = next_end
    return specs


def _apply_chunk_padding(
    specs: List[AudioChunkSpec],
    duration: float,
    padding_seconds: float,
) -> List[AudioChunkSpec]:
    padding_seconds = max(0.0, padding_seconds)
    padded: List[AudioChunkSpec] = []
    for spec in specs:
        start_s = max(0.0, spec.start_s - padding_seconds)
        end_s = min(duration, spec.end_s + padding_seconds)
        if padded and start_s < padded[-1].end_s:
            previous = padded[-1]
            previous_boundary = previous.speech_end_s if previous.speech_end_s is not None else previous.end_s
            current_boundary = spec.speech_start_s if spec.speech_start_s is not None else spec.start_s
            boundary = (previous_boundary + current_boundary) / 2.0
            boundary = max(previous.start_s, min(previous.end_s, boundary))
            padded[-1] = AudioChunkSpec(
                start_s=previous.start_s,
                end_s=boundary,
                method=previous.method,
                speech_start_s=previous.speech_start_s,
                speech_end_s=previous.speech_end_s,
            )
            start_s = max(start_s, boundary)
        if end_s - start_s > 0.05:
            padded.append(
                AudioChunkSpec(
                    start_s=start_s,
                    end_s=end_s,
                    method=spec.method,
                    speech_start_s=spec.speech_start_s,
                    speech_end_s=spec.speech_end_s,
                )
            )
    return padded


def _write_audio_chunk_specs(
    ffmpeg: str,
    input_path: Path,
    specs: List[AudioChunkSpec],
    method_prefix: str,
) -> List[ASRAudioChunk]:
    output_dir = ASR_CHUNK_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_audio_stem(input_path)
    run_id = uuid4().hex[:10]
    chunks: List[ASRAudioChunk] = []
    for output_index, spec in enumerate(specs):
        output_path = output_dir / f"{safe_stem}.asrchunk.{method_prefix}.{run_id}.{output_index:05d}.wav"
        if not _write_audio_chunk(ffmpeg, input_path, output_path, spec.start_s, spec.end_s):
            return []
        chunks.append(
            ASRAudioChunk(
                path=output_path,
                index=len(chunks),
                start_s=spec.start_s,
                end_s=spec.end_s,
                method=spec.method,
                speech_start_s=spec.speech_start_s,
                speech_end_s=spec.speech_end_s,
            )
        )
    return chunks


def _write_audio_chunk(ffmpeg: str, input_path: Path, output_path: Path, start_s: float, end_s: float) -> bool:
    duration = max(0.05, end_s - start_s)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _split_audio_for_asr(input_path: Path, chunk_seconds: float, duration: Optional[float]) -> List[ASRAudioChunk]:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return []
    output_dir = ASR_CHUNK_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_audio_stem(input_path)
    run_id = uuid4().hex[:10]
    output_pattern = output_dir / f"{safe_stem}.asrchunk.{run_id}.%05d.wav"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-acodec",
        "pcm_s16le",
        "-f",
        "segment",
        "-segment_time",
        f"{chunk_seconds:.3f}",
        "-reset_timestamps",
        "1",
        str(output_pattern),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return []
    paths = sorted(output_dir.glob(f"{safe_stem}.asrchunk.{run_id}.*.wav"))
    chunks: List[ASRAudioChunk] = []
    for index, path in enumerate(paths):
        start_s = index * chunk_seconds
        end_s = min(start_s + chunk_seconds, duration) if duration is not None else None
        chunks.append(ASRAudioChunk(path=path, index=index, start_s=start_s, end_s=end_s, method="fixed"))
    return chunks


def _audio_duration_seconds(path: Path) -> Optional[float]:
    if path.suffix.lower() == ".wav":
        try:
            import wave

            with wave.open(str(path), "rb") as handle:
                frame_rate = handle.getframerate()
                if frame_rate:
                    return handle.getnframes() / float(frame_rate)
        except Exception:
            pass
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return None
    return _duration_from_ffmpeg_output((result.stderr or "") + "\n" + (result.stdout or ""))


def _duration_from_ffmpeg_output(output: str) -> Optional[float]:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return (hours * 3600.0) + (minutes * 60.0) + seconds


def _safe_audio_stem(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._")
    return stem or "audio"
