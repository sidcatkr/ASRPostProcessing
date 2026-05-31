from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Iterable, List

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.correction_parser import parse_correction_response
from asrpostprocessing.schemas import CorrectionResult, RAGContext, SearchResult, TranscriptResult


class VLLMChatASRAdapter:
    def transcribe(self, audio_path: str, config: ExperimentConfig, keyword_instruction: str = "") -> TranscriptResult:
        payload = {
            "model": config.asr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _asr_instruction(keyword_instruction)},
                        {"type": "audio_url", "audio_url": {"url": _data_url(audio_path)}},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        data = _post_chat(config.asr_base_url, payload, config.request_timeout_s)
        text = _extract_message_text(data)
        parsed = _parse_asr_text(text)
        return TranscriptResult(
            language=parsed.get("language") or config.language,
            text=parsed.get("text") or text,
            metadata={"backend": "vllm_chat", "raw": data, "parsed": parsed},
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
        }
        data = _post_chat(config.post_base_url, payload, config.request_timeout_s)
        response_text = _extract_message_text(data)
        result = parse_correction_response(response_text, chunk_text)
        result.metadata.setdefault("backend", "vllm_openai")
        result.metadata.setdefault("raw", data)
        return result


def _asr_instruction(keyword_instruction: str) -> str:
    base = "Transcribe the Korean conversational audio faithfully. Preserve spoken meaning and do not invent content."
    if keyword_instruction:
        return base + "\n\n" + keyword_instruction
    return base


def _postprocess_prompt(
    chunk_text: str,
    config: ExperimentConfig,
    contexts: List[RAGContext],
    search_results: List[SearchResult],
) -> str:
    context_block = "\n\n".join(f"[{ctx.context_id}] {ctx.text}" for ctx in contexts) or "(none)"
    search_block = "\n".join(f"- {item.title}: {item.snippet} ({item.url})" for item in search_results) or "(none)"
    keywords = ", ".join(config.keywords) or "(none)"
    strength_policy = _postprocess_strength_policy(config.postprocess_strength)
    return f"""
Raw transcript chunk:
{chunk_text}

Keywords:
{keywords}

Retrieved context:
{context_block}

Search results:
{search_block}

Correction strength:
{strength_policy}

Rules:
- Preserve the original meaning.
- Correct only clear ASR errors such as Korean phonetic renderings of technical terms.
- If uncertain, keep the original phrase.
- Do not add facts that are not supported by the audio transcript or context.
- Return compact JSON with keys: corrected_text, edits, risk, used_context_ids.
- Each edit item must include before, after, reason, confidence.
""".strip()


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


def _post_chat(base_url: str, payload: dict, timeout_s: float) -> dict:
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise RuntimeError("The requests package is required for vLLM/OpenAI-compatible backends.") from exc
    url = base_url.rstrip("/") + "/chat/completions"
    response = requests.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


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
    try:
        from qwen_asr import parse_asr_output  # type: ignore

        parsed = parse_asr_output(text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, tuple) and len(parsed) >= 2:
            return {"language": parsed[0], "text": parsed[1]}
        if isinstance(parsed, str):
            return {"text": parsed}
    except Exception:
        pass
    return {"text": text.strip()}


def _data_url(audio_path: str) -> str:
    path = Path(audio_path)
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
