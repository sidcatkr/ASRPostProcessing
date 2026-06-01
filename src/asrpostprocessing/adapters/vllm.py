from __future__ import annotations

import base64
import json
import mimetypes
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional
from uuid import uuid4

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.correction_parser import parse_correction_response
from asrpostprocessing.preprocess import ffmpeg_executable
from asrpostprocessing.schemas import CorrectionResult, RAGContext, SearchResult, TranscriptResult, TranscriptSegment

ASR_CHUNK_OUTPUT_DIR = Path("outputs/asr_chunks")
ASR_CHUNK_THRESHOLD_RATIO = 1.1


class ASRAudioChunk(NamedTuple):
    path: Path
    index: int
    start_s: float
    end_s: Optional[float]


class VLLMChatASRAdapter:
    def transcribe(self, audio_path: str, config: ExperimentConfig, keyword_instruction: str = "") -> TranscriptResult:
        chunks = _asr_audio_chunks(audio_path, config)
        if len(chunks) > 1:
            return self._transcribe_chunks(chunks, config, keyword_instruction)
        return self._transcribe_one(audio_path, config, keyword_instruction)

    def _transcribe_one(self, audio_path: str, config: ExperimentConfig, keyword_instruction: str = "") -> TranscriptResult:
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
        data = _post_chat(config.asr_base_url, payload, _asr_request_timeout_s(config), "ASR")
        text = _extract_message_text(data)
        parsed = _parse_asr_text(text)
        return TranscriptResult(
            language=parsed.get("language") or config.language,
            text=parsed.get("text") or text,
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
        for chunk in chunks:
            result = self._transcribe_one(str(chunk.path), config, keyword_instruction=keyword_instruction)
            text = result.text.strip()
            if text:
                texts.append(text)
            language = result.language or language
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_s=chunk.start_s,
                    end_s=chunk.end_s,
                    metadata={"chunk_index": chunk.index, "audio_path": str(chunk.path), "asr_metadata": result.metadata},
                )
            )
            chunk_metadata.append(
                {
                    "index": chunk.index,
                    "audio_path": str(chunk.path),
                    "start_s": chunk.start_s,
                    "end_s": chunk.end_s,
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
                "chunk_seconds": float(getattr(config, "asr_chunk_seconds", 15.0) or 15.0),
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
- Do not include reasoning or analysis text.
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
            "for ASR, use shorter asr_chunk_seconds or increase asr_request_timeout_s."
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


def _asr_request_timeout_s(config: ExperimentConfig) -> float:
    return max(30.0, float(getattr(config, "asr_request_timeout_s", config.request_timeout_s) or config.request_timeout_s))


def _asr_audio_chunks(audio_path: str, config: ExperimentConfig) -> List[ASRAudioChunk]:
    path = Path(audio_path)
    duration = _audio_duration_seconds(path)
    chunk_seconds = max(5.0, float(getattr(config, "asr_chunk_seconds", 15.0) or 15.0))
    if duration is None or duration <= chunk_seconds * ASR_CHUNK_THRESHOLD_RATIO:
        return [ASRAudioChunk(path=path, index=0, start_s=0.0, end_s=duration)]
    chunks = _split_audio_for_asr(path, chunk_seconds, duration)
    if chunks:
        return chunks
    raise RuntimeError(
        f"ASR input is {duration:.1f}s, which is too long for one vLLM ASR request, "
        "but ffmpeg could not create audio chunks."
    )


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
        chunks.append(ASRAudioChunk(path=path, index=index, start_s=start_s, end_s=end_s))
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
