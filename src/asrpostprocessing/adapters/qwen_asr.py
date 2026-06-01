from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from asrpostprocessing.config import ExperimentConfig, normalize_asr_chunking_strategy
from asrpostprocessing.schemas import TranscriptResult, TranscriptSegment

from .vllm import (
    ASRAudioChunk,
    _asr_audio_chunks,
    _asr_instruction,
    _clean_asr_transcript_text,
    _filter_asr_language_drift,
    _rolling_asr_context,
)

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


class QwenASRPackageAdapter:
    """Direct `qwen-asr` package backend for CUDA servers.

    The public `qwen-asr` Python API accepts a `context` string rather than a
    decoder hotword knob, so keyword/context bias is passed as bounded prompt
    context instead of being treated as a guaranteed term constraint.
    """

    def __init__(self, backend: str):
        self.backend = backend

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
        model = _get_model(config, self.backend)
        kwargs: Dict[str, Any] = {"audio": audio_path}
        if config.language:
            kwargs["language"] = _qwen_language(config.language)
        context = _asr_instruction(keyword_instruction, previous_context)
        if context:
            kwargs["context"] = context
        results = model.transcribe(**kwargs)
        result = results[0] if isinstance(results, list) else results
        raw_text = getattr(result, "text", "")
        text = _clean_asr_transcript_text(raw_text)
        text, filtered_reason = _filter_asr_language_drift(text, config.language)
        language = getattr(result, "language", None) or config.language
        segments = _segments_from_timestamps(getattr(result, "time_stamps", None))
        return TranscriptResult(
            language=language,
            text=text,
            segments=segments,
            metadata={
                "backend": f"qwen_asr_{self.backend}",
                "used_context": bool(context),
                "raw_result_type": type(result).__name__,
                "cleaned_asr_text": raw_text != text,
                "filtered_asr_text_reason": filtered_reason,
            },
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
                "backend": f"qwen_asr_{self.backend}",
                "chunked": True,
                "chunking_strategy": normalize_asr_chunking_strategy(getattr(config, "asr_chunking_strategy", "silence")),
                "chunk_seconds": float(getattr(config, "asr_chunk_seconds", 120.0) or 120.0),
                "chunk_padding_seconds": float(getattr(config, "asr_chunk_padding_seconds", 0.5) or 0.0),
                "context_chars": context_chars,
                "chunks": chunk_metadata,
            },
        )


def _segments_from_timestamps(time_stamps) -> List[TranscriptSegment]:
    segments: List[TranscriptSegment] = []
    for timestamp in time_stamps or []:
        if isinstance(timestamp, dict):
            segments.append(
                TranscriptSegment(
                    text=str(timestamp.get("text", "")),
                    start_s=_to_float(timestamp.get("start")),
                    end_s=_to_float(timestamp.get("end")),
                    metadata={"raw": timestamp},
                )
            )
    return segments


def _get_model(config: ExperimentConfig, backend: str):
    key = (backend, config.asr_model)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    try:
        import torch  # type: ignore
        from qwen_asr import Qwen3ASRModel  # type: ignore
    except Exception as exc:
        raise RuntimeError("Install `qwen-asr[vllm]` on the CUDA server to use the qwen_asr backend.") from exc
    if backend == "vllm":
        model = Qwen3ASRModel.LLM(
            model=config.asr_model,
            gpu_memory_utilization=0.7,
            max_inference_batch_size=8,
            max_new_tokens=4096,
        )
    else:
        model = Qwen3ASRModel.from_pretrained(
            config.asr_model,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=8,
            max_new_tokens=4096,
        )
    _MODEL_CACHE[key] = model
    return model


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _qwen_language(language: str):
    mapping = {"ko": "Korean", "kr": "Korean", "korean": "Korean", "en": "English", "english": "English"}
    return mapping.get((language or "").lower(), language)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
