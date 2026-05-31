from __future__ import annotations

from typing import Any, Dict, Tuple

from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.schemas import TranscriptResult, TranscriptSegment

_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


class QwenASRPackageAdapter:
    """Direct `qwen-asr` package backend for CUDA servers.

    Keyword Bias is not passed here because the public `qwen-asr` Python API
    exposes audio/language/timestamp controls, not a documented hotword decoder
    knob. Use `vllm_chat` when ASR prompt/context bias is part of an experiment.
    """

    def __init__(self, backend: str):
        self.backend = backend

    def transcribe(self, audio_path: str, config: ExperimentConfig, keyword_instruction: str = "") -> TranscriptResult:
        model = _get_model(config, self.backend)
        kwargs: Dict[str, Any] = {"audio": audio_path}
        if config.language:
            kwargs["language"] = _qwen_language(config.language)
        results = model.transcribe(**kwargs)
        result = results[0] if isinstance(results, list) else results
        text = getattr(result, "text", "")
        language = getattr(result, "language", None) or config.language
        segments = []
        for timestamp in getattr(result, "time_stamps", None) or []:
            if isinstance(timestamp, dict):
                segments.append(
                    TranscriptSegment(
                        text=str(timestamp.get("text", "")),
                        start_s=_to_float(timestamp.get("start")),
                        end_s=_to_float(timestamp.get("end")),
                        metadata={"raw": timestamp},
                    )
                )
        return TranscriptResult(
            language=language,
            text=text,
            segments=segments,
            metadata={"backend": f"qwen_asr_{self.backend}", "keyword_instruction_ignored": bool(keyword_instruction)},
        )


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


def _qwen_language(language: str):
    mapping = {"ko": "Korean", "kr": "Korean", "korean": "Korean", "en": "English", "english": "English"}
    return mapping.get((language or "").lower(), language)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
