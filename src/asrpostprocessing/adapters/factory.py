from __future__ import annotations

from asrpostprocessing.config import ExperimentConfig

from .mock import MockASRAdapter, MockPostProcessAdapter
from .qwen_asr import QwenASRPackageAdapter
from .vllm import VLLMChatASRAdapter, VLLMOpenAIPostProcessAdapter


def build_asr_adapter(config: ExperimentConfig):
    backend = (config.asr_backend or "mock").lower()
    if backend == "mock":
        return MockASRAdapter()
    if backend in {"vllm", "vllm_chat", "openai_audio"}:
        return VLLMChatASRAdapter()
    if backend in {"qwen_asr_vllm", "qwen-asr-vllm"}:
        return QwenASRPackageAdapter("vllm")
    if backend in {"qwen_asr_transformers", "qwen-asr-transformers"}:
        return QwenASRPackageAdapter("transformers")
    raise ValueError(f"Unsupported ASR backend: {config.asr_backend}")


def build_postprocess_adapter(config: ExperimentConfig):
    backend = (config.post_backend or "mock").lower()
    if backend == "mock":
        return MockPostProcessAdapter()
    if backend in {"vllm", "vllm_openai", "openai"}:
        return VLLMOpenAIPostProcessAdapter()
    raise ValueError(f"Unsupported post-processing backend: {config.post_backend}")
