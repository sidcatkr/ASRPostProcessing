from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from .config import ExperimentConfig
from .preprocess import ffmpeg_executable


@dataclass
class DoctorCheck:
    name: str
    status: str
    detail: str

    def to_dict(self):
        return asdict(self)


def run_doctor(config: ExperimentConfig, check_endpoints: bool = False) -> List[DoctorCheck]:
    checks = [
        _check_python(),
        _check_package("gradio", required=True),
        _check_package("requests", required=True),
        _check_package("yaml", required=False),
        _check_package("tensorboard", required=False),
        _check_output_dirs(config),
        _check_model_residency(config),
    ]
    if _needs_nvidia(config):
        checks.append(_check_nvidia())
    if config.auto_start_model_servers and _needs_nvidia(config):
        auto_start_asr = (config.asr_backend or "").lower() in {"vllm", "vllm_chat", "openai_audio"}
        auto_start_post = bool(config.enable_llm_postprocess) and (config.post_backend or "").lower() in {
            "vllm",
            "vllm_openai",
            "openai",
        }
        if auto_start_asr:
            checks.append(_check_package("qwen_asr", required=True))
        if auto_start_asr or auto_start_post:
            checks.append(_check_package("vllm", required=True))
        if auto_start_post:
            checks.append(_check_vllm_executable())
    if config.asr_backend.startswith("qwen_asr"):
        checks.append(_check_package("qwen_asr", required=True))
    if config.enable_rag and config.rag_embedding_backend == "faiss":
        checks.append(_check_package("faiss", required=True))
        checks.append(_check_package("sentence_transformers", required=True))
    if _preprocess_enabled(config):
        checks.append(_check_ffmpeg_preprocess_support())
    if check_endpoints:
        checks.append(_check_openai_endpoint("asr_endpoint", config.asr_base_url))
        checks.append(_check_openai_endpoint("post_endpoint", config.post_base_url))
    return checks


def doctor_as_json(checks: List[DoctorCheck]) -> str:
    return json.dumps([check.to_dict() for check in checks], ensure_ascii=False, indent=2)


def has_failures(checks: List[DoctorCheck]) -> bool:
    return any(check.status == "fail" for check in checks)


def _check_python() -> DoctorCheck:
    version = sys.version_info
    ok = version >= (3, 12)
    detail = f"{platform.python_version()} at {sys.executable}"
    return DoctorCheck("python>=3.12", "ok" if ok else "fail", detail)


def _check_package(module: str, required: bool) -> DoctorCheck:
    found = importlib.util.find_spec(module) is not None
    if found:
        return DoctorCheck(f"package:{module}", "ok", "installed")
    return DoctorCheck(f"package:{module}", "fail" if required else "warn", "not installed")


def _check_nvidia() -> DoctorCheck:
    if not shutil.which("nvidia-smi"):
        return DoctorCheck("nvidia-smi", "fail", "not found on PATH")
    try:
        result = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
    except Exception as exc:
        return DoctorCheck("nvidia-smi", "fail", str(exc))
    detail = result.stdout.strip() or "nvidia-smi returned no GPU list"
    return DoctorCheck("nvidia-smi", "ok", detail)


def _check_executable(name: str, missing_detail: str) -> DoctorCheck:
    executable = shutil.which(name)
    if not executable:
        return DoctorCheck(name, "fail", f"not found on PATH; {missing_detail}")
    return DoctorCheck(name, "ok", executable)


def _check_vllm_executable() -> DoctorCheck:
    executable = shutil.which("vllm")
    if executable:
        return DoctorCheck("vllm", "ok", executable)
    sibling = Path(sys.executable).with_name("vllm")
    if sibling.exists():
        return DoctorCheck("vllm", "ok", str(sibling))
    return DoctorCheck("vllm", "fail", "not found on PATH or next to the active Python executable")


def _check_output_dirs(config: ExperimentConfig) -> DoctorCheck:
    try:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.runs_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return DoctorCheck("output_dirs", "fail", str(exc))
    return DoctorCheck("output_dirs", "ok", f"{config.output_dir}, {config.runs_dir}")


def _check_model_residency(config: ExperimentConfig) -> DoctorCheck:
    mode = (config.model_residency or "parallel").lower()
    if mode not in {"parallel", "sequential", "stage_replicas"}:
        return DoctorCheck("model_residency", "fail", f"unsupported mode: {config.model_residency}")
    if mode in {"sequential", "stage_replicas"} and not config.auto_start_model_servers:
        return DoctorCheck(
            "model_residency",
            "warn",
            f"{mode} mode only unloads model servers that this app auto-starts",
        )
    detail = {
        "parallel": "all required servers stay loaded",
        "sequential": "one stage server is loaded at a time",
        "stage_replicas": "all configured GPUs are reused by each model stage",
    }[mode]
    return DoctorCheck("model_residency", "ok", detail)


def _check_ffmpeg_preprocess_support() -> DoctorCheck:
    executable = ffmpeg_executable()
    if executable:
        return DoctorCheck("preprocess:ffmpeg", "ok", executable)
    return DoctorCheck(
        "preprocess:ffmpeg",
        "warn",
        "not found on PATH and imageio-ffmpeg is not installed; 16-bit PCM WAV preprocessing still works",
    )


def _check_openai_endpoint(name: str, base_url: str) -> DoctorCheck:
    try:
        import requests  # type: ignore

        response = requests.get(base_url.rstrip("/") + "/models", timeout=5)
        response.raise_for_status()
    except Exception as exc:
        return DoctorCheck(name, "fail", f"{base_url}: {exc}")
    return DoctorCheck(name, "ok", base_url)


def _needs_nvidia(config: ExperimentConfig) -> bool:
    return config.asr_backend != "mock" or config.post_backend != "mock"


def _preprocess_enabled(config: ExperimentConfig) -> bool:
    legacy_model = (config.preprocess_model or "none").lower()
    return (
        bool(config.enable_noise_reduction)
        or bool(config.enable_volume_normalization)
        or (bool(config.enable_preprocess) and legacy_model != "none")
    )
