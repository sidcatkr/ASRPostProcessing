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
    ]
    if _needs_nvidia(config):
        checks.append(_check_nvidia())
    if config.asr_backend.startswith("qwen_asr"):
        checks.append(_check_package("qwen_asr", required=True))
    if config.enable_rag and config.rag_embedding_backend == "faiss":
        checks.append(_check_package("faiss", required=True))
        checks.append(_check_package("sentence_transformers", required=True))
    if config.enable_preprocess and config.preprocess_model.lower() == "rnnoise":
        checks.append(_check_external_preprocess("rnnoise", config.rnnoise_command, "ASRPP_RNNOISE_COMMAND"))
    if config.enable_preprocess and config.preprocess_model.lower() in {"bs-roformer", "bs_roformer", "bsroformer"}:
        checks.append(_check_external_preprocess("bs_roformer", config.bs_roformer_command, "ASRPP_BS_ROFORMER_COMMAND"))
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


def _check_output_dirs(config: ExperimentConfig) -> DoctorCheck:
    try:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.runs_dir).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return DoctorCheck("output_dirs", "fail", str(exc))
    return DoctorCheck("output_dirs", "ok", f"{config.output_dir}, {config.runs_dir}")


def _check_external_preprocess(name: str, command: str, env_name: str) -> DoctorCheck:
    if command:
        executable = command.split()[0]
        if "{" in executable:
            return DoctorCheck(f"preprocess:{name}", "ok", "command template configured")
        return DoctorCheck(f"preprocess:{name}", "ok" if shutil.which(executable) else "warn", command)
    return DoctorCheck(f"preprocess:{name}", "warn", f"set config command or {env_name} before using this preprocessor")


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
