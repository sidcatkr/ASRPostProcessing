from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from .config import ExperimentConfig


StatusCallback = Callable[[str], None]


@dataclass
class ModelServerSpec:
    name: str
    model: str
    backend: str
    base_url: str
    host: str
    port: int
    gpu: str
    command_template: str
    log_path: str


@dataclass
class ModelServerStatus:
    name: str
    base_url: str
    status: str
    detail: str
    pid: Optional[int] = None
    log_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _PendingServer:
    spec: ModelServerSpec
    process: Optional[subprocess.Popen]
    final_status: str
    detail: str


_LOCK = threading.Lock()
_PROCESSES: Dict[str, subprocess.Popen] = {}


def ensure_model_servers(config: ExperimentConfig, status_callback: Optional[StatusCallback] = None) -> List[ModelServerStatus]:
    if not config.auto_start_model_servers:
        return []
    specs = _server_specs(config)
    if not specs:
        return []
    statuses: List[ModelServerStatus] = []
    pending: List[_PendingServer] = []
    with _LOCK:
        for spec in specs:
            status = _prepare_server(spec, status_callback)
            if isinstance(status, ModelServerStatus):
                statuses.append(status)
            else:
                pending.append(status)
    for item in pending:
        _wait_until_ready(item.spec, float(config.server_start_timeout_s), item.process)
        statuses.append(
            ModelServerStatus(
                item.spec.name,
                item.spec.base_url,
                item.final_status,
                item.detail,
                pid=item.process.pid if item.process is not None else None,
                log_path=item.spec.log_path,
            )
        )
    return statuses


def _server_specs(config: ExperimentConfig) -> List[ModelServerSpec]:
    specs: List[ModelServerSpec] = []
    log_dir = Path(config.server_log_dir)
    if _uses_external_asr_server(config):
        specs.append(
            _make_spec(
                "asr",
                config.asr_model,
                config.asr_backend,
                config.asr_base_url,
                config.asr_server_host,
                config.asr_server_gpu,
                config.asr_server_command,
                log_dir,
            )
        )
    if _uses_external_post_server(config):
        specs.append(
            _make_spec(
                "post",
                config.post_model,
                config.post_backend,
                config.post_base_url,
                config.post_server_host,
                config.post_server_gpu,
                config.post_server_command,
                log_dir,
            )
        )
    return specs


def _uses_external_asr_server(config: ExperimentConfig) -> bool:
    return (config.asr_backend or "").lower() in {"vllm", "vllm_chat", "openai_audio"}


def _uses_external_post_server(config: ExperimentConfig) -> bool:
    return bool(config.enable_llm_postprocess) and (config.post_backend or "").lower() in {"vllm", "vllm_openai", "openai"}


def _make_spec(
    name: str,
    model: str,
    backend: str,
    base_url: str,
    host: str,
    gpu: str,
    command_template: str,
    log_dir: Path,
) -> ModelServerSpec:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.hostname or not parsed.port:
        raise ValueError(f"{name} base URL must include scheme, host, and port: {base_url}")
    return ModelServerSpec(
        name=name,
        model=model,
        backend=backend,
        base_url=base_url,
        host=host,
        port=int(parsed.port),
        gpu=str(gpu),
        command_template=command_template,
        log_path=str(log_dir / f"{name}_vllm.log"),
    )


def _prepare_server(spec: ModelServerSpec, status_callback: Optional[StatusCallback]) -> ModelServerStatus | _PendingServer:
    key = f"{spec.name}:{spec.base_url}"
    if _endpoint_ready(spec.base_url):
        return ModelServerStatus(spec.name, spec.base_url, "ready", "endpoint already ready", log_path=spec.log_path)

    process = _PROCESSES.get(key)
    if process is not None and process.poll() is None:
        _emit(status_callback, f"{spec.name} model server is starting on {spec.base_url}")
        return _PendingServer(spec, process, "ready", "managed process became ready")

    if _tcp_port_open(spec.base_url):
        _emit(status_callback, f"{spec.name} port is open; waiting for OpenAI-compatible readiness at {spec.base_url}")
        return _PendingServer(spec, None, "ready", "existing process became ready")

    try:
        process = _start_process(spec)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Cannot auto-start {spec.name} model server because executable was not found: {exc.filename}. "
            "Install vLLM on the GPU server or set a custom server command in the UI/config."
        ) from exc
    _PROCESSES[key] = process
    _emit(status_callback, f"Started {spec.name} model server pid={process.pid}; waiting for {spec.base_url}")
    return _PendingServer(spec, process, "started", "server started and became ready")


def _start_process(spec: ModelServerSpec) -> subprocess.Popen:
    Path(spec.log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(spec.log_path, "ab")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        if spec.command_template:
            command = spec.command_template.format(
                model=spec.model,
                host=spec.host,
                port=spec.port,
                base_url=spec.base_url,
                gpu=spec.gpu,
                log_path=spec.log_path,
            )
            return subprocess.Popen(
                command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        return subprocess.Popen(
            _default_command(spec),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_file.close()


def _default_command(spec: ModelServerSpec) -> List[str]:
    command = ["vllm", "serve", spec.model, "--host", spec.host, "--port", str(spec.port)]
    if spec.name == "post":
        command.extend(
            [
                "--tensor-parallel-size",
                "1",
                "--max-model-len",
                "262144",
                "--reasoning-parser",
                "qwen3",
                "--language-model-only",
            ]
        )
    return command


def _wait_until_ready(spec: ModelServerSpec, timeout_s: float, process: Optional[subprocess.Popen]) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _endpoint_ready(spec.base_url):
            return
        if process is not None and process.poll() is not None:
            tail = _tail_log(spec.log_path)
            raise RuntimeError(
                f"{spec.name} model server exited before becoming ready. "
                f"pid={process.pid}, returncode={process.returncode}, log={spec.log_path}\n{tail}"
            )
        time.sleep(3.0)
    raise RuntimeError(
        f"{spec.name} model server did not become ready within {timeout_s:.0f}s at {spec.base_url}. "
        f"Check log: {spec.log_path}"
    )


def _endpoint_ready(base_url: str) -> bool:
    try:
        import requests  # type: ignore

        response = requests.get(base_url.rstrip("/") + "/models", timeout=3)
        return 200 <= response.status_code < 300
    except Exception:
        return False


def _tcp_port_open(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if not parsed.hostname or not parsed.port:
        return False
    host = "127.0.0.1" if parsed.hostname in {"0.0.0.0", "::"} else parsed.hostname
    try:
        with socket.create_connection((host, int(parsed.port)), timeout=1):
            return True
    except OSError:
        return False


def _tail_log(path: str, max_bytes: int = 4096) -> str:
    log_path = Path(path)
    if not log_path.exists():
        return "(no log file written)"
    data = log_path.read_bytes()
    return data[-max_bytes:].decode("utf-8", errors="replace").strip()


def _emit(callback: Optional[StatusCallback], message: str) -> None:
    if callback is not None:
        callback(message)
