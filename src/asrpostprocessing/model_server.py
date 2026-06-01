from __future__ import annotations

import os
import re
import site
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse

from .config import ExperimentConfig


StatusCallback = Callable[[str], None]


@dataclass
class ModelServerSpec:
    name: str
    stage: str
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


def ensure_model_servers(
    config: ExperimentConfig,
    status_callback: Optional[StatusCallback] = None,
    names: Optional[Iterable[str]] = None,
) -> List[ModelServerStatus]:
    if not config.auto_start_model_servers:
        return []
    specs = _filter_specs(_server_specs(config), names)
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


def stop_model_servers(
    config: ExperimentConfig,
    status_callback: Optional[StatusCallback] = None,
    names: Optional[Iterable[str]] = None,
) -> List[ModelServerStatus]:
    specs = _filter_specs(_server_specs(config), names)
    if not specs:
        return []
    statuses: List[ModelServerStatus] = []
    pending: List[Tuple[ModelServerSpec, subprocess.Popen]] = []
    with _LOCK:
        for spec in specs:
            process = _PROCESSES.pop(_process_key(spec), None)
            if process is None:
                statuses.append(
                    ModelServerStatus(
                        spec.name,
                        spec.base_url,
                        "not_managed",
                        "no auto-started process is registered in this app process",
                        log_path=spec.log_path,
                    )
                )
            else:
                pending.append((spec, process))
    for spec, process in pending:
        statuses.append(_stop_process(spec, process, float(config.server_shutdown_timeout_s), status_callback))
    return statuses


def _server_specs(config: ExperimentConfig) -> List[ModelServerSpec]:
    specs: List[ModelServerSpec] = []
    log_dir = Path(config.server_log_dir)
    lanes = _pipeline_lanes(config)
    if lanes:
        for lane in lanes:
            lane_name = _safe_name(str(lane.get("name") or "lane"))
            if _uses_external_asr_server(config):
                specs.append(
                    _make_spec(
                        f"asr_{lane_name}",
                        "asr",
                        str(lane.get("asr_model") or config.asr_model),
                        config.asr_backend,
                        str(lane.get("asr_base_url") or config.asr_base_url),
                        str(lane.get("asr_server_host") or config.asr_server_host),
                        str(lane.get("asr_server_gpu") or lane.get("asr_gpu") or config.asr_server_gpu),
                        str(lane.get("asr_server_command") or config.asr_server_command),
                        log_dir,
                    )
                )
            if _uses_external_post_server(config):
                specs.append(
                    _make_spec(
                        f"post_{lane_name}",
                        "post",
                        str(lane.get("post_model") or config.post_model),
                        config.post_backend,
                        str(lane.get("post_base_url") or config.post_base_url),
                        str(lane.get("post_server_host") or config.post_server_host),
                        str(lane.get("post_server_gpu") or lane.get("post_gpu") or config.post_server_gpu),
                        str(lane.get("post_server_command") or config.post_server_command),
                        log_dir,
                    )
                )
        return specs
    if _uses_external_asr_server(config):
        specs.append(
            _make_spec(
                "asr",
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


def _filter_specs(specs: List[ModelServerSpec], names: Optional[Iterable[str]]) -> List[ModelServerSpec]:
    selected = _normalize_names(names)
    if selected is None:
        return specs
    return [spec for spec in specs if spec.name in selected or spec.stage in selected]


def _normalize_names(names: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if names is None:
        return None
    if isinstance(names, str):
        names = [names]
    selected = {str(name).strip().lower() for name in names if str(name).strip()}
    aliases = {"llm": "post", "postprocess": "post", "post_processing": "post", "asr": "asr", "post": "post"}
    return {aliases.get(name, name) for name in selected}


def _uses_external_asr_server(config: ExperimentConfig) -> bool:
    return (config.asr_backend or "").lower() in {"vllm", "vllm_chat", "openai_audio"}


def _uses_external_post_server(config: ExperimentConfig) -> bool:
    return bool(config.enable_llm_postprocess) and (config.post_backend or "").lower() in {"vllm", "vllm_openai", "openai"}


def _make_spec(
    name: str,
    stage: str,
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
        stage=stage,
        model=model,
        backend=backend,
        base_url=base_url,
        host=host,
        port=int(parsed.port),
        gpu=str(gpu),
        command_template=command_template,
        log_path=str(log_dir / f"{name}_vllm.log"),
    )


def _pipeline_lanes(config: ExperimentConfig) -> List[Dict[str, object]]:
    lanes = list(getattr(config, "pipeline_lanes", []) or [])
    if lanes:
        return lanes
    asr_urls = list(getattr(config, "asr_base_urls", []) or [])
    post_urls = list(getattr(config, "post_base_urls", []) or [])
    count = max(len(asr_urls), len(post_urls))
    if count <= 0:
        return []
    generated: List[Dict[str, object]] = []
    for index in range(count):
        lane: Dict[str, object] = {"name": f"lane_{index}"}
        if index < len(asr_urls):
            lane["asr_base_url"] = asr_urls[index]
        if index < len(post_urls):
            lane["post_base_url"] = post_urls[index]
        generated.append(lane)
    return generated


def _safe_name(value: str) -> str:
    normalized = (value or "lane").strip().lower().replace("-", "_")
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in normalized)
    return safe or "lane"


def _prepare_server(spec: ModelServerSpec, status_callback: Optional[StatusCallback]) -> ModelServerStatus | _PendingServer:
    key = _process_key(spec)
    if _endpoint_ready(spec.base_url):
        return ModelServerStatus(spec.name, spec.base_url, "ready", "endpoint already ready", log_path=spec.log_path)

    process = _PROCESSES.get(key)
    if process is not None and process.poll() is None:
        _emit(status_callback, f"{spec.name} model server is starting on {spec.base_url}")
        return _PendingServer(spec, process, "ready", "managed process became ready")

    if _tcp_port_open(spec.base_url):
        raise RuntimeError(
            f"{spec.name} model server port {spec.port} is already open, but {spec.base_url}/models is not an "
            "OpenAI-compatible model endpoint. Change the base URL port in the UI/config or stop the process "
            "that is using the port."
        )

    try:
        process = _start_process(spec)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Cannot auto-start {spec.name} model server because executable was not found: {exc.filename}. "
            "Install the required GPU serving package or set a custom server command in the UI/config."
        ) from exc
    _PROCESSES[key] = process
    _emit(status_callback, f"Started {spec.name} model server pid={process.pid}; waiting for {spec.base_url}")
    return _PendingServer(spec, process, "started", "server started and became ready")


def _process_key(spec: ModelServerSpec) -> str:
    return f"{spec.name}:{spec.base_url}"


def _start_process(spec: ModelServerSpec) -> subprocess.Popen:
    Path(spec.log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(spec.log_path, "ab")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["LD_LIBRARY_PATH"] = _with_nvidia_library_paths(env.get("LD_LIBRARY_PATH", ""))

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
    if spec.name == "asr":
        return [
            sys.executable,
            "-m",
            "asrpostprocessing.qwen_asr_serve_compat",
            spec.model,
            "--host",
            spec.host,
            "--port",
            str(spec.port),
            "--gpu-memory-utilization",
            "0.7",
            "--max-model-len",
            "32768",
            "--attention-backend",
            "TRITON_ATTN",
            "--enforce-eager",
        ]
    return [
        "vllm",
        "serve",
        spec.model,
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--language-model-only",
        "--quantization",
        "bitsandbytes",
        "--load-format",
        "bitsandbytes",
        "--enforce-eager",
        "--attention-backend",
        "TRITON_ATTN",
        "--gpu-memory-utilization",
        "0.6",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
    ]


def _with_nvidia_library_paths(current: str) -> str:
    existing = [part for part in current.split(os.pathsep) if part]
    additions: List[str] = []
    for base in site.getsitepackages():
        package_dir = Path(base) / "nvidia"
        for relative in ("cu13/lib", "cuda_runtime/lib", "nvjitlink/lib"):
            path = package_dir / relative
            if path.is_dir():
                additions.append(str(path))
    merged: List[str] = []
    for path in additions + existing:
        if path not in merged:
            merged.append(path)
    return os.pathsep.join(merged)


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


def _stop_process(
    spec: ModelServerSpec,
    process: subprocess.Popen,
    timeout_s: float,
    status_callback: Optional[StatusCallback],
) -> ModelServerStatus:
    if process.poll() is not None:
        return ModelServerStatus(
            spec.name,
            spec.base_url,
            "exited",
            f"managed process had already exited with returncode={process.returncode}",
            pid=process.pid,
            log_path=spec.log_path,
        )
    _emit(status_callback, f"Stopping {spec.name} model server pid={process.pid}")
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
        return ModelServerStatus(
            spec.name,
            spec.base_url,
            "stopped",
            "managed process terminated to free VRAM",
            pid=process.pid,
            log_path=spec.log_path,
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
        return ModelServerStatus(
            spec.name,
            spec.base_url,
            "killed",
            f"managed process did not exit within {timeout_s:.0f}s and was killed",
            pid=process.pid,
            log_path=spec.log_path,
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
    text = data[-max_bytes:].decode("utf-8", errors="replace").strip()
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _emit(callback: Optional[StatusCallback], message: str) -> None:
    if callback is not None:
        callback(message)
