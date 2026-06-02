from __future__ import annotations

import os
import re
import signal
import shlex
import shutil
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
    gpu_memory_utilization: str = "auto"
    gpu_memory_utilization_max: float = 0.90
    gpu_memory_reserved_mb: int = 256


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
    if _stage_replicas(config):
        for index, item in enumerate(_stage_server_pairs(config)):
            base_url, gpu = item
            if _uses_external_asr_server(config):
                specs.append(
                    _make_spec(
                        f"asr_stage_{index}",
                        "asr",
                        config.asr_model,
                        config.asr_backend,
                        base_url,
                        config.asr_server_host,
                        gpu,
                        config.asr_server_command,
                        log_dir,
                        config.server_gpu_memory_utilization,
                        config.server_gpu_memory_utilization_max,
                        config.server_gpu_memory_reserved_mb,
                    )
                )
            if _uses_external_post_server(config):
                specs.append(
                    _make_spec(
                        f"post_stage_{index}",
                        "post",
                        config.post_model,
                        config.post_backend,
                        base_url,
                        config.post_server_host,
                        gpu,
                        config.post_server_command,
                        log_dir,
                        config.server_gpu_memory_utilization,
                        config.server_gpu_memory_utilization_max,
                        config.server_gpu_memory_reserved_mb,
                    )
                )
        return specs
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
                        str(lane.get("server_gpu_memory_utilization") or config.server_gpu_memory_utilization),
                        float(lane.get("server_gpu_memory_utilization_max") or config.server_gpu_memory_utilization_max),
                        int(lane.get("server_gpu_memory_reserved_mb") or config.server_gpu_memory_reserved_mb),
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
                        str(lane.get("server_gpu_memory_utilization") or config.server_gpu_memory_utilization),
                        float(lane.get("server_gpu_memory_utilization_max") or config.server_gpu_memory_utilization_max),
                        int(lane.get("server_gpu_memory_reserved_mb") or config.server_gpu_memory_reserved_mb),
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
                config.server_gpu_memory_utilization,
                config.server_gpu_memory_utilization_max,
                config.server_gpu_memory_reserved_mb,
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
                config.server_gpu_memory_utilization,
                config.server_gpu_memory_utilization_max,
                config.server_gpu_memory_reserved_mb,
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
    gpu_memory_utilization: str,
    gpu_memory_utilization_max: float,
    gpu_memory_reserved_mb: int,
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
        gpu_memory_utilization=str(gpu_memory_utilization or "auto"),
        gpu_memory_utilization_max=float(gpu_memory_utilization_max),
        gpu_memory_reserved_mb=int(gpu_memory_reserved_mb),
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


def _stage_replicas(config: ExperimentConfig) -> bool:
    return str(getattr(config, "model_residency", "") or "").strip().lower() == "stage_replicas"


def _stage_server_pairs(config: ExperimentConfig) -> List[Tuple[str, str]]:
    base_urls = [str(url).strip() for url in (getattr(config, "stage_server_base_urls", []) or []) if str(url).strip()]
    gpus = [str(gpu).strip() for gpu in (getattr(config, "stage_server_gpus", []) or []) if str(gpu).strip()]
    count = min(len(base_urls), len(gpus))
    return [(base_urls[index], gpus[index]) for index in range(count)]


def _safe_name(value: str) -> str:
    normalized = (value or "lane").strip().lower().replace("-", "_")
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in normalized)
    return safe or "lane"


def _prepare_server(spec: ModelServerSpec, status_callback: Optional[StatusCallback]) -> ModelServerStatus | _PendingServer:
    key = _process_key(spec)
    if _endpoint_ready(spec.base_url, spec.model):
        return ModelServerStatus(spec.name, spec.base_url, "ready", "endpoint already ready", log_path=spec.log_path)

    process = _PROCESSES.get(key)
    if process is not None and process.poll() is None:
        _emit(status_callback, f"{spec.name} model server is starting on {spec.base_url}")
        return _PendingServer(spec, process, "ready", "managed process became ready")

    if _tcp_port_open(spec.base_url):
        detail = _open_port_detail(spec)
        raise RuntimeError(
            f"{spec.name} model server port {spec.port} is already open, but {spec.base_url}/models is not serving "
            f"the expected model {spec.model!r}. {detail} Change the base URL port in the UI/config or stop "
            "the process that is using the port."
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
    env["VLLM_CACHE_ROOT"] = _vllm_cache_root(spec)
    gpu_memory_utilization = _gpu_memory_utilization_for_spec(spec)

    try:
        if spec.command_template:
            command = spec.command_template.format(
                model=spec.model,
                host=spec.host,
                port=spec.port,
                base_url=spec.base_url,
                gpu=spec.gpu,
                log_path=spec.log_path,
                gpu_memory_utilization=gpu_memory_utilization,
                vllm_cache_root=env["VLLM_CACHE_ROOT"],
                python=shlex.quote(sys.executable),
                python_dir=shlex.quote(str(Path(sys.executable).parent)),
                vllm=shlex.quote(_vllm_executable()),
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
    gpu_memory_utilization = _gpu_memory_utilization_for_spec(spec)
    if spec.stage == "asr":
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
            gpu_memory_utilization,
            "--max-model-len",
            "65536",
            "--attention-backend",
            "TRITON_ATTN",
            "--enforce-eager",
        ]
    return [
        _vllm_executable(),
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
        gpu_memory_utilization,
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
    ]


def _vllm_executable() -> str:
    executable = shutil.which("vllm")
    if executable:
        return executable
    sibling = Path(sys.executable).with_name("vllm")
    if sibling.exists():
        return str(sibling)
    return "vllm"


def _gpu_memory_utilization_for_spec(spec: ModelServerSpec) -> str:
    requested = str(spec.gpu_memory_utilization or "auto").strip().lower()
    if requested not in {"auto", "adaptive"}:
        return requested
    ratios = _free_memory_ratios(spec.gpu, spec.gpu_memory_reserved_mb)
    if not ratios:
        return _format_gpu_memory_utilization(spec.gpu_memory_utilization_max)
    ratio = min(spec.gpu_memory_utilization_max, min(ratios))
    ratio = max(0.05, min(0.99, ratio))
    return _format_gpu_memory_utilization(ratio)


def _free_memory_ratios(gpu: str, reserved_mb: int) -> List[float]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    ratios: List[float] = []
    for index in _gpu_indices(gpu):
        try:
            result = subprocess.run(
                [
                    executable,
                    f"--id={index}",
                    "--query-gpu=memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                total_mb = float(parts[0])
                free_mb = float(parts[1])
            except ValueError:
                continue
            if total_mb > 0:
                ratios.append(max(0.0, (free_mb - float(reserved_mb)) / total_mb))
    return ratios


def _gpu_indices(gpu: str) -> List[str]:
    tokens = [part.strip() for part in re.split(r"[,\s]+", str(gpu or "")) if part.strip()]
    return [token for token in tokens if token.isdigit()]


def _format_gpu_memory_utilization(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _vllm_cache_root(spec: ModelServerSpec) -> str:
    return str(Path(spec.log_path).parent / "vllm_cache" / _safe_name(spec.name))


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
        if _endpoint_ready(spec.base_url, spec.model):
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
        port_released = _wait_until_port_closed(spec, timeout_s, status_callback)
        status = "exited" if port_released else "exited_port_open"
        detail = f"managed process had already exited with returncode={process.returncode}"
        if not port_released:
            detail += f", but port {spec.port} is still open"
        return ModelServerStatus(
            spec.name,
            spec.base_url,
            status,
            detail,
            pid=process.pid,
            log_path=spec.log_path,
        )
    _emit(status_callback, f"Stopping {spec.name} model server pid={process.pid}")
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
        port_released = _wait_until_port_closed(spec, timeout_s, status_callback)
        status = "stopped" if port_released else "stopped_port_open"
        detail = "managed process terminated and port released to free VRAM" if port_released else (
            f"managed process terminated, but port {spec.port} is still open"
        )
        return ModelServerStatus(
            spec.name,
            spec.base_url,
            status,
            detail,
            pid=process.pid,
            log_path=spec.log_path,
        )
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        process.wait(timeout=10)
        port_released = _wait_until_port_closed(spec, timeout_s, status_callback)
        status = "killed" if port_released else "killed_port_open"
        detail = f"managed process did not exit within {timeout_s:.0f}s and was killed"
        if port_released:
            detail += "; port released"
        else:
            detail += f", but port {spec.port} is still open"
        return ModelServerStatus(
            spec.name,
            spec.base_url,
            status,
            detail,
            pid=process.pid,
            log_path=spec.log_path,
        )


def _signal_process_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(process.pid), sig)
        return
    except (OSError, ProcessLookupError):
        pass
    if sig == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _endpoint_ready(base_url: str, expected_model: str = "") -> bool:
    payload, _ = _fetch_models_payload(base_url)
    if payload is None:
        return False
    if not expected_model:
        return True
    return _models_payload_contains(payload, expected_model)


def _wait_until_port_closed(
    spec: ModelServerSpec,
    timeout_s: float,
    status_callback: Optional[StatusCallback],
) -> bool:
    if not _tcp_port_open(spec.base_url):
        return True
    _emit(status_callback, f"Waiting for {spec.name} port {spec.port} to close before reusing the stage endpoint.")
    deadline = time.time() + max(0.0, timeout_s)
    while time.time() < deadline:
        time.sleep(0.5)
        if not _tcp_port_open(spec.base_url):
            return True
    return not _tcp_port_open(spec.base_url)


def _fetch_models_payload(base_url: str) -> Tuple[Optional[object], str]:
    try:
        import requests  # type: ignore

        response = requests.get(base_url.rstrip("/") + "/models", timeout=3)
        if not 200 <= response.status_code < 300:
            return None, f"{base_url}/models returned HTTP {response.status_code}"
        try:
            return response.json(), ""
        except Exception as exc:
            return None, f"{base_url}/models did not return JSON: {exc}"
    except Exception as exc:
        return None, f"{base_url}/models request failed: {exc}"


def _open_port_detail(spec: ModelServerSpec) -> str:
    payload, error = _fetch_models_payload(spec.base_url)
    if payload is None:
        return error
    model_ids = _models_payload_model_ids(payload)
    if model_ids:
        return f"Found model(s) on that endpoint: {', '.join(model_ids)}."
    return "The endpoint responded to /models, but no model id/root/model fields were found."


def _models_payload_contains(payload: object, expected_model: str) -> bool:
    expected = str(expected_model or "").strip()
    if not expected:
        return True
    return expected in _models_payload_model_ids(payload)


def _models_payload_model_ids(payload: object) -> List[str]:
    if not isinstance(payload, dict):
        return []
    items = payload.get("data")
    if not isinstance(items, list):
        return []
    model_ids: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        values = [item.get("id"), item.get("root"), item.get("model")]
        for value in values:
            normalized = str(value or "").strip()
            if normalized and normalized not in model_ids:
                model_ids.append(normalized)
    return model_ids


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
