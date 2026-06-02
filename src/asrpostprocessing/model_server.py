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
    runtime_options: "_ServerRuntimeOptions"


@dataclass
class _ServerRuntimeOptions:
    gpu_memory_utilization: str
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    detail: str = ""


_LOCK = threading.Lock()
_PROCESSES: Dict[str, subprocess.Popen] = {}
_ADAPTIVE_START_RETRY_SCALES = (0.85, 0.70, 0.55)
_NEAR_FULL_CAPACITY_SCALE = 0.98


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
        item = _wait_until_ready_with_adaptive_retries(item, float(config.server_start_timeout_s), status_callback)
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
                unmanaged = _stop_unmanaged_stage_endpoint(spec, float(config.server_shutdown_timeout_s), status_callback)
                if unmanaged is not None:
                    statuses.append(unmanaged)
                else:
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
        if _reclaim_unmanaged_stage_endpoint(spec, status_callback):
            if _endpoint_ready(spec.base_url, spec.model):
                return ModelServerStatus(
                    spec.name,
                    spec.base_url,
                    "ready",
                    "endpoint became ready after reclaiming stale stage process",
                    log_path=spec.log_path,
                )
            if not _tcp_port_open(spec.base_url):
                _emit(status_callback, f"{spec.name} reclaimed stale stage endpoint on port {spec.port}; starting expected model.")
            else:
                _emit(status_callback, f"{spec.name} reclaimed stale stage endpoint, but port {spec.port} is still open.")
        if _tcp_port_open(spec.base_url):
            detail = _open_port_detail(spec)
            raise RuntimeError(
                f"{spec.name} model server port {spec.port} is already open, but {spec.base_url}/models is not serving "
                f"the expected model {spec.model!r}. {detail} Change the base URL port in the UI/config or stop "
                "the process that is using the port."
            )

    runtime_options = _runtime_options_for_spec(spec)
    if runtime_options.detail:
        _emit(status_callback, f"{spec.name} launch profile: {runtime_options.detail}")
    try:
        process = _start_process(spec, runtime_options)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Cannot auto-start {spec.name} model server because executable was not found: {exc.filename}. "
            "Install the required GPU serving package or set a custom server command in the UI/config."
        ) from exc
    _PROCESSES[key] = process
    _emit(status_callback, f"Started {spec.name} model server pid={process.pid}; waiting for {spec.base_url}")
    return _PendingServer(spec, process, "started", "server started and became ready", runtime_options)


def _process_key(spec: ModelServerSpec) -> str:
    return f"{spec.name}:{spec.base_url}"


def _start_process(
    spec: ModelServerSpec,
    runtime_options: Optional[_ServerRuntimeOptions] = None,
) -> subprocess.Popen:
    Path(spec.log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(spec.log_path, "ab")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PATH"] = _with_python_executable_dir(env.get("PATH", ""))
    env["LD_LIBRARY_PATH"] = _with_nvidia_library_paths(env.get("LD_LIBRARY_PATH", ""))
    env["VLLM_CACHE_ROOT"] = _vllm_cache_root(spec)
    runtime_options = runtime_options or _runtime_options_for_spec(spec)

    try:
        if spec.command_template:
            command = spec.command_template.format(
                model=spec.model,
                host=spec.host,
                port=spec.port,
                base_url=spec.base_url,
                gpu=spec.gpu,
                log_path=spec.log_path,
                gpu_memory_utilization=runtime_options.gpu_memory_utilization,
                max_model_len=runtime_options.max_model_len,
                max_num_seqs=runtime_options.max_num_seqs,
                max_num_batched_tokens=runtime_options.max_num_batched_tokens,
                vllm_cache_root=env["VLLM_CACHE_ROOT"],
                python=shlex.quote(sys.executable),
                python_dir=shlex.quote(str(Path(sys.executable).parent)),
                vllm=shlex.quote(_vllm_executable()),
            )
            command = _apply_runtime_options_to_command(command, runtime_options)
            return subprocess.Popen(
                command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )

        return subprocess.Popen(
            _default_command(spec, runtime_options),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_file.close()


def _default_command(
    spec: ModelServerSpec,
    runtime_options: Optional[_ServerRuntimeOptions] = None,
) -> List[str]:
    runtime_options = runtime_options or _runtime_options_for_spec(spec)
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
            runtime_options.gpu_memory_utilization,
            "--max-model-len",
            str(runtime_options.max_model_len),
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
        str(runtime_options.max_model_len),
        "--language-model-only",
        "--quantization",
        "bitsandbytes",
        "--load-format",
        "bitsandbytes",
        "--enforce-eager",
        "--attention-backend",
        "TRITON_ATTN",
        "--gpu-memory-utilization",
        runtime_options.gpu_memory_utilization,
        "--max-num-seqs",
        str(runtime_options.max_num_seqs),
        "--max-num-batched-tokens",
        str(runtime_options.max_num_batched_tokens),
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
    return _runtime_options_for_spec(spec).gpu_memory_utilization


def _runtime_options_for_spec(spec: ModelServerSpec, retry_scale: float = 1.0) -> _ServerRuntimeOptions:
    requested = str(spec.gpu_memory_utilization or "auto").strip().lower()
    max_model_len = _adaptive_int_option(spec, "--max-model-len", _default_max_model_len(spec.stage))
    max_num_seqs = _adaptive_int_option(spec, "--max-num-seqs", _default_max_num_seqs(spec.stage))
    max_num_batched_tokens = _adaptive_int_option(
        spec,
        "--max-num-batched-tokens",
        _default_max_num_batched_tokens(spec.stage),
    )
    if requested not in {"auto", "adaptive"}:
        return _ServerRuntimeOptions(
            requested,
            max_model_len,
            max_num_seqs,
            max_num_batched_tokens,
        )
    snapshots = _free_memory_snapshots(spec.gpu, spec.gpu_memory_reserved_mb)
    if not snapshots:
        return _ServerRuntimeOptions(
            _format_gpu_memory_utilization(spec.gpu_memory_utilization_max),
            max_model_len,
            max_num_seqs,
            max_num_batched_tokens,
            "nvidia-smi unavailable; using configured max GPU memory utilization",
        )
    raw_ratio = min(snapshot[3] for snapshot in snapshots)
    capped_ratio = min(spec.gpu_memory_utilization_max, raw_ratio)
    ratio = max(0.05, min(0.99, capped_ratio * max(0.05, min(1.0, retry_scale))))
    capacity_scale = _capacity_scale_for_ratio(ratio, spec.gpu_memory_utilization_max)
    max_model_len = _scale_capacity(max_model_len, _minimum_max_model_len(spec.stage, max_model_len), capacity_scale, 1024)
    max_num_seqs = _scale_capacity(max_num_seqs, 1, capacity_scale, 1)
    max_num_batched_tokens = _scale_capacity(
        max_num_batched_tokens,
        _minimum_max_num_batched_tokens(max_num_batched_tokens),
        capacity_scale,
        256,
    )
    snapshot_summary = "; ".join(
        f"GPU{index} free {free_mb:.0f}/{total_mb:.0f} MiB after {spec.gpu_memory_reserved_mb} MiB reserve"
        for index, total_mb, free_mb, _ratio in snapshots
    )
    return _ServerRuntimeOptions(
        _format_gpu_memory_utilization(ratio),
        max_model_len,
        max_num_seqs,
        max_num_batched_tokens,
        (
            f"{snapshot_summary}; gpu-memory-utilization={_format_gpu_memory_utilization(ratio)}, "
            f"max-model-len={max_model_len}, max-num-seqs={max_num_seqs}, "
            f"max-num-batched-tokens={max_num_batched_tokens}"
        ),
    )


def _free_memory_ratios(gpu: str, reserved_mb: int) -> List[float]:
    return [snapshot[3] for snapshot in _free_memory_snapshots(gpu, reserved_mb)]


def _free_memory_snapshots(gpu: str, reserved_mb: int) -> List[Tuple[str, float, float, float]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    snapshots: List[Tuple[str, float, float, float]] = []
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
                reserved_free_mb = max(0.0, free_mb - float(reserved_mb))
                snapshots.append((index, total_mb, reserved_free_mb, reserved_free_mb / total_mb))
    return snapshots


def _gpu_indices(gpu: str) -> List[str]:
    tokens = [part.strip() for part in re.split(r"[,\s]+", str(gpu or "")) if part.strip()]
    return [token for token in tokens if token.isdigit()]


def _format_gpu_memory_utilization(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _capacity_scale_for_ratio(ratio: float, configured_max: float) -> float:
    scale = max(0.05, min(1.0, ratio / max(0.05, configured_max)))
    if scale >= _NEAR_FULL_CAPACITY_SCALE:
        return 1.0
    return scale


def _adaptive_int_option(spec: ModelServerSpec, option: str, default: int) -> int:
    if spec.command_template:
        value = _command_option_int(spec.command_template, option)
        if value is not None:
            return value
    return default


def _command_option_int(command_template: str, option: str) -> Optional[int]:
    match = re.search(rf"{re.escape(option)}(?:=|\s+)(\d+)", command_template)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _apply_runtime_options_to_command(command: str, runtime_options: _ServerRuntimeOptions) -> str:
    replacements = {
        "--max-model-len": str(runtime_options.max_model_len),
        "--max-num-seqs": str(runtime_options.max_num_seqs),
        "--max-num-batched-tokens": str(runtime_options.max_num_batched_tokens),
    }
    for option, value in replacements.items():
        command = _replace_command_option(command, option, value)
    return command


def _replace_command_option(command: str, option: str, value: str) -> str:
    pattern = rf"({re.escape(option)}(?:=|\s+))\S+"
    return re.sub(pattern, lambda match: match.group(1) + value, command)


def _scale_capacity(max_value: int, min_value: int, scale: float, step: int) -> int:
    max_value = max(1, int(max_value))
    min_value = max(1, min(int(min_value), max_value))
    value = max(min_value, int(max_value * scale))
    if step > 1:
        value = max(min_value, (value // step) * step)
    return min(max_value, value)


def _default_max_model_len(stage: str) -> int:
    return 65536 if stage == "asr" else 2048


def _default_max_num_seqs(stage: str) -> int:
    return 1


def _default_max_num_batched_tokens(stage: str) -> int:
    return 2048


def _minimum_max_model_len(stage: str, max_value: int) -> int:
    minimum = 8192 if stage == "asr" else 2048
    return min(max_value, minimum)


def _minimum_max_num_batched_tokens(max_value: int) -> int:
    return min(max_value, 2048)


def _vllm_cache_root(spec: ModelServerSpec) -> str:
    return str(Path(spec.log_path).parent / "vllm_cache" / _safe_name(spec.name))


def _with_python_executable_dir(current: str) -> str:
    python_dir = str(Path(sys.executable).parent)
    existing = [part for part in current.split(os.pathsep) if part]
    merged: List[str] = []
    for path in [python_dir] + existing:
        if path and path not in merged:
            merged.append(path)
    return os.pathsep.join(merged)


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


def _wait_until_ready_with_adaptive_retries(
    item: _PendingServer,
    timeout_s: float,
    status_callback: Optional[StatusCallback],
) -> _PendingServer:
    attempt = 0
    while True:
        try:
            _wait_until_ready(item.spec, timeout_s, item.process)
            return item
        except RuntimeError as exc:
            if not _should_retry_adaptive_start(item.spec, attempt, exc):
                raise
            _unregister_process(item.spec, item.process)
            if item.process is not None and item.process.poll() is None:
                _signal_process_group(item.process, signal.SIGTERM)
            _wait_until_port_closed(item.spec, min(timeout_s, 10.0), status_callback)
            retry_scale = _ADAPTIVE_START_RETRY_SCALES[attempt]
            runtime_options = _runtime_options_for_spec(item.spec, retry_scale=retry_scale)
            _emit(
                status_callback,
                f"{item.spec.name} model server failed with the current VRAM profile; "
                f"retrying on GPU {item.spec.gpu} with smaller capacity: {runtime_options.detail}",
            )
            process = _start_process(item.spec, runtime_options)
            with _LOCK:
                _PROCESSES[_process_key(item.spec)] = process
            _emit(status_callback, f"Restarted {item.spec.name} model server pid={process.pid}; waiting for {item.spec.base_url}")
            item = _PendingServer(
                item.spec,
                process,
                "started",
                "server started and became ready after adaptive VRAM retry",
                runtime_options,
            )
            attempt += 1


def _should_retry_adaptive_start(spec: ModelServerSpec, attempt: int, exc: RuntimeError) -> bool:
    if attempt >= len(_ADAPTIVE_START_RETRY_SCALES):
        return False
    requested = str(spec.gpu_memory_utilization or "auto").strip().lower()
    if requested not in {"auto", "adaptive"}:
        return False
    message = str(exc).lower()
    if "exited before becoming ready" not in message:
        return False
    resource_markers = (
        "engine core initialization failed",
        "failed core proc",
        "cuda",
        "memory",
        "out of memory",
        "oom",
        "kv cache",
        "cache blocks",
    )
    return any(marker in message for marker in resource_markers)


def _unregister_process(spec: ModelServerSpec, process: Optional[subprocess.Popen]) -> None:
    if process is None:
        return
    key = _process_key(spec)
    with _LOCK:
        registered = _PROCESSES.get(key)
        if registered is process:
            _PROCESSES.pop(key, None)


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


def _reclaim_unmanaged_stage_endpoint(spec: ModelServerSpec, status_callback: Optional[StatusCallback]) -> bool:
    if not _stage_replica_spec(spec):
        return False
    payload, _error = _fetch_models_payload(spec.base_url)
    if payload is None or _models_payload_contains(payload, spec.model):
        return False
    stale_models = _models_payload_model_ids(payload)
    pids = _safe_stage_server_pids(spec)
    if not pids:
        return False
    model_detail = f" serving {', '.join(stale_models)}" if stale_models else ""
    _emit(
        status_callback,
        f"{spec.name} found stale stage process on port {spec.port}{model_detail}; reclaiming same-user process(es) {pids}.",
    )
    return _terminate_unmanaged_stage_pids(spec, pids, status_callback)


def _stop_unmanaged_stage_endpoint(
    spec: ModelServerSpec,
    timeout_s: float,
    status_callback: Optional[StatusCallback],
) -> Optional[ModelServerStatus]:
    if not _stage_replica_spec(spec) or not _tcp_port_open(spec.base_url):
        return None
    pids = _safe_stage_server_pids(spec)
    if not pids:
        return None
    _emit(status_callback, f"Stopping unmanaged {spec.name} stage server process(es) {pids} on port {spec.port}")
    stopped = _terminate_unmanaged_stage_pids(spec, pids, status_callback, timeout_s=timeout_s)
    return ModelServerStatus(
        spec.name,
        spec.base_url,
        "stopped_unmanaged" if stopped else "unmanaged_port_open",
        (
            "unmanaged same-user stage server terminated and port released"
            if stopped
            else f"unmanaged same-user stage server was signaled, but port {spec.port} is still open"
        ),
        pid=pids[0] if pids else None,
        log_path=spec.log_path,
    )


def _stage_replica_spec(spec: ModelServerSpec) -> bool:
    return spec.stage in {"asr", "post"} and re.match(r"^(?:asr|post)_stage_\d+$", spec.name or "") is not None


def _safe_stage_server_pids(spec: ModelServerSpec) -> List[int]:
    pids = _listening_pids_for_port(spec.port)
    safe: List[int] = []
    for pid in pids:
        if _pid_owned_by_current_user(pid) and _stage_server_cmdline(_cmdline_for_pid(pid)):
            safe.append(pid)
    return safe


def _terminate_unmanaged_stage_pids(
    spec: ModelServerSpec,
    pids: List[int],
    status_callback: Optional[StatusCallback],
    timeout_s: float = 15.0,
) -> bool:
    for pid in pids:
        _signal_pid_group(pid, signal.SIGTERM)
    if _wait_until_port_closed(spec, timeout_s, status_callback):
        return True
    for pid in pids:
        if _pid_exists(pid):
            _signal_pid_group(pid, signal.SIGKILL)
    return _wait_until_port_closed(spec, min(timeout_s, 10.0), status_callback)


def _listening_pids_for_port(port: int) -> List[int]:
    pids: List[int] = []
    for pid in _listening_pids_from_ss(port):
        if pid not in pids:
            pids.append(pid)
    if pids:
        return pids
    for pid in _listening_pids_from_lsof(port):
        if pid not in pids:
            pids.append(pid)
    if pids:
        return pids
    for pid in _listening_pids_from_fuser(port):
        if pid not in pids:
            pids.append(pid)
    return pids


def _listening_pids_from_ss(port: int) -> List[int]:
    executable = shutil.which("ss")
    if not executable:
        return []
    try:
        result = subprocess.run([executable, "-ltnp"], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    pids: List[int] = []
    port_pattern = re.compile(rf":{int(port)}(?:\s|$)")
    for line in result.stdout.splitlines():
        if not port_pattern.search(line):
            continue
        for match in re.findall(r"pid=(\d+)", line):
            pid = int(match)
            if pid not in pids:
                pids.append(pid)
    return pids


def _listening_pids_from_lsof(port: int) -> List[int]:
    executable = shutil.which("lsof")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return _int_lines(result.stdout)


def _listening_pids_from_fuser(port: int) -> List[int]:
    executable = shutil.which("fuser")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "-n", "tcp", str(int(port))],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return _int_tokens(result.stdout + "\n" + result.stderr)


def _int_lines(value: str) -> List[int]:
    pids: List[int] = []
    for line in value.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        if pid not in pids:
            pids.append(pid)
    return pids


def _int_tokens(value: str) -> List[int]:
    pids: List[int] = []
    for token in re.findall(r"\b\d+\b", value or ""):
        pid = int(token)
        if pid not in pids:
            pids.append(pid)
    return pids


def _pid_owned_by_current_user(pid: int) -> bool:
    try:
        return os.stat(f"/proc/{int(pid)}").st_uid == os.geteuid()
    except OSError:
        pass
    try:
        result = subprocess.run(["ps", "-o", "uid=", "-p", str(int(pid))], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    try:
        return int(result.stdout.strip()) == os.geteuid()
    except ValueError:
        return False


def _cmdline_for_pid(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        if raw:
            return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        pass
    try:
        result = subprocess.run(["ps", "-o", "command=", "-p", str(int(pid))], capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _stage_server_cmdline(command: str) -> bool:
    normalized = f" {command or ''} "
    if " asrpostprocessing.qwen_asr_serve_compat " in normalized:
        return True
    if re.search(r"(?:^|[/\s])vllm(?:\s|$)", command or "") and " serve " in normalized:
        return True
    if " vllm.entrypoints.openai.api_server " in normalized:
        return True
    return False


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _signal_pid_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(int(pid)), sig)
        return
    except (OSError, ProcessLookupError):
        pass
    try:
        os.kill(int(pid), sig)
    except (OSError, ProcessLookupError):
        pass


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
