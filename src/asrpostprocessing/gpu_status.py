from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, List


def query_gpu_status() -> Dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {
            "available": False,
            "timestamp": time.time(),
            "error": "nvidia-smi executable was not found on PATH.",
            "gpus": [],
            "processes": [],
        }

    status: Dict[str, Any] = {
        "available": True,
        "timestamp": time.time(),
        "nvidia_smi": executable,
        "gpus": [],
        "processes": [],
        "warnings": [],
    }

    gpu_result = _run_nvidia_smi(
        executable,
        [
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit,pstate",
            "--format=csv,noheader,nounits",
        ],
    )
    if gpu_result.returncode != 0:
        status["available"] = False
        status["error"] = _result_error(gpu_result)
        return status
    status["gpus"] = [_parse_gpu_line(line) for line in gpu_result.stdout.splitlines() if line.strip()]

    process_result = _run_nvidia_smi(
        executable,
        [
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    )
    if process_result.returncode == 0:
        status["processes"] = [
            _parse_process_line(line) for line in process_result.stdout.splitlines() if line.strip()
        ]
    else:
        status["warnings"].append(f"compute process query failed: {_result_error(process_result)}")
    return status


def _run_nvidia_smi(executable: str, args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run([executable, *args], capture_output=True, text=True, timeout=10)


def _parse_gpu_line(line: str) -> Dict[str, Any]:
    parts = [part.strip() for part in line.split(",")]
    while len(parts) < 10:
        parts.append("")
    total = _to_int(parts[2])
    used = _to_int(parts[3])
    free = _to_int(parts[4])
    return {
        "index": _to_int(parts[0]),
        "name": parts[1],
        "memory_total_mb": total,
        "memory_used_mb": used,
        "memory_free_mb": free,
        "memory_used_percent": round((used / total) * 100.0, 2) if total else None,
        "gpu_utilization_percent": _to_int(parts[5]),
        "temperature_c": _to_int(parts[6]),
        "power_draw_w": _to_float(parts[7]),
        "power_limit_w": _to_float(parts[8]),
        "performance_state": parts[9] or None,
    }


def _parse_process_line(line: str) -> Dict[str, Any]:
    parts = [part.strip() for part in line.split(",", 2)]
    while len(parts) < 3:
        parts.append("")
    return {
        "pid": _to_int(parts[0]),
        "process_name": parts[1],
        "used_memory_mb": _to_int(parts[2]),
    }


def _to_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _result_error(result: subprocess.CompletedProcess) -> str:
    detail = (result.stderr or result.stdout or "").strip()
    return detail or f"nvidia-smi exited with code {result.returncode}"
