from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List
from urllib.parse import urlsplit, urlunsplit

from .config import ExperimentConfig

PROMETHEUS_SAMPLE_RE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def vllm_metrics_endpoint_pool(config: ExperimentConfig) -> List[str]:
    endpoints: List[str] = []
    if str(config.asr_backend or "").lower() in {"vllm", "vllm_chat", "openai_audio"}:
        endpoints.extend(_lane_urls(config, "asr_base_url", "asr_model"))
        endpoints.extend(str(item).strip() for item in (config.asr_base_urls or []) if str(item).strip())
        endpoints.append(config.asr_base_url)
    if config.enable_llm_postprocess and str(config.post_backend or "").lower() in {"vllm", "vllm_openai", "openai"}:
        endpoints.extend(_lane_urls(config, "post_base_url", "post_model"))
        endpoints.extend(str(item).strip() for item in (config.post_base_urls or []) if str(item).strip())
        endpoints.append(config.post_base_url)
    return _dedupe(endpoints)


def query_vllm_metrics(base_urls: Iterable[str], timeout_s: float = 0.75) -> Dict[str, Any]:
    endpoints: Dict[str, Any] = {}
    for base_url in _dedupe(str(item).strip() for item in base_urls if str(item).strip()):
        metrics_url = _metrics_url(base_url)
        try:
            import requests  # type: ignore

            response = requests.get(metrics_url, timeout=timeout_s)
            response.raise_for_status()
            raw_metrics = parse_prometheus_metrics(response.text)
            endpoints[base_url] = {
                "available": True,
                "metrics_url": metrics_url,
                "counters": summarize_vllm_counters(raw_metrics),
            }
        except Exception as exc:
            endpoints[base_url] = {
                "available": False,
                "metrics_url": metrics_url,
                "error": str(exc),
                "counters": {},
            }
    return {"timestamp": time.time(), "endpoints": endpoints}


def parse_prometheus_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = PROMETHEUS_SAMPLE_RE.match(stripped)
        if not match:
            continue
        name, value = match.groups()
        if name.endswith(("_bucket", "_created")):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        metrics[name] = metrics.get(name, 0.0) + number
    return metrics


def summarize_vllm_counters(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "preemption_count": _sum_names(metrics, ["preempt"]),
        "prompt_tokens": _sum_names(metrics, ["prompt_tokens_total", "prompt_token_total"]),
        "generation_tokens": _sum_names(
            metrics,
            ["generation_tokens_total", "generated_tokens_total", "output_tokens_total", "completion_tokens_total"],
        ),
        "request_success_count": _sum_names(metrics, ["request_success", "requests_success", "requests_total"]),
    }


def diff_vllm_metrics(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_endpoints = before.get("endpoints") if isinstance(before, dict) else {}
    after_endpoints = after.get("endpoints") if isinstance(after, dict) else {}
    if not isinstance(before_endpoints, dict) or not isinstance(after_endpoints, dict):
        return _empty_delta(before, after)
    endpoint_deltas: Dict[str, Dict[str, Any]] = {}
    total_delta = {
        "preemption_count": 0.0,
        "prompt_tokens": 0.0,
        "generation_tokens": 0.0,
        "total_tokens": 0.0,
        "request_success_count": 0.0,
    }
    warnings: List[str] = []
    available = False
    for endpoint in sorted(set(before_endpoints) | set(after_endpoints)):
        before_item = before_endpoints.get(endpoint) or {}
        after_item = after_endpoints.get(endpoint) or {}
        if not before_item.get("available") or not after_item.get("available"):
            error = after_item.get("error") or before_item.get("error") or "metrics unavailable"
            endpoint_deltas[endpoint] = {"available": False, "error": error}
            warnings.append(f"{endpoint}: {error}")
            continue
        before_counters = before_item.get("counters") if isinstance(before_item.get("counters"), dict) else {}
        after_counters = after_item.get("counters") if isinstance(after_item.get("counters"), dict) else {}
        delta = {
            "preemption_count": _counter_delta(before_counters, after_counters, "preemption_count"),
            "prompt_tokens": _counter_delta(before_counters, after_counters, "prompt_tokens"),
            "generation_tokens": _counter_delta(before_counters, after_counters, "generation_tokens"),
            "request_success_count": _counter_delta(before_counters, after_counters, "request_success_count"),
        }
        delta["total_tokens"] = delta["prompt_tokens"] + delta["generation_tokens"]
        endpoint_deltas[endpoint] = {"available": True, **delta}
        for key in total_delta:
            total_delta[key] += float(delta.get(key) or 0.0)
        available = True
    return {
        "available": available,
        "before_timestamp": before.get("timestamp") if isinstance(before, dict) else None,
        "after_timestamp": after.get("timestamp") if isinstance(after, dict) else None,
        "delta": total_delta,
        "endpoint_deltas": endpoint_deltas,
        "warnings": warnings,
    }


def _metrics_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    path = f"{path}/metrics" if path else "/metrics"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _lane_urls(config: ExperimentConfig, url_key: str, model_key: str) -> List[str]:
    model_value = str(getattr(config, model_key, "") or "")
    urls: List[str] = []
    for lane in config.pipeline_lanes or []:
        if not isinstance(lane, dict):
            continue
        url = str(lane.get(url_key) or "").strip()
        lane_model = str(lane.get(model_key) or "").strip()
        if url and (not lane_model or lane_model == model_value):
            urls.append(url)
    return urls


def _dedupe(values: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _sum_names(metrics: Dict[str, float], needles: List[str]) -> float:
    total = 0.0
    for name, value in metrics.items():
        normalized = name.lower()
        if any(needle in normalized for needle in needles):
            total += float(value)
    return total


def _counter_delta(before: Dict[str, Any], after: Dict[str, Any], key: str) -> float:
    before_value = _to_float(before.get(key))
    after_value = _to_float(after.get(key))
    if before_value is None or after_value is None:
        return 0.0
    return max(0.0, after_value - before_value)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "available": False,
        "before_timestamp": before.get("timestamp") if isinstance(before, dict) else None,
        "after_timestamp": after.get("timestamp") if isinstance(after, dict) else None,
        "delta": {},
        "endpoint_deltas": {},
        "warnings": ["metrics snapshots were not dictionaries"],
    }
