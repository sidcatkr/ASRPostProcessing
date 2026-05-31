from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from .config import ExperimentConfig, dump_resolved_yaml
from .schemas import Edit, MetricsResult


class RunLogger:
    def __init__(self, config: ExperimentConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.output_dir = Path(config.output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = Path(config.runs_dir) / run_id
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.output_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_edits(self, edits: Iterable[Edit]) -> Path:
        path = self.output_dir / "edits.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for edit in edits:
                handle.write(json.dumps(edit.to_dict(), ensure_ascii=False) + "\n")
        return path

    def write_config(self) -> Path:
        path = self.output_dir / "config.resolved.yaml"
        path.write_text(dump_resolved_yaml(self.config), encoding="utf-8")
        return path

    def write_tensorboard_metrics(self, metrics: MetricsResult) -> Path:
        fallback = self.runs_dir / "metrics.tsv"
        data = {
            **metrics.to_dict(),
            "keyword_bias_weight": self.config.keyword_bias_weight,
            "noise_reduction_strength": self.config.noise_reduction_strength,
            "volume_normalization_strength": self.config.volume_normalization_strength,
            "rag_strength": self.config.rag_strength,
            "postprocess_strength": self.config.postprocess_strength,
            "search_strength": self.config.search_strength,
        }
        fallback.write_text(
            "\n".join(f"{key}\t{value}" for key, value in sorted(data.items()) if value is not None) + "\n",
            encoding="utf-8",
        )
        if not _write_torch_tensorboard_scalars(self.runs_dir, data):
            _write_tensorboard_scalars(self.runs_dir, data)
        return fallback


def make_run_id(prefix: str = "run") -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    millis = int((time.time() % 1) * 1000)
    return f"{prefix}-{timestamp}-{millis:03d}"


def _write_torch_tensorboard_scalars(log_dir: Path, data: Dict[str, Any]) -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore

        writer = SummaryWriter(log_dir=str(log_dir))
        for key, value in data.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, global_step=0)
        writer.flush()
        writer.close()
        return True
    except Exception:
        return False


def _write_tensorboard_scalars(log_dir: Path, data: Dict[str, Any]) -> bool:
    try:
        from tensorboard.compat.proto.event_pb2 import Event  # type: ignore
        from tensorboard.compat.proto.summary_pb2 import Summary  # type: ignore
        from tensorboard.summary.writer.event_file_writer import EventFileWriter  # type: ignore

        writer = EventFileWriter(str(log_dir))
        wall_time = time.time()
        for key, value in data.items():
            if not isinstance(value, (int, float)):
                continue
            summary = Summary(value=[Summary.Value(tag=key, simple_value=float(value))])
            writer.add_event(Event(wall_time=wall_time, step=0, summary=summary))
        writer.flush()
        writer.close()
        return True
    except Exception:
        return False
