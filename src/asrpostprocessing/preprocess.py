from __future__ import annotations

import os
import shlex
import subprocess
import wave
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .config import ExperimentConfig, clamp01


@dataclass
class PreprocessResult:
    audio_path: str
    applied: bool
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def preprocess_audio(audio_path: str, config: ExperimentConfig) -> PreprocessResult:
    plan = _preprocess_plan(config)
    if not plan:
        return PreprocessResult(audio_path=audio_path, applied=False, metadata={"steps": []})

    current_path = audio_path
    warnings: List[str] = []
    steps: List[Dict[str, Any]] = []
    applied = False
    for step in plan:
        if step["type"] == "noise_reduction":
            result = _noise_reduce(current_path, config, step["model"])
        elif step["type"] == "volume_normalization":
            result = _normalize_wav(current_path, config)
        else:
            continue
        warnings.extend(result.warnings)
        steps.append(
            {
                "step": step["type"],
                "model": result.metadata.get("model", step.get("model")),
                "input_audio_path": current_path,
                "output_audio_path": result.audio_path,
                "applied": result.applied,
                "warnings": result.warnings,
                "metadata": result.metadata,
            }
        )
        if result.applied:
            current_path = result.audio_path
            applied = True
    return PreprocessResult(
        audio_path=current_path,
        applied=applied,
        warnings=warnings,
        metadata={
            "steps": steps,
            "execution_order": [step["type"] for step in plan],
            "legacy_enable_preprocess": bool(config.enable_preprocess),
        },
        steps=steps,
    )


def _preprocess_plan(config: ExperimentConfig) -> List[Dict[str, str]]:
    plan: List[Dict[str, str]] = []
    noise_model = (config.noise_reduction_model or "none").lower()
    enable_noise = bool(config.enable_noise_reduction) and noise_model != "none"
    enable_volume = bool(config.enable_volume_normalization)

    legacy_model = (config.preprocess_model or "none").lower()
    if bool(config.enable_preprocess):
        if legacy_model in {"rnnoise", "bs-roformer", "bs_roformer", "bsroformer"} and not enable_noise:
            noise_model = legacy_model
            enable_noise = True
        elif legacy_model in {"volume", "volume_normalization", "normalize"} and not enable_volume:
            enable_volume = True

    if enable_noise:
        plan.append({"type": "noise_reduction", "model": noise_model})
    if enable_volume:
        plan.append({"type": "volume_normalization", "model": "volume_normalization"})
    return plan


def _noise_reduce(audio_path: str, config: ExperimentConfig, model: str) -> PreprocessResult:
    model = (model or "none").lower()
    if model == "rnnoise":
        command = config.rnnoise_command or os.environ.get("ASRPP_RNNOISE_COMMAND", "")
        return _run_external_preprocessor(audio_path, config, "rnnoise", command, config.noise_reduction_strength)
    if model in {"bs-roformer", "bs_roformer", "bsroformer"}:
        command = config.bs_roformer_command or os.environ.get("ASRPP_BS_ROFORMER_COMMAND", "")
        return _run_external_preprocessor(audio_path, config, "bs_roformer", command, config.noise_reduction_strength)
    return PreprocessResult(
        audio_path=audio_path,
        applied=False,
        warnings=[f"Unknown noise reduction model: {config.noise_reduction_model}"],
        metadata={"model": config.noise_reduction_model, "fallback": "input_audio"},
    )


def _run_external_preprocessor(
    audio_path: str,
    config: ExperimentConfig,
    model_name: str,
    command_template: str,
    strength_value: float,
) -> PreprocessResult:
    if not command_template:
        env_name = "ASRPP_RNNOISE_COMMAND" if model_name == "rnnoise" else "ASRPP_BS_ROFORMER_COMMAND"
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=[
                f"{model_name} requires an external command template. "
                f"Set config field `{model_name}_command` or environment variable `{env_name}`."
            ],
            metadata={"model": model_name, "fallback": "original_audio"},
        )
    input_path = Path(audio_path)
    output_dir = Path(config.output_dir) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.{model_name}.wav"
    format_values = {
        "input": str(input_path),
        "output": str(output_path),
        "strength": str(clamp01(strength_value)),
    }
    try:
        if "{input}" in command_template or "{output}" in command_template:
            command = command_template.format(**format_values)
        else:
            command = f"{command_template} {shlex.quote(str(input_path))} {shlex.quote(str(output_path))}"
        subprocess.run(shlex.split(command), check=True, capture_output=True, text=True)
    except Exception as exc:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=[f"{model_name} preprocessing failed: {exc}"],
            metadata={"model": model_name, "fallback": "original_audio"},
        )
    if not output_path.exists():
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=[f"{model_name} command completed but did not create {output_path}."],
            metadata={"model": model_name, "fallback": "original_audio"},
        )
    return PreprocessResult(
        audio_path=str(output_path),
        applied=True,
        metadata={"model": model_name, "command": command_template, "strength": clamp01(strength_value)},
    )


def _normalize_wav(audio_path: str, config: ExperimentConfig) -> PreprocessResult:
    input_path = _volume_input_path(audio_path, config)
    if input_path is None:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["Volume normalization requires PCM WAV input or configured ffmpeg_command for conversion."],
            metadata={"model": "volume_normalization", "fallback": "input_audio"},
        )
    if input_path.suffix.lower() != ".wav":
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["Volume normalization currently supports PCM WAV input only."],
            metadata={"model": "volume_normalization"},
        )
    output_dir = Path(config.output_dir) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.normalized.wav"
    strength = clamp01(config.volume_normalization_strength)
    try:
        with wave.open(str(input_path), "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(reader.getnframes())
        if params.sampwidth != 2:
            return PreprocessResult(
                audio_path=audio_path,
                applied=False,
                warnings=["Volume normalization supports 16-bit PCM WAV in this MVP."],
                metadata={"sample_width": params.sampwidth},
            )
        rms = _pcm16_rms(frames)
        if rms == 0:
            return PreprocessResult(
                audio_path=audio_path,
                applied=False,
                warnings=["Input WAV is silent; normalization skipped."],
                metadata={"rms": rms},
            )
        target_rms = max(1, int(32767 * (10 ** (float(config.volume_target_dbfs) / 20.0))))
        factor = 1.0 + ((target_rms / float(rms)) - 1.0) * strength
        normalized, clipped_samples = _pcm16_mul(frames, factor)
        with wave.open(str(output_path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(normalized)
        return PreprocessResult(
            audio_path=str(output_path),
            applied=True,
            metadata={
                "model": "volume_normalization",
                "input_rms": rms,
                "target_rms": target_rms,
                "target_dbfs": float(config.volume_target_dbfs),
                "strength": strength,
                "gain_factor": factor,
                "clipped_samples": clipped_samples,
                "converted_input_path": str(input_path) if str(input_path) != audio_path else "",
            },
        )
    except Exception as exc:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=[f"Volume normalization failed: {exc}"],
            metadata={"model": "volume_normalization", "fallback": "original_audio"},
        )


def _volume_input_path(audio_path: str, config: ExperimentConfig) -> Path | None:
    input_path = Path(audio_path)
    if input_path.suffix.lower() == ".wav":
        return input_path
    command_template = config.ffmpeg_command or os.environ.get("ASRPP_FFMPEG_COMMAND", "")
    if not command_template:
        return None
    output_dir = Path(config.output_dir) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.volume-input.wav"
    format_values = {"input": str(input_path), "output": str(output_path)}
    try:
        if "{input}" in command_template or "{output}" in command_template:
            command = command_template.format(**format_values)
        else:
            command = f"{command_template} -y -i {shlex.quote(str(input_path))} {shlex.quote(str(output_path))}"
        subprocess.run(shlex.split(command), check=True, capture_output=True, text=True)
    except Exception:
        return None
    return output_path if output_path.exists() else None


def _pcm16_rms(frames: bytes) -> int:
    samples = _pcm16_samples(frames)
    if not samples:
        return 0
    total = sum(sample * sample for sample in samples)
    return int((total / len(samples)) ** 0.5)


def _pcm16_mul(frames: bytes, factor: float) -> tuple[bytes, int]:
    samples = _pcm16_samples(frames)
    clipped = 0
    for index, sample in enumerate(samples):
        value = int(round(sample * factor))
        clipped_value = max(-32768, min(32767, value))
        if clipped_value != value:
            clipped += 1
        samples[index] = clipped_value
    return samples.tobytes(), clipped


def _pcm16_samples(frames: bytes) -> array:
    samples = array("h")
    samples.frombytes(frames)
    if samples.itemsize != 2:
        raise RuntimeError("array('h') is not 16-bit on this platform.")
    if os.sys.byteorder != "little":
        samples.byteswap()
    return samples
