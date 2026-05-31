from __future__ import annotations

import os
import shlex
import subprocess
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .config import ExperimentConfig, clamp01


@dataclass
class PreprocessResult:
    audio_path: str
    applied: bool
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def preprocess_audio(audio_path: str, config: ExperimentConfig) -> PreprocessResult:
    model = (config.preprocess_model or "none").lower()
    if not config.enable_preprocess or model == "none":
        return PreprocessResult(audio_path=audio_path, applied=False, metadata={"model": "none"})
    if model in {"volume", "volume_normalization", "normalize"}:
        return _normalize_wav(audio_path, config)
    if model == "rnnoise":
        command = config.rnnoise_command or os.environ.get("ASRPP_RNNOISE_COMMAND", "")
        return _run_external_preprocessor(audio_path, config, "rnnoise", command)
    if model in {"bs-roformer", "bs_roformer", "bsroformer"}:
        command = config.bs_roformer_command or os.environ.get("ASRPP_BS_ROFORMER_COMMAND", "")
        return _run_external_preprocessor(audio_path, config, "bs_roformer", command)
    return PreprocessResult(
        audio_path=audio_path,
        applied=False,
        warnings=[f"Unknown preprocess model: {config.preprocess_model}"],
        metadata={"model": config.preprocess_model, "fallback": "original_audio"},
    )


def _run_external_preprocessor(audio_path: str, config: ExperimentConfig, model_name: str, command_template: str) -> PreprocessResult:
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
        "strength": str(clamp01(config.preprocess_strength)),
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
        metadata={"model": model_name, "command": command_template, "strength": clamp01(config.preprocess_strength)},
    )


def _normalize_wav(audio_path: str, config: ExperimentConfig) -> PreprocessResult:
    input_path = Path(audio_path)
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
    strength = clamp01(config.preprocess_strength)
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
        target_rms = int(32767 * (0.06 + 0.12 * strength))
        factor = 1.0 + ((target_rms / float(rms)) - 1.0) * strength
        normalized = _pcm16_mul(frames, factor)
        with wave.open(str(output_path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(normalized)
        return PreprocessResult(
            audio_path=str(output_path),
            applied=True,
            metadata={"model": "volume_normalization", "input_rms": rms, "target_rms": target_rms, "factor": factor},
        )
    except Exception as exc:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=[f"Volume normalization failed: {exc}"],
            metadata={"model": "volume_normalization", "fallback": "original_audio"},
        )


def _pcm16_rms(frames: bytes) -> int:
    samples = _pcm16_samples(frames)
    if not samples:
        return 0
    total = sum(sample * sample for sample in samples)
    return int((total / len(samples)) ** 0.5)


def _pcm16_mul(frames: bytes, factor: float) -> bytes:
    samples = _pcm16_samples(frames)
    for index, sample in enumerate(samples):
        value = int(round(sample * factor))
        samples[index] = max(-32768, min(32767, value))
    return samples.tobytes()


def _pcm16_samples(frames: bytes) -> array:
    samples = array("h")
    samples.frombytes(frames)
    if samples.itemsize != 2:
        raise RuntimeError("array('h') is not 16-bit on this platform.")
    if os.sys.byteorder != "little":
        samples.byteswap()
    return samples
