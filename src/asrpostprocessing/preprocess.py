from __future__ import annotations

import os
import shutil
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
    if model in {"rnnoise", "bs-roformer", "bs_roformer", "bsroformer", "basic", "built-in", "built_in", "denoise"}:
        return _denoise_audio(audio_path, config, model)
    return PreprocessResult(
        audio_path=audio_path,
        applied=False,
        warnings=[f"Unknown noise reduction model: {config.noise_reduction_model}"],
        metadata={"model": config.noise_reduction_model, "fallback": "input_audio"},
    )


def _denoise_audio(audio_path: str, config: ExperimentConfig, model_name: str) -> PreprocessResult:
    input_path = Path(audio_path)
    output_dir = Path(config.output_dir) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = _safe_preprocess_name(model_name)
    output_path = output_dir / f"{input_path.stem}.{safe_model}.denoised.wav"
    strength = clamp01(config.noise_reduction_strength)
    if input_path.suffix.lower() == ".wav":
        result = _denoise_pcm16_wav(input_path, output_path, model_name, strength)
        if result is not None:
            return result
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["Noise reduction for this input requires ffmpeg on PATH or 16-bit PCM WAV input."],
            metadata={"model": model_name, "fallback": "original_audio"},
        )
    return _denoise_with_ffmpeg(ffmpeg, input_path, output_path, model_name, strength)


def _denoise_pcm16_wav(input_path: Path, output_path: Path, model_name: str, strength: float) -> PreprocessResult | None:
    try:
        with wave.open(str(input_path), "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(reader.getnframes())
    except (EOFError, wave.Error):
        return None
    if params.sampwidth != 2:
        return None
    rms = _pcm16_rms(frames)
    if rms == 0:
        return PreprocessResult(
            audio_path=str(input_path),
            applied=False,
            warnings=["Input WAV is silent; noise reduction skipped."],
            metadata={"model": model_name, "processor": "pcm_noise_gate", "rms": rms},
        )
    threshold = max(24, int(rms * (0.35 + strength)))
    attenuation = max(0.05, 1.0 - (0.8 * strength))
    denoised, attenuated_samples = _pcm16_noise_gate(frames, threshold, attenuation)
    with wave.open(str(output_path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(denoised)
    return PreprocessResult(
        audio_path=str(output_path),
        applied=True,
        metadata={
            "model": model_name,
            "processor": "pcm_noise_gate",
            "strength": strength,
            "input_rms": rms,
            "threshold": threshold,
            "attenuation": attenuation,
            "attenuated_samples": attenuated_samples,
        },
    )


def _denoise_with_ffmpeg(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    model_name: str,
    strength: float,
) -> PreprocessResult:
    noise_floor = -60.0 + (35.0 * strength)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-af",
        f"afftdn=nf={noise_floor:.1f}",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as exc:
        return PreprocessResult(
            audio_path=str(input_path),
            applied=False,
            warnings=[f"Noise reduction failed: {exc}"],
            metadata={"model": model_name, "processor": "ffmpeg_afftdn", "fallback": "original_audio"},
        )
    if not output_path.exists():
        return PreprocessResult(
            audio_path=str(input_path),
            applied=False,
            warnings=[f"Noise reduction did not create {output_path}."],
            metadata={"model": model_name, "processor": "ffmpeg_afftdn", "fallback": "original_audio"},
        )
    return PreprocessResult(
        audio_path=str(output_path),
        applied=True,
        metadata={
            "model": model_name,
            "processor": "ffmpeg_afftdn",
            "strength": strength,
            "noise_floor_db": noise_floor,
        },
    )


def _normalize_wav(audio_path: str, config: ExperimentConfig) -> PreprocessResult:
    input_path = _volume_input_path(audio_path, config, "volume-input")
    if input_path is None:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["Volume normalization for this input requires ffmpeg on PATH or 16-bit PCM WAV input."],
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
            converted_path = _convert_audio_to_pcm16_wav(Path(audio_path), config, "volume-input")
            if converted_path is not None and converted_path != input_path:
                input_path = converted_path
                with wave.open(str(input_path), "rb") as reader:
                    params = reader.getparams()
                    frames = reader.readframes(reader.getnframes())
            if params.sampwidth != 2:
                return PreprocessResult(
                    audio_path=audio_path,
                    applied=False,
                    warnings=["Volume normalization requires 16-bit PCM WAV input or ffmpeg on PATH for conversion."],
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


def _volume_input_path(audio_path: str, config: ExperimentConfig, tag: str) -> Path | None:
    input_path = Path(audio_path)
    if input_path.suffix.lower() == ".wav":
        return input_path
    return _convert_audio_to_pcm16_wav(input_path, config, tag)


def _convert_audio_to_pcm16_wav(input_path: Path, config: ExperimentConfig, tag: str) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    output_dir = Path(config.output_dir) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}.{tag}.wav"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return output_path if output_path.exists() else None


def _safe_preprocess_name(value: str) -> str:
    normalized = (value or "preprocess").lower().replace("-", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in normalized)


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


def _pcm16_noise_gate(frames: bytes, threshold: int, attenuation: float) -> tuple[bytes, int]:
    samples = _pcm16_samples(frames)
    attenuated = 0
    for index, sample in enumerate(samples):
        if abs(sample) > threshold:
            continue
        value = int(round(sample * attenuation))
        if value != sample:
            attenuated += 1
        samples[index] = value
    return samples.tobytes(), attenuated


def _pcm16_samples(frames: bytes) -> array:
    samples = array("h")
    samples.frombytes(frames)
    if samples.itemsize != 2:
        raise RuntimeError("array('h') is not 16-bit on this platform.")
    if os.sys.byteorder != "little":
        samples.byteswap()
    return samples
