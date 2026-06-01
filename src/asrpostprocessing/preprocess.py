from __future__ import annotations

import os
import shutil
import subprocess
import wave
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from .cache import cache_json_path, file_sha256, read_json, stable_json_hash, write_json_atomic
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
    cache_path = _preprocess_cache_path(audio_path, config, plan) if config.preprocess_cache_enabled else None
    if cache_path is not None:
        cached = read_json(cache_path)
        if cached and cached.get("audio_path") and Path(str(cached["audio_path"])).exists():
            cached.setdefault("metadata", {})
            cached["metadata"]["cache_hit"] = True
            cached["metadata"]["cache_path"] = str(cache_path)
            return _preprocess_result_from_dict(cached)

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
    result = PreprocessResult(
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
    if cache_path is not None and result.applied:
        payload = result.to_dict()
        payload.setdefault("metadata", {})
        payload["metadata"]["cache_hit"] = False
        payload["metadata"]["cache_path"] = str(cache_path)
        write_json_atomic(cache_path, payload)
        result.metadata["cache_hit"] = False
        result.metadata["cache_path"] = str(cache_path)
    return result


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
    if config.noise_reduction_command:
        return _denoise_with_custom_command(audio_path, config, model)
    if model in {"afftdn", "ffmpeg_afftdn", "basic", "built-in", "built_in", "denoise"}:
        return _denoise_audio(audio_path, config, model)
    if model in {"deepfilternet2", "deep_filter_net2", "deep-filter-net2", "deepfilternet2_pf", "deepfilternet2-pf"}:
        return _denoise_with_deepfilternet(
            audio_path,
            config,
            model,
            "DeepFilterNet2",
            postfilter=model.endswith("_pf") or model.endswith("-pf"),
        )
    if model in {"deepfilternet3", "deep_filter_net3", "deep-filter-net3"}:
        return _denoise_with_deepfilternet(audio_path, config, model, "DeepFilterNet3", postfilter=False)
    if model == "rnnoise":
        return _denoise_with_rnnoise(audio_path, config, model)
    if model in {"bs-roformer", "bs_roformer", "bsroformer"}:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=[
                "BS-RoFormer is not wired as a speech denoise backend yet. "
                "Use afftdn, rnnoise, DeepFilterNet2/3, or set noise_reduction_command for an external backend."
            ],
            metadata={"model": model, "fallback": "original_audio"},
        )
    return PreprocessResult(
        audio_path=audio_path,
        applied=False,
        warnings=[f"Unknown noise reduction model: {config.noise_reduction_model}"],
        metadata={"model": config.noise_reduction_model, "fallback": "input_audio"},
    )


def _denoise_audio(audio_path: str, config: ExperimentConfig, model_name: str) -> PreprocessResult:
    input_path = Path(audio_path)
    safe_model = _safe_preprocess_name(model_name)
    output_path = _preprocessed_output_path(config, input_path, safe_model, "denoised")
    strength = clamp01(config.noise_reduction_strength)
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["Noise reduction requires ffmpeg or imageio-ffmpeg to create browser-playable preview audio."],
            metadata={"model": model_name, "fallback": "original_audio"},
        )
    return _denoise_with_ffmpeg(ffmpeg, input_path, output_path, model_name, strength)


def _denoise_with_custom_command(audio_path: str, config: ExperimentConfig, model_name: str) -> PreprocessResult:
    input_path = Path(audio_path)
    output_path = _preprocessed_output_path(config, input_path, _safe_preprocess_name(model_name), "external")
    command = str(config.noise_reduction_command).format(
        input=str(input_path),
        output=str(output_path),
        strength=clamp01(config.noise_reduction_strength),
        model=model_name,
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
    except Exception as exc:
        return PreprocessResult(
            audio_path=str(input_path),
            applied=False,
            warnings=[f"External noise reduction failed: {exc}"],
            metadata={"model": model_name, "processor": "external_command", "fallback": "original_audio"},
        )
    if not output_path.exists():
        return PreprocessResult(
            audio_path=str(input_path),
            applied=False,
            warnings=[f"External noise reduction did not create {output_path}."],
            metadata={"model": model_name, "processor": "external_command", "fallback": "original_audio"},
        )
    metadata = {"model": model_name, "processor": "external_command", "strength": clamp01(config.noise_reduction_strength)}
    metadata.update(_wav_metadata(output_path))
    return PreprocessResult(audio_path=str(output_path), applied=True, metadata=metadata)


def _denoise_with_rnnoise(audio_path: str, config: ExperimentConfig, model_name: str) -> PreprocessResult:
    denoise = shutil.which("denoise")
    ffmpeg = ffmpeg_executable()
    if not denoise:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["RNNoise requires the pyrnnoise `denoise` command or noise_reduction_command."],
            metadata={"model": model_name, "processor": "pyrnnoise", "fallback": "original_audio"},
        )
    if not ffmpeg:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["RNNoise preprocessing requires ffmpeg to prepare 48 kHz WAV input."],
            metadata={"model": model_name, "processor": "pyrnnoise", "fallback": "original_audio"},
        )
    input_path = Path(audio_path)
    prepared_path = _convert_audio_to_pcm16_wav(input_path, config, "rnnoise-input", sample_rate=48000, channels=1)
    if prepared_path is None:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["RNNoise preprocessing could not prepare a 48 kHz WAV input."],
            metadata={"model": model_name, "processor": "pyrnnoise", "fallback": "original_audio"},
        )
    enhanced_path = _preprocessed_output_path(config, input_path, "rnnoise", "enhanced")
    output_path = _preprocessed_output_path(config, input_path, "rnnoise", "denoised")
    try:
        subprocess.run([denoise, str(prepared_path), str(enhanced_path)], check=True, capture_output=True, text=True)
    except Exception as exc:
        return PreprocessResult(
            audio_path=str(input_path),
            applied=False,
            warnings=[f"RNNoise preprocessing failed: {exc}"],
            metadata={"model": model_name, "processor": "pyrnnoise", "fallback": "original_audio"},
        )
    if not enhanced_path.exists():
        return PreprocessResult(
            audio_path=str(input_path),
            applied=False,
            warnings=[f"RNNoise did not create {enhanced_path}."],
            metadata={"model": model_name, "processor": "pyrnnoise", "fallback": "original_audio"},
        )
    return _mix_or_copy_enhanced(prepared_path, enhanced_path, output_path, config, model_name, "pyrnnoise")


def _denoise_with_deepfilternet(
    audio_path: str,
    config: ExperimentConfig,
    model_name: str,
    model_id: str,
    postfilter: bool,
) -> PreprocessResult:
    deepfilter = shutil.which("deepFilter") or shutil.which("deep-filter")
    ffmpeg = ffmpeg_executable()
    if not deepfilter:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["DeepFilterNet preprocessing requires the `deepFilter` or `deep-filter` command."],
            metadata={"model": model_name, "processor": "deepfilternet", "fallback": "original_audio"},
        )
    if not ffmpeg:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["DeepFilterNet preprocessing requires ffmpeg to prepare 48 kHz WAV input."],
            metadata={"model": model_name, "processor": "deepfilternet", "fallback": "original_audio"},
        )
    input_path = Path(audio_path)
    prepared_path = _convert_audio_to_pcm16_wav(input_path, config, "deepfilternet-input", sample_rate=48000, channels=1)
    if prepared_path is None:
        return PreprocessResult(
            audio_path=audio_path,
            applied=False,
            warnings=["DeepFilterNet preprocessing could not prepare a 48 kHz WAV input."],
            metadata={"model": model_name, "processor": "deepfilternet", "fallback": "original_audio"},
        )
    output_path = _preprocessed_output_path(config, input_path, _safe_preprocess_name(model_name), "denoised")
    out_dir = output_path.parent / f".{output_path.stem}.deepfilter"
    out_dir.mkdir(parents=True, exist_ok=True)
    command_candidates = _deepfilter_commands(deepfilter, out_dir, prepared_path, model_id, postfilter)
    last_error = ""
    enhanced_path: Path | None = None
    for command in command_candidates:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            enhanced_path = _latest_wav(out_dir)
            if enhanced_path is not None:
                break
        except Exception as exc:
            last_error = str(exc)
    if enhanced_path is None:
        return PreprocessResult(
            audio_path=str(input_path),
            applied=False,
            warnings=[f"DeepFilterNet preprocessing failed: {last_error or 'no output wav'}"],
            metadata={"model": model_name, "processor": "deepfilternet", "fallback": "original_audio"},
        )
    result = _mix_or_copy_enhanced(prepared_path, enhanced_path, output_path, config, model_name, "deepfilternet")
    result.metadata["model_id"] = model_id
    result.metadata["postfilter"] = bool(postfilter)
    return result


def _deepfilter_commands(
    executable: str,
    out_dir: Path,
    input_path: Path,
    model_id: str,
    postfilter: bool,
) -> List[List[str]]:
    name = Path(executable).name
    base_option = "--out-dir" if name == "deep-filter" else "--output-dir"
    model_option = "--model" if name == "deep-filter" else "--model-base-dir"
    base = [executable, base_option, str(out_dir)]
    if postfilter:
        base.append("--pf")
    commands: List[List[str]] = []
    if model_id != "DeepFilterNet2":
        commands.append([*base, model_option, model_id, str(input_path)])
    commands.append([*base, str(input_path)])
    return commands


def _mix_or_copy_enhanced(
    original_path: Path,
    enhanced_path: Path,
    output_path: Path,
    config: ExperimentConfig,
    model_name: str,
    processor: str,
) -> PreprocessResult:
    strength = clamp01(config.noise_reduction_strength)
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return PreprocessResult(
            audio_path=str(original_path),
            applied=False,
            warnings=[f"{processor} output requires ffmpeg for final WAV conversion."],
            metadata={"model": model_name, "processor": processor, "fallback": "original_audio"},
        )
    if strength <= 0.0:
        shutil.copyfile(original_path, output_path)
        mix_strategy = "original_only"
    elif strength >= 1.0:
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(enhanced_path),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "aresample=async=1:first_pts=0,alimiter=limit=0.98",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except Exception as exc:
            return PreprocessResult(
                audio_path=str(original_path),
                applied=False,
                warnings=[f"{processor} final WAV conversion failed: {exc}"],
                metadata={"model": model_name, "processor": processor, "fallback": "original_audio"},
            )
        mix_strategy = "enhanced_only"
    else:
        original_weight = 1.0 - strength
        enhanced_weight = strength
        filter_complex = (
            f"[0:a][1:a]amix=inputs=2:duration=first:weights='{original_weight:.6f} {enhanced_weight:.6f}':"
            "normalize=0,alimiter=limit=0.98,aresample=async=1:first_pts=0[a]"
        )
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(original_path),
            "-i",
            str(enhanced_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[a]",
            "-acodec",
            "pcm_s16le",
            "-f",
            "wav",
            str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except Exception as exc:
            return PreprocessResult(
                audio_path=str(original_path),
                applied=False,
                warnings=[f"{processor} alpha mix failed: {exc}"],
                metadata={"model": model_name, "processor": processor, "fallback": "original_audio"},
            )
        mix_strategy = "alpha_mix"
    metadata = {
        "model": model_name,
        "processor": processor,
        "strength": strength,
        "mix_strategy": mix_strategy,
        "enhanced_audio_path": str(enhanced_path),
        "prepared_input_path": str(original_path),
        "output_format": "wav_pcm_s16le",
    }
    metadata.update(_wav_metadata(output_path))
    return PreprocessResult(audio_path=str(output_path), applied=output_path.exists(), metadata=metadata)


def _denoise_with_ffmpeg(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    model_name: str,
    strength: float,
) -> PreprocessResult:
    noise_reduction_db = 4.0 + (12.0 * strength)
    noise_floor = -60.0 + (18.0 * strength)
    gain_smooth = int(8 + (24 * strength))
    audio_filter = (
        f"afftdn=nr={noise_reduction_db:.1f}:nf={noise_floor:.1f}:gs={gain_smooth},"
        "aresample=async=1:first_pts=0"
    )
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        audio_filter,
        "-acodec",
        "pcm_s16le",
        "-f",
        "wav",
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
    metadata = {
        "model": model_name,
        "processor": "ffmpeg_afftdn",
        "strength": strength,
        "noise_reduction_db": noise_reduction_db,
        "noise_floor_db": noise_floor,
        "gain_smooth": gain_smooth,
        "output_format": "wav_pcm_s16le",
    }
    metadata.update(_wav_metadata(output_path))
    return PreprocessResult(audio_path=str(output_path), applied=True, metadata=metadata)


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
    output_path = _preprocessed_output_path(config, input_path, "normalized")
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
        requested_factor = 1.0 + ((target_rms / float(rms)) - 1.0) * strength
        peak_abs = _pcm16_peak_abs(frames)
        peak_limit = int(32767 * 0.98)
        peak_limited = False
        factor = requested_factor
        if peak_abs > 0:
            max_factor_without_clipping = peak_limit / float(peak_abs)
            if factor > max_factor_without_clipping:
                factor = max_factor_without_clipping
                peak_limited = True
        normalized, clipped_samples = _pcm16_mul(frames, factor)
        with wave.open(str(output_path), "wb") as writer:
            writer.setparams(params)
            writer.writeframes(normalized)
        warnings = ["Volume normalization gain was peak-limited to avoid clipping."] if peak_limited else []
        metadata = {
            "model": "volume_normalization",
            "input_rms": rms,
            "input_peak": peak_abs,
            "target_rms": target_rms,
            "target_dbfs": float(config.volume_target_dbfs),
            "strength": strength,
            "requested_gain_factor": requested_factor,
            "gain_factor": factor,
            "peak_limited": peak_limited,
            "peak_limit": peak_limit,
            "clipped_samples": clipped_samples,
            "converted_input_path": str(input_path) if str(input_path) != audio_path else "",
        }
        metadata.update(_wav_metadata(output_path))
        return PreprocessResult(audio_path=str(output_path), applied=True, warnings=warnings, metadata=metadata)
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


def _convert_audio_to_pcm16_wav(
    input_path: Path,
    config: ExperimentConfig,
    tag: str,
    sample_rate: int | None = None,
    channels: int | None = None,
) -> Path | None:
    ffmpeg = ffmpeg_executable()
    if not ffmpeg:
        return None
    output_path = _preprocessed_output_path(config, input_path, tag)
    audio_filter = "aresample=async=1:first_pts=0"
    if sample_rate:
        audio_filter += f",aresample={int(sample_rate)}"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        audio_filter,
    ]
    if channels:
        command.extend(["-ac", str(int(channels))])
    command.extend(["-acodec", "pcm_s16le", "-f", "wav", str(output_path)])
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return output_path if output_path.exists() else None


def ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _wav_metadata(path: Path) -> Dict[str, Any]:
    try:
        with wave.open(str(path), "rb") as reader:
            frames = reader.getnframes()
            rate = reader.getframerate()
            return {
                "channels": reader.getnchannels(),
                "sample_width": reader.getsampwidth(),
                "sample_rate": rate,
                "frames": frames,
                "duration_seconds": frames / float(rate) if rate else 0.0,
            }
    except Exception:
        return {}


def _safe_preprocess_name(value: str) -> str:
    normalized = (value or "preprocess").lower().replace("-", "_")
    return "".join(char if char.isalnum() or char == "_" else "_" for char in normalized)


def _preprocessed_output_path(config: ExperimentConfig, input_path: Path, *parts: str) -> Path:
    output_dir = Path(config.output_dir) / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_preprocess_name(input_path.stem)
    suffix = ".".join(_safe_preprocess_name(part) for part in parts if part)
    token = uuid4().hex[:10]
    return output_dir / f"{stem}.{suffix}.{token}.wav"


def _preprocess_cache_path(audio_path: str, config: ExperimentConfig, plan: List[Dict[str, str]]) -> Path:
    payload = {
        "audio_sha256": file_sha256(audio_path),
        "plan": plan,
        "noise_reduction_model": config.noise_reduction_model,
        "noise_reduction_command": config.noise_reduction_command,
        "noise_reduction_strength": clamp01(config.noise_reduction_strength),
        "volume_normalization_strength": clamp01(config.volume_normalization_strength),
        "volume_target_dbfs": float(config.volume_target_dbfs),
    }
    return cache_json_path(config.cache_dir, "preprocess", stable_json_hash(payload))


def _preprocess_result_from_dict(payload: Dict[str, Any]) -> PreprocessResult:
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    return PreprocessResult(
        audio_path=str(payload.get("audio_path") or ""),
        applied=bool(payload.get("applied")),
        warnings=[str(item) for item in warnings],
        metadata=metadata,
        steps=[item for item in steps if isinstance(item, dict)],
    )


def _latest_wav(directory: Path) -> Path | None:
    paths = [path for path in directory.rglob("*.wav") if path.is_file()]
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


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


def _pcm16_peak_abs(frames: bytes) -> int:
    samples = _pcm16_samples(frames)
    if not samples:
        return 0
    return max(abs(sample) for sample in samples)


def _pcm16_samples(frames: bytes) -> array:
    samples = array("h")
    samples.frombytes(frames)
    if samples.itemsize != 2:
        raise RuntimeError("array('h') is not 16-bit on this platform.")
    if os.sys.byteorder != "little":
        samples.byteswap()
    return samples
