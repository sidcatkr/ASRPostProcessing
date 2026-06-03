from __future__ import annotations

from types import SimpleNamespace


def _install_torchaudio_info_compat() -> None:
    try:
        import torchaudio  # type: ignore
    except Exception:
        return
    if hasattr(torchaudio, "info"):
        return

    def info(filepath, *args, **kwargs):  # noqa: ANN001, ANN202 - mirrors torchaudio's dynamic API.
        try:
            import soundfile as sf  # type: ignore

            metadata = sf.info(str(filepath))
            return SimpleNamespace(
                sample_rate=int(metadata.samplerate),
                num_frames=int(metadata.frames),
                num_channels=int(metadata.channels),
                bits_per_sample=0,
                encoding=str(metadata.format or "UNKNOWN"),
            )
        except Exception:
            import wave

            with wave.open(str(filepath), "rb") as handle:
                return SimpleNamespace(
                    sample_rate=int(handle.getframerate()),
                    num_frames=int(handle.getnframes()),
                    num_channels=int(handle.getnchannels()),
                    bits_per_sample=int(handle.getsampwidth()) * 8,
                    encoding="PCM_S",
                )

    torchaudio.info = info  # type: ignore[attr-defined]


_install_torchaudio_info_compat()
