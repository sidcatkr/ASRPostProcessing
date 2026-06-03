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


def _install_torchaudio_wav_io_compat() -> None:
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import torchaudio  # type: ignore
    except Exception:
        return

    def load(  # noqa: ANN001, ANN202 - mirrors torchaudio's dynamic API.
        filepath,
        frame_offset=0,
        num_frames=-1,
        normalize=True,
        channels_first=True,
        *args,
        **kwargs,
    ):
        try:
            import soundfile as sf  # type: ignore

            frames = None if int(num_frames or -1) < 0 else int(num_frames)
            dtype = "float32" if normalize else "int16"
            data, sample_rate = sf.read(
                str(filepath),
                start=max(0, int(frame_offset or 0)),
                frames=frames,
                dtype=dtype,
                always_2d=True,
            )
        except Exception:
            import wave

            with wave.open(str(filepath), "rb") as handle:
                sample_rate = handle.getframerate()
                channels = handle.getnchannels()
                if frame_offset:
                    handle.setpos(max(0, int(frame_offset)))
                frames_to_read = handle.getnframes() if int(num_frames or -1) < 0 else int(num_frames)
                raw = handle.readframes(frames_to_read)
                data = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
                if normalize:
                    data = data.astype("float32") / 32768.0
        tensor = torch.from_numpy(np.asarray(data))
        if channels_first and tensor.ndim == 2:
            tensor = tensor.transpose(0, 1)
        return tensor.contiguous(), int(sample_rate)

    def save(filepath, src, sample_rate, channels_first=True, *args, **kwargs):  # noqa: ANN001, ANN202
        tensor = torch.as_tensor(src).detach().cpu()
        if channels_first and tensor.ndim == 2:
            tensor = tensor.transpose(0, 1)
        array = tensor.numpy()
        try:
            import soundfile as sf  # type: ignore

            subtype = "PCM_16" if str(array.dtype) == "int16" else None
            sf.write(str(filepath), array, int(sample_rate), subtype=subtype)
        except Exception:
            import wave

            if str(array.dtype) != "int16":
                array = np.clip(array, -1.0, 1.0)
                array = (array * 32767.0).astype("<i2")
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            with wave.open(str(filepath), "wb") as handle:
                handle.setnchannels(int(array.shape[1]))
                handle.setsampwidth(2)
                handle.setframerate(int(sample_rate))
                handle.writeframes(array.astype("<i2", copy=False).tobytes())

    torchaudio.load = load  # type: ignore[assignment]
    torchaudio.save = save  # type: ignore[assignment]


_install_torchaudio_wav_io_compat()


def _force_single_process_dataloader() -> None:
    try:
        import torch.utils.data as torch_data  # type: ignore
    except Exception:
        return

    original = torch_data.DataLoader
    if getattr(original, "_asrpp_deepfilter_compat", False):
        return

    class DataLoader(original):  # type: ignore[misc, valid-type]
        _asrpp_deepfilter_compat = True

        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN204
            kwargs["num_workers"] = 0
            kwargs.pop("prefetch_factor", None)
            super().__init__(*args, **kwargs)

    torch_data.DataLoader = DataLoader


_force_single_process_dataloader()
