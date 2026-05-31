# Readiness Audit Against notes.md

Status: conditionally ready for GPU-server experiments, not fully proven on the current Mac.

## Implemented

- Qwen model defaults are configured as `Qwen/Qwen3-ASR-1.7B` for ASR and `Qwen/Qwen3.5-9B` for post-processing.
- Audio input, reference transcript input, independent noise-reduction and volume-normalization toggles, Keyword Bias, LLM post-processing, RAG, Search, model selection, weights, raw/corrected transcript viewer, diff, metrics, edits, preprocess metadata, model-server status, and run status are wired through the Gradio UI.
- Server GPU/VRAM status is visible in the Gradio UI and available from `asrpp gpu`.
- Pipeline order follows `notes.md`: audio input, optional preprocess, ASR, raw transcript, chunking, LLM/RAG/Search post-process, corrected transcript, CER/WER evaluation.
- GPU model residency is configurable: `parallel` keeps ASR and post-processing servers loaded for speed, while `sequential` starts ASR, unloads it, then starts the post-processing model for lower VRAM usage.
- Keyword Bias is implemented as ASR chat prompt/context, with weight levels mapped to increasing hint strength.
- LLM post-processing is chunked and returns structured correction JSON with edit logs.
- RAG supports inline text and uploaded `.txt`, `.md`/`.markdown`, `.csv`, `.json`, and `.pdf` files, with lexical fallback and optional FAISS + sentence-transformers retrieval.
- Search is optional, supports DuckDuckGo Instant Answer by default or a custom endpoint, and caches results for reproducible experiments.
- CER/WER, raw-vs-corrected deltas, latency, and a lightweight preservation proxy are written to outputs, `metrics.tsv`, and TensorBoard event logs.
- Sweep runs notes.md research conditions A, B1, B2, B3, C, D, E, F, and G, and writes `sweep_summary.csv` plus `sweep_analysis.json` with sweet-spot, over-bias, over-RAG, and over-postprocess detection.
- `asrpp doctor` checks Python, dependencies, NVIDIA GPU presence, optional model packages, `ffmpeg` availability for compressed-audio preprocessing, output dirs, and optional vLLM endpoints.

## Not Proven On This Machine

- Real Qwen3-ASR/Qwen3.5 inference could not be executed because this machine has no NVIDIA GPU and local Python is 3.9, not the configured Python 3.12 target.
- Gradio UI could not be launched locally because `gradio` is not installed here.
- TensorBoard event writing is implemented through `torch.utils.tensorboard` when available and a direct `tensorboard` event-writer fallback otherwise.
- Noise reduction and volume normalization work internally for 16-bit PCM WAV. Compressed-audio preprocessing uses `ffmpeg` from `PATH` when conversion is needed.
- Search quality depends on the provider. DuckDuckGo Instant Answer is usable without an API key, but domain-specific technical search may need a stronger custom endpoint.

## Required GPU-Server Gate Before Real Experiments

Run these commands on the NVIDIA server after installation:

```bash
asrpp doctor --config configs/cuda.yaml --check-endpoints
asrpp doctor --config configs/cuda_sequential.yaml
PYTHONPATH=src python -m unittest discover -s tests
ASRPP_GPU_SMOKE_AUDIO=/path/to/sample.wav \
ASRPP_GPU_SMOKE_REFERENCE="$(cat /path/to/reference.txt)" \
PYTHONPATH=src python -m unittest tests.test_gpu_smoke
```

If all pass, the project is ready for controlled experiments using `asrpp ui`, `asrpp run`, and `asrpp sweep`.
