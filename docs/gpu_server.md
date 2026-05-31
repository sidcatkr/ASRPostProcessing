# NVIDIA GPU Server Setup

This project is intended to run model inference on a CUDA/NVIDIA server.

## Environment

```bash
conda create -n asrpp python=3.12 -y
conda activate asrpp
pip install -U pip setuptools wheel
pip install -e ".[rag]"
pip install -U "qwen-asr[vllm]"
pip install -U --extra-index-url https://wheels.vllm.ai/nightly "vllm[audio]>=0.22" bitsandbytes
```

`qwen-asr[vllm]` installs the Qwen3-ASR server implementation. The app starts it through `python -m asrpostprocessing.qwen_asr_serve_compat` so the project can keep a small vLLM compatibility shim for recent vLLM builds. Install optional accelerators such as `flash-attn` only if the server GPU and Python/CUDA wheel set support them.
`Qwen/Qwen3.5-9B` requires a recent vLLM build with `Qwen3_5ForConditionalGeneration` support. The tested 16 GB RTX 5000 server also needs bitsandbytes quantization for the post-processing model to fit on one GPU.

## Model Residency Modes

Two GPU residency modes are supported because the lab server may not have enough free VRAM to keep both Qwen models loaded.

- `configs/cuda.yaml`: `model_residency: parallel`. The ASR server and post-processing LLM server are both loaded and kept ready. This is faster after warmup and is the preferred mode when both GPUs have enough free VRAM.
- `configs/cuda_sequential.yaml`: `model_residency: sequential`. The pipeline starts the ASR model for transcription, terminates that managed server to free VRAM, then starts the post-processing LLM for correction. This is slower but keeps only one managed stage model loaded at a time.

Sequential unloading only applies to model servers auto-started by this app. If you manually started servers in another shell, stop them yourself before using the low-VRAM config.

## Serve Models

`configs/cuda.yaml` and `configs/cuda_sequential.yaml` enable `auto_start_model_servers: true`, so pressing Run in the Gradio UI will start the required model server if `/v1/models` is not already ready. They use ports `18000` and `18001` by default because some shared GPU servers reserve `8000` and `8001` for JupyterHub. Logs are written to `outputs/model_servers/asr_vllm.log` and `outputs/model_servers/post_vllm.log`.

Use manual serving only when you want to pre-warm models before opening the UI, or when you need custom serving flags.
The app auto-start path and `scripts/serve_gpu.sh` add the conda NVIDIA CUDA library directory to `LD_LIBRARY_PATH` for bitsandbytes. If you run `vllm serve` by hand and see `libnvJitLink.so.13` errors, export the matching env path first:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m asrpostprocessing.qwen_asr_serve_compat Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 18000 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 32768 \
  --attention-backend TRITON_ATTN \
  --enforce-eager
```

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 18001 \
  --dtype float16 \
  --max-model-len 2048 \
  --language-model-only \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --enforce-eager \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.6 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048
```

The ASR default caps `--max-model-len` at 32768 because Qwen3-ASR advertises 65536 context by default, which requires more KV cache than the tested 16 GB RTX 5000 nodes expose at `--gpu-memory-utilization 0.7`. It also forces `TRITON_ATTN` plus eager mode because FlashInfer prefill failed on the tested RTX 5000 with CUDA invalid-argument errors during audio requests. The post-processing default uses 2K context because transcript chunks are short and the tested RTX 5000 needs 4-bit quantization plus eager mode to serve Qwen3.5 reliably. It also uses `TRITON_ATTN` because FlashInfer prefill produced CUDA invalid-argument failures for Qwen3.5 requests on the tested server. Increase `--max-model-len` or remove quantization through custom server commands only when the target GPU has enough VRAM.

For manual parallel warmup, use:

```bash
scripts/serve_gpu.sh parallel
```

## Run

Before experiments, run the readiness check.

```bash
asrpp doctor --config configs/cuda.yaml --check-endpoints
```

Check current GPU and VRAM usage from the shell:

```bash
asrpp gpu
```

```bash
conda run --no-capture-output -n asrpp asrpp ui --config configs/cuda.yaml --host 0.0.0.0 --port 7860
```

With the default CUDA config, the first Run click may take several minutes because it loads both `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3.5-9B`. Subsequent runs reuse the already-ready endpoints.
The Gradio UI also exposes a `Server GPU / VRAM status` panel with a refresh button, so you can confirm whether ports `18000` and `18001` have loaded models into VRAM.

RAG accepts inline text plus uploaded `.txt`, `.md`/`.markdown`, `.csv`, `.json`, and `.pdf` files. PDF extraction uses `pypdf` from the `rag` extra, then retrieval uses FAISS + sentence-transformers when available and the built-in lexical retriever otherwise.

When VRAM is tight, launch the UI with the sequential config instead:

```bash
conda run --no-capture-output -n asrpp asrpp ui --config configs/cuda_sequential.yaml --host 0.0.0.0 --port 7860
```

```bash
asrpp run \
  --config configs/cuda.yaml \
  --audio sample.wav \
  --reference sample_reference.txt \
  --keyword "Claude Code" \
  --keyword "Boolean" \
  --enable-keyword-bias
```

## Notes

- Keyword Bias is implemented as ASR chat prompt/context because Qwen3-ASR does not expose a documented hotword decoding parameter.
- `auto_start_model_servers` starts Qwen3-ASR through the compatibility wrapper and the post-processing model through `vllm serve`. The `qwen_asr_*` backends load through the Python package instead.
- `model_residency: parallel` keeps all required managed servers loaded. `model_residency: sequential` loads and unloads the ASR and post-processing stages one at a time.
- Override `asr_server_command` or `post_server_command` when the server needs cluster-specific launch flags. Command templates may use `{model}`, `{host}`, `{port}`, `{base_url}`, `{gpu}`, and `{log_path}`.
- Noise reduction and volume normalization are separate experimental variables. Configure `rnnoise_command` or `bs_roformer_command` when using those preprocessors. Command templates can use `{input}`, `{output}`, and `{strength}`.
- Search defaults to DuckDuckGo Instant Answer. Set `search_provider: endpoint` and configure `search_endpoint` if you need a stronger or internal search service.
- For timestamp-aware chunking, serve Qwen3-ASR with Qwen3-ForcedAligner and return timestamp metadata through the ASR backend.
- Search is optional and cached under `outputs/search_cache` for reproducible comparisons.
- WER/CER metrics are written to each run directory under `runs/<run_id>/` as both `metrics.tsv` and TensorBoard event files. Use `asrpp tensorboard --launch --logdir runs --port 6006` when you want the dashboard.
