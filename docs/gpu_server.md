# NVIDIA GPU Server Setup

This project is intended to run model inference on a CUDA/NVIDIA server.

## Environment

```bash
conda create -n asrpp python=3.12 -y
conda activate asrpp
pip install -U pip setuptools wheel
pip install -e ".[rag]"
pip install -U "qwen-asr[vllm]"
```

`qwen-asr[vllm]` installs the compatible vLLM/CUDA runtime used by the default model-server startup path. Install optional accelerators such as `flash-attn` only if the server GPU and Python/CUDA wheel set support them.

## Serve Models

`configs/cuda.yaml` enables `auto_start_model_servers: true`, so pressing Run in the Gradio UI will start both vLLM servers if `/v1/models` is not already ready. Logs are written to `outputs/model_servers/asr_vllm.log` and `outputs/model_servers/post_vllm.log`.

Use manual serving only when you want to pre-warm models before opening the UI, or when you need custom vLLM flags.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16
```

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --reasoning-parser qwen3 \
  --language-model-only
```

The post-processing default uses 8K context because transcript chunks are short and the tested csgpu nodes expose 16 GB RTX 5000 GPUs. Increase `--max-model-len` through `post_server_command` only when the target GPU has enough VRAM.

## Run

Before experiments, run the readiness check.

```bash
asrpp doctor --config configs/cuda.yaml --check-endpoints
```

```bash
conda run --no-capture-output -n asrpp asrpp ui --config configs/cuda.yaml --host 0.0.0.0 --port 7860
```

With the default CUDA config, the first Run click may take several minutes because it loads both `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3.5-9B`. Subsequent runs reuse the already-ready endpoints.

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
- `auto_start_model_servers` only manages external vLLM/OpenAI-compatible backends. The `qwen_asr_*` backends load through the Python package instead.
- Override `asr_server_command` or `post_server_command` when the server needs cluster-specific launch flags. Command templates may use `{model}`, `{host}`, `{port}`, `{base_url}`, `{gpu}`, and `{log_path}`.
- Configure `rnnoise_command` or `bs_roformer_command` when using those preprocessors. Command templates can use `{input}`, `{output}`, and `{strength}`.
- Search defaults to DuckDuckGo Instant Answer. Set `search_provider: endpoint` and configure `search_endpoint` if you need a stronger or internal search service.
- For timestamp-aware chunking, serve Qwen3-ASR with Qwen3-ForcedAligner and return timestamp metadata through the ASR backend.
- Search is optional and cached under `outputs/search_cache` for reproducible comparisons.
