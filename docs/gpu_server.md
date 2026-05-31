# NVIDIA GPU Server Setup

This project is intended to run model inference on a CUDA/NVIDIA server.

## Environment

```bash
conda create -n asrpp python=3.12 -y
conda activate asrpp
pip install -e ".[rag]"
pip install -U qwen-asr[vllm]
pip install -U vllm[audio] --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
pip install -U flash-attn --no-build-isolation
```

## Serve Models

Run ASR and post-processing on separate ports so experiments can isolate ASR Keyword Bias from LLM/RAG post-processing.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-ASR-1.7B \
  --host 0.0.0.0 \
  --port 8000
```

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.5-9B \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

## Run

Before experiments, run the readiness check.

```bash
asrpp doctor --config configs/cuda.yaml --check-endpoints
```

```bash
asrpp ui --config configs/cuda.yaml --host 0.0.0.0 --port 7860
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
- Configure `rnnoise_command` or `bs_roformer_command` when using those preprocessors. Command templates can use `{input}`, `{output}`, and `{strength}`.
- Search defaults to DuckDuckGo Instant Answer. Set `search_provider: endpoint` and configure `search_endpoint` if you need a stronger or internal search service.
- For timestamp-aware chunking, serve Qwen3-ASR with Qwen3-ForcedAligner and return timestamp metadata through the ASR backend.
- Search is optional and cached under `outputs/search_cache` for reproducible comparisons.
