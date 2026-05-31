#!/usr/bin/env bash
set -euo pipefail

ASR_PORT="${ASR_PORT:-8000}"
POST_PORT="${POST_PORT:-8001}"
ASR_GPU="${ASR_GPU:-0}"
POST_GPU="${POST_GPU:-1}"

if [[ "${1:-}" == "asr" ]]; then
  CUDA_VISIBLE_DEVICES="$ASR_GPU" vllm serve Qwen/Qwen3-ASR-1.7B \
    --host 0.0.0.0 \
    --port "$ASR_PORT"
elif [[ "${1:-}" == "post" ]]; then
  CUDA_VISIBLE_DEVICES="$POST_GPU" vllm serve Qwen/Qwen3.5-9B \
    --host 0.0.0.0 \
    --port "$POST_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --language-model-only
else
  echo "Usage: $0 {asr|post}" >&2
  exit 2
fi
