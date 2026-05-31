#!/usr/bin/env bash
set -euo pipefail

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
POST_MODEL="${POST_MODEL:-Qwen/Qwen3.5-9B}"
ASR_PORT="${ASR_PORT:-18000}"
POST_PORT="${POST_PORT:-18001}"
ASR_GPU="${ASR_GPU:-0}"
POST_GPU="${POST_GPU:-1}"

if [[ "${1:-}" == "asr" ]]; then
  CUDA_VISIBLE_DEVICES="$ASR_GPU" qwen-asr-serve "$ASR_MODEL" \
    --host 0.0.0.0 \
    --port "$ASR_PORT" \
    --gpu-memory-utilization 0.7
elif [[ "${1:-}" == "post" ]]; then
  CUDA_VISIBLE_DEVICES="$POST_GPU" vllm serve "$POST_MODEL" \
    --host 0.0.0.0 \
    --port "$POST_PORT" \
    --dtype float16 \
    --max-model-len 8192
elif [[ "${1:-}" == "parallel" ]]; then
  "$0" asr &
  asr_pid="$!"
  "$0" post &
  post_pid="$!"
  trap 'kill "$asr_pid" "$post_pid" 2>/dev/null || true' INT TERM EXIT
  wait "$asr_pid" "$post_pid"
else
  echo "Usage: $0 {asr|post|parallel}" >&2
  exit 2
fi
