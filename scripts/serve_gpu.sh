#!/usr/bin/env bash
set -euo pipefail

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
POST_MODEL="${POST_MODEL:-Qwen/Qwen3.5-9B}"
ASR_PORT="${ASR_PORT:-18000}"
POST_PORT="${POST_PORT:-18001}"
ASR_GPU="${ASR_GPU:-0}"
POST_GPU="${POST_GPU:-1}"
ASR_MAX_MODEL_LEN="${ASR_MAX_MODEL_LEN:-32768}"
POST_MAX_MODEL_LEN="${POST_MAX_MODEL_LEN:-2048}"
PYTHON_BIN="${PYTHON_BIN:-${ASRPP_PY:-python}}"
VLLM_BIN="${VLLM_BIN:-}"

for libdir in "${CONDA_PREFIX:-}/lib/python"*/site-packages/nvidia/cu13/lib; do
  if [[ -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
  fi
done

vllm_bin() {
  if [[ -n "$VLLM_BIN" ]]; then
    printf '%s\n' "$VLLM_BIN"
    return
  fi
  "$PYTHON_BIN" - <<'PY'
import shutil
import sys
from pathlib import Path

executable = shutil.which("vllm")
if executable:
    print(executable)
else:
    sibling = Path(sys.executable).with_name("vllm")
    print(sibling if sibling.exists() else "vllm")
PY
}

if [[ "${1:-}" == "asr" ]]; then
  CUDA_VISIBLE_DEVICES="$ASR_GPU" "$PYTHON_BIN" -m asrpostprocessing.qwen_asr_serve_compat "$ASR_MODEL" \
    --host 0.0.0.0 \
    --port "$ASR_PORT" \
    --gpu-memory-utilization 0.7 \
    --max-model-len "$ASR_MAX_MODEL_LEN" \
    --attention-backend TRITON_ATTN \
    --enforce-eager
elif [[ "${1:-}" == "post" ]]; then
  CUDA_VISIBLE_DEVICES="$POST_GPU" "$(vllm_bin)" serve "$POST_MODEL" \
    --host 0.0.0.0 \
    --port "$POST_PORT" \
    --dtype float16 \
    --max-model-len "$POST_MAX_MODEL_LEN" \
    --language-model-only \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --gpu-memory-utilization 0.6 \
    --max-num-seqs 1 \
    --max-num-batched-tokens "$POST_MAX_MODEL_LEN"
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
