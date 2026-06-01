#!/usr/bin/env bash
set -euo pipefail

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
POST_MODEL="${POST_MODEL:-Qwen/Qwen3.5-9B}"
LANES="${LANES:-0:1:18000:18001,2:3:18002:18003}"
ASR_MAX_MODEL_LEN="${ASR_MAX_MODEL_LEN:-65536}"
POST_MAX_MODEL_LEN="${POST_MAX_MODEL_LEN:-8192}"
POST_MAX_NUM_SEQS="${POST_MAX_NUM_SEQS:-8}"
POST_MAX_BATCHED_TOKENS="${POST_MAX_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

for libdir in "${CONDA_PREFIX:-}/lib/python"*/site-packages/nvidia/cu13/lib; do
  if [[ -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
  fi
done

start_asr() {
  local gpu="$1"
  local port="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -m asrpostprocessing.qwen_asr_serve_compat "$ASR_MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$ASR_MAX_MODEL_LEN"
}

start_post() {
  local gpu="$1"
  local port="$2"
  CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$POST_MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --dtype bfloat16 \
    --max-model-len "$POST_MAX_MODEL_LEN" \
    --language-model-only \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$POST_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$POST_MAX_BATCHED_TOKENS"
}

start_all() {
  local pids=()
  IFS=',' read -ra lane_specs <<< "$LANES"
  for lane in "${lane_specs[@]}"; do
    IFS=':' read -r asr_gpu post_gpu asr_port post_port <<< "$lane"
    start_asr "$asr_gpu" "$asr_port" &
    pids+=("$!")
    start_post "$post_gpu" "$post_port" &
    pids+=("$!")
  done
  trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM EXIT
  wait "${pids[@]}"
}

case "${1:-all}" in
  asr-a) start_asr "${ASR_GPU:-0}" "${ASR_PORT:-18000}" ;;
  post-a) start_post "${POST_GPU:-1}" "${POST_PORT:-18001}" ;;
  asr-b) start_asr "${ASR_GPU:-2}" "${ASR_PORT:-18002}" ;;
  post-b) start_post "${POST_GPU:-3}" "${POST_PORT:-18003}" ;;
  all|parallel) start_all ;;
  *)
    echo "Usage: $0 {all|parallel|asr-a|post-a|asr-b|post-b}" >&2
    echo "LANES format: asr_gpu:post_gpu:asr_port:post_port[,..]" >&2
    exit 2
    ;;
esac
