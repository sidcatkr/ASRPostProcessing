#!/usr/bin/env bash
set -euo pipefail

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
POST_MODEL="${POST_MODEL:-Qwen/Qwen3.5-9B}"
LANES="${LANES:-0:1:18000:18001,2:3:18002:18003}"
STAGE_GPUS="${STAGE_GPUS:-0,1,2,3}"
STAGE_PORTS="${STAGE_PORTS:-18000,18001,18002,18003}"
ASR_MAX_MODEL_LEN="${ASR_MAX_MODEL_LEN:-65536}"
POST_MAX_MODEL_LEN="${POST_MAX_MODEL_LEN:-8192}"
POST_MAX_NUM_SEQS="${POST_MAX_NUM_SEQS:-8}"
POST_MAX_BATCHED_TOKENS="${POST_MAX_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-auto}"
GPU_MEMORY_UTILIZATION_MAX="${GPU_MEMORY_UTILIZATION_MAX:-0.90}"
GPU_MEMORY_RESERVED_MB="${GPU_MEMORY_RESERVED_MB:-256}"
VLLM_CACHE_ROOT_BASE="${VLLM_CACHE_ROOT_BASE:-outputs/model_servers_l4x4/vllm_cache}"

for libdir in "${CONDA_PREFIX:-}/lib/python"*/site-packages/nvidia/cu13/lib; do
  if [[ -d "$libdir" ]]; then
    export LD_LIBRARY_PATH="$libdir:${LD_LIBRARY_PATH:-}"
  fi
done

gpu_memory_utilization() {
  local gpu="$1"
  if [[ "$GPU_MEMORY_UTILIZATION" != "auto" && "$GPU_MEMORY_UTILIZATION" != "adaptive" ]]; then
    printf '%s\n' "$GPU_MEMORY_UTILIZATION"
    return
  fi
  python - "$gpu" "$GPU_MEMORY_UTILIZATION_MAX" "$GPU_MEMORY_RESERVED_MB" <<'PY'
import subprocess
import sys

gpu, max_value, reserved_mb = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ratios = []
for token in gpu.replace(",", " ").split():
    if not token.isdigit():
        continue
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={token}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        continue
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            total_mb = float(parts[0])
            free_mb = float(parts[1])
        except ValueError:
            continue
        if total_mb > 0:
            ratios.append(max(0.0, (free_mb - reserved_mb) / total_mb))
value = min(max_value, min(ratios)) if ratios else max_value
value = max(0.05, min(0.99, value))
print(f"{value:.4f}".rstrip("0").rstrip("."))
PY
}

cache_root_for() {
  local stage="$1"
  local gpu="$2"
  local port="$3"
  printf '%s/%s_gpu%s_port%s\n' "$VLLM_CACHE_ROOT_BASE" "$stage" "$gpu" "$port"
}

start_asr() {
  local gpu="$1"
  local port="$2"
  local gpu_memory
  local cache_root
  gpu_memory="$(gpu_memory_utilization "$gpu")"
  cache_root="$(cache_root_for asr "$gpu" "$port")"
  mkdir -p "$cache_root"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_CACHE_ROOT="$cache_root" python -m asrpostprocessing.qwen_asr_serve_compat "$ASR_MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --gpu-memory-utilization "$gpu_memory" \
    --max-model-len "$ASR_MAX_MODEL_LEN"
}

start_post() {
  local gpu="$1"
  local port="$2"
  local gpu_memory
  local cache_root
  gpu_memory="$(gpu_memory_utilization "$gpu")"
  cache_root="$(cache_root_for post "$gpu" "$port")"
  mkdir -p "$cache_root"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_CACHE_ROOT="$cache_root" vllm serve "$POST_MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --dtype bfloat16 \
    --max-model-len "$POST_MAX_MODEL_LEN" \
    --language-model-only \
    --gpu-memory-utilization "$gpu_memory" \
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

start_stage() {
  local stage="$1"
  local pids=()
  IFS=',' read -ra gpus <<< "$STAGE_GPUS"
  IFS=',' read -ra ports <<< "$STAGE_PORTS"
  if [[ "${#gpus[@]}" -ne "${#ports[@]}" ]]; then
    echo "STAGE_GPUS and STAGE_PORTS must have the same length" >&2
    exit 2
  fi
  for index in "${!gpus[@]}"; do
    if [[ "$stage" == "asr" ]]; then
      start_asr "${gpus[$index]}" "${ports[$index]}" &
    else
      start_post "${gpus[$index]}" "${ports[$index]}" &
    fi
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
  asr-stage|asr-all-gpus) start_stage asr ;;
  post-stage|post-all-gpus) start_stage post ;;
  all|parallel) start_all ;;
  *)
    echo "Usage: $0 {all|parallel|asr-a|post-a|asr-b|post-b|asr-stage|post-stage}" >&2
    echo "LANES format: asr_gpu:post_gpu:asr_port:post_port[,..]" >&2
    echo "STAGE_GPUS/STAGE_PORTS format: gpu,gpu,... and port,port,..." >&2
    exit 2
    ;;
esac
