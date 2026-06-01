# Gradio UI 기반 ASR 후처리 실험 보고서

## 1. 실험 목적

본 실험의 목적은 README에 정의된 연구 질문을 Gradio UI와 L4 x4 서버 환경에서 재현 가능하게 검증하는 것이다.

핵심 질문은 다음과 같다.

1. LLM-only 후처리와 RAG-augmented 후처리 중 어느 조건이 한국어 대화 ASR의 CER/WER을 더 낮추는가?
2. Keyword Bias를 ASR 단계에 함께 적용하면 후처리 성능이 개선되는가?
3. Keyword Bias, RAG, Search, noise reduction, volume normalization의 강도가 과하면 오히려 오류가 늘어나는가?
4. 후처리 모델은 raw ASR의 의미를 보존하면서 발음 기반 오인식을 얼마나 안정적으로 교정하는가?

실험은 수동 Run과 Auto Experiment를 함께 사용한다. 수동 Run은 기준 조건과 개별 기능의 동작을 확인하는 데 사용하고, Auto Experiment는 여러 조합을 같은 오디오와 reference에서 한 번에 비교하는 데 사용한다.

## 2. 실험 환경

### 2.1 모델

- ASR: `Qwen/Qwen3-ASR-1.7B`
- Post-processing LLM: `Qwen/Qwen3.5-9B`
- Noise reduction 후보: `afftdn`, `RNNoise`, `DeepFilterNet2`, `DeepFilterNet2-PF`, `DeepFilterNet3`, `BS-RoFormer`
- RAG embedding 기본값: `intfloat/multilingual-e5-base`

### 2.2 L4 x4 lane 구성

기본 설정 파일은 `configs/l4x4.yaml`이다.

| Lane | Preprocess GPU | ASR GPU / endpoint | Post GPU / endpoint |
| --- | --- | --- | --- |
| lane_a | GPU 1 | GPU 0 / `http://127.0.0.1:18000/v1` | GPU 1 / `http://127.0.0.1:18001/v1` |
| lane_b | GPU 3 | GPU 2 / `http://127.0.0.1:18002/v1` | GPU 3 / `http://127.0.0.1:18003/v1` |

Gradio UI의 primary ASR/POST URL과 GPU 입력값은 단일 서버 fallback이다. L4 x4 병렬 실행 기준은 `pipeline_lanes`, `asr_base_urls`, `post_base_urls`, `preprocess_gpu`이다. UI에서는 `Configured pipeline lanes`에 다음과 같은 형태가 보여야 한다.

```text
lane_a: PRE GPU 1 -> ASR GPU 0 http://127.0.0.1:18000/v1 -> POST GPU 1 http://127.0.0.1:18001/v1
lane_b: PRE GPU 3 -> ASR GPU 2 http://127.0.0.1:18002/v1 -> POST GPU 3 http://127.0.0.1:18003/v1
```

## 3. 서버 준비 및 상태 확인

원격 서버에서는 이미 열려 있는 `csgpu2` tmux session을 사용한다.

```bash
tmux attach -t csgpu2
cd ~/hcilabs/ASRPostProcessing
conda activate asrpp
export ASRPP_PREPROCESS_VENV="$PWD/.venv-preprocess"
export PATH="$HOME/.local/bin:$PATH"
scripts/serve_l4x4.sh all
```

Gradio UI는 같은 config로 실행한다.

```bash
PYTHONPATH=src asrpp ui --config configs/l4x4.yaml --host 127.0.0.1 --port 7860
```

실행 후 다음 명령으로 상태를 확인한다.

```bash
for port in 18000 18001 18002 18003; do
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/v1/models" >/dev/null \
    && echo "port ${port} up" || echo "port ${port} down"
done

curl -fsS --max-time 3 http://127.0.0.1:7860/ >/dev/null && echo gradio_up

nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

정상 상태에서는 18000, 18001, 18002, 18003이 모두 up이어야 하고, 네 GPU에 `VLLM::EngineCore`가 resident 상태로 떠 있어야 한다.

## 4. Gradio UI 사용 절차

### 4.1 접속 및 기본 확인

브라우저에서 `http://127.0.0.1:7860`에 접속한다. 접속 후 먼저 다음 항목을 확인한다.

- `Configured pipeline lanes`가 lane_a/lane_b 모두 표시되는지 확인한다.
- `Primary ASR base URL`, `Primary post-processing LLM API URL`은 fallback 값이므로 L4 x4 lane이 보이면 그대로 둔다.
- `Refresh GPU status`를 눌러 현재 GPU process와 utilization을 확인한다.

### 4.2 입력 데이터

1. `Audio`에 WAV 또는 지원되는 오디오 파일을 업로드한다.
2. 가능하면 같은 오디오에 대응되는 reference transcript를 넣는다. CER/WER 비교에는 reference가 필요하다.
3. 도메인 용어가 있으면 `Keywords`에 줄바꿈 또는 쉼표로 입력한다.
4. RAG를 평가할 경우 관련 문서를 업로드하거나 `RAG inline text`에 배경 지식을 넣는다.

Reference가 없는 경우에도 raw/corrected transcript, near-miss, fallback 여부, artifact marker 등은 확인할 수 있지만 CER/WER 중심의 정량 비교는 약해진다.

### 4.3 전처리 설정

전처리 실험 축은 다음과 같다.

- `Noise reduction`: 잡음 제거 사용 여부
- `Noise reduction model`: `afftdn`, `RNNoise`, `DeepFilterNet2`, `DeepFilterNet2-PF`, `DeepFilterNet3`, `BS-RoFormer`
- `Noise reduction strength`: 잡음 제거 강도
- `Volume normalization`: 볼륨 정규화 사용 여부
- `Volume normalization strength`: 볼륨 정규화 강도
- `Volume target dBFS`: 목표 음량

`Preview preprocessed audio`를 먼저 눌러 전처리 결과를 들어본다. DeepFilterNet/custom/RNNoise subprocess는 lane의 `preprocess_gpu`를 `CUDA_VISIBLE_DEVICES`로 받아 GPU 1 또는 GPU 3에서 실행된다. 단, `afftdn`과 volume normalization은 ffmpeg/CPU 기반이므로 GPU utilization이 오르지 않을 수 있다.

전처리 결과가 cache hit이면 GPU 작업 없이 기존 파일을 즉시 재사용한다. GPU 사용 여부를 확인하려면 run artifact의 `preprocess.json`에서 `metadata.cache_hit`, `preprocess_gpu`, `cuda_visible_devices`를 확인한다.

### 4.4 ASR 설정

긴 오디오에서는 ASR chunking 설정이 중요하다.

- `ASR chunking strategy`
  - `silence`: 무음 지점 중심으로 chunk를 나눈다. 기본 권장값이다.
  - `fixed`: 지정한 초 단위로 자른다.
  - `none`: 긴 오디오를 하나의 ASR 요청으로 보낸다. baseline 비교용이다.
- `ASR chunk seconds`: chunk 최대 길이
- `ASR request timeout`: ASR endpoint 전용 timeout
- `Rolling context chars`: 이전 chunk transcript를 다음 chunk에 제공하는 문맥 길이
- `ASR chunk parallelism`: rolling context가 0일 때 ASR chunk 병렬 처리 worker 수

정확도 우선 실험은 `silence`, `asr_context_chars > 0`을 권장한다. 처리량 비교 실험은 `asr_context_chars=0`, `asr_chunk_parallelism=2` 이상으로 둔다.

### 4.5 후처리 설정

후처리 실험 축은 다음과 같다.

- `LLM postprocess`: Qwen3.5-9B 후처리 사용 여부
- `Postprocess strength`: LLM 교정 강도
- `RAG`: retrieval context 사용 여부
- `RAG strength`: RAG context 반영 강도
- `RAG top-k`: 검색할 context 개수
- `Search`: 외부 검색 사용 여부
- `Search strength`: 검색 결과 반영 강도

RAG와 Search는 LLM 후처리에 종속된 조건으로 본다. LLM 후처리가 꺼진 상태에서 RAG/Search만 켠 조합은 실험 matrix에서 valid condition으로 취급하지 않는다.

### 4.6 수동 Run

수동 Run은 한 조건을 깊게 확인할 때 사용한다.

1. Auto Experiment Mode를 끈다.
2. 전처리, Keyword Bias, LLM, RAG, Search 토글을 원하는 조건으로 설정한다.
3. `Run`을 누른다.
4. 결과 영역에서 다음을 확인한다.
   - Raw transcript
   - Corrected transcript
   - Diff
   - Metrics
   - Edits
   - Preprocess result
   - Server status
   - Run status

Run status에는 run id, model residency, pipeline lanes, ASR chunking, artifact path가 표시된다.

### 4.7 Auto Experiment

Auto Experiment는 README의 "모든 경우의 수" 실험을 자동화한다. 수동 토글은 Auto Mode에서 "실험에 포함할 축"으로 해석된다.

권장 순서는 다음과 같다.

1. `Auto Experiment Mode`를 켠다.
2. 빠른 확인은 `core_ablation`으로 시작한다.
3. 본 실험은 `full_valid`를 사용한다.
4. 최종 후보 축만 남긴 뒤 `full_strength_sweep`을 실행한다.
5. 모델 자체 비교가 필요할 때만 `Include model combinations`를 켠다.

`full_valid`는 다음 조합을 만든다.

- Pre/ASR modes: none, Keyword, Noise, Volume, Keyword+Noise, Keyword+Volume, Noise+Volume, Keyword+Noise+Volume
- Post modes: none, LLM, LLM+RAG, LLM+Search, LLM+RAG+Search
- 기본 총 40 conditions

`full_strength_sweep`은 active 축에 strength grid를 추가한다. 기본 grid는 다음과 같다.

- Keyword Bias: `0.25`, `0.5`, `0.75`, `1.0`
- Noise/Volume/Post/RAG/Search: `0.25`, `0.5`, `0.75`
- RAG top-k: `3`, `5`, `8`, `12`

## 5. 권장 실험 순서

### 5.1 Phase 0: 환경 검증

목적은 실험 전에 서버와 UI가 정상 상태인지 확인하는 것이다.

1. 네 endpoint와 Gradio가 모두 up인지 확인한다.
2. `Configured pipeline lanes`가 PRE/ASR/POST GPU를 모두 표시하는지 확인한다.
3. 작은 오디오로 수동 Run을 1회 실행한다.
4. artifact에 `preprocess.json`, `asr_quality.json`, `correction_quality.json`, `vllm_metrics.json`이 생기는지 확인한다.

### 5.2 Phase 1: Baseline

같은 audio/reference에서 아래 조건을 먼저 만든다.

| 조건 | 목적 |
| --- | --- |
| Raw ASR only | 후처리 없는 기준 CER/WER |
| Noise only | 전처리가 raw ASR에 주는 영향 |
| Volume only | 음량 정규화 영향 |
| Keyword only | ASR keyword bias 영향 |
| LLM only | 후처리 기본 효과 |
| LLM + RAG | RAG 효과 |
| LLM + Search | Search 효과 |
| LLM + RAG + Search | 외부 지식 결합 효과 |

Phase 1은 수동 Run 또는 Auto Experiment `core_ablation`으로 진행한다.

### 5.3 Phase 2: Full Valid Matrix

`full_valid`로 40개 조건을 실행한다.

목표는 각 기능이 단독 또는 결합 상태에서 baseline 대비 좋아지는지 확인하는 것이다. 특히 다음 bucket을 본다.

- baseline 대비 CER/WER 개선
- raw ASR보다 악화된 조건
- Keyword Bias 과보정 후보
- RAG/Search로 의미가 바뀐 후보
- Noise/Volume 전처리 때문에 ASR이 악화된 후보

### 5.4 Phase 3: Strength Sweep

Phase 2에서 가능성이 있는 축만 남기고 `full_strength_sweep`을 실행한다. 모든 축을 무조건 켜면 비용이 커지고 해석이 어려워진다.

예시:

- Keyword가 유효하면 keyword weight sweep
- RAG가 유효하면 RAG strength와 top-k sweep
- Noise reduction이 유효하면 DeepFilterNet/RNNoise model 및 strength sweep
- Search가 불안정하면 Search strength를 낮추거나 제외

### 5.5 Phase 4: 최종 후보 재검증

최종 후보 조건은 최소 2개 이상의 다른 오디오 subset에서 다시 실행한다.

권장 subset:

- clean speech
- noisy speech
- BGM 포함
- technical/domain terms
- code-switching
- long audio

## 6. 결과 artifact와 판정 기준

Auto Experiment 결과는 `outputs/auto-experiment-*` 아래에 저장된다.

주요 파일:

- `auto_experiment_summary.csv`: condition별 metric, latency, cache hit, GPU/VRAM peak, vLLM token delta
- `auto_experiment_analysis.json`: best/worst, baseline 대비 악화, effect summary
- `auto_experiment_conditions.json`: 생성된 condition matrix

개별 run artifact:

- `raw.txt`
- `processed.txt`
- `metrics.json`
- `preprocess.json`
- `asr_quality.json`
- `correction_quality.json`
- `vllm_metrics.json`

판정 우선순위:

1. `cer_normalized_no_space` 개선
2. `wer_eojeol` 개선
3. raw ASR 의미 보존
4. keyword near-miss 감소
5. hallucination 또는 과보정 없음
6. latency와 throughput
7. fallback 없이 정상 backend 사용

최종 후보는 baseline 대비 CER/WER이 낮고, `worse_than_baseline` 또는 over-correction bucket에 들어가지 않아야 한다.

## 7. GPU 사용률 해석

GPU가 "노는지"는 한 순간의 `GPU-Util`만으로 판단하지 않는다. 다음 세 가지를 함께 본다.

1. 네 GPU에 vLLM process가 resident인지
2. 각 endpoint의 request/token delta가 증가하는지
3. preprocess metadata에 `preprocess_gpu`가 기록되는지

현재 L4 x4 설계는 다음과 같다.

- GPU 0/2: ASR endpoint
- GPU 1/3: post-processing endpoint
- GPU 1/3: DeepFilterNet/custom/RNNoise preprocess subprocess

Auto Experiment는 ASR cache group이 준비되는 즉시 같은 group의 후처리 condition을 condition worker에 제출한다. 따라서 ASR priming 전체 완료를 기다리지 않고 ASR과 post-processing이 겹쳐 실행된다.

다만 다음 경우에는 GPU utilization이 낮게 보일 수 있다.

- 실행할 작업이 없는 idle 상태
- 첫 ASR cache group이 아직 생성되지 않은 초반 구간
- 후처리가 꺼진 condition
- 전처리가 cache hit인 경우
- `afftdn` 또는 volume normalization처럼 CPU/ffmpeg 기반 전처리를 실행하는 경우
- 마지막 tail 구간에서 남은 작업 수가 GPU 수보다 적은 경우

GPU 문제 재확인 명령:

```bash
for port in 18000 18001 18002 18003; do
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/v1/models" >/dev/null \
    && echo "port ${port} up" || echo "port ${port} down"
done

nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

전처리 GPU routing 확인:

```bash
find outputs -name preprocess.json -print | tail -5
```

`preprocess.json`에서 DeepFilterNet/custom/RNNoise를 사용한 run은 다음 metadata가 보여야 한다.

```json
{
  "preprocess_gpu": "1",
  "cuda_visible_devices": "1"
}
```

lane_b run이면 값은 `"3"`이어야 한다.

## 8. 문제 발생 시 대응

### 8.1 `18003`이 down인 경우

GPU 3 post server가 내려간 상태이다. 다른 사용자 프로세스를 죽이지 말고, 우리 post-b만 다시 띄운다.

```bash
POST_GPU=3 POST_PORT=18003 scripts/serve_l4x4.sh post-b
```

### 8.2 전처리 중 GPU 사용률이 오르지 않는 경우

다음을 먼저 확인한다.

1. 선택한 noise model이 GPU backend인지 확인한다. `afftdn`은 CPU/ffmpeg 기반이다.
2. `preprocess_cache_hit`이 true인지 확인한다. cache hit이면 GPU를 쓰지 않는다.
3. `preprocess.json`에 `preprocess_gpu`와 `cuda_visible_devices`가 있는지 확인한다.
4. DeepFilterNet/RNNoise command가 실제 설치되어 있는지 확인한다.

### 8.3 UI에서 primary GPU만 보이는 것처럼 보이는 경우

Primary GPU 입력칸은 fallback이다. 실제 L4 x4 병렬 기준은 `Configured pipeline lanes`이다. 이 JSON에 lane_a/lane_b가 모두 보여야 한다.

### 8.4 일반 Run에서 GPU 일부만 바쁜 경우

일반 Run은 한 오디오/한 조건을 처리하므로 전체 Auto Experiment만큼 모든 GPU를 지속적으로 채우지 못할 수 있다. 처리량 극대화 확인은 Auto Experiment 또는 sweep으로 판단한다.

## 9. 보고서 작성 템플릿

실험을 마친 뒤 결과 보고서는 다음 형식으로 정리한다.

```markdown
## 실험 ID

- 날짜:
- 오디오:
- Reference:
- Config:
- Run/Auto Experiment ID:

## 조건

- Auto Experiment coverage:
- Include model combinations:
- Preprocess axes:
- Postprocess axes:
- Strength grids:

## 결과 요약

- Best CER:
- Best WER:
- Best latency-quality tradeoff:
- Worse-than-baseline cases:

## 관찰

- Keyword Bias 효과:
- Noise/Volume 효과:
- LLM 효과:
- RAG/Search 효과:
- 과보정/환각 여부:

## GPU/성능

- Endpoint status:
- vLLM token delta:
- Peak GPU utilization:
- Preprocess GPU metadata:
- Cache hit ratio:

## 결론

- 채택 조건:
- 제외 조건:
- 다음 실험:
```

## 10. 완료 기준

실험 한 세트가 완료되었다고 판단하려면 다음 증거가 있어야 한다.

- 같은 audio/reference에서 baseline과 비교 조건이 모두 존재한다.
- `auto_experiment_summary.csv`와 `auto_experiment_analysis.json`이 생성되었다.
- best condition이 baseline보다 CER/WER을 개선했다.
- worse-than-baseline과 over-correction 후보를 검토했다.
- `vllm_metrics.json` 또는 summary의 vLLM delta로 endpoint 사용을 확인했다.
- GPU 0/1/2/3 process residency를 확인했다.
- DeepFilterNet/custom/RNNoise를 쓴 경우 `preprocess_gpu` metadata를 확인했다.
