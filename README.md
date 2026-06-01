ASR (Qwen/Qwen3-ASR-1.7B)
Post Processing model (Qwen/Qwen3.5-9B)

# LLM/RAG 기반 후처리는 한국어 대화 ASR 결과의 CER/WER을 낮출 수 있는가?

1. LLM-only 후처리와 RAG-augmented 후처리의 차이는 무엇인가?
2. Keyword Bias를 ASR 단계에 함께 적용하면 후처리 성능이 더 좋아지는가?
3. Keyword Bias 또는 RAG의 강도가 과도할 때 오히려 CER/WER이 악화되는가?
4. 후처리 모델은 어느 정도까지 원문 의미를 보존할 수 있는가?

## ASR Pipeline:

    Audio Input (WAV format)
        ↓ - 이 과정에 먼저 Keyword Bias 적용해도 됨
    Pre Process - 잡음 제거, Volume Normalization... 사용할 모델: BS-RoFormer or RNNoise, etc...
    **BS-RoFormer: Band Split Rotary Position Embedding Transformer**: 회전형 위치 인코딩인 RoPE를 적용. 기존 Transformer보다 입력 간의 상대적 위치 관계를 잘 반영할 수 있어, 긴 시간 흐름을 가지는 음성 처리에 유리합니다.

        ↓
    Qwen 3 ASR
        ↓
    RAW Transcript (Output)
        ↓ - 여기에 Chunking 적용. (전체 RAW Transcript를 Post Process에 전달하면 Hallucination 등 문제가 발생할 수 있음. 따라서 일정 단위로 Chunking하여 처리.)
    Post Process (LLM, RAG, Search, etc)
        ↓
    Corrected Transcript
        ↓
    CER/WER Evaluation

## Types of Post Processing:

### LLM-Based Post Processing (Base)

RAW Transcript에서의 잘못된 발음 (불련 -> Boolean, 포물 -> for문) 을 교정한다.

### RAG(Retrieval-Augmented-Generation), 검색 증강 생성

사전에 준비된 관련 데이터 또는 사용자가 입력한 키워드 기반으로 LLM-Based Post Processing을 보조한다.

**예상 Pipeline:**

    RAW input: "클러드 코드로 포물 작성 보조"
        ↓
    LLM-Based Post Processing 적용 (LLM 모델이 잘못된 발음 교정, RAG 사용하여 사용자가 입력한 데이터에서 키워드나 단어, 지식 등을 참고하여 잘못 전사된 단어 수정, 만약 모호하거나 입력된 데이터에 없는 경우 search tool 참조하여 정확도 향상)
        ↓
    최종 출력

### 연구에서 중점적으로 볼 것

- LLM-Based Post Processing + RAG만 적용한 경우
- LLM-Based Post Processing + RAG + 사전 Keyword Bias도 함꼐 적용한 경우

이 두 가지 경우의 정확도를 비교하고 post-process 및 keyword bias 가중치를 얼마나 적용해야 정확도가 좋아지는가?

예상: LLM을 학습할 때 과적합이 발생하는 것처럼 가중치를 과도하게 적용할 시 오히려 정확도가 하락하는 현상이 발생할 것. LLM의 loss 값을 분석하여 sweet spot을 찾는 것처럼 가중치 조정을 통해 가장 정확도가 좋은 부분을 찾는 것이 중요.

## Gradio GUI

이 Lab Project는 관리 및 시각화의 용이성을 위해 Gradio GUI 및 TensorBoard를 컨트롤 및 시각화 도구로 사용한다.

Gradio GUI에 포함될 기능은 다음과 같다:

- Audio 입력: WAV 혹은 다른 format (but wav 선호)으로 Audio를 Gradio GUI 상에서 직접 녹화 혹은 PC Storage에서 파일을 업로드하는 방식으로 구현한다.
- 전처리 / 후처리 Selection: 이 연구에서의 핵심 중 하나는 전/후처리응 적용했을 떄와 그렇지 않았을 때의 CER/WER을 비교하는 것이다. 따라서 전처리 / 후처리, 구체적으로는 전처리에서 Keyword Bias(활성화된 경우 키워드 입력 창 띄워야 함.), Pre Process(잡음 제거, Volume Normalization), 후처리에서 LLM-Based Post Processing, RAG, Search tools 적용 여부 등을 각각 토글 방식으로 적용할지 말지 사용자가 결정할 수 있어야 한다.
- Model Selection: 우리가 사용 확정된 모델은 Qwen3-ASR-1.7B 모델과 Qwen3.5-9B 모델이다. 전처리 모델은 BS-Roformer, RNNoise 등 여러 종류가 있으므로 전처리 모델을 선택할 수 있게 허용해야 한다.
- 모델 가중치: 전처리, 후처리에서 적용 가중치를 개별로 세밀하게 조절할 수 있어야 한다. 가중치 정도에 따라 변화하는 정확도를 측정하는 것도 실험 목적의 일부이기 때문이다.
- RAG 입력: RAG를 사용하기 위한 데이터를 입력 및 업로드할 수 있는 창과 기능을 마련해야 한다.
- Transcription Viewer: 진행 상황 및 RAW, Processed Transcript를 볼 수 있는 창을 마련해야 한다.

## L4 x4 서버 운영

기본 L4 x4 실행 설정은 `configs/l4x4.yaml`과 `scripts/serve_l4x4.sh`를 사용한다. 기본 실행 모드는 `stage_replicas`이며, ASR stage에서는 4개 GPU가 모두 ASR replica로, post-processing stage에서는 4개 GPU가 모두 post LLM replica로 재사용된다.

    GPU 0: PRE + ASR/POST stage replica endpoint 18000
    GPU 1: PRE + ASR/POST stage replica endpoint 18001
    GPU 2: PRE + ASR/POST stage replica endpoint 18002
    GPU 3: PRE + ASR/POST stage replica endpoint 18003

Gradio UI의 primary ASR/POST GPU와 URL 입력값은 단일 서버 fallback이다. `configs/l4x4.yaml`로 UI를 띄우면 실제 stage 실행 기준은 `stage_server_base_urls`, `stage_server_gpus`, `preprocess_gpus`이다. `preprocess_gpu`는 DeepFilterNet/custom/RNNoise subprocess에 `CUDA_VISIBLE_DEVICES`로 전달되어 전처리 stage도 GPU pool에 분산된다.

`auto_experiment_saturate_lanes`가 켜져 있으면 Auto Experiment뿐 아니라 일반 Run에서도 stage/pipeline lane 수를 기준으로 ASR chunk worker, postprocess worker, condition worker를 자동 보정한다. ASR rolling context가 꺼져 있는 처리량 우선 설정에서는 audio chunk가 모든 ASR endpoint에 분산된다.

Gradio로 업로드한 큰 오디오 파일은 `upload_cache_enabled: true`일 때 `upload_cache_dir` 아래 content-addressed cache로 고정된다. 같은 파일을 다시 실행하면 Gradio 임시 업로드 경로가 아니라 cache hit 경로를 사용하므로 큰 파일 재전송, 임시 파일 삭제, 반복 실행 I/O 낭비를 줄일 수 있다.

### 서버 띄우기

이미 열려 있는 tmux session을 사용한다.

    tmux attach -t csgpu2

repo root에서 conda/env를 활성화한 뒤 UI를 실행한다. 모델 서버는 Run 또는 Auto Experiment가 시작될 때 stage별로 자동 실행된다.

    cd ~/hcilabs/ASRPostProcessing
    conda activate asrpp
    export ASRPP_PREPROCESS_VENV="$PWD/.venv-preprocess"
    export PATH="$HOME/.local/bin:$PATH"
    PYTHONPATH=src asrpp ui --config configs/l4x4.yaml --host 127.0.0.1 --port 7860

서버만 직접 확인할 때는 stage별로 하나씩 띄운다. `asr-stage`는 네 GPU에 ASR replica를 올리고, `post-stage`는 네 GPU에 post-processing replica를 올린다. 둘은 같은 port를 재사용하므로 동시에 실행하지 않는다.

    STAGE_GPUS=0,1,2,3 STAGE_PORTS=18000,18001,18002,18003 scripts/serve_l4x4.sh asr-stage
    STAGE_GPUS=0,1,2,3 STAGE_PORTS=18000,18001,18002,18003 scripts/serve_l4x4.sh post-stage

서버가 이미 떠 있는지 확인하려면 다음을 사용한다.

    PYTHONPATH=src python -m asrpostprocessing doctor --config configs/l4x4.yaml --check-endpoints

    for port in 18000 18001 18002 18003; do
      printf "metrics_%s=" "$port"
      curl -fsS --max-time 3 "http://127.0.0.1:${port}/metrics" | wc -l
    done

    nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits

### 서버 내리기

가장 안전한 방법은 실행 중인 Gradio 또는 stage script pane에서 `Ctrl-C`를 보내는 것이다. Gradio가 자동으로 띄운 stage model server는 stage가 끝나면 종료된다.

tmux 밖에서 내릴 때는 다음을 사용한다.

    tmux send-keys -t csgpu2 C-c

내린 뒤 endpoint와 GPU process를 다시 확인한다.

    for port in 18000 18001 18002 18003; do
      curl -fsS --max-time 3 "http://127.0.0.1:${port}/models" || echo "port ${port} stopped"
    done

    nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader,nounits

다른 사용자의 GPU process나 이 script가 띄우지 않은 process는 종료하지 않는다. 수동으로 띄운 model server를 정리해야 할 때도 먼저 `nvidia-smi`와 command line을 확인하고, 본인이 시작한 PID만 종료한다.
