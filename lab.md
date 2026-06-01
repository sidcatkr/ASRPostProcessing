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

## 현재 구현된 기능

- Gradio GUI에서 일반 오디오 입력과 긴 오디오 파일 입력을 모두 받을 수 있다.
- Mock backend와 vLLM/OpenAI-compatible backend를 선택할 수 있다.
- Qwen3-ASR-1.7B ASR 서버와 Qwen3.5-9B 후처리 서버를 자동 시작하거나, 이미 떠 있는 서버에 연결할 수 있다.
- `vllm_chat`뿐 아니라 direct `qwen_asr_*` backend에서도 ASR audio chunking과 rolling context를 같은 방식으로 적용한다.
- 서버 구동 방식은 parallel residency와 sequential residency를 지원한다.
- 전처리 preview에서 volume normalization과 noise reduction 선택 상태를 확인할 수 있다.
- Volume normalization은 peak를 기준으로 gain을 제한해 ASR 입력 전에 clipping이 새로 생기지 않도록 한다.
- Keyword Bias, LLM 후처리, RAG, Search를 각각 독립적으로 켜고 끌 수 있다.
- 기본 balanced 후처리 강도(`postprocess_strength: 0.5`)에서도 keyword list가 제공되면 `사학 연구`/`서면 연구` 같은 가까운 ASR near-miss를 `선행 연구`처럼 등록된 keyword로 교정한다.
- RAW transcript, corrected transcript, diff, CER/WER 계열 metric, edit list, preprocess 결과, server status를 UI에서 확인할 수 있다.
- 실행 artifact에는 `asr_quality.json`이 포함되어 chunk별 길이/문자 밀도, preprocessing warning, clipping 여부, 권장 재실험 조건을 확인할 수 있다.
- CLI `asrpp asr-quality`로 같은 오디오를 여러 ASR chunk/preprocess 조건에서 비교하고 JSON 리포트를 만들 수 있다.
- Korean ASR 모드에서는 `language None<asr_text>` 같은 Qwen artifact와 중국어/CJK drift가 transcript 안에 섞여 들어온 경우 후처리 전에 제거한다.
- 제거된 language drift는 `asr_quality.json`의 `language_drift.filtered_reasons`와 chunk metadata에 기록한다.
- 실행 중 GPU/VRAM snapshot과 최근 진행 이벤트를 표시한다.
- ASR 요청에는 post-processing 요청과 별도의 timeout(`asr_request_timeout_s`)을 적용한다.
- 긴 오디오는 ASR 전 단계에서 audio chunk로 나누어 vLLM ASR endpoint에 순차 요청할 수 있다.
- chunked ASR에서는 이전 chunk transcript의 최근 일부를 다음 요청에 rolling context로 넣어 긴 발화의 문맥 단절을 줄인다.

## ASR Audio Chunking 구현

ASR audio chunking은 RAW transcript 후처리 chunking과 다른 단계이다. RAW transcript chunking은 후처리 LLM의 hallucination과 context overload를 줄이기 위한 텍스트 단계이고, ASR audio chunking은 긴 audio payload가 vLLM ASR 서버에서 timeout 또는 context 문제를 만들지 않도록 ASR 요청 전에 audio 자체를 나누는 단계이다.

지원하는 전략은 다음과 같다:

- `asr_chunking_strategy: none`: 긴 오디오도 하나의 ASR 요청으로 보낸다. baseline 비교용이다.
- `asr_chunking_strategy: fixed`: `asr_chunk_seconds` 단위로 ffmpeg segment를 만든다. 30초, 60초, 120초 fixed chunk 실험에 사용한다.
- `asr_chunking_strategy: silence`: ffmpeg `silencedetect`로 무음 구간을 찾고 speech boundary를 우선 보존해서 chunk를 만든다. 실패하거나 무음 구간을 찾지 못하면 fixed chunking으로 fallback한다.

주요 설정값은 다음과 같다:

- `asr_request_timeout_s`: ASR endpoint 전용 timeout. 기본값은 300초이다.
- `asr_chunk_seconds`: ASR chunk 최대 길이. 기본값은 120초이다. `/tmp` 문제 샘플의 첫 120초 비교에서 30초 chunk보다 120초 요청이 오인식 후보와 지연 시간이 모두 적었다.
- `asr_chunk_padding_seconds`: silence-aware chunk 양끝에 붙일 padding. 기본값은 0.5초이다.
- `asr_silence_threshold_db`: 무음 판정 dB threshold. 기본값은 -35dB이다.
- `asr_min_silence_seconds`: 무음으로 인정할 최소 길이. 기본값은 0.6초이다.
- `asr_context_chars`: 다음 ASR chunk 요청에 참고용으로 넣을 이전 transcript의 최근 문자 수. 기본값은 240자이며 0이면 비활성화된다.

chunked ASR 결과는 전체 transcript text로 합쳐지고, 각 chunk는 `TranscriptSegment`로 보존된다. Segment metadata에는 chunk index, audio path, chunk method, speech start/end, rolling context 길이, 원 ASR 응답 metadata가 들어간다. 따라서 CER/WER뿐 아니라 어떤 chunk 전략과 문맥 조건에서 오류가 생겼는지도 추적할 수 있다.

## 기존 방식 대비 발전점

기존 방식은 긴 오디오를 하나의 ASR 요청으로 보내거나, 단순 fixed-duration audio chunk로 자르는 방식이었다. 이 방식은 구현은 단순하지만 다음 문제가 있었다:

- 긴 payload에서 vLLM ASR endpoint timeout이 발생하기 쉽다.
- fixed chunk boundary가 말 중간을 자르면 단어 누락이나 반복 전사가 생길 수 있다.
- chunk 길이와 timeout을 실험 조건으로 분리해서 비교하기 어렵다.
- chunk metadata가 부족해서 어떤 구간에서 품질이 떨어졌는지 추적하기 어렵다.

현재 방식은 다음 점이 개선되었다:

- baseline(`none`), fixed 30초, fixed 60초, fixed 120초, silence-aware/VAD-style 120초를 같은 코드 경로에서 비교할 수 있다.
- OpenAI-compatible vLLM endpoint와 direct qwen-asr package backend 모두에서 같은 chunk/context 조건을 비교할 수 있다.
- silence-aware 전략은 가능한 한 무음 지점에서 chunk를 나누므로 말 중간 절단 위험을 줄인다.
- padding을 추가해 chunk boundary 근처 음성이 잘리는 문제를 완화한다.
- 이전 chunk transcript를 bounded rolling context로 전달해 강의식 장문 오디오에서 주제와 문장 흐름이 끊기는 문제를 완화한다.
- 전처리에서 볼륨을 키울 때 peak-limited gain을 사용해 clipped sample이 ASR 품질을 망치는 위험을 줄인다.
- ASR 결과에 중국어/Han drift가 섞여도 후처리 LLM으로 넘기기 전에 제거하므로, `/tmp/processed.txt`처럼 외국어 artifact가 그럴듯한 한국어 문장으로 번역되는 위험을 줄인다.
- keyword-guided near-miss 보정을 기본 balanced 강도에서 적용해 `/tmp/processed.txt`에 남아 있던 `사학 연구` 같은 명확한 domain-term 오인식이 후처리 뒤에도 그대로 남는 문제를 줄인다.
- silence detection이 실패해도 fixed chunking으로 fallback하므로 실험 실행이 중단되지 않는다.
- ASR 전용 timeout을 둬서 후처리 LLM timeout과 ASR timeout을 별도로 조정할 수 있다.
- UI에서 chunking strategy, chunk length, padding, silence threshold, minimum silence, ASR timeout, rolling context 길이를 직접 바꿀 수 있다.
- chunk별 metadata가 남기 때문에 오류 분석과 재현이 쉬워진다.
- `asr_quality.json`을 별도 artifact로 남겨 `/tmp/raw.txt`처럼 transcript만 남은 상황에서도 preprocess와 chunk 조건을 추적할 수 있다.
- `asrpp asr-quality --sample-seconds 120 --chunk-seconds 30 --chunk-seconds 60 --chunk-seconds 120`처럼 `/tmp` 문제 구간만 잘라 비교할 수 있다.

## 실험 비교 축

권장 비교 조건은 다음과 같다:

- No audio chunk: `asr_chunking_strategy=none`
- Fixed 30s: `asr_chunking_strategy=fixed`, `asr_chunk_seconds=30`
- Fixed 60s: `asr_chunking_strategy=fixed`, `asr_chunk_seconds=60`
- Fixed 120s: `asr_chunking_strategy=fixed`, `asr_chunk_seconds=120`
- Silence-aware 120s: `asr_chunking_strategy=silence`, `asr_chunk_seconds=120`
- Rolling context off/on: `asr_context_chars=0`과 기본값 `asr_context_chars=240`

이 조건들을 같은 audio, 같은 keyword/RAG/post-process 설정에서 비교하면 audio chunking과 rolling context가 CER/WER과 timeout 안정성에 주는 영향을 분리해서 볼 수 있다.
