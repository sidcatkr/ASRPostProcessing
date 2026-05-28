# 한국어 대화 ASR 결과의 후처리 기반 품질 개선 및 오류 분석

## Overview

 

## Motivation

회의, 강의, 연구실 미팅과 같은 실제 대화 환경에서는 일반 문장뿐만 아니라 다음과 같은 표현이 자주 섞인다.

- LLM
- RAG
- Fine-tuning
- LoRA
- Docker
- containerd
- Claude Code
- Qwen
- Gemma
- ASR / STT
- GPU 모델명
- 프로젝트명 및 고유명사

이러한 단어들은 한국어 문장 안에서 발음되거나 한글식으로 섞여 말해질 때 ASR 결과에서 잘못 변환될 가능성이 있다. 예를 들어 `Claude Code`가 `클러드 코드`로, `containerd`가 `컨테이너 디`로, `nerdctl`이 `너드씨티엘`처럼 인식될 수 있다.

실제로 이산수학 강의에서 `boolean`을 `불련`, `for 문`을 `포물`이라고 인식한 사례를 직접 경험하였다.

본 프로젝트는 이러한 오류를 단순히 수동으로 수정하는 것이 아니라, LLM과 RAG, search tools 등을 활용하여 자동 후처리 파이프라인으로 개선할 수 있는지 확인한다.

## Research Goal

본 프로젝트의 최종 목표는 특정 기술 용어만을 정확히 맞추는 것이 아니라, 한국어 대화 transcript 전체의 품질을 개선할 수 있는 후처리 구조를 실험하는 것이다.

주요 목표는 다음과 같다.

1. 한국어 대화 음성에 대해 baseline ASR 결과를 생성한다.
2. ASR 결과에서 오류가 발생하는 유형을 분석한다.
3. LLM 기반 후처리 시스템을 이용해 transcript를 교정한다.
4. 필요할 경우 glossary, RAG, search tool 등을 활용해 모호한 표현을 보정한다.
5. 원본 ASR 결과와 후처리 결과의 CER/WER 변화를 비교한다.
6. 후처리 과정에서 발생할 수 있는 hallucination 및 오수정 사례를 분석한다.
7. 결과를 사람이 이해하기 쉬운 형태로 시각화한다.

## Research Questions

본 프로젝트에서 확인하고자 하는 질문은 다음과 같다.

1. LLM 기반 후처리는 한국어 ASR 결과의 CER/WER을 낮출 수 있는가?
2. 단순 glossary 기반 교정과 LLM 기반 교정은 어떤 차이를 보이는가?
3. RAG 또는 search tool을 결합하면 고유명사 및 기술 용어 교정에 도움이 되는가?
4. LLM 후처리는 원문 의미를 유지하면서 오류만 수정할 수 있는가?
5. 후처리 과정에서 hallucination이나 과도한 재작성은 얼마나 발생하는가?

## System Architecture

전체 파이프라인은 다음과 같이 구성된다.

```text
Audio Input
    ↓
ASR Model
    ↓
Raw Transcript
    ↓
Chunking
    ↓
Glossary / RAG / Search Tool
    ↓
LLM Post-Processor
    ↓
Corrected Transcript
    ↓
Evaluation
    ↓
Visualization
````

## Pipeline

### 1. ASR

음성 입력을 ASR 모델에 전달하여 raw transcript를 생성한다.

초기 실험에서는 Qwen ASR 계열 모델을 사용하며, 필요에 따라 Whisper 또는 다른 ASR 모델과 비교할 수 있다.

### 2. Chunking

긴 transcript를 한 번에 후처리하면 LLM이 문맥을 과도하게 재작성하거나 hallucination을 발생시킬 가능성이 있다. 따라서 transcript를 일정 길이의 chunk로 나누어 처리한다.

예상 chunk 단위는 다음과 같다.

* 문장 단위
* 30초 ~ 90초 단위
* 일정 글자 수 단위

### 3. Glossary Matching

ASR 결과에 등장할 수 있는 기술 용어, 고유명사, 프로젝트명을 glossary로 관리한다.

예시:

```json
[
  {
    "term": "Claude Code",
    "aliases": ["클로드 코드", "클러드 코드"],
    "description": "Claude 기반 AI coding agent"
  },
  {
    "term": "containerd",
    "aliases": ["컨테이너디", "컨테이너 디", "컨테이너드"],
    "description": "container runtime"
  },
  {
    "term": "nerdctl",
    "aliases": ["너드씨티엘", "너드 시티엘"],
    "description": "containerd compatible CLI"
  }
]
```

### 4. LLM Post-Processing

LLM은 raw transcript와 glossary 또는 retrieved context를 입력받아 교정된 transcript를 생성한다.

후처리 모델의 규칙은 다음과 같다.

* 원문의 의미를 바꾸지 않는다.
* 명백한 ASR 오류만 수정한다.
* 없는 내용을 추가하지 않는다.
* 불확실한 경우 원문을 유지한다.
* 수정한 항목과 이유를 함께 기록한다.
* 결과는 구조화된 JSON 형태로 출력한다.

예상 출력 형식:

```json
{
  "corrected_text": "Claude Code로 Fine-tuning하고 containerd를 nerdctl로 실행했습니다.",
  "edits": [
    {
      "before": "클러드 코드",
      "after": "Claude Code",
      "reason": "glossary와 문맥상 일치하는 기술 용어",
      "confidence": 0.92
    }
  ],
  "risk": "low"
}
```

### 5. Evaluation

원본 ASR 결과와 후처리 결과를 reference transcript와 비교하여 품질 변화를 측정한다.

사용할 수 있는 지표는 다음과 같다.

* CER(Character Error Rate)
* WER(Word Error Rate)
* 수정 개수
* 정답 수정률
* 오수정률
* hallucination 발생 사례
* latency

한국어는 띄어쓰기 기준이 불안정할 수 있으므로, WER뿐만 아니라 CER을 주요 지표로 사용한다.

### 6. Visualization

후처리 결과를 사람이 쉽게 확인할 수 있도록 시각화한다.

예상 시각화 항목:

* Raw transcript
* Corrected transcript
* Diff view
* CER/WER 변화
* 오류 유형별 분포
* 수정 성공/실패 사례
* hallucination 사례
* 처리 시간

## Experiment Design

비교 실험은 다음 방식으로 진행한다.

```text
A. Raw ASR output
B. Glossary-only correction
C. LLM-only correction
D. RAG + LLM correction
E. Search tool + LLM correction
```

각 방식에 대해 CER/WER, 오류 유형, latency, hallucination 여부를 비교한다.

## Models

### ASR Model

초기 ASR 모델은 Qwen ASR 계열 모델을 사용할 예정이다.

### Post-Processing Model

후처리 LLM은 별도 serving 없이 `transformers`로 직접 로딩해 실험한다. 현재 lab 서버의 Quadro RTX 5000 환경에서는 서버 최적화보다 재현 가능한 단일 프로세스 실행이 더 중요하므로, 초기 실험은 PyTorch + Transformers 기준으로 진행한다.

초기 후보 모델:

* Qwen3.5-9B
* Qwen2.5-14B-Instruct
* EXAONE-3.5-7.8B-Instruct
* 기타 local LLM

## Hardware Environment

현재 실험 서버 환경은 다음과 같다.

```text
GPU: Quadro RTX 5000 16GB × 2
CUDA: 13.0
Runtime: Python + PyTorch + Transformers
Package Manager: uv
Serving: None, single-process local inference
```

기본 실험에서는 ASR과 후처리를 분리하여 실행한다.

```text
Mode 1:
GPU 0 → ASR
GPU 1 → Post-processing LLM

Mode 2:
GPU 0,1 → Post-processing LLM batch inference with device_map=auto
```

## Transformers Inference

샘플 후처리 스크립트는 모델을 직접 로딩하고, prompt를 chat template으로 변환한 뒤 JSON 후처리 결과를 생성한다.

빠른 smoke test는 모델을 로딩하지 않는 glossary backend로 실행한다.

```bash
.venv/bin/python examples/sample_postprocess.py --backend glossary --limit 2 --strict
```

실제 LLM 후처리 실험은 다음처럼 실행한다.

```bash
.venv/bin/python examples/sample_postprocess.py --backend transformers --limit 1
```

`Qwen/Qwen3.5-9B`가 GPU 메모리에 빡빡하면 더 작은 모델을 지정한다.

```bash
.venv/bin/python examples/sample_postprocess.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --backend transformers \
  --limit 1
```

결과를 JSONL로 저장하려면 다음과 같이 실행한다.

```bash
.venv/bin/python examples/sample_postprocess.py \
  --backend transformers \
  --limit 2 \
  --output outputs/sample_postprocess.jsonl
```

모델 호출 없이 prompt payload만 확인한다.

```bash
.venv/bin/python examples/sample_postprocess.py --dry-run
```

기존 테스트 엔트리포인트도 같은 샘플을 실행한다.

```bash
.venv/bin/python src/test.py --backend glossary --limit 1
```

## Expected Output

최종적으로 다음과 같은 결과물을 만드는 것을 목표로 한다.

1. 한국어 대화 ASR raw transcript
2. LLM/RAG 기반 corrected transcript
3. 수정 항목 JSON log
4. CER/WER 비교 결과
5. 오류 유형 분석
6. hallucination 및 오수정 사례 분석
7. 결과 시각화 화면

## Project Scope

본 단기 프로젝트에서는 ASR 모델 자체를 처음부터 학습하는 것보다, 기존 ASR 결과에 후처리 시스템을 결합하여 품질을 개선하는 방식에 집중한다.

시간이 허용될 경우 다음 확장을 고려한다.

* ASR model fine-tuning
* LoRA 기반 ASR adaptation
* 여러 ASR 모델 비교
* 여러 post-processing LLM 비교
* RAG/search tool 기반 자동 용어 보정
* 실시간 회의 transcript 후처리

## Limitations

LLM 기반 후처리는 transcript 품질을 개선할 가능성이 있지만, 다음과 같은 위험이 있다.

* 원문 의미 변경
* 없는 내용 추가
* 과도한 문장 재작성
* 고유명사 오수정
* reference transcript 부족
* chunk 경계로 인한 문맥 손실
* latency 증가

따라서 본 프로젝트에서는 단순히 결과가 자연스러워졌는지뿐만 아니라, 실제 CER/WER이 개선되었는지와 hallucination이 발생했는지를 함께 분석한다.

## Future Work

향후에는 다음 방향으로 확장할 수 있다.

* 실제 회의 녹음 데이터 기반 평가
* domain-specific glossary 자동 생성
* RAG 기반 고유명사 검색 시스템 고도화
* ASR confidence score와 LLM 후처리 결합
* streaming ASR 후처리
* local LLM 기반 privacy-preserving transcript correction
* fine-tuned ASR model과 post-processing pipeline 비교

## Summary

본 프로젝트는 한국어 대화 ASR 결과의 품질을 개선하기 위해 LLM, RAG, glossary, search tool 기반 후처리 구조를 실험한다.

핵심 목표는 ASR 모델 하나의 성능만을 비교하는 것이 아니라, 실제 회의나 강의 환경에서 생성된 transcript를 사람이 더 정확하게 읽고 활용할 수 있도록 만드는 후처리 파이프라인을 설계하고 평가하는 것이다.