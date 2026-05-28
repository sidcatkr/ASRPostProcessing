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
