# Gradio UI 기반 ASR 후처리 실험 보고서

## 1. 실험 목적

본 실험의 목적은 한국어 대화 음성 ASR 결과에 전처리, Keyword Bias, LLM 후처리, RAG, Search를 적용했을 때 실제로 CER/WER이 개선되는지 확인하는 것이다.

실험에서 확인할 질문은 다음과 같다.

1. LLM 후처리는 raw ASR 결과보다 발음 오류와 문장 오류를 더 잘 교정하는가?
2. RAG를 함께 사용하면 LLM-only 후처리보다 도메인 용어와 문맥 교정이 좋아지는가?
3. Keyword Bias를 ASR 단계에 적용하면 도메인 키워드가 더 정확하게 전사되는가?
4. Noise reduction과 volume normalization은 ASR 품질을 개선하는가, 아니면 일부 음성에서는 오히려 악화시키는가?
5. Search는 RAG에 없는 외부 지식을 보완하는가, 아니면 잘못된 정보로 인해 의미 왜곡을 만드는가?
6. Keyword Bias, RAG, Search, 전처리, 후처리 강도를 높였을 때 어느 지점부터 과보정이 발생하는가?
7. 여러 기능을 함께 사용할 때 가장 안정적으로 CER/WER을 낮추는 조합은 무엇인가?

이 실험은 단순히 최고 점수 하나를 찾는 것이 아니라, 어떤 기능이 어떤 상황에서 도움이 되고 어떤 상황에서 악화 요인이 되는지 분리해 최종적으로 재현 가능한 최적 조합을 찾는 것을 목표로 한다.

## 2. 실험 방법

### 2.1 실험 데이터 준비

같은 실험 조건을 여러 종류의 음성에 반복 적용한다.

사용할 음성 유형은 다음과 같다.

- 깨끗한 한국어 대화 음성
- 잡음이 포함된 한국어 대화 음성
- 배경음 또는 BGM이 포함된 음성
- 기술 용어와 프로젝트명이 포함된 음성
- 코드 관련 발화가 포함된 음성
- 한국어와 영어가 섞인 음성
- 긴 오디오

각 오디오에는 가능한 한 reference transcript를 준비한다. Reference가 있어야 CER/WER을 기준으로 정량 비교할 수 있다.

### 2.1.1 오디오 제작용 테스트 문장

아래 문장을 그대로 읽어서 오디오를 만들고, Reference transcript에는 같은 문장을 붙여 넣는다.

```text
오늘 실험에서는 Qwen3-ASR 1.7B와 Qwen3.5 9B 후처리 모델을 함께 사용한다. GPU 0번부터 3번까지 유휴 자원 없이 병렬 처리하고, DeepFilterNet2, RNNoise, BS-RoFormer 전처리 후보를 비교한다. 키워드 바이어스는 씨에스지피유투, 하드코딩 금지, 자동 실험, 청크 병합, 알에이지 검색을 포함한다. 원문과 교정문 사이의 diff가 화면에 보여야 하며, 기준 전사문이 있으면 CER과 WER 값이 반드시 계산되어야 한다. 잡음이 있는 긴 오디오에서도 같은 문장을 반복하지 말고, 숫자 18000번 포트와 7860번 포트를 정확히 구분한다.
```

검증 포인트는 다음과 같다.

- Diff 출력이 비어 있지 않은지 확인한다.
- Reference transcript가 있을 때 CER/WER 값이 계산되는지 확인한다.
- 영어 모델명, 숫자, 포트 번호, GPU 번호가 유지되는지 확인한다.
- 전처리와 후처리 조합을 바꿔도 같은 reference 기준으로 비교한다.

### 2.2 Baseline 측정

먼저 아무 보정도 적용하지 않은 raw ASR 결과를 baseline으로 만든다.

절차:

1. Gradio UI에 오디오를 입력한다.
2. Reference transcript를 입력한다.
3. Keyword Bias, Noise reduction, Volume normalization, LLM 후처리, RAG, Search를 모두 끈다.
4. Run을 실행한다.
5. raw transcript와 CER/WER을 baseline으로 기록한다.

이 baseline은 이후 모든 조건의 비교 기준으로 사용한다.

### 2.3 단일 기능 실험

각 기능이 독립적으로 어떤 효과를 내는지 확인한다.

실험 조건은 다음과 같다.

| 조건 | 목적 |
| --- | --- |
| Keyword Bias only | 도메인 키워드가 ASR 단계에서 더 잘 반영되는지 확인 |
| Noise reduction only | 잡음 제거가 raw ASR 정확도를 개선하는지 확인 |
| Volume normalization only | 음량 정규화가 ASR 안정성을 높이는지 확인 |
| LLM only | 후처리 모델이 발음 오류와 문장 오류를 줄이는지 확인 |
| LLM + RAG | RAG가 도메인 용어와 문맥 교정에 추가 이득을 주는지 확인 |
| LLM + Search | 검색이 외부 지식 보완에 도움이 되는지 확인 |

각 조건은 baseline과 같은 오디오, 같은 reference에서 실행한다.

기록할 항목은 다음과 같다.

- CER/WER
- baseline 대비 개선량
- 잘 고쳐진 단어와 문장
- 새로 생긴 오류
- 의미가 바뀐 문장
- 과보정 사례

### 2.4 조합 실험

단일 기능 실험 후, 실제 사용 가능성이 높은 조합을 비교한다.

주요 조합은 다음과 같다.

1. Keyword Bias + LLM
2. Keyword Bias + LLM + RAG
3. Keyword Bias + LLM + Search
4. Keyword Bias + LLM + RAG + Search
5. Noise reduction + LLM
6. Noise reduction + Keyword Bias + LLM
7. Noise reduction + Volume normalization + Keyword Bias + LLM + RAG

이 단계에서는 기능을 많이 켤수록 성능이 좋아지는지, 아니면 특정 조합에서만 좋아지는지 확인한다.

### 2.5 Gradio UI 수동 Run 절차

수동 Run은 한 조건을 자세히 확인할 때 사용한다.

절차:

1. Audio 입력에 실험할 오디오를 업로드한다.
2. Reference transcript를 입력한다.
3. Keywords에 도메인 용어를 입력한다.
4. RAG를 사용할 경우 관련 문서를 업로드하거나 inline text를 입력한다.
5. 사용할 전처리, Keyword Bias, LLM, RAG, Search 토글을 설정한다.
6. 필요한 strength 값을 조정한다.
7. Run을 실행한다.
8. Raw transcript, Corrected transcript, Diff, Metrics, Edits를 확인한다.
9. 결과를 baseline과 비교해 기록한다.

전처리를 사용하는 경우에는 Run 전에 Preview preprocessed audio를 먼저 실행해 음성이 과하게 왜곡되지 않았는지 확인한다. 다만 전처리 preview에서 듣기에 좋아도 ASR 결과가 나빠질 수 있으므로 최종 판단은 CER/WER과 corrected transcript를 기준으로 한다.

### 2.6 Auto Experiment 절차

Auto Experiment는 여러 조합을 자동으로 실행할 때 사용한다.

절차:

1. 먼저 baseline을 수동 Run으로 만든다.
2. 단일 기능 몇 개를 수동 Run으로 확인한다.
3. Auto Experiment Mode를 켠다.
4. 실험에 포함할 축의 토글을 켠다.
5. 빠른 확인은 `core_ablation`으로 실행한다.
6. 본 실험은 `full_valid`로 실행한다.
7. 좋은 후보가 좁혀지면 `full_strength_sweep`으로 강도별 성능을 비교한다.

Auto Experiment 결과에서는 다음을 확인한다.

- baseline보다 좋아진 조건
- baseline보다 나빠진 조건
- 단일 기능보다 조합이 좋은 조건
- Keyword Bias 또는 RAG가 과하게 작동한 조건
- Search로 인해 의미가 바뀐 조건
- 전처리 때문에 ASR이 악화된 조건

### 2.7 강도 비교 방법

기능의 사용 여부만 비교하지 않고 강도 변화에 따른 성능도 비교한다.

비교할 강도 축은 다음과 같다.

- Keyword Bias strength
- Noise reduction strength
- Volume normalization strength
- LLM postprocess strength
- RAG strength
- RAG top-k
- Search strength

확인할 질문은 다음과 같다.

1. 강도가 올라갈수록 CER/WER이 계속 낮아지는가?
2. 특정 강도 이후 CER/WER이 다시 높아지는가?
3. 도메인 용어는 좋아졌지만 일반 문장이 나빠지는가?
4. corrected transcript가 reference에는 가까워졌지만 원문 의미를 바꾸는가?

이 단계의 목표는 각 기능의 sweet spot을 찾는 것이다.

### 2.8 결과 비교 방법

각 실험 조건은 다음 순서로 비교한다.

1. baseline 대비 CER/WER 개선 여부를 확인한다.
2. 가장 낮은 CER/WER을 가진 조건을 1차 후보로 둔다.
3. corrected transcript를 직접 읽어 의미 보존 여부를 확인한다.
4. 도메인 용어와 keyword near-miss가 실제로 줄었는지 확인한다.
5. 과보정, hallucination, 잘못된 Search/RAG 반영 사례를 제거한다.
6. 여러 음성 유형에서 같은 개선 경향이 반복되는지 확인한다.
7. 가장 안정적으로 개선되는 조건을 최종 후보로 선택한다.

최종 결론은 "어떤 기능을 켜면 항상 좋다"가 아니라, "어떤 데이터 조건에서 어떤 기능 조합과 강도가 좋았다"는 형식으로 작성한다.
