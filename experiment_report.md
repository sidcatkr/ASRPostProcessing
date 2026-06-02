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

아래 문장을 그대로 읽어서 오디오를 만들고, Reference transcript에는 같은 문장을 붙여 넣는다. 짧은 문장만 사용하면 CER/WER 분모가 작아져 0.n% 단위 차이가 과소평가될 수 있으므로, 최소한 일반 길이 이상을 사용하고 본 실험에서는 긴 낭독 또는 매우 긴 낭독을 사용한다.

짧은 sanity check:

```text
오늘 실험에서는 Qwen3-ASR 1.7B와 Qwen3.5 9B 후처리 모델을 함께 사용한다. GPU 0번부터 3번까지 유휴 자원 없이 병렬 처리하고, DeepFilterNet2, RNNoise, BS-RoFormer 전처리 후보를 비교한다. 키워드 바이어스는 씨에스지피유투, 하드코딩 금지, 자동 실험, 청크 병합, 알에이지 검색을 포함한다. 원문과 교정문 사이의 diff가 화면에 보여야 하며, 기준 전사문이 있으면 CER과 WER 값이 반드시 계산되어야 한다. 잡음이 있는 긴 오디오에서도 같은 문장을 반복하지 말고, 숫자 18000번 포트와 7860번 포트를 정확히 구분한다.
```

일반 길이 reference:

```text
오늘 실험에서는 Qwen3-ASR 1.7B와 Qwen3.5 9B 후처리 모델을 함께 사용한다. 목표는 raw ASR 결과에 전처리, keyword bias, LLM 후처리, RAG, Search를 적용했을 때 CER과 WER이 실제로 낮아지는지 확인하는 것이다. 서버는 GPU 0번부터 GPU 3번까지 네 장의 NVIDIA L40을 사용하며, ASR stage와 post-processing stage는 각각 모든 GPU replica를 활용한다. 전처리 후보에는 DeepFilterNet2, DeepFilterNet3, RNNoise, BS-RoFormer, FFmpeg afftdn이 포함된다. 키워드 목록에는 씨에스지피유투, 자동 실험, 하드코딩 금지, 청크 병합, 알에이지 검색, 브이엘엘엠, 포트 18000번, 포트 7860번을 넣는다. 사용자는 Gradio 화면에서 각 조건의 CER/WER뿐 아니라 raw transcript와 corrected transcript 사이의 글자 단위 diff를 펼쳐서 확인한다. 이 문장은 일반적인 연구 실험 설명 길이에 맞춰 작성되었으며, 숫자, 영어 모델명, 한국어 기술 용어, 발음이 헷갈리는 외래어를 함께 포함한다.
```

긴 낭독 reference:

```text
이번 실험의 핵심 목적은 한국어 대화 음성에서 발생하는 ASR 오류를 체계적으로 분석하고, 전처리와 후처리 조합이 실제 품질 개선으로 이어지는지 검증하는 것이다. 먼저 baseline 조건에서는 keyword bias, noise reduction, volume normalization, LLM post-processing, RAG, Search를 모두 끄고 raw transcript만 생성한다. 그 다음 keyword bias only, noise reduction only, volume normalization only, LLM only, LLM plus RAG, LLM plus Search, LLM plus RAG plus Search 조건을 차례대로 실행한다. 모든 조건은 같은 reference transcript를 사용하며, 각 조건의 CER과 WER을 baseline과 비교한다. 단순히 숫자가 낮아졌는지만 보는 것이 아니라, 어떤 단어가 수정되었고 어떤 문장이 새로 왜곡되었는지도 함께 확인한다.

실험 오디오는 깨끗한 음성, 약한 팬 소음이 섞인 음성, 배경 음악이 작게 깔린 음성, 마이크 입력이 낮은 음성, 발화 속도가 빠른 음성, 문장 사이 휴지가 긴 음성으로 나누어 준비한다. 기술 용어는 Qwen3-ASR, Qwen3.5, DeepFilterNet2, RNNoise, BS-RoFormer, vLLM, CUDA_VISIBLE_DEVICES, tensor parallel, stage replica, cache priming, chunk merge, CER, WER, RAG, Search를 포함한다. 한국어 발음으로는 큐웬 쓰리 에이에스알, 큐웬 삼점오, 딥필터넷 투, 알엔노이즈, 비에스 로포머, 브이엘엘엠, 쿠다 비저블 디바이시스, 텐서 패러럴, 스테이지 레플리카, 캐시 프라이밍, 청크 머지, 씨이에알, 더블유이에알, 알에이지, 서치를 함께 말한다.

성능 검증에서는 GPU가 단순히 메모리만 예약하고 놀고 있는 상태를 성공으로 보지 않는다. ASR stage에서는 네 개의 ASR replica가 모두 준비되어야 하며, 긴 오디오를 여러 chunk로 나누어 각 endpoint에 병렬 요청을 보내야 한다. Post-processing stage에서도 텍스트 chunk를 충분히 잘게 나누고 postprocess_parallelism을 높여 모든 post LLM replica가 요청을 받도록 한다. 만약 GPU 0번에 다른 사용자의 Python process가 4GB 정도 VRAM을 사용하고 있다면, 이 실험 시스템은 GPU 0번을 포기하지 않고 남은 VRAM에 맞는 serving profile을 계산해야 한다. 반대로 GPU 0번이 비어 있으면 다음 실행에서 최대 memory utilization과 최대 batch capacity를 사용해야 한다.

최종 결과 화면에서는 각 condition의 CER, WER, delta CER, delta WER, ASR endpoint, post endpoint, preprocess GPU, peak GPU utilization, peak VRAM, ASR cache hit 여부를 확인한다. 또한 volume__llm_rag_search_model_6e0874dd 같은 case identifier 오른쪽에는 접힌 diff 버튼이 있어야 한다. 사용자가 그 버튼을 누르면 raw transcript와 corrected transcript 사이의 변경점이 글자 단위로 표시되어야 한다. 예를 들어 불련이 Boolean으로 바뀌거나 포물이 for문으로 바뀌는 경우, 전체 문단을 하나의 변경 블록으로 칠하지 말고 실제로 바뀐 글자 주변만 세밀하게 표시해야 한다. 이 방식은 모델이 의미를 보존했는지, 도메인 용어만 고쳤는지, 또는 RAG와 Search 때문에 새로운 환각을 넣었는지 판단하는 데 필요하다.
```

매우 긴 낭독 reference:

```text
본 실험은 README에 정의된 ASR post-processing pipeline을 실제 서버 환경에서 재현하고 검증하기 위한 절차이다. 오디오는 Gradio UI에 업로드하고, reference transcript는 이 문단 전체를 그대로 붙여 넣는다. 실험자는 먼저 모든 기능을 끈 baseline을 실행하여 raw transcript와 CER/WER을 기록한다. 이후 자동 실험 모드를 켜고 keyword bias, noise reduction, volume normalization, LLM post-processing, RAG, Search의 가능한 조합을 실행한다. full_valid 모드에서는 유효한 기능 조합을 빠짐없이 평가하고, full_strength_sweep 모드에서는 keyword bias weight, noise reduction strength, volume normalization strength, postprocess strength, RAG strength, RAG top-k, search strength를 여러 단계로 바꾸어 sweet spot을 찾는다.

테스트 문장에는 일반 대화체와 기술 설명체를 모두 포함한다. 예를 들어 회의 참가자가 "오늘은 씨에스지피유투 서버에서 자동 실험을 돌리고, 포트 7860번의 Gradio UI와 포트 18000번부터 18003번까지의 vLLM endpoint를 확인하겠습니다"라고 말한다. 다른 참가자는 "Qwen3-ASR 1.7B가 긴 오디오를 silence-aware chunking으로 나누고, Qwen3.5 9B가 후처리 chunk를 받아 도메인 용어를 보정합니다"라고 답한다. 또 다른 참가자는 "DeepFilterNet2와 RNNoise는 잡음 제거 효과가 다를 수 있고, BS-RoFormer는 긴 시간 흐름의 음성 분리에 강점이 있지만 실제 backend 연결 여부를 확인해야 합니다"라고 덧붙인다. 이러한 문장은 ASR이 숫자, 포트, 영어 모델명, 한국어 발음, 하이픈, 약어를 얼마나 정확히 처리하는지 확인하기 위해 필요하다.

정량 평가에서는 reference transcript가 반드시 있어야 한다. Reference가 없으면 CER과 WER은 계산되지 않으며, corrected transcript가 좋아 보이더라도 엄격한 결론을 내릴 수 없다. Reference가 있을 때는 raw CER, corrected CER, raw WER, corrected WER, delta CER, delta WER을 모두 기록한다. delta 값이 양수이면 corrected transcript가 baseline보다 좋아졌다는 뜻이고, delta 값이 음수이면 후처리가 오히려 악화되었다는 뜻이다. 그러나 숫자만으로는 충분하지 않다. 예를 들어 LLM이 "브이엘엘엠"을 "vLLM"으로 바꾸는 것은 좋은 수정일 수 있지만, "실험 조건을 비교한다"를 "모든 조건이 성공했다"로 바꾸면 의미 왜곡이다. 따라서 각 method별 diff를 반드시 열어 보고, 실제 변경된 단어와 문장 단위를 확인해야 한다.

서버 성능 평가에서는 모든 가용 GPU를 적극적으로 사용한다. 네 장의 L40이 비어 있다면 ASR stage는 네 개의 ASR replica를 동시에 띄우고, post-processing stage는 네 개의 post LLM replica를 동시에 띄운다. Auto Experiment는 condition worker를 충분히 늘려 요청 큐가 비지 않도록 하며, 일반 Run도 stage replica pool에 맞춰 ASR chunk parallelism과 postprocess parallelism을 보정한다. GPU acceleration이 가능한 전처리 backend는 CUDA_VISIBLE_DEVICES를 통해 preprocess GPU pool에 분산한다. 특정 GPU에 외부 process가 떠 있으면 그 process를 종료하지 않는다. 대신 nvidia-smi에서 확인한 free VRAM을 기준으로 gpu-memory-utilization, max-model-len, max-num-seqs, max-num-batched-tokens를 계산하여 남은 자원을 사용한다. 이 정책은 특정 PID나 GPU 번호를 하드코딩하지 않고, 실행 시점의 실제 서버 상태를 기준으로 동작해야 한다.

실험자는 결과를 해석할 때 다음 질문을 순서대로 확인한다. 첫째, baseline 대비 CER/WER이 개선되었는가. 둘째, 개선된 조건이 여러 오디오 유형에서도 반복되는가. 셋째, keyword bias가 도메인 용어를 정확히 살렸는가. 넷째, noise reduction이 잡음을 줄였지만 음성을 손상시키지는 않았는가. 다섯째, volume normalization이 작은 음성을 키웠지만 clipping을 만들지는 않았는가. 여섯째, LLM post-processing이 ASR 오류만 고치고 원래 의미를 유지했는가. 일곱째, RAG와 Search가 실제 context를 보완했는가, 아니면 새로운 잘못된 정보를 넣었는가. 여덟째, 각 method의 diff에서 변경된 글자와 단어가 사람이 납득할 수 있는 수준인가.

이 reference는 실험 결과가 너무 쉽게 0퍼센트 근처로 떨어지는 문제를 피하기 위해 길게 작성되었다. 짧은 문장 하나로만 테스트하면 한두 글자 차이가 전체 CER/WER에 과도하게 반영되거나 반대로 조건 간 차이가 거의 보이지 않는다. 긴 reference를 사용하면 모델명, 포트 번호, GPU 번호, 기술 용어, 일반 문장, 긴 문맥, 반복되지 않는 표현이 함께 평가된다. 따라서 오디오 제작자는 이 문단을 끊지 말고 자연스럽게 읽되, 문장 사이에 적절한 휴지를 둔다. 실험자는 같은 reference를 그대로 붙여 넣고, Auto Experiment가 생성한 모든 case의 diff와 CER/WER을 함께 확인한다.
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
