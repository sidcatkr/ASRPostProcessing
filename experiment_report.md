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

### 1.1 구현된 실험 기능

현재 Gradio UI에는 단일 실행과 자동 실험을 모두 수행할 수 있는 기능이 구현되어 있다. 단일 실행에서는 오디오와 reference transcript를 입력하고, Keyword Bias, Noise reduction, Volume normalization, LLM 후처리, RAG, Search를 각각 토글해 결과를 비교할 수 있다. 자동 실험에서는 사용자가 켠 범위 안에서 baseline, 단일 기능, 조합 조건, 강도 sweep, 모델 후보 조합을 자동으로 생성해 여러 조건을 반복 실행한다.

정량 평가는 CER/WER, baseline 대비 개선량, latency, GPU/VRAM 사용량을 기록한다. CER/WER은 공백, 줄바꿈, 문장부호, 기호를 제외한 한 줄 내용 문자열 기준으로 계산하며, `cer_strict`는 참고용으로 별도 기록한다. 결과 화면에는 raw transcript, corrected transcript, CER/WER, edits, preprocessing 정보, model server 상태, GPU 상태가 표시된다.

Diff view에는 삭제, 삽입, 대체가 구분되어 표시되며, 각 변경의 reference/raw 텍스트와 corrected 텍스트를 함께 확인할 수 있다. Reference가 있는 경우 CER/WER error monitor가 오류 위치와 오류 기여도를 보여준다. 단일 실행과 자동 실험 결과는 보고서에 활용할 수 있도록 HTML diff export 파일로도 저장된다.

서버 실행은 L4 x4 GPU 환경을 기준으로 구성되어 있다. Run 또는 Auto Experiment가 필요할 때 stage model server를 올리고, 종료 후 ASR/Post-processing 모델을 offload해 유휴 GPU 점유를 줄인다. VRAM이 부족하거나 일부 GPU에 다른 프로세스가 있을 때는 가용 GPU와 남은 VRAM을 기준으로 가능한 stage replica만 사용하도록 구성되어 있다.

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

일반 글 reference:

```text
토요일 오전 아홉 시, 해든마을 주민센터 앞마당에는 비에 젖은 은행잎 냄새와 갓 구운 빵 냄새가 함께 떠올랐다. 강서윤은 유리병에 담은 매실청 세 상자와 라벤더색 천 가방 열두 개를 가지런히 놓았고, 민재호는 작은 컵 4,500원, 큰 컵 7,800원이라고 쓴 가격표를 나무 상자에 기대어 세웠다. 노을정류장으로 가는 742번 버스가 임시 정류장으로 우회한다는 안내 방송이 두 번 흘러나왔지만, 장터에 모인 사람들은 대체로 서두르지 않았다. 삼도시장 빵집에서 가져온 호두 식빵은 점심 전에 모두 팔렸고, 끝까지 남은 것은 보리차 두 병, 파란 우산 하나, 그리고 누군가 놓고 간 얇은 노트뿐이었다.
```

대화와 안내 reference:

```text
안내 직원은 전화기를 어깨에 받친 채 서류를 넘겼다. "Omar Lee 님의 강연 장비 예약을 확인하겠습니다. 예약번호는 S-17-바람노트이고, 연락처는 02-314-1592, 수령 시간은 오후 두 시 삼십 분입니다." 전화한 사람은 잠시 망설이다가 말했다. "이름이 오마르 리처럼 들릴 수 있는데, 명찰에는 Omar Lee라고 적어 주세요. 장비 상자에는 검은색 HDMI 케이블 두 개와 무선 마이크 네 개가 들어 있어야 합니다." 직원은 다시 물었다. "수령 장소는 서리풀 도서관 1층 안내 데스크가 맞습니까?" 상대는 "맞습니다. 다만 B-219 보관실 열쇠는 강연 장비와 섞지 말아 주세요"라고 답했다.

잠시 뒤 같은 데스크에서 여행 설명회 안내 방송이 시작되었다. "2026년 7월 14일 화요일 오르트 항구 당일 코스 참가자는 신분증, 얇은 겉옷, 개인 컵, 충전된 보조 배터리를 준비해 주세요. 첫 배는 오전 8시 40분에 출발하고, 바람이 강하면 두 번째 배는 9시 15분으로 늦춰질 수 있습니다. 수평선 예약번호와 탑승자 이름이 다르면 창구에서 다시 확인해야 합니다. 해든마을, 노을정류장, 오르트 항구는 서로 다른 장소이므로 안내 문자를 받을 때 이름을 혼동하지 마세요."
```

어려운 설명문 reference:

```text
청연초서는 조선 후기 어느 이름 없는 서리가 남긴 필사본으로 전해진다. 표지는 낡았지만 제목만은 푸른 먹으로 또렷하게 남아 있고, 첫 장에는 비가 그친 뒤 강가의 세곡 창고를 점검했다는 짧은 기록이 있다. 본문에는 당시 장부의 숫자와 개인의 감상이 이상하게 섞여 있다. 어느 쪽은 곡식 스물네 섬, 소금 일곱 되, 젖은 종이 여섯 장을 적고, 다른 쪽은 강물 위에 떠 있던 검은 달과 멀리서 들려온 북소리를 적는다. 그래서 청연초서는 단순한 회계 문서도 아니고 완전한 일기도 아니다.

Asteria-9 관측 기록은 전혀 다른 성격의 글이다. 라그랑주 점 L2 근처에서 열린 관측 창은 11분 40초에 불과했고, 카시오페아 델타 구역의 배경광은 예상보다 낮았다. 에리다누스 관측 노트에는 같은 날짜 새벽 3시 12분의 하늘 밝기 값이 따로 적혀 있으며, 수평선 관측번호 HZ-0419가 붙어 있다. 이 번호는 오르트 항구 여행의 수평선 예약번호와 무관하다. 문서 끝에는 Muller-Lyer illusion과 뮐러-라이어 착시의 표기를 통일하라는 메모, 브라키스토크론 곡선을 최단 시간 강하 곡선으로 설명하라는 메모, 노르덴펠트 지수를 실제 경제 지표로 오해하지 말라는 메모가 함께 남아 있다.
```

시 reference:

```text
해가 낮게 걸린 저녁,
노을정류장에는 젖은 발자국이 남고
파란 우산 하나가 벤치 끝에서
자기 차례를 기다린다.

해든마을의 창문마다
작은 불빛이 천천히 켜질 때,
강서윤은 빈 유리병을 씻고
민재호는 지워진 가격표를 다시 쓴다.

오르트 항구 쪽 바람은
밤 8시 5분의 배를 흔들고,
The Quiet Harbor라는 제목은
아직 읽히지 않은 책등에서 조용히 빛난다.
```

작문 reference:

```text
내가 기억하는 가장 좋은 안내문은 길을 잃은 사람을 부끄럽게 만들지 않는 문장이었다. 어느 여름, 나는 강변 12로에서 라벤더 3번길로 가는 길을 잘못 들었고, 휴대전화 배터리는 4퍼센트밖에 남지 않았다. 그때 노을정류장 앞에 붙은 작은 안내문이 보였다. 안내문은 길을 틀렸다고 말하지 않았다. 대신 "왼쪽 골목으로 120미터를 걸으면 해든마을 주민센터가 보입니다"라고 적혀 있었다. 나는 그 문장이 좋았다. 명령하지 않고, 겁주지 않고, 필요한 만큼만 정확했기 때문이다.

좋은 글도 그와 비슷하다고 생각한다. 어려운 말이 필요한 곳에서는 어려운 말을 피하지 않되, 독자가 길을 잃지 않도록 주변을 밝혀야 한다. 청연초서, 브라키스토크론 곡선, 노르덴펠트 지수처럼 낯선 단어가 나오더라도 문맥이 단단하면 독자는 천천히 따라올 수 있다. 반대로 쉬운 단어만 쓰더라도 숫자와 이름과 시간이 흐트러지면 글은 금방 믿기 어려워진다. 그래서 나는 정확한 문장을 좋아한다. 정확한 문장은 차갑지 않다. 오히려 읽는 사람에게 돌아갈 길을 남겨 두는 친절한 문장에 가깝다.
```

장문 혼합 reference:

```text
새벽 배송 센터에서 박하준은 은색 카트 네 대를 밀고 들어왔다. 첫 번째 카트에는 라벤더 3번길로 갈 푸른유리컵 여섯 개가 실려 있었고, 두 번째 카트에는 은하우체통 모양 저금통 두 개가 있었다. 세 번째 카트에는 검은색 천 가방과 남색 목도리, 얇은 노트, 작은 카드 지갑이 들어 있었는데, 노트 표지에는 바람노트라는 손글씨 스티커가 붙어 있었다. 공항 분실물 센터에서 KJ-4821 항공편의 좌석 17C를 찾던 승객이 바로 그 노트를 설명했기 때문에, 박하준은 상자를 다른 물품과 섞지 않았다.

오전에는 한유진 교사가 공개 수업을 했다. 그는 물의 순환을 설명하며 증발, 응결, 침투, 빗물받이, 경사면이라는 단어를 칠판에 적었다. 한 학생은 비가 온 뒤 운동장에 물이 고이는 까닭을 배수구에 낙엽이 쌓였기 때문이라고 말했고, 다른 학생은 흙의 기울기 때문이라고 말했다. 오후에는 병원 접수 창구에서 김태린의 예약 시간이 확인되었다. 6월 18일 목요일 오전 11시 10분, 정형외과 3번 진료실이었다. 보호자는 아세트아미노펜은 가끔 먹지만 이부프로펜은 최근 한 달 동안 먹지 않았다고 말했다.

저녁에는 서리풀 도서관에서 작은 낭독회가 열렸다. 추천 도서는 김라온의 산책의 문장, 정이현의 푸른 계산서, 그리고 The Quiet Harbor였다. Omar Lee는 강연 장비를 찾으러 와서 예약번호 S-17-바람노트를 확인했고, 류다인은 B-219 보관실에서 청연초서의 표지 사진을 다시 살폈다. 한쪽 탁자에서는 이오세비오 알바레스, 나디아 크웬, 임하록, 세라핀 권, 미나토 유진이 Asteria-9 관측 로그와 에리다누스 관측 노트를 대조했다. 수평선 관측번호 HZ-0419는 천문 기록용이고, 여행 안내 문자에 적힌 수평선 예약번호는 오르트 항구 탑승 확인용이었다.
```

RAG와 Keyword Bias 준비:

- ElevenLabs reference audio manifest: `experiment_assets/audio_references/manifest.md`
- 주제별 audio: `experiment_assets/audio_references/audio/*.mp3`
- 주제별 reference transcript: `experiment_assets/audio_references/references/*.txt`
- 주제별 RAG 파일: `experiment_assets/audio_references/rag/*.md`
- 주제별 Keyword Bias 목록: `experiment_assets/audio_references/keywords/*.txt` comma-separated
- RAG 파일: `experiment_assets/general_rag_context.md`
- Keyword Bias 목록: `experiment_assets/general_keyword_bias_terms.txt` comma-separated
- 사용 방법: 한 오디오를 실험할 때는 manifest에서 같은 row의 audio, reference, RAG, keyword 파일을 함께 사용한다. Keyword Bias 입력에는 comma-separated terms를 그대로 붙여 넣는다. 여러 주제를 섞어 긴 실험을 할 때만 통합 RAG와 통합 keyword 목록을 사용한다. RAG가 켜진 조건과 꺼진 조건은 같은 reference 기준으로 함께 비교한다.

검증 포인트는 다음과 같다.

- Diff 출력이 비어 있지 않은지 확인한다.
- Reference transcript가 있을 때 CER/WER 값이 계산되는지 확인한다.
- CER/WER은 띄어쓰기, 줄바꿈, 문장부호, 기호만 다른 경우 오류로 세지 않는다. 내용이 같은데 공백이나 쉼표, 마침표, 물음표 같은 표기만 다른 결과가 우수 조건에서 밀려나면 안 된다.
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
