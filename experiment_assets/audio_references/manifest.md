# Audio Reference Asset Manifest

ElevenLabs로 생성한 reference audio와 같은 주제의 transcript, RAG context, Keyword Bias terms를 한 곳에 묶은 manifest이다.

| Topic | Audio | Reference | RAG | Keywords |
| --- | --- | --- | --- | --- |
| 짧은 sanity check | `audio/sanity_check.mp3` | `references/sanity_check.txt` | `rag/sanity_check.md` | `keywords/sanity_check.txt` |
| 일반 글 | `audio/general_prose.mp3` | `references/general_prose.txt` | `rag/general_prose.md` | `keywords/general_prose.txt` |
| 대화와 안내 | `audio/dialogue_and_guide.mp3` | `references/dialogue_and_guide.txt` | `rag/dialogue_and_guide.md` | `keywords/dialogue_and_guide.txt` |
| 어려운 설명문 | `audio/difficult_expository.mp3` | `references/difficult_expository.txt` | `rag/difficult_expository.md` | `keywords/difficult_expository.txt` |
| 시 | `audio/poem.mp3` | `references/poem.txt` | `rag/poem.md` | `keywords/poem.txt` |
| 작문 | `audio/essay.mp3` | `references/essay.txt` | `rag/essay.md` | `keywords/essay.txt` |
| 장문 혼합 | `audio/long_mixed.mp3` | `references/long_mixed.txt` | `rag/long_mixed.md` | `keywords/long_mixed.txt` |

실험 시 같은 row의 audio, reference, RAG, keywords를 함께 사용한다. 전체 통합 RAG와 keyword 목록은 상위 `experiment_assets/general_rag_context.md`, `experiment_assets/general_keyword_bias_terms.txt`에 남겨 둔다.
