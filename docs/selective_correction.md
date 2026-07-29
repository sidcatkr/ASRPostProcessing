# Selective ASR correction plan

The current default direction is precision-first: keep the raw ASR text unless a proposed edit is clearly better.

## Architecture

```text
Audio -> ASR baseline -> candidate edit proposal -> verifier/reranker -> final transcript
```

The post-processing LLM is asked for minimal edit proposals. Runtime code verifies each edit against the raw chunk and applies only accepted edits. The raw span is always the fallback candidate.

## Default policy

- Search remains disabled by default; use it later only for entity spelling verification.
- Post-processing decoding is deterministic (`temperature=0.0`, `top_p=1.0`).
- `postprocess_strength` changes acceptance thresholds rather than model sampling randomness:
  - conservative: confidence >= 0.95
  - balanced: confidence >= 0.90
  - aggressive: confidence >= 0.80
- Edits must match an exact `before` span in the raw chunk, must not overlap, and must stay within `selective_max_edit_ratio` and `selective_max_edits_per_100_chars`.

## Model exploration

Keep Qwen3-ASR as a baseline, but run comparable ASR adapters/configs for other candidates such as NVIDIA NeMo/Parakeet or Whisper-family models. Compare them with the same downstream verifier enabled so model quality and post-processing selection are not conflated.
