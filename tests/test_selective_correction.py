from asrpostprocessing.config import ExperimentConfig
from asrpostprocessing.schemas import CorrectionResult, Edit
from asrpostprocessing.selective_correction import verify_and_apply_correction


def test_selective_correction_applies_high_confidence_exact_span():
    config = ExperimentConfig.from_mapping({"postprocess_strength": 0.5})
    result = CorrectionResult(
        corrected_text="Claude Code로 for문을 작성합니다.",
        edits=[Edit(before="포문", after="for문", reason="keyword", confidence=0.93)],
    )

    verified = verify_and_apply_correction("Claude Code로 포문을 작성합니다.", result, config)

    assert verified.corrected_text == "Claude Code로 for문을 작성합니다."
    assert len(verified.edits) == 1
    assert verified.metadata["selective_correction"]["applied_count"] == 1


def test_selective_correction_rejects_low_confidence_rewrite():
    config = ExperimentConfig.from_mapping({"postprocess_strength": 0.5})
    result = CorrectionResult(
        corrected_text="완전히 다른 문장입니다.",
        edits=[Edit(before="포문", after="for문", reason="guess", confidence=0.4)],
    )

    verified = verify_and_apply_correction("Claude Code로 포문을 작성합니다.", result, config)

    assert verified.corrected_text == "Claude Code로 포문을 작성합니다."
    assert verified.edits == []
    assert verified.risk == "unchanged"
    assert verified.metadata["selective_correction"]["rejected_count"] == 1


def test_selective_correction_can_be_disabled_for_ablation():
    config = ExperimentConfig.from_mapping({"enable_selective_correction": False})
    result = CorrectionResult(corrected_text="교정", edits=[])

    verified = verify_and_apply_correction("원문", result, config)

    assert verified.corrected_text == "교정"
    assert verified.metadata["selective_correction"]["enabled"] is False
