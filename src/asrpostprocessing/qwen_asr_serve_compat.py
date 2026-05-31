from __future__ import annotations

import importlib
import sys
import types


def _install_vllm_compat() -> None:
    try:
        import vllm.inputs as inputs  # type: ignore
    except Exception:
        return
    if "vllm.inputs.data" in sys.modules:
        return
    module = types.ModuleType("vllm.inputs.data")
    for name in dir(inputs):
        if not name.startswith("__"):
            setattr(module, name, getattr(inputs, name))
    sys.modules["vllm.inputs.data"] = module

    try:
        multimodal_inputs = importlib.import_module("vllm.multimodal.inputs")
        multimodal_parse = importlib.import_module("vllm.multimodal.parse")
    except Exception:
        return
    for name in ("ModalityData", "MultiModalDataDict"):
        if not hasattr(multimodal_inputs, name):
            if hasattr(multimodal_parse, name):
                setattr(multimodal_inputs, name, getattr(multimodal_parse, name))
            elif hasattr(inputs, name):
                setattr(multimodal_inputs, name, getattr(inputs, name))


def _patch_qwen_asr_vllm_compat() -> None:
    try:
        qwen3_asr = importlib.import_module("qwen_asr.core.vllm_backend.qwen3_asr")
    except Exception:
        return

    processing_info = getattr(qwen3_asr, "Qwen3ASRProcessingInfo", None)
    parser_cls = getattr(qwen3_asr, "Qwen3ASRMultiModalDataParser", None)
    if processing_info is None or parser_cls is None or "get_data_parser" in processing_info.__dict__:
        return

    def get_data_parser(self):  # type: ignore[no-untyped-def]
        feature_extractor = self.get_feature_extractor()
        kwargs = {"target_sr": feature_extractor.sampling_rate}
        if hasattr(self, "_get_expected_hidden_size"):
            kwargs["expected_hidden_size"] = self._get_expected_hidden_size()
        return parser_cls(**kwargs)

    processing_info.get_data_parser = get_data_parser


_install_vllm_compat()


def main() -> None:
    from qwen_asr.cli.serve import main as qwen_asr_main  # type: ignore

    _patch_qwen_asr_vllm_compat()
    qwen_asr_main()


if __name__ == "__main__":
    main()
