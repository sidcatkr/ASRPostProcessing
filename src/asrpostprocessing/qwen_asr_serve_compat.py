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


_install_vllm_compat()


def main() -> None:
    from qwen_asr.cli.serve import main as qwen_asr_main  # type: ignore

    qwen_asr_main()


if __name__ == "__main__":
    main()
