from __future__ import annotations

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


_install_vllm_compat()


def main() -> None:
    from qwen_asr.cli.serve import main as qwen_asr_main  # type: ignore

    qwen_asr_main()


if __name__ == "__main__":
    main()
