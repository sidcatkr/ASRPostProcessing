from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import ExperimentConfig
from .pipeline import PipelineRunner
from .text import make_diff_html


def launch_ui(config_path: Optional[str] = None, host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    try:
        import gradio as gr  # type: ignore
    except Exception as exc:
        raise RuntimeError("Gradio is required for `asrpp ui`. Install project dependencies first.") from exc

    initial_config = ExperimentConfig()
    if config_path:
        from .config import load_config

        initial_config = load_config(config_path)

    with gr.Blocks(title="ASR Post-Processing Lab") as demo:
        gr.Markdown("# ASR Post-Processing Lab")
        with gr.Row():
            audio = gr.Audio(label="Audio", type="filepath", sources=["upload", "microphone"])
            with gr.Column():
                reference_text = gr.Textbox(label="Reference transcript", lines=6)
                reference_file = gr.File(label="Reference file (.txt)", file_types=[".txt"])

        with gr.Accordion("Pipeline controls", open=True):
            with gr.Row():
                enable_preprocess = gr.Checkbox(label="Preprocess", value=initial_config.enable_preprocess)
                preprocess_model = gr.Dropdown(
                    ["none", "volume_normalization", "RNNoise", "BS-RoFormer"],
                    value=initial_config.preprocess_model,
                    label="Preprocess model",
                )
                preprocess_strength = gr.Slider(0, 1, value=initial_config.preprocess_strength, step=0.05, label="Preprocess strength")
            with gr.Row():
                enable_keyword_bias = gr.Checkbox(label="Keyword Bias", value=initial_config.enable_keyword_bias)
                keyword_bias_weight = gr.Slider(0, 1, value=initial_config.keyword_bias_weight, step=0.25, label="Keyword Bias weight")
                keywords = gr.Textbox(label="Keywords", value=", ".join(initial_config.keywords), placeholder="Claude Code, Boolean, for문")
            with gr.Row():
                enable_llm = gr.Checkbox(label="LLM post-process", value=initial_config.enable_llm_postprocess)
                postprocess_strength = gr.Slider(0, 1, value=initial_config.postprocess_strength, step=0.05, label="Post-process strength")
            with gr.Row():
                enable_rag = gr.Checkbox(label="RAG", value=initial_config.enable_rag)
                rag_strength = gr.Slider(0, 1, value=initial_config.rag_strength, step=0.05, label="RAG strength")
                rag_top_k = gr.Slider(1, 10, value=initial_config.rag_top_k, step=1, label="RAG top-k")
            rag_text = gr.Textbox(label="RAG text", value=initial_config.rag_inline_text, lines=6)
            rag_files = gr.File(label="RAG files", file_count="multiple", file_types=[".txt", ".md", ".csv", ".json"])
            with gr.Row():
                enable_search = gr.Checkbox(label="Search", value=initial_config.enable_search)
                search_strength = gr.Slider(0, 1, value=initial_config.search_strength, step=0.05, label="Search strength")
                search_provider = gr.Dropdown(
                    ["duckduckgo", "endpoint", "none"],
                    value=initial_config.search_provider,
                    label="Search provider",
                )
                search_endpoint = gr.Textbox(value=initial_config.search_endpoint, label="Search endpoint")
            with gr.Row():
                asr_model = gr.Textbox(value=initial_config.asr_model, label="ASR model")
                post_model = gr.Textbox(value=initial_config.post_model, label="Post model")
            with gr.Row():
                asr_base_url = gr.Textbox(value=initial_config.asr_base_url, label="ASR base URL")
                post_base_url = gr.Textbox(value=initial_config.post_base_url, label="Post base URL")
            with gr.Row():
                asr_backend = gr.Dropdown(
                    ["vllm_chat", "qwen_asr_vllm", "qwen_asr_transformers", "mock"],
                    value=initial_config.asr_backend,
                    label="ASR backend",
                )
                post_backend = gr.Dropdown(["vllm_openai", "mock"], value=initial_config.post_backend, label="Post backend")
                run_button = gr.Button("Run", variant="primary")

        with gr.Row():
            raw_output = gr.Textbox(label="Raw transcript", lines=12)
            corrected_output = gr.Textbox(label="Corrected transcript", lines=12)
        diff_output = gr.HTML(label="Diff")
        with gr.Row():
            metrics_output = gr.JSON(label="Metrics")
            edits_output = gr.JSON(label="Edits")
        progress_output = gr.Textbox(label="Run status", lines=4)

        run_button.click(
            fn=run_from_ui,
            inputs=[
                audio,
                reference_text,
                reference_file,
                enable_preprocess,
                preprocess_model,
                preprocess_strength,
                enable_keyword_bias,
                keyword_bias_weight,
                keywords,
                enable_llm,
                postprocess_strength,
                enable_rag,
                rag_strength,
                rag_top_k,
                rag_text,
                rag_files,
                enable_search,
                search_strength,
                search_provider,
                search_endpoint,
                asr_model,
                post_model,
                asr_base_url,
                post_base_url,
                asr_backend,
                post_backend,
            ],
            outputs=[raw_output, corrected_output, diff_output, metrics_output, edits_output, progress_output],
        )

    return demo.launch(server_name=host, server_port=port, share=share)


def run_from_ui(
    audio_path: Optional[str],
    reference_text: str,
    reference_file: Any,
    enable_preprocess: bool,
    preprocess_model: str,
    preprocess_strength: float,
    enable_keyword_bias: bool,
    keyword_bias_weight: float,
    keywords: str,
    enable_llm: bool,
    postprocess_strength: float,
    enable_rag: bool,
    rag_strength: float,
    rag_top_k: int,
    rag_text: str,
    rag_files: Any,
    enable_search: bool,
    search_strength: float,
    search_provider: str,
    search_endpoint: str,
    asr_model: str,
    post_model: str,
    asr_base_url: str,
    post_base_url: str,
    asr_backend: str,
    post_backend: str,
) -> Tuple[str, str, str, dict, list, str]:
    if not audio_path:
        return "", "", "", {}, [], "No audio input provided."
    reference = _read_reference_from_ui(reference_text, reference_file)
    config = ExperimentConfig(
        asr_model=asr_model or "Qwen/Qwen3-ASR-1.7B",
        post_model=post_model or "Qwen/Qwen3.5-9B",
        asr_backend=asr_backend,
        post_backend=post_backend,
        asr_base_url=asr_base_url or "http://127.0.0.1:8000/v1",
        post_base_url=post_base_url or "http://127.0.0.1:8001/v1",
        enable_preprocess=bool(enable_preprocess),
        preprocess_model=preprocess_model,
        preprocess_strength=float(preprocess_strength),
        enable_keyword_bias=bool(enable_keyword_bias),
        keyword_bias_weight=float(keyword_bias_weight),
        keywords=_split_keywords(keywords),
        enable_llm_postprocess=bool(enable_llm),
        postprocess_strength=float(postprocess_strength),
        enable_rag=bool(enable_rag),
        rag_strength=float(rag_strength),
        rag_top_k=int(rag_top_k),
        rag_inline_text=rag_text or "",
        rag_files=_file_paths(rag_files),
        enable_search=bool(enable_search),
        search_strength=float(search_strength),
        search_provider=search_provider or "duckduckgo",
        search_endpoint=search_endpoint or "",
    )
    try:
        output = PipelineRunner(config).run(audio_path=audio_path, reference_text=reference)
    except Exception as exc:
        return "", "", "", {}, [], f"Run failed: {exc}"
    return (
        output.raw.text,
        output.correction.corrected_text,
        output.diff_html or make_diff_html(output.raw.text, output.correction.corrected_text),
        output.metrics.to_dict(),
        [edit.to_dict() for edit in output.correction.edits],
        (
            f"Run ID: {output.run_id}\n"
            f"Output: {output.output_dir}\n"
            f"TensorBoard: tensorboard --logdir {config.runs_dir} --port {config.tensorboard_port}\n"
            f"Artifacts: {json.dumps(output.artifacts, ensure_ascii=False)}"
        ),
    )


def _split_keywords(value: str) -> List[str]:
    return [item.strip() for item in (value or "").replace("\n", ",").split(",") if item.strip()]


def _read_reference_from_ui(reference_text: str, reference_file: Any) -> Optional[str]:
    if reference_text and reference_text.strip():
        return reference_text.strip()
    paths = _file_paths(reference_file)
    if paths:
        return Path(paths[0]).read_text(encoding="utf-8").strip()
    return None


def _file_paths(files: Any) -> List[str]:
    if not files:
        return []
    if not isinstance(files, list):
        files = [files]
    paths: List[str] = []
    for item in files:
        if isinstance(item, str):
            paths.append(item)
        elif hasattr(item, "name"):
            paths.append(str(item.name))
        elif isinstance(item, dict) and item.get("name"):
            paths.append(str(item["name"]))
    return paths
