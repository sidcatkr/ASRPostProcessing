from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .config import ExperimentConfig
from .gpu_status import query_gpu_status
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
            with gr.Accordion("ASR Keyword Bias", open=True):
                with gr.Row():
                    enable_keyword_bias = gr.Checkbox(label="Keyword Bias", value=initial_config.enable_keyword_bias)
                    keyword_bias_weight = gr.Slider(0, 1, value=initial_config.keyword_bias_weight, step=0.25, label="Keyword Bias weight")
                    keywords = gr.Textbox(label="Keywords", value=", ".join(initial_config.keywords), placeholder="Claude Code, Boolean, for문")
            with gr.Accordion("Pre Process", open=True):
                with gr.Row():
                    enable_noise_reduction = gr.Checkbox(label="Noise reduction", value=initial_config.enable_noise_reduction)
                    noise_reduction_model = gr.Dropdown(
                        ["none", "RNNoise", "BS-RoFormer"],
                        value=initial_config.noise_reduction_model,
                        label="Noise reduction model",
                    )
                    noise_reduction_strength = gr.Slider(
                        0,
                        1,
                        value=initial_config.noise_reduction_strength,
                        step=0.05,
                        label="Noise reduction strength",
                    )
                with gr.Row():
                    enable_volume_normalization = gr.Checkbox(label="Volume normalization", value=initial_config.enable_volume_normalization)
                    volume_normalization_strength = gr.Slider(
                        0,
                        1,
                        value=initial_config.volume_normalization_strength,
                        step=0.05,
                        label="Volume normalization strength",
                    )
                    volume_target_dbfs = gr.Slider(-40, -6, value=initial_config.volume_target_dbfs, step=1, label="Volume target dBFS")
                with gr.Row():
                    rnnoise_command = gr.Textbox(value=initial_config.rnnoise_command, label="RNNoise command")
                    bs_roformer_command = gr.Textbox(value=initial_config.bs_roformer_command, label="BS-RoFormer command")
                    ffmpeg_command = gr.Textbox(value=initial_config.ffmpeg_command, label="FFmpeg command for non-WAV")
            with gr.Row():
                enable_llm = gr.Checkbox(label="LLM post-process", value=initial_config.enable_llm_postprocess)
                postprocess_strength = gr.Slider(0, 1, value=initial_config.postprocess_strength, step=0.05, label="Post-process strength")
            with gr.Row():
                enable_rag = gr.Checkbox(label="RAG", value=initial_config.enable_rag)
                rag_strength = gr.Slider(0, 1, value=initial_config.rag_strength, step=0.05, label="RAG strength")
                rag_top_k = gr.Slider(1, 10, value=initial_config.rag_top_k, step=1, label="RAG top-k")
            rag_text = gr.Textbox(label="RAG text", value=initial_config.rag_inline_text, lines=6)
            rag_files = gr.File(
                label="RAG files (.txt, .md, .csv, .json, .pdf)",
                file_count="multiple",
                file_types=[".txt", ".md", ".markdown", ".csv", ".json", ".pdf"],
            )
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
                post_model = gr.Textbox(value=initial_config.post_model, label="Post-processing LLM model")
            with gr.Row():
                asr_base_url = gr.Textbox(value=initial_config.asr_base_url, label="ASR base URL")
                post_base_url = gr.Textbox(value=initial_config.post_base_url, label="Post-processing LLM API URL")
            with gr.Accordion("Model server startup", open=True):
                with gr.Row():
                    auto_start_model_servers = gr.Checkbox(
                        label="Start required model servers when Run is pressed",
                        value=initial_config.auto_start_model_servers,
                    )
                    model_residency = gr.Dropdown(
                        [
                            ("All required models stay loaded (fast, high VRAM)", "parallel"),
                            ("One model at a time (slow, low VRAM)", "sequential"),
                        ],
                        value=initial_config.model_residency,
                        label="Model residency",
                    )
                with gr.Row():
                    asr_server_gpu = gr.Textbox(value=initial_config.asr_server_gpu, label="ASR server GPU")
                    post_server_gpu = gr.Textbox(value=initial_config.post_server_gpu, label="Post-processing server GPU")
                    server_start_timeout_s = gr.Slider(
                        60,
                        1800,
                        value=initial_config.server_start_timeout_s,
                        step=30,
                        label="Server start timeout seconds",
                    )
                    server_shutdown_timeout_s = gr.Slider(
                        5,
                        180,
                        value=initial_config.server_shutdown_timeout_s,
                        step=5,
                        label="Server shutdown timeout seconds",
                    )
                with gr.Row():
                    server_log_dir = gr.Textbox(value=initial_config.server_log_dir, label="Server log directory")
                    asr_server_host = gr.Textbox(value=initial_config.asr_server_host, label="ASR server bind host")
                    post_server_host = gr.Textbox(value=initial_config.post_server_host, label="Post-processing server bind host")
                asr_server_command = gr.Textbox(
                    value=initial_config.asr_server_command,
                    label="Custom ASR server command",
                    placeholder="Leave empty to use python -m asrpostprocessing.qwen_asr_serve_compat {model}",
                )
                post_server_command = gr.Textbox(
                    value=initial_config.post_server_command,
                    label="Custom post-processing server command",
                    placeholder="Leave empty to use the Qwen3.5 vLLM command",
                )
            with gr.Row():
                asr_backend = gr.Dropdown(
                    [
                        ("Qwen ASR OpenAI-compatible server", "vllm_chat"),
                        ("qwen-asr package via vLLM", "qwen_asr_vllm"),
                        ("qwen-asr package via Transformers", "qwen_asr_transformers"),
                        ("Mock ASR for UI testing", "mock"),
                    ],
                    value=initial_config.asr_backend,
                    label="ASR backend",
                )
                post_backend = gr.Dropdown(
                    [
                        ("vLLM OpenAI-compatible API", "vllm_openai"),
                        ("Mock post-processor for UI testing", "mock"),
                    ],
                    value=initial_config.post_backend,
                    label="Post-processing backend",
                )
                run_button = gr.Button("Run", variant="primary")
            progress_output = gr.Textbox(label="Run status", lines=4)

        with gr.Row():
            raw_output = gr.Textbox(label="Raw transcript", lines=12)
            corrected_output = gr.Textbox(label="Corrected transcript", lines=12)
        diff_output = gr.HTML(label="Diff")
        with gr.Row():
            metrics_output = gr.JSON(label="Metrics")
            edits_output = gr.JSON(label="Edits")
        with gr.Row():
            preprocess_output = gr.JSON(label="Preprocess")
            server_output = gr.JSON(label="Model servers")
        with gr.Row():
            gpu_output = gr.JSON(label="Server GPU / VRAM status", value=query_gpu_status())
            refresh_gpu_button = gr.Button("Refresh GPU status")

        run_button.click(
            fn=run_from_ui_stream,
            inputs=[
                audio,
                reference_text,
                reference_file,
                enable_keyword_bias,
                keyword_bias_weight,
                keywords,
                enable_noise_reduction,
                noise_reduction_model,
                noise_reduction_strength,
                enable_volume_normalization,
                volume_normalization_strength,
                volume_target_dbfs,
                rnnoise_command,
                bs_roformer_command,
                ffmpeg_command,
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
                auto_start_model_servers,
                server_start_timeout_s,
                server_log_dir,
                asr_server_gpu,
                post_server_gpu,
                asr_server_host,
                post_server_host,
                asr_server_command,
                post_server_command,
                asr_backend,
                post_backend,
                model_residency,
                server_shutdown_timeout_s,
            ],
            outputs=[
                raw_output,
                corrected_output,
                diff_output,
                metrics_output,
                edits_output,
                preprocess_output,
                server_output,
                progress_output,
                gpu_output,
            ],
        )
        refresh_gpu_button.click(fn=query_gpu_status, outputs=gpu_output)

    demo.queue()
    return demo.launch(server_name=host, server_port=port, share=share)


def run_from_ui_stream(*args):
    yield "", "", "", {}, [], {}, [], "Preparing run and checking model servers...", query_gpu_status()
    yield run_from_ui(*args)


def run_from_ui(
    audio_path: Optional[str],
    reference_text: str,
    reference_file: Any,
    enable_keyword_bias: bool,
    keyword_bias_weight: float,
    keywords: str,
    enable_noise_reduction: bool,
    noise_reduction_model: str,
    noise_reduction_strength: float,
    enable_volume_normalization: bool,
    volume_normalization_strength: float,
    volume_target_dbfs: float,
    rnnoise_command: str,
    bs_roformer_command: str,
    ffmpeg_command: str,
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
    auto_start_model_servers: bool,
    server_start_timeout_s: float,
    server_log_dir: str,
    asr_server_gpu: str,
    post_server_gpu: str,
    asr_server_host: str,
    post_server_host: str,
    asr_server_command: str,
    post_server_command: str,
    asr_backend: str,
    post_backend: str,
    model_residency: str = "parallel",
    server_shutdown_timeout_s: float = 30.0,
) -> Tuple[str, str, str, dict, list, dict, list, str, dict]:
    if not audio_path:
        return "", "", "", {}, [], {}, [], "No audio input provided.", query_gpu_status()
    reference = _read_reference_from_ui(reference_text, reference_file)
    config = ExperimentConfig(
        asr_model=asr_model or "Qwen/Qwen3-ASR-1.7B",
        post_model=post_model or "Qwen/Qwen3.5-9B",
        asr_backend=asr_backend,
        post_backend=post_backend,
        asr_base_url=asr_base_url or "http://127.0.0.1:18000/v1",
        post_base_url=post_base_url or "http://127.0.0.1:18001/v1",
        auto_start_model_servers=bool(auto_start_model_servers),
        model_residency=model_residency or "parallel",
        server_start_timeout_s=float(server_start_timeout_s),
        server_shutdown_timeout_s=float(server_shutdown_timeout_s),
        server_log_dir=server_log_dir or "outputs/model_servers",
        asr_server_gpu=asr_server_gpu or "0",
        post_server_gpu=post_server_gpu or "1",
        asr_server_host=asr_server_host or "0.0.0.0",
        post_server_host=post_server_host or "0.0.0.0",
        asr_server_command=asr_server_command or "",
        post_server_command=post_server_command or "",
        enable_preprocess=False,
        preprocess_model="none",
        preprocess_strength=0.0,
        enable_noise_reduction=bool(enable_noise_reduction),
        noise_reduction_model=noise_reduction_model or "none",
        noise_reduction_strength=float(noise_reduction_strength),
        enable_volume_normalization=bool(enable_volume_normalization),
        volume_normalization_strength=float(volume_normalization_strength),
        volume_target_dbfs=float(volume_target_dbfs),
        rnnoise_command=rnnoise_command or "",
        bs_roformer_command=bs_roformer_command or "",
        ffmpeg_command=ffmpeg_command or "",
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
        hint = ""
        if (asr_backend != "mock" or post_backend != "mock") and not auto_start_model_servers:
            hint = (
                "\n\nCurrent backends require running model servers. "
                "For UI-only testing, set ASR backend to 'Mock ASR for UI testing' "
                "and Post-processing backend to 'Mock post-processor for UI testing'."
            )
        elif auto_start_model_servers:
            hint = "\n\nAutomatic model server startup is enabled. Check the server log path in the error above."
        return "", "", "", {}, [], {}, [], f"Run failed: {exc}{hint}", query_gpu_status()
    server_lines = _format_server_statuses(output.server_statuses)
    return (
        output.raw.text,
        output.correction.corrected_text,
        output.diff_html or make_diff_html(output.raw.text, output.correction.corrected_text),
        output.metrics.to_dict(),
        [edit.to_dict() for edit in output.correction.edits],
        output.preprocess,
        output.server_statuses,
        (
            f"Run ID: {output.run_id}\n"
            f"Model residency: {config.model_residency}\n"
            f"{server_lines}"
            f"Output: {output.output_dir}\n"
            f"TensorBoard: tensorboard --logdir {config.runs_dir} --port {config.tensorboard_port}\n"
            f"Artifacts: {json.dumps(output.artifacts, ensure_ascii=False)}"
        ),
        query_gpu_status(),
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


def _format_server_statuses(statuses: List[dict]) -> str:
    if not statuses:
        return ""
    lines = ["Model servers:"]
    for item in statuses:
        pid = f" pid={item['pid']}" if item.get("pid") else ""
        log_path = f" log={item['log_path']}" if item.get("log_path") else ""
        lines.append(f"- {item['name']}: {item['status']} at {item['base_url']}{pid}{log_path}")
    return "\n".join(lines) + "\n"
