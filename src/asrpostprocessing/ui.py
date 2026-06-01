from __future__ import annotations

import inspect
import json
import os
import time
from html import escape
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from .auto_experiment import run_auto_experiment
from .config import ExperimentConfig
from .gpu_status import query_gpu_status
from .pipeline import PipelineRunner
from .preprocess import preprocess_audio
from .text import make_diff_html

RUN_STATUS_POLL_INTERVAL_S = 1.0
RUN_STATUS_RECENT_EVENT_LIMIT = 8
NOISE_REDUCTION_MODEL_CHOICES = [
    ("None", "none"),
    ("FFmpeg afftdn", "afftdn"),
    ("RNNoise", "rnnoise"),
    ("DeepFilterNet2", "deepfilternet2"),
    ("DeepFilterNet2-PF", "deepfilternet2_pf"),
    ("DeepFilterNet3", "deepfilternet3"),
    ("BS-RoFormer", "bs-roformer"),
]
RunOutput = Tuple[str, str, str, dict, list, dict, list, str, Optional[str], str, dict]


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
        initial_config_state = gr.State(value=initial_config.to_dict())
        gr.Markdown("# ASR Post-Processing Lab")
        with gr.Row():
            audio = gr.Audio(label="Audio", type="filepath", sources=["upload", "microphone"])
            large_audio_file = gr.File(
                label="Long audio file (.mp3, .wav)",
                file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm", ".aac"],
            )
            with gr.Column():
                reference_text = gr.Textbox(label="Reference transcript", lines=6)
                reference_file = gr.File(label="Reference file (.txt)", file_types=[".txt"])

        with gr.Accordion("Pipeline controls", open=True):
            with gr.Accordion("ASR Keyword Bias", open=True):
                with gr.Row():
                    enable_keyword_bias = gr.Checkbox(label="Keyword Bias", value=initial_config.enable_keyword_bias)
                    keyword_bias_weight = gr.Slider(0, 1, value=initial_config.keyword_bias_weight, step=0.25, label="Keyword Bias weight")
                    keywords = gr.Textbox(label="Keywords", value=", ".join(initial_config.keywords), placeholder="term A, product name, acronym")
            with gr.Accordion("Pre Process", open=True):
                with gr.Row():
                    enable_noise_reduction = gr.Checkbox(label="Noise reduction", value=initial_config.enable_noise_reduction)
                    noise_reduction_model = gr.Dropdown(
                        NOISE_REDUCTION_MODEL_CHOICES,
                        value=_canonical_noise_reduction_model(initial_config.noise_reduction_model),
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
                preview_preprocess_button = gr.Button("Preview preprocessed audio")
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
                asr_base_url = gr.Textbox(value=initial_config.asr_base_url, label="Primary ASR base URL")
                post_base_url = gr.Textbox(value=initial_config.post_base_url, label="Primary post-processing LLM API URL")
            with gr.Accordion("ASR request chunking", open=True):
                with gr.Row():
                    asr_chunking_strategy = gr.Dropdown(
                        [
                            ("Silence-aware / VAD-style", "silence"),
                            ("Fixed-duration segments", "fixed"),
                            ("No audio chunking", "none"),
                        ],
                        value=initial_config.asr_chunking_strategy,
                        label="ASR audio chunking",
                    )
                    asr_chunk_seconds = gr.Slider(5, 120, value=initial_config.asr_chunk_seconds, step=5, label="ASR max chunk seconds")
                    asr_request_timeout_s = gr.Slider(
                        30,
                        900,
                        value=initial_config.asr_request_timeout_s,
                        step=30,
                        label="ASR request timeout seconds",
                    )
                with gr.Row():
                    asr_chunk_padding_seconds = gr.Slider(
                        0,
                        5,
                        value=initial_config.asr_chunk_padding_seconds,
                        step=0.1,
                        label="Silence chunk padding seconds",
                    )
                    asr_silence_threshold_db = gr.Slider(
                        -80,
                        -10,
                        value=initial_config.asr_silence_threshold_db,
                        step=1,
                        label="Silence threshold dB",
                    )
                    asr_min_silence_seconds = gr.Slider(
                        0.1,
                        5.0,
                        value=initial_config.asr_min_silence_seconds,
                        step=0.1,
                        label="Minimum silence seconds",
                    )
                    asr_context_chars = gr.Slider(
                        0,
                        2000,
                        value=initial_config.asr_context_chars,
                        step=40,
                        label="ASR rolling context chars",
                    )
                    asr_chunk_parallelism = gr.Slider(
                        1,
                        16,
                        value=initial_config.asr_chunk_parallelism,
                        step=1,
                        label="ASR chunk workers",
                    )
            with gr.Accordion("Auto Experiment", open=False):
                with gr.Row():
                    auto_experiment_mode = gr.Checkbox(label="Auto Experiment Mode", value=False)
                    auto_experiment_coverage = gr.Dropdown(
                        [
                            ("Core ablation", "core_ablation"),
                            ("Full valid combination", "full_valid"),
                            ("Full + strength sweep", "full_strength_sweep"),
                        ],
                        value="full_valid",
                        label="Coverage",
                    )
                with gr.Row():
                    auto_experiment_parallelism = gr.Slider(
                        1,
                        32,
                        value=initial_config.auto_experiment_parallelism,
                        step=1,
                        label="Condition workers",
                    )
                    postprocess_parallelism = gr.Slider(
                        1,
                        16,
                        value=initial_config.postprocess_parallelism,
                        step=1,
                        label="Postprocess chunk workers",
                    )
                    enable_cache = gr.Checkbox(label="Use preprocess/ASR cache", value=True)
                    auto_experiment_saturate_lanes = gr.Checkbox(
                        label="Saturate available lanes",
                        value=initial_config.auto_experiment_saturate_lanes,
                    )
                with gr.Row():
                    auto_experiment_include_models = gr.Checkbox(
                        label="Include model combinations",
                        value=initial_config.auto_experiment_include_models,
                    )
                with gr.Row():
                    auto_experiment_asr_models = gr.Textbox(
                        label="ASR models for Auto Experiment",
                        value=", ".join(initial_config.auto_experiment_asr_models),
                        placeholder="Qwen/Qwen3-ASR-1.7B, ...",
                    )
                    auto_experiment_post_models = gr.Textbox(
                        label="Post models for Auto Experiment",
                        value=", ".join(initial_config.auto_experiment_post_models),
                        placeholder="Qwen/Qwen3.5-9B, ...",
                    )
                with gr.Row():
                    auto_experiment_noise_models = gr.Textbox(
                        label="Noise models for Auto Experiment",
                        value=", ".join(initial_config.auto_experiment_noise_models),
                        placeholder="afftdn, deepfilternet2, deepfilternet2_pf, deepfilternet3, rnnoise",
                    )
                    auto_experiment_rag_embedding_models = gr.Textbox(
                        label="RAG embedding models for Auto Experiment",
                        value=", ".join(initial_config.auto_experiment_rag_embedding_models),
                        placeholder="intfloat/multilingual-e5-base, ...",
                    )
                with gr.Row():
                    auto_experiment_keyword_weights = gr.Textbox(
                        label="Keyword weight sweep",
                        value=", ".join(str(value) for value in initial_config.auto_experiment_keyword_weights),
                    )
                    auto_experiment_noise_strengths = gr.Textbox(
                        label="Noise strength sweep",
                        value=", ".join(str(value) for value in initial_config.auto_experiment_noise_strengths),
                    )
                    auto_experiment_volume_strengths = gr.Textbox(
                        label="Volume strength sweep",
                        value=", ".join(str(value) for value in initial_config.auto_experiment_volume_strengths),
                    )
                with gr.Row():
                    auto_experiment_postprocess_strengths = gr.Textbox(
                        label="Postprocess strength sweep",
                        value=", ".join(str(value) for value in initial_config.auto_experiment_postprocess_strengths),
                    )
                    auto_experiment_rag_strengths = gr.Textbox(
                        label="RAG strength sweep",
                        value=", ".join(str(value) for value in initial_config.auto_experiment_rag_strengths),
                    )
                    auto_experiment_rag_top_ks = gr.Textbox(
                        label="RAG top-k sweep",
                        value=", ".join(str(value) for value in initial_config.auto_experiment_rag_top_ks),
                    )
                with gr.Row():
                    auto_experiment_search_strengths = gr.Textbox(
                        label="Search strength sweep",
                        value=", ".join(str(value) for value in initial_config.auto_experiment_search_strengths),
                    )
            with gr.Accordion("Model server startup", open=True):
                with gr.Row():
                    auto_start_model_servers = gr.Checkbox(
                        label="Start required model servers when Run is pressed",
                        value=initial_config.auto_start_model_servers,
                    )
                    model_residency = gr.Dropdown(
                        [
                            ("All required models stay loaded (fast, high VRAM)", "parallel"),
                            ("All GPUs per stage (reload ASR/POST between stages)", "stage_replicas"),
                            ("One model at a time (slow, low VRAM)", "sequential"),
                        ],
                        value=initial_config.model_residency,
                        label="Model residency",
                    )
                with gr.Row():
                    asr_server_gpu = gr.Textbox(value=initial_config.asr_server_gpu, label="Primary ASR server GPU")
                    post_server_gpu = gr.Textbox(value=initial_config.post_server_gpu, label="Primary post-processing server GPU")
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
                gr.JSON(label="Configured pipeline lanes", value=_pipeline_lane_summary(initial_config))
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
        diff_output = gr.HTML(label="Inline diff")
        with gr.Row():
            metrics_output = gr.JSON(label="Metrics")
            server_output = gr.JSON(label="Model servers")
        with gr.Accordion("Edits", open=False):
            edits_output = gr.JSON(label="Edits")
        with gr.Row():
            preprocess_output = gr.JSON(label="Preprocess")
            preprocessed_audio_output = gr.Audio(
                label="Preprocessed audio preview",
                type="filepath",
                format="wav",
                interactive=False,
                editable=False,
            )
        preprocessed_audio_player_output = gr.HTML(label="Preprocessed audio timeline")
        with gr.Row():
            gpu_output = gr.JSON(label="Server GPU / VRAM status", value=query_gpu_status())
            refresh_gpu_button = gr.Button("Refresh GPU status")

        run_button.click(
            fn=run_from_ui_stream,
            inputs=[
                audio,
                large_audio_file,
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
                asr_chunking_strategy,
                asr_chunk_seconds,
                asr_chunk_padding_seconds,
                asr_silence_threshold_db,
                asr_min_silence_seconds,
                asr_request_timeout_s,
                asr_context_chars,
                asr_chunk_parallelism,
                auto_experiment_mode,
                auto_experiment_coverage,
                auto_experiment_parallelism,
                postprocess_parallelism,
                enable_cache,
                auto_experiment_saturate_lanes,
                auto_experiment_include_models,
                auto_experiment_asr_models,
                auto_experiment_post_models,
                auto_experiment_noise_models,
                auto_experiment_rag_embedding_models,
                auto_experiment_keyword_weights,
                auto_experiment_noise_strengths,
                auto_experiment_volume_strengths,
                auto_experiment_postprocess_strengths,
                auto_experiment_rag_strengths,
                auto_experiment_rag_top_ks,
                auto_experiment_search_strengths,
                initial_config_state,
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
                preprocessed_audio_output,
                preprocessed_audio_player_output,
                gpu_output,
            ],
        )
        preview_preprocess_button.click(
            fn=preview_preprocessed_audio_from_ui,
            inputs=[
                audio,
                large_audio_file,
                enable_noise_reduction,
                noise_reduction_model,
                noise_reduction_strength,
                enable_volume_normalization,
                volume_normalization_strength,
                volume_target_dbfs,
            ],
            outputs=[preprocessed_audio_output, preprocessed_audio_player_output, preprocess_output, progress_output],
        )
        refresh_gpu_button.click(fn=query_gpu_status, outputs=gpu_output)

    demo.queue()
    return demo.launch(**_launch_kwargs(host, port, share))


def run_from_ui_stream(*args):
    started = time.time()
    progress_state: Dict[str, Any] = {"started": started, "current": "Preparing run.", "events": []}
    result_queue: Queue[Tuple[str, Any]] = Queue(maxsize=1)

    def record_progress(message: str) -> None:
        event = {"elapsed_s": time.time() - started, "message": message}
        progress_state["current"] = message
        progress_state["events"] = [*progress_state["events"], event][-RUN_STATUS_RECENT_EVENT_LIMIT:]

    def worker() -> None:
        try:
            result_queue.put(("result", run_from_ui(*args, status_callback=record_progress)))
        except Exception as exc:
            result_queue.put(("error", exc))

    record_progress("Preparing run and checking model servers.")
    thread = Thread(target=worker, daemon=True)
    thread.start()
    gpu_status = query_gpu_status()
    yield _empty_run_output(_format_live_run_status(progress_state, gpu_status), gpu_status)

    while thread.is_alive():
        try:
            kind, payload = result_queue.get(timeout=RUN_STATUS_POLL_INTERVAL_S)
        except Empty:
            gpu_status = query_gpu_status()
            yield _empty_run_output(_format_live_run_status(progress_state, gpu_status), gpu_status)
            continue
        if kind == "result":
            yield payload
            return
        yield _unexpected_run_error(payload)
        return

    try:
        kind, payload = result_queue.get_nowait()
    except Empty:
        yield _unexpected_run_error(RuntimeError("Run worker ended without returning a result."))
        return
    if kind == "result":
        yield payload
    else:
        yield _unexpected_run_error(payload)


def _empty_run_output(status: str, gpu_status: dict) -> RunOutput:
    return "", "", "", {}, [], {}, [], status, None, "", gpu_status


def _unexpected_run_error(exc: BaseException) -> RunOutput:
    return "", "", "", {}, [], {}, [], f"Run failed unexpectedly: {exc}", None, "", query_gpu_status()


def _format_live_run_status(progress_state: Dict[str, Any], gpu_status: dict) -> str:
    elapsed = _format_seconds(time.time() - float(progress_state.get("started", time.time())))
    current = str(progress_state.get("current") or "Running.")
    lines = [
        f"Run in progress. Elapsed: {elapsed}",
        f"Current stage: {current}",
        _gpu_snapshot_line(gpu_status),
        _gpu_process_line(gpu_status),
        "",
        "Recent events:",
    ]
    events = progress_state.get("events") or []
    for event in events[-RUN_STATUS_RECENT_EVENT_LIMIT:]:
        event_elapsed = _format_seconds(float(event.get("elapsed_s", 0.0)))
        lines.append(f"- +{event_elapsed} {event.get('message', '')}")
    lines.extend(
        [
            "",
            "Note: vLLM reserves VRAM when the server starts. 0% GPU util between samples can still be normal while a request is queued, preprocessing, transferring data, or between decode bursts.",
        ]
    )
    return "\n".join(lines)


def _gpu_snapshot_line(gpu_status: dict) -> str:
    if not gpu_status.get("available"):
        return f"GPU snapshot: unavailable ({gpu_status.get('error', 'nvidia-smi failed')})"
    summaries = []
    for gpu in gpu_status.get("gpus", []):
        index = gpu.get("index")
        used = gpu.get("memory_used_mb")
        total = gpu.get("memory_total_mb")
        util = gpu.get("gpu_utilization_percent")
        temp = gpu.get("temperature_c")
        power = _format_power(gpu.get("power_draw_w"), gpu.get("power_limit_w"))
        pstate = gpu.get("performance_state") or "P?"
        summaries.append(f"GPU{index}: util {util}%, VRAM {used}/{total} MiB, power {power}, temp {temp}C, {pstate}")
    return "GPU snapshot: " + ("; ".join(summaries) if summaries else "no GPUs reported")


def _format_power(draw: Any, limit: Any) -> str:
    if isinstance(draw, (int, float)) and isinstance(limit, (int, float)):
        return f"{draw:.0f}/{limit:.0f}W"
    if isinstance(draw, (int, float)):
        return f"{draw:.0f}W"
    return "unknown"


def _gpu_process_line(gpu_status: dict) -> str:
    processes = gpu_status.get("processes") or []
    if not processes:
        return "GPU compute processes: none reported"
    summaries = []
    for process in processes[:6]:
        name = Path(str(process.get("process_name", ""))).name or "process"
        summaries.append(f"pid {process.get('pid')} {name} {process.get('used_memory_mb')} MiB")
    suffix = f"; +{len(processes) - 6} more" if len(processes) > 6 else ""
    return "GPU compute processes: " + "; ".join(summaries) + suffix


def preview_preprocessed_audio_from_ui(
    audio_path: Optional[str],
    large_audio_file: Any,
    enable_noise_reduction: bool,
    noise_reduction_model: str,
    noise_reduction_strength: float,
    enable_volume_normalization: bool,
    volume_normalization_strength: float,
    volume_target_dbfs: float,
) -> Tuple[Optional[str], str, dict, str]:
    audio_path = _resolve_audio_path(audio_path, large_audio_file)
    if not audio_path:
        return None, "", {}, "No audio input provided."
    config = _preprocess_config_from_ui(
        enable_noise_reduction,
        noise_reduction_model,
        noise_reduction_strength,
        enable_volume_normalization,
        volume_normalization_strength,
        volume_target_dbfs,
    )
    try:
        result = preprocess_audio(audio_path, config)
    except Exception as exc:
        return None, "", {}, f"Preprocess preview failed: {exc}"
    preview_path = _preview_audio_path(result.to_dict())
    preview_html = _audio_timeline_html(preview_path, result.to_dict())
    if result.applied:
        status = "Preprocessed audio ready."
    elif result.steps:
        status = "Preprocessing could not be applied; previewing the input audio."
    else:
        status = "No preprocessing selected; previewing the input audio."
    if result.warnings:
        status += "\n" + "\n".join(result.warnings)
    return preview_path, preview_html, result.to_dict(), status


def run_from_ui(
    audio_path: Optional[str],
    large_audio_file: Any,
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
    asr_chunking_strategy: str = "silence",
    asr_chunk_seconds: float = 120.0,
    asr_chunk_padding_seconds: float = 0.5,
    asr_silence_threshold_db: float = -35.0,
    asr_min_silence_seconds: float = 0.6,
    asr_request_timeout_s: float = 300.0,
    asr_context_chars: int = 240,
    asr_chunk_parallelism: int = 1,
    auto_experiment_mode: bool = False,
    auto_experiment_coverage: str = "full_valid",
    auto_experiment_parallelism: int = 1,
    postprocess_parallelism: int = 1,
    enable_cache: bool = True,
    auto_experiment_saturate_lanes: bool = True,
    auto_experiment_include_models: bool = False,
    auto_experiment_asr_models: str = "",
    auto_experiment_post_models: str = "",
    auto_experiment_noise_models: str = "",
    auto_experiment_rag_embedding_models: str = "",
    auto_experiment_keyword_weights: str = "",
    auto_experiment_noise_strengths: str = "",
    auto_experiment_volume_strengths: str = "",
    auto_experiment_postprocess_strengths: str = "",
    auto_experiment_rag_strengths: str = "",
    auto_experiment_rag_top_ks: str = "",
    auto_experiment_search_strengths: str = "",
    base_config_state: Optional[Dict[str, Any]] = None,
    *,
    status_callback: Optional[Callable[[str], None]] = None,
) -> RunOutput:
    audio_path = _resolve_audio_path(audio_path, large_audio_file)
    if not audio_path:
        return "", "", "", {}, [], {}, [], "No audio input provided.", None, "", query_gpu_status()
    reference = _read_reference_from_ui(reference_text, reference_file)
    config_values = _config_from_ui_state(base_config_state).to_dict()
    config_values.update(
        {
            "asr_model": asr_model or "Qwen/Qwen3-ASR-1.7B",
            "post_model": post_model or "Qwen/Qwen3.5-9B",
            "asr_backend": asr_backend,
            "post_backend": post_backend,
            "asr_base_url": asr_base_url or "http://127.0.0.1:18000/v1",
            "post_base_url": post_base_url or "http://127.0.0.1:18001/v1",
            "asr_request_timeout_s": float(asr_request_timeout_s or 300.0),
            "asr_chunking_strategy": asr_chunking_strategy or "silence",
            "asr_chunk_seconds": float(asr_chunk_seconds or 120.0),
            "asr_chunk_padding_seconds": float(asr_chunk_padding_seconds or 0.0),
            "asr_silence_threshold_db": float(asr_silence_threshold_db or -35.0),
            "asr_min_silence_seconds": float(asr_min_silence_seconds or 0.6),
            "asr_context_chars": int(asr_context_chars or 0),
            "asr_chunk_parallelism": int(asr_chunk_parallelism or 1),
            "auto_start_model_servers": bool(auto_start_model_servers),
            "model_residency": model_residency or "parallel",
            "server_start_timeout_s": float(server_start_timeout_s),
            "server_shutdown_timeout_s": float(server_shutdown_timeout_s),
            "server_log_dir": server_log_dir or "outputs/model_servers",
            "asr_server_gpu": asr_server_gpu or "0",
            "post_server_gpu": post_server_gpu or "1",
            "asr_server_host": asr_server_host or "0.0.0.0",
            "post_server_host": post_server_host or "0.0.0.0",
            "asr_server_command": asr_server_command or "",
            "post_server_command": post_server_command or "",
            "enable_preprocess": False,
            "preprocess_model": "none",
            "preprocess_strength": 0.0,
            "postprocess_parallelism": int(postprocess_parallelism or 1),
            "auto_experiment_parallelism": int(auto_experiment_parallelism or 1),
            "auto_experiment_saturate_lanes": bool(auto_experiment_saturate_lanes),
            "auto_experiment_include_models": bool(auto_experiment_include_models),
            "auto_experiment_asr_models": _split_keywords(auto_experiment_asr_models),
            "auto_experiment_post_models": _split_keywords(auto_experiment_post_models),
            "auto_experiment_noise_models": _split_keywords(auto_experiment_noise_models),
            "auto_experiment_rag_embedding_models": _split_keywords(auto_experiment_rag_embedding_models),
            "auto_experiment_keyword_weights": _split_float_grid(auto_experiment_keyword_weights),
            "auto_experiment_noise_strengths": _split_float_grid(auto_experiment_noise_strengths),
            "auto_experiment_volume_strengths": _split_float_grid(auto_experiment_volume_strengths),
            "auto_experiment_postprocess_strengths": _split_float_grid(auto_experiment_postprocess_strengths),
            "auto_experiment_rag_strengths": _split_float_grid(auto_experiment_rag_strengths),
            "auto_experiment_rag_top_ks": _split_int_grid(auto_experiment_rag_top_ks),
            "auto_experiment_search_strengths": _split_float_grid(auto_experiment_search_strengths),
            "asr_cache_enabled": bool(enable_cache),
            "preprocess_cache_enabled": bool(enable_cache),
            "enable_noise_reduction": bool(enable_noise_reduction),
            "noise_reduction_model": noise_reduction_model or "none",
            "noise_reduction_strength": float(noise_reduction_strength),
            "enable_volume_normalization": bool(enable_volume_normalization),
            "volume_normalization_strength": float(volume_normalization_strength),
            "volume_target_dbfs": float(volume_target_dbfs),
            "enable_keyword_bias": bool(enable_keyword_bias),
            "keyword_bias_weight": float(keyword_bias_weight),
            "keywords": _split_keywords(keywords),
            "enable_llm_postprocess": bool(enable_llm),
            "postprocess_strength": float(postprocess_strength),
            "enable_rag": bool(enable_rag),
            "rag_strength": float(rag_strength),
            "rag_top_k": int(rag_top_k),
            "rag_inline_text": rag_text or "",
            "rag_files": _file_paths(rag_files),
            "enable_search": bool(enable_search),
            "search_strength": float(search_strength),
            "search_provider": search_provider or "duckduckgo",
            "search_endpoint": search_endpoint or "",
        }
    )
    config = ExperimentConfig.from_mapping(config_values)
    if auto_experiment_mode:
        try:
            report = run_auto_experiment(
                audio_path=audio_path,
                base_config=config,
                reference_text=reference,
                rag_inline_text=rag_text or "",
                mode=auto_experiment_coverage or "full_valid",
                status_callback=status_callback,
            )
        except Exception as exc:
            return "", "", "", {}, [], {}, [], f"Auto experiment failed: {exc}", None, "", query_gpu_status()
        return (
            "",
            "",
            "",
            report.get("analysis", {}),
            [],
            {"auto_experiment": True, "condition_count": report.get("condition_count")},
            [],
            (
                f"Auto Experiment ID: {report['run_id']}\n"
                f"Coverage: {report['mode']}\n"
                f"Conditions: {report['condition_count']}\n"
                f"Summary: {report['summary_csv']}\n"
                f"Analysis: {report['analysis_json']}\n"
                f"Output: {report['output_dir']}"
            ),
            None,
            "",
            query_gpu_status(),
        )
    try:
        output = PipelineRunner(config, status_callback=status_callback).run(audio_path=audio_path, reference_text=reference)
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
        return "", "", "", {}, [], {}, [], f"Run failed: {exc}{hint}", None, "", query_gpu_status()
    server_lines = _format_server_statuses(output.server_statuses)
    preview_path = _preview_audio_path(output.preprocess)
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
            f"{_format_pipeline_lanes(config)}"
            f"ASR chunking: {config.asr_chunking_strategy} "
            f"({config.asr_chunk_seconds:g}s, timeout {config.asr_request_timeout_s:g}s, "
            f"context {config.asr_context_chars} chars, workers {config.asr_chunk_parallelism})\n"
            f"{server_lines}"
            f"Output: {output.output_dir}\n"
            f"TensorBoard: tensorboard --logdir {config.runs_dir} --port {config.tensorboard_port}\n"
            f"Artifacts: {json.dumps(output.artifacts, ensure_ascii=False)}"
        ),
        preview_path,
        _audio_timeline_html(preview_path, output.preprocess),
        query_gpu_status(),
    )


def _preprocess_config_from_ui(
    enable_noise_reduction: bool,
    noise_reduction_model: str,
    noise_reduction_strength: float,
    enable_volume_normalization: bool,
    volume_normalization_strength: float,
    volume_target_dbfs: float,
) -> ExperimentConfig:
    return ExperimentConfig(
        enable_preprocess=False,
        preprocess_model="none",
        preprocess_strength=0.0,
        enable_noise_reduction=bool(enable_noise_reduction),
        noise_reduction_model=noise_reduction_model or "none",
        noise_reduction_strength=float(noise_reduction_strength),
        enable_volume_normalization=bool(enable_volume_normalization),
        volume_normalization_strength=float(volume_normalization_strength),
        volume_target_dbfs=float(volume_target_dbfs),
    )


def _preview_audio_path(preprocess: dict) -> Optional[str]:
    if isinstance(preprocess, dict) and preprocess.get("steps") and not preprocess.get("applied"):
        return None
    path = preprocess.get("audio_path") if isinstance(preprocess, dict) else None
    if not path:
        return None
    return str(path) if Path(str(path)).exists() else None


def _audio_timeline_html(audio_path: Optional[str], preprocess: dict) -> str:
    if isinstance(preprocess, dict) and not preprocess.get("applied"):
        return ""
    if not audio_path:
        return ""
    path = Path(audio_path)
    if not path.exists():
        return ""
    duration = _preprocess_duration(preprocess)
    duration_label = _format_seconds(duration) if duration is not None else ""
    url = "/gradio_api/file=" + quote(str(path), safe="/")
    separator = "&" if "?" in url else "?"
    url = f"{url}{separator}v={path.stat().st_mtime_ns}"
    escaped_url = escape(url, quote=True)
    escaped_name = escape(path.name)
    duration_html = f'<div class="asrpp-audio-time">Duration: {escape(duration_label)}</div>' if duration_label else ""
    return (
        '<div class="asrpp-audio-preview">'
        f'<audio controls preload="metadata" src="{escaped_url}" style="width:100%;"></audio>'
        f'<div class="asrpp-audio-file">{escaped_name}</div>'
        f"{duration_html}"
        "</div>"
    )


def _preprocess_duration(preprocess: dict) -> Optional[float]:
    if not isinstance(preprocess, dict):
        return None
    steps = preprocess.get("steps")
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if not isinstance(step, dict) or not step.get("applied"):
            continue
        metadata = step.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("duration_seconds"), (int, float)):
            return float(metadata["duration_seconds"])
    return None


def _format_seconds(value: float) -> str:
    total_seconds = max(0, int(round(value)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def _resolve_audio_path(audio_path: Optional[str], large_audio_file: Any = None) -> Optional[str]:
    file_paths = _file_paths(large_audio_file)
    if file_paths:
        return file_paths[0]
    return audio_path


def _launch_kwargs(host: str, port: int, share: bool) -> dict:
    kwargs = {"server_name": host, "server_port": port, "share": share}
    max_file_size = os.environ.get("ASRPP_GRADIO_MAX_FILE_SIZE", "2gb")
    if max_file_size:
        try:
            import gradio as gr  # type: ignore

            supports_max_file_size = "max_file_size" in inspect.signature(gr.Blocks.launch).parameters
        except Exception:
            supports_max_file_size = False
        if supports_max_file_size:
            kwargs["max_file_size"] = max_file_size
    try:
        kwargs["allowed_paths"] = [str((Path.cwd() / "outputs").resolve())]
    except Exception:
        pass
    return kwargs


def _config_from_ui_state(value: Any) -> ExperimentConfig:
    if isinstance(value, ExperimentConfig):
        return ExperimentConfig.from_mapping(value.to_dict())
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ExperimentConfig()
    if isinstance(value, dict):
        return ExperimentConfig.from_mapping(value)
    return ExperimentConfig()


def _pipeline_lane_summary(config: ExperimentConfig) -> Dict[str, Any]:
    lanes = []
    for lane in config.pipeline_lanes:
        if not isinstance(lane, dict):
            continue
        lanes.append(
            {
                "name": lane.get("name", ""),
                "asr_base_url": lane.get("asr_base_url", ""),
                "post_base_url": lane.get("post_base_url", ""),
                "asr_server_gpu": lane.get("asr_server_gpu", lane.get("asr_gpu", "")),
                "post_server_gpu": lane.get("post_server_gpu", lane.get("post_gpu", "")),
                "preprocess_gpu": lane.get("preprocess_gpu", ""),
            }
        )
    return {
        "primary": {
            "asr_base_url": config.asr_base_url,
            "post_base_url": config.post_base_url,
            "asr_server_gpu": config.asr_server_gpu,
            "post_server_gpu": config.post_server_gpu,
            "preprocess_gpu": config.preprocess_gpu,
        },
        "asr_base_urls": list(config.asr_base_urls),
        "post_base_urls": list(config.post_base_urls),
        "stage_server_base_urls": list(config.stage_server_base_urls),
        "stage_server_gpus": list(config.stage_server_gpus),
        "preprocess_gpus": list(config.preprocess_gpus),
        "pipeline_lanes": lanes,
    }


def _format_pipeline_lanes(config: ExperimentConfig) -> str:
    lanes = _pipeline_lane_summary(config)["pipeline_lanes"]
    if config.model_residency == "stage_replicas" and config.stage_server_base_urls:
        pairs = []
        for index, base_url in enumerate(config.stage_server_base_urls):
            gpu = config.stage_server_gpus[index] if index < len(config.stage_server_gpus) else "?"
            preprocess_gpu = config.preprocess_gpus[index] if index < len(config.preprocess_gpus) else gpu
            pairs.append(f"stage_{index}: PRE GPU {preprocess_gpu} -> ASR/POST GPU {gpu} {base_url}")
        return "Stage replicas: " + "; ".join(pairs) + "\n"
    if not lanes:
        return ""
    parts = []
    for lane in lanes:
        name = lane.get("name") or "lane"
        asr_gpu = lane.get("asr_server_gpu") or "?"
        post_gpu = lane.get("post_server_gpu") or "?"
        preprocess_gpu = lane.get("preprocess_gpu") or "?"
        asr_url = lane.get("asr_base_url") or "primary ASR"
        post_url = lane.get("post_base_url") or "primary post"
        parts.append(
            f"{name}: PRE GPU {preprocess_gpu} -> ASR GPU {asr_gpu} {asr_url} -> POST GPU {post_gpu} {post_url}"
        )
    return "Pipeline lanes: " + "; ".join(parts) + "\n"


def _canonical_noise_reduction_model(value: str) -> str:
    normalized = str(value or "none").strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "false": "none",
        "afftdn": "afftdn",
        "ffmpeg_afftdn": "afftdn",
        "basic": "afftdn",
        "built_in": "afftdn",
        "denoise": "afftdn",
        "rnnoise": "rnnoise",
        "deepfilternet2": "deepfilternet2",
        "deep_filter_net2": "deepfilternet2",
        "deepfilternet2_pf": "deepfilternet2_pf",
        "deep_filter_net2_pf": "deepfilternet2_pf",
        "deepfilternet3": "deepfilternet3",
        "deep_filter_net3": "deepfilternet3",
        "bs_roformer": "bs-roformer",
        "bsroformer": "bs-roformer",
    }
    return aliases.get(normalized, normalized.replace("_", "-") if normalized.startswith("bs_") else normalized)


def _split_keywords(value: str) -> List[str]:
    return [item.strip() for item in (value or "").replace("\n", ",").split(",") if item.strip()]


def _split_float_grid(value: str) -> List[float]:
    values: List[float] = []
    for item in _split_keywords(value):
        try:
            number = max(0.0, min(1.0, float(item)))
        except ValueError:
            continue
        if number > 0.0 and number not in values:
            values.append(number)
    return values


def _split_int_grid(value: str) -> List[int]:
    values: List[int] = []
    for item in _split_keywords(value):
        try:
            number = max(1, int(item))
        except ValueError:
            continue
        if number not in values:
            values.append(number)
    return values


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
