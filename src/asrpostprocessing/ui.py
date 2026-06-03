from __future__ import annotations

import inspect
import json
import os
import shutil
import time
from html import escape
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from .auto_experiment import run_auto_experiment
from .cache import cache_file_by_sha256
from .config import ExperimentConfig
from .gpu_status import query_gpu_status
from .pipeline import PipelineRunner
from .preprocess import preprocess_audio
from .text import make_character_diff_html, make_diff_export_document, make_diff_html

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
                    placeholder="Leave empty to use the default command; custom commands can use {python}, {vllm}, {model}",
                )
                post_server_command = gr.Textbox(
                    value=initial_config.post_server_command,
                    label="Custom post-processing server command",
                    placeholder="Leave empty to use the default command; custom commands can use {python}, {vllm}, {model}",
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
    config = _preprocess_config_from_ui(
        enable_noise_reduction,
        noise_reduction_model,
        noise_reduction_strength,
        enable_volume_normalization,
        volume_normalization_strength,
        volume_target_dbfs,
    )
    audio_path, upload_cache = _resolve_audio_input(audio_path, large_audio_file, config)
    if not audio_path:
        return None, "", {}, "No audio input provided."
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
    upload_cache_status = _format_upload_cache_status(upload_cache)
    if upload_cache_status:
        status = f"{upload_cache_status}{status}"
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
    base_config = _config_from_ui_state(base_config_state)
    selected_asr_model = _text_or_default(asr_model, base_config.asr_model)
    selected_post_model = _text_or_default(post_model, base_config.post_model)
    selected_asr_base_url = _text_or_default(asr_base_url, base_config.asr_base_url)
    selected_post_base_url = _text_or_default(post_base_url, base_config.post_base_url)
    include_model_axis = bool(auto_experiment_include_models)
    config_values = base_config.to_dict()
    config_values.update(
        {
            "asr_model": selected_asr_model,
            "post_model": selected_post_model,
            "asr_backend": asr_backend,
            "post_backend": post_backend,
            "asr_base_url": selected_asr_base_url,
            "post_base_url": selected_post_base_url,
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
            "auto_experiment_include_models": include_model_axis,
            "auto_experiment_asr_models": _auto_experiment_model_grid(
                auto_experiment_asr_models,
                base_config.auto_experiment_asr_models,
                selected_asr_model,
                bool(auto_experiment_mode),
                include_model_axis,
            ),
            "auto_experiment_post_models": _auto_experiment_model_grid(
                auto_experiment_post_models,
                base_config.auto_experiment_post_models,
                selected_post_model,
                bool(auto_experiment_mode),
                include_model_axis,
            ),
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
    audio_path, upload_cache = _resolve_audio_input(audio_path, large_audio_file, config)
    if not audio_path:
        return "", "", "", {}, [], {}, [], "No audio input provided.", None, "", query_gpu_status()
    reference = _read_reference_from_ui(reference_text, reference_file, audio_path, upload_cache)
    upload_cache_status = _format_upload_cache_status(upload_cache)
    if upload_cache_status and status_callback:
        status_callback(upload_cache_status.strip())
    for message in _apply_runtime_saturation(config):
        if status_callback:
            status_callback(message)
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
        metrics_payload = dict(report.get("analysis", {}))
        metrics_payload["strict_audit"] = report.get("audit", {})
        metrics_payload["best_methods"] = metrics_payload.get("best_methods", [])
        metrics_payload["per_case_cer_wer"] = _auto_experiment_metric_rows(report)
        if not reference:
            metrics_payload["reference_required"] = "CER/WER require a reference transcript or .txt reference file."
        diff_html = _auto_experiment_diff_html(report, reference)
        return (
            "",
            "",
            diff_html,
            metrics_payload,
            [],
            {"auto_experiment": True, "condition_count": report.get("condition_count")},
            [],
            (
                f"{upload_cache_status}"
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
    metrics_payload = _metrics_payload(output.metrics.to_dict(), reference)
    diff_html = _display_diff_html(reference, output.raw.text, output.correction.corrected_text)
    diff_export_path = Path(output.output_dir) / "diff_export.html"
    _write_diff_export(
        diff_export_path,
        diff_html,
        title=f"Transcript Diff: {output.run_id}",
        metadata={"Run ID": output.run_id, "View": "Reference/Raw -> Corrected"},
    )
    diff_html = _prepend_diff_export_panel(diff_html, diff_export_path)
    _record_diff_export_artifact(output, diff_export_path)
    return (
        output.raw.text,
        output.correction.corrected_text,
        diff_html,
        metrics_payload,
        [edit.to_dict() for edit in output.correction.edits],
        output.preprocess,
        output.server_statuses,
        (
            f"{upload_cache_status}"
            f"Run ID: {output.run_id}\n"
            f"Model residency: {config.model_residency}\n"
            f"{_format_pipeline_lanes(config)}"
            f"ASR chunking: {config.asr_chunking_strategy} "
            f"({config.asr_chunk_seconds:g}s, timeout {config.asr_request_timeout_s:g}s, "
            f"context {config.asr_context_chars} chars, workers {config.asr_chunk_parallelism})\n"
            f"{server_lines}"
            f"Output: {output.output_dir}\n"
            f"Diff export: {diff_export_path}\n"
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


def _display_diff_html(reference_text: Optional[str], raw_text: str, corrected_text: str) -> str:
    sections: List[str] = []
    if reference_text:
        sections.append(
            _diff_section(
                "Reference -> Corrected",
                make_diff_html(reference_text, corrected_text, "Reference", "Corrected", show_error_monitor=True),
            )
        )
    if not reference_text or raw_text != corrected_text:
        sections.append(_diff_section("Raw -> Corrected", make_diff_html(raw_text, corrected_text, "Raw", "Corrected")))
    if not sections:
        sections.append(_diff_section("Raw -> Corrected", make_diff_html(raw_text, corrected_text, "Raw", "Corrected")))
    return '<div class="asrpp-diff-stack">' + "\n".join(sections) + "</div>"


def _write_diff_export(path: Path, body_html: str, title: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_diff_export_document(body_html, title=title, metadata=metadata), encoding="utf-8")
    return path


def _record_diff_export_artifact(output: Any, export_path: Path) -> None:
    output.artifacts["diff_export_html"] = str(export_path)
    result_path = Path(str(output.artifacts.get("result") or ""))
    if not result_path.exists():
        return
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    artifacts["diff_export_html"] = str(export_path)
    payload["artifacts"] = artifacts
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepend_diff_export_panel(body_html: str, export_path: Path) -> str:
    escaped_path = escape(str(export_path))
    href = _gradio_file_href(export_path)
    return (
        '<div class="asrpp-diff-export-panel" '
        'style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;'
        'margin:0 0 10px 0;padding:8px 10px;border:1px solid var(--border-color-primary,#d0d7de);'
        'border-radius:var(--block-radius,8px);background:var(--background-fill-secondary,rgba(127,127,127,.08));'
        'font-size:13px;">'
        f'<a href="{href}" target="_blank" download '
        'style="font-weight:650;text-decoration:none;color:var(--link-text-color,#2563eb);">Export HTML</a>'
        f'<span style="color:var(--body-text-color-subdued,#6e7781);overflow-wrap:anywhere;">{escaped_path}</span>'
        "</div>"
        f"{body_html}"
    )


def _gradio_file_href(path: Path) -> str:
    resolved = path.resolve()
    url = "/gradio_api/file=" + quote(str(resolved), safe="/")
    try:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}v={resolved.stat().st_mtime_ns}"
    except OSError:
        return url


def _diff_section(title: str, body: str) -> str:
    return (
        '<section class="asrpp-diff-section">'
        f'<h3 style="font-size:14px;margin:8px 0 6px 0;">{escape(title)}</h3>'
        f"{body}"
        "</section>"
    )


def _auto_experiment_diff_html(report: Dict[str, Any], reference_text: Optional[str]) -> str:
    analysis = report.get("analysis") if isinstance(report.get("analysis"), dict) else {}
    best = analysis.get("best_by_cer") or analysis.get("best_by_wer") or analysis.get("baseline")
    lines = [
        '<div class="asrpp-auto-diff-summary">',
        "<h3>Auto Experiment Diff</h3>",
    ]
    if not reference_text:
        lines.append("<p>CER/WER and reference diff require a reference transcript or .txt reference file.</p>")
    if isinstance(best, dict) and best:
        lines.append(
            "<p>"
            f"Best/comparable case: {_auto_experiment_case_tags_html(best)} "
            f"CER={escape(_format_rate_cell(best.get('cer_normalized_no_space')))} "
            f"WER={escape(_format_rate_cell(best.get('wer_eojeol')))}"
            "</p>"
        )
        output_dir = best.get("output_dir")
        if output_dir:
            lines.append(f"<p>Per-case diff artifact: {escape(str(Path(str(output_dir)) / 'diff.html'))}</p>")
    else:
        lines.append("<p>No comparable row is available yet. Check the summary CSV after the run completes.</p>")
    lines.append(_auto_experiment_best_methods_html(report))
    lines.append(_auto_experiment_audit_html(report))
    lines.append(_auto_experiment_results_table(report, reference_text))
    lines.append(f"<p>Summary CSV: {escape(str(report.get('summary_csv') or ''))}</p>")
    lines.append("</div>")
    body = "\n".join(lines)
    export_path = _auto_experiment_diff_export_path(report)
    if export_path is None:
        return body
    _write_diff_export(
        export_path,
        body,
        title=f"Auto Experiment Diff: {report.get('run_id') or 'run'}",
        metadata={"Run ID": report.get("run_id"), "View": "Auto Experiment Diff"},
    )
    return _prepend_diff_export_panel(body, export_path)


def _auto_experiment_diff_export_path(report: Dict[str, Any]) -> Optional[Path]:
    output_dir = str(report.get("output_dir") or "").strip()
    if not output_dir:
        return None
    return Path(output_dir) / "auto_experiment_diff_export.html"


def _auto_experiment_audit_html(report: Dict[str, Any]) -> str:
    audit = report.get("audit")
    if not isinstance(audit, dict):
        analysis = report.get("analysis") if isinstance(report.get("analysis"), dict) else {}
        audit = analysis.get("audit") if isinstance(analysis.get("audit"), dict) else {}
    if not audit:
        return "<p>Strict audit unavailable for this run.</p>"
    gates = audit.get("gates") if isinstance(audit.get("gates"), dict) else {}
    gate_rows = "\n".join(
        "<tr>"
        f"<td>{escape(_audit_gate_label(key))}</td>"
        f'<td class="{escape(_audit_gate_class(value))}">{escape(_audit_gate_value(value))}</td>'
        "</tr>"
        for key, value in gates.items()
    )
    return f"""
<section class="asrpp-auto-audit">
  <h3>Strict Experiment Audit</h3>
  <div class="asrpp-audit-grid">
    <div><span>Verdict</span><strong>{escape(str(audit.get("verdict") or "unknown"))}</strong></div>
    <div><span>Rows</span><strong>{escape(str(audit.get("row_count", "")))}/{escape(str(audit.get("expected_case_count", "")))}</strong></div>
    <div><span>Failed</span><strong>{escape(str(audit.get("failed_count", "")))}</strong></div>
    <div><span>CER/WER Rows</span><strong>{escape(str(audit.get("cer_wer_row_count", "")))}</strong></div>
    <div><span>Baseline CER</span><strong>{escape(_format_rate_cell(audit.get("baseline_cer_normalized_no_space")))}</strong></div>
    <div><span>Best CER Δ</span><strong>{escape(_format_signed_rate_cell(audit.get("best_cer_improvement_vs_baseline")))}</strong></div>
    <div><span>Best WER Δ</span><strong>{escape(_format_signed_rate_cell(audit.get("best_wer_improvement_vs_baseline")))}</strong></div>
    <div><span>ASR Cache Groups</span><strong>{escape(str(audit.get("observed_asr_cache_group_count", "")))}/{escape(str(audit.get("expected_asr_cache_group_count", "")))}</strong></div>
    <div><span>Actual ASR Cache Keys</span><strong>{escape(str(audit.get("actual_asr_cache_key_count", "")))}</strong></div>
    <div><span>Peak GPU</span><strong>{escape(_format_percent_metric_cell(audit.get("peak_gpu_utilization_percent")))}</strong></div>
    <div><span>Peak VRAM MB</span><strong>{escape(_format_metric_cell(audit.get("peak_vram_mb")))}</strong></div>
    <div class="asrpp-audit-wide"><span>ASR URLs</span><strong>{escape(_join_audit_values(audit.get("observed_asr_base_urls")))}</strong></div>
    <div class="asrpp-audit-wide"><span>Post URLs</span><strong>{escape(_join_audit_values(audit.get("observed_post_base_urls")))}</strong></div>
    <div><span>PRE GPUs</span><strong>{escape(_join_audit_values(audit.get("observed_preprocess_gpus")))}</strong></div>
  </div>
  <p>{escape(str(audit.get("conclusion") or ""))}</p>
  <table class="asrpp-audit-gates">
    <tbody>
      {gate_rows}
    </tbody>
  </table>
</section>
"""


def _auto_experiment_best_methods_html(report: Dict[str, Any]) -> str:
    rows = _auto_experiment_best_method_rows(report)
    if not rows:
        return "<p>No best-method summary is available yet.</p>"
    body = "\n".join(
        "<tr>"
        f"<td><span class=\"asrpp-best-badge\">{escape(str(row.get('badge') or 'Best'))}</span></td>"
        f"<td>{escape(str(row.get('method') or ''))}</td>"
        f'<td>{_auto_experiment_case_tags_html(row)}</td>'
        f"<td class=\"metric\">{escape(_format_rate_cell(row.get('cer_normalized_no_space')))}</td>"
        f"<td class=\"metric\">{escape(_format_rate_cell(row.get('wer_eojeol')))}</td>"
        f"<td class=\"metric\">{escape(_format_signed_rate_cell(row.get('delta_cer_vs_baseline')))}</td>"
        f"<td class=\"metric\">{escape(_format_signed_rate_cell(row.get('delta_wer_vs_baseline')))}</td>"
        f"<td class=\"metric\">{escape(_format_count_cell(row.get('rag_context_count')))}</td>"
        f"<td class=\"metric\">{escape(_format_count_cell(row.get('search_result_count')))}</td>"
        "</tr>"
        for row in rows
    )
    return f"""
<section class="asrpp-best-methods">
  <h3>Best Methods</h3>
  <table>
    <thead>
      <tr>
        <th>Best</th>
        <th>Method</th>
        <th>Case</th>
        <th>CER</th>
        <th>WER</th>
        <th>ΔCER</th>
        <th>ΔWER</th>
        <th>RAG Ctx</th>
        <th>Search Hits</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</section>
"""


def _auto_experiment_results_table(report: Dict[str, Any], reference_text: Optional[str] = None) -> str:
    rows = _auto_experiment_metric_rows(report, reference_text=reference_text)
    if not rows:
        return "<p>No per-condition CER/WER rows are available.</p>"
    body = "\n".join(_auto_experiment_result_row_html(index + 1, row) for index, row in enumerate(rows))
    return f"""
<style>
  .asrpp-auto-results {{ margin-top: 12px; max-height: 520px; overflow: auto; border: 1px solid #3f3f46; border-radius: 6px; }}
  .asrpp-auto-results table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .asrpp-auto-results th, .asrpp-auto-results td {{ padding: 7px 8px; border-bottom: 1px solid #3f3f46; vertical-align: top; }}
  .asrpp-auto-results th {{ position: sticky; top: 0; background: #18181b; text-align: left; z-index: 1; }}
  .asrpp-auto-results td.metric {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .asrpp-auto-results td.features {{ min-width: 150px; }}
  .asrpp-auto-case-tags {{ display: flex; flex-wrap: wrap; gap: 4px; min-width: 132px; }}
  .asrpp-auto-case-tag {{ display: inline-flex; align-items: center; border: 1px solid #71717a; border-radius: 999px; padding: 1px 7px; background: #27272a; color: #f4f4f5; font-size: 12px; line-height: 1.55; white-space: nowrap; }}
  .asrpp-auto-case-tag.baseline {{ border-color: #93c5fd; color: #bfdbfe; }}
  .asrpp-auto-case-tag.model {{ border-color: #c4b5fd; color: #ddd6fe; }}
  .asrpp-auto-condition-subtitle {{ display: block; margin-top: 2px; color: #a1a1aa; font-size: 11px; }}
  .asrpp-auto-results .failed {{ color: #fca5a5; }}
  .asrpp-best-methods {{ margin-top: 12px; border: 1px solid #3f3f46; border-radius: 6px; padding: 10px 12px; }}
  .asrpp-best-methods table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .asrpp-best-methods th, .asrpp-best-methods td {{ padding: 6px 7px; border-top: 1px solid #3f3f46; text-align: left; }}
  .asrpp-best-badge {{ display: inline-block; border: 1px solid #86efac; color: #86efac; border-radius: 4px; padding: 1px 5px; font-size: 12px; white-space: nowrap; }}
  .asrpp-best-list {{ display: flex; gap: 4px; flex-wrap: wrap; min-width: 78px; }}
  .asrpp-auto-audit {{ margin-top: 12px; border: 1px solid #3f3f46; border-radius: 6px; padding: 10px 12px; }}
  .asrpp-audit-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin: 8px 0; }}
  .asrpp-audit-grid div {{ border: 1px solid #3f3f46; border-radius: 4px; padding: 7px 8px; }}
  .asrpp-audit-grid span {{ display: block; color: #a1a1aa; font-size: 12px; }}
  .asrpp-audit-grid strong {{ display: block; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }}
  .asrpp-audit-grid .asrpp-audit-wide {{ grid-column: span 2; }}
  .asrpp-audit-gates {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }}
  .asrpp-audit-gates td {{ padding: 5px 6px; border-top: 1px solid #3f3f46; }}
  .asrpp-audit-pass {{ color: #86efac; }}
  .asrpp-audit-fail {{ color: #fca5a5; }}
  .asrpp-route {{ max-width: 150px; overflow-wrap: anywhere; font-size: 12px; color: #d4d4d8; }}
  .asrpp-case-diff-cell {{ min-width: 110px; }}
  .asrpp-case-diff summary {{ cursor: pointer; display: inline-flex; align-items: center; border: 1px solid #71717a; border-radius: 4px; padding: 3px 7px; font-size: 12px; color: #f4f4f5; background: #27272a; }}
  .asrpp-case-diff-body {{ margin-top: 8px; min-width: min(760px, 80vw); max-width: 980px; }}
  .asrpp-case-diff-subtitle {{ margin: 10px 0 6px; color: #d4d4d8; font-size: 12px; font-weight: 650; }}
        </style>
<h3>Auto Experiment CER/WER by condition</h3>
<div class="asrpp-auto-results">
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Best</th>
        <th>Case</th>
        <th>Condition</th>
        <th>Enabled</th>
        <th>CER</th>
        <th>WER</th>
        <th>ΔCER vs baseline</th>
        <th>ΔWER vs baseline</th>
        <th>ASR URL</th>
        <th>Post URL</th>
        <th>PRE GPU</th>
        <th>Peak GPU</th>
        <th>ASR Cache</th>
        <th>RAG Ctx</th>
        <th>Search Hits</th>
        <th>Diff</th>
        <th>Risk/Error</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</div>
"""


def _auto_experiment_result_row_html(index: int, row: Dict[str, Any]) -> str:
    risk_or_error = row.get("error") or row.get("risk") or ""
    risk_class = ' class="failed"' if row.get("error") else ""
    return (
        "<tr>"
        f'<td class="metric">{index}</td>'
        f'<td><span class="asrpp-best-list">{_auto_experiment_best_badge_html(row.get("best_badges"))}</span></td>'
        f'<td>{_auto_experiment_case_tags_html(row)}</td>'
        f"<td>{_auto_experiment_condition_label_html(row)}</td>"
        f'<td class="features">{escape(str(row.get("enabled_features") or "baseline"))}</td>'
        f'<td class="metric">{escape(str(row.get("cer") or "n/a"))}</td>'
        f'<td class="metric">{escape(str(row.get("wer") or "n/a"))}</td>'
        f'<td class="metric">{escape(str(row.get("delta_cer_vs_baseline") or "n/a"))}</td>'
        f'<td class="metric">{escape(str(row.get("delta_wer_vs_baseline") or "n/a"))}</td>'
        f'<td class="asrpp-route">{escape(str(row.get("asr_base_url") or ""))}</td>'
        f'<td class="asrpp-route">{escape(str(row.get("post_base_url") or ""))}</td>'
        f'<td class="metric">{escape(str(row.get("preprocess_gpu") or ""))}</td>'
        f'<td class="metric">{escape(str(row.get("peak_gpu_utilization_percent") or ""))}</td>'
        f'<td class="metric">{escape(str(row.get("asr_cache_hit") or ""))}</td>'
        f'<td class="metric">{escape(str(row.get("rag_context_count") or ""))}</td>'
        f'<td class="metric">{escape(str(row.get("search_result_count") or ""))}</td>'
        f'<td class="asrpp-case-diff-cell">{row.get("diff_html") or ""}</td>'
        f"<td{risk_class}>{escape(str(risk_or_error))}</td>"
        "</tr>"
    )


def _auto_experiment_metric_rows(report: Dict[str, Any], reference_text: Optional[str] = None) -> List[Dict[str, Any]]:
    raw_rows = report.get("rows")
    if not isinstance(raw_rows, list):
        return []
    best_badges = _auto_experiment_best_badges(report)
    rows = [
        _auto_experiment_metric_row(row, best_badges.get(str(row.get("case_id") or ""), []), reference_text)
        for row in raw_rows
        if isinstance(row, dict)
    ]
    return sorted(rows, key=_auto_experiment_metric_sort_key)


def _auto_experiment_metric_row(
    row: Dict[str, Any],
    best_badges: Optional[List[str]] = None,
    reference_text: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "case_id": row.get("case_id") or row.get("condition_id") or "",
        "condition_id": row.get("condition_id") or "",
        "label": row.get("label") or row.get("condition_id") or "",
        "best_badges": best_badges or [],
        "enabled_features": _auto_experiment_enabled_features(row),
        "cer": _format_rate_cell(row.get("cer_normalized_no_space")),
        "wer": _format_rate_cell(row.get("wer_eojeol")),
        "delta_cer_vs_baseline": _format_signed_rate_cell(row.get("delta_cer_vs_baseline")),
        "delta_wer_vs_baseline": _format_signed_rate_cell(row.get("delta_wer_vs_baseline")),
        "asr_base_url": row.get("asr_base_url") or "",
        "post_base_url": row.get("post_base_url") or "",
        "preprocess_gpu": row.get("preprocess_gpu") or "",
        "peak_gpu_utilization_percent": _format_percent_metric_cell(row.get("peak_gpu_utilization_percent")),
        "asr_cache_hit": row.get("asr_cache_hit") if row.get("asr_cache_hit") not in (None, "") else "",
        "rag_context_count": _format_count_cell(row.get("rag_context_count")),
        "search_result_count": _format_count_cell(row.get("search_result_count")),
        "diff_html": _auto_experiment_case_diff_html(row, reference_text),
        "risk": row.get("risk") or "",
        "error": row.get("error") or "",
    }


def _auto_experiment_case_diff_html(row: Dict[str, Any], reference_text: Optional[str] = None) -> str:
    if row.get("error"):
        return ""
    output_dir = str(row.get("output_dir") or "").strip()
    if not output_dir:
        return ""
    raw_path = Path(output_dir) / "raw_transcript.txt"
    corrected_path = Path(output_dir) / "corrected_transcript.txt"
    if not raw_path.exists() or not corrected_path.exists():
        return ""
    try:
        raw_text = raw_path.read_text(encoding="utf-8")
        corrected_text = corrected_path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if reference_text:
        diff = (
            '<div class="asrpp-case-diff-subtitle">Reference CER/WER</div>'
            + make_diff_html(reference_text, corrected_text, "Reference", "Corrected", show_error_monitor=True)
            + '<div class="asrpp-case-diff-subtitle">Raw -> Corrected edits</div>'
            + make_character_diff_html(raw_text, corrected_text, "Raw", "Corrected")
        )
    else:
        diff = make_character_diff_html(raw_text, corrected_text, "Raw", "Corrected")
    case_id = escape(str(row.get("case_id") or row.get("condition_id") or "case"))
    return (
        f'<details class="asrpp-case-diff">'
        f"<summary>Diff</summary>"
        f'<div class="asrpp-case-diff-body" aria-label="Character-level diff for {case_id}">{diff}</div>'
        "</details>"
    )


def _auto_experiment_metric_sort_key(row: Dict[str, Any]) -> Tuple[int, float, float, str]:
    cer = _metric_sort_value(row.get("cer"))
    wer = _metric_sort_value(row.get("wer"))
    failed = 1 if row.get("error") else 0
    return failed, cer, wer, str(row.get("case_id") or "")


def _auto_experiment_best_method_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    analysis = report.get("analysis") if isinstance(report.get("analysis"), dict) else {}
    methods = analysis.get("best_methods")
    if isinstance(methods, list) and methods:
        return [method for method in methods if isinstance(method, dict)]
    fallback = []
    for badge, key in [
        ("Best CER", "best_by_cer"),
        ("Best WER", "best_by_wer"),
        ("Best speed/quality", "best_latency_quality_tradeoff"),
    ]:
        row = analysis.get(key)
        if not isinstance(row, dict) or not row:
            continue
        fallback.append(
            {
                "badge": badge,
                "case_id": row.get("case_id") or "",
                "condition_id": row.get("condition_id") or "",
                "method": _auto_experiment_enabled_features(row),
                "cer_normalized_no_space": row.get("cer_normalized_no_space"),
                "wer_eojeol": row.get("wer_eojeol"),
                "delta_cer_vs_baseline": row.get("delta_cer_vs_baseline"),
                "delta_wer_vs_baseline": row.get("delta_wer_vs_baseline"),
                "rag_context_count": row.get("rag_context_count"),
                "search_result_count": row.get("search_result_count"),
            }
        )
    return fallback


def _auto_experiment_best_badges(report: Dict[str, Any]) -> Dict[str, List[str]]:
    badges: Dict[str, List[str]] = {}
    for method in _auto_experiment_best_method_rows(report):
        case_id = str(method.get("case_id") or "")
        badge = str(method.get("badge") or "")
        if not case_id or not badge:
            continue
        existing = badges.setdefault(case_id, [])
        if badge not in existing:
            existing.append(badge)
    return badges


def _auto_experiment_best_badge_html(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "".join(f'<span class="asrpp-best-badge">{escape(str(item))}</span>' for item in value if str(item).strip())


def _auto_experiment_enabled_features(row: Dict[str, Any]) -> str:
    features = []
    feature_map = [
        ("keyword_bias_enabled", "Keyword"),
        ("noise_reduction_enabled", "Noise"),
        ("volume_normalization_enabled", "Volume"),
        ("llm_postprocess_enabled", "LLM"),
        ("rag_enabled", "RAG"),
        ("search_enabled", "Search"),
    ]
    for key, label in feature_map:
        if _truthy(row.get(key)):
            features.append(label)
    return " + ".join(features) if features else "Baseline"


def _auto_experiment_case_tags_html(row: Dict[str, Any]) -> str:
    raw_case_id = str(row.get("case_id") or row.get("condition_id") or "").strip()
    tags = _auto_experiment_case_tags(row)
    if not tags:
        tags = [raw_case_id or "Case"]
    spans = []
    for tag in tags:
        css_class = "asrpp-auto-case-tag"
        lowered = tag.lower()
        if "baseline" in lowered:
            css_class += " baseline"
        elif "model" in lowered:
            css_class += " model"
        spans.append(f'<span class="{css_class}">{escape(tag)}</span>')
    title = f' title="{escape(raw_case_id)}"' if raw_case_id else ""
    return f'<span class="asrpp-auto-case-tags"{title}>{"".join(spans)}</span>'


def _auto_experiment_case_tags(row: Dict[str, Any]) -> List[str]:
    case_id = str(row.get("case_id") or "").strip()
    condition_id = str(row.get("condition_id") or "").strip()
    enabled_features = str(row.get("enabled_features") or "").strip()
    method = str(row.get("method") or "").strip()
    label = str(row.get("label") or "").strip()
    tags: List[str] = []

    if condition_id == "baseline" or case_id.startswith("baseline") or label in {"Baseline", "All off baseline"}:
        tags.append("Baseline")
    elif enabled_features and enabled_features.lower() != "baseline":
        tags.extend(_split_feature_tags(enabled_features))
    elif method and method.lower() != "baseline":
        tags.extend(_split_feature_tags(method))
    elif label and label not in {"Baseline", "All off baseline"}:
        tags.extend(_split_feature_tags(label.split("(", 1)[0]))

    tag_parts_source = condition_id or _strip_case_model_suffix(case_id)
    for part in [item for item in tag_parts_source.split("__") if item]:
        parsed = _auto_experiment_condition_part_tag(part)
        if parsed and parsed not in tags:
            tags.append(parsed)
    if _has_case_model_suffix(case_id) and "Model" not in tags:
        tags.append("Model")
    return tags


def _split_feature_tags(value: str) -> List[str]:
    aliases = {
        "keyword bias": "Keyword",
        "keyword": "Keyword",
        "noise reduction": "Noise",
        "noise": "Noise",
        "volume normalization": "Volume",
        "volume": "Volume",
        "llm": "LLM",
        "rag": "RAG",
        "search": "Search",
    }
    tags: List[str] = []
    for raw_part in value.replace(",", "+").split("+"):
        normalized = " ".join(raw_part.strip().lower().split())
        if not normalized:
            continue
        tag = aliases.get(normalized) or raw_part.strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _auto_experiment_condition_part_tag(part: str) -> str:
    feature_tags = {
        "baseline": "Baseline",
        "keyword": "Keyword",
        "noise": "Noise",
        "volume": "Volume",
        "llm": "LLM",
        "rag": "RAG",
        "search": "Search",
    }
    if part in feature_tags:
        return feature_tags[part]
    if part.startswith("kw"):
        return f"KW {_display_sweep_value(part[2:])}"
    if part.startswith("noise") and len(part) > len("noise"):
        return f"Noise {_display_sweep_value(part[len('noise') :])}"
    if part.startswith("vol"):
        return f"Vol {_display_sweep_value(part[3:])}"
    if part.startswith("post"):
        return f"Post {_display_sweep_value(part[4:])}"
    if part.startswith("rag") and len(part) > len("rag"):
        return f"RAG {_display_sweep_value(part[3:])}"
    if part.startswith("topk"):
        return f"Top-k {part[4:]}"
    if part.startswith("search"):
        return f"Search {_display_sweep_value(part[6:])}"
    if part.startswith("nmodel_"):
        return "Noise model"
    if part.startswith("emb_"):
        return "RAG emb"
    return ""


def _display_sweep_value(value: str) -> str:
    text = str(value or "").strip().replace("p", ".")
    return text or "sweep"


def _auto_experiment_condition_label_html(row: Dict[str, Any]) -> str:
    condition_id = str(row.get("condition_id") or "").strip()
    label = str(row.get("label") or condition_id or "").strip()
    if condition_id == "baseline" or label in {"Baseline", "All off baseline"}:
        label = "Baseline (all toggles off)"
    if not label:
        label = "Condition"
    subtitle = ""
    case_id = str(row.get("case_id") or "").strip()
    if _has_case_model_suffix(case_id):
        subtitle = '<span class="asrpp-auto-condition-subtitle">model variant</span>'
    return f"{escape(label)}{subtitle}"


def _has_case_model_suffix(case_id: str) -> bool:
    return "__model_" in case_id or "_model_" in case_id


def _strip_case_model_suffix(case_id: str) -> str:
    for marker in ("__model_", "_model_"):
        if marker in case_id:
            return case_id.split(marker, 1)[0]
    return case_id


def _format_metric_cell(value: Any) -> str:
    number = _metric_float_or_none(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def _format_signed_metric_cell(value: Any) -> str:
    number = _metric_float_or_none(value)
    if number is None:
        return ""
    return f"{number:+.4f}"


def _format_rate_cell(value: Any) -> str:
    number = _metric_float_or_none(value)
    if number is None:
        return ""
    return f"{number * 100.0:.4f}%"


def _format_signed_rate_cell(value: Any) -> str:
    number = _metric_float_or_none(value)
    if number is None:
        return ""
    return f"{number * 100.0:+.4f}%"


def _format_percent_metric_cell(value: Any) -> str:
    formatted = _format_metric_cell(value)
    return f"{formatted}%" if formatted else ""


def _format_count_cell(value: Any) -> str:
    number = _metric_float_or_none(value)
    if number is None:
        return ""
    return str(int(number))


def _join_audit_values(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value if str(item).strip()) or ""
    return str(value or "")


def _audit_gate_label(key: str) -> str:
    labels = {
        "reference_provided": "Reference provided",
        "all_expected_cases_finished": "All expected cases finished",
        "condition_coverage_complete": "Condition coverage complete",
        "no_failed_cases": "No failed cases",
        "baseline_present": "Baseline present",
        "cer_wer_available_for_all_rows": "CER/WER available for all rows",
        "asr_cache_groups_observed": "ASR cache groups observed",
    }
    return labels.get(key, key.replace("_", " ").strip().title())


def _audit_gate_class(value: Any) -> str:
    return "asrpp-audit-pass" if _truthy(value) else "asrpp-audit-fail"


def _audit_gate_value(value: Any) -> str:
    return "PASS" if _truthy(value) else "FAIL"


def _metric_sort_value(value: Any) -> float:
    number = _metric_float_or_none(value)
    return number if number is not None else 999999.0


def _metric_float_or_none(value: Any) -> Optional[float]:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _metrics_payload(metrics: Dict[str, Any], reference_text: Optional[str]) -> Dict[str, Any]:
    payload = dict(metrics)
    payload["reference_provided"] = bool(reference_text)
    if not reference_text:
        payload["reference_required"] = "CER/WER require a reference transcript or .txt reference file."
    return payload


def _format_seconds(value: float) -> str:
    total_seconds = max(0, int(round(value)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def _resolve_audio_path(audio_path: Optional[str], large_audio_file: Any = None) -> Optional[str]:
    resolved, _ = _resolve_audio_input(audio_path, large_audio_file, ExperimentConfig(upload_cache_enabled=False))
    return resolved


def _resolve_audio_input(
    audio_path: Optional[str],
    large_audio_file: Any = None,
    config: Optional[ExperimentConfig] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    file_paths = _file_paths(large_audio_file)
    resolved = file_paths[0] if file_paths else audio_path
    if not resolved:
        return None, {}
    config = config or ExperimentConfig()
    if not bool(getattr(config, "upload_cache_enabled", True)):
        return resolved, {"enabled": False, "path": resolved}
    path = Path(str(resolved))
    if not path.exists() or not path.is_file():
        return resolved, {}
    try:
        cached = cache_file_by_sha256(path, getattr(config, "upload_cache_dir", "outputs/upload_cache"), "audio")
    except Exception as exc:
        return resolved, {"enabled": True, "source_path": str(path), "error": str(exc)}
    payload = cached.to_dict()
    payload["enabled"] = True
    sidecar_path = _mirror_reference_sidecar(path, Path(cached.cached_path))
    if sidecar_path is not None:
        payload["reference_sidecar_path"] = str(sidecar_path)
    return cached.cached_path, payload


def _format_upload_cache_status(cache: Dict[str, Any]) -> str:
    if not cache or not cache.get("enabled"):
        return ""
    if cache.get("error"):
        return f"Audio upload cache skipped: {cache['error']}\n"
    path = str(cache.get("cached_path") or "")
    action = "hit" if cache.get("cache_hit") else "stored"
    size_mb = _format_mebibytes(cache.get("size_bytes"))
    size_part = f" ({size_mb})" if size_mb else ""
    return f"Audio upload cache {action}: {path}{size_part}\n"


def _format_mebibytes(value: Any) -> str:
    try:
        size_bytes = float(value)
    except (TypeError, ValueError):
        return ""
    if size_bytes < 0:
        return ""
    return f"{size_bytes / (1024.0 * 1024.0):.1f} MiB"


def _mirror_reference_sidecar(source_audio: Path, cached_audio: Path) -> Optional[Path]:
    source_sidecar = source_audio.with_suffix(".txt")
    if not source_sidecar.exists() or not source_sidecar.is_file():
        return None
    target_sidecar = cached_audio.with_suffix(".txt")
    if target_sidecar.exists() and target_sidecar.stat().st_mtime_ns >= source_sidecar.stat().st_mtime_ns:
        return target_sidecar
    target_sidecar.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source_sidecar, target_sidecar)
    except Exception:
        return None
    return target_sidecar


def _apply_runtime_saturation(config: ExperimentConfig) -> List[str]:
    if not bool(getattr(config, "auto_experiment_saturate_lanes", True)):
        return []
    lane_count = _runtime_lane_count(config)
    if lane_count <= 1:
        return []
    messages: List[str] = []
    if int(getattr(config, "asr_context_chars", 0) or 0) <= 0 and int(getattr(config, "asr_chunk_parallelism", 1) or 1) < lane_count:
        previous = int(config.asr_chunk_parallelism)
        config.asr_chunk_parallelism = lane_count
        messages.append(f"Runtime lane saturation: ASR chunk workers {previous} -> {config.asr_chunk_parallelism}.")
    if int(getattr(config, "postprocess_parallelism", 1) or 1) < lane_count:
        previous = int(config.postprocess_parallelism)
        config.postprocess_parallelism = lane_count
        messages.append(f"Runtime lane saturation: postprocess workers {previous} -> {config.postprocess_parallelism}.")
    if int(getattr(config, "auto_experiment_parallelism", 1) or 1) < lane_count:
        previous = int(config.auto_experiment_parallelism)
        config.auto_experiment_parallelism = lane_count
        messages.append(f"Runtime lane saturation: condition workers {previous} -> {config.auto_experiment_parallelism}.")
    return messages


def _runtime_lane_count(config: ExperimentConfig) -> int:
    if config.model_residency == "stage_replicas":
        return max(
            1,
            len([item for item in config.stage_server_base_urls if str(item).strip()]),
            len([item for item in config.stage_server_gpus if str(item).strip()]),
            len([item for item in config.preprocess_gpus if str(item).strip()]),
        )
    lane_count = len([lane for lane in (config.pipeline_lanes or []) if isinstance(lane, dict)])
    return max(
        1,
        lane_count,
        len([item for item in config.asr_base_urls if str(item).strip()]),
        len([item for item in config.post_base_urls if str(item).strip()]),
    )


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


def _text_or_default(value: str, fallback: str) -> str:
    return str(value).strip() if str(value or "").strip() else str(fallback or "").strip()


def _auto_experiment_model_grid(
    value: str,
    configured_values: List[str],
    selected_model: str,
    auto_mode: bool,
    include_models: bool,
) -> List[str]:
    models = _split_keywords(value)
    if not models and auto_mode and include_models:
        models = [str(item).strip() for item in configured_values if str(item).strip()]
        if selected_model:
            models.append(selected_model)
    deduped: List[str] = []
    for model in models:
        if model and model not in deduped:
            deduped.append(model)
    return deduped


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


def _read_reference_from_ui(
    reference_text: str,
    reference_file: Any,
    audio_path: Optional[str] = None,
    upload_cache: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if reference_text and reference_text.strip():
        return reference_text.strip()
    paths = _file_paths(reference_file)
    if paths:
        return Path(paths[0]).read_text(encoding="utf-8").strip()
    for candidate in _reference_sidecar_candidates(audio_path, upload_cache or {}):
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if text:
            return text
    return None


def _reference_sidecar_candidates(audio_path: Optional[str], upload_cache: Dict[str, Any]) -> List[Path]:
    candidates: List[Path] = []
    direct_sidecar = upload_cache.get("reference_sidecar_path")
    if direct_sidecar:
        path = Path(str(direct_sidecar))
        if path.exists():
            candidates.append(path)
    for value in [upload_cache.get("source_path"), audio_path, upload_cache.get("cached_path")]:
        if not value:
            continue
        path = Path(str(value)).with_suffix(".txt")
        if path.exists() and path not in candidates:
            candidates.append(path)
    return candidates


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
