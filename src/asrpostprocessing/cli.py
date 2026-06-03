from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .config import load_config
from .doctor import doctor_as_json, has_failures, run_doctor
from .asr_quality_compare import run_asr_quality_compare
from .auto_experiment import preview_auto_experiment, run_auto_experiment
from .pipeline import PipelineRunner, read_reference
from .sweep import run_sweep, shard_manifest
from .transcript_quality import build_transcript_quality_report


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="asrpp", description="ASR post-processing lab CLI")
    subcommands = parser.add_subparsers(dest="command", required=True)

    ui_parser = subcommands.add_parser("ui", help="Launch Gradio GUI")
    ui_parser.add_argument("--config")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true")

    run_parser = subcommands.add_parser("run", help="Run one audio/reference sample")
    run_parser.add_argument("--audio", required=True)
    run_parser.add_argument("--reference")
    run_parser.add_argument("--reference-text")
    run_parser.add_argument("--config")
    run_parser.add_argument("--run-id")
    _add_backend_overrides(run_parser)

    sweep_parser = subcommands.add_parser("sweep", help="Run a weight grid over a manifest CSV")
    sweep_parser.add_argument("--manifest", required=True)
    sweep_parser.add_argument("--config")
    sweep_parser.add_argument("--jobs", type=int, help="Parallel sweep workers. Defaults to config sweep_parallelism.")

    shard_parser = subcommands.add_parser("manifest-shard", help="Split a manifest CSV into round-robin shards")
    shard_parser.add_argument("--manifest", required=True)
    shard_parser.add_argument("--num-shards", type=int, required=True)
    shard_parser.add_argument("--out", required=True)
    shard_parser.add_argument("--prefix", default="shard")

    auto_parser = subcommands.add_parser("auto-experiment", help="Run the valid toggle-condition matrix for one audio sample")
    auto_parser.add_argument("--audio")
    auto_parser.add_argument("--reference")
    auto_parser.add_argument("--reference-text")
    auto_parser.add_argument("--config")
    auto_parser.add_argument(
        "--mode",
        choices=["core_ablation", "full_valid", "full_strength_sweep"],
    )
    auto_parser.add_argument("--preview", action="store_true", help="Print the generated condition matrix without running it")
    _add_backend_overrides(auto_parser)

    asr_quality_parser = subcommands.add_parser("asr-quality", help="Compare ASR-only quality across chunk/preprocess settings")
    asr_quality_parser.add_argument("--audio", required=True)
    asr_quality_parser.add_argument("--config")
    asr_quality_parser.add_argument("--output")
    asr_quality_parser.add_argument("--chunk-seconds", type=float, action="append")
    asr_quality_parser.add_argument("--strategy", choices=["fixed", "silence", "none"], action="append")
    asr_quality_parser.add_argument("--preprocess-mode", choices=["none", "configured", "both"], default="both")
    asr_quality_parser.add_argument("--sample-seconds", type=float)
    asr_quality_parser.add_argument("--sample-start-s", type=float, action="append")
    _add_backend_overrides(asr_quality_parser)

    transcript_quality_parser = subcommands.add_parser(
        "transcript-quality",
        help="Analyze existing raw/corrected transcript text files",
    )
    transcript_quality_parser.add_argument("--raw", required=True)
    transcript_quality_parser.add_argument("--corrected")
    transcript_quality_parser.add_argument("--config")
    transcript_quality_parser.add_argument("--output")
    transcript_quality_parser.add_argument("--keyword", action="append", dest="keywords")

    tb_parser = subcommands.add_parser("tensorboard", help="Show or launch TensorBoard for runs/")
    tb_parser.add_argument("--logdir", default="runs")
    tb_parser.add_argument("--port", type=int, default=6006)
    tb_parser.add_argument("--launch", action="store_true")

    doctor_parser = subcommands.add_parser("doctor", help="Check local experiment readiness")
    doctor_parser.add_argument("--config", default="configs/cuda.yaml")
    doctor_parser.add_argument("--check-endpoints", action="store_true")
    doctor_parser.add_argument("--json", action="store_true")

    gpu_parser = subcommands.add_parser("gpu", help="Show NVIDIA GPU and VRAM status")
    gpu_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "ui":
        from .ui import launch_ui

        launch_ui(config_path=args.config, host=args.host, port=args.port, share=args.share)
        return 0
    if args.command == "run":
        config = load_config(args.config, overrides=_backend_overrides(args))
        reference = args.reference_text or read_reference(args.reference)
        with contextlib.redirect_stdout(sys.stderr):
            output = PipelineRunner(config).run(audio_path=args.audio, reference_text=reference, run_id=args.run_id)
        print(json.dumps(output.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "sweep":
        config = load_config(args.config)
        summary_path = run_sweep(args.manifest, config, jobs=args.jobs)
        print(str(summary_path))
        return 0
    if args.command == "manifest-shard":
        paths = shard_manifest(args.manifest, args.num_shards, args.out, prefix=args.prefix)
        print(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2))
        return 0
    if args.command == "auto-experiment":
        config = load_config(args.config, overrides=_backend_overrides(args))
        mode = args.mode or config.auto_experiment_coverage
        if args.preview:
            preview = preview_auto_experiment(config, mode=mode)
            print(
                json.dumps(
                    {
                        "mode": preview["mode"],
                        "condition_count": preview["condition_count"],
                        "case_count": preview["case_count"],
                        "asr_cache_group_count": preview["asr_cache_group_count"],
                        "model_axis_enabled": preview["model_axis_enabled"],
                        "conditions": [condition.to_dict() for condition in preview["conditions"]],
                        "cases": [case.to_dict() for case in preview["cases"]],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if not args.audio:
            parser.error("auto-experiment --audio is required unless --preview is used")
        reference = args.reference_text or read_reference(args.reference)
        with contextlib.redirect_stdout(sys.stderr):
            report = run_auto_experiment(
                audio_path=args.audio,
                base_config=config,
                reference_text=reference,
                mode=mode,
            )
        output_path = Path(report["output_dir"]) / "auto_experiment_result.json"
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(output_path))
        return 0
    if args.command == "asr-quality":
        config = load_config(args.config, overrides=_backend_overrides(args))
        with contextlib.redirect_stdout(sys.stderr):
            output_path = run_asr_quality_compare(
                audio_path=args.audio,
                base_config=config,
                output_path=args.output,
                chunk_seconds=args.chunk_seconds,
                strategies=args.strategy,
                preprocess_mode=args.preprocess_mode,
                sample_seconds=args.sample_seconds,
                sample_start_s=args.sample_start_s or 0.0,
            )
        print(str(output_path))
        return 0
    if args.command == "transcript-quality":
        config = load_config(args.config)
        if args.keywords:
            config.keywords = args.keywords
        report = build_transcript_quality_report(args.raw, config, corrected_path=args.corrected)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(str(output_path))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "tensorboard":
        command = ["tensorboard", "--logdir", args.logdir, "--port", str(args.port)]
        if args.launch:
            if not shutil.which("tensorboard"):
                raise RuntimeError("tensorboard executable was not found.")
            subprocess.run(command, check=True)
        else:
            print(" ".join(command))
        return 0
    if args.command == "doctor":
        config = load_config(args.config)
        checks = run_doctor(config, check_endpoints=args.check_endpoints)
        if args.json:
            print(doctor_as_json(checks))
        else:
            for check in checks:
                print(f"{check.status.upper():5} {check.name}: {check.detail}")
        return 1 if has_failures(checks) else 0
    if args.command == "gpu":
        from .gpu_status import query_gpu_status

        status = query_gpu_status()
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(_format_gpu_status(status))
        return 0 if status.get("available") else 1
    parser.print_help()
    return 1


def _format_gpu_status(status: Dict[str, Any]) -> str:
    if not status.get("available"):
        return f"GPU status unavailable: {status.get('error', 'unknown error')}"
    lines = ["GPU / VRAM status:"]
    for gpu in status.get("gpus", []):
        lines.append(
            "GPU {index}: {name} | VRAM {used}/{total} MiB ({percent}%) | util {util}% | temp {temp}C".format(
                index=gpu.get("index"),
                name=gpu.get("name"),
                used=gpu.get("memory_used_mb"),
                total=gpu.get("memory_total_mb"),
                percent=gpu.get("memory_used_percent"),
                util=gpu.get("gpu_utilization_percent"),
                temp=gpu.get("temperature_c"),
            )
        )
    processes = status.get("processes", [])
    if processes:
        lines.append("GPU processes:")
        for process in processes:
            lines.append(
                "- pid={pid} memory={memory} MiB process={name}".format(
                    pid=process.get("pid"),
                    memory=process.get("used_memory_mb"),
                    name=process.get("process_name"),
                )
            )
    else:
        lines.append("GPU processes: none")
    for warning in status.get("warnings", []):
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def _add_backend_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asr-backend", choices=["mock", "vllm_chat", "qwen_asr_vllm", "qwen_asr_transformers"])
    parser.add_argument("--post-backend", choices=["mock", "vllm_openai"])
    parser.add_argument("--asr-model")
    parser.add_argument("--post-model")
    parser.add_argument("--asr-base-url")
    parser.add_argument("--post-base-url")
    parser.add_argument("--keyword", action="append", dest="keywords")
    parser.add_argument("--enable-keyword-bias", action="store_true")
    parser.add_argument("--keyword-bias-weight", type=float)
    parser.add_argument("--postprocess-strength", type=float)
    parser.add_argument("--enable-rag", action="store_true")
    parser.add_argument(
        "--rag-file",
        action="append",
        dest="rag_files",
        help="RAG source file: .txt, .md, .csv, .json, or .pdf",
    )
    parser.add_argument("--rag-strength", type=float)
    parser.add_argument("--enable-search", action="store_true")
    parser.add_argument("--search-provider", choices=["duckduckgo", "endpoint", "none"])
    parser.add_argument("--search-endpoint")
    parser.add_argument("--enable-noise-reduction", action="store_true")
    parser.add_argument(
        "--noise-reduction-model",
        choices=[
            "none",
            "afftdn",
            "ffmpeg_afftdn",
            "RNNoise",
            "rnnoise",
            "DeepFilterNet2",
            "deepfilternet2",
            "DeepFilterNet2-PF",
            "deepfilternet2_pf",
            "DeepFilterNet3",
            "deepfilternet3",
            "BS-RoFormer",
            "bs_roformer",
        ],
    )
    parser.add_argument("--noise-reduction-command")
    parser.add_argument("--noise-reduction-strength", type=float)
    parser.add_argument("--enable-volume-normalization", action="store_true")
    parser.add_argument("--volume-normalization-strength", type=float)
    parser.add_argument("--volume-target-dbfs", type=float)
    parser.add_argument("--asr-request-timeout-s", type=float)
    parser.add_argument("--asr-chunking-strategy", choices=["silence", "fixed", "none"])
    parser.add_argument("--asr-chunk-seconds", type=float)
    parser.add_argument("--asr-chunk-padding-seconds", type=float)
    parser.add_argument("--asr-silence-threshold-db", type=float)
    parser.add_argument("--asr-min-silence-seconds", type=float)
    parser.add_argument("--asr-context-chars", type=int)
    parser.add_argument("--asr-chunk-parallelism", type=int)
    parser.add_argument("--auto-start-model-servers", action="store_true")
    parser.add_argument("--no-auto-start-model-servers", action="store_true")
    residency_group = parser.add_mutually_exclusive_group()
    residency_group.add_argument("--model-residency", choices=["parallel", "sequential", "stage_replicas"])
    residency_group.add_argument("--sequential-model-loading", action="store_true")
    residency_group.add_argument("--parallel-model-loading", action="store_true")
    parser.add_argument("--server-shutdown-timeout-s", type=float)
    parser.add_argument("--postprocess-parallelism", type=int)
    parser.add_argument("--auto-experiment-parallelism", type=int)
    parser.add_argument("--auto-experiment-include-models", action="store_true")
    parser.add_argument("--auto-experiment-asr-model", action="append", dest="auto_experiment_asr_models")
    parser.add_argument("--auto-experiment-post-model", action="append", dest="auto_experiment_post_models")
    parser.add_argument("--auto-experiment-noise-model", action="append", dest="auto_experiment_noise_models")
    parser.add_argument(
        "--auto-experiment-rag-embedding-model",
        action="append",
        dest="auto_experiment_rag_embedding_models",
    )
    parser.add_argument("--auto-experiment-keyword-weight", type=float, action="append", dest="auto_experiment_keyword_weights")
    parser.add_argument("--auto-experiment-noise-strength", type=float, action="append", dest="auto_experiment_noise_strengths")
    parser.add_argument("--auto-experiment-volume-strength", type=float, action="append", dest="auto_experiment_volume_strengths")
    parser.add_argument("--auto-experiment-postprocess-strength", type=float, action="append", dest="auto_experiment_postprocess_strengths")
    parser.add_argument("--auto-experiment-rag-strength", type=float, action="append", dest="auto_experiment_rag_strengths")
    parser.add_argument("--auto-experiment-rag-top-k", type=int, action="append", dest="auto_experiment_rag_top_ks")
    parser.add_argument("--auto-experiment-search-strength", type=float, action="append", dest="auto_experiment_search_strengths")
    parser.add_argument("--no-auto-experiment-saturate-lanes", action="store_true")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--enable-cache", action="store_true")
    cache_group.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--cache-dir")


def _backend_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "asr_backend": args.asr_backend,
        "post_backend": args.post_backend,
        "asr_model": args.asr_model,
        "post_model": args.post_model,
        "asr_base_url": args.asr_base_url,
        "post_base_url": args.post_base_url,
        "keyword_bias_weight": args.keyword_bias_weight,
        "postprocess_strength": args.postprocess_strength,
        "rag_strength": args.rag_strength,
        "search_provider": args.search_provider,
        "search_endpoint": args.search_endpoint,
        "noise_reduction_model": args.noise_reduction_model,
        "noise_reduction_command": args.noise_reduction_command,
        "noise_reduction_strength": args.noise_reduction_strength,
        "volume_normalization_strength": args.volume_normalization_strength,
        "volume_target_dbfs": args.volume_target_dbfs,
        "asr_request_timeout_s": args.asr_request_timeout_s,
        "asr_chunking_strategy": args.asr_chunking_strategy,
        "asr_chunk_seconds": args.asr_chunk_seconds,
        "asr_chunk_padding_seconds": args.asr_chunk_padding_seconds,
        "asr_silence_threshold_db": args.asr_silence_threshold_db,
        "asr_min_silence_seconds": args.asr_min_silence_seconds,
        "asr_context_chars": args.asr_context_chars,
        "asr_chunk_parallelism": args.asr_chunk_parallelism,
        "model_residency": args.model_residency,
        "server_shutdown_timeout_s": args.server_shutdown_timeout_s,
        "postprocess_parallelism": args.postprocess_parallelism,
        "auto_experiment_parallelism": args.auto_experiment_parallelism,
        "auto_experiment_asr_models": args.auto_experiment_asr_models,
        "auto_experiment_post_models": args.auto_experiment_post_models,
        "auto_experiment_noise_models": args.auto_experiment_noise_models,
        "auto_experiment_rag_embedding_models": args.auto_experiment_rag_embedding_models,
        "auto_experiment_keyword_weights": args.auto_experiment_keyword_weights,
        "auto_experiment_noise_strengths": args.auto_experiment_noise_strengths,
        "auto_experiment_volume_strengths": args.auto_experiment_volume_strengths,
        "auto_experiment_postprocess_strengths": args.auto_experiment_postprocess_strengths,
        "auto_experiment_rag_strengths": args.auto_experiment_rag_strengths,
        "auto_experiment_rag_top_ks": args.auto_experiment_rag_top_ks,
        "auto_experiment_search_strengths": args.auto_experiment_search_strengths,
        "cache_dir": args.cache_dir,
    }
    if args.auto_experiment_include_models:
        overrides["auto_experiment_include_models"] = True
    if args.no_auto_experiment_saturate_lanes:
        overrides["auto_experiment_saturate_lanes"] = False
    if args.enable_cache:
        overrides["asr_cache_enabled"] = True
        overrides["preprocess_cache_enabled"] = True
    if args.disable_cache:
        overrides["asr_cache_enabled"] = False
        overrides["preprocess_cache_enabled"] = False
    if args.auto_start_model_servers:
        overrides["auto_start_model_servers"] = True
    if args.no_auto_start_model_servers:
        overrides["auto_start_model_servers"] = False
    if args.sequential_model_loading:
        overrides["model_residency"] = "sequential"
    if args.parallel_model_loading:
        overrides["model_residency"] = "parallel"
    if args.keywords:
        overrides["keywords"] = args.keywords
    if args.enable_keyword_bias:
        overrides["enable_keyword_bias"] = True
    if args.enable_rag:
        overrides["enable_rag"] = True
    if args.rag_files:
        overrides["rag_files"] = args.rag_files
    if args.enable_search:
        overrides["enable_search"] = True
    if args.enable_noise_reduction:
        overrides["enable_noise_reduction"] = True
    if args.enable_volume_normalization:
        overrides["enable_volume_normalization"] = True
    return overrides
