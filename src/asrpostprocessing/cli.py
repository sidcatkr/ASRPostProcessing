from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from typing import Any, Dict, Optional

from .config import load_config
from .doctor import doctor_as_json, has_failures, run_doctor
from .pipeline import PipelineRunner, read_reference
from .sweep import run_sweep


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

    tb_parser = subcommands.add_parser("tensorboard", help="Show or launch TensorBoard for runs/")
    tb_parser.add_argument("--logdir", default="runs")
    tb_parser.add_argument("--port", type=int, default=6006)
    tb_parser.add_argument("--launch", action="store_true")

    doctor_parser = subcommands.add_parser("doctor", help="Check local experiment readiness")
    doctor_parser.add_argument("--config", default="configs/cuda.yaml")
    doctor_parser.add_argument("--check-endpoints", action="store_true")
    doctor_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "ui":
        from .ui import launch_ui

        launch_ui(config_path=args.config, host=args.host, port=args.port, share=args.share)
        return 0
    if args.command == "run":
        config = load_config(args.config, overrides=_backend_overrides(args))
        reference = args.reference_text or read_reference(args.reference)
        output = PipelineRunner(config).run(audio_path=args.audio, reference_text=reference, run_id=args.run_id)
        print(json.dumps(output.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "sweep":
        config = load_config(args.config)
        summary_path = run_sweep(args.manifest, config)
        print(str(summary_path))
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
    parser.print_help()
    return 1


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
    parser.add_argument("--enable-rag", action="store_true")
    parser.add_argument("--rag-file", action="append", dest="rag_files")
    parser.add_argument("--rag-strength", type=float)
    parser.add_argument("--enable-search", action="store_true")
    parser.add_argument("--search-provider", choices=["duckduckgo", "endpoint", "none"])
    parser.add_argument("--search-endpoint")
    parser.add_argument("--auto-start-model-servers", action="store_true")
    parser.add_argument("--no-auto-start-model-servers", action="store_true")


def _backend_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides = {
        "asr_backend": args.asr_backend,
        "post_backend": args.post_backend,
        "asr_model": args.asr_model,
        "post_model": args.post_model,
        "asr_base_url": args.asr_base_url,
        "post_base_url": args.post_base_url,
        "keyword_bias_weight": args.keyword_bias_weight,
        "rag_strength": args.rag_strength,
        "search_provider": args.search_provider,
        "search_endpoint": args.search_endpoint,
    }
    if args.auto_start_model_servers:
        overrides["auto_start_model_servers"] = True
    if args.no_auto_start_model_servers:
        overrides["auto_start_model_servers"] = False
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
    return overrides
