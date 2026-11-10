"""Command line interface for CaRINE MBPP benchmarking sweeps."""

import argparse
import os
import sys
import time
from typing import List
import pandas as pd

from e1.evaluator import evaluate_mbpp_model, load_mbpp_dataset
from e1.ollama_client import OllamaClient
from e1.telemetry import get_hardware_info


def sanitize_filename(name: str) -> str:
    """Replaces unsafe characters in model name for filenames."""
    return name.replace(":", "_").replace("/", "_").replace(".", "_")


def run_sweep(
    models: List[str],
    num_samples: int = 50,
    output_dir: str = "results",
    timeout_sec: int = 5,
    unload_after_eval: bool = True,
) -> pd.DataFrame:
    """Executes benchmark sweep across all specified models and saves CSV outputs."""
    os.makedirs(output_dir, exist_ok=True)
    hw = get_hardware_info()
    print("=" * 70)
    print("CaRINE BENCHMARK: QUANTIZATION HARDWARE & REASONING SWEEP")
    print(f"Host Hardware: {hw['physical_cores']}C/{hw['logical_cores']}T, Total RAM: {hw['total_memory_gib']} GiB")
    print(f"Models to evaluate: {models}")
    print(f"Sample count per model: {num_samples} MBPP tasks")
    print(f"Subprocess timeout: {timeout_sec}s")
    print(f"Results destination: {output_dir}")
    print("=" * 70)

    client = OllamaClient()
    available_models = client.list_models()
    print(f"[*] Locally detected Ollama models: {available_models}")

    # Load dataset once
    print("[*] Loading MBPP (sanitized, test split)...")
    dataset = load_mbpp_dataset()
    print(f"[✓] Dataset loaded with {len(dataset)} total tasks.")

    all_summaries = []

    for model_name in models:
        slug = sanitize_filename(model_name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        df_tasks, summary = evaluate_mbpp_model(
            model_name=model_name,
            dataset=dataset,
            num_samples=num_samples,
            client=client,
            timeout_sec=timeout_sec,
            unload_after_eval=unload_after_eval,
            verbose=True,
        )

        # Save per-task detailed results
        task_csv_path = os.path.join(output_dir, f"mbpp_tasks_{slug}_{num_samples}tasks_{timestamp}.csv")
        df_tasks.to_csv(task_csv_path, index=False)
        print(f"    [+] Saved task-level metrics: {task_csv_path}")

        all_summaries.append(summary)

    # Save aggregate summary
    df_summary = pd.DataFrame(all_summaries)
    summary_csv_path = os.path.join(output_dir, f"mbpp_summary_{num_samples}tasks_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 70)
    print(df_summary.to_markdown(index=False))
    print(f"\n[✓] Summary saved to: {summary_csv_path}")

    return df_summary


def main():
    parser = argparse.ArgumentParser(
        description="CaRINE Quantization Benchmark CLI for Ollama on MBPP"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:3b-text-q8_0", "llama3.2:3b"],
        help="Model tags to benchmark in Ollama (default: llama3.2:3b-text-q8_0 llama3.2:3b)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of tasks to evaluate from sanitized MBPP test split (default: 50)",
    )
    default_out = "e1/results" if os.path.isdir("e1") else "results"
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_out,
        help=f"Directory to save CSV and summary reports (default: {default_out})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Subprocess timeout for code execution in seconds (default: 5)",
    )
    parser.add_argument(
        "--no-unload",
        action="store_true",
        help="Do not unload model after each benchmark run",
    )

    args = parser.parse_args()

    run_sweep(
        models=args.models,
        num_samples=args.samples,
        output_dir=args.output_dir,
        timeout_sec=args.timeout,
        unload_after_eval=not args.no_unload,
    )


if __name__ == "__main__":
    main()

