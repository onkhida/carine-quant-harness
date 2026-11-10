"""Reasoning retention (MBPP pass@1) and hardware performance evaluation loop."""

import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from datasets import load_dataset

from e1.ollama_client import OllamaClient
from e1.prompts import build_mbpp_prompt, extract_python_code, extract_target_function_name
from e1.sandbox import sandbox_execute_code
from e1.telemetry import get_ollama_rss_gib


def load_mbpp_dataset(split: str = "test"):
    """Loads Google's sanitized MBPP benchmark dataset with offline cache support."""
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        return load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    except Exception:
        import os
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        return load_dataset("google-research-datasets/mbpp", "sanitized", split=split)


def evaluate_mbpp_model(
    model_name: str,
    dataset=None,
    num_samples: int = 50,
    client: Optional[OllamaClient] = None,
    timeout_sec: int = 5,
    unload_after_eval: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates an Ollama model on a slice of the MBPP dataset.
    Extracts hardware efficiency metrics (throughput, TTFT/prompt speed, peak RSS)
    and evaluates task retention (pass@1).
    """
    if client is None:
        client = OllamaClient()

    if dataset is None:
        if verbose:
            print("[*] Loading MBPP (sanitized) dataset...")
        dataset = load_mbpp_dataset()

    sample_count = min(num_samples, len(dataset))
    eval_slice = dataset.select(range(sample_count))

    if verbose:
        print(f"\n{'='*70}")
        print(f"[*] Starting MBPP Benchmark: {model_name} ({sample_count} tasks)")
        print(f"{'='*70}")

    # Cold-start mitigation: warm-up query to page model weights into memory
    if verbose:
        print("    [+] Running warm-up query (cold-start mitigation)...", end="", flush=True)
    _ = client.generate(model_name, "def warm_up(): pass", num_predict=16)
    if verbose:
        print(" [DONE]")

    task_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(eval_slice):
        task_id = item["task_id"]
        problem_desc = item.get("prompt") or item.get("text", "")
        test_assertions = item.get("test_list", [])
        test_imports = item.get("test_imports", [])

        # Step 1: Dynamically extract target function name from assertions
        expected_fn = extract_target_function_name(test_assertions)

        # Step 2: Build signature-enforced prompt
        prompt = build_mbpp_prompt(problem_desc, function_name=expected_fn)

        # Step 3: Query Ollama
        gen_metrics = client.generate(
            model=model_name,
            prompt=prompt,
            temperature=0.0,
            num_predict=256,
        )

        # Step 4: Capture Ollama process resident memory footprint (RSS)
        rss_gib = get_ollama_rss_gib()

        # Step 5: Sandbox code extraction & execution
        raw_response = gen_metrics["response"]
        code = extract_python_code(raw_response)
        exec_res = sandbox_execute_code(
            code=code,
            assertions=test_assertions,
            test_imports=test_imports,
            timeout_sec=timeout_sec,
        )

        status_str = exec_res["status"]
        is_pass = 1 if status_str == "PASS" else 0

        record = {
            "model": model_name,
            "task_index": idx,
            "task_id": task_id,
            "function_name": expected_fn or "unknown",
            "status": status_str,
            "passed": is_pass,
            "eval_rate_tok_sec": round(gen_metrics["eval_rate_tok_sec"], 2),
            "eval_count": gen_metrics["eval_count"],
            "eval_duration_sec": round(gen_metrics["eval_duration_sec"], 3),
            "prompt_eval_rate_tok_sec": round(gen_metrics["prompt_eval_rate_tok_sec"], 2),
            "prompt_eval_count": gen_metrics["prompt_eval_count"],
            "prompt_eval_duration_sec": round(gen_metrics["prompt_eval_duration_sec"], 3),
            "total_duration_sec": round(gen_metrics["total_duration_sec"], 3),
            "rss_gib": round(rss_gib, 3),
            "generated_code": code,
        }
        task_records.append(record)

        if verbose:
            short_status = status_str.split()[0]
            tok_s = gen_metrics["eval_rate_tok_sec"]
            p_tok_s = gen_metrics["prompt_eval_rate_tok_sec"]
            fn_label = f"[{expected_fn}]" if expected_fn else ""
            print(
                f"    Task {idx+1:02d}/{sample_count} (ID {task_id:>3}) {fn_label:<20}: "
                f"[{short_status:<7}] -> Gen: {tok_s:>5.1f} tok/s | Prompt: {p_tok_s:>6.1f} tok/s | RSS: {rss_gib:.2f} GiB"
            )

    df_tasks = pd.DataFrame(task_records)

    # Compute aggregate summary statistics
    passed_total = df_tasks["passed"].sum()
    pass_rate = (passed_total / sample_count) * 100.0 if sample_count > 0 else 0.0
    avg_gen_rate = df_tasks["eval_rate_tok_sec"].mean() if sample_count > 0 else 0.0
    avg_prompt_rate = df_tasks["prompt_eval_rate_tok_sec"].mean() if sample_count > 0 else 0.0
    avg_latency = df_tasks["total_duration_sec"].mean() if sample_count > 0 else 0.0
    peak_rss = df_tasks["rss_gib"].max() if sample_count > 0 else 0.0

    summary = {
        "Model": model_name,
        "Total Tasks": sample_count,
        "Passed Tasks": int(passed_total),
        "MBPP pass@1 (%)": round(pass_rate, 2),
        "Gen Throughput (tok/s)": round(avg_gen_rate, 2),
        "Prompt Eval Speed (tok/s)": round(avg_prompt_rate, 2),
        "Avg Latency (s)": round(avg_latency, 2),
        "Peak RSS (GiB)": round(peak_rss, 2),
    }

    if verbose:
        print(f"\n[✓] Finished {model_name}: pass@1 = {pass_rate:.1f}% ({passed_total}/{sample_count}), "
              f"Throughput = {avg_gen_rate:.1f} tok/s, Peak RSS = {peak_rss:.2f} GiB")

    # Flush allocator cache and unload model if requested to avoid memory bleed into next run
    if unload_after_eval:
        if verbose:
            print("    [+] Unloading model to reclaim allocator memory...", end="", flush=True)
        client.unload_model(model_name)
        time.sleep(2.0)
        if verbose:
            print(" [DONE]")

    return df_tasks, summary

