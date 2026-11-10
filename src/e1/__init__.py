"""CaRINE LLM Quantization Benchmarking Suite."""

from e1.cli import main, run_sweep
from e1.evaluator import evaluate_mbpp_model, load_mbpp_dataset
from e1.ollama_client import OllamaClient
from e1.perplexity import evaluate_wikitext_perplexity
from e1.prompts import build_mbpp_prompt, extract_python_code, extract_target_function_name
from e1.sandbox import sandbox_execute_code
from e1.telemetry import get_hardware_info, get_ollama_rss_gib

__all__ = [
    "main",
    "run_sweep",
    "evaluate_mbpp_model",
    "load_mbpp_dataset",
    "OllamaClient",
    "evaluate_wikitext_perplexity",
    "build_mbpp_prompt",
    "extract_python_code",
    "extract_target_function_name",
    "sandbox_execute_code",
    "get_hardware_info",
    "get_ollama_rss_gib",
]
