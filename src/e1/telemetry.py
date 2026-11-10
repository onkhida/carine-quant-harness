"""System telemetry and resource tracking for Ollama and local LLM processes."""

import os
import psutil
from typing import List, Optional


def get_hardware_info() -> dict:
    """Returns basic host hardware telemetry."""
    return {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_memory_gib": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "available_memory_gib": round(psutil.virtual_memory().available / (1024 ** 3), 2),
    }


def get_ollama_processes() -> List[psutil.Process]:
    """Finds all active Ollama daemon, server, and model runner processes."""
    matched = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "ollama" in name or "llama" in name or "ollama" in cmdline or "llama" in cmdline:
                matched.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matched


def get_ollama_rss_gib() -> float:
    """
    Captures the aggregate physical resident memory footprint (RSS in GiB)
    of all active Ollama engine runner and daemon processes.
    Falls back to current process RSS if no Ollama process is detected.
    """
    total_rss_bytes = 0
    found_ollama = False

    for proc in get_ollama_processes():
        try:
            mem = proc.memory_info()
            if mem and mem.rss > 0:
                total_rss_bytes += mem.rss
                found_ollama = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if found_ollama:
        return total_rss_bytes / (1024 ** 3)

    # Fallback to current process if Ollama runs in-process or unreachable
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0

