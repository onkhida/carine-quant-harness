"""Ollama API client with hardware metrics extraction and memory management."""

import time
import requests
from typing import Any, Dict, List, Optional


class OllamaClient:
    """Client for local Ollama daemon inference and telemetry extraction."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        num_predict: int = 256,
        keep_alive: Optional[str] = None,
        timeout: int = 240,
    ) -> Dict[str, Any]:
        """
        Sends a generation request to Ollama and parses both response text
        and hardware performance metrics.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # Parse hardware performance metrics from Ollama engine
        eval_count = data.get("eval_count", 0)
        eval_dur_ns = data.get("eval_duration", 0)
        eval_rate = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0

        prompt_eval_count = data.get("prompt_eval_count", 0)
        prompt_eval_dur_ns = data.get("prompt_eval_duration", 0)
        prompt_eval_rate = (
            (prompt_eval_count / (prompt_eval_dur_ns / 1e9))
            if prompt_eval_dur_ns > 0
            else 0.0
        )

        total_dur_ns = data.get("total_duration", 0)
        load_dur_ns = data.get("load_duration", 0)

        return {
            "response": data.get("response", ""),
            "eval_count": eval_count,
            "eval_duration_sec": eval_dur_ns / 1e9,
            "eval_rate_tok_sec": eval_rate,
            "prompt_eval_count": prompt_eval_count,
            "prompt_eval_duration_sec": prompt_eval_dur_ns / 1e9,
            "prompt_eval_rate_tok_sec": prompt_eval_rate,
            "total_duration_sec": total_dur_ns / 1e9,
            "load_duration_sec": load_dur_ns / 1e9,
            "raw": data,
        }

    def unload_model(self, model: str) -> bool:
        """
        Signals Ollama to immediately unload the model from memory (RAM/VRAM),
        mitigating allocator page retention and memory bleed between runs.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "keep_alive": 0,
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Lists all locally available models in Ollama."""
        url = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m["name"] for m in models]
        except Exception:
            return []

