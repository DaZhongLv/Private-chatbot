"""
This module contains functions for interacting with external services,
such as the Ollama API for model management.
"""
from __future__ import annotations

import requests


def get_ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        r.raise_for_status()
        data = r.json()
        models = data.get("models", [])
        return [m.get("name") for m in models if m.get("name")]
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Ollama not reachable at http://localhost:11434/api/tags: {e}")
        return []
    except Exception as e:
        print(f"[WARN] Unexpected error while fetching Ollama models: {e}")
        return []

