from __future__ import annotations

import requests

from doc_agent.llm.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()
