import os
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self):
        self.base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", "")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.base_url:
            raise ValueError("LLM_BASE_URL 未配置")
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        if not self.model:
            raise ValueError("LLM_MODEL 未配置")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]