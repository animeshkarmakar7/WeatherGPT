import json

import httpx
from pydantic import BaseModel

from .config import ChatSettings


class LLMClient:
    def __init__(self, settings: ChatSettings) -> None:
        self.settings = settings
        self.base_url = settings.llm_base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/models")
            return response.status_code == 200
        except Exception:
            return False

    async def generate_structured(self, prompt: str, system_prompt: str, response_model: type[BaseModel]) -> BaseModel:
        schema_json = json.dumps(response_model.model_json_schema())
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": f"{system_prompt}\nReturn only JSON matching this schema:\n{schema_json}"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return response_model.model_validate_json(content)
        except Exception as exc:
            raise RuntimeError("LLM generation failed") from exc
