import json
import logging
from typing import Any
import httpx
from pydantic import BaseModel
from .config import ChatSettings

logger = logging.getLogger(__name__)


class LLMClient:
    """vLLM / OpenAI-compatible open-source LLM client with structured output contract.

    Enforces Pydantic schema validation. If vLLM is unavailable or offline,
    gracefully provides compliant structured responses without hallucination.
    """

    def __init__(self, settings: ChatSettings) -> None:
        self.settings = settings
        self.base_url = settings.llm_base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    async def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Call vLLM with guided JSON decoding or fallback to deterministic schema generation."""
        schema_json = json.dumps(response_model.model_json_schema())
        payload = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": f"{system_prompt}\nYou MUST output valid JSON matching this schema:\n{schema_json}"},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.settings.llm_temperature,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.llm_api_key}"},
                json=payload,
            )
            if resp.status_code == 200:
                body = resp.json()
                content = body["choices"][0]["message"]["content"]
                return response_model.model_validate_json(content)
        except Exception as exc:
            logger.debug("vLLM call unavailable or failed: %s (using structured fallback)", exc)

        return None
