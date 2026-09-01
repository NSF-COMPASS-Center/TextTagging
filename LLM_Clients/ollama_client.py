from typing import Optional, Type

import ollama

from .base import LLMClient, ResponseModel


class OllamaClient(LLMClient):
    def __init__(self, model: str = "llama3.1", host: Optional[str] = None):
        self.model = model
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]

    def generate_structured(
        self, system_prompt: str, user_prompt: str, response_model: Type[ResponseModel]
    ) -> ResponseModel:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            format=response_model.model_json_schema(),
        )
        return response_model.model_validate_json(response["message"]["content"])
