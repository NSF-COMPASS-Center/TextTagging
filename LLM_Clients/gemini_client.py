import os
from typing import Optional, Type

from google import genai
from google.genai import types

from .base import LLMClient, ResponseModel


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        self.model = model
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text

    def generate_structured(
        self, system_prompt: str, user_prompt: str, response_model: Type[ResponseModel]
    ) -> ResponseModel:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        return response.parsed
