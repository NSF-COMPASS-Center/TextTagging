from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class LLMClient(ABC):
    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Send a system/user prompt pair and return the raw text response."""

    @abstractmethod
    def generate_structured(
        self, system_prompt: str, user_prompt: str, response_model: Type[ResponseModel]
    ) -> ResponseModel:
        """Send a system/user prompt pair and return output parsed into response_model."""

    def generate_then_format(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[ResponseModel],
        formatting_prompt: str = "Format the following response into the requested structure.",
    ) -> ResponseModel:
        """Run a raw generation call, then a second call that formats that raw output into response_model."""
        raw_response = self.generate_text(system_prompt, user_prompt)
        return self.generate_structured(formatting_prompt, raw_response, response_model)
