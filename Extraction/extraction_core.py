# Raw-then-format LLM extraction of a group of params from one chunk of text.

import os
from typing import Dict, List

from pydantic import BaseModel

from LLM_Clients.base import LLMClient

from .params import ParamDefinition

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "extraction_system.md")
_USER_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "extraction_user.md")


class _ExtractedValue(BaseModel):
    param_name: str
    value: str


class _ExtractionResult(BaseModel):
    values: List[_ExtractedValue]


def _load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def extract_group_from_chunk(group: List[ParamDefinition], text: str, client: LLMClient) -> Dict[str, List[str]]:
    """Run one raw+format LLM call extracting group's params from text.

    Works identically whether group has 1 param (by-tag mode) or several
    (grouped mode / semantic baseline).
    """
    system_prompt_template = _load_prompt(_SYSTEM_PROMPT_PATH)
    user_prompt_template = _load_prompt(_USER_PROMPT_PATH)

    definitions = "\n".join(f"- {param.name}: {param.definition}" for param in group)
    system_prompt = system_prompt_template.format(definitions=definitions)
    user_prompt = user_prompt_template.format(text=text)

    result = client.generate_then_format(system_prompt, user_prompt, _ExtractionResult)

    valid_names = {param.name for param in group}
    extracted: Dict[str, List[str]] = {param.name: [] for param in group}
    for item in result.values:
        if item.param_name in valid_names:
            extracted[item.param_name].append(item.value)

    return extracted
