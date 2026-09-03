# at the base level, LLM gets sys and user prompt, looks at the tags it can choose from, and assign tags to the chunks based on what it observes

# what we need to provide to LLM: prompts, legal tags, definitions/context for each tag

# for now, have blank placeholder prompts. We will have a directory with .md files that contain corresponding prompts.

# Right now, one raw call where the tags+definitions and prompt are provided for LLM to choose tags. Then a formatting call with the LLM client stuff that was built.
    # later, want to potentially add multiple calls to prioritize certain tags over others.
    # actually, implement both. There can be a list of list of tags, the first list being the most important tags, the second list being the second most important tags, etc.
    # default should just be the one call unless specified by a list of list of tags w/their definitions/context

import json
import os
from typing import List, Optional, Union

from pydantic import BaseModel

from LLM_Clients.base import LLMClient

from .schema import RunConfig, RunResult, TagMatch, TaggedRecord, save_run

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
_SYSTEM_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "tagging_system.md")
_USER_PROMPT_PATH = os.path.join(_PROMPTS_DIR, "tagging_user.md")


class TagDefinition:
    def __init__(self, name: str, definition: str) -> None:
        self.name = name
        self.definition = definition

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TagDefinition):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


def create_tag_definitions(json_path: str) -> List[TagDefinition]:
    with open(json_path, "r") as f:
        data = json.load(f)

    return [TagDefinition(name, definition) for name, definition in data.items()]


class _TagAssignment(BaseModel):
    tag_name: str
    evidence: str


class _TagAssignments(BaseModel):
    tags: List[_TagAssignment]


def _load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def tag_record_with_tier(tier: List[TagDefinition], record: TaggedRecord, client: LLMClient) -> TaggedRecord:
    """Run a single raw+format LLM call pair, choosing among tier's tags, and append TagMatches."""
    system_prompt = _load_prompt(_SYSTEM_PROMPT_PATH)
    user_prompt_template = _load_prompt(_USER_PROMPT_PATH)

    tag_context = "\n".join(f"- {tag.name}: {tag.definition}" for tag in tier)
    user_prompt = f"{user_prompt_template}\n\nAvailable tags:\n{tag_context}\n\nText:\n{record.text}"

    assignments = client.generate_then_format(system_prompt, user_prompt, _TagAssignments)

    valid_names = {tag.name for tag in tier}
    for assignment in assignments.tags:
        if assignment.tag_name in valid_names:
            record.matches.append(
                TagMatch(tag_name=assignment.tag_name, evidence={"reasoning": assignment.evidence})
            )

    return record


def tag_record_with_all_tiers(tiers: List[List[TagDefinition]], record: TaggedRecord, client: LLMClient) -> TaggedRecord:
    """Run every tier, in priority order, accumulating matches onto the record."""
    for tier in tiers:
        record = tag_record_with_tier(tier, record, client)

    return record


def run_llm_tagging(
    tags: Union[List[TagDefinition], List[List[TagDefinition]]],
    records: List[TaggedRecord],
    client: LLMClient,
    config: RunConfig,
    output_dir: Optional[str] = None,
) -> RunResult:
    """Run every record through every tier of tags and wrap the results in a RunResult.

    `tags` may be a flat List[TagDefinition] (single call, the default) or a List[List[TagDefinition]]
    of priority tiers, most important first (one raw+format call pair per tier).

    If output_dir is provided, the RunResult is also persisted to disk as JSON
    (see schema.save_run); leaving it as None keeps the run in-memory only.
    """
    tiers = tags if tags and isinstance(tags[0], list) else [tags]

    records = [tag_record_with_all_tiers(tiers, record, client) for record in records]
    result = RunResult(config=config, records=records)

    if output_dir is not None:
        save_run(result, output_dir)

    return result
