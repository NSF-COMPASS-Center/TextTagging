# Repeatable output payload for tagging runs.
#
# The goal: any tagger implementation (string-match today, LLM-based later)
# produces the same shapes here, so runs made with different chunk sizes,
# models, or tagging methods can be saved, reloaded, and compared apples-to-apples.

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TagMatch:
    tag_name: str
    matched_phrase: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaggedRecord:
    text: str
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    matches: List[TagMatch] = field(default_factory=list)


@dataclass
class RunConfig:
    run_id: str
    tagger_method: str
    tag_source: Optional[str] = None
    model: Optional[str] = None
    chunk_size: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class RunResult:
    config: RunConfig
    records: List[TaggedRecord]


def make_run_config(tagger_method: str, **kwargs: Any) -> RunConfig:
    return RunConfig(
        run_id=uuid.uuid4().hex,
        tagger_method=tagger_method,
        created_at=datetime.now().isoformat(),
        **kwargs,
    )


def make_record(text: str, id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> TaggedRecord:
    return TaggedRecord(text=text, id=id, metadata=metadata or {})


def to_dict(run_result: RunResult) -> Dict[str, Any]:
    return asdict(run_result)


def from_dict(data: Dict[str, Any]) -> RunResult:
    config = RunConfig(**data["config"])
    records = [
        TaggedRecord(
            text=record["text"],
            id=record.get("id"),
            metadata=record.get("metadata", {}),
            matches=[TagMatch(**match) for match in record.get("matches", [])],
        )
        for record in data["records"]
    ]
    return RunResult(config=config, records=records)


def save_run(run_result: RunResult, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{run_result.config.run_id}.json")

    with open(path, "w") as f:
        json.dump(to_dict(run_result), f, indent=2)

    return path


def load_run(path: str) -> RunResult:
    with open(path, "r") as f:
        data = json.load(f)

    return from_dict(data)
