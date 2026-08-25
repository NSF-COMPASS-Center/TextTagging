# Function 1: takes a tag and a TaggedRecord, matches the tag's phrases against the record's text,
    # and appends a TagMatch to the record if a match is found.

# Function 2: scales function 1. For each tag, run against the record and attach all matches.

# Function 3: scales function 2 across a list of records, wrapping everything in a RunResult
    # (the repeatable output payload for a tagging run) with optional persistence to disk.

import re
from typing import List, Optional

from .schema import RunConfig, RunResult, TagMatch, TaggedRecord, save_run
from .tag import Tag

_SPECIAL_CHARS_RE = re.compile(r"[^\w\s]")


def _clean(text: str) -> str:
    return _SPECIAL_CHARS_RE.sub("", text).lower()


def tag_record(tag: Tag, record: TaggedRecord) -> TaggedRecord:
    """Match a single tag's phrases against record.text and attach a TagMatch if found."""
    cleaned_text = _clean(record.text)

    for phrase in tag.phrases:
        cleaned_phrase = _clean(phrase)
        if re.search(r"\b" + re.escape(cleaned_phrase) + r"\b", cleaned_text):
            record.matches.append(TagMatch(tag_name=tag.name, matched_phrase=phrase))
            break

    return record


def tag_record_with_all(tags: List[Tag], record: TaggedRecord) -> TaggedRecord:
    """Run record against every tag in tags, attaching all matches."""
    for tag in tags:
        record = tag_record(tag, record)

    return record


def run_string_match_tagging(
    tags: List[Tag],
    records: List[TaggedRecord],
    config: RunConfig,
    output_dir: Optional[str] = None,
) -> RunResult:
    """Run every record against every tag and wrap the results in a RunResult.

    If output_dir is provided, the RunResult is also persisted to disk as JSON
    (see schema.save_run); leaving it as None keeps the run in-memory only.
    """
    records = [tag_record_with_all(tags, record) for record in records]
    result = RunResult(config=config, records=records)

    if output_dir is not None:
        save_run(result, output_dir)

    return result
