# Denormalized export of a RunResult, one row per (chunk, tag match),
# for loading straight into pandas/a spreadsheet without touching schema.py.

import json
import os
from typing import Any, Dict, List

from .schema import RunResult


def flatten_records(run_result: RunResult) -> List[Dict[str, Any]]:
    """One row per (chunk, tag match).

    Chunks with zero matches still get a single row with tag_name=None so
    unmatched chunks aren't invisible.
    """
    rows: List[Dict[str, Any]] = []

    for record in run_result.records:
        base: Dict[str, Any] = {
            "run_id": run_result.config.run_id,
            "tagger_method": run_result.config.tagger_method,
            "chunk_id": record.id,
            "section_titles": record.metadata.get("section_titles"),
            "chunk_text": record.text,
        }

        if not record.matches:
            rows.append({**base, "tag_name": None, "matched_phrase": None, "evidence": {}})
            continue

        for match in record.matches:
            rows.append({
                **base,
                "tag_name": match.tag_name,
                "matched_phrase": match.matched_phrase,
                "evidence": match.evidence,
            })

    return rows


def export_flat(run_result: RunResult, output_dir: str) -> str:
    """Write {output_dir}/{run_id}_flat.jsonl using flatten_records."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{run_result.config.run_id}_flat.jsonl")

    with open(path, "w") as f:
        for row in flatten_records(run_result):
            f.write(json.dumps(row) + "\n")

    return path
