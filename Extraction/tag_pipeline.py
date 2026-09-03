# Tag-based extraction: pull parameter values out of chunks that a Taggers
# RunResult already tagged.
#
# Two modes:
#   grouped  - cap+union chunks across all tags, batch params 3-4 per LLM call
#   by_tag   - each param maps to the tag of the same name; only that tag's
#              (capped) chunks are used, one param per call

from typing import List, Literal, Optional, Union

from LLM_Clients.base import LLMClient

from Taggers import RunResult, TaggedRecord, load_run

from .extraction_core import extract_group_from_chunk
from .params import ParamDefinition, create_param_definitions, group_params
from .schema import ExtractionStore, coerce_types, make_store, save_store, update_store


def select_chunks_with_cap(run_result: RunResult, tag_name: str, max_chunks: int) -> List[TaggedRecord]:
    """Chunks carrying tag_name among their matches, first max_chunks in order of appearance."""
    selected = []
    for record in run_result.records:
        if any(match.tag_name == tag_name for match in record.matches):
            selected.append(record)
            if len(selected) >= max_chunks:
                break

    return selected


def _all_tag_names(run_result: RunResult) -> List[str]:
    names = []
    seen = set()
    for record in run_result.records:
        for match in record.matches:
            if match.tag_name not in seen:
                names.append(match.tag_name)
                seen.add(match.tag_name)

    return names


def run_extraction_grouped(
    run_result: RunResult,
    params: List[ParamDefinition],
    client: LLMClient,
    max_chunks_per_tag: int,
    group_size: int,
    max_calls: Optional[int] = None,
) -> ExtractionStore:
    chunks_by_id = {}
    for tag_name in _all_tag_names(run_result):
        for record in select_chunks_with_cap(run_result, tag_name, max_chunks_per_tag):
            chunks_by_id[record.id] = record

    param_groups = group_params(params, target_size=group_size)
    store = make_store(params)
    calls_made = 0

    for record in chunks_by_id.values():
        for group in param_groups:
            if max_calls is not None and calls_made >= max_calls:
                print(f"[tag_pipeline] hit max_calls={max_calls} - stopping extraction early")
                return store

            try:
                extracted = extract_group_from_chunk(group, record.text, client)
            except Exception as e:
                print(f"[tag_pipeline] extraction call failed on chunk {record.id!r}, group {[p.name for p in group]}: {e} - skipping")
                calls_made += 1
                continue

            update_store(store, extracted)
            calls_made += 1

    return store


def run_extraction_by_tag(
    run_result: RunResult,
    params: List[ParamDefinition],
    client: LLMClient,
    max_chunks_per_tag: int,
    max_calls: Optional[int] = None,
) -> ExtractionStore:
    store = make_store(params)
    tag_names = set(_all_tag_names(run_result))
    calls_made = 0

    for param in params:
        if param.name not in tag_names:
            print(f"[tag_pipeline] no tag named '{param.name}' in this run - skipping")
            continue

        for record in select_chunks_with_cap(run_result, param.name, max_chunks_per_tag):
            if max_calls is not None and calls_made >= max_calls:
                print(f"[tag_pipeline] hit max_calls={max_calls} - stopping extraction early")
                return store

            try:
                extracted = extract_group_from_chunk([param], record.text, client)
            except Exception as e:
                print(f"[tag_pipeline] extraction call failed on chunk {record.id!r}, param {param.name!r}: {e} - skipping")
                calls_made += 1
                continue

            update_store(store, extracted)
            calls_made += 1

    return store


def run_extraction_on_tags(
    run: Union[str, RunResult],
    params_json_path: str,
    client: LLMClient,
    mode: Literal["grouped", "by_tag"] = "grouped",
    max_chunks_per_tag: int = 20,
    group_size: int = 4,
    max_calls: Optional[int] = None,
    output_path: Optional[str] = None,
) -> ExtractionStore:
    """max_calls caps the total number of raw+format LLM call pairs made across
    this whole run (a hard ceiling on extraction cost per batch), independent of
    how many chunks/tags/params are involved -- extraction stops early once hit."""
    run_result = load_run(run) if isinstance(run, str) else run
    params = create_param_definitions(params_json_path)

    if mode == "grouped":
        store = run_extraction_grouped(run_result, params, client, max_chunks_per_tag, group_size, max_calls)
    elif mode == "by_tag":
        store = run_extraction_by_tag(run_result, params, client, max_chunks_per_tag, max_calls)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    store = coerce_types(store, params)

    if output_path is not None:
        save_store(store, output_path)

    return store
