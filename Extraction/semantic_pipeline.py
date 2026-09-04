# Chroma-backed semantic extraction: query a persistent per-document
# collection (built by chroma_store.get_or_rebuild_collection, which already
# carries tag and header metadata on every chunk) under one of three filter
# combinations:
#   semantic                      - no filtering, plain semantic search
#   semantic_tag_filtered         - restrict to chunks tagged with the param's
#                                    name (falls back to unfiltered semantic
#                                    search for a param with no matching tag
#                                    anywhere in the document)
#   semantic_tag_header_filtered  - tag filter (with the same fallback) plus
#                                    excluding chunks under a
#                                    header_discard_phrases section

import json
from typing import Dict, List, Optional, Set

from LLM_Clients.base import LLMClient

from .extraction_core import extract_group_from_chunk
from .params import ParamDefinition, create_param_definitions, group_params
from .schema import ExtractionStore, coerce_types, make_store, save_store, update_store

_DEFAULT_FALLBACK_CHUNK_COUNT = 5
_ANCHOR_CHUNKS_PER_FIELD = 4
_ANCHOR_SENTENCES_PER_CHUNK = 5


def create_query_definitions(json_path: str) -> Dict[str, List[str]]:
    """Load {param_name: [query1, query2, ...]} - retrieval queries per param.

    Distinct from the params JSON (type + definition, used for the extraction
    prompt) - this file only drives what gets searched for in the vector db.
    """
    with open(json_path, "r") as f:
        return json.load(f)


def target_chunks_per_field(sentences_per_chunk: Optional[int], fallback: int = _DEFAULT_FALLBACK_CHUNK_COUNT) -> int:
    """How many deduplicated chunks to retrieve per field.

    Scales inversely with sentence-based chunk size, anchored at 4 chunks/field
    for a 5-sentence chunk size (so total retrieved sentence-context per field
    stays roughly constant). Char-based chunking (max_chunk_size) has no clean
    sentence-equivalent, so it uses a fixed `fallback` count instead.
    """
    if sentences_per_chunk is None:
        return fallback

    return max(1, round(_ANCHOR_CHUNKS_PER_FIELD * _ANCHOR_SENTENCES_PER_CHUNK / sentences_per_chunk))


def collect_all_tag_names(collection: "chromadb.Collection") -> Set[str]:
    names: Set[str] = set()
    for meta in collection.get()["metadatas"]:
        tags = meta.get("tags", "")
        if tags:
            names.update(tags.split(","))

    return names


def retrieve_chunks_for_param(
    collection: "chromadb.Collection",
    param: ParamDefinition,
    queries_by_param: Dict[str, List[str]],
    target_count: int,
    tag_filter: bool,
    header_filter: bool,
    all_tag_names: Set[str],
    seen: Set[str],
) -> List[str]:
    """Up to target_count new (not already in `seen`) chunk texts for `param`,
    querying each of its queries (falling back to its definition) until the
    target is hit or the collection is exhausted.

    `seen` is shared across every param in the group so chunks already
    claimed by an earlier param in the group don't count toward this param's
    target (dedup across the whole group, not just within one param).
    """
    collection_size = collection.count()
    if collection_size == 0:
        return []

    effective_tag_filter = tag_filter and param.name in all_tag_names
    queries = queries_by_param.get(param.name) or [param.definition]

    collected: List[str] = []
    for query in queries:
        result = collection.query(query_texts=[query], n_results=collection_size)
        for text, meta in zip(result["documents"][0], result["metadatas"][0]):
            if text in seen:
                continue
            if header_filter and meta.get("header_excluded"):
                continue
            if effective_tag_filter:
                tags = meta.get("tags", "")
                tag_names = tags.split(",") if tags else []
                if param.name not in tag_names:
                    continue

            collected.append(text)
            seen.add(text)
            if len(collected) >= target_count:
                break
        if len(collected) >= target_count:
            break

    return collected


def retrieve_merged_chunks_for_group(
    collection: "chromadb.Collection",
    group: List[ParamDefinition],
    queries_by_param: Dict[str, List[str]],
    target_count: int,
    tag_filter: bool,
    header_filter: bool,
    all_tag_names: Set[str],
) -> List[str]:
    """Merged, deduplicated chunk texts for every param in the group -
    each param contributes up to target_count chunks not already claimed by
    an earlier param in the group."""
    merged: List[str] = []
    seen: Set[str] = set()

    for param in group:
        merged.extend(
            retrieve_chunks_for_param(collection, param, queries_by_param, target_count, tag_filter, header_filter, all_tag_names, seen)
        )

    return merged


def run_semantic_pipeline(
    collection: "chromadb.Collection",
    params_json_path: str,
    queries_json_path: Optional[str],
    client: LLMClient,
    group_size: int = 4,
    sentences_per_chunk: Optional[int] = None,
    tag_filter: bool = False,
    header_filter: bool = False,
    fallback_chunk_count: int = _DEFAULT_FALLBACK_CHUNK_COUNT,
    output_path: Optional[str] = None,
) -> ExtractionStore:
    params = create_param_definitions(params_json_path)
    queries_by_param = create_query_definitions(queries_json_path) if queries_json_path else {}
    param_groups = group_params(params, target_size=group_size)

    target_count = target_chunks_per_field(sentences_per_chunk, fallback=fallback_chunk_count)
    all_tag_names = collect_all_tag_names(collection) if tag_filter else set()

    store = make_store(params)
    for group in param_groups:
        chunks_for_group = retrieve_merged_chunks_for_group(
            collection, group, queries_by_param, target_count, tag_filter, header_filter, all_tag_names
        )
        for chunk_text in chunks_for_group:
            try:
                extracted = extract_group_from_chunk(group, chunk_text, client)
            except Exception as e:
                print(f"[semantic_pipeline] extraction call failed for group {[p.name for p in group]}: {e} - skipping")
                continue

            update_store(store, extracted)

    store = coerce_types(store, params)

    if output_path is not None:
        save_store(store, output_path)

    return store
