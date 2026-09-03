# Semantic-search baseline: independently chunk the markdown, embed into an
# ephemeral (non-persistent) ChromaDB collection, retrieve chunks per
# parameter via semantic search, and run the same extraction logic as
# tag_pipeline.py.

import json
import uuid
from typing import Dict, List, Optional

import chromadb

from Document_Parsing import Chunk, parse_markdown
from LLM_Clients.base import LLMClient

from .extraction_core import extract_group_from_chunk
from .params import ParamDefinition, create_param_definitions, group_params
from .schema import ExtractionStore, coerce_types, make_store, save_store, update_store


def build_ephemeral_collection(chunks: List[Chunk]) -> "chromadb.Collection":
    """In-memory-only ChromaDB collection - never touches disk."""
    client = chromadb.EphemeralClient()
    # Unique name per call: EphemeralClient's in-process backend persists collection
    # state across instances within the same process, so a fixed name collides
    # ("already exists") the second time this runs in a batch/loop.
    collection = client.create_collection(name=f"chunks_{uuid.uuid4().hex}")

    collection.add(
        documents=[chunk.text for chunk in chunks],
        ids=[str(i) for i in range(len(chunks))],
        metadatas=[{"section_titles": " > ".join(chunk.metadata.get("section_titles", []))} for chunk in chunks],
    )

    return collection


def create_query_definitions(json_path: str) -> Dict[str, List[str]]:
    """Load {param_name: [query1, query2, ...]} - retrieval queries per param.

    Distinct from the params JSON (type + definition, used for the extraction
    prompt) - this file only drives what gets searched for in the vector db.
    """
    with open(json_path, "r") as f:
        return json.load(f)


def retrieve_merged_chunks_for_group(
    collection: "chromadb.Collection",
    group: List[ParamDefinition],
    queries_by_param: Dict[str, List[str]],
    top_k: int,
) -> List[str]:
    """For each param in the group, run every one of its queries (falling back
    to its definition if it has none), merging/deduping across queries and
    then across params into one chunk-text list.
    """
    merged: List[str] = []
    seen = set()

    for param in group:
        queries = queries_by_param.get(param.name) or [param.definition]
        for query in queries:
            result = collection.query(query_texts=[query], n_results=top_k)
            for text in result["documents"][0]:
                if text not in seen:
                    merged.append(text)
                    seen.add(text)

    return merged


def run_semantic_baseline(
    markdown_path: str,
    params_json_path: str,
    queries_json_path: Optional[str],
    client: LLMClient,
    top_k: int = 5,
    max_chunk_size: Optional[int] = None,
    sentences_per_chunk: Optional[int] = None,
    group_size: int = 4,
    output_path: Optional[str] = None,
) -> ExtractionStore:
    with open(markdown_path, "r") as f:
        text = f.read()
    chunks = parse_markdown(text, max_chunk_size=max_chunk_size, sentences_per_chunk=sentences_per_chunk)

    collection = build_ephemeral_collection(chunks)
    params = create_param_definitions(params_json_path)
    queries_by_param = create_query_definitions(queries_json_path) if queries_json_path else {}
    param_groups = group_params(params, target_size=group_size)

    store = make_store(params)
    for group in param_groups:
        for chunk_text in retrieve_merged_chunks_for_group(collection, group, queries_by_param, top_k):
            try:
                extracted = extract_group_from_chunk(group, chunk_text, client)
            except Exception as e:
                print(f"[semantic_baseline] extraction call failed for group {[p.name for p in group]}: {e} - skipping")
                continue

            update_store(store, extracted)

    store = coerce_types(store, params)

    if output_path is not None:
        save_store(store, output_path)

    return store
