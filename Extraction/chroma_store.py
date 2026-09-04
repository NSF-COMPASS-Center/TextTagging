# Persistent (disk-backed) per-document ChromaDB collection: chunks are
# indexed once per run with tag and header metadata attached, so the same
# collection can be queried under different filter combinations by
# semantic_pipeline.py (plain semantic / tag-filtered / tag+header-filtered).
#
# Rebuilt from scratch on every run (delete + recreate) rather than
# incrementally updated -- keeps rebuild semantics simple and avoids stale
# entries from a previous run with different chunking/tagging settings.

from typing import Dict, List, Optional

import chromadb

from Document_Parsing import Chunk, is_header_excluded
from Taggers import RunResult


def build_tag_lookup(run_result: RunResult) -> Dict[str, List[str]]:
    """chunk_id -> matched tag names, from the RunResult's TaggedRecords.

    Only chunks that were tagged (i.e. survived header-phrase filtering
    before tagging) appear here; chunks absent from this map are treated as
    untagged (empty tag list) when building Chroma metadata.
    """
    lookup: Dict[str, List[str]] = {}
    for record in run_result.records:
        if record.id is None:
            continue
        lookup[record.id] = [match.tag_name for match in record.matches]

    return lookup


def get_or_rebuild_collection(
    persist_path: str,
    collection_name: str,
    chunks: List[Chunk],
    tag_lookup: Dict[str, List[str]],
    header_discard_phrases: Optional[List[str]],
) -> "chromadb.Collection":
    """(Re)build a persistent Chroma collection from `chunks` (the full,
    unfiltered document corpus), stamping each chunk with its section
    breadcrumb, matched tag names, and whether it falls under a
    header_discard_phrases section -- so a single collection supports every
    extraction_type's filtering needs.
    """
    client = chromadb.PersistentClient(path=persist_path)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # collection didn't exist yet -- nothing to delete

    collection = client.create_collection(name=collection_name)

    collection.add(
        documents=[chunk.text for chunk in chunks],
        ids=[chunk.id for chunk in chunks],
        metadatas=[
            {
                "section_titles": " > ".join(chunk.metadata.get("section_titles", [])),
                "tags": ",".join(tag_lookup.get(chunk.id, [])),
                "header_excluded": is_header_excluded(chunk, header_discard_phrases),
            }
            for chunk in chunks
        ],
    )

    return collection
