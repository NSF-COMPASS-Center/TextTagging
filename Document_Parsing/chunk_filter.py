# Discard chunks whose markdown header breadcrumb (metadata["section_titles"])
# matches a configured phrase -- e.g. dropping "Introduction"/"Conclusion"
# sections before tagging/extraction.

from typing import List, Optional

from .chunk import Chunk


def is_header_excluded(chunk: Chunk, phrases: Optional[List[str]]) -> bool:
    """True if chunk's header breadcrumb (metadata["section_titles"]) matches
    any of the given phrases (case-insensitive substring match, checked
    against every ancestor header title, not just the chunk's immediate one).
    """
    if not phrases:
        return False

    lowered_phrases = [phrase.lower() for phrase in phrases]
    titles = chunk.metadata.get("section_titles", [])
    return any(phrase in title.lower() for title in titles for phrase in lowered_phrases)


def filter_chunks_by_header_phrases(chunks: List[Chunk], phrases: Optional[List[str]]) -> List[Chunk]:
    if not phrases:
        return chunks

    return [chunk for chunk in chunks if not is_header_excluded(chunk, phrases)]
