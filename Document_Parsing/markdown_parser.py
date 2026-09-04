# Parses markdown text into Chunks, splitting on ATX headers (# .. ######)
# and tagging each chunk with the header hierarchy it falls under, as
# metadata["section_titles"].

import re
from typing import List, Optional

import pysbd

from .chunk import Chunk

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_FENCE_RE = re.compile(r"^(```|~~~)")

_segmenter = pysbd.Segmenter(language="en", clean=False)


def _split_to_size(text: str, max_chunk_size: Optional[int]) -> List[str]:
    if max_chunk_size is None or len(text) <= max_chunk_size:
        return [text]

    return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]


def _split_by_sentences(text: str, sentences_per_chunk: int) -> List[str]:
    """Group text into chunks of `sentences_per_chunk` sentences each (pysbd).

    Sentence groups never span a header boundary - this only splits the body
    text within a single header section, so a section's last group may be
    smaller than sentences_per_chunk.
    """
    sentences = [s.strip() for s in _segmenter.segment(text) if s.strip()]
    if not sentences:
        return []

    return [
        " ".join(sentences[i:i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]


def _split_body(text: str, max_chunk_size: Optional[int], sentences_per_chunk: Optional[int]) -> List[str]:
    if max_chunk_size is not None and sentences_per_chunk is not None:
        raise ValueError("max_chunk_size and sentences_per_chunk are mutually exclusive")

    if sentences_per_chunk is not None:
        return _split_by_sentences(text, sentences_per_chunk)

    return _split_to_size(text, max_chunk_size)


def parse_markdown(
    text: str,
    max_chunk_size: Optional[int] = None,
    sentences_per_chunk: Optional[int] = None,
) -> List[Chunk]:
    breadcrumb: List[Optional[str]] = [None] * 6
    chunks: List[Chunk] = []
    buffer: List[str] = []
    in_fence = False
    chunk_index = 0

    def finalize() -> None:
        nonlocal chunk_index
        body = "\n".join(buffer).strip()
        buffer.clear()

        if not body:
            return

        section_titles = [title for title in breadcrumb if title is not None]
        for piece in _split_body(body, max_chunk_size, sentences_per_chunk):
            chunks.append(Chunk(text=piece, metadata={"section_titles": list(section_titles)}, id=f"c{chunk_index}"))
            chunk_index += 1

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            buffer.append(line)
            continue

        header_match = None if in_fence else _HEADER_RE.match(line)

        if header_match:
            finalize()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            breadcrumb[level - 1] = title
            for i in range(level, 6):
                breadcrumb[i] = None
        else:
            buffer.append(line)

    finalize()
    return chunks
