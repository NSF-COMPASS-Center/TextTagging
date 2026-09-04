# chunk class / object

# attributes: length of the string, text content of the chunk, metadata (e.g. section titles it falls under, tags)

from typing import Any, Dict, Optional


class Chunk:
    def __init__(self, text: str, metadata: Optional[Dict[str, Any]] = None, id: Optional[str] = None) -> None:
        self.text = text
        self.length = len(text)
        self.metadata = metadata or {}
        self.id = id

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return self.text
