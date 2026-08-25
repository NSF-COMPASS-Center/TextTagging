# 1. Has the tag class.
# Attributes: name (name of the tag, which is a string), phrases (list of strings that correspond to the tag)
# Methods: __init__ (initialize the tag object), __str__ (return the tag name), __repr__ (return the tag name), __eq__ (compare two tags), __hash__ (hash the tag name)

# 2. Have a 'create_tag' function. There will be json files that contain the tag name as the key and the list of phrases as values.

import json
from typing import List


class Tag:
    def __init__(self, name: str, phrases: List[str]) -> None:
        self.name = name
        self.phrases = phrases

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tag):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

# TODO: as we get more complex in this repo make sure this payload and logic holds up and isn't lazy 
def create_tags(json_path: str) -> List[Tag]:
    with open(json_path, "r") as f:
        data = json.load(f)

    return [Tag(name, phrases) for name, phrases in data.items()]
