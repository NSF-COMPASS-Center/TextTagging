# Extraction parameter definitions.
#
# Loaded from a JSON file shaped {param_name: [type, definition]}, e.g.
# {"award_amount": ["int", "The total dollar amount of the award"]}.
# `param_type` is a hint used only for post-extraction coercion (see schema.coerce_types) -
# the LLM is always asked for a plain string at extraction time.

import json
import math
from typing import List


class ParamDefinition:
    def __init__(self, name: str, param_type: str, definition: str) -> None:
        self.name = name
        self.param_type = param_type
        self.definition = definition

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParamDefinition):
            return NotImplemented
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


def create_param_definitions(json_path: str) -> List[ParamDefinition]:
    with open(json_path, "r") as f:
        data = json.load(f)

    params = []
    for name, value in data.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Param '{name}' must map to a [type, definition] list, got {value!r}")
        param_type, definition = value
        params.append(ParamDefinition(name, param_type, definition))

    return params


def group_params(params: List[ParamDefinition], target_size: int = 4) -> List[List[ParamDefinition]]:
    """Split params into groups sized as evenly as possible, each at most target_size.

    e.g. 10 params with target_size=4 -> groups of 4/3/3, not 4/4/2.
    """
    if not params:
        return []

    num_groups = max(1, math.ceil(len(params) / target_size))
    base, extra = divmod(len(params), num_groups)

    groups = []
    i = 0
    for g in range(num_groups):
        size = base + 1 if g < extra else base
        groups.append(params[i:i + size])
        i += size

    return groups
