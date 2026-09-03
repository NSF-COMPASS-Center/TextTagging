# The extraction store: {param_name: [unique_value, unique_value, ...]}.
# Deduplicated per parameter, persisted so repeated runs/cycles keep accumulating.

import json
from typing import Any, Dict, List, Union

from .params import ParamDefinition

ExtractionStore = Dict[str, List[Union[str, int]]]


def make_store(params: List[ParamDefinition]) -> ExtractionStore:
    return {param.name: [] for param in params}


def update_store(store: ExtractionStore, new_values: Dict[str, List[str]]) -> ExtractionStore:
    """Append only values not already present per key (order-preserving dedup)."""
    for param_name, values in new_values.items():
        existing = store.setdefault(param_name, [])
        seen = set(existing)
        for value in values:
            if value not in seen:
                existing.append(value)
                seen.add(value)

    return store


_COERCERS = {"int": int, "float": float}


def coerce_types(store: ExtractionStore, params: List[ParamDefinition]) -> ExtractionStore:
    """For params declared 'int'/'float', try that conversion on every stored value; keep as string on failure."""
    for param in params:
        coerce = _COERCERS.get(param.param_type)
        if coerce is None:
            continue

        coerced: List[Any] = []
        for value in store.get(param.name, []):
            try:
                coerced.append(coerce(value))
            except (ValueError, TypeError):
                coerced.append(value)
        store[param.name] = coerced

    return store


def save_store(store: ExtractionStore, path: str) -> str:
    with open(path, "w") as f:
        json.dump(store, f, indent=2)

    return path


def load_store(path: str) -> ExtractionStore:
    with open(path, "r") as f:
        return json.load(f)
