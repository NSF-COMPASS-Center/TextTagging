"""Runner: markdown file -> chunks -> tag(s) -> saved RunResult(s) + flat export
-> optional extraction (tag-based and/or semantic-search baseline).

Tagging is configurable via CLI or config. Extraction (which pipeline(s) to run,
params/queries paths, extraction LLM provider/model, batching, etc.) is config-only --
set it in the YAML config passed via --config.

Usage:
    python runner.py --config run.yaml
    python runner.py --input doc.md --tagger string_match --string-match-tags tags.json
    python runner.py --config run.yaml --tagger llm   # CLI overrides the config value
"""

import argparse
import copy
import os
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from Document_Parsing import Chunk, parse_markdown
from Extraction import ExtractionStore, run_extraction_on_tags, run_semantic_baseline
from LLM_Clients import GeminiClient, LLMClient, OllamaClient, OpenAIClient
from Taggers import (
    RunResult,
    TaggedRecord,
    create_tag_definitions,
    create_tags,
    export_flat,
    make_record,
    make_run_config,
    run_llm_tagging,
    run_string_match_tagging,
)

_TAGGER_CHOICES = ("string_match", "llm", "both")
_EXTRACT_CHOICES = ("tags", "semantic", "both")
_EXTRACTION_MODE_CHOICES = ("grouped", "by_tag")
_PROVIDER_CLIENTS = {"openai": OpenAIClient, "gemini": GeminiClient, "ollama": OllamaClient}

_PAPER_TYPE_CHOICES = ("loss_survival", "uv", "cfs")
_PAPER_TYPE_FILES = {
    "string_match_tags": "string_tags.json",
    "llm_tags": "llm_tags.json",
    "extraction_params": "params.json",
    "semantic_queries": "queries.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to a YAML config file")
    parser.add_argument("--input", help="Path to the markdown file to tag")
    parser.add_argument(
        "--paper-type",
        choices=_PAPER_TYPE_CHOICES,
        help="Paper type; auto-fills string_match_tags/llm_tags/extraction_params/semantic_queries "
             "from tags/<paper_type>/*.json unless those are set explicitly",
    )
    parser.add_argument("--tagger", choices=_TAGGER_CHOICES, help="Which tagger(s) to run")
    parser.add_argument("--max-chunk-size", type=int, help="Max characters per chunk (mutually exclusive with --sentences-per-chunk)")
    parser.add_argument("--sentences-per-chunk", type=int, help="Group N sentences per chunk within each header section (mutually exclusive with --max-chunk-size)")
    parser.add_argument("--string-match-tags", help="JSON path for create_tags (name -> [phrases])")
    parser.add_argument("--llm-tags", help="JSON path for create_tag_definitions (name -> definition)")
    parser.add_argument("--tagging-llm-provider", choices=sorted(_PROVIDER_CLIENTS), help="LLM provider used for the 'llm' tagger")
    parser.add_argument("--tagging-llm-model", help="LLM model name used for the 'llm' tagger")
    parser.add_argument("--tagging-llm-host", help="Host URL for the tagging LLM provider (ollama only; defaults to localhost)")
    parser.add_argument("--output-dir", help="Directory to write RunResult + flat export files to")

    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_settings(args: argparse.Namespace) -> Dict[str, Any]:
    settings: Dict[str, Any] = {
        "output_dir": "output",
        "extraction_mode": "grouped",
        "extraction_max_chunks_per_tag": 20,
        "extraction_group_size": 4,
        "semantic_top_k": 5,
    }

    if args.config:
        settings.update(load_config(args.config))

    for key in (
        "input", "paper_type", "tagger", "max_chunk_size", "sentences_per_chunk", "string_match_tags",
        "llm_tags", "tagging_llm_provider", "tagging_llm_model", "tagging_llm_host", "output_dir",
    ):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value

    if settings.get("max_chunk_size") is not None and settings.get("sentences_per_chunk") is not None:
        raise ValueError("max_chunk_size and sentences_per_chunk are mutually exclusive")

    paper_type = settings.get("paper_type")
    if paper_type:
        if paper_type not in _PAPER_TYPE_CHOICES:
            raise ValueError(f"Unknown paper_type {paper_type!r}; expected one of {_PAPER_TYPE_CHOICES}")
        for key, filename in _PAPER_TYPE_FILES.items():
            settings.setdefault(key, os.path.join("tags", paper_type, filename))

    if not settings.get("input"):
        raise ValueError("No input markdown file given (use --input or config.input)")
    if not settings.get("tagger"):
        raise ValueError("No tagger method given (use --tagger or config.tagger)")

    extract = settings.get("extract")
    if extract:
        if extract not in _EXTRACT_CHOICES:
            raise ValueError(f"Unknown config.extract {extract!r}; expected one of {_EXTRACT_CHOICES}")
        if not settings.get("extraction_params"):
            raise ValueError("config.extract requires config.extraction_params")
        extraction_mode = settings.get("extraction_mode")
        if extraction_mode not in _EXTRACTION_MODE_CHOICES:
            raise ValueError(f"Unknown config.extraction_mode {extraction_mode!r}; expected one of {_EXTRACTION_MODE_CHOICES}")

    return settings


def chunks_to_records(chunks: List[Chunk]) -> List[TaggedRecord]:
    return [make_record(text=c.text, id=str(i), metadata=c.metadata) for i, c in enumerate(chunks)]


_LLM_DEFAULTS = {
    # Tagging defaults to a local Ollama model (no API key/cost) unless a
    # config overrides tagging_llm_provider/tagging_llm_model.
    "tagging": {"provider": "ollama", "model": "gpt-oss:20b"},
    "extraction": {"provider": "openai", "model": None},
}


def build_llm_client(settings: Dict[str, Any], prefix: str) -> LLMClient:
    """Build an LLMClient from settings[f"{prefix}_llm_provider/model/host"],
    falling back to _LLM_DEFAULTS[prefix] for whichever of provider/model isn't set.

    `prefix` is "tagging" or "extraction" so the two call sites can be configured
    (via config.tagging_llm_* / config.extraction_llm_*) with different
    providers/models -- e.g. a local model for tagging, a hosted one for extraction.
    """
    defaults = _LLM_DEFAULTS.get(prefix, {})
    provider = settings.get(f"{prefix}_llm_provider") or defaults.get("provider", "openai")
    client_cls = _PROVIDER_CLIENTS[provider]

    kwargs: Dict[str, Any] = {}
    model = settings.get(f"{prefix}_llm_model") or defaults.get("model")
    if model:
        kwargs["model"] = model
    if provider == "ollama" and settings.get(f"{prefix}_llm_host"):
        kwargs["host"] = settings[f"{prefix}_llm_host"]
    return client_cls(**kwargs)


def _chunk_config_kwargs(settings: Dict[str, Any]) -> Dict[str, Any]:
    """chunk_size/params kwargs for make_run_config, recording whichever chunking
    scheme (char-based or sentence-based) this run actually used."""
    sentences_per_chunk = settings.get("sentences_per_chunk")
    if sentences_per_chunk is not None:
        return {"params": {"sentences_per_chunk": sentences_per_chunk}}
    return {"chunk_size": settings.get("max_chunk_size")}


def run_string_match(settings: Dict[str, Any], chunks: List[Chunk]) -> RunResult:
    tags_path = settings.get("string_match_tags")
    if not tags_path:
        raise ValueError("string_match tagger requires --string-match-tags / config.string_match_tags")

    tags = create_tags(tags_path)
    records = chunks_to_records(chunks)
    config = make_run_config("string_match", tag_source=tags_path, **_chunk_config_kwargs(settings))
    return run_string_match_tagging(tags, records, config, output_dir=settings["output_dir"])


def run_llm(settings: Dict[str, Any], chunks: List[Chunk]) -> RunResult:
    tags_path = settings.get("llm_tags")
    if not tags_path:
        raise ValueError("llm tagger requires --llm-tags / config.llm_tags")

    tag_defs = create_tag_definitions(tags_path)
    records = chunks_to_records(chunks)
    client = build_llm_client(settings, "tagging")
    config = make_run_config(
        "llm",
        tag_source=tags_path,
        model=settings.get("tagging_llm_model"),
        **_chunk_config_kwargs(settings),
    )
    return run_llm_tagging(tag_defs, records, client, config, output_dir=settings["output_dir"])


def run_tag_extraction(settings: Dict[str, Any], result: RunResult, client: LLMClient) -> ExtractionStore:
    output_path = os.path.join(settings["output_dir"], f"{result.config.run_id}_extraction.json")
    return run_extraction_on_tags(
        result,
        settings["extraction_params"],
        client,
        mode=settings["extraction_mode"],
        max_chunks_per_tag=settings["extraction_max_chunks_per_tag"],
        group_size=settings["extraction_group_size"],
        max_calls=settings.get("extraction_max_calls"),
        output_path=output_path,
    )


def run_semantic_extraction(settings: Dict[str, Any], client: LLMClient) -> ExtractionStore:
    output_path = os.path.join(settings["output_dir"], "semantic_baseline_extraction.json")
    return run_semantic_baseline(
        settings["input"],
        settings["extraction_params"],
        settings.get("semantic_queries"),
        client,
        top_k=settings["semantic_top_k"],
        max_chunk_size=settings.get("max_chunk_size"),
        sentences_per_chunk=settings.get("sentences_per_chunk"),
        group_size=settings["extraction_group_size"],
        output_path=output_path,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = build_settings(args)

    with open(settings["input"], "r") as f:
        text = f.read()
    chunks = parse_markdown(
        text, max_chunk_size=settings.get("max_chunk_size"), sentences_per_chunk=settings.get("sentences_per_chunk"),
    )

    tagger = settings["tagger"]
    results: List[RunResult] = []

    if tagger in ("string_match", "both"):
        results.append(run_string_match(settings, copy.deepcopy(chunks)))
    if tagger in ("llm", "both"):
        results.append(run_llm(settings, copy.deepcopy(chunks)))

    print(f"Parsed {len(chunks)} chunk(s) from {settings['input']}")
    for result in results:
        match_count = sum(len(record.matches) for record in result.records)
        flat_path = export_flat(result, settings["output_dir"])
        print(
            f"[{result.config.tagger_method}] run_id={result.config.run_id} "
            f"records={len(result.records)} matches={match_count} "
            f"-> {settings['output_dir']}/{result.config.run_id}.json, {flat_path}"
        )

    extract = settings.get("extract")
    if extract:
        client: Optional[LLMClient] = None

        if extract in ("tags", "both"):
            client = client or build_llm_client(settings, "extraction")
            for result in results:
                store = run_tag_extraction(settings, result, client)
                total = sum(len(v) for v in store.values())
                print(f"[extraction:tags:{result.config.tagger_method}] {total} unique value(s) extracted")

        if extract in ("semantic", "both"):
            client = client or build_llm_client(settings, "extraction")
            store = run_semantic_extraction(settings, client)
            total = sum(len(v) for v in store.values())
            print(f"[extraction:semantic] {total} unique value(s) extracted")


if __name__ == "__main__":
    main()
