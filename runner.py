"""Runner: markdown file -> chunks -> tag(s) -> saved RunResult(s) + flat export
-> optional Chroma-backed semantic extraction.

Tagging is configurable via CLI or config. Extraction (extraction_type,
params/queries paths, extraction LLM provider/model, etc.) is config-only --
set it in the YAML config passed via --config. Tag and header metadata always
get attached to the persistent per-document Chroma collection; extraction_type
picks whether/how they filter retrieval:
  semantic                      - plain semantic search, no filtering
  semantic_tag_filtered         - restrict to chunks tagged with the param's
                                   name (falls back to unfiltered search for a
                                   param with no matching tag in the document)
  semantic_tag_header_filtered  - tag filter (same fallback) plus excluding
                                   chunks under a header_discard_phrases section

Usage:
    python runner.py --config run.yaml
    python runner.py --input doc.md --tagger string_match --string-match-tags tags.json
    python runner.py --config run.yaml --tagger llm   # CLI overrides the config value
"""

import argparse
import copy
import os
import re
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from Document_Parsing import Chunk, filter_chunks_by_header_phrases, parse_markdown
from Extraction import (
    ExtractionStore,
    build_tag_lookup,
    get_or_rebuild_collection,
    run_semantic_pipeline,
)
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
_EXTRACTION_TYPE_CHOICES = ("semantic", "semantic_tag_filtered", "semantic_tag_header_filtered")
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
        "extraction_group_size": 4,
        "semantic_fallback_chunk_count": 5,
        "header_discard_phrases": [],
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

    extraction_type = settings.get("extraction_type")
    if extraction_type:
        if extraction_type not in _EXTRACTION_TYPE_CHOICES:
            raise ValueError(f"Unknown config.extraction_type {extraction_type!r}; expected one of {_EXTRACTION_TYPE_CHOICES}")
        if not settings.get("extraction_params"):
            raise ValueError("config.extraction_type requires config.extraction_params")

    return settings


def chunks_to_records(chunks: List[Chunk]) -> List[TaggedRecord]:
    return [make_record(text=c.text, id=c.id, metadata=c.metadata) for c in chunks]


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


def parse_full_chunks(settings: Dict[str, Any]) -> List[Chunk]:
    """The full, unfiltered document corpus (stable chunk ids, no header-phrase
    filtering) -- this is what gets indexed into Chroma. Tagging runs against a
    header-filtered subset of this same list (see run_tagging)."""
    with open(settings["input"], "r") as f:
        text = f.read()
    return parse_markdown(text, max_chunk_size=settings.get("max_chunk_size"), sentences_per_chunk=settings.get("sentences_per_chunk"))


def run_tagging(settings: Dict[str, Any], full_chunks: List[Chunk]) -> List[RunResult]:
    tagging_chunks = filter_chunks_by_header_phrases(full_chunks, settings.get("header_discard_phrases"))
    tagger = settings["tagger"]

    results: List[RunResult] = []
    if tagger in ("string_match", "both"):
        results.append(run_string_match(settings, copy.deepcopy(tagging_chunks)))
    if tagger in ("llm", "both"):
        results.append(run_llm(settings, copy.deepcopy(tagging_chunks)))

    return results


def merged_tag_lookup(results: List[RunResult]) -> Dict[str, List[str]]:
    """Union tag names per chunk id across every tagger run (relevant when
    tagger: both produced two RunResults over the same tagging_chunks)."""
    merged: Dict[str, List[str]] = {}
    for result in results:
        for chunk_id, tag_names in build_tag_lookup(result).items():
            existing = merged.setdefault(chunk_id, [])
            for name in tag_names:
                if name not in existing:
                    existing.append(name)

    return merged


def chroma_collection_name(input_path: str) -> str:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", stem)


def build_document_collection(settings: Dict[str, Any], full_chunks: List[Chunk], tag_results: List[RunResult]) -> "chromadb.Collection":
    tag_lookup = merged_tag_lookup(tag_results)
    persist_path = os.path.join(settings["output_dir"], "chroma_db")
    collection_name = chroma_collection_name(settings["input"])
    return get_or_rebuild_collection(persist_path, collection_name, full_chunks, tag_lookup, settings.get("header_discard_phrases"))


def run_extraction(settings: Dict[str, Any], collection: "chromadb.Collection", client: LLMClient) -> ExtractionStore:
    extraction_type = settings["extraction_type"]
    output_path = os.path.join(settings["output_dir"], f"{extraction_type}_extraction.json")

    return run_semantic_pipeline(
        collection,
        settings["extraction_params"],
        settings.get("semantic_queries"),
        client,
        group_size=settings["extraction_group_size"],
        sentences_per_chunk=settings.get("sentences_per_chunk"),
        tag_filter=extraction_type in ("semantic_tag_filtered", "semantic_tag_header_filtered"),
        header_filter=extraction_type == "semantic_tag_header_filtered",
        fallback_chunk_count=settings["semantic_fallback_chunk_count"],
        output_path=output_path,
    )


def main() -> None:
    load_dotenv()
    args = parse_args()
    settings = build_settings(args)

    full_chunks = parse_full_chunks(settings)
    print(f"Parsed {len(full_chunks)} chunk(s) from {settings['input']}")

    results = run_tagging(settings, full_chunks)
    for result in results:
        match_count = sum(len(record.matches) for record in result.records)
        flat_path = export_flat(result, settings["output_dir"])
        print(
            f"[{result.config.tagger_method}] run_id={result.config.run_id} "
            f"records={len(result.records)} matches={match_count} "
            f"-> {settings['output_dir']}/{result.config.run_id}.json, {flat_path}"
        )

    extraction_type = settings.get("extraction_type")
    if extraction_type:
        client = build_llm_client(settings, "extraction")
        collection = build_document_collection(settings, full_chunks, results)
        store = run_extraction(settings, collection, client)
        total = sum(len(v) for v in store.values())
        print(f"[extraction:{extraction_type}] {total} unique value(s) extracted")


if __name__ == "__main__":
    main()
