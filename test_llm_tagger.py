"""Quick manual test: run the LLM tagger on a couple of chunks from a random
paper and dump {chunk text, tag(s), reasoning} to a JSON file for inspection.

Usage:
    python test_llm_tagger.py
    python test_llm_tagger.py --paper-type cfs --num-chunks 3
    python test_llm_tagger.py --input papers/cfs/.../some-with-image-refs.md
"""

import argparse
import glob
import json
import os
import random

from dotenv import load_dotenv

from Document_Parsing import parse_markdown
from LLM_Clients import GeminiClient, OllamaClient, OpenAIClient
from Taggers import create_tag_definitions, run_llm_tagging
from runner import chunks_to_records, make_run_config

_PAPER_TYPE_CHOICES = ("loss_survival", "uv", "cfs")
_PROVIDER_CLIENTS = {"openai": OpenAIClient, "gemini": GeminiClient, "ollama": OllamaClient}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Markdown file to sample chunks from (default: a random paper)")
    parser.add_argument(
        "--paper-type", choices=_PAPER_TYPE_CHOICES,
        help="Restrict the random paper pick to this type's tags/llm_tags.json (default: random type too)",
    )
    parser.add_argument("--num-chunks", type=int, default=2, help="How many chunks to tag (default: 2)")
    parser.add_argument("--max-chunk-size", type=int, default=2000, help="Chunking size passed to parse_markdown")
    parser.add_argument("--provider", choices=sorted(_PROVIDER_CLIENTS), default="openai", help="LLM provider for the tagger")
    parser.add_argument("--model", default="gpt-4o", help="Model name for the chosen provider")
    parser.add_argument("--host", help="Host URL (ollama only; defaults to localhost)")
    parser.add_argument("--output", default="output/test_llm_tagger.json", help="Where to write the results JSON")
    parser.add_argument("--seed", type=int, help="Random seed, for reproducible chunk/paper picks")
    return parser.parse_args()


def pick_random_paper(paper_type: str) -> str:
    candidates = glob.glob(f"papers/{paper_type}/figures_with_markdown/*/*.md")
    if not candidates:
        raise FileNotFoundError(f"No markdown files found under papers/{paper_type}/figures_with_markdown/")
    return random.choice(candidates)


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    paper_type = args.paper_type or random.choice(_PAPER_TYPE_CHOICES)
    input_path = args.input or pick_random_paper(paper_type)
    llm_tags_path = os.path.join("tags", paper_type, "llm_tags.json")

    with open(input_path, "r") as f:
        text = f.read()
    chunks = parse_markdown(text, max_chunk_size=args.max_chunk_size)
    if not chunks:
        raise ValueError(f"No chunks parsed from {input_path}")

    sample = random.sample(chunks, k=min(args.num_chunks, len(chunks)))
    records = chunks_to_records(sample)

    tag_defs = create_tag_definitions(llm_tags_path)
    client_cls = _PROVIDER_CLIENTS[args.provider]
    client_kwargs = {"model": args.model}
    if args.provider == "ollama" and args.host:
        client_kwargs["host"] = args.host
    client = client_cls(**client_kwargs)
    config = make_run_config(
        "llm", tag_source=llm_tags_path, model=f"{args.provider}:{args.model}", chunk_size=args.max_chunk_size
    )
    result = run_llm_tagging(tag_defs, records, client, config)

    output = {
        "input": input_path,
        "paper_type": paper_type,
        "llm_tags": llm_tags_path,
        "provider": args.provider,
        "model": args.model,
        "chunks": [
            {
                "id": record.id,
                "text": record.text,
                "tags": [
                    {"tag_name": match.tag_name, "reasoning": match.evidence.get("reasoning")}
                    for match in record.matches
                ],
            }
            for record in result.records
        ],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Tagged {len(sample)} chunk(s) from {input_path} ({paper_type}) -> {args.output}")


if __name__ == "__main__":
    main()
