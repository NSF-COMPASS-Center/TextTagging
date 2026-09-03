# Tagging analysis functions and outputs that should be human readable

# After ingesting an output from running tagging, should be able to report:
    # chunk size
    # Total number of records tagged
    # Total number of records that were not tagged
    # List of which markdown headers had the most associated tagged records
    # List of which markdown headers had the least associated tagged records
    # was there oversaturation of tags?
    # was there undersaturation of tags?
    # any other insights I may be missing in this docstring

import argparse
from typing import Any, Dict, Union

import pandas as pd

from Taggers import RunResult, flatten_records, load_run

# A tag firing on more chunks than this fraction is flagged as oversaturated;
# fewer than this fraction (but still > 0) is flagged as undersaturated.
_OVERSATURATION_THRESHOLD = 0.75
_UNDERSATURATION_THRESHOLD = 0.05


def load_run_as_df(run: Union[str, RunResult]) -> pd.DataFrame:
    """Load a RunResult (or path to its saved JSON) into a flat, match-level DataFrame.

    One row per (chunk, tag match); a chunk with N tags appears N times, a chunk
    with 0 tags appears once with tag_name=None (see Taggers.flatten_records).
    """
    run_result = load_run(run) if isinstance(run, str) else run
    return pd.DataFrame(flatten_records(run_result))


def chunk_level_df(flat_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the flat match-level frame to one row per chunk.

    Columns: chunk_id, section_titles, chunk_text, chunk_length, tag_count.
    """
    def _section_path(section_titles: Any) -> str:
        if not section_titles:
            return "(no section)"
        return " > ".join(section_titles)

    chunks = (
        flat_df.groupby("chunk_id", sort=False)
        .agg(
            section_titles=("section_titles", "first"),
            chunk_text=("chunk_text", "first"),
            tag_count=("tag_name", lambda s: s.notna().sum()),
        )
        .reset_index()
    )
    chunks["chunk_length"] = chunks["chunk_text"].str.len()
    chunks["section_path"] = chunks["section_titles"].apply(_section_path)
    return chunks


def chunk_size_stats(chunk_df: pd.DataFrame) -> Dict[str, float]:
    lengths = chunk_df["chunk_length"]
    return {
        "count": int(lengths.count()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 1),
        "median": float(lengths.median()),
    }


def tagged_vs_untagged_counts(chunk_df: pd.DataFrame) -> Dict[str, Any]:
    total = len(chunk_df)
    tagged = int((chunk_df["tag_count"] > 0).sum())
    untagged = total - tagged
    return {
        "total_records": total,
        "tagged_records": tagged,
        "untagged_records": untagged,
        "pct_tagged": round(100 * tagged / total, 1) if total else 0.0,
    }


def section_tag_counts(chunk_df: pd.DataFrame) -> pd.Series:
    """Count of tagged chunks per markdown header path, sorted descending.

    Headers with zero tagged chunks appear at the bottom (count 0), which is
    itself informative for the "least associated" question.
    """
    tagged = chunk_df.assign(is_tagged=chunk_df["tag_count"] > 0)
    counts = tagged.groupby("section_path")["is_tagged"].sum().astype(int)
    return counts.sort_values(ascending=False)


def tag_frequency(flat_df: pd.DataFrame) -> pd.Series:
    """Count of distinct chunks each tag appears on, sorted descending."""
    matched = flat_df[flat_df["tag_name"].notna()]
    return matched.groupby("tag_name")["chunk_id"].nunique().sort_values(ascending=False)


def tags_per_chunk_distribution(chunk_df: pd.DataFrame) -> pd.Series:
    """Histogram: number of chunks having 0, 1, 2, ... tags."""
    return chunk_df["tag_count"].value_counts().sort_index()


def tag_saturation(chunk_df: pd.DataFrame, flat_df: pd.DataFrame) -> Dict[str, Any]:
    """Per-chunk and per-tag saturation signals.

    Per-chunk: a distribution skewed toward many tags per chunk suggests
    over-tagging; a spike at 0 suggests under-tagging/coverage gaps.
    Per-tag: a tag firing on nearly every chunk is likely too broad
    (oversaturated, low signal); one firing on almost none is likely
    under-triggered (undersaturated - weak phrase list or definition).
    """
    total_chunks = len(chunk_df)
    freq = tag_frequency(flat_df)
    coverage = (freq / total_chunks) if total_chunks else freq

    oversaturated = coverage[coverage > _OVERSATURATION_THRESHOLD].index.tolist()
    undersaturated = coverage[coverage < _UNDERSATURATION_THRESHOLD].index.tolist()

    return {
        "chunks_per_tag_count": tags_per_chunk_distribution(chunk_df),
        "chunk_frequency_per_tag": freq,
        "tag_coverage_pct": (coverage * 100).round(1),
        "oversaturated_tags": oversaturated,
        "undersaturated_tags": undersaturated,
    }


def tag_cooccurrence(flat_df: pd.DataFrame) -> pd.DataFrame:
    """Square matrix: how often each pair of tags fires on the same chunk.

    Redundant/overlapping tags are a common cause of per-chunk oversaturation;
    this pinpoints which tags to consider consolidating.
    """
    matched = flat_df[flat_df["tag_name"].notna()]
    indicator = pd.crosstab(matched["chunk_id"], matched["tag_name"]).clip(upper=1)
    return indicator.T.dot(indicator)


def phrase_usage(flat_df: pd.DataFrame) -> pd.Series:
    """(string_match runs) count of distinct chunks each matched phrase fired on."""
    matched = flat_df[flat_df["matched_phrase"].notna()]
    return matched.groupby("matched_phrase")["chunk_id"].nunique().sort_values(ascending=False)


def evidence_stats(flat_df: pd.DataFrame) -> Dict[str, Any]:
    """(llm runs) how often tag matches carry reasoning, and how substantial it is."""
    matched = flat_df[flat_df["tag_name"].notna()]
    reasoning = matched["evidence"].apply(
        lambda e: e.get("reasoning") if isinstance(e, dict) else None
    )
    has_reasoning = reasoning.fillna("").str.strip().str.len() > 0

    return {
        "records_with_evidence": int(has_reasoning.sum()),
        "records_missing_evidence": int((~has_reasoning).sum()),
        "avg_evidence_length": round(float(reasoning.dropna().str.len().mean()), 1)
        if has_reasoning.any()
        else 0.0,
    }


def summary_report(run: Union[str, RunResult]) -> Dict[str, Any]:
    """Bundle every stat above into one human-readable report structure."""
    flat_df = load_run_as_df(run)
    chunk_df = chunk_level_df(flat_df)
    tagger_method = flat_df["tagger_method"].iloc[0] if not flat_df.empty else None

    report: Dict[str, Any] = {
        "tagger_method": tagger_method,
        "chunk_size": chunk_size_stats(chunk_df),
        "tagged_vs_untagged": tagged_vs_untagged_counts(chunk_df),
        "section_tag_counts": section_tag_counts(chunk_df),
        "saturation": tag_saturation(chunk_df, flat_df),
        "tag_cooccurrence": tag_cooccurrence(flat_df),
    }

    if tagger_method == "string_match":
        report["phrase_usage"] = phrase_usage(flat_df)
    elif tagger_method == "llm":
        report["evidence_stats"] = evidence_stats(flat_df)

    return report


def format_report(report: Dict[str, Any], top_n: int = 5) -> str:
    lines = []

    lines.append(f"=== Tagging Analysis Report ({report['tagger_method']}) ===\n")

    cs = report["chunk_size"]
    lines.append("Chunk size (characters):")
    lines.append(f"  count={cs['count']}  min={cs['min']}  max={cs['max']}  mean={cs['mean']}  median={cs['median']}\n")

    tv = report["tagged_vs_untagged"]
    lines.append("Coverage:")
    lines.append(
        f"  {tv['tagged_records']}/{tv['total_records']} chunks tagged "
        f"({tv['pct_tagged']}%), {tv['untagged_records']} untagged\n"
    )

    sections = report["section_tag_counts"]
    lines.append(f"Top {top_n} headers by tagged records:")
    for section, count in sections.head(top_n).items():
        lines.append(f"  {count:>4}  {section}")
    lines.append(f"\nBottom {top_n} headers by tagged records:")
    for section, count in sections.tail(top_n).items():
        lines.append(f"  {count:>4}  {section}")
    lines.append("")

    sat = report["saturation"]
    lines.append("Tags per chunk distribution:")
    for tag_count, num_chunks in sat["chunks_per_tag_count"].items():
        lines.append(f"  {tag_count} tag(s): {num_chunks} chunk(s)")
    lines.append("")

    lines.append("Tag frequency (chunks matched / coverage %):")
    for tag_name, count in sat["chunk_frequency_per_tag"].items():
        pct = sat["tag_coverage_pct"].get(tag_name, 0.0)
        lines.append(f"  {tag_name}: {count} chunks ({pct}%)")
    lines.append("")

    if sat["oversaturated_tags"]:
        lines.append(f"Possibly OVERSATURATED (fires on >{int(_OVERSATURATION_THRESHOLD * 100)}% of chunks):")
        for tag_name in sat["oversaturated_tags"]:
            lines.append(f"  - {tag_name}")
        lines.append("")

    if sat["undersaturated_tags"]:
        lines.append(f"Possibly UNDERSATURATED (fires on <{int(_UNDERSATURATION_THRESHOLD * 100)}% of chunks):")
        for tag_name in sat["undersaturated_tags"]:
            lines.append(f"  - {tag_name}")
        lines.append("")

    cooc = report["tag_cooccurrence"]
    if not cooc.empty:
        lines.append("Tag co-occurrence matrix:")
        lines.append("  " + cooc.to_string().replace("\n", "\n  "))
        lines.append("")

    if "phrase_usage" in report:
        lines.append("Phrase usage (string_match):")
        for phrase, count in report["phrase_usage"].items():
            lines.append(f"  {phrase}: {count} chunks")
        lines.append("")

    if "evidence_stats" in report:
        ev = report["evidence_stats"]
        lines.append("LLM evidence quality:")
        lines.append(
            f"  with_reasoning={ev['records_with_evidence']}  "
            f"missing_reasoning={ev['records_missing_evidence']}  "
            f"avg_reasoning_length={ev['avg_evidence_length']}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a human-readable analysis report for a tagging run.")
    parser.add_argument("--run", required=True, help="Path to a saved RunResult JSON file")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top/bottom sections to show")
    args = parser.parse_args()

    report = summary_report(args.run)
    print(format_report(report, top_n=args.top_n))


if __name__ == "__main__":
    main()
