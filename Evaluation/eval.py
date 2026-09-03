"""Evaluate LLM extraction output against hand-labeled ground truth.

Ground truth is a .xlsx export (e.g. from a Google Sheet) with a 'pdf_name'
column identifying the paper each row belongs to, and one column per
extracted field (matching the param_name keys used by the Extraction
pipeline). Multiple rows can share a pdf_name; the ground-truth value set
for a field is the set of unique non-null values across those rows.

Ground-truth column headers commonly differ from our param names in casing/
spacing/punctuation (e.g. "Coagulant Dose", "Virus", "pH_before_treatment",
"airborne/stationary" vs. coagulant_dose, virus, ph_before_treatment,
airborne_stationary). Column names are matched to ExtractionStore keys on a
normalized form (see normalize_field_name): lowercased, whitespace/hyphens/
slashes collapsed to underscores - not exact string equality.

Extraction output for a given paper is expected at
<output_dir>/<pdf_name>/<extraction-glob> (see runner.py/run_batch.py
--output-dir), where <pdf_name> is the source markdown filename without its
.md extension.

The ground truth's 'pdf_name' column holds the *original PDF* filename
(e.g. "1 Fate of Coronaviruses ... Ferric Chloride.pdf"), while the output
directory is named after the *markdown* filename produced by document
parsing, which commonly differs in case, spacing/punctuation, and can carry
an extra suffix like "-with-image-refs" (e.g.
"1-Fate-of-Coronaviruses-...-Ferric-Chloride-with-image-refs"). Matching
between the two is done on a normalized form (see normalize_pdf_name):
lowercased, extension stripped, whitespace/underscores collapsed to hyphens,
and known suffixes (_PDF_NAME_SUFFIXES_TO_STRIP) stripped. Ground-truth rows
are matched to output directories by this normalized key, not by exact
string equality.

Each paper type (loss_survival / uv / cfs) has its own ground-truth .xlsx --
pass the matching one via --ground-truth for whichever --output-dir you're
evaluating.

Usage (run as a module from the repo root, so the Extraction package resolves):
    python -m Evaluation.eval --ground-truth ground_truth.xlsx
    python -m Evaluation.eval --ground-truth gt.xlsx --output-dir output --pdf-name my_paper
"""

import argparse
import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz

from Extraction import ExtractionStore, load_store

GroundTruth = Dict[str, Dict[str, List[Any]]]

_DEFAULT_OUTPUT_DIR = "output"
_DEFAULT_EXTRACTION_GLOB = "*_extraction.json"
_DEFAULT_REPORT_PATH = "Evaluation/eval_report.csv"
_DEFAULT_FUZZY_THRESHOLD = 90.0
_PDF_NAME_COLUMN = "pdf_name"

# Suffixes document parsing is known to append to a markdown filename that
# aren't part of the paper's identity, stripped during pdf_name normalization.
_PDF_NAME_SUFFIXES_TO_STRIP = ("-with-image-refs",)


def normalize_pdf_name(name: str) -> str:
    """Canonicalize a pdf_name (ground truth) or markdown stem (output dir) for matching.

    Lowercases, strips the extension (.pdf/.md/...), collapses whitespace/
    underscores/repeated hyphens into single hyphens, and strips any known
    non-identity suffix (see _PDF_NAME_SUFFIXES_TO_STRIP).
    """
    stem = os.path.splitext(str(name).strip())[0]
    stem = stem.lower()
    stem = re.sub(r"[\s_]+", "-", stem)
    stem = re.sub(r"-+", "-", stem).strip("-")

    for suffix in _PDF_NAME_SUFFIXES_TO_STRIP:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return stem


def normalize_field_name(name: str) -> str:
    """Canonicalize a ground-truth column name or ExtractionStore key for matching.

    Lowercases and collapses whitespace/hyphens/slashes/repeated underscores
    into single underscores, so ground-truth columns like "Coagulant Dose",
    "Virus", "pH_before_treatment", "airborne/stationary" line up with our
    snake_case param names (coagulant_dose, virus, ph_before_treatment,
    airborne_stationary) despite differing casing/spacing/punctuation.
    """
    key = str(name).strip().lower()
    key = re.sub(r"[\s/-]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", required=True, help="Path to the ground-truth .xlsx file")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR, help="Base directory containing <pdf_name>/ subdirectories of extraction output")
    parser.add_argument("--extraction-glob", default=_DEFAULT_EXTRACTION_GLOB, help="Glob pattern (within <output_dir>/<pdf_name>/) for the extraction JSON to evaluate")
    parser.add_argument("--pdf-name", help="Restrict evaluation to a single pdf_name")
    parser.add_argument("--report", default=_DEFAULT_REPORT_PATH, help="Path to write the per-field CSV report")
    parser.add_argument("--fuzzy-threshold", type=float, default=_DEFAULT_FUZZY_THRESHOLD, help="Minimum rapidfuzz token_sort_ratio (0-100) for a fuzzy string match")
    return parser.parse_args()


def load_ground_truth(xlsx_path: str) -> GroundTruth:
    """Group xlsx rows by normalized pdf_name; collect unique non-null values per field column."""
    df = pd.read_excel(xlsx_path)
    if _PDF_NAME_COLUMN not in df.columns:
        raise ValueError(f"Ground truth is missing required column '{_PDF_NAME_COLUMN}'")

    field_columns = [c for c in df.columns if c != _PDF_NAME_COLUMN]
    ground_truth: GroundTruth = {}
    seen_raw_names: Dict[str, str] = {}

    for pdf_name, group in df.groupby(_PDF_NAME_COLUMN):
        raw_name = str(pdf_name)
        norm_name = normalize_pdf_name(raw_name)

        if norm_name in seen_raw_names:
            print(
                f"[warn] ground truth pdf_name {raw_name!r} normalizes to the same key "
                f"({norm_name!r}) as {seen_raw_names[norm_name]!r}; rows will be merged"
            )
        seen_raw_names[norm_name] = raw_name

        fields: Dict[str, List[Any]] = ground_truth.get(norm_name, {})
        seen_field_names: Dict[str, str] = {}
        for column in field_columns:
            field_key = normalize_field_name(column)
            if field_key in seen_field_names and seen_field_names[field_key] != column:
                print(
                    f"[warn] ground truth column {column!r} normalizes to the same field "
                    f"({field_key!r}) as {seen_field_names[field_key]!r}; values will be merged"
                )
            seen_field_names[field_key] = column

            values = group[column].dropna().tolist()
            existing = fields.get(field_key, [])
            fields[field_key] = list(dict.fromkeys(existing + values))
        ground_truth[norm_name] = fields

    return ground_truth


def discover_pdf_dirs(output_dir: str) -> Dict[str, str]:
    """Map normalized pdf_name -> actual <output_dir> subdirectory name."""
    mapping: Dict[str, str] = {}
    if not os.path.isdir(output_dir):
        return mapping

    for entry in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, entry)):
            mapping[normalize_pdf_name(entry)] = entry

    return mapping


def find_extraction_file(doc_dir: str, glob_pattern: str) -> Optional[str]:
    matches = glob.glob(os.path.join(doc_dir, glob_pattern))

    if not matches:
        print(f"[warn] no extraction file found in {doc_dir}")
        return None

    if len(matches) > 1:
        matches.sort(key=os.path.getmtime, reverse=True)
        skipped = ", ".join(matches[1:])
        print(f"[warn] multiple extraction files in {doc_dir!r}; using most recent {matches[0]!r}, skipping: {skipped}")

    return matches[0]


def normalize_value(value: Any) -> Tuple[str, Optional[float]]:
    """Return (normalized_string, numeric_value_or_None)."""
    text = str(value).strip().lower()
    text = " ".join(text.split())

    numeric_text = text
    for ch in ("$", ",", "%"):
        numeric_text = numeric_text.replace(ch, "")
    numeric_text = numeric_text.strip()

    try:
        numeric_value: Optional[float] = float(numeric_text)
    except ValueError:
        numeric_value = None

    return text, numeric_value


def values_match(gt_value: Any, extracted_value: Any, fuzzy_threshold: float) -> bool:
    gt_text, gt_number = normalize_value(gt_value)
    ext_text, ext_number = normalize_value(extracted_value)

    if gt_number is not None and ext_number is not None:
        return gt_number == ext_number

    if gt_text == ext_text:
        return True

    return fuzz.token_sort_ratio(gt_text, ext_text) >= fuzzy_threshold


def match_field(gt_values: List[Any], extracted_values: List[Any], fuzzy_threshold: float) -> Tuple[int, int, int]:
    """Greedy one-to-one bipartite match; return (tp, fp, fn)."""
    candidates = []
    for gi, gt_value in enumerate(gt_values):
        for ei, extracted_value in enumerate(extracted_values):
            if values_match(gt_value, extracted_value, fuzzy_threshold):
                gt_text, _ = normalize_value(gt_value)
                ext_text, _ = normalize_value(extracted_value)
                score = 100.0 if gt_text == ext_text else fuzz.token_sort_ratio(gt_text, ext_text)
                candidates.append((score, gi, ei))

    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_gt = set()
    matched_ext = set()
    tp = 0
    for _score, gi, ei in candidates:
        if gi in matched_gt or ei in matched_ext:
            continue
        matched_gt.add(gi)
        matched_ext.add(ei)
        tp += 1

    fn = len(gt_values) - len(matched_gt)
    fp = len(extracted_values) - len(matched_ext)
    return tp, fp, fn


def evaluate(
    ground_truth: GroundTruth,
    output_dir: str,
    glob_pattern: str,
    fuzzy_threshold: float,
    only_pdf_name: Optional[str] = None,
) -> pd.DataFrame:
    pdf_names = [normalize_pdf_name(only_pdf_name)] if only_pdf_name else list(ground_truth.keys())
    pdf_dirs = discover_pdf_dirs(output_dir)

    field_counts: Dict[str, Dict[str, int]] = {}
    seen_extracted_fields: set = set()

    for pdf_name in pdf_names:
        fields = ground_truth.get(pdf_name)
        if fields is None:
            print(f"[warn] pdf_name={pdf_name!r} not found in ground truth")
            continue

        doc_dir = pdf_dirs.get(pdf_name)
        if doc_dir is None:
            print(f"[warn] no output directory under {output_dir!r} matches pdf_name={pdf_name!r}")
            extraction_path = None
        else:
            extraction_path = find_extraction_file(os.path.join(output_dir, doc_dir), glob_pattern)

        store: ExtractionStore = load_store(extraction_path) if extraction_path else {}
        seen_extracted_fields.update(store.keys())

        for field, gt_values in fields.items():
            extracted_values = store.get(field, [])
            tp, fp, fn = match_field(gt_values, extracted_values, fuzzy_threshold)

            counts = field_counts.setdefault(field, {"tp": 0, "fp": 0, "fn": 0})
            counts["tp"] += tp
            counts["fp"] += fp
            counts["fn"] += fn

    unknown_fields = {f for f in field_counts if f not in seen_extracted_fields}
    if unknown_fields:
        print(f"[warn] ground truth field(s) never seen in any extraction store: {sorted(unknown_fields)}")

    rows = []
    for field, counts in field_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"field": field, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})

    report = pd.DataFrame(rows).sort_values("f1", ascending=True).reset_index(drop=True)

    if not report.empty:
        summary = {
            "field": "MACRO_AVERAGE",
            "tp": report["tp"].sum(),
            "fp": report["fp"].sum(),
            "fn": report["fn"].sum(),
            "precision": report["precision"].mean(),
            "recall": report["recall"].mean(),
            "f1": report["f1"].mean(),
        }
        report = pd.concat([report, pd.DataFrame([summary])], ignore_index=True)

    return report


def main() -> None:
    args = parse_args()
    ground_truth = load_ground_truth(args.ground_truth)
    report = evaluate(
        ground_truth,
        output_dir=args.output_dir,
        glob_pattern=args.extraction_glob,
        fuzzy_threshold=args.fuzzy_threshold,
        only_pdf_name=args.pdf_name,
    )

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    report.to_csv(args.report, index=False)

    print(report.to_string(index=False))
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
