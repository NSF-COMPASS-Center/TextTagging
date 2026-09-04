"""Compare every extraction_type's output already produced for one paper
(separate runner.py invocations into the same output dir, one per
extraction_type) against ground truth, side by side, in a single N-column
HTML.

Usage (run as a module from the repo root, so the Extraction package resolves):
    python -m Evaluation.compare_all --ground-truth gt.xlsx --pdf-name my_paper \
        --params tags/loss_survival/params.json
    python -m Evaluation.compare_all --ground-truth gt.xlsx --pdf-name my_paper \
        --params tags/loss_survival/params.json --output-dir output --out my_paper_all.html
"""

import argparse
import glob
import os

from Extraction.params import create_param_definitions
from Extraction.schema import load_store

from .compare import build_comparison_rows, write_comparison_html
from .eval import discover_pdf_dirs, load_ground_truth, normalize_pdf_name

_DEFAULT_OUTPUT_DIR = "output"
_DEFAULT_GLOB = "*_extraction.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ground-truth", required=True, help="Path to the ground-truth .xlsx file")
    parser.add_argument("--pdf-name", required=True, help="The paper to compare (matches an <output_dir>/<pdf_name>/ subdirectory)")
    parser.add_argument("--params", required=True, help="Extraction params JSON path ({name: [type, definition]})")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR, help="Base directory containing <pdf_name>/ subdirectories of extraction output")
    parser.add_argument("--glob", default=_DEFAULT_GLOB, help="Glob (within <output_dir>/<pdf_name>/) for the extraction JSON file(s) to compare")
    parser.add_argument("--fuzzy-threshold", type=float, default=90.0, help="Minimum rapidfuzz token_sort_ratio (0-100) for a fuzzy string match")
    parser.add_argument("--out", help="Output HTML path (default: <output_dir>/<pdf_name>/all_comparison.html)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pdf_dirs = discover_pdf_dirs(args.output_dir)
    norm_name = normalize_pdf_name(args.pdf_name)
    doc_dir_name = pdf_dirs.get(norm_name)
    if doc_dir_name is None:
        raise ValueError(f"No output directory under {args.output_dir!r} matches pdf_name={args.pdf_name!r}")
    doc_dir = os.path.join(args.output_dir, doc_dir_name)

    store_paths = sorted(glob.glob(os.path.join(doc_dir, args.glob)))
    if not store_paths:
        raise ValueError(f"No extraction files matching {args.glob!r} found in {doc_dir!r}")

    stores = {}
    for path in store_paths:
        name = os.path.basename(path)
        if name.endswith("_extraction.json"):
            name = name[: -len("_extraction.json")]
        stores[name] = load_store(path)

    params = create_param_definitions(args.params)
    ground_truth = load_ground_truth(args.ground_truth)
    gt_fields = ground_truth.get(norm_name, {})

    rows = build_comparison_rows(stores, params, ground_truth_fields=gt_fields, fuzzy_threshold=args.fuzzy_threshold)

    out_path = args.out or os.path.join(doc_dir, "all_comparison.html")
    write_comparison_html(rows, out_path, title=args.pdf_name, store_names=list(stores.keys()))

    print(f"Compared {len(stores)} extraction store(s) ({', '.join(stores.keys())}) for {args.pdf_name!r} -> {out_path}")


if __name__ == "__main__":
    main()
