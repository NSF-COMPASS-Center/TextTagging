"""Batch runner: for one paper type, run tagging (string_match or llm --
whichever the config's `tagger` says; `tagger: both` isn't supported here,
use runner.py directly for that) + tag analysis + tag-based/semantic
extraction over a list of input markdown papers, then (optionally) evaluate
the extraction output against hand-labeled ground truth.

Each input's output lands in <output_dir>/<pdf_name>/ (pdf_name = the input
filename's stem), matching the directory convention Evaluation/eval.py expects.

Usage:
    python run_batch.py --config run_loss_survival.yaml --input a.md b.md c.md d.md e.md
    python run_batch.py --config run_cfs.yaml --input a.md b.md c.md d.md e.md \
        --ground-truth ground_truth_cfs.xlsx
"""

import argparse
import os
from typing import List, Optional

from dotenv import load_dotenv

import runner
from Evaluation.compare import build_comparison_rows, write_comparison_csv, write_comparison_html
from Evaluation.eval import GroundTruth, evaluate, load_ground_truth, normalize_pdf_name
from Extraction.params import create_param_definitions
from Tagging_Analysis.tag_analysis import format_report, summary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML config for this paper type (e.g. run_cfs.yaml)")
    parser.add_argument("--input", nargs="+", required=True, help="Markdown file(s) to run the pipeline on")
    parser.add_argument(
        "--output-dir",
        help="Base output directory; overrides the config's output_dir. "
             "Each input gets its own <output_dir>/<pdf_name>/ subdirectory",
    )
    parser.add_argument("--top-n", type=int, default=5, help="top_n passed to tag_analysis.format_report")
    parser.add_argument("--ground-truth", help="Path to a ground-truth .xlsx; if given, evaluation runs after all inputs finish")
    parser.add_argument("--extraction-glob", default="*_extraction.json", help="Glob (within each pdf_name dir) for the extraction file(s) to evaluate")
    parser.add_argument("--eval-report", help="Where to write the evaluation CSV (default: <output_dir>/eval_report.csv)")
    return parser.parse_args()


def _cli_namespace(config: str, input_path: str, output_dir: str) -> argparse.Namespace:
    """Namespace with every field runner.build_settings reads via getattr.

    Only config/input/output_dir are set here; tagger, paper_type, extraction
    settings, etc. all come from the YAML config so every input in the batch
    runs with identical settings.
    """
    return argparse.Namespace(
        config=config,
        input=input_path,
        paper_type=None,
        tagger=None,
        max_chunk_size=None,
        sentences_per_chunk=None,
        string_match_tags=None,
        llm_tags=None,
        tagging_llm_provider=None,
        tagging_llm_model=None,
        tagging_llm_host=None,
        output_dir=output_dir,
    )


def run_one(
    config: str,
    input_path: str,
    base_output_dir: str,
    top_n: int,
    ground_truth: Optional[GroundTruth] = None,
) -> None:
    pdf_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(base_output_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    args = _cli_namespace(config, input_path, output_dir)
    settings = runner.build_settings(args)

    tagger = settings["tagger"]
    if tagger not in ("string_match", "llm"):
        raise ValueError(
            f"run_batch.py only supports a single tagger method (string_match or llm), "
            f"got tagger: {tagger!r} in {config} -- tagger: both isn't supported here (use runner.py directly)"
        )

    extract = settings.get("extract")
    client = None
    semantic_store = None
    tag_store = None

    # Semantic runs first: it re-parses the markdown independently and doesn't
    # depend on tagging at all, so it's run before string-match tagging/analysis.
    if extract in ("semantic", "both"):
        client = client or runner.build_llm_client(settings, "extraction")
        semantic_store = runner.run_semantic_extraction(settings, client)
        print(f"[{pdf_name}] extraction:semantic {sum(len(v) for v in semantic_store.values())} unique value(s)")

    with open(settings["input"], "r") as f:
        text = f.read()
    chunks = runner.parse_markdown(
        text, max_chunk_size=settings.get("max_chunk_size"), sentences_per_chunk=settings.get("sentences_per_chunk"),
    )

    if tagger == "string_match":
        result = runner.run_string_match(settings, chunks)
    else:
        result = runner.run_llm(settings, chunks)

    flat_path = runner.export_flat(result, settings["output_dir"])
    match_count = sum(len(record.matches) for record in result.records)
    print(f"[{pdf_name}] tagging ({tagger}): run_id={result.config.run_id} records={len(result.records)} matches={match_count} -> {flat_path}")

    report_text = format_report(summary_report(result), top_n=top_n)
    report_path = os.path.join(output_dir, f"{result.config.run_id}_analysis.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"[{pdf_name}] tag analysis -> {report_path}")

    if extract in ("tags", "both"):
        client = client or runner.build_llm_client(settings, "extraction")
        tag_store = runner.run_tag_extraction(settings, result, client)
        print(f"[{pdf_name}] extraction:tags {sum(len(v) for v in tag_store.values())} unique value(s)")

    # Side-by-side comparison of the two extraction methods, independent of
    # eval.py/ground truth -- always written when both ran, so there's a
    # manual sanity check even if evaluation has no ground truth or fails.
    if tag_store is not None and semantic_store is not None:
        params = create_param_definitions(settings["extraction_params"])
        gt_fields = (ground_truth or {}).get(normalize_pdf_name(pdf_name), {})
        rows = build_comparison_rows(tag_store, semantic_store, params, ground_truth_fields=gt_fields)
        csv_path = write_comparison_csv(rows, os.path.join(output_dir, f"{pdf_name}_comparison.csv"))
        html_path = write_comparison_html(rows, os.path.join(output_dir, f"{pdf_name}_comparison.html"), title=pdf_name)
        print(f"[{pdf_name}] comparison -> {csv_path}, {html_path}")


def run_evaluation(ground_truth: GroundTruth, output_dir: str, glob_pattern: str, report_path: str) -> None:
    report = evaluate(ground_truth, output_dir=output_dir, glob_pattern=glob_pattern, fuzzy_threshold=90.0)

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    report.to_csv(report_path, index=False)

    print(f"\n=== Evaluation ({output_dir}) ===")
    print(report.to_string(index=False))
    print(f"Evaluation report -> {report_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    config_data = runner.load_config(args.config)
    output_dir = args.output_dir or config_data.get("output_dir", "output")

    # Loaded once up front (not just in run_evaluation) so per-paper comparison
    # files can show ground truth alongside tag-based/semantic values even if
    # the evaluation step at the end fails or --eval-report isn't wanted.
    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else None

    failed = []
    for input_path in args.input:
        try:
            run_one(args.config, input_path, output_dir, args.top_n, ground_truth=ground_truth)
        except Exception as e:
            print(f"[{input_path}] FAILED: {e} - continuing with remaining papers")
            failed.append(input_path)

    if failed:
        print(f"\n{len(failed)} paper(s) failed and were skipped:")
        for path in failed:
            print(f"  - {path}")

    if ground_truth is not None:
        report_path = args.eval_report or os.path.join(output_dir, "eval_report.csv")
        try:
            run_evaluation(ground_truth, output_dir, args.extraction_glob, report_path)
        except Exception as e:
            print(f"\n[eval] evaluation FAILED: {e}")
            print(
                f"[eval] this does not affect the extraction outputs already saved under "
                f"{output_dir}/<pdf_name>/*_extraction.json and semantic_baseline_extraction.json -- "
                f"see each paper's <pdf_name>_comparison.csv/html for a manual side-by-side check instead."
            )


if __name__ == "__main__":
    main()
