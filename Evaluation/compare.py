# Side-by-side comparison of one or more extraction stores for one paper,
# field by field -- plus ground truth when available. Meant as a manual
# sanity check independent of eval.py -- useful when there's no ground truth
# yet, or eval.py itself fails for some reason. Store-count-agnostic: a run
# through runner.py/run_batch.py produces a single store (one extraction_type
# per invocation), while Evaluation/compare_all.py can pass several stores
# (from separate runs on the same document) to compare side by side.

import csv
import html
import itertools
from typing import Any, Dict, List, Optional

from Extraction.params import ParamDefinition
from Extraction.schema import ExtractionStore

from .eval import values_match

_DEFAULT_FUZZY_THRESHOLD = 90.0


def _csv_fieldnames(store_names: List[str]) -> List[str]:
    return (
        ["field", "ground_truth_values"]
        + [f"{name}_values" for name in store_names]
        + [f"{name}_matches_gt" for name in store_names]
        + (["stores_agree"] if len(store_names) >= 2 else [])
    )


def build_comparison_rows(
    stores: Dict[str, ExtractionStore],
    params: List[ParamDefinition],
    ground_truth_fields: Optional[Dict[str, List[Any]]] = None,
    fuzzy_threshold: float = _DEFAULT_FUZZY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """One row per param: values from each store (keyed by store/extraction_type
    name) side by side, plus (when there are >=2 stores) whether any pair of
    them agree with each other, and (when ground_truth_fields is given, keyed
    the same way as ExtractionStore -- see Evaluation.eval's
    normalize_field_name) whether each store matches ground truth. Same fuzzy
    matching logic as Evaluation.eval, for consistency with the real eval
    pipeline."""
    ground_truth_fields = ground_truth_fields or {}
    store_names = list(stores.keys())

    rows = []
    for param in params:
        values_by_store = {name: store.get(param.name, []) for name, store in stores.items()}
        gt_values = ground_truth_fields.get(param.name, [])

        row: Dict[str, Any] = {
            "field": param.name,
            "ground_truth_values": "; ".join(str(v) for v in gt_values),
        }
        for name, values in values_by_store.items():
            row[f"{name}_values"] = "; ".join(str(v) for v in values)
            row[f"{name}_matches_gt"] = (
                any(values_match(g, v, fuzzy_threshold) for g in gt_values for v in values) if gt_values else None
            )

        if len(store_names) >= 2:
            row["stores_agree"] = any(
                values_match(a, b, fuzzy_threshold)
                for values_a, values_b in itertools.combinations(values_by_store.values(), 2)
                for a in values_a
                for b in values_b
            )

        rows.append(row)
    return rows


def write_comparison_csv(rows: List[Dict[str, Any]], path: str, store_names: List[str]) -> str:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fieldnames(store_names))
        writer.writeheader()
        writer.writerows(rows)

    return path


def _bool_cell(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def write_comparison_html(rows: List[Dict[str, Any]], path: str, title: str, store_names: List[str]) -> str:
    has_gt = any(r["ground_truth_values"] for r in rows)
    multi_store = len(store_names) >= 2

    gt_hits = sum(1 for r in rows if any(r[f"{name}_matches_gt"] for name in store_names)) if has_gt else 0
    gt_total = sum(1 for r in rows if r["ground_truth_values"])
    agree_count = sum(1 for r in rows if r.get("stores_agree")) if multi_store else 0

    body_rows = []
    for r in rows:
        any_values = any(r[f"{name}_values"] for name in store_names)
        if r["ground_truth_values"]:
            row_class = "gt_hit" if any(r[f"{name}_matches_gt"] for name in store_names) else "gt_miss"
        elif multi_store and r.get("stores_agree"):
            row_class = "agree"
        elif not any_values:
            row_class = "empty"
        elif multi_store:
            row_class = "disagree"
        else:
            row_class = ""

        value_cells = "".join(f'<td>{html.escape(r[f"{name}_values"]) or "&mdash;"}</td>' for name in store_names)
        gt_match_cells = "".join(f'<td>{_bool_cell(r[f"{name}_matches_gt"])}</td>' for name in store_names) if has_gt else ""
        agree_cell = f'<td>{_bool_cell(r.get("stores_agree"))}</td>' if multi_store else ""

        body_rows.append(
            f'<tr class="{row_class}">'
            f'<td>{html.escape(r["field"])}</td>'
            + (f'<td>{html.escape(r["ground_truth_values"]) or "&mdash;"}</td>' if has_gt else "")
            + value_cells
            + gt_match_cells
            + agree_cell
            + "</tr>"
        )

    gt_header = "<th>Ground truth</th>" if has_gt else ""
    gt_match_headers = "".join(f"<th>{html.escape(name)} matches GT?</th>" for name in store_names) if has_gt else ""
    agree_header = "<th>Stores agree?</th>" if multi_store else ""
    value_headers = "".join(f"<th>{html.escape(name)} values</th>" for name in store_names)

    if has_gt:
        summary = f"{gt_hits}/{gt_total} field(s) with ground truth had at least one store match it (fuzzy match, same logic as Evaluation.eval)"
    elif multi_store:
        summary = f"{agree_count}/{len(rows)} field(s) agree across stores (no ground truth given for this paper)"
    else:
        summary = f"{len(rows)} field(s) extracted (no ground truth given for this paper)"

    document = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)} - extraction comparison</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.1rem; }}
  .summary {{ color: #555; margin-bottom: 1rem; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: top; font-size: 0.9rem; }}
  th {{ background: #f4f4f4; }}
  tr.agree {{ background: #eaf7ea; }}
  tr.disagree {{ background: #fdecea; }}
  tr.empty {{ background: #fafafa; color: #999; }}
  tr.gt_hit {{ background: #eaf7ea; }}
  tr.gt_miss {{ background: #fdecea; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="summary">{summary}</div>
<table>
<tr><th>Field</th>{gt_header}{value_headers}{gt_match_headers}{agree_header}</tr>
{''.join(body_rows)}
</table>
</body>
</html>
"""

    with open(path, "w") as f:
        f.write(document)

    return path
