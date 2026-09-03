# Side-by-side comparison of the tag-based extraction store against the
# semantic-baseline extraction store for one paper, field by field -- plus
# ground truth when available. Meant as a manual sanity check independent of
# eval.py -- useful when there's no ground truth yet, or eval.py itself
# fails for some reason.

import csv
import html
from typing import Any, Dict, List, Optional

from Extraction.params import ParamDefinition
from Extraction.schema import ExtractionStore

from .eval import values_match

_DEFAULT_FUZZY_THRESHOLD = 90.0

_CSV_FIELDNAMES = [
    "field",
    "ground_truth_values",
    "tag_based_values",
    "semantic_values",
    "tag_count",
    "semantic_count",
    "tag_matches_gt",
    "semantic_matches_gt",
    "tag_semantic_agree",
]


def build_comparison_rows(
    tag_store: ExtractionStore,
    semantic_store: ExtractionStore,
    params: List[ParamDefinition],
    ground_truth_fields: Optional[Dict[str, List[Any]]] = None,
    fuzzy_threshold: float = _DEFAULT_FUZZY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """One row per param: values from each store side by side, plus whether
    tag-based/semantic agree with each other and (when ground_truth_fields is
    given, keyed the same way as ExtractionStore -- see Evaluation.eval's
    normalize_field_name) with ground truth. Same fuzzy matching logic as
    Evaluation.eval, for consistency with the real eval pipeline."""
    ground_truth_fields = ground_truth_fields or {}

    rows = []
    for param in params:
        tag_values = tag_store.get(param.name, [])
        semantic_values = semantic_store.get(param.name, [])
        gt_values = ground_truth_fields.get(param.name, [])

        tag_semantic_agree = any(values_match(t, s, fuzzy_threshold) for t in tag_values for s in semantic_values)
        tag_matches_gt = any(values_match(g, t, fuzzy_threshold) for g in gt_values for t in tag_values) if gt_values else None
        semantic_matches_gt = any(values_match(g, s, fuzzy_threshold) for g in gt_values for s in semantic_values) if gt_values else None

        rows.append(
            {
                "field": param.name,
                "ground_truth_values": "; ".join(str(v) for v in gt_values),
                "tag_based_values": "; ".join(str(v) for v in tag_values),
                "semantic_values": "; ".join(str(v) for v in semantic_values),
                "tag_count": len(tag_values),
                "semantic_count": len(semantic_values),
                "tag_matches_gt": tag_matches_gt,
                "semantic_matches_gt": semantic_matches_gt,
                "tag_semantic_agree": tag_semantic_agree,
            }
        )
    return rows


def write_comparison_csv(rows: List[Dict[str, Any]], path: str) -> str:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return path


def _bool_cell(value: Optional[bool]) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def write_comparison_html(rows: List[Dict[str, Any]], path: str, title: str) -> str:
    has_gt = any(r["ground_truth_values"] for r in rows)
    gt_hits = sum(1 for r in rows if r["tag_matches_gt"] or r["semantic_matches_gt"])
    gt_total = sum(1 for r in rows if r["ground_truth_values"])
    agree_count = sum(1 for r in rows if r["tag_semantic_agree"])

    body_rows = []
    for r in rows:
        if r["ground_truth_values"]:
            row_class = "gt_hit" if (r["tag_matches_gt"] or r["semantic_matches_gt"]) else "gt_miss"
        elif r["tag_semantic_agree"]:
            row_class = "agree"
        elif not r["tag_based_values"] and not r["semantic_values"]:
            row_class = "empty"
        else:
            row_class = "disagree"

        body_rows.append(
            f'<tr class="{row_class}">'
            f'<td>{html.escape(r["field"])}</td>'
            + (f'<td>{html.escape(r["ground_truth_values"]) or "&mdash;"}</td>' if has_gt else "")
            + f'<td>{html.escape(r["tag_based_values"]) or "&mdash;"}</td>'
            f'<td>{html.escape(r["semantic_values"]) or "&mdash;"}</td>'
            + (
                f'<td>{_bool_cell(r["tag_matches_gt"])}</td><td>{_bool_cell(r["semantic_matches_gt"])}</td>'
                if has_gt
                else ""
            )
            + f'<td>{_bool_cell(r["tag_semantic_agree"])}</td>'
            f"</tr>"
        )

    gt_header = "<th>Ground truth</th>" if has_gt else ""
    gt_match_headers = "<th>Tag matches GT?</th><th>Semantic matches GT?</th>" if has_gt else ""
    summary = (
        f"{gt_hits}/{gt_total} field(s) with ground truth had at least one method match it (fuzzy match, same logic as Evaluation.eval)"
        if has_gt
        else f"{agree_count}/{len(rows)} field(s) agree between tag-based and semantic extraction (no ground truth given for this paper)"
    )

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
<tr><th>Field</th>{gt_header}<th>Tag-based values</th><th>Semantic values</th>{gt_match_headers}<th>Tag/semantic agree?</th></tr>
{''.join(body_rows)}
</table>
</body>
</html>
"""

    with open(path, "w") as f:
        f.write(document)

    return path
