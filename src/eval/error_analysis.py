"""Surface validation errors for manual review.

This does not write the final error-analysis report by itself — recurring
failure-mode judgment calls need a human (or human-like) reader looking at
actual examples. It groups errors by (true, predicted) confusion pair and
prints/saves the underlying records so that judgment call can be made and
written up in reports/error_analysis.md.

Run: python -m src.eval.error_analysis --predictions reports/validation_predictions_slm.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default=str(REPORTS_DIR / "validation_predictions_slm.jsonl"))
    p.add_argument("--top-pairs", type=int, default=10)
    p.add_argument("--examples-per-pair", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.predictions, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    errors = [r for r in rows if not r["correct"]]
    print(f"Total validation rows: {len(rows)}; errors: {len(errors)} ({len(errors) / len(rows) * 100:.1f}%)\n")

    pair_counts = Counter((r["true_category"], r["predicted_category"]) for r in errors)
    print("Confusion pairs (true -> predicted), most frequent first:")
    for (true, pred), count in pair_counts.most_common(args.top_pairs):
        print(f"  {true:>12} -> {pred:<12} : {count}")

    grouped = {pair: [] for pair in pair_counts}
    for r in errors:
        grouped[(r["true_category"], r["predicted_category"])].append(r)

    dump = {}
    for pair, count in pair_counts.most_common(args.top_pairs):
        examples = grouped[pair][: args.examples_per_pair]
        dump[f"{pair[0]}_to_{pair[1]}"] = [
            {
                "record_id": e["record_id"],
                "server_slug": e["server_slug"],
                "name": e["name"],
                "description": e["description"][:400],
            }
            for e in examples
        ]

    out_path = REPORTS_DIR / "error_analysis" / "confusion_examples.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, indent=2)
    print(f"\nSaved grouped examples to {out_path}")


if __name__ == "__main__":
    main()
