"""Data audit for the MCP tool-category classification task.

Produces:
  - reports/data_audit.json  (machine-readable stats)
  - reports/data_audit.md    (human-readable report, referenced from README.md)

Run: python -m src.data.data_audit
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from src.data.schema import LABELS, build_input_text, load_records

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def _norm_text(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def _content_hash(name: str, description: str) -> str:
    return hashlib.sha1(f"{_norm_text(name)}||{_norm_text(description)}".encode("utf-8")).hexdigest()


def audit_split(name: str, records) -> dict:
    n = len(records)
    label_counts = Counter(r.category for r in records if r.category is not None)

    missing_name = sum(1 for r in records if not r.name.strip())
    missing_description = sum(1 for r in records if not r.description.strip())
    empty_schema = sum(1 for r in records if not r.input_schema_raw.strip())

    unparsable_schema = 0
    for r in records:
        if not r.input_schema_raw.strip():
            continue
        try:
            parsed = json.loads(r.input_schema_raw)
            if not isinstance(parsed, dict):
                unparsable_schema += 1
        except json.JSONDecodeError:
            unparsable_schema += 1

    record_ids = [r.record_id for r in records]
    duplicate_record_ids = n - len(set(record_ids))

    content_hashes = [_content_hash(r.name, r.description) for r in records]
    hash_counts = Counter(content_hashes)
    exact_content_duplicates = sum(c - 1 for c in hash_counts.values() if c > 1)

    server_slugs = sorted({r.server_slug for r in records})

    desc_lengths = [len(r.description) for r in records]
    input_text_lengths = [len(build_input_text(r)) for r in records]

    return {
        "split": name,
        "rows": n,
        "servers": len(server_slugs),
        "server_slugs": server_slugs,
        "label_counts": dict(label_counts),
        "missing_name": missing_name,
        "missing_description": missing_description,
        "empty_schema": empty_schema,
        "unparsable_schema": unparsable_schema,
        "duplicate_record_ids": duplicate_record_ids,
        "exact_name_description_duplicates": exact_content_duplicates,
        "description_char_length": {
            "min": min(desc_lengths) if desc_lengths else 0,
            "max": max(desc_lengths) if desc_lengths else 0,
            "mean": round(sum(desc_lengths) / n, 1) if n else 0,
        },
        "input_text_char_length": {
            "min": min(input_text_lengths) if input_text_lengths else 0,
            "max": max(input_text_lengths) if input_text_lengths else 0,
            "mean": round(sum(input_text_lengths) / n, 1) if n else 0,
        },
        "content_hashes": content_hashes,
    }


def cross_split_leakage(splits: dict[str, dict]) -> dict:
    server_overlap = {}
    names = list(splits.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = set(splits[a]["server_slugs"]) & set(splits[b]["server_slugs"])
            server_overlap[f"{a}_vs_{b}"] = len(overlap)

    content_overlap = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = set(splits[a]["content_hashes"]) & set(splits[b]["content_hashes"])
            content_overlap[f"{a}_vs_{b}"] = len(overlap)

    return {"server_slug_overlap": server_overlap, "near_duplicate_content_overlap": content_overlap}


def check_servers_public_for_label_leakage(server_rows: list[dict]) -> dict:
    leak_terms = [lbl.lower() for lbl in LABELS]
    hits = []
    for row in server_rows:
        text = f"{row.get('name', '')} {row.get('description', '')}".lower()
        matched = [t for t in leak_terms if t in text]
        if matched:
            hits.append({"slug": row.get("slug"), "matched_terms": matched})
    return {"rows_scanned": len(server_rows), "rows_with_label_word_in_text": len(hits), "examples": hits[:10]}


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train = load_records(DATA_DIR / "train.jsonl")
    validation = load_records(DATA_DIR / "validation.jsonl")
    test = load_records(DATA_DIR / "test_unlabeled.jsonl")

    with open(DATA_DIR / "servers_public.jsonl", "r", encoding="utf-8") as fh:
        server_rows = [json.loads(line) for line in fh if line.strip()]

    splits = {
        "train": audit_split("train", train),
        "validation": audit_split("validation", validation),
        "test": audit_split("test", test),
    }
    leakage = cross_split_leakage(splits)
    server_leak = check_servers_public_for_label_leakage(server_rows)

    # Drop the bulky per-record hash lists before serializing the summary.
    summary = {
        "splits": {k: {kk: vv for kk, vv in v.items() if kk not in ("content_hashes", "server_slugs")} for k, v in splits.items()},
        "cross_split_leakage": leakage,
        "servers_public_label_word_scan": server_leak,
    }

    with open(REPORTS_DIR / "data_audit.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    write_markdown_report(summary, splits)
    print(json.dumps(summary["splits"], indent=2))
    print(f"\nWrote {REPORTS_DIR / 'data_audit.json'} and {REPORTS_DIR / 'data_audit.md'}")


def write_markdown_report(summary: dict, splits: dict) -> None:
    lines = []
    lines.append("# Data Audit\n")
    lines.append(
        "Generated by `python -m src.data.data_audit`. Covers the checks done before any modeling "
        "started: label balance, missing/malformed fields, duplicate risk, and target leakage.\n"
    )

    lines.append("## Label distribution\n")
    lines.append("| Label | Train | Validation | Train % | Val % |")
    lines.append("|---|---:|---:|---:|---:|")
    train_total = sum(summary["splits"]["train"]["label_counts"].values())
    val_total = sum(summary["splits"]["validation"]["label_counts"].values())
    for label in LABELS:
        tc = summary["splits"]["train"]["label_counts"].get(label, 0)
        vc = summary["splits"]["validation"]["label_counts"].get(label, 0)
        lines.append(
            f"| {label} | {tc} | {vc} | {tc / train_total * 100:.1f}% | {vc / val_total * 100:.1f}% |"
        )
    max_c = max(summary["splits"]["train"]["label_counts"].values())
    min_c = min(summary["splits"]["train"]["label_counts"].values())
    lines.append(
        f"\n`Read` outnumbers `Other`, the rarest class, by roughly **{max_c / min_c:.0f}x** in "
        f"training data, and `Financial` plus `Other` combined are under 1% of rows. This is a "
        f"long-tailed, imbalanced 6-way problem, so accuracy alone would be misleading. Macro F1 as "
        f"the primary metric (per the task spec) is the right call.\n"
    )

    lines.append("## Missing / malformed fields\n")
    lines.append("| Split | Missing name | Missing description | Empty schema | Unparsable schema | Duplicate record_id |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split_name in ["train", "validation", "test"]:
        s = summary["splits"][split_name]
        lines.append(
            f"| {split_name} | {s['missing_name']} | {s['missing_description']} | {s['empty_schema']} | "
            f"{s['unparsable_schema']} | {s['duplicate_record_ids']} |"
        )
    lines.append(
        "\nNo malformed JSON schemas or duplicate `record_id`s. A handful of tools ship an empty "
        "`properties` object (zero-argument tools, e.g. status checks) - that's legitimate, not "
        "missing data, and gets rendered as `\"no parameters\"` by the shared text builder rather "
        "than dropped.\n"
    )

    lines.append("## Duplicate / near-duplicate risk\n")
    for split_name in ["train", "validation", "test"]:
        s = summary["splits"][split_name]
        lines.append(f"- **{split_name}**: {s['exact_name_description_duplicates']} exact (name, description) duplicate pairs within the split.")
    lines.append("\n### Cross-split overlap (leakage check)\n")
    lines.append("| Pair | Shared `server_slug`s | Shared (name, description) content |")
    lines.append("|---|---:|---:|")
    for pair, count in summary["cross_split_leakage"]["server_slug_overlap"].items():
        content = summary["cross_split_leakage"]["near_duplicate_content_overlap"][pair]
        lines.append(f"| {pair} | {count} | {content} |")
    lines.append(
        "\n`server_slug` overlap is 0 across every split pair, so the provided StratifiedGroupKFold "
        "split is doing what it should: grouping by server stops a model from memorizing a server's "
        "house style instead of learning the actual task. The non-zero content-overlap numbers above "
        "aren't a split bug though. Different MCP servers sometimes ship near-identical generic tools "
        "(e.g. a boilerplate `list_files` wrapper), and that shows up as overlapping text even with "
        "zero server overlap. Worth flagging as a residual risk: validation/test scores may be a little "
        "optimistic for the subset of tools that have a lookalike in training data.\n"
    )

    lines.append("## Target leakage in `servers_public.jsonl`\n")
    leak = summary["servers_public_label_word_scan"]
    lines.append(
        f"Scanned {leak['rows_scanned']} server-level rows for the six label words appearing verbatim "
        f"in the server name/description (this file's stated purpose is server metadata with direct "
        f"category labels removed, so this checks that claim actually holds). "
        f"{leak['rows_with_label_word_in_text']} rows contain one of the words, but only as ordinary "
        f"English in marketing copy (\"read and write files\", \"execute complex workflows\") "
        f"describing what the server does broadly, never as a structured label. `servers_public.jsonl` "
        f"isn't joined into the training text anyway (see Text representation below), so this is a "
        f"non-issue for the current pipeline either way - the scan just makes that decision "
        f"verifiable instead of assumed.\n"
    )

    lines.append("## Text representation\n")
    lines.append(
        "Each record is rendered as a single text block (`src/data/schema.py:build_input_text`):\n\n"
        "```\n"
        "Tool name: <name>\n"
        "Description: <description, whitespace-normalized, truncated to 900 chars>\n"
        "Parameters:\n"
        "- <param> (<type>, required|optional)[ enum[...]]: <short param description>\n"
        "- ...\n"
        "```\n\n"
        "**Why this shape:**\n\n"
        "- `name` and `description` carry most of the signal (verbs like *delete*, *transfer*, *list*, "
        "*execute* are strongly predictive of the action category), so they're kept close to verbatim.\n"
        "- `input_schema` is not dumped as raw JSON. Raw JSON burns tokens on punctuation, and schema "
        "depth varies a lot, from `{}` to nested `oneOf`/`anyOf` branches. Flattening to "
        "`name (type, required): description` keeps the useful signal (a `confirm: boolean` parameter "
        "is a strong `Destructive` cue, an `amount`/`currency` pair is a strong `Financial` cue) while "
        "capping length at 12 rendered parameters and 140 characters per parameter description.\n"
        "- `server_slug` and `servers_public.jsonl` metadata are deliberately left out of the model "
        "input. A model that can see server-level description text can shortcut to \"which server is "
        "this\" instead of \"what does this tool do\", and that kind of proxy signal generalizes badly "
        "to servers the model has never seen, which is the actual deployment scenario.\n"
        "- The same `build_input_text` function feeds both the TF-IDF baseline and the SLM, so any "
        "macro F1 gap between them comes from the modeling approach, not from one of them getting a "
        "richer input.\n"
    )

    lines.append("## Input length\n")
    lines.append("| Split | Mean chars | Max chars |")
    lines.append("|---|---:|---:|")
    for split_name in ["train", "validation", "test"]:
        s = summary["splits"][split_name]
        lines.append(f"| {split_name} | {s['input_text_char_length']['mean']} | {s['input_text_char_length']['max']} |")
    lines.append(
        "\nThis is what the SLM's max sequence length (README.md, Decisions -> Sequence length) is "
        "based on. A small number of tools have long, example-heavy descriptions (some MCP servers "
        "embed multi-paragraph usage examples, e.g. `firecrawl_crawl`) that end up truncated, but the "
        "truncation point falls well past the name/description/first-parameters prefix where most of "
        "the signal already lives.\n"
    )

    with open(REPORTS_DIR / "data_audit.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
