#!/usr/bin/env python
"""Inference entry point for the MCP tool-category classifier.

Produces a `record_id,category` CSV for a JSONL file of tool records, using
either the fine-tuned SLM (a LoRA adapter directory) or the TF-IDF baseline
(a joblib pipeline) — the model type is auto-detected from --model-path.

Usage:
    python predict.py \
        --model-path models/slm/adapter \
        --input data/test_unlabeled.jsonl \
        --output predictions.csv

    python predict.py \
        --model-path models/baseline/tfidf_logreg.joblib \
        --input data/test_unlabeled.jsonl \
        --output predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from src.data.schema import ID_TO_LABEL, LABELS, build_input_text, load_records

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache" / "huggingface"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True, help="LoRA adapter directory, joblib file, or HF model id")
    p.add_argument("--input", required=True, help="Input JSONL file (train/validation/test schema)")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect)")
    return p.parse_args()


def predict_baseline(model_path: str, texts: list[str]) -> list[str]:
    import joblib

    pipeline = joblib.load(model_path)
    return list(pipeline.predict(texts))


def predict_slm(model_path: str, texts: list[str], batch_size: int, max_length: int, device: str | None) -> list[str]:
    import torch
    from peft import AutoPeftModelForSequenceClassification
    from transformers import AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR)
    # num_labels isn't stored in adapter_config.json, so it must be passed
    # explicitly or the base model loads with the transformers default
    # (num_labels=2) and the saved 6-way classification head fails to load.
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        model_path, cache_dir=CACHE_DIR, num_labels=len(LABELS)
    )
    # Set at training time on the in-memory model only, so it isn't part of
    # the saved checkpoint's base config and has to be set again here.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(resolved_device)
    model.eval()

    predictions: list[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(resolved_device)
            logits = model(**encoded).logits
            batch_preds = logits.argmax(dim=-1).tolist()
            predictions.extend(ID_TO_LABEL[i] for i in batch_preds)

    return predictions


def resolve_model_kind(model_path: str) -> str:
    path = Path(model_path)
    if path.is_dir() and (path / "adapter_config.json").exists():
        return "slm"
    if path.is_file() and path.suffix in (".joblib", ".pkl"):
        return "baseline"
    if path.is_dir() and (path / "config.json").exists():
        return "slm"  # merged / non-adapter transformers checkpoint
    raise ValueError(
        f"Could not determine model type for --model-path={model_path!r}. "
        "Expected a LoRA adapter directory (containing adapter_config.json), "
        "a transformers checkpoint directory (containing config.json), or a "
        "baseline .joblib file."
    )


def main() -> None:
    args = parse_args()

    records = load_records(args.input)
    if not records:
        print(f"No records found in {args.input}", file=sys.stderr)
        sys.exit(1)

    texts = [build_input_text(r) for r in records]

    kind = resolve_model_kind(args.model_path)
    print(f"Model kind: {kind} ({args.model_path})")

    start = time.perf_counter()
    if kind == "baseline":
        predictions = predict_baseline(args.model_path, texts)
    else:
        predictions = predict_slm(args.model_path, texts, args.batch_size, args.max_length, args.device)
    elapsed = time.perf_counter() - start

    invalid = [p for p in predictions if p not in LABELS]
    if invalid:
        # Should be unreachable — both code paths select from a fixed
        # 6-way head/label set — but we fail loudly rather than silently
        # emit a bad row, since the grader expects every row to be valid.
        raise RuntimeError(f"{len(invalid)} predictions fell outside {LABELS}: {invalid[:5]}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["record_id", "category"])
        for record, category in zip(records, predictions):
            writer.writerow([record.record_id, category])

    print(f"Wrote {len(records)} predictions to {output_path}")
    print(f"Elapsed: {elapsed:.2f}s ({len(records) / elapsed:.1f} records/sec)")


if __name__ == "__main__":
    main()
