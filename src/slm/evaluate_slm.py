"""Evaluate the fine-tuned SLM adapter on the validation split.

Mirrors src/baseline/train_baseline.py's metrics so the two are directly
comparable in README.md. Also records model size, peak memory, and
latency/throughput, and the (trivially zero) invalid-output rate.

Run: python -m src.slm.evaluate_slm --adapter-path models/slm/adapter
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import AutoPeftModelForSequenceClassification
from transformers import AutoTokenizer

from src.data.schema import ID_TO_LABEL, LABELS, build_input_text, load_records
from src.eval.metrics import compute_metrics, dump_metrics_json, save_confusion_matrix_png

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "huggingface"
REPORTS_DIR = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter-path", default=str(ROOT / "models" / "slm" / "adapter"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=320)
    return p.parse_args()


def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), cache_dir=CACHE_DIR)
    # num_labels isn't stored in adapter_config.json, so it must be passed
    # explicitly here or the base model loads with the transformers default
    # (num_labels=2) and the saved 6-way classification head fails to load.
    model = AutoPeftModelForSequenceClassification.from_pretrained(
        str(adapter_path), cache_dir=CACHE_DIR, num_labels=len(LABELS)
    )
    # Set at training time on the in-memory model only, so it isn't part of
    # the saved checkpoint's base config and has to be set again here.
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    val_records = load_records(DATA_DIR / "validation.jsonl")
    texts = [build_input_text(r) for r in val_records]
    y_true = [r.category for r in val_records]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Single-record latency, matching how predict.py is used in the
    # deployment scenario described by the task (one JSONL file at a time,
    # no guaranteed large-batch workload).
    warmup_texts = texts[: min(10, len(texts))]
    with torch.no_grad():
        for t in warmup_texts:
            enc = tokenizer([t], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            model(**enc)

    latency_sample = texts[: min(300, len(texts))]
    lat_start = time.perf_counter()
    with torch.no_grad():
        for t in latency_sample:
            enc = tokenizer([t], truncation=True, max_length=args.max_length, padding=True, return_tensors="pt").to(device)
            model(**enc)
    lat_elapsed = time.perf_counter() - lat_start
    per_record_ms = (lat_elapsed / len(latency_sample)) * 1000

    # Batched predictions for the actual metrics (this is how predict.py
    # scores a full file — batched, not one-at-a-time).
    all_preds: list[str] = []
    batch_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            batch_texts = texts[start : start + args.batch_size]
            enc = tokenizer(
                batch_texts, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            preds = logits.argmax(dim=-1).tolist()
            all_preds.extend(ID_TO_LABEL[i] for i in preds)
    batch_elapsed = time.perf_counter() - batch_start

    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None

    invalid_count = sum(1 for p in all_preds if p not in LABELS)

    metrics = compute_metrics(y_true, all_preds)
    metrics["model"] = {
        "type": "Qwen2.5-0.5B-Instruct + LoRA (sequence classification head)",
        "device": device,
        "adapter_size_mb": round(dir_size_mb(adapter_path), 2),
        "single_record_latency_ms": round(per_record_ms, 3),
        "batched_throughput_records_per_sec": round(len(texts) / batch_elapsed, 1),
        "peak_gpu_memory_mb": round(peak_mem_mb, 1) if peak_mem_mb else None,
        "invalid_output_rate": invalid_count / len(all_preds),
    }

    dump_metrics_json(metrics, REPORTS_DIR / "slm_metrics.json")
    save_confusion_matrix_png(
        metrics["confusion_matrix"]["matrix"],
        metrics["confusion_matrix"]["labels"],
        REPORTS_DIR / "figures" / "slm_confusion_matrix.png",
        title="SLM (Qwen2.5-0.5B + LoRA) — Validation Confusion Matrix",
    )

    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(json.dumps(metrics["per_class"], indent=2))
    print(f"\nSaved metrics to {REPORTS_DIR / 'slm_metrics.json'}")

    # Persist per-record predictions for the error-analysis pass.
    predictions_path = REPORTS_DIR / "validation_predictions_slm.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as fh:
        for record, pred, true in zip(val_records, all_preds, y_true):
            fh.write(
                json.dumps(
                    {
                        "record_id": record.record_id,
                        "server_slug": record.server_slug,
                        "name": record.name,
                        "description": record.description,
                        "true_category": true,
                        "predicted_category": pred,
                        "correct": pred == true,
                    }
                )
                + "\n"
            )
    print(f"Saved per-record validation predictions to {predictions_path}")


if __name__ == "__main__":
    main()
