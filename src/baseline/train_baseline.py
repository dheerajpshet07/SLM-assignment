"""TF-IDF + Logistic Regression baseline.

Required by the task spec as a non-neural reference point: if the SLM can't
beat this, the added complexity (GPU fine-tuning, LoRA adapters, generative
decoding) isn't earning its keep.

Run: python -m src.baseline.train_baseline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.schema import build_input_text, load_records
from src.eval.metrics import compute_metrics, dump_metrics_json, save_confusion_matrix_png

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "baseline"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 20260714


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=4.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_records = load_records(DATA_DIR / "train.jsonl")
    val_records = load_records(DATA_DIR / "validation.jsonl")

    x_train = [build_input_text(r) for r in train_records]
    y_train = [r.category for r in train_records]
    x_val = [build_input_text(r) for r in val_records]
    y_val = [r.category for r in val_records]

    pipeline = build_pipeline()

    train_start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - train_start

    # Latency: single-record predict, matching how predict.py will be used
    # at inference time (no batching assumptions baked into the number).
    warmup = x_val[: min(20, len(x_val))]
    for text in warmup:
        pipeline.predict([text])
    latency_sample = x_val[: min(500, len(x_val))]
    lat_start = time.perf_counter()
    for text in latency_sample:
        pipeline.predict([text])
    lat_elapsed = time.perf_counter() - lat_start
    per_record_ms = (lat_elapsed / len(latency_sample)) * 1000

    y_pred = pipeline.predict(x_val)
    metrics = compute_metrics(y_val, y_pred)

    model_path = MODEL_DIR / "tfidf_logreg.joblib"
    joblib.dump(pipeline, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    metrics["model"] = {
        "type": "TF-IDF (word 1-2gram) + Logistic Regression (class_weight=balanced)",
        "vocab_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "train_seconds": round(train_seconds, 2),
        "model_size_mb": round(model_size_mb, 2),
        "single_record_latency_ms": round(per_record_ms, 3),
        "throughput_records_per_sec": round(1000 / per_record_ms, 1),
    }

    dump_metrics_json(metrics, REPORTS_DIR / "baseline_metrics.json")
    save_confusion_matrix_png(
        metrics["confusion_matrix"]["matrix"],
        metrics["confusion_matrix"]["labels"],
        REPORTS_DIR / "figures" / "baseline_confusion_matrix.png",
        title="Baseline (TF-IDF + LogReg) — Validation Confusion Matrix",
    )

    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(json.dumps(metrics["per_class"], indent=2))
    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {REPORTS_DIR / 'baseline_metrics.json'}")

    predictions_path = REPORTS_DIR / "validation_predictions_baseline.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as fh:
        for record, pred, true in zip(val_records, y_pred, y_val):
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
