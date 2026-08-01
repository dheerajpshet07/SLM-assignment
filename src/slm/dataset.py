"""Tokenization helpers shared by SLM training and inference."""
from __future__ import annotations

from typing import Any

from datasets import Dataset

from src.data.schema import LABEL_TO_ID, ToolRecord, build_input_text

MAX_SEQ_LENGTH = 320


def records_to_hf_dataset(records: list[ToolRecord], with_labels: bool) -> Dataset:
    texts = [build_input_text(r) for r in records]
    data: dict[str, Any] = {
        "record_id": [r.record_id for r in records],
        "text": texts,
    }
    if with_labels:
        data["labels"] = [LABEL_TO_ID[r.category] for r in records]
    return Dataset.from_dict(data)


def make_tokenize_fn(tokenizer, max_length: int = MAX_SEQ_LENGTH):
    def tokenize(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return tokenize
