"""LoRA fine-tuning of a small causal LM as a 6-way tool-category classifier.

Model/task formulation
-----------------------
We fine-tune Qwen2.5-0.5B-Instruct with a sequence-classification head
(`AutoModelForSequenceClassification`, 6-way softmax) rather than treating
this as open-ended text generation. Two reasons:

1. Deterministic, always-valid output. A softmax over exactly 6 logits +
   argmax can only ever produce one of the 6 known labels — there is no
   parsing step, no risk of the model emitting free text, a 7th label, or
   nothing at all. A generative formulation would need constrained decoding
   or output validation/retry logic to get the same guarantee.
2. Cheaper inference. One forward pass per record, no autoregressive
   decoding loop — directly relevant to the "reduce cost and latency"
   question in the take-home brief.

Model choice: Qwen2.5-0.5B-Instruct (494M params) was chosen over larger
Qwen2.5 checkpoints (1.5B/3B, still within the assignment's 3B ceiling)
because this machine's GPU is a shared 6GB laptop card (RTX 4050), not a
dedicated 24GB card. 0.5B leaves comfortable headroom for LoRA optimizer
state, activations, and other GPU consumers (this is a personal laptop, not
a training rig) while still being a genuine instruction-tuned SLM. See
README.md for the accuracy/compute trade-off discussion.

Run: python -m src.slm.train_slm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from src.data.schema import ID_TO_LABEL, LABEL_TO_ID, LABELS, load_records
from src.slm.dataset import MAX_SEQ_LENGTH, make_tokenize_fn, records_to_hf_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "huggingface"

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "slm"
SEED = 20260714

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--epochs", type=int, default=8)
    # Small defaults: this is a 6GB laptop GPU (RTX 4050), not a dedicated
    # 24GB card, and it's shared with the rest of the desktop. Effective
    # batch size (train_batch_size * grad_accum_steps) is what matters for
    # optimization; grad accumulation keeps it at 32 without the memory hit
    # of a large per-step batch.
    p.add_argument("--train-batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--class-weight-power", type=float, default=0.5,
                    help="0 = no reweighting, 1 = full inverse-frequency, 0.5 = sqrt-dampened (default)")
    p.add_argument("--early-stopping-patience", type=int, default=3)
    p.add_argument("--max-length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--max-train-samples", type=int, default=None, help="debug: subsample training set")
    p.add_argument("--max-eval-samples", type=int, default=None, help="debug: subsample validation set")
    p.add_argument("--no-grad-checkpointing", action="store_true",
                    help="disable gradient checkpointing (faster, uses more VRAM)")
    return p.parse_args()


def compute_class_weights(train_records, power: float) -> torch.Tensor:
    """Inverse-frequency class weights, dampened by `power` to avoid letting
    the rarest classes (Financial: 111 rows, Other: 76 rows out of 22k)
    dominate the gradient. power=1.0 reproduces sklearn's
    ``class_weight="balanced"`` (used by the baseline); power=0.5 pulls
    the most extreme weights toward 1.0 while still upweighting the tail.
    We default to 0.5 because pilot runs with power=1.0 showed noisy,
    high-variance loss on `Other` batches without a matching Recall gain.
    """
    counts = np.array([sum(1 for r in train_records if r.category == label) for label in LABELS])
    n_samples = len(train_records)
    n_classes = len(LABELS)
    # sklearn's "balanced" formula (n_samples / (n_classes * count)); computed
    # by hand, with a floor of 1, so this also tolerates debug subsamples
    # that don't cover every class (sklearn's compute_class_weight raises in
    # that case).
    balanced = n_samples / (n_classes * np.maximum(counts, 1))
    dampened = balanced ** power
    return torch.tensor(dampened, dtype=torch.float32)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device, dtype=logits.dtype)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
        return (loss, outputs) if return_outputs else loss


def build_compute_metrics():
    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        labels = eval_pred.label_ids
        return {
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "accuracy": (preds == labels).mean(),
        }

    return compute_metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(SEED)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    print(f"Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=CACHE_DIR)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        # The classification head is randomly initialized (not part of the
        # pretrained checkpoint), so it must be fully trained rather than
        # LoRA-adapted on top of random weights.
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Required for gradient checkpointing to work through frozen base-model
    # layers with only LoRA adapters trainable (otherwise no tensor in the
    # graph requires grad at the checkpoint boundary and backprop breaks).
    model.enable_input_require_grads()

    train_records = load_records(DATA_DIR / "train.jsonl")
    val_records = load_records(DATA_DIR / "validation.jsonl")
    if args.max_train_samples:
        train_records = train_records[: args.max_train_samples]
    if args.max_eval_samples:
        val_records = val_records[: args.max_eval_samples]

    train_ds = records_to_hf_dataset(train_records, with_labels=True)
    val_ds = records_to_hf_dataset(val_records, with_labels=True)

    tokenize_fn = make_tokenize_fn(tokenizer, max_length=args.max_length)
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text", "record_id"])
    val_ds_tokenized = val_ds.map(tokenize_fn, batched=True, remove_columns=["text", "record_id"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    class_weights = compute_class_weights(train_records, power=args.class_weight_power)
    print("Class weights:", dict(zip(LABELS, class_weights.tolist())))

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=25,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=torch.cuda.is_available() and not args.no_grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds_tokenized,
        data_collator=collator,
        compute_metrics=build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
        class_weights=class_weights,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_start = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - train_start

    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
    )

    # trainer.model is now the best checkpoint (load_best_model_at_end=True).
    final_adapter_dir = output_dir / "adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    training_summary = {
        "model_id": args.model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": LORA_TARGET_MODULES,
        "class_weight_power": args.class_weight_power,
        "class_weights": dict(zip(LABELS, class_weights.tolist())),
        "max_seq_length": args.max_length,
        "epochs_configured": args.epochs,
        "epochs_completed": trainer.state.epoch,
        "best_checkpoint_step": trainer.state.best_global_step,
        "best_metric_macro_f1": trainer.state.best_metric,
        "early_stopping_patience": args.early_stopping_patience,
        "effective_batch_size": args.train_batch_size * args.grad_accum_steps,
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_memory_mb": round(peak_mem_mb, 1) if peak_mem_mb else None,
        "seed": SEED,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(training_summary, fh, indent=2)

    print(json.dumps(training_summary, indent=2))
    print(f"\nSaved best adapter to {final_adapter_dir}")


if __name__ == "__main__":
    main()
