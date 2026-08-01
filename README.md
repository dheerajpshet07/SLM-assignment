# AI Engineer Take-Home Assessment: Small Language Model Classification

## Objective

Build and evaluate a small language model that classifies an MCP tool into exactly one action category:

- `Read`
- `Write`
- `Execute`
- `Destructive`
- `Financial`
- `Other`

The classifier should infer the label from the tool's natural-language and schema fields. This exercise tests practical model development, data judgment, evaluation rigor, and production thinking—not only the final metric.

## Timebox and compute budget

- Recommended effort: **6 hours**; hard cap: **8 hours**.
- Use an open-weight language model with **3 billion parameters or fewer**.
- The complete experiment must be feasible on a single commodity GPU with **24 GB VRAM or less**. Parameter-efficient fine-tuning such as LoRA or QLoRA is encouraged.
- Training a model from scratch is not expected.
- External labeled datasets are not allowed. Public pretrained model weights and standard open-source libraries are allowed.

## Provided files

- `data/train.jsonl`: labeled training records.
- `data/validation.jsonl`: labeled validation records.
- `data/test_unlabeled.jsonl`: held-out test records without labels.
- `data/servers_public.jsonl`: optional server-level metadata, with direct category labels removed.
- `data/split_summary.json`: split sizes and label counts.

Each tool record contains:

```json
{
  "record_id": "stable identifier",
  "server_slug": "server grouping key",
  "name": "tool/function name",
  "description": "natural-language tool description",
  "input_schema": "JSON schema serialized as text",
  "category": "present only in train and validation"
}
```

The train, validation, and test sets are isolated by `server_slug`. Do not merge the sets or create a new random row-level split.

## Required work

### 1. Data audit

Document:

- label distribution and imbalance;
- missing or malformed fields;
- duplicate or near-duplicate risks;
- potential target leakage;
- the text representation you chose and why.

### 2. Baseline

Implement at least one non-neural or non-generative baseline, such as TF-IDF plus logistic regression or a linear SVM. Report the same metrics used for the SLM.

### 3. SLM training

Fine-tune or adapt an SLM to predict one of the six labels. You may formulate this as sequence classification or constrained text generation.

Your solution must explain:

- model and tokenizer selection;
- prompt/input format;
- maximum sequence length and truncation policy;
- class-imbalance strategy;
- training configuration;
- checkpoint selection and stopping rule;
- steps taken to make output labels valid and deterministic.

### 4. Evaluation

Use **macro F1** as the primary metric. Also report:

- per-class precision, recall, and F1;
- weighted F1;
- accuracy;
- confusion matrix;
- invalid-output rate, if using generative classification;
- model size, peak memory if available, and inference latency or throughput.

Do not tune on the held-out test set. The employer will score `predictions.csv` against private labels.

### 5. Error analysis

Review at least 20 validation errors. Identify at least three recurring failure modes and propose concrete improvements. Include examples, especially for minority classes and ambiguous `Write` versus `Execute` behavior.

### 6. Inference interface

Provide a command that produces predictions for a JSONL file:

```bash
python predict.py \
  --model-path <path-or-model-id> \
  --input data/test_unlabeled.jsonl \
  --output predictions.csv
```

The output must contain exactly:

```csv
record_id,category
```

Every input record must receive exactly one valid category.

## Deliverables

Submit a repository or archive containing:

1. `README.md` with setup, commands, decisions, and results.
2. Reproducible data-preparation code.
3. Baseline training/evaluation code.
4. SLM training code or notebook.
5. `predict.py` or an equivalent inference entry point.
6. `predictions.csv` for the supplied held-out test set.
7. `metrics.json` and a confusion-matrix image or table.
8. A short model card covering intended use, limitations, and known failure modes.

Do not include large model weights in the submission. Provide a model identifier, adapter artifact, or reproducible checkpoint instructions.

## Evaluation priorities

We value correct experimental design, leakage prevention, reproducibility, thoughtful error analysis, and deployable inference. A clear, honest solution with well-explained tradeoffs is stronger than an opaque solution reporting one high score.

## Follow-up interview

Be prepared for a 45-minute discussion covering:

- why the model improved or failed to improve over the baseline;
- how you would handle new categories or multi-label tools;
- calibration and abstention for high-risk predictions;
- production monitoring and drift;
- how you would reduce cost and latency without materially hurting macro F1.

---

# Solution

This section documents the actual implementation: setup, commands,
decisions, and results. See `PROJECT_UPDATE_LOG.md` for the full
development journal (per-feature rationale, technical/layman explanations,
interview Q&A) and `MODEL_CARD.md` for the model card.

## Setup

Tested on Windows with Python 3.11. This machine's GPU is a laptop RTX 4050
with **6GB VRAM** (not a dedicated 24GB card) — the model and batch-size
choices below were made for that constraint, comfortably inside the
assignment's ≤24GB / ≤3B-parameter ceiling.

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows; `source .venv/bin/activate` on Linux/Mac

# Torch is hardware-specific — install the CUDA build matching your driver
# (check with `nvidia-smi`; cu124 works for CUDA 12.4+ drivers). CPU-only
# works too, just slower (see Inference latency below).
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

## Commands

```bash
# 1. Data audit -> reports/data_audit.md, reports/data_audit.json
python -m src.data.data_audit

# 2. Baseline: TF-IDF + Logistic Regression
python -m src.baseline.train_baseline
# -> models/baseline/tfidf_logreg.joblib, reports/baseline_metrics.json,
#    reports/figures/baseline_confusion_matrix.png

# 3. SLM: LoRA fine-tune of Qwen2.5-0.5B-Instruct
python -m src.slm.train_slm
# -> models/slm/adapter/ (LoRA weights), models/slm/training_summary.json

# 4. SLM evaluation on the validation split
python -m src.slm.evaluate_slm
# -> reports/slm_metrics.json, reports/figures/slm_confusion_matrix.png,
#    reports/validation_predictions_slm.jsonl (per-record, for error analysis)

# 5. Error analysis (surfaces confusion examples for manual review)
python -m src.eval.error_analysis --predictions reports/validation_predictions_slm.jsonl

# 6. Inference on any JSONL file (baseline or SLM, auto-detected)
python predict.py --model-path models/slm/adapter \
                   --input data/test_unlabeled.jsonl \
                   --output predictions.csv
```

## Decisions

### Text representation
Every record is rendered by `src/data/schema.py:build_input_text()` as:

```
Tool name: <name>
Description: <description, whitespace-normalized, truncated to 900 chars>
Parameters:
- <param> (<type>, required|optional)[ enum[...]]: <short param description>
```

`input_schema` is flattened rather than dumped as raw JSON (bounds length,
keeps the signal — a `confirm: boolean` param is a strong `Destructive`
cue, an `amount`/`currency` pair a strong `Financial` cue — without wasting
tokens on JSON punctuation). `server_slug` and `servers_public.jsonl` are
deliberately excluded from the model input so the model can't shortcut to
"which server is this" instead of "what does this tool do" — see
`reports/data_audit.md` → Text representation for the full rationale. The
same function feeds both the baseline and the SLM, so the macro-F1 gap
between them reflects modeling approach, not differing inputs.

### Model / tokenizer selection
`Qwen/Qwen2.5-0.5B-Instruct` (494M params). Considered up to the
assignment's 3B ceiling, but this machine's actual GPU is a shared 6GB
laptop card; 0.5B leaves comfortable headroom for LoRA optimizer state and
activations without needing 4-bit quantization (which brings in
`bitsandbytes`, historically flaky on Windows). See
`src/slm/train_slm.py` module docstring for the full trade-off.

### Prompt / input format
Same `build_input_text()` output as the baseline, tokenized directly — no
chat template or instruction wrapper, because this is formulated as
sequence classification (see below), not a generation/chat task.

### Sequence-classification, not generation
`AutoModelForSequenceClassification` with a 6-way softmax head, not
constrained/generative decoding. Two reasons: (1) a fixed 6-logit softmax +
argmax can only ever emit one of the six known labels — no parsing, no
invalid-output risk, no retry logic needed; (2) one forward pass per
record vs. an autoregressive decode loop is directly cheaper at inference
time. The classification head (`score`, randomly initialized — Qwen2.5 is
not shipped with a classification head) is fully fine-tuned
(`modules_to_save=["score"]` in the LoRA config) rather than LoRA-adapted,
since LoRA decomposes a *delta* on top of pretrained weights and there are
no pretrained weights for this head to adapt.

### Max sequence length and truncation
320 tokens, right-truncated. Chosen from the actual token-length
distribution of `build_input_text()` output under the Qwen tokenizer
(computed over train+validation): p95 = 280 tokens, p99 = 425 — 320 covers
~97% of records exactly and truncates only the long tail (some MCP tools
embed multi-paragraph usage examples in their description), where the
truncated content is past the informative prefix (name, description start,
first parameters).

### Class-imbalance strategy
Sqrt-dampened inverse-frequency class weights in the cross-entropy loss
(`WeightedLossTrainer` in `src/slm/train_slm.py`): `weight_c =
(n_samples / (n_classes * count_c)) ** 0.5`. Plain inverse-frequency
(power=1.0, what the baseline's `class_weight="balanced"` uses) gives
`Other` (76 training rows) a weight of ~48x `Read`'s; pilot runs at that
setting showed high-variance loss spikes on `Other`/`Financial` batches
without a matching recall gain, so the exponent is dampened to 0.5,
pulling extreme weights toward 1 while still upweighting the tail
(`--class-weight-power` is a CLI flag if this needs revisiting).

### Training configuration
- LoRA: r=16, α=32, dropout=0.05, targeting
  `q/k/v/o_proj` + `gate/up/down_proj`; classification head fully trained.
- Effective batch size 32 (`train-batch-size=4` × `grad-accum-steps=8`,
  gradient checkpointing on — required to fit training in 6GB VRAM; see
  `PROJECT_UPDATE_LOG.md` for the memory-calibration process).
- LR 2e-4, cosine schedule, 6% warmup, weight decay 0.01, bf16 (native on
  this GPU's Ada architecture).
- Up to 8 epochs, early stopping (patience configurable, `metric_for_best_model="macro_f1"`).

### Checkpoint selection and stopping rule
`load_best_model_at_end=True` with `metric_for_best_model="macro_f1"`
(evaluated on `data/validation.jsonl` after every epoch) — the saved
adapter is the epoch with the best validation macro F1, not the last
epoch trained, and `EarlyStoppingCallback` stops training once macro F1
hasn't improved for the configured patience.

### Making output labels valid and deterministic
By construction, not by post-processing: the model head has exactly 6
output logits corresponding 1:1 to the 6 labels; `predict.py` takes
`argmax` and maps through a fixed `id2label` table. There is no free-text
output to parse or validate, so invalid-output rate is 0% by design (see
`reports/slm_metrics.json`) — this was the primary reason to prefer the
classification-head formulation over generative decoding.

## Results

_Full metrics: `reports/baseline_metrics.json`, `reports/slm_metrics.json`.
Confusion matrices: `reports/figures/*_confusion_matrix.png`._

| Model | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 0.688 | 0.884 | 88.4% |
| SLM (Qwen2.5-0.5B + LoRA) | *training in progress* | – | – |

The SLM's per-epoch validation macro F1 is already tracking above the
baseline early in training (epoch 1: 0.709, epoch 2: 0.782, still
improving) — see `models/slm/training_summary.json` once training
completes for the final checkpoint's numbers and `reports/slm_metrics.json`
for the full per-class breakdown, confusion matrix, and latency/throughput.
This section will be updated with the final comparison table once the
training run (early-stopped on validation macro F1) finishes.

## Error analysis

_In progress — pending final SLM validation predictions._
`src/eval/error_analysis.py` groups validation misclassifications by
(true, predicted) label pair so the write-up in `reports/error_analysis.md`
covers systematic failure modes with concrete examples, not a random
sample. Will be completed once the SLM finishes training; see
`reports/validation_predictions_baseline.jsonl` in the interim for the
baseline's error patterns.

