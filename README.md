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

This section covers setup, commands, the reasoning behind the main design
choices, and results. Model card is in `MODEL_CARD.md`.

## Setup

Built and tested on Windows / Python 3.11. GPU is a laptop RTX 4050 with
6GB VRAM, not the 24GB the assignment budgets for, so batch size and model
size below were picked for that reality rather than the assignment ceiling.

```bash
python -m venv .venv
.venv/Scripts/activate   # Windows; `source .venv/bin/activate` on Linux/Mac

# torch is hardware-specific, so it's not in requirements.txt. Check your
# driver with nvidia-smi and pick a matching CUDA build, e.g.:
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
`src/data/schema.py:build_input_text()` turns every record into one text
block:

```
Tool name: <name>
Description: <description, whitespace-normalized, truncated to 900 chars>
Parameters:
- <param> (<type>, required|optional)[ enum[...]]: <short param description>
```

I flattened `input_schema` into that parameter list instead of dumping raw
JSON. Raw JSON burns tokens on braces and quotes, and a flattened
`confirm: boolean` or `amount`/`currency` pair is a much more direct signal
for `Destructive`/`Financial` than the same thing buried in nested JSON.
`server_slug` and `servers_public.jsonl` are left out on purpose too — a
model that can see which server a tool came from can shortcut to "which
server is this" instead of actually reading the tool description, and that
shortcut won't hold up on a server it's never seen. Full writeup in
`reports/data_audit.md`.

Both the baseline and the SLM train on the exact same `build_input_text()`
output, so whatever F1 gap shows up between them is about the model, not
about one of them getting a better prompt.

### Model choice
`Qwen/Qwen2.5-0.5B-Instruct`, 494M params. The assignment allows up to 3B,
but I sized down for the actual GPU in this laptop (6GB, shared with
everything else running on the machine) rather than the assignment's
ceiling. 0.5B leaves enough headroom for LoRA training without reaching
for 4-bit quantization, which would pull in `bitsandbytes` — not the most
reliable thing to depend on for a Windows setup.

### Classification head, not generation
The model predicts through a 6-way softmax head
(`AutoModelForSequenceClassification`) rather than generating the label as
text. Two reasons. First, output validity: argmax over 6 fixed logits can
only ever land on one of the six real labels, so there's no parsing step
and nothing to go wrong there. Second, it's just cheaper at inference, one
forward pass instead of a token-by-token decode. The head itself (`score`)
is randomly initialized, since Qwen2.5 doesn't ship with a classifier, so
it's fully trained rather than LoRA-adapted (`modules_to_save=["score"]`).
There's no pretrained weight there for a LoRA delta to sit on top of.

### Sequence length
320 tokens, truncated from the right. I checked the actual token-length
distribution under the Qwen tokenizer before picking a number: p95 is 280
tokens, p99 is 425, so 320 covers roughly 97% of records outright. What
gets cut is mostly a handful of tools with long, example-heavy
descriptions, and the cut happens after the name/description/first few
parameters, which is where most of the signal already is.

### Class imbalance
The loss uses sqrt-dampened inverse-frequency class weights
(`WeightedLossTrainer` in `src/slm/train_slm.py`):
`weight_c = (n_samples / (n_classes * count_c)) ** 0.5`. Full inverse
frequency (what the baseline's `class_weight="balanced"` uses) puts `Other`
at roughly 48x the weight of `Read`, and in early testing that just made
training noisy on `Other`/`Financial` batches without actually improving
recall on them. Dampening the exponent to 0.5 keeps the same ordering but
pulls the extremes back toward 1. `--class-weight-power` is exposed as a
flag if this needs revisiting.

### Training setup
LoRA at r=16, alpha=32, dropout=0.05, on all attention and MLP projections,
plus the fully-trained classification head. Batch size and gradient
checkpointing came out of actually testing what fits in 6GB: a few short
runs at increasing batch sizes while watching peak memory usage, landing on
`--train-batch-size 32 --eval-batch-size 64 --grad-accum-steps 1` with
checkpointing on. LR 2e-4 with cosine decay and 6% warmup, weight decay
0.01, bf16 (the 4050's Ada architecture supports it natively). Up to 6
epochs with early stopping, `metric_for_best_model="macro_f1"`.

### Checkpoint selection
`load_best_model_at_end=True`, gated on validation macro F1 checked after
every epoch. Whatever gets saved is the best epoch, not just the last one,
and training stops once macro F1 hasn't improved for a couple of epochs in
a row.

### Valid, deterministic output
This falls out of using a classification head rather than something bolted
on after the fact. Six logits, one argmax, one label. There's no free-text
output to validate, so the invalid-output rate is 0% by construction
(`reports/slm_metrics.json` reports it anyway, for the record).

## Results

Full metrics in `reports/baseline_metrics.json` and `reports/slm_metrics.json`
(also consolidated in `metrics.json` at the repo root), confusion matrices
in `reports/figures/`. Validation split, 4,429 rows.

| Model | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| Baseline (TF-IDF + LogReg) | 0.688 | 0.884 | 88.4% |
| SLM (Qwen2.5-0.5B + LoRA) | **0.830** | 0.950 | 95.1% |

The SLM clears the baseline by 0.14 macro F1, a bigger gap than the
per-epoch trend suggested early on (epoch 1: 0.709, epoch 2: 0.782, epoch
3: 0.805, epoch 4: 0.830, the best checkpoint, per
`models/slm/training_summary.json`). Epoch 5 came in slightly lower (0.829)
with eval loss rising noticeably, an early overfitting signal, so
`load_best_model_at_end` correctly kept epoch 4 rather than a later one.

Per-class F1: Read 0.969, Write 0.927, Destructive 0.960, Execute 0.866,
Financial 0.857, Other 0.400. The two classes that matter most for a
tool-gating use case, `Destructive` and `Financial`, are both strong.
`Other` is the weak point: only 15 validation examples and no coherent
internal pattern (see error analysis below).

Deployment numbers: 44.5MB adapter, 0% invalid-output rate (by
construction), ~28 records/sec batched on this GPU, full detail in
`reports/slm_metrics.json`.

## Error analysis

Full write-up in `reports/error_analysis.md`. Short version: the SLM drops
errors from 514 (11.6%, baseline) to 215 (4.9%). Over half of what's left
is Read/Write confusion, concentrated in two patterns worth knowing about:

1. **Umbrella tools with no verb to key on.** Things like an Azure tool
   described only as "Work with Azure SQL Database servers," which
   genuinely could be either read or write depending on runtime arguments
   the static description doesn't expose. This is a data/taxonomy
   limitation, not something more training fixes.
2. **The description's own wording points at the wrong category.** A
   `dbt` `compile` tool explicitly says "without running" and still gets
   predicted `Execute`; a tool literally described as "Writes
   .brand-preview.html" is labeled `Execute` and predicted `Read`. The
   model leans on individual verbs ("update," "execute," "compile") more
   than the sentence they sit in.

`Other` is the standout weak class (recall 0.27), and its errors span
completely unrelated tools: an image-analysis tool, a Chinese astrology
calculator, an AWS Lambda invoker. There's no shared pattern for a model
to learn with only 76 training examples covering that much variety.

