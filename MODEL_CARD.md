# Model Card — MCP Tool Category Classifier

## Overview

| | |
|---|---|
| Task | 6-way single-label classification of an MCP tool's action category |
| Labels | `Read`, `Write`, `Execute`, `Destructive`, `Financial`, `Other` |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (494M params) |
| Adaptation | LoRA (r=16, alpha=32, dropout=0.05) on attention + MLP projections; classification head fully fine-tuned |
| Formulation | Sequence classification (softmax over 6 fixed logits), not generation |
| Input | Tool name, description, and a flattened `input_schema` parameter list (see README.md → Text representation) |
| Max sequence length | 320 tokens, right-truncated |
| Training hardware | Single RTX 4050 laptop GPU, 6GB VRAM |

## Intended use

Classifying MCP (Model Context Protocol) tool definitions by what they
actually do when invoked: read-only, mutates state, runs arbitrary
commands, destructive, moves money, or none of the above. The main use
case is feeding an approval/guardrail policy in an agent framework, e.g.
requiring extra confirmation before a `Destructive` or `Financial` tool
gets called automatically. Shipped as batch classification over a JSONL
file (`predict.py`), not a real-time service.

## Out of scope

- **Not a safety boundary on its own.** A `Destructive` tool mislabeled as
  `Write` is exactly the kind of miss this model exists to catch, so it
  shouldn't be the only thing standing between an agent and a risky action.
  Pair it with human review or a stricter rule-based allowlist for anything
  that actually gates access.
- **Not validated outside MCP tool schemas.** Trained and evaluated only on
  the name/description/JSON-Schema shape this dataset uses. Unverified on
  OpenAPI specs, bare function signatures, or anything without a
  description field.
- **Not multi-label.** A tool that both reads and writes gets forced into
  one label. See the assignment's follow-up-interview notes at the top of
  this repo's README for how a multi-label version would work.

## Training data

`data/train.jsonl`: 22,143 rows across 1,448 MCP servers, split from
validation/test by `server_slug` (grouped split provided by the employer,
zero server overlap confirmed in `reports/data_audit.md`). Heavily
imbalanced: `Read` is about 64% of training rows, `Financial` and `Other`
combined are under 1%. Handled with sqrt-dampened inverse-frequency class
weighting in the loss; see `src/slm/train_slm.py`.

## Performance (validation split, 4,429 rows)

Full breakdown in `reports/slm_metrics.json` and
`reports/figures/slm_confusion_matrix.png`. Best checkpoint was epoch 4
(macro F1 peaked there; epoch 5 dropped slightly with rising eval loss, an
overfitting signal, so `load_best_model_at_end` kept epoch 4):

| Metric | Value |
|---|---|
| Macro F1 | 0.830 |
| Weighted F1 | 0.950 |
| Accuracy | 95.1% |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Read | 0.962 | 0.977 | 0.969 | 2817 |
| Write | 0.930 | 0.925 | 0.927 | 1087 |
| Execute | 0.931 | 0.809 | 0.866 | 235 |
| Destructive | 0.968 | 0.953 | 0.960 | 253 |
| Financial | 0.778 | 0.955 | 0.857 | 22 |
| Other | 0.800 | 0.267 | 0.400 | 15 |

Baseline (TF-IDF + Logistic Regression) for reference: macro F1 0.688,
accuracy 88.4%, in `reports/baseline_metrics.json`.

Deployment footprint: 44.5MB LoRA adapter, ~28 records/sec batched
throughput on an RTX 4050, 0% invalid-output rate by construction. Full
detail in `reports/slm_metrics.json`.

## Known failure modes

Full write-up with examples in `reports/error_analysis.md`. Summary: 215
validation errors (4.9%, down from the baseline's 514/11.6%). Three
recurring patterns:

1. **Umbrella tools with no verb to key on**, e.g. an Azure tool described
   only as "Work with Azure SQL Database servers." These genuinely mix
   read and write behavior depending on runtime arguments the static
   description never exposes. Not fixable by more training; it's a
   data/taxonomy limitation.
2. **Description wording pointing at the wrong category.** A `dbt`
   `compile` tool explicitly says "without running" and still gets
   predicted `Execute`; a tool described as "Writes .brand-preview.html"
   is labeled `Execute` but predicted `Read`. The model leans on
   individual verbs more than full sentence context.
3. **`Other` has no coherent internal pattern.** Recall is 0.267 on just
   15 validation examples spanning unrelated tools (image analysis, an
   astrology calculator, an AWS Lambda invoker). With 76 training examples
   covering that much variety, there's little shared signal to learn from.

## Limitations

- **Leans on the description.** The model only sees name, description, and
  parameter schema. 316 training rows and 42 test rows have an empty
  description, and those get judged on name and parameters alone, which is
  noticeably less reliable.
- **Rare classes are still rare.** Class weighting helps recall on
  `Financial`/`Other`, but with only 111 and 76 training examples
  respectively, the model just hasn't seen much variety in what those
  categories look like. Treat predictions in those two classes with more
  skepticism than a `Read`/`Write` prediction.
- **English-centric.** Training descriptions are overwhelmingly English;
  untested on anything else.
- **No calibration or abstention.** The model always outputs one of six
  labels, even on a genuinely ambiguous input. There's no "not sure."
  Softmax probabilities exist but haven't been calibrated (e.g. temperature
  scaling) or checked for reliability as confidence estimates.
- **Static snapshot.** Trained on a mid-2026 crawl of public MCP servers.
  As tool-naming conventions in the ecosystem shift, accuracy will drift,
  and nothing here retrains automatically.

## Reproduction

```bash
python -m src.slm.train_slm      # trains and saves models/slm/adapter
python -m src.slm.evaluate_slm   # scores models/slm/adapter on validation
python predict.py --model-path models/slm/adapter \
                   --input data/test_unlabeled.jsonl \
                   --output predictions.csv
```

Full model weights aren't included in this submission. The adapter
directory (`models/slm/adapter/`, LoRA weights only, a few tens of MB) is
included; the base model (`Qwen/Qwen2.5-0.5B-Instruct`) downloads from the
Hugging Face Hub on first run.
