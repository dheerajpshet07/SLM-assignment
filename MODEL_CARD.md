# Model Card — MCP Tool Category Classifier

## Overview

| | |
|---|---|
| Task | 6-way single-label classification of an MCP tool's action category |
| Labels | `Read`, `Write`, `Execute`, `Destructive`, `Financial`, `Other` |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` (494M params) |
| Adaptation | LoRA (r=16, α=32, dropout=0.05) on attention + MLP projections; classification head fully fine-tuned |
| Formulation | Sequence classification (softmax over 6 fixed logits), not generation |
| Input | `Tool name`, `Description`, flattened `input_schema` parameter list (see README.md → Text Representation) |
| Max sequence length | 320 tokens, right-truncated |
| Training hardware | Single RTX 4050 laptop GPU, 6GB VRAM |

## Intended use

- Bulk-labeling or triaging MCP (Model Context Protocol) tool definitions by
  the *effect* they have when invoked (read-only vs. mutates state vs. runs
  arbitrary commands vs. destructive vs. moves money vs. none of the above),
  e.g. to drive an approval/guardrail policy in an agent framework that
  should require extra confirmation before invoking `Destructive` or
  `Financial` tools.
- Batch classification of a JSONL export of tool definitions
  (`predict.py`), not a real-time/streaming service as shipped.

## Out-of-scope / not intended use

- **Not a safety boundary by itself.** A misclassified `Destructive` tool
  labeled `Write` is a false negative on the exact axis this model exists
  to catch. See Limitations — this must be one signal among several
  (e.g. paired with human review or a stricter rule-based allowlist) for
  any actual access-control decision, not the sole gate.
- **Not validated on non-MCP tool schemas.** Trained and evaluated only on
  MCP tool records with the `name` + `description` + JSON Schema
  `input_schema` shape provided in this dataset; behavior on other API
  description formats (OpenAPI specs, raw function signatures without
  descriptions, etc.) is unverified.
- **Not multi-label.** A tool that both reads and writes is forced into one
  label. See README.md → Follow-up discussion for how this could be
  extended.

## Training data

- `data/train.jsonl` (22,143 rows, 1,448 MCP servers), split from
  validation/test by `server_slug` (`StratifiedGroupKFold`, provided by the
  employer, verified zero-overlap in `reports/data_audit.md`).
- Severely imbalanced: `Read` is ~64% of training rows; `Financial` (111
  rows) and `Other` (76 rows) combined are under 1%. Addressed via
  sqrt-dampened inverse-frequency class weighting in the loss (see
  `src/slm/train_slm.py`).

## Performance (validation split, 4,429 rows)

_See `reports/slm_metrics.json` and `reports/figures/slm_confusion_matrix.png`
for the full breakdown once training completes; summary below._

Training is early-stopped on validation macro F1, so the number below is a
lower bound as of the last completed epoch, not the final saved checkpoint:

| Epoch | Macro F1 | Accuracy |
|---|---|---|
| 1 | 0.709 | 91.9% |
| 2 | 0.782 | 93.8% |

Baseline reference (TF-IDF + Logistic Regression): macro F1 0.688,
accuracy 88.4% (`reports/baseline_metrics.json`).

## Known failure modes

_Pending — see `reports/error_analysis.md` once the SLM finishes training
and its validation predictions can be analyzed. `src/eval/error_analysis.py`
is built and validated (tested against the baseline's errors already);
running it against the SLM's own mistakes is the last step._

## Limitations

- **Description-dependent.** The model only sees `name` + `description` +
  parameter schema — a tool with a misleading or missing description (316
  training rows and 42 test rows have an empty description, per the data
  audit) is judged on name and parameters alone, which is markedly less
  reliable.
- **Rare classes stay rare classes.** Class weighting improves recall on
  `Financial`/`Other` versus an unweighted model, but with only 111/76
  training examples respectively, the model has seen far less variety in
  what these categories look like than it has for `Read`. Treat
  low-support-class predictions with more skepticism than high-support-class
  ones — see README.md → Calibration & abstention discussion.
- **English-centric.** Tool descriptions in the training data are
  overwhelmingly English; behavior on non-English descriptions is
  untested.
- **No calibrated confidence / abstention.** The shipped model always emits
  one of six labels, even for genuinely ambiguous or out-of-distribution
  inputs — there is no "I don't know." Softmax probabilities are available
  from the logits but were not calibrated (e.g. temperature scaling) or
  validated for reliability as confidence estimates.
- **Static snapshot.** Trained on a mid-2026 crawl of public MCP servers;
  tool-naming conventions and MCP ecosystem patterns will drift, and the
  model isn't automatically kept current (see README.md → Production
  monitoring and drift discussion).

## Reproduction

```bash
python -m src.slm.train_slm      # trains and saves models/slm/adapter
python -m src.slm.evaluate_slm   # scores models/slm/adapter on validation
python predict.py --model-path models/slm/adapter \
                   --input data/test_unlabeled.jsonl \
                   --output predictions.csv
```

Model weights are not included in this submission (per instructions); the
adapter directory (`models/slm/adapter/`, LoRA weights only, a few tens of
MB) is included, and the base model (`Qwen/Qwen2.5-0.5B-Instruct`) is
pulled from the Hugging Face Hub on first run.
