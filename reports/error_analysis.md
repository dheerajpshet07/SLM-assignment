# Error Analysis

SLM (Qwen2.5-0.5B + LoRA) on the validation split: 215 errors out of 4,429
rows (4.9%), down from the baseline's 514 (11.6%). Examples below are read
directly from `reports/error_analysis/confusion_examples.json`, generated
by `src/eval/error_analysis.py`, which groups every misclassification by
(true, predicted) label pair. Full per-record predictions are in
`reports/validation_predictions_slm.jsonl`.

## Where the errors are

| True -> Predicted | Count |
|---|---:|
| Write -> Read | 71 |
| Read -> Write | 50 |
| Execute -> Read | 25 |
| Execute -> Write | 20 |
| Other -> Read | 10 |
| Read -> Execute | 9 |
| Destructive -> Write | 5 |
| Write -> Destructive | 5 |
| Destructive -> Read | 4 |
| Write -> Financial | 3 |

Read/Write confusion alone accounts for over half of all errors (121 of
215). Execute is the next biggest source, mostly confused with Read and
Write. Destructive and Financial barely show up as error sources at all,
which matches their strong per-class F1 (0.96 and 0.86) in
`reports/slm_metrics.json` - the model is genuinely good at spotting the
high-stakes categories, which is the more important property for this
task's actual use case.

## Failure mode 1: umbrella tools with no verb to key on

The single biggest Write<->Read source is tools whose description names a
resource but never says what the tool actually does to it. Several Azure
tools show this exactly:

- `datadog` - "Work with Azure Native ISV services including Datadog" (true `Write`, predicted `Read`)
- `foundry` - "Work with Azure AI Foundry models and deployments" (true `Write`, predicted `Read`)
- `servicebus` - "Work with Azure Service Bus messaging" (true `Write`, predicted `Read`)
- `sql` - "Work with Azure SQL Database servers and firewall rules" (true `Write`, predicted `Read`)

"Work with X" gives the model nothing to work with. These are almost
certainly umbrella tools that route to both read and write sub-operations
depending on runtime arguments the static description doesn't expose - a
genuinely multi-label situation being forced into one of six buckets. No
amount of better training fixes this; it's a property of the tool
description, not something the model failed to learn. A fix here is
upstream of modeling: either the label needs to reflect the riskier of the
two behaviors (write dominates read for a gating use case), or these tools
need multi-label support (see the follow-up-interview notes at the top of
this repo's README).

## Failure mode 2: the description's verb points at a different category than the label does

A few cases where the tool description contains a word that flatly
contradicts its own label:

- `brand_preview` - "Generate a visual proof page... **Writes** .brand/brand-preview.html" - true label `Execute`, predicted `Read`. The description literally says "Writes," the model ignored that and guessed `Read` anyway (case 1 shows some tools are read-only despite touching a file for caching, so this may be a case where the model latched onto "preview"/"visual" as read-flavored words instead).
- `smart_translate_workflow` - "**Execute** intelligent translation workflow..." - true label `Read`, predicted... `Execute`. The description's own first word matches the model's guess, not the ground-truth label.
- `compile` (dbt) - "Generate executable SQL from models **without running**" - true `Read`, predicted `Execute`. "Compile" and "executable" read as Execute-flavored even though the description explicitly says nothing gets run.
- `parse` (dbt) - "Parse and validate project files" - true `Read`, predicted `Execute`. Same pattern: "parse" sounds procedural, but this is static validation, not an action with side effects.

This is the ambiguous Write-vs-Execute (and here, Read-vs-Execute) boundary
the assignment brief specifically calls out. The taxonomy's intended
distinction (does invoking this change state, or run something with
side effects, vs. just inspect/validate) doesn't always line up with which
verb a tool author reached for when naming or describing their tool.
`compile` and `parse` are the clearest examples: both explicitly disclaim
side effects in their own descriptions, and the model still leaned on the
surface-level verb.

## Failure mode 3: control/status tools named with the wrong-sounding verb

- `mcp_ado_pipelines_update_build_stage` - "Update a build stage (cancel, retry, or run)" - true `Execute`, predicted `Write`. "Update" is one of the strongest `Write` signals in this dataset, but this tool controls whether a CI pipeline stage runs, cancels, or retries - it's manipulating execution state, not writing data.
- `scout.reddit.result` / `scout.x.result` - "Get [X] scout run status and results by run ID" - true `Execute`, predicted `Read`. Polling a running job's status and results reads like a pure read operation, but these are labeled `Execute`, presumably because they're part of an execution/job-lifecycle tool family rather than standalone data access.

Both directions of this pattern point at the same root cause: the model
(reasonably) treats "update" as a write-verb and "get status" as a
read-verb, but this taxonomy sometimes categorizes by *what kind of
resource* is being touched (a running job/pipeline) rather than by the
verb alone. That's a genuinely hard distinction to learn from
name+description text without more explicit signal that a tool belongs to
an execution-control family.

## Failure mode 4: `Other` has no internal pattern to learn

`Other` recall is 0.267 (4 of 15 validation examples correct) - by far the
weakest number in the whole per-class table, and 10 of its 15 examples
were predicted as `Read`. The examples that get missed span completely
unrelated domains:

- `understand_image` - "Analyze images with AI vision..." (true `Other`, predicted `Read`)
- `qimen_dunjia_calculate` - a Chinese divination/astrology calculator (true `Other`, predicted `Read`)
- `repurpose_content` - reformats a blog post into platform-specific social copy (true `Other`, predicted `Read`)
- `lambda_function` - "invoking a specific AWS Lambda function with parameters" (true `Other`, predicted `Read`)

There's no shared vocabulary or structure tying these together - `Other`
is whatever doesn't fit the other five categories, which is exactly the
kind of class a model struggles to learn a decision boundary for,
regardless of how the loss is weighted. With only 76 training examples
covering this much semantic variety, the model has effectively seen a
handful of scattered one-off cases rather than a coherent pattern, and it
defaults to the safest guess (`Read`, the majority class) when nothing
matches.

## What this suggests for next steps

1. **Failure modes 1 and 4 are data/taxonomy problems, not modeling
   problems.** No amount of further training fixes an umbrella tool with
   no verb to key on, or a catch-all class with 76 examples spanning
   unrelated domains. Better fixes: multi-label support for failure mode
   1, and either more `Other` examples or a narrower definition of what
   belongs in it for failure mode 4.
2. **Failure modes 2 and 3 are more tractable.** Both come from the model
   over-indexing on individual verbs ("update," "execute," "compile")
   instead of the fuller sentence context that would disambiguate them
   ("without running," "cancel, retry, or run a *stage*"). A model with
   more capacity, or more training steps on these specific patterns, might
   close some of this gap - worth checking whether a 1.5B/3B run
   (see README.md, Model choice) does better here specifically, not just
   on the aggregate macro F1.
3. **The high-stakes classes are already solid.** `Destructive` (F1 0.96)
   and `Financial` (F1 0.86) barely appear as error sources. For a
   tool-gating use case, that's the more important property to have gotten
   right than fixing the Read/Write boundary - see README.md, Results, for
   the full per-class breakdown.
