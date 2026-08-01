"""Shared data loading and text-representation utilities.

Both the baseline (TF-IDF) and the SLM pipelines build their input text from
these same functions, so the two models are compared on identical inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

MAX_PARAMS_RENDERED = 12
MAX_PARAM_DESC_CHARS = 140
MAX_TOOL_DESC_CHARS = 900


@dataclass
class ToolRecord:
    record_id: str
    server_slug: str
    name: str
    description: str
    input_schema_raw: str
    category: str | None = None


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc


def load_records(path: str | Path) -> list[ToolRecord]:
    records = []
    for row in read_jsonl(path):
        records.append(
            ToolRecord(
                record_id=row["record_id"],
                server_slug=row["server_slug"],
                name=row.get("name", "") or "",
                description=row.get("description", "") or "",
                input_schema_raw=row.get("input_schema", "") or "",
                category=row.get("category"),
            )
        )
    return records


def _parse_schema(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _describe_property(name: str, spec: Any, required: bool) -> str:
    if not isinstance(spec, dict):
        return f"- {name} ({'required' if required else 'optional'})"

    type_hint = spec.get("type")
    if isinstance(type_hint, list):
        type_hint = "|".join(str(t) for t in type_hint)
    if type_hint is None:
        if "enum" in spec:
            type_hint = "enum"
        elif "oneOf" in spec or "anyOf" in spec:
            type_hint = "union"
        else:
            type_hint = "any"

    piece = f"- {name} ({type_hint}, {'required' if required else 'optional'})"

    enum_vals = spec.get("enum")
    if isinstance(enum_vals, list) and enum_vals:
        shown = ", ".join(str(v) for v in enum_vals[:6])
        piece += f" enum[{shown}]"

    desc = spec.get("description")
    if isinstance(desc, str) and desc.strip():
        desc = " ".join(desc.split())[:MAX_PARAM_DESC_CHARS]
        piece += f": {desc}"

    return piece


def summarize_schema(raw: str) -> str:
    """Render a JSON Schema string as a compact, human-readable parameter list.

    Dumping raw JSON schemas verbatim wastes tokens on syntax (braces,
    quoting) that carries no signal for this task, and some schemas are deep
    enough to blow past reasonable sequence-length budgets. We instead pull
    out the fields that actually matter for judging tool *effect*: parameter
    names, types, required-ness, enums, and short descriptions.
    """
    schema = _parse_schema(raw)
    if schema is None:
        return "no parameters" if not raw.strip() else "unparsable schema"

    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "no parameters"

    required = set(schema.get("required") or [])
    lines = []
    for i, (prop_name, prop_spec) in enumerate(properties.items()):
        if i >= MAX_PARAMS_RENDERED:
            remaining = len(properties) - MAX_PARAMS_RENDERED
            lines.append(f"- ... and {remaining} more parameter(s)")
            break
        lines.append(_describe_property(prop_name, prop_spec, prop_name in required))

    return "\n".join(lines)


def build_input_text(record: ToolRecord) -> str:
    """Compose the single text prompt fed to both the baseline and the SLM.

    Format: name, then description (truncated), then a flattened parameter
    summary. Field labels ("Tool name:", "Description:", ...) are kept so a
    language model can rely on consistent structure, and TF-IDF picks up the
    same tokens regardless of the literal labels.
    """
    name = record.name.strip() or "unknown_tool"
    description = " ".join(record.description.split())[:MAX_TOOL_DESC_CHARS]
    if not description:
        description = "(no description provided)"
    params = summarize_schema(record.input_schema_raw)

    return (
        f"Tool name: {name}\n"
        f"Description: {description}\n"
        f"Parameters:\n{params}"
    )
