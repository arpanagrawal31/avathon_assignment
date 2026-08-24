"""
Pydantic schema for email triage extraction.
"""

import json
from typing import Literal

from pydantic import BaseModel, field_validator


class ActionItem(BaseModel):
    owner: str | None = None
    task: str
    deadline: str | None = None


class TriageOutput(BaseModel):
    intent: Literal[
        "request", "status_update", "scheduling",
        "approval_request", "escalation", "fyi", "other"
    ]
    urgency: Literal["low", "medium", "high"]
    requires_response: bool
    action_items: list[ActionItem]
    escalation_flag: bool

    @field_validator("action_items", mode="before")
    @classmethod
    def coerce_none_to_list(cls, v):
        """Accept null as empty list. LLMs sometimes emit null instead of []."""
        return v if v is not None else []


# ── Validation helpers ──────────────────────────────────────

def validate_json_string(raw: str) -> tuple[TriageOutput | None, str | None]:
    """Parse a raw JSON string and validate against schema"""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"

    try:
        model = TriageOutput.model_validate(data)
        return model, None
    except Exception as e:
        return None, f"Schema validation failed: {e}"


def check_grounding(label: TriageOutput, email_body: str) -> dict:
    """Check that owner/deadline fields are verbatim substrings of the email"""
    body_lower = email_body.lower()
    details = []

    for i, item in enumerate(label.action_items):
        for field_name in ("owner", "deadline"):
            value = getattr(item, field_name)
            if value is not None:
                supported = value.lower() in body_lower
                details.append({
                    "action_item_idx": i,
                    "field": field_name,
                    "value": value,
                    "supported": supported,
                })

    total = len(details)
    unsupported = sum(1 for d in details if not d["supported"])

    return {
        "total_spans": total,
        "unsupported": unsupported,
        "unsupported_rate": unsupported / total if total > 0 else 0.0,
        "details": details,
    }
