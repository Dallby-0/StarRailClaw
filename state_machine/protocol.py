from __future__ import annotations

import json
import re
from typing import Any


def _closed_object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


def state_bootstrap_json_schema() -> dict[str, Any]:
    """Strict schema for the single identify + intent + bootstrap Responses call."""
    level = {"type": "string", "enum": ["high", "mid", "low"]}
    bbox = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
    empty_object = _closed_object({})

    text_element = _closed_object({
        "type": {"type": "string", "enum": ["text_line"]},
        "text": {"type": "string"},
        "bbox": bbox,
        "brief": {"type": "string"},
        "stability": level,
        "discrimination": level,
    })
    pattern_element = _closed_object({
        "type": {"type": "string", "enum": ["pattern"]},
        "bbox": bbox,
        "brief": {"type": "string"},
        "stability": level,
        "discrimination": level,
    })

    fixed_point = _closed_object({
        "type": {"type": "string", "enum": ["fixed_point"]},
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
    })
    run_preset = _closed_object({
        "type": {"type": "string", "enum": ["run_preset"]},
        "name": {"type": "string", "enum": ["wait_till_combat_end", "find_and_interact_with_next_object"]},
    })
    region_template = _closed_object({
        "type": {"type": "string", "enum": ["region_template"]},
        "template_bbox": bbox,
        "search_rect": bbox,
        "threshold": {"type": "number", "minimum": 0, "maximum": 1},
    })
    expected_after = _closed_object({
        "screen_should_change": {"type": "boolean"},
        "exit_likely": {"type": "boolean"},
    })
    event = _closed_object({"type": {"type": "string"}})
    step = _closed_object({
        "step_id": {"type": "string"},
        "resolver": {"anyOf": [fixed_point, region_template, run_preset]},
        "expected_after": expected_after,
        "emits_on_success": event,
        "brief": {"type": "string"},
    })
    intent_route = _closed_object({
        "intent_kind": {"type": "string"},
        "phase": {"type": "string"},
    })
    operation = _closed_object({
        "operation": {"type": "string"},
        "is_default": {"type": "boolean"},
        "intent_scope": {"type": "string", "enum": ["intent_invariant", "intent_specific"]},
        "intent_effect": {"type": "string", "enum": ["preserve", "advance", "complete", "none"]},
        "safety": {"type": "string", "enum": ["low_risk", "reversible", "commit", "destructive"]},
        "expected_event": {"type": "string"},
        "intent_routes": {"type": "array", "items": intent_route, "maxItems": 4},
        "steps": {"type": "array", "items": step, "minItems": 1, "maxItems": 2},
    })

    transition = _closed_object({
        "from_phase": {"type": "string"},
        "event": {"type": "string"},
        "next_phase": {"type": "string"},
        "status": {"type": "string", "enum": ["running", "completed"]},
        # Strict structured output cannot safely express arbitrary JSON maps.
        # The initial visual proposal therefore carries no dynamic fact patch;
        # deterministic runtime events may enrich facts later.
        "fact_patch": empty_object,
    })
    intent_proposal = _closed_object({
        "kind": {"type": "string"},
        "phase": {"type": "string"},
        "params": empty_object,
        "facts": empty_object,
        "transitions": {"type": "array", "items": transition, "maxItems": 8},
        "completion": _closed_object({"event": {"type": "string"}}),
    })

    return _closed_object({
        "page_summary": {"type": "string"},
        "slug": {"type": "string", "pattern": "^[a-z0-9]+(?:_[a-z0-9]+)*$"},
        "possible_page_type": {"type": "string"},
        "elements": {"type": "array", "items": {"anyOf": [text_element, pattern_element]}, "maxItems": 8},
        "intent_assessment": _closed_object({
            "relation": {"type": "string", "enum": ["expected_step", "blocking_overlay", "completion_evidence", "unrelated", "contradiction", "unknown"]},
            "reason": {"type": "string"},
        }),
        "intent_proposal": {"anyOf": [{"type": "null"}, intent_proposal]},
        "bootstrap_operations": {"type": "array", "items": operation, "maxItems": 2},
    })


def _load_json_lenient(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass

    # Vision models occasionally omit a comma between adjacent numeric bbox
    # coordinates, e.g. [26 25, 62 80].  Repair only numeric adjacency inside
    # JSON arrays; do not attempt broad prose/JSON reconstruction.
    def repair_numeric_array(match: re.Match[str]) -> str:
        body = re.sub(r"(?<=\d)[ \t]+(?=-?\d)", ", ", match.group(1))
        return f"[{body}]"

    repaired = re.sub(r"\[([0-9,\.\-+ \t\r\n]+)\]", repair_numeric_array, text)
    try:
        value = json.loads(repaired)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def parse_state_payload(text: str) -> dict[str, Any] | None:
    payload = _load_json_lenient(text)
    if payload is None:
        return None
    required = {"page_summary", "slug", "elements", "bootstrap_operations"}
    if not required.issubset(payload.keys()):
        return None
    if not isinstance(payload.get("elements"), list) or not isinstance(payload.get("bootstrap_operations"), list):
        return None
    payload.setdefault("possible_page_type", "none")
    return payload
