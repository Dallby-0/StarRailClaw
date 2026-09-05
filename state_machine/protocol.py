from __future__ import annotations

import json
import math
import re
from typing import Any


def _closed_object(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


def state_bootstrap_json_schema(available_tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Strict schema for the single identify + intent + bootstrap Responses call."""
    if available_tools is None:
        from state_machine.presets import tool_catalog

        available_tools = tool_catalog()
    tool_names = sorted({
        str(item.get("name") or "").strip()
        for item in available_tools
        if str(item.get("name") or "").strip()
    })
    level = {"type": "string", "enum": ["high", "mid", "low"]}
    bbox = {
        "type": "array",
        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
        "minItems": 4,
        "maxItems": 4,
    }
    role = {"type": "string", "enum": ["identity", "identity_support", "interaction", "instance", "diagnostic"]}

    text_element = _closed_object({
        "type": {"type": "string", "enum": ["text_line"]},
        "text": {"type": "string"},
        "bbox": bbox,
        "brief": {"type": "string"},
        "role": role,
        "stability": level,
        "discrimination": level,
    })
    pattern_element = _closed_object({
        "type": {"type": "string", "enum": ["pattern"]},
        "bbox": bbox,
        "brief": {"type": "string"},
        "role": role,
        "stability": level,
        "discrimination": level,
    })

    point = _closed_object({
        "type": {"type": "string", "enum": ["point"]},
        "x": {"type": "integer", "minimum": 0, "maximum": 1000},
        "y": {"type": "integer", "minimum": 0, "maximum": 1000},
        "coordinate_space": {"type": "string", "enum": ["logical"]},
        "source": {"type": "string", "enum": ["bootstrap"]},
    })
    region_template = _closed_object({
        "type": {"type": "string", "enum": ["region_template"]},
        "template_bbox": bbox,
        "search_rect": bbox,
        "threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "coordinate_space": {"type": "string", "enum": ["logical"]},
    })
    text_target = _closed_object({
        "type": {"type": "string", "enum": ["text_target"]},
        "rect": bbox,
        "texts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
        "coordinate_space": {"type": "string", "enum": ["logical"]},
    })
    template_hint = _closed_object({
        "id": {"type": "string"},
        "type": {"type": "string", "enum": ["template"]},
        "template_bbox": bbox,
        "rect": bbox,
        "threshold": {"type": "number", "minimum": 0, "maximum": 1},
        "coordinate_space": {"type": "string", "enum": ["logical"]},
    })
    text_hint = _closed_object({
        "id": {"type": "string"},
        "type": {"type": "string", "enum": ["text"]},
        "rect": bbox,
        "texts": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
        "coordinate_space": {"type": "string", "enum": ["logical"]},
    })
    line_hint = _closed_object({
        "id": {"type": "string"},
        "type": {"type": "string", "enum": ["line_count"]},
        "rect": bbox,
        "min": {"type": "integer", "minimum": 0, "maximum": 20},
        "max": {"type": "integer", "minimum": 0, "maximum": 20},
        "coordinate_space": {"type": "string", "enum": ["logical"]},
    })
    hint = {"anyOf": [template_hint, text_hint, line_hint]}

    def deferred(variant: dict[str, Any]) -> dict[str, Any]:
        props = dict(variant["properties"])
        props["materialize_after"] = {"type": "string"}
        return _closed_object(props)

    deferred_hint = {"anyOf": [deferred(template_hint), deferred(text_hint), deferred(line_hint)]}
    event = _closed_object({"type": {"type": "string"}})
    effect_hint = _closed_object({
        "id": {"type": "string"},
        "probe": hint,
        "expected": {"type": "string", "enum": ["pass", "becomes_pass", "becomes_fail"]},
    })
    provider = _closed_object({
        "provider_id": {"type": "string"},
        "base_priority": {"type": "integer", "minimum": -100, "maximum": 100},
        "repeat_policy": {"type": "string", "enum": ["once_per_visit", "after_confirmed_effect"]},
        "locators": {"type": "array", "items": {"anyOf": [point, region_template, text_target]}, "minItems": 1, "maxItems": 4},
        "hints": {"type": "array", "items": hint, "maxItems": 3},
        "deferred_hints": {"type": "array", "items": deferred_hint, "maxItems": 2},
        "effect_hints": {"type": "array", "items": effect_hint, "maxItems": 2},
        "successors": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
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
        "expected_event": {"type": "string"},
        "intent_routes": {"type": "array", "items": intent_route, "maxItems": 4},
        "providers": {"type": "array", "items": provider, "minItems": 1, "maxItems": 4},
    })
    execution_variants: list[dict[str, Any]] = [
        _closed_object({
            "kind": {"type": "string", "enum": ["reactive_2d"]},
            "bootstrap_operations": {"type": "array", "items": operation, "minItems": 1, "maxItems": 2},
        })
    ]
    if tool_names:
        execution_variants.append(_closed_object({
            "kind": {"type": "string", "enum": ["invoke_tool"]},
            "tool_name": {"type": "string", "enum": tool_names},
        }))
    execution_variants.append(_closed_object({
        "kind": {"type": "string", "enum": ["cannot_handle"]},
        "reason": {"type": "string"},
    }))

    transition = _closed_object({
        "from_phase": {"type": "string"},
        "event": {"type": "string"},
        "next_phase": {"type": "string"},
        "status": {"type": "string", "enum": ["running", "completed"]},
    })
    intent_proposal = _closed_object({
        "kind": {"type": "string"},
        "phase": {"type": "string"},
        "transitions": {"type": "array", "items": transition, "maxItems": 8},
        "completion": _closed_object({"event": {"type": "string"}}),
    })

    return _closed_object({
        "page_summary": {"type": "string"},
        "slug": {"type": "string", "pattern": "^[a-z0-9]+(?:_[a-z0-9]+)*$"},
        "possible_page_type": {"type": "string"},
        "scene_mode": {"type": "string", "enum": ["ui_2d", "scene_3d", "unknown"]},
        "page_family": {"type": "string", "pattern": "^[a-z0-9]+(?:_[a-z0-9]+)*$"},
        "surface_relation": {"type": "string", "enum": ["same_surface_step", "same_family_new_surface", "different_surface", "uncertain"]},
        "common_identity": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "elements": {"type": "array", "items": {"anyOf": [text_element, pattern_element]}, "maxItems": 8},
        "intent_assessment": _closed_object({
            "relation": {"type": "string", "enum": ["expected_step", "blocking_overlay", "completion_evidence", "unrelated", "contradiction", "unknown"]},
            "reason": {"type": "string"},
        }),
        "intent_proposal": {"anyOf": [{"type": "null"}, intent_proposal]},
        "execution": {"anyOf": execution_variants},
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


def _schema_error(value: Any, schema: dict[str, Any], path: str = "$") -> str | None:
    """Validate the JSON Schema subset used by state_bootstrap_json_schema.

    Keeping this validator local avoids making the FSM runtime depend on an
    optional third-party package.  The schema intentionally uses only this
    small, recursively validated subset.
    """
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        if any(_schema_error(value, candidate, path) is None for candidate in alternatives):
            return None
        return f"{path}: value does not match any allowed schema"

    expected = schema.get("type")
    type_ok = {
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        "boolean": isinstance(value, bool),
    }.get(expected, True)
    if not type_ok:
        return f"{path}: expected {expected}, got {type(value).__name__}"

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{path}: value {value!r} is not in enum"

    if expected == "object":
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        missing = [name for name in required if name not in value]
        if missing:
            return f"{path}: missing required properties {missing}"
        if schema.get("additionalProperties") is False:
            extra = [name for name in value if name not in properties]
            if extra:
                return f"{path}: unexpected properties {extra}"
        for name, child in value.items():
            child_schema = properties.get(name)
            if isinstance(child_schema, dict):
                error = _schema_error(child, child_schema, f"{path}.{name}")
                if error is not None:
                    return error

    if expected == "array":
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            return f"{path}: expected at least {schema['minItems']} items"
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            return f"{path}: expected at most {schema['maxItems']} items"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, child in enumerate(value):
                error = _schema_error(child, item_schema, f"{path}[{index}]")
                if error is not None:
                    return error

    if expected == "string" and isinstance(schema.get("pattern"), str):
        if re.search(schema["pattern"], value) is None:
            return f"{path}: value {value!r} does not match required pattern"

    if expected in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path}: value {value!r} is below minimum {schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path}: value {value!r} is above maximum {schema['maximum']}"

    return None


def parse_state_payload(text: str) -> dict[str, Any] | None:
    payload = _load_json_lenient(text)
    if payload is None:
        return None
    from state_machine.presets import get_tool, tool_catalog

    catalog = tool_catalog()
    error = _schema_error(payload, state_bootstrap_json_schema(catalog))
    if error is not None:
        print(f"[fsm][llm][json_schema][invalid] {error}")
        return None
    scene_mode = str(payload.get("scene_mode") or "")
    execution = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
    kind = str(execution.get("kind") or "")
    if kind == "reactive_2d" and scene_mode != "ui_2d":
        print(f"[fsm][llm][json_schema][invalid] reactive_2d cannot handle scene_mode={scene_mode!r}")
        return None
    if kind == "invoke_tool":
        tool = get_tool(str(execution.get("tool_name") or ""))
        if tool is None or scene_mode not in tool.supported_scene_modes:
            print(f"[fsm][llm][json_schema][invalid] tool does not support scene_mode={scene_mode!r}")
            return None
    return payload
