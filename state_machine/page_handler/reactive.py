from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from state_machine.time_utils import now_iso as _now_iso


PROVIDER_STATUSES = {"proposed", "canary", "active", "disabled"}
PROVIDER_SCOPES = {"instance", "family"}
REPEAT_POLICIES = {"once_per_visit", "after_confirmed_effect"}
DEFAULT_BUDGETS = {
    "soft_actions": 6,
    "hard_actions": 12,
    "max_repairs": 2,
    "max_seconds": 180,
    "max_expensive_probes": 2,
}
GLOBAL_HARD_ACTIONS = 120
GLOBAL_HARD_REPAIRS = 20


def _slug(raw: Any, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or fallback


def _rect(raw: Any) -> list[int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        values = [max(0, min(1000, int(value))) for value in raw]
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _bounded_int(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _bounded_float(raw: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def normalize_locator(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type") or "").strip()
    if kind == "point":
        try:
            x = max(0, min(999, int(raw.get("x"))))
            y = max(0, min(999, int(raw.get("y"))))
        except (TypeError, ValueError):
            return None
        if str(raw.get("coordinate_space") or "") != "logical":
            return None
        return {
            "type": "point",
            "x": x,
            "y": y,
            "coordinate_space": "logical",
            "source": str(raw.get("source") or "bootstrap"),
        }
    if kind == "region_template":
        if str(raw.get("coordinate_space") or "") != "logical":
            return None
        search_rect = _rect(raw.get("search_rect"))
        template_bbox = _rect(raw.get("template_bbox") or raw.get("bbox"))
        template_path = str(raw.get("template_path") or "").strip()
        if not template_path and template_bbox is None:
            return None
        return {
            "type": kind,
            "template_path": template_path,
            "template_bbox": template_bbox,
            "search_rect": search_rect,
            "threshold": _bounded_float(raw.get("threshold", 0.82), 0.82, 0.0, 1.0),
            "coordinate_space": "logical",
        }
    if kind == "text_target":
        if str(raw.get("coordinate_space") or "") != "logical":
            return None
        rect = _rect(raw.get("rect") or raw.get("search_rect"))
        texts = raw.get("texts") if isinstance(raw.get("texts"), list) else [raw.get("text") or raw.get("contains")]
        texts = [str(value).strip() for value in texts if str(value or "").strip()]
        if rect is None or not texts:
            return None
        return {"type": kind, "rect": rect, "texts": texts[:6], "match": "contains", "coordinate_space": "logical"}
    return None


def normalize_hint(raw: Any, *, deferred: bool = False, index: int = 1) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type") or raw.get("kind") or "").strip()
    if kind in {"text_present", "text_contains"}:
        kind = "text"
    if kind in {"template_present", "region_template"}:
        kind = "template"
    hint: dict[str, Any] = {
        "id": _slug(raw.get("id"), f"hint_{index}"),
        "type": kind,
        "status": "deferred" if deferred else str(raw.get("status") or "confirmed"),
    }
    if str(raw.get("coordinate_space") or "") != "logical":
        return None
    if kind == "text":
        rect = _rect(raw.get("rect"))
        texts = raw.get("texts") if isinstance(raw.get("texts"), list) else [raw.get("text") or raw.get("contains")]
        texts = [str(value).strip() for value in texts if str(value or "").strip()]
        if rect is None or not texts:
            return None
        hint.update({"rect": rect, "texts": texts[:6], "cost": "ocr"})
    elif kind == "line_count":
        rect = _rect(raw.get("rect"))
        if rect is None:
            return None
        minimum = _bounded_int(raw.get("min", raw.get("minimum", 1)), 1, 0, 100)
        maximum = _bounded_int(raw.get("max", raw.get("maximum", minimum)), minimum, minimum, 100)
        hint.update({"rect": rect, "min": minimum, "max": maximum, "cost": "detection"})
    elif kind == "template":
        rect = _rect(raw.get("rect") or raw.get("search_rect"))
        bbox = _rect(raw.get("template_bbox") or raw.get("bbox"))
        path = str(raw.get("template_path") or "").strip()
        if not path and bbox is None:
            return None
        hint.update({
            "rect": rect,
            "template_bbox": bbox,
            "template_path": path,
            "threshold": _bounded_float(raw.get("threshold", 0.82), 0.82, 0.0, 1.0),
            "cost": "cheap",
        })
    else:
        return None
    hint["coordinate_space"] = "logical"
    if deferred:
        predecessor = str(raw.get("materialize_after") or "").strip()
        if not predecessor:
            return None
        hint["materialize_after"] = _slug(predecessor, predecessor)
    return hint


def normalize_effect_hint(raw: Any, *, index: int = 1) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    probe = normalize_hint(raw.get("probe") if isinstance(raw.get("probe"), dict) else raw, index=index)
    if probe is None:
        return None
    expected = str(raw.get("expected") or "pass")
    if expected not in {"pass", "becomes_pass", "becomes_fail"}:
        return None
    return {"id": _slug(raw.get("id"), f"effect_{index}"), "probe": probe, "expected": expected}


def normalize_provider(raw: Any, *, index: int = 1, default_priority: int = 0) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    provider_id = _slug(raw.get("provider_id") or raw.get("id") or raw.get("step_id"), f"provider_{index}")
    raw_locators = raw.get("locators") if isinstance(raw.get("locators"), list) else []
    if not raw_locators and isinstance(raw.get("resolver"), dict):
        raw_locators = [raw["resolver"]]
    locators = [locator for item in raw_locators[:4] if (locator := normalize_locator(item)) is not None]
    if not locators:
        return None
    status = str(raw.get("status") or "proposed")
    scope = str(raw.get("scope") or "instance")
    repeat = str(raw.get("repeat_policy") or "once_per_visit")
    provider: dict[str, Any] = {
        "provider_id": provider_id,
        "status": status if status in PROVIDER_STATUSES else "proposed",
        "scope": scope if scope in PROVIDER_SCOPES else "instance",
        "base_priority": _bounded_int(raw.get("base_priority", default_priority), default_priority, -100, 100),
        "repeat_policy": repeat if repeat in REPEAT_POLICIES else "once_per_visit",
        "locators": locators,
        "hints": [hint for i, item in enumerate(raw.get("hints") if isinstance(raw.get("hints"), list) else [], 1) if (hint := normalize_hint(item, index=i)) is not None],
        "deferred_hints": [hint for i, item in enumerate(raw.get("deferred_hints") if isinstance(raw.get("deferred_hints"), list) else [], 1) if (hint := normalize_hint(item, deferred=True, index=i)) is not None],
        "effect_hints": [hint for i, item in enumerate(raw.get("effect_hints") if isinstance(raw.get("effect_hints"), list) else [], 1) if (hint := normalize_effect_hint(item, index=i)) is not None],
        "successors": [_slug(value, "") for value in raw.get("successors", []) if _slug(value, "")][:4] if isinstance(raw.get("successors"), list) else [],
        "emits_on_success": dict(raw.get("emits_on_success") or {}),
        "brief": str(raw.get("brief") or raw.get("label") or ""),
        "success_count": _bounded_int(raw.get("success_count", 0), 0, 0, 1_000_000),
        "result_counts": dict(raw.get("result_counts") or {}),
        "successful_visits": [str(value) for value in raw.get("successful_visits", [])][-8:] if isinstance(raw.get("successful_visits"), list) else [],
        "created_at": str(raw.get("created_at") or _now_iso()),
        "updated_at": _now_iso(),
    }
    if isinstance(raw.get("generalization_evidence"), dict):
        provider["generalization_evidence"] = deepcopy(raw["generalization_evidence"])
    return provider


def normalize_budgets(raw: Any) -> dict[str, int]:
    source = raw if isinstance(raw, dict) else {}
    soft = _bounded_int(source.get("soft_actions"), DEFAULT_BUDGETS["soft_actions"], 1, 20)
    hard = _bounded_int(source.get("hard_actions"), DEFAULT_BUDGETS["hard_actions"], soft, 40)
    return {
        "soft_actions": soft,
        "hard_actions": hard,
        "max_repairs": _bounded_int(source.get("max_repairs"), DEFAULT_BUDGETS["max_repairs"], 0, 3),
        "max_seconds": _bounded_int(source.get("max_seconds"), DEFAULT_BUDGETS["max_seconds"], 10, 600),
        "max_expensive_probes": _bounded_int(source.get("max_expensive_probes"), DEFAULT_BUDGETS["max_expensive_probes"], 0, 4),
    }


def normalize_operation(raw: Any, *, index: int = 1) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    operation = _slug(raw.get("operation"), f"operation_{index}")
    raw_providers = raw.get("providers") if isinstance(raw.get("providers"), list) else []
    providers = [provider for i, item in enumerate(raw_providers[:12], 1) if (provider := normalize_provider(item, index=i, default_priority=100 - i * 10)) is not None]
    if not providers:
        return None
    provider_ids = [str(provider.get("provider_id") or "") for provider in providers]
    if len(provider_ids) != len(set(provider_ids)):
        return None
    return {
        "operation": operation,
        "intent_scope": str(raw.get("intent_scope") or "intent_specific"),
        "intent_effect": str(raw.get("intent_effect") or "none"),
        "expected_event": str(raw.get("expected_event") or "") or None,
        "providers": providers,
        "budgets": normalize_budgets(raw.get("budgets")),
    }


def initial_cursor(*, visit_id: str, operation: str, now_monotonic: float = 0.0) -> dict[str, Any]:
    return {
        "visit_id": visit_id,
        "operation": operation,
        "started_monotonic": float(now_monotonic),
        "total_actions": 0,
        "actions_since_repair": 0,
        "repair_count": 0,
        "attempt_counts": {},
        "confirmed_effect_epoch": 0,
        "last_provider_id": None,
        "successor_ids": [],
        "materialized_hints": {},
        "repair_hashes": [],
        "history": [],
        "updated_at": _now_iso(),
    }


def cursor_for(runtime: dict[str, Any], visit_id: str, operation: str, *, now_monotonic: float = 0.0) -> dict[str, Any]:
    cursors = runtime.get("reactive_visits")
    if not isinstance(cursors, dict):
        cursors = {}
        runtime["reactive_visits"] = cursors
    key = f"{visit_id}|{operation}"
    cursor = cursors.get(key)
    if not isinstance(cursor, dict):
        cursor = initial_cursor(visit_id=visit_id, operation=operation, now_monotonic=now_monotonic)
        cursors[key] = cursor
    return cursor


def clear_visit(runtime: dict[str, Any], visit_id: str) -> None:
    cursors = runtime.get("reactive_visits")
    if isinstance(cursors, dict):
        for key in [key for key in cursors if key.startswith(f"{visit_id}|")]:
            del cursors[key]


def reset_run_local_progress(runtime: dict[str, Any]) -> None:
    runtime["page_visit"] = None
    runtime["reactive_visits"] = {}
    runtime["reactive_total_actions"] = 0
    runtime["reactive_total_repairs"] = 0


def provider_available(provider: dict[str, Any], cursor: dict[str, Any]) -> bool:
    if str(provider.get("status")) == "disabled":
        return False
    count = int((cursor.get("attempt_counts") or {}).get(str(provider.get("provider_id")), 0) or 0)
    if count == 0:
        return True
    return str(provider.get("repeat_policy")) == "after_confirmed_effect" and int(cursor.get("confirmed_effect_epoch", 0) or 0) >= count


def provider_score(provider: dict[str, Any], cursor: dict[str, Any], hint_results: list[str]) -> float:
    score = float(provider.get("base_priority", 0) or 0)
    status = str(provider.get("status") or "proposed")
    score += {"active": 12.0, "canary": 4.0, "proposed": 0.0}.get(status, -1000.0)
    score += min(8.0, float(provider.get("success_count", 0) or 0) * 2.0)
    successor_ids = {str(value) for value in cursor.get("successor_ids", [])}
    if str(provider.get("provider_id")) in successor_ids:
        score += 24.0
    for result in hint_results:
        score += {"pass": 60.0, "unknown": 0.0, "fail": -60.0}.get(result, 0.0)
    return score


def rank_providers(operation: dict[str, Any], cursor: dict[str, Any], hint_results: dict[str, list[str]]) -> list[dict[str, Any]]:
    candidates = [item for item in operation.get("providers", []) if isinstance(item, dict) and provider_available(item, cursor)]
    return sorted(
        candidates,
        key=lambda item: (
            -provider_score(item, cursor, hint_results.get(str(item.get("provider_id")), [])),
            str(item.get("provider_id") or ""),
        ),
    )


def record_attempt(cursor: dict[str, Any], provider: dict[str, Any], *, executed: bool = True) -> None:
    provider_id = str(provider.get("provider_id") or "")
    counts = cursor.get("attempt_counts") if isinstance(cursor.get("attempt_counts"), dict) else {}
    cursor["attempt_counts"] = counts
    counts[provider_id] = int(counts.get(provider_id, 0) or 0) + 1
    if executed:
        cursor["total_actions"] = int(cursor.get("total_actions", 0) or 0) + 1
        cursor["actions_since_repair"] = int(cursor.get("actions_since_repair", 0) or 0) + 1
    cursor["last_provider_id"] = provider_id
    cursor["updated_at"] = _now_iso()


def mark_confirmed_effect(cursor: dict[str, Any]) -> None:
    cursor["confirmed_effect_epoch"] = int(cursor.get("confirmed_effect_epoch", 0) or 0) + 1
    cursor["actions_since_repair"] = 0
    # A productive action makes the visit-local repair attempts stale.  The
    # global hard limit remains the final runaway guard.
    cursor["repair_count"] = 0
    cursor["updated_at"] = _now_iso()


def update_successor_context(cursor: dict[str, Any], provider: dict[str, Any], result: str) -> None:
    usable_unverified = result == "unverified" and not provider.get("effect_hints")
    if result in {"confirmed", "transitioned"} or usable_unverified:
        cursor["successor_ids"] = [str(value) for value in provider.get("successors", [])]
    else:
        cursor["successor_ids"] = []
    cursor["updated_at"] = _now_iso()


def normalized_patch_hash(patch: Any) -> str:
    if not isinstance(patch, dict):
        return ""
    return json.dumps(deepcopy(patch), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
