from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from state_machine.time_utils import now_iso as _now_iso


PROVIDER_STATUSES = {"proposed", "canary", "active", "disabled", "superseded"}
PROVIDER_SCOPES = {"instance", "family"}
REPEAT_POLICIES = {"once_per_visit", "after_confirmed_effect"}
DEFAULT_BUDGETS = {
    "soft_actions": 8,
    "hard_actions": 16,
    "max_repairs": 2,
    "max_seconds": 180,
    "max_expensive_probes": 2,
}
GLOBAL_HARD_ACTIONS = 120
GLOBAL_HARD_REPAIRS = 20
PROGRESSING_MAX_REPAIRS = 8
CHAIN_BONUS = 72.0


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
    if kind == "click_region":
        if str(raw.get("coordinate_space") or "") != "logical":
            return None
        rect = _rect(raw.get("rect"))
        if rect is None:
            return None
        point = raw.get("preferred_point")
        if isinstance(point, (list, tuple)) and len(point) == 2:
            preferred = [max(rect[0], min(rect[2] - 1, int(point[0]))), max(rect[1], min(rect[3] - 1, int(point[1])))]
        else:
            preferred = [(rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2]
        candidates: list[dict[str, Any]] = []
        for item in (raw.get("candidate_points") if isinstance(raw.get("candidate_points"), list) else [])[:4]:
            candidate = item.get("point") if isinstance(item, dict) else item
            if not isinstance(candidate, (list, tuple)) or len(candidate) != 2:
                continue
            candidate = [max(rect[0], min(rect[2] - 1, int(candidate[0]))), max(rect[1], min(rect[3] - 1, int(candidate[1])))]
            candidates.append({"point": candidate, "successes": _bounded_int(item.get("successes", 0) if isinstance(item, dict) else 0, 0, 0, 100000), "no_effects": _bounded_int(item.get("no_effects", 0) if isinstance(item, dict) else 0, 0, 0, 100000), "source": str(item.get("source") or "learned") if isinstance(item, dict) else "learned"})
        if not any(item["point"] == preferred for item in candidates):
            candidates.insert(0, {"point": preferred, "successes": 0, "no_effects": 0, "source": "preferred"})
        return {"type": kind, "rect": rect, "preferred_point": preferred, "candidate_points": candidates[:4], "coordinate_space": "logical"}
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


def normalize_guard(raw: Any, *, index: int = 1) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type") or raw.get("kind") or "").strip()
    if kind in {"text_present", "text_contains"}:
        kind = "text"
    if kind in {"template_present", "region_template"}:
        kind = "template"
    guard: dict[str, Any] = {
        "id": _slug(raw.get("id"), f"guard_{index}"),
        "type": kind,
        "status": str(raw.get("status") or "active"),
    }
    if str(raw.get("coordinate_space") or "") != "logical":
        return None
    if kind == "text":
        rect = _rect(raw.get("rect"))
        texts = raw.get("texts") if isinstance(raw.get("texts"), list) else [raw.get("text") or raw.get("contains")]
        texts = [str(value).strip() for value in texts if str(value or "").strip()]
        if rect is None or not texts:
            return None
        guard.update({"rect": rect, "texts": texts[:6], "cost": "ocr"})
    elif kind == "line_count":
        rect = _rect(raw.get("rect"))
        if rect is None:
            return None
        target = _bounded_int(raw.get("target", raw.get("min", 1)), 1, 0, 100)
        tolerance = _bounded_int(raw.get("tolerance", 1), 1, 0, 10)
        guard.update({"rect": rect, "target": target, "tolerance": tolerance, "cost": "detection"})
    elif kind == "template":
        rect = _rect(raw.get("rect") or raw.get("search_rect"))
        bbox = _rect(raw.get("template_bbox") or raw.get("bbox"))
        path = str(raw.get("template_path") or "").strip()
        if not path and bbox is None:
            return None
        guard.update({
            "rect": rect,
            "template_bbox": bbox,
            "template_path": path,
            "threshold": _bounded_float(raw.get("threshold", 0.82), 0.82, 0.0, 1.0),
            "cost": "cheap",
        })
    else:
        return None
    guard["coordinate_space"] = "logical"
    guard["group"] = _slug(raw.get("group"), str(guard["id"]))
    if guard["status"] not in {"provisional", "active"}:
        guard["status"] = "provisional"
    if isinstance(raw.get("evidence"), dict):
        evidence = raw["evidence"]
        guard["evidence"] = {
            "positive_visits": [str(value) for value in evidence.get("positive_visits", [])][-8:],
            "negative_checks": _bounded_int(evidence.get("negative_checks", 0), 0, 0, 1000),
            "false_positive_count": _bounded_int(evidence.get("false_positive_count", 0), 0, 0, 1000),
        }
    if raw.get("learned_from_watch"):
        guard["learned_from_watch"] = _slug(raw.get("learned_from_watch"), "watch")
    return guard


def normalize_watch(raw: Any, *, index: int = 1) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or str(raw.get("coordinate_space") or "") != "logical":
        return None
    rect = _rect(raw.get("rect"))
    modalities = raw.get("modalities") if isinstance(raw.get("modalities"), list) else [raw.get("modality")]
    modalities = [str(value) for value in modalities if str(value) == "appearance"]
    if rect is None or not modalities:
        return None
    after_provider = _slug(raw.get("after_provider"), "") if raw.get("after_provider") else ""
    return {
        "id": _slug(raw.get("id"), f"watch_{index}"),
        "rect": rect,
        "modalities": modalities,
        "after_provider": after_provider or None,
        "coordinate_space": "logical",
    }


def normalize_effect_hint(raw: Any, *, index: int = 1) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    probe = normalize_guard(raw.get("probe") if isinstance(raw.get("probe"), dict) else raw, index=index)
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
        "operation_key": _slug(raw.get("operation_key"), provider_id),
        "status": status if status in PROVIDER_STATUSES else "proposed",
        "scope": scope if scope in PROVIDER_SCOPES else "instance",
        "base_priority": _bounded_int(raw.get("base_priority", default_priority), default_priority, -100, 100),
        "repeat_policy": repeat if repeat in REPEAT_POLICIES else "once_per_visit",
        "locators": locators,
        "guards": [guard for i, item in enumerate(raw.get("guards") if isinstance(raw.get("guards"), list) else [], 1) if (guard := normalize_guard(item, index=i)) is not None],
        "watches": [watch for i, item in enumerate(raw.get("watches") if isinstance(raw.get("watches"), list) else [], 1) if (watch := normalize_watch(item, index=i)) is not None],
        "effects": [effect for i, item in enumerate(raw.get("effects") if isinstance(raw.get("effects"), list) else [], 1) if (effect := normalize_effect_hint(item, index=i)) is not None],
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
        "entry_providers": [_slug(value, "") for value in raw.get("entry_providers", []) if _slug(value, "")][:4] if isinstance(raw.get("entry_providers"), list) else [],
        "providers": providers,
        "provider_archive": [deepcopy(item) for item in raw.get("provider_archive", []) if isinstance(item, dict)][-8:] if isinstance(raw.get("provider_archive"), list) else [],
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
        "repair_hashes": [],
        "history": [],
        "updated_at": _now_iso(),
    }


def effective_repair_limit(cursor: dict[str, Any], configured_limit: int) -> int:
    if str(cursor.get("progress_assessment") or "") == "progressing":
        return max(int(configured_limit), PROGRESSING_MAX_REPAIRS)
    return int(configured_limit)


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
    runtime["text_only_reentry_state_id"] = None
    runtime["text_only_reentry_count"] = 0


def provider_available(provider: dict[str, Any], cursor: dict[str, Any]) -> bool:
    if str(provider.get("status")) in {"disabled", "superseded"}:
        return False
    count = int((cursor.get("attempt_counts") or {}).get(str(provider.get("provider_id")), 0) or 0)
    if count == 0:
        return True
    return str(provider.get("repeat_policy")) == "after_confirmed_effect" and int(cursor.get("confirmed_effect_epoch", 0) or 0) >= count


def provider_score_components(provider: dict[str, Any], cursor: dict[str, Any], guard_details: list[Any], *, entry_bonus: bool = False) -> dict[str, float]:
    components = {"base": float(provider.get("base_priority", 0) or 0), "status": 0.0, "history": 0.0, "successor": 0.0, "entry": CHAIN_BONUS if entry_bonus else 0.0, "guards": 0.0}
    status = str(provider.get("status") or "proposed")
    components["status"] = {"active": 12.0, "canary": 4.0, "proposed": 0.0}.get(status, -1000.0)
    components["history"] = min(8.0, float(provider.get("success_count", 0) or 0) * 2.0)
    successor_ids = {str(value) for value in cursor.get("successor_ids", [])}
    if str(provider.get("provider_id")) in successor_ids:
        components["successor"] = CHAIN_BONUS
    guards = [item for item in provider.get("guards", []) if isinstance(item, dict)]
    grouped: dict[str, list[float]] = {}
    for index, detail in enumerate(guard_details):
        status = str(guards[index].get("status") or "provisional") if index < len(guards) else "provisional"
        weight = 60.0 if status == "active" else 12.0
        result = str(detail.get("result") or "unknown") if isinstance(detail, dict) else str(detail)
        strength = float(detail.get("strength", 1.0) or 0.0) if isinstance(detail, dict) else 1.0
        value = {"pass": weight * strength, "unknown": 0.0, "fail": -weight * strength}.get(result, 0.0)
        group = str(guards[index].get("group") or guards[index].get("id") or index) if index < len(guards) else str(index)
        grouped.setdefault(group, []).append(value)
    components["guards"] = max(-60.0, min(60.0, sum(max(values) for values in grouped.values())))
    return components


def provider_score(provider: dict[str, Any], cursor: dict[str, Any], guard_details: list[Any]) -> float:
    return sum(provider_score_components(provider, cursor, guard_details).values())


def rank_provider_details(operation: dict[str, Any], cursor: dict[str, Any], guard_details: dict[str, list[Any]]) -> list[dict[str, Any]]:
    candidates = [item for item in operation.get("providers", []) if isinstance(item, dict) and provider_available(item, cursor)]
    fresh = int(cursor.get("total_actions", 0) or 0) == 0 and not cursor.get("attempt_counts")
    entries = {str(value) for value in operation.get("entry_providers", [])}
    if not entries:
        referenced = {str(value) for item in candidates for value in item.get("successors", [])}
        entries = {str(item.get("provider_id")) for item in candidates if item.get("successors") and str(item.get("provider_id")) not in referenced}
    if fresh and entries:
        entry_candidates = [item for item in candidates if str(item.get("provider_id") or "") in entries]
        if entry_candidates:
            candidates = entry_candidates
    ranked: list[dict[str, Any]] = []
    for provider in candidates:
        provider_id = str(provider.get("provider_id") or "")
        components = provider_score_components(provider, cursor, guard_details.get(provider_id, []), entry_bonus=fresh and provider_id in entries)
        ranked.append({"provider": provider, "provider_id": provider_id, "score": sum(components.values()), "components": components})
    return sorted(ranked, key=lambda item: (-item["score"], item["provider_id"]))


def rank_providers(operation: dict[str, Any], cursor: dict[str, Any], guard_details: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return [item["provider"] for item in rank_provider_details(operation, cursor, guard_details)]


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


def update_successor_context(
    cursor: dict[str, Any],
    provider: dict[str, Any],
    result: str,
    effect_details: list[dict[str, Any]] | None = None,
) -> None:
    has_evaluable_effect = any(
        isinstance(detail, dict) and detail.get("matched") is not None
        for detail in (effect_details or [])
    )
    usable_unverified = result == "unverified" and (
        not provider.get("effects") or (effect_details is not None and not has_evaluable_effect)
    )
    if result in {"confirmed", "transitioned"} or usable_unverified:
        cursor["successor_ids"] = [str(value) for value in provider.get("successors", [])]
    else:
        cursor["successor_ids"] = []
    cursor["updated_at"] = _now_iso()


def normalized_patch_hash(patch: Any) -> str:
    if not isinstance(patch, dict):
        return ""
    return json.dumps(deepcopy(patch), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
