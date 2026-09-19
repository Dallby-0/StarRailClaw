from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from state_machine.page_handler.reactive import normalize_operation, normalize_provider
from state_machine.time_utils import now_iso as _now_iso


HANDLER_SCHEMA_VERSION = "reactive_handler.v3.1"
TRACE_LIMIT = 40


def _normalize_intent_route(raw: Any, operation: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    kinds = raw.get("intent_kinds") if isinstance(raw.get("intent_kinds"), list) else [raw.get("intent_kind") or raw.get("kind")]
    phases = raw.get("phases") if isinstance(raw.get("phases"), list) else [raw.get("phase") or "*"]
    kinds = [str(value) for value in kinds if str(value or "")]
    if not kinds:
        return None
    return {
        "intent_kinds": kinds,
        "phases": [str(value) for value in phases if str(value or "")] or ["*"],
        "operation": str(raw.get("operation") or operation),
        "intent_effect": str(raw.get("intent_effect") or "advance"),
        "expected_event": str(raw.get("expected_event") or "") or None,
        "params": dict(raw.get("params") or {}),
    }


def _empty_handler() -> dict[str, Any]:
    now = _now_iso()
    return {
        "schema_version": HANDLER_SCHEMA_VERSION,
        "default_operation": None,
        "intent_routes": [],
        "operation_policies": {},
        "episode_trace": [],
        "created_at": now,
        "updated_at": now,
    }


def ensure_page_handler(meta: dict[str, Any]) -> dict[str, Any]:
    raw = meta.get("page_handler")
    if raw is None:
        raw = _empty_handler()
        meta["page_handler"] = raw
    if not isinstance(raw, dict):
        raise ValueError("page_handler must be an object")
    version = str(raw.get("schema_version") or "")
    if version != HANDLER_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported page handler schema {version or '<missing>'!r}; "
            "create a fresh state workspace for reactive_handler.v3.1"
        )
    policies = raw.get("operation_policies")
    if not isinstance(policies, dict):
        raise ValueError("reactive_handler.v3.1 operation_policies must be an object")
    normalized: dict[str, dict[str, Any]] = {}
    for index, (name, policy) in enumerate(policies.items(), 1):
        operation = normalize_operation(policy, index=index)
        if operation is None or operation["operation"] != str(name):
            raise ValueError(f"invalid reactive operation policy: {name!r}")
        normalized[str(name)] = operation
    raw["operation_policies"] = normalized
    if not isinstance(raw.get("intent_routes"), list):
        raw["intent_routes"] = []
    if not isinstance(raw.get("episode_trace"), list):
        raw["episode_trace"] = []
    raw["updated_at"] = _now_iso()
    return raw


def handler_from_bootstrap(bootstrap_operations: Any) -> dict[str, Any]:
    handler = _empty_handler()
    if not isinstance(bootstrap_operations, list):
        return handler
    defaults: list[str] = []
    for index, raw in enumerate(bootstrap_operations[:4], 1):
        operation = normalize_operation(raw, index=index)
        if operation is None:
            continue
        name = operation["operation"]
        handler["operation_policies"][name] = operation
        if bool(raw.get("is_default")):
            defaults.append(name)
        routes = raw.get("intent_routes") if isinstance(raw.get("intent_routes"), list) else []
        handler["intent_routes"].extend(route for item in routes if (route := _normalize_intent_route(item, name)) is not None)
    if len(defaults) == 1:
        name = defaults[0]
        operation = handler["operation_policies"][name]
        handler["default_operation"] = {
            "operation": name,
            "intent_scope": operation["intent_scope"],
            "intent_effect": operation["intent_effect"],
            "expected_event": operation["expected_event"],
        }
    elif len(handler["operation_policies"]) == 1:
        name, operation = next(iter(handler["operation_policies"].items()))
        handler["default_operation"] = {
            "operation": name,
            "intent_scope": operation["intent_scope"],
            "intent_effect": operation["intent_effect"],
            "expected_event": operation["expected_event"],
        }
    return handler


def _provider_signature(provider: dict[str, Any]) -> tuple[Any, ...]:
    return (str(provider.get("provider_id") or ""),)


def _provider_behavior_signature(provider: dict[str, Any]) -> str:
    locators = provider.get("locators") if isinstance(provider.get("locators"), list) else []
    point = next((item for item in locators if isinstance(item, dict) and item.get("type") == "point"), None)
    if point is not None:
        behavior: Any = {
            "type": "point",
            "x": point.get("x"),
            "y": point.get("y"),
            "coordinate_space": point.get("coordinate_space"),
        }
    else:
        behavior = [
            {key: value for key, value in item.items() if key not in {"source", "template_path"}}
            for item in locators
            if isinstance(item, dict)
        ]
    return json.dumps(behavior, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _click_region(provider: dict[str, Any]) -> list[int] | None:
    locator = next((item for item in provider.get("locators", []) if isinstance(item, dict) and item.get("type") == "click_region"), None)
    return locator.get("rect") if isinstance(locator, dict) and isinstance(locator.get("rect"), list) else None


def _region_overlap(left: list[int], right: list[int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    smaller = min((left[2] - left[0]) * (left[3] - left[1]), (right[2] - right[0]) * (right[3] - right[1]))
    return float(intersection) / float(max(1, smaller))


def _semantic_mergeable(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("operation_key") or "") != str(right.get("operation_key") or ""):
        return False
    left_rect, right_rect = _click_region(left), _click_region(right)
    if left_rect is None or right_rect is None or _region_overlap(left_rect, right_rect) < 0.8:
        return False
    for key in ("successors", "effects"):
        a, b = left.get(key) or [], right.get(key) or []
        if a and b and a != b:
            return False
    return True


def _merge_semantic_provider(previous: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(incoming)
    merged["provider_id"] = str(previous.get("provider_id") or incoming.get("provider_id") or "")
    merged["base_priority"] = max(int(previous.get("base_priority", 0) or 0), int(incoming.get("base_priority", 0) or 0))
    status_rank = {"disabled": -1, "proposed": 0, "canary": 1, "active": 2}
    merged["status"] = max((str(previous.get("status") or "proposed"), str(incoming.get("status") or "proposed")), key=lambda value: status_rank.get(value, -1))
    merged["success_count"] = int(previous.get("success_count", 0) or 0)
    merged["result_counts"] = dict(previous.get("result_counts") or {})
    merged["successful_visits"] = list(previous.get("successful_visits") or [])
    merged["created_at"] = previous.get("created_at", incoming.get("created_at"))
    for key in ("watches", "effects", "successors"):
        if not merged.get(key):
            merged[key] = deepcopy(previous.get(key) or [])
    locators = [deepcopy(item) for item in previous.get("locators", [])]
    for locator in incoming.get("locators", []):
        if locator.get("type") != "click_region":
            if locator not in locators:
                locators.append(deepcopy(locator))
            continue
        existing = next((item for item in locators if item.get("type") == "click_region"), None)
        if existing is None:
            locators.append(deepcopy(locator))
            continue
        known = {tuple(item.get("point", [])) for item in existing.get("candidate_points", [])}
        for candidate in locator.get("candidate_points", []):
            if tuple(candidate.get("point", [])) not in known:
                existing.setdefault("candidate_points", []).append(deepcopy(candidate))
        existing["candidate_points"] = existing.get("candidate_points", [])[:4]
    merged["locators"] = locators[:4]
    group = f"{merged.get('operation_key')}_variants"
    guards: list[dict[str, Any]] = []
    known_guards: set[str] = set()
    for guard in list(previous.get("guards", [])) + list(incoming.get("guards", [])):
        signature = json.dumps({key: value for key, value in guard.items() if key not in {"id", "status", "evidence", "group"}}, ensure_ascii=False, sort_keys=True)
        if signature in known_guards:
            continue
        item = deepcopy(guard)
        item["group"] = group
        guards.append(item)
        known_guards.add(signature)
    merged["guards"] = guards[:6]
    return merged


def merge_operation(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    if not isinstance(existing, dict):
        return deepcopy(incoming), {str(item.get("provider_id")) for item in incoming.get("providers", []) if isinstance(item, dict)}
    merged = deepcopy(existing)
    providers = merged.get("providers") if isinstance(merged.get("providers"), list) else []
    merged["providers"] = providers
    touched: set[str] = set()
    provider_id_map: dict[str, str] = {}
    for provider in incoming.get("providers", []):
        if not isinstance(provider, dict):
            continue
        incoming_id = str(provider.get("provider_id") or "")
        signature = _provider_signature(provider)
        previous = next(
            (
                item for item in providers
                if isinstance(item, dict)
                and (_provider_signature(item) == signature or _semantic_mergeable(item, provider))
            ),
            None,
        )
        if previous is not None:
            semantic_merge = _provider_signature(previous) != signature
            if semantic_merge:
                archive = merged.setdefault("provider_archive", [])
                archive.append({"reason": "semantic_region_merge", "merged_into": previous.get("provider_id"), "provider": deepcopy(provider)})
                del archive[:-8]
            provider = _merge_semantic_provider(previous, provider)
            providers.remove(previous)
        providers.append(deepcopy(provider))
        merged_id = str(provider.get("provider_id") or "")
        provider_id_map[incoming_id] = merged_id
        touched.add(merged_id)
    for key in ("intent_scope", "intent_effect", "expected_event", "budgets"):
        if incoming.get(key) is not None:
            merged[key] = deepcopy(incoming[key])
    if isinstance(incoming.get("entry_providers"), list):
        merged["entry_providers"] = [provider_id_map.get(str(value), str(value)) for value in incoming["entry_providers"]]
    return merged, touched


def merge_bootstrap_operations(handler: dict[str, Any], operations: Any) -> set[str]:
    touched: set[str] = set()
    if not isinstance(operations, list):
        return touched
    policies = handler["operation_policies"]
    for index, raw in enumerate(operations[:4], 1):
        operation = normalize_operation(raw, index=index)
        if operation is None:
            continue
        name = operation["operation"]
        policies[name], changed = merge_operation(policies.get(name), operation)
        touched.update(changed)
    handler["updated_at"] = _now_iso()
    return touched


def apply_handler_patch(handler: dict[str, Any], patch: Any, *, operation: str) -> set[str]:
    if not isinstance(patch, dict):
        return set()
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    handler["operation_policies"] = policies
    policy = policies.get(operation)
    touched: set[str] = set()
    raw_providers = patch.get("providers") if isinstance(patch.get("providers"), list) else []
    candidates = patch.get("generalization_candidates") if isinstance(patch.get("generalization_candidates"), list) else []
    incoming: list[dict[str, Any]] = []
    existing_ids = {
        str(item.get("provider_id") or "")
        for item in (policy.get("providers", []) if isinstance(policy, dict) else [])
        if isinstance(item, dict)
    }
    tagged = [(raw, False) for raw in raw_providers] + [(raw, True) for raw in candidates]
    for index, (raw, family_candidate) in enumerate(tagged, 1):
        candidate = dict(raw) if isinstance(raw, dict) else raw
        if isinstance(candidate, dict) and family_candidate:
            candidate["scope"] = "family"
            candidate["status"] = "proposed"
            base_id = str(candidate.get("provider_id") or "family_candidate")
            unique_id = base_id
            suffix = 2
            while unique_id in existing_ids:
                unique_id = f"{base_id}_family_{suffix}"
                suffix += 1
            candidate["provider_id"] = unique_id
        provider = normalize_provider(candidate, index=index)
        if provider is not None:
            incoming.append(provider)
            existing_ids.add(str(provider.get("provider_id") or ""))
    if not incoming:
        return set()
    if not isinstance(policy, dict):
        policy = {
            "operation": operation,
            "intent_scope": "intent_specific",
            "intent_effect": "advance",
            "expected_event": None,
            "providers": [],
            "budgets": {},
        }
        policies[operation] = policy
    merged, touched = merge_operation(policy, {**policy, "providers": incoming})
    policies[operation] = merged
    handler["updated_at"] = _now_iso()
    return touched


def materialize_provider_templates(handler: dict[str, Any], state_dir: Path, frame_rgb, vision, provider_ids: set[str]) -> None:
    for policy in handler.get("operation_policies", {}).values():
        if not isinstance(policy, dict):
            continue
        for provider in policy.get("providers", []):
            if not isinstance(provider, dict) or str(provider.get("provider_id")) not in provider_ids:
                continue
            for index, locator in enumerate(provider.get("locators", []), 1):
                if not isinstance(locator, dict) or locator.get("type") != "region_template" or locator.get("template_path"):
                    continue
                bbox = locator.get("template_bbox")
                if not isinstance(bbox, list):
                    continue
                path = state_dir / f"reactive_locator_{provider['provider_id']}_{index}.png"
                vision.save_template_from_rect(frame_rgb, bbox, path)
                locator["template_path"] = str(path)
            for collection in ("guards",):
                for index, guard in enumerate(provider.get(collection, []), 1):
                    if not isinstance(guard, dict) or guard.get("type") != "template" or guard.get("template_path"):
                        continue
                    bbox = guard.get("template_bbox")
                    if not isinstance(bbox, list):
                        continue
                    path = state_dir / f"reactive_hint_{provider['provider_id']}_{index}.png"
                    vision.save_template_from_rect(frame_rgb, bbox, path)
                    guard["template_path"] = str(path)
            for index, effect in enumerate(provider.get("effects", []), 1):
                probe = effect.get("probe") if isinstance(effect, dict) else None
                if not isinstance(probe, dict) or probe.get("type") != "template" or probe.get("template_path"):
                    continue
                bbox = probe.get("template_bbox")
                if not isinstance(bbox, list):
                    continue
                path = state_dir / f"reactive_effect_{provider['provider_id']}_{index}.png"
                vision.save_template_from_rect(frame_rgb, bbox, path)
                probe["template_path"] = str(path)


def append_episode(handler: dict[str, Any], episode: dict[str, Any]) -> None:
    trace = handler.get("episode_trace") if isinstance(handler.get("episode_trace"), list) else []
    handler["episode_trace"] = trace
    item = dict(episode)
    item.setdefault("created_at", _now_iso())
    trace.append(item)
    if len(trace) > TRACE_LIMIT:
        del trace[:-TRACE_LIMIT]
    handler["updated_at"] = _now_iso()


def record_provider_result(operation: dict[str, Any], provider_id: str, result: str, *, visit_id: str) -> None:
    provider = next((item for item in operation.get("providers", []) if isinstance(item, dict) and str(item.get("provider_id")) == provider_id), None)
    if not isinstance(provider, dict):
        return
    counts = provider.get("result_counts") if isinstance(provider.get("result_counts"), dict) else {}
    provider["result_counts"] = counts
    counts[result] = int(counts.get(result, 0) or 0) + 1
    if result in {"confirmed", "transitioned", "confirmed_by_successor"}:
        visits = provider.get("successful_visits") if isinstance(provider.get("successful_visits"), list) else []
        provider["successful_visits"] = visits
        if visit_id not in visits:
            visits.append(visit_id)
            del visits[:-8]
        provider["success_count"] = int(provider.get("success_count", 0) or 0) + 1
        if str(provider.get("scope")) == "family":
            provider["status"] = "active" if len(visits) >= 2 else "canary"
        else:
            provider["status"] = "active"
    provider["updated_at"] = _now_iso()


def handler_summary(handler: dict[str, Any], *, limit_trace: int = 8) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for name, policy in handler.get("operation_policies", {}).items():
        if not isinstance(policy, dict):
            continue
        policies[str(name)] = {
            "intent_scope": policy.get("intent_scope"),
            "intent_effect": policy.get("intent_effect"),
            "budgets": policy.get("budgets"),
            "entry_providers": policy.get("entry_providers", []),
            "providers": [
                {
                    "provider_id": item.get("provider_id"),
                    "scope": item.get("scope"),
                    "status": item.get("status"),
                    "base_priority": item.get("base_priority"),
                    "locators": item.get("locators", []),
                    "guards": item.get("guards", []),
                    "watches": item.get("watches", []),
                    "effects": item.get("effects", []),
                    "successors": item.get("successors", []),
                    "result_counts": item.get("result_counts", {}),
                }
                for item in policy.get("providers", []) if isinstance(item, dict)
            ],
            "provider_archive": policy.get("provider_archive", [])[-8:],
        }
    trace = handler.get("episode_trace") if isinstance(handler.get("episode_trace"), list) else []
    return {
        "schema_version": handler.get("schema_version"),
        "default_operation": handler.get("default_operation"),
        "intent_routes": handler.get("intent_routes", []),
        "operation_policies": policies,
        "recent_trace": trace[-limit_trace:],
    }
