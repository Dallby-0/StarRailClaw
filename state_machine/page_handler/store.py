from __future__ import annotations

from pathlib import Path
from typing import Any

from state_machine.time_utils import now_iso as _now_iso

HANDLER_SCHEMA_VERSION = "progressive_handler.v1"
TRACE_LIMIT = 40
STRATEGY_RESULTS = {"verified_success", "partial_progress", "no_effect", "wrong_transition", "unsafe_effect", "state_mismatch", "transient_unknown"}


def _slug(raw: Any, fallback: str) -> str:
    text = str(raw or "").strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or fallback


def ensure_page_handler(meta: dict[str, Any]) -> dict[str, Any]:
    handler = meta.get("page_handler")
    if not isinstance(handler, dict):
        handler = {}
        meta["page_handler"] = handler
    handler.setdefault("schema_version", HANDLER_SCHEMA_VERSION)
    handler.setdefault("default_operation", None)
    if not isinstance(handler.get("intent_routes"), list):
        handler["intent_routes"] = []
    if not isinstance(handler.get("operation_policies"), dict):
        handler["operation_policies"] = {}
    if not isinstance(handler.get("episode_trace"), list):
        handler["episode_trace"] = []
    handler["updated_at"] = _now_iso()
    return handler


def _normalize_resolver(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    rtype = str(raw.get("type") or "fixed_point").strip()
    if rtype == "click":
        rtype = "fixed_point"
    if rtype == "fixed_point":
        try:
            return {"type": rtype, "x": int(raw.get("x")), "y": int(raw.get("y"))}
        except Exception:
            return None
    if rtype == "run_preset":
        name = str(raw.get("name") or "").strip()
        return {"type": rtype, "name": name} if name else None
    if rtype == "region_template":
        return {
            "type": rtype,
            "template_path": str(raw.get("template_path") or ""),
            "search_rect": raw.get("search_rect"),
            "template_bbox": raw.get("template_bbox") or raw.get("bbox"),
            "threshold": float(raw.get("threshold", 0.82)),
        }
    return None


def normalize_strategy(raw: Any, *, operation: str, index: int = 1) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    strategy_id = _slug(raw.get("strategy_id") or raw.get("id"), f"{operation}_strategy_{index}")
    raw_steps = raw.get("steps") if isinstance(raw.get("steps"), list) else [raw]
    steps: list[dict[str, Any]] = []
    for step_idx, step in enumerate(raw_steps[:2], start=1):
        if not isinstance(step, dict):
            continue
        resolver = _normalize_resolver(step.get("resolver") or step.get("action") or step)
        if resolver is None:
            continue
        steps.append({
            "step_id": _slug(step.get("step_id"), f"step_{step_idx}"),
            "resolver": resolver,
            "preconditions": dict(step.get("preconditions") or {}),
            "expected_after": dict(step.get("expected_after") or {}),
            "emits_on_success": dict(step.get("emits_on_success") or {}),
            "brief": str(step.get("brief") or step.get("label") or ""),
        })
    if not steps:
        return None
    status = str(raw.get("status", "proposed"))
    if status not in {"proposed", "probation", "active", "degraded", "quarantined"}:
        status = "proposed"
    return {
        "strategy_id": strategy_id,
        "operation": operation,
        "level": max(0, int(raw.get("level", 0) or 0)),
        "status": status,
        "safety": str(raw.get("safety", "low_risk")),
        "steps": steps,
        "success_count": int(raw.get("success_count", 0) or 0),
        "fail_count": int(raw.get("fail_count", 0) or 0),
        "result_counts": dict(raw.get("result_counts") or {}),
        "created_at": str(raw.get("created_at") or _now_iso()),
        "updated_at": _now_iso(),
    }


def normalize_operation(raw: Any, *, index: int = 1) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    operation = _slug(raw.get("operation"), f"operation_{index}")
    strategies_raw = raw.get("strategies") if isinstance(raw.get("strategies"), list) else []
    if not strategies_raw and isinstance(raw.get("steps"), list):
        strategies_raw = [{"strategy_id": f"{operation}_fixed_v1", "level": 0, "status": "proposed", "steps": raw["steps"], "safety": raw.get("safety", "low_risk")}]
    strategies = []
    for strategy_idx, item in enumerate(strategies_raw, start=1):
        if not isinstance(item, dict):
            continue
        candidate = dict(item)
        candidate.setdefault("safety", raw.get("safety", "low_risk"))
        strategy = normalize_strategy(candidate, operation=operation, index=strategy_idx)
        if strategy is not None:
            strategies.append(strategy)
    if not strategies:
        return None
    return {
        "operation": operation,
        "intent_scope": str(raw.get("intent_scope", "intent_specific")),
        "intent_effect": str(raw.get("intent_effect", "none")),
        "safety": str(raw.get("safety", "low_risk")),
        "expected_event": str(raw.get("expected_event") or "") or None,
        "strategies": strategies,
    }


def handler_from_bootstrap(bootstrap_operations: Any) -> dict[str, Any]:
    handler = {"schema_version": HANDLER_SCHEMA_VERSION, "default_operation": None, "intent_routes": [], "operation_policies": {}, "episode_trace": [], "created_at": _now_iso(), "updated_at": _now_iso()}
    if not isinstance(bootstrap_operations, list):
        return handler
    for idx, raw in enumerate(bootstrap_operations[:4], start=1):
        operation = normalize_operation(raw, index=idx)
        if operation is None:
            continue
        name = operation["operation"]
        handler["operation_policies"][name] = operation
        if bool(raw.get("is_default", False)):
            handler["default_operation"] = {"operation": name, "intent_scope": operation["intent_scope"], "intent_effect": operation["intent_effect"], "safety": operation["safety"], "expected_event": operation["expected_event"]}
        routes = raw.get("intent_routes") if isinstance(raw.get("intent_routes"), list) else []
        handler["intent_routes"].extend(route for route in routes if isinstance(route, dict))
    return handler


def handler_summary(handler: dict[str, Any], *, limit_trace: int = 8) -> dict[str, Any]:
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    compact: dict[str, Any] = {}
    for operation, policy in policies.items():
        if not isinstance(policy, dict):
            continue
        compact[str(operation)] = {
            "intent_scope": policy.get("intent_scope"), "intent_effect": policy.get("intent_effect"), "safety": policy.get("safety"),
            "strategies": [{"strategy_id": item.get("strategy_id"), "level": item.get("level"), "status": item.get("status"), "success_count": item.get("success_count", 0), "fail_count": item.get("fail_count", 0)} for item in policy.get("strategies", []) if isinstance(item, dict)],
        }
    trace = handler.get("episode_trace") if isinstance(handler.get("episode_trace"), list) else []
    return {"schema_version": handler.get("schema_version"), "default_operation": handler.get("default_operation"), "intent_routes": handler.get("intent_routes", []), "operation_policies": compact, "recent_trace": trace[-limit_trace:]}


def select_strategy(handler: dict[str, Any], operation: str, *, excluded: set[str] | None = None) -> dict[str, Any] | None:
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    policy = policies.get(operation)
    if not isinstance(policy, dict):
        return None
    excluded = excluded or set()
    candidates = [item for item in policy.get("strategies", []) if isinstance(item, dict) and str(item.get("strategy_id")) not in excluded and str(item.get("status", "proposed")) in {"proposed", "probation", "active"}]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (int(item.get("level", 0) or 0), 0 if item.get("status") == "active" else 1, -int(item.get("success_count", 0) or 0)))
    return candidates[0]


def apply_handler_patch(handler: dict[str, Any], patch: Any) -> set[str]:
    touched: set[str] = set()
    if not isinstance(patch, dict):
        return touched
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    handler["operation_policies"] = policies
    for op_idx, raw in enumerate(patch.get("operations") if isinstance(patch.get("operations"), list) else [], start=1):
        operation = normalize_operation(raw, index=op_idx)
        if operation is None:
            continue
        name = operation["operation"]
        existing = policies.get(name)
        if not isinstance(existing, dict):
            policies[name] = operation
            touched.update(str(item["strategy_id"]) for item in operation["strategies"])
            continue
        by_id = {str(item.get("strategy_id")): item for item in existing.get("strategies", []) if isinstance(item, dict)}
        for strategy in operation["strategies"]:
            old = by_id.get(str(strategy["strategy_id"]))
            if isinstance(old, dict):
                strategy.update({"success_count": int(old.get("success_count", 0) or 0), "fail_count": int(old.get("fail_count", 0) or 0), "result_counts": dict(old.get("result_counts") or {}), "created_at": old.get("created_at", strategy["created_at"])})
                existing["strategies"].remove(old)
            existing.setdefault("strategies", []).append(strategy)
            touched.add(str(strategy["strategy_id"]))
    if isinstance(patch.get("default_operation"), dict):
        handler["default_operation"] = dict(patch["default_operation"])
    if isinstance(patch.get("intent_routes"), list):
        handler["intent_routes"] = [item for item in patch["intent_routes"] if isinstance(item, dict)]
    handler["updated_at"] = _now_iso()
    return touched


def materialize_strategy_templates(handler: dict[str, Any], state_dir: Path, frame_rgb, vision, strategy_ids: set[str]) -> None:
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    for policy in policies.values():
        if not isinstance(policy, dict):
            continue
        for strategy in policy.get("strategies", []):
            if not isinstance(strategy, dict) or str(strategy.get("strategy_id")) not in strategy_ids:
                continue
            for step_idx, step in enumerate(strategy.get("steps", []), start=1):
                resolver = step.get("resolver") if isinstance(step, dict) and isinstance(step.get("resolver"), dict) else None
                if not isinstance(resolver, dict) or resolver.get("type") != "region_template" or resolver.get("template_path"):
                    continue
                bbox = resolver.get("template_bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                path = state_dir / f"action_template_{strategy['strategy_id']}_{step_idx}.png"
                vision.save_template_from_rect(frame_rgb, [int(v) for v in bbox], path)
                resolver["template_path"] = str(path)


def append_episode(handler: dict[str, Any], episode: dict[str, Any]) -> None:
    trace = handler.get("episode_trace") if isinstance(handler.get("episode_trace"), list) else []
    handler["episode_trace"] = trace
    item = dict(episode)
    item.setdefault("created_at", _now_iso())
    trace.append(item)
    if len(trace) > TRACE_LIMIT:
        del trace[:-TRACE_LIMIT]
    handler["updated_at"] = _now_iso()


def mark_strategy_result(handler: dict[str, Any], operation: str, strategy_id: str, result: str) -> None:
    if result not in STRATEGY_RESULTS:
        raise ValueError(f"unsupported strategy result: {result}")
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    policy = policies.get(operation)
    if not isinstance(policy, dict):
        return
    strategy = next((item for item in policy.get("strategies", []) if isinstance(item, dict) and str(item.get("strategy_id")) == strategy_id), None)
    if not isinstance(strategy, dict):
        return
    counts = strategy.get("result_counts") if isinstance(strategy.get("result_counts"), dict) else {}
    strategy["result_counts"] = counts
    counts[result] = int(counts.get(result, 0) or 0) + 1
    if result in {"verified_success", "partial_progress"}:
        strategy["success_count"] = int(strategy.get("success_count", 0) or 0) + 1
        strategy["status"] = "active"
    elif result in {"no_effect", "wrong_transition", "unsafe_effect"}:
        strategy["fail_count"] = int(strategy.get("fail_count", 0) or 0) + 1
        strategy["status"] = "quarantined" if result == "unsafe_effect" else "degraded"
    strategy["updated_at"] = _now_iso()


def promote_operation_to_default(handler: dict[str, Any], operation: str) -> bool:
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    policy = policies.get(operation)
    if not isinstance(policy, dict):
        return False
    if str(policy.get("intent_scope")) != "intent_invariant" or str(policy.get("safety")) not in {"low_risk", "reversible"}:
        return False
    handler["default_operation"] = {
        "operation": operation,
        "intent_scope": "intent_invariant",
        "intent_effect": str(policy.get("intent_effect", "none")),
        "safety": str(policy.get("safety")),
        "expected_event": policy.get("expected_event"),
    }
    handler["updated_at"] = _now_iso()
    return True
