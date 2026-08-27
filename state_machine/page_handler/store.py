from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from typing import Any

from state_machine.time_utils import now_iso as _now_iso

HANDLER_SCHEMA_VERSION = "progressive_handler.v1"
TRACE_LIMIT = 40
STRATEGY_RESULTS = {"verified_success", "verified_reentry", "partial_progress", "progress_unknown", "no_effect", "wrong_transition", "unsafe_effect", "state_mismatch", "transient_unknown"}


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
    _migrate_legacy_partial_reviews(handler)
    _fold_single_exit_sibling_into_default(handler)
    handler["updated_at"] = _now_iso()
    return handler


def _migrate_legacy_partial_reviews(handler: dict[str, Any]) -> set[str]:
    """Correct strategies degraded by the old partial-review result mapping.

    Older runtimes stored an LLM `partial_needs_continue` verdict as
    `no_effect`, then degraded the strategy that had actually advanced the
    page. Only episodes carrying that explicit semantic verdict are corrected;
    ordinary degraded strategies remain untouched.
    """
    trace = handler.get("episode_trace") if isinstance(handler.get("episode_trace"), list) else []
    affected: set[tuple[str, str]] = set()
    for episode in trace:
        if not isinstance(episode, dict) or str(episode.get("result") or "") != "no_effect":
            continue
        review = episode.get("same_state_review")
        if not isinstance(review, dict) or str(review.get("verdict") or "") != "partial_needs_continue":
            continue
        command = episode.get("command") if isinstance(episode.get("command"), dict) else {}
        operation = str(command.get("operation") or "")
        strategy_id = str(episode.get("strategy_id") or "")
        if operation and strategy_id:
            affected.add((operation, strategy_id))

    if not affected:
        return set()
    migrated: set[str] = set()
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    for operation, strategy_id in affected:
        policy = policies.get(operation)
        if not isinstance(policy, dict):
            continue
        strategy = next(
            (item for item in policy.get("strategies", []) if isinstance(item, dict) and str(item.get("strategy_id") or "") == strategy_id),
            None,
        )
        if not isinstance(strategy, dict) or str(strategy.get("status") or "") != "degraded":
            continue
        counts = strategy.get("result_counts") if isinstance(strategy.get("result_counts"), dict) else {}
        no_effect_count = int(counts.get("no_effect", 0) or 0)
        if no_effect_count > 0:
            counts["no_effect"] = no_effect_count - 1
            if counts["no_effect"] <= 0:
                counts.pop("no_effect", None)
        counts["partial_progress"] = int(counts.get("partial_progress", 0) or 0) + 1
        strategy["result_counts"] = counts
        strategy["fail_count"] = max(0, int(strategy.get("fail_count", 0) or 0) - 1)
        strategy["status"] = "active" if int(strategy.get("success_count", 0) or 0) > 0 else "proposed"
        strategy["legacy_partial_review_migrated"] = True
        strategy["updated_at"] = _now_iso()
        migrated.add(strategy_id)

    if migrated:
        handler["updated_at"] = _now_iso()
    return migrated


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
            "expected_after": dict(step.get("expected_after") or {"state_relation": "may_leave", "reentry_policy": "forbid"}),
            "emits_on_success": dict(step.get("emits_on_success") or {}),
            "brief": str(step.get("brief") or step.get("label") or ""),
        })
    if not steps:
        return None
    status = str(raw.get("status", "proposed"))
    if status not in {"proposed", "active", "degraded", "quarantined"}:
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
        # Optional local recovery profiles.  They are intentionally kept
        # outside learned strategies so exploratory probing cannot poison the
        # strategy ranking or success counts.
        "exploration": [dict(item) for item in raw.get("exploration", []) if isinstance(item, dict)],
        "allow_exploration": bool(raw.get("allow_exploration", False)),
        "strategies": strategies,
    }


def _normalize_intent_route(raw: Any, operation: str) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    kinds = raw.get("intent_kinds")
    if not isinstance(kinds, list):
        kind = raw.get("intent_kind") or raw.get("kind")
        kinds = [kind] if kind else []
    phases = raw.get("phases")
    if not isinstance(phases, list):
        phase = raw.get("phase")
        phases = [phase] if phase else ["*"]
    if not kinds:
        return None
    return {
        "intent_kinds": [str(value) for value in kinds if str(value)],
        "phases": [str(value) for value in phases if str(value)] or ["*"],
        "operation": str(raw.get("operation") or operation),
        "intent_effect": str(raw.get("intent_effect", "advance")),
        "expected_event": str(raw.get("expected_event") or "") or None,
        "params": dict(raw.get("params") or {}),
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
        handler["intent_routes"].extend(route for item in routes if (route := _normalize_intent_route(item, name)) is not None)
    # A single low-risk bootstrap operation is an unambiguous answer to the
    # combined "identify + advance" request.  Treat it as the default even if
    # the model omitted is_default or emitted a contradictory intent_scope.
    # Multiple choices and commit/destructive operations still require an
    # explicit command/intent and therefore remain unresolved.
    if handler["default_operation"] is None and len(handler["operation_policies"]) == 1:
        only = next(iter(handler["operation_policies"].values()))
        if isinstance(only, dict) and str(only.get("safety")) in {"low_risk", "reversible"}:
            handler["default_operation"] = {
                "operation": str(only["operation"]),
                "intent_scope": str(only.get("intent_scope", "intent_specific")),
                "intent_effect": str(only.get("intent_effect", "none")),
                "safety": str(only.get("safety")),
                "expected_event": only.get("expected_event"),
            }
    _fold_single_exit_sibling_into_default(handler)
    return handler


def _fold_single_exit_sibling_into_default(handler: dict[str, Any]) -> bool:
    """Normalize a model-split `select` + `confirm` into one two-step strategy."""
    default = handler.get("default_operation") if isinstance(handler.get("default_operation"), dict) else None
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    if not isinstance(default, dict):
        return False
    default_name = str(default.get("operation") or "")
    current = policies.get(default_name)
    if not isinstance(current, dict):
        return False
    current_strategies = [item for item in current.get("strategies", []) if isinstance(item, dict)]
    if not current_strategies or any(len(item.get("steps", [])) != 1 for item in current_strategies):
        return False
    if not all(
        str(item["steps"][0].get("expected_after", {}).get("state_relation")) == "must_remain"
        for item in current_strategies
    ):
        return False

    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for name, policy in policies.items():
        if name == default_name or not isinstance(policy, dict):
            continue
        for strategy in policy.get("strategies", []):
            if not isinstance(strategy, dict):
                continue
            steps = strategy.get("steps") if isinstance(strategy.get("steps"), list) else []
            if len(steps) == 1 and str(steps[0].get("expected_after", {}).get("state_relation")) == "must_leave":
                candidates.append((policy, strategy))
    if len(candidates) != 1:
        return False

    continuation_policy, continuation_strategy = candidates[0]
    continuation_step = continuation_strategy["steps"][0]
    safety = str(current.get("safety", "reversible"))
    for strategy in current_strategies:
        strategy["steps"].append(deepcopy(continuation_step))
        strategy["strategy_id"] = _slug(f"{strategy.get('strategy_id')}_and_{continuation_strategy.get('strategy_id')}", f"{default_name}_two_step")
        strategy["safety"] = safety
    current["expected_event"] = continuation_policy.get("expected_event") or current.get("expected_event")
    default["expected_event"] = current.get("expected_event")
    handler["updated_at"] = _now_iso()
    return True


def handler_summary(handler: dict[str, Any], *, limit_trace: int = 8) -> dict[str, Any]:
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    compact: dict[str, Any] = {}
    for operation, policy in policies.items():
        if not isinstance(policy, dict):
            continue
        compact[str(operation)] = {
            "intent_scope": policy.get("intent_scope"), "intent_effect": policy.get("intent_effect"), "safety": policy.get("safety"),
            "exploration": [
                {"id": item.get("id"), "kind": item.get("kind")}
                for item in policy.get("exploration", [])
                if isinstance(item, dict)
            ],
            "strategies": [{"strategy_id": item.get("strategy_id"), "level": item.get("level"), "status": item.get("status"), "success_count": item.get("success_count", 0), "fail_count": item.get("fail_count", 0)} for item in policy.get("strategies", []) if isinstance(item, dict)],
        }
    trace = handler.get("episode_trace") if isinstance(handler.get("episode_trace"), list) else []
    return {"schema_version": handler.get("schema_version"), "default_operation": handler.get("default_operation"), "intent_routes": handler.get("intent_routes", []), "operation_policies": compact, "recent_trace": trace[-limit_trace:]}


def continuation_patch_from_sibling(handler: dict[str, Any], operation: str) -> dict[str, Any] | None:
    """Reuse an already learned sibling operation that exits the current page.

    This repairs a common bootstrap modeling error where the model emits
    `select` and `confirm` as parallel operations even though they are ordered
    steps of one default page advance operation.
    """
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    current = policies.get(operation) if isinstance(policies.get(operation), dict) else {}
    for sibling_name, policy in policies.items():
        if sibling_name == operation or not isinstance(policy, dict):
            continue
        for strategy in policy.get("strategies", []):
            if not isinstance(strategy, dict):
                continue
            steps = strategy.get("steps") if isinstance(strategy.get("steps"), list) else []
            if not steps or not any(
                isinstance(step, dict)
                and isinstance(step.get("expected_after"), dict)
                and str(step["expected_after"].get("state_relation")) == "must_leave"
                for step in steps
            ):
                continue
            copied = deepcopy(strategy)
            copied["strategy_id"] = _slug(f"{operation}_continue_{strategy.get('strategy_id')}", f"{operation}_continuation")
            copied["operation"] = operation
            copied["level"] = max(1, int(copied.get("level", 0) or 0) + 1)
            copied["status"] = "proposed"
            # The sibling is being folded into an already authorized default
            # page advance operation. Keep the default operation's safety
            # boundary instead of importing a model's overly broad "commit"
            # label for an ordinary card-confirm button.
            copied["safety"] = str(current.get("safety", "reversible"))
            copied.pop("success_count", None)
            copied.pop("fail_count", None)
            copied.pop("result_counts", None)
            return {
                "operations": [{
                    "operation": operation,
                    "intent_scope": str(current.get("intent_scope", "intent_specific")),
                    "intent_effect": str(current.get("intent_effect", "advance")),
                    "safety": str(current.get("safety", "reversible")),
                    "expected_event": policy.get("expected_event") or current.get("expected_event") or "",
                    "strategies": [copied],
                }]
            }
    return None


def select_strategy(handler: dict[str, Any], operation: str, *, excluded: set[str] | None = None, preferred_id: str | None = None) -> dict[str, Any] | None:
    policies = handler.get("operation_policies") if isinstance(handler.get("operation_policies"), dict) else {}
    policy = policies.get(operation)
    if not isinstance(policy, dict):
        return None
    excluded = excluded or set()
    candidates = [item for item in policy.get("strategies", []) if isinstance(item, dict) and str(item.get("strategy_id")) not in excluded and str(item.get("status", "proposed")) in {"proposed", "active"}]
    if not candidates:
        return None
    if preferred_id:
        preferred = next((item for item in candidates if str(item.get("strategy_id") or "") == preferred_id), None)
        if preferred is not None:
            return preferred
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
        normalized_routes = []
        for item in patch["intent_routes"]:
            operation = str(item.get("operation") or "") if isinstance(item, dict) else ""
            route = _normalize_intent_route(item, operation)
            if route is not None:
                normalized_routes.append(route)
        handler["intent_routes"] = normalized_routes
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
    if result in {"verified_success", "verified_reentry"}:
        strategy["success_count"] = int(strategy.get("success_count", 0) or 0) + 1
        strategy["status"] = "active"
    elif result == "partial_progress":
        # An intermediate step is not evidence that the complete strategy
        # works.  In particular, do not promote a multi-step strategy whose
        # confirm/exit step has never succeeded.
        pass
    elif result == "progress_unknown":
        # A bounded continuation lease permits this action even when the local
        # observer cannot prove page-local progress. It is neither success
        # evidence nor a reason to degrade the strategy.
        pass
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
