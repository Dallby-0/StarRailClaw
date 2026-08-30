from __future__ import annotations

from typing import Any

from state_machine.time_utils import now_iso as _now_iso
from state_machine.page_handler.reactive import initial_cursor


NO_PROGRESS_LIMIT = 2
OPERATION_ATTEMPT_LIMIT = 6
OPERATION_REPAIR_LIMIT = 1
OPERATION_REVIEW_LIMIT = 2
DEFAULT_CONTINUATION_ACTIONS = 6
MAX_CONTINUATION_ACTIONS = 12


def _guard(runtime: dict[str, Any]) -> dict[str, Any]:
    value = runtime.get("no_progress_guard")
    if not isinstance(value, dict):
        value = {}
        runtime["no_progress_guard"] = value
    return value


def _key(visit_id: str, operation: str, strategy_id: str) -> str:
    return f"{visit_id}|{operation}|{strategy_id}"


def record_attempt(runtime: dict[str, Any], visit_id: str, operation: str) -> int:
    attempts = runtime.get("visit_operation_attempts")
    if not isinstance(attempts, dict):
        attempts = {}
        runtime["visit_operation_attempts"] = attempts
    key = f"{visit_id}|{operation}"
    attempts[key] = int(attempts.get(key, 0) or 0) + 1
    return int(attempts[key])


def operation_exhausted(runtime: dict[str, Any], visit_id: str, operation: str) -> bool:
    attempts = runtime.get("visit_operation_attempts")
    count = int(attempts.get(f"{visit_id}|{operation}", 0) or 0) if isinstance(attempts, dict) else 0
    return count >= OPERATION_ATTEMPT_LIMIT


def record_repair(runtime: dict[str, Any], visit_id: str, operation: str) -> int:
    repairs = runtime.get("visit_operation_repairs")
    if not isinstance(repairs, dict):
        repairs = {}
        runtime["visit_operation_repairs"] = repairs
    key = f"{visit_id}|{operation}"
    repairs[key] = int(repairs.get(key, 0) or 0) + 1
    return int(repairs[key])


def repair_exhausted(runtime: dict[str, Any], visit_id: str, operation: str) -> bool:
    repairs = runtime.get("visit_operation_repairs")
    count = int(repairs.get(f"{visit_id}|{operation}", 0) or 0) if isinstance(repairs, dict) else 0
    return count >= OPERATION_REPAIR_LIMIT


def record_review(runtime: dict[str, Any], visit_id: str, operation: str) -> int:
    reviews = runtime.get("visit_operation_reviews")
    if not isinstance(reviews, dict):
        reviews = {}
        runtime["visit_operation_reviews"] = reviews
    key = f"{visit_id}|{operation}"
    reviews[key] = int(reviews.get(key, 0) or 0) + 1
    return int(reviews[key])


def review_exhausted(runtime: dict[str, Any], visit_id: str, operation: str) -> bool:
    reviews = runtime.get("visit_operation_reviews")
    count = int(reviews.get(f"{visit_id}|{operation}", 0) or 0) if isinstance(reviews, dict) else 0
    return count >= OPERATION_REVIEW_LIMIT


def _continuations(runtime: dict[str, Any]) -> dict[str, Any]:
    value = runtime.get("visit_operation_continuations")
    if not isinstance(value, dict):
        value = {}
        runtime["visit_operation_continuations"] = value
    return value


def continuation_for(runtime: dict[str, Any], visit_id: str, operation: str) -> dict[str, Any] | None:
    entry = _continuations(runtime).get(f"{visit_id}|{operation}")
    if not isinstance(entry, dict) or int(entry.get("remaining_actions", 0) or 0) <= 0:
        return None
    return entry


def grant_continuation(
    runtime: dict[str, Any],
    visit_id: str,
    operation: str,
    strategy_id: str,
    *,
    max_additional_actions: int = DEFAULT_CONTINUATION_ACTIONS,
) -> dict[str, Any]:
    actions = max(1, min(int(max_additional_actions or DEFAULT_CONTINUATION_ACTIONS), MAX_CONTINUATION_ACTIONS))
    entry = {
        "visit_id": visit_id,
        "operation": operation,
        "strategy_id": strategy_id,
        "remaining_actions": actions,
        "granted_actions": actions,
        "updated_at": _now_iso(),
    }
    _continuations(runtime)[f"{visit_id}|{operation}"] = entry
    return entry


def consume_continuation(runtime: dict[str, Any], visit_id: str, operation: str, strategy_id: str) -> int | None:
    entry = continuation_for(runtime, visit_id, operation)
    if entry is None or str(entry.get("strategy_id") or "") != strategy_id:
        return None
    entry["remaining_actions"] = max(0, int(entry.get("remaining_actions", 0) or 0) - 1)
    entry["updated_at"] = _now_iso()
    return int(entry["remaining_actions"])


def clear_continuation(runtime: dict[str, Any], visit_id: str, operation: str) -> None:
    _continuations(runtime).pop(f"{visit_id}|{operation}", None)


def _reactive_cursors(runtime: dict[str, Any]) -> dict[str, Any]:
    value = runtime.get("visit_operation_reactive")
    if not isinstance(value, dict):
        value = {}
        runtime["visit_operation_reactive"] = value
    return value


def reactive_cursor_for(runtime: dict[str, Any], visit_id: str, operation: str) -> dict[str, Any]:
    key = f"{visit_id}|{operation}"
    cursors = _reactive_cursors(runtime)
    cursor = cursors.get(key)
    if not isinstance(cursor, dict):
        cursor = initial_cursor(visit_id=visit_id, operation=operation)
        cursors[key] = cursor
    return cursor


def clear_reactive_cursor(runtime: dict[str, Any], visit_id: str, operation: str) -> None:
    _reactive_cursors(runtime).pop(f"{visit_id}|{operation}", None)


def record_no_progress(runtime: dict[str, Any], visit_id: str, operation: str, strategy_id: str, result: str) -> int:
    guard = _guard(runtime)
    key = _key(visit_id, operation, strategy_id)
    entry = guard.get(key) if isinstance(guard.get(key), dict) else {}
    entry.update({
        "visit_id": visit_id,
        "operation": operation,
        "strategy_id": strategy_id,
        "count": int(entry.get("count", 0) or 0) + 1,
        "last_result": result,
        "updated_at": _now_iso(),
    })
    guard[key] = entry
    return int(entry["count"])


def strategy_blocked_for_visit(runtime: dict[str, Any], visit_id: str, operation: str, strategy_id: str) -> bool:
    entry = _guard(runtime).get(_key(visit_id, operation, strategy_id))
    return isinstance(entry, dict) and int(entry.get("count", 0) or 0) >= NO_PROGRESS_LIMIT


def blocked_strategies(runtime: dict[str, Any], visit_id: str, operation: str) -> set[str]:
    prefix = f"{visit_id}|{operation}|"
    return {
        key[len(prefix):]
        for key, entry in _guard(runtime).items()
        if key.startswith(prefix) and isinstance(entry, dict) and int(entry.get("count", 0) or 0) >= NO_PROGRESS_LIMIT
    }


def clear_visit_progress(runtime: dict[str, Any], visit_id: str) -> None:
    guard = _guard(runtime)
    for key in [key for key in guard if key.startswith(f"{visit_id}|")]:
        del guard[key]
    attempts = runtime.get("visit_operation_attempts")
    if isinstance(attempts, dict):
        for key in [key for key in attempts if key.startswith(f"{visit_id}|")]:
            del attempts[key]
    repairs = runtime.get("visit_operation_repairs")
    if isinstance(repairs, dict):
        for key in [key for key in repairs if key.startswith(f"{visit_id}|")]:
            del repairs[key]
    reviews = runtime.get("visit_operation_reviews")
    if isinstance(reviews, dict):
        for key in [key for key in reviews if key.startswith(f"{visit_id}|")]:
            del reviews[key]
    continuations = runtime.get("visit_operation_continuations")
    if isinstance(continuations, dict):
        for key in [key for key in continuations if key.startswith(f"{visit_id}|")]:
            del continuations[key]
    exploration = runtime.get("visit_operation_exploration")
    if isinstance(exploration, dict):
        for key in [key for key in exploration if key.startswith(f"{visit_id}|")]:
            del exploration[key]
    reactive = runtime.get("visit_operation_reactive")
    if isinstance(reactive, dict):
        for key in [key for key in reactive if key.startswith(f"{visit_id}|")]:
            del reactive[key]


def reset_run_local_progress(runtime: dict[str, Any]) -> None:
    """Discard visit-scoped controller state when a new process run starts.

    These values describe attempts made against a concrete on-screen page
    instance. Reusing them after a process restart can make the first matched
    page appear exhausted before the new run performs any action.
    """
    runtime["page_visit"] = None
    for key in (
        "no_progress_guard",
        "visit_operation_attempts",
        "visit_operation_repairs",
        "visit_operation_reviews",
        "visit_operation_continuations",
        "visit_operation_exploration",
        "visit_operation_reactive",
    ):
        runtime[key] = {}
