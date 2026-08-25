from __future__ import annotations

from typing import Any

from state_machine.time_utils import now_iso as _now_iso


NO_PROGRESS_LIMIT = 2
OPERATION_ATTEMPT_LIMIT = 6
OPERATION_REPAIR_LIMIT = 1


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
