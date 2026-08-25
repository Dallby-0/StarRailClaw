from __future__ import annotations

from typing import Any

from state_machine.intent import reduce_intent_event
from state_machine.page_handler.store import append_episode, ensure_page_handler, mark_strategy_result
from state_machine.progress_guard import clear_visit_progress, record_no_progress
from state_machine.time_utils import now_iso as _now_iso
from state_machine.visit import start_new_visit


def _save_runtime_value(runtime: dict[str, Any]) -> None:
    from state_machine.io import _save_runtime

    _save_runtime(runtime)


def classify_pending_settlement(pending: dict[str, Any], resolved_state_id: str) -> str:
    source_state = str(pending.get("state_id") or "")
    expected = pending.get("expected_after") if isinstance(pending.get("expected_after"), dict) else {}
    departure = str(pending.get("departure_evidence") or "weak")
    relation = str(expected.get("state_relation") or "may_leave")
    reentry_policy = str(expected.get("reentry_policy") or "forbid")
    if resolved_state_id == source_state:
        return "verified_reentry" if departure == "confirmed" and reentry_policy == "new_visit" else "no_effect"
    if relation == "must_remain":
        return "wrong_transition"
    return "verified_success"


def settle_pending_operation(runtime: dict[str, Any], resolved_state_id: str) -> dict[str, Any] | None:
    pending = runtime.get("pending_operation")
    if not isinstance(pending, dict) or pending.get("status") != "awaiting_resolution":
        return None
    source_state = str(pending.get("state_id") or "")
    visit_id = str(pending.get("visit_id") or "")
    operation = str(pending.get("operation") or "")
    strategy_id = str(pending.get("strategy_id") or "")
    departure = str(pending.get("departure_evidence") or "weak")
    result = classify_pending_settlement(pending, resolved_state_id)

    state_path_raw = str(pending.get("state_dir") or "")
    if state_path_raw:
        from pathlib import Path
        from state_machine.io import _load_json, _save_json

        state_path = Path(state_path_raw) / "state.json"
        if state_path.exists():
            state_meta = _load_json(state_path)
            handler = ensure_page_handler(state_meta)
            mark_strategy_result(handler, operation, strategy_id, result)
            append_episode(handler, {
                "state_id": source_state,
                "visit_id": visit_id,
                "operation": operation,
                "strategy_id": strategy_id,
                "step_id": pending.get("step_id"),
                "result": result,
                "settled_after_resolution": True,
                "resolved_state_id": resolved_state_id,
                "departure_evidence": departure,
            })
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)

    if result in {"no_effect", "wrong_transition"}:
        count = record_no_progress(runtime, visit_id, operation, strategy_id, result)
    else:
        count = 0
        clear_visit_progress(runtime, visit_id)
        event = pending.get("success_event")
        reduce_intent_event(runtime, event if isinstance(event, dict) else None)

    if result == "verified_reentry":
        new_visit = start_new_visit(runtime, source_state, reason="settled_reentry")
    elif result == "verified_success":
        new_visit = start_new_visit(runtime, resolved_state_id, reason="settled_transition")
    else:
        new_visit = None

    runtime["pending_operation"] = None
    _save_runtime_value(runtime)
    return {
        "result": result,
        "source_state_id": source_state,
        "resolved_state_id": resolved_state_id,
        "visit_id": visit_id,
        "new_visit_id": new_visit.get("visit_id") if isinstance(new_visit, dict) else None,
        "operation": operation,
        "strategy_id": strategy_id,
        "no_progress_count": count,
        "allow_self_edge": result == "verified_reentry",
    }
