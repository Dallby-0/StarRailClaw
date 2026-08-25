from __future__ import annotations

from typing import Any

from state_machine.time_utils import now_iso as _now_iso


def current_visit(runtime: dict[str, Any], state_id: str | None = None) -> dict[str, Any] | None:
    visit = runtime.get("page_visit")
    if not isinstance(visit, dict):
        return None
    if state_id is not None and str(visit.get("state_id")) != str(state_id):
        return None
    return visit


def start_new_visit(runtime: dict[str, Any], state_id: str, *, reason: str) -> dict[str, Any]:
    seq = int(runtime.get("page_visit_seq", 0) or 0) + 1
    runtime["page_visit_seq"] = seq
    visit = {
        "state_id": str(state_id),
        "visit_seq": seq,
        "visit_id": f"{state_id}:{seq}",
        "entered_at": _now_iso(),
        "reason": reason,
    }
    runtime["page_visit"] = visit
    return visit


def ensure_page_visit(runtime: dict[str, Any], state_id: str) -> dict[str, Any]:
    existing = current_visit(runtime, state_id)
    return existing if existing is not None else start_new_visit(runtime, state_id, reason="state_selected")
