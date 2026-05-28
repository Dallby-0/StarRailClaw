from __future__ import annotations

import uuid
from typing import Any

from state_machine.constants import FORCE_RESET_AT, HARD_LIMIT, MID_LIMIT, SOFT_LIMIT


def _maybe_rotate_session(runtime: dict[str, Any]) -> bool:
    turns = int(runtime.get("llm_turn_count", 0))
    tier = str(runtime.get("session_tier", "soft"))
    if turns >= FORCE_RESET_AT:
        runtime["pending_refresh"] = True
        return True
    if tier == "soft" and turns >= SOFT_LIMIT and runtime.get("last_transition_ok", False):
        runtime["session_tier"] = "mid"
        runtime["pending_refresh"] = True
        return True
    if tier == "mid" and turns >= MID_LIMIT and runtime.get("last_transition_ok", False):
        runtime["session_tier"] = "hard"
        runtime["pending_refresh"] = True
        return True
    if tier == "hard" and turns >= HARD_LIMIT:
        runtime["pending_refresh"] = True
        return True
    return False


def _refresh_session_if_needed(runtime: dict[str, Any], base_session_id: str) -> str:
    if not runtime.get("llm_session_id") or runtime.get("pending_refresh", False):
        runtime["llm_session_id"] = f"{base_session_id}-fsm-{uuid.uuid4().hex[:8]}"
        runtime["llm_turn_count"] = 0
        runtime["pending_refresh"] = False
        runtime["last_transition_ok"] = False
    return str(runtime["llm_session_id"])
