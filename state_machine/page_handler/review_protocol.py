from __future__ import annotations

import json
from typing import Any


SAME_STATE_VERDICTS = {
    "completed_same_visit",
    "completed_new_visit",
    "partial_needs_continue",
    "ineffective",
    "wrong_effect",
    "uncertain",
}


def parse_same_state_review(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, dict) or str(payload.get("verdict") or "") not in SAME_STATE_VERDICTS:
        return None
    patch = payload.get("handler_patch")
    if patch is not None and not isinstance(patch, dict):
        return None
    event = payload.get("event")
    if event is not None and not isinstance(event, dict):
        return None
    continuation = payload.get("continuation")
    if continuation is not None and not isinstance(continuation, dict):
        return None
    payload.setdefault("handler_patch", {"operations": []})
    payload.setdefault("event", {})
    payload.setdefault("continuation", {})
    return payload


def result_for_same_state_verdict(verdict: str, *, final_step: bool) -> str:
    if verdict == "completed_new_visit":
        return "verified_reentry"
    if verdict == "completed_same_visit":
        return "verified_success" if final_step else "partial_progress"
    if verdict == "partial_needs_continue":
        # The semantic reviewer has positively identified page-local progress.
        # Keeping this as no_effect both contradicts the review and degrades the
        # strategy that produced the progress.
        return "partial_progress"
    if verdict == "wrong_effect":
        return "wrong_transition"
    return "no_effect"
