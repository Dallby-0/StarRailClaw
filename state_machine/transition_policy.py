from __future__ import annotations

from typing import Any

from state_machine.graph import _get_reachable_targets
from state_machine.io import _save_runtime
from state_machine.logger import FsmRunLogger
from state_machine.matching import MatchResult


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _pick_reachable_transition_candidate(matches: list[MatchResult], state_id: str, reachable: set[str]) -> tuple[MatchResult | None, str]:
    reachable_hits = [m for m in matches if m.success and m.state_id in reachable and m.state_id != state_id]
    if len(reachable_hits) == 1:
        return reachable_hits[0], "strong"
    if len(reachable_hits) > 1:
        return None, "ambiguous_reachable"
    return None, "none"


def _defer_unknown_transition(
    *,
    runtime: dict[str, Any],
    state_id: str,
    action_id: str,
    logger: FsmRunLogger | None,
    frame,
    reason: str,
    effort: str | None = None,
) -> tuple[bool, Any]:
    runtime["last_state_id"] = None
    runtime["last_transition_ok"] = True
    runtime["pending_from_state_id"] = state_id
    runtime["pending_action_id"] = action_id
    if effort is not None:
        runtime["repair_fail_count"] = 0
    _save_runtime(runtime)
    fields: dict[str, Any] = {"from_state": state_id, "action_id": action_id, "to_state": None, "reason": reason}
    if effort is not None:
        fields["effort"] = effort
    _log(logger, f"[fsm][transition][to-unknown] from={state_id} action={action_id} reason={reason}", "transition", **fields)
    return True, frame


def _resolve_transition_after_progress(
    *,
    state_id: str,
    action_id: str,
    matches: list[MatchResult],
    graph: dict[str, Any],
    runtime: dict[str, Any],
    prefer_reachable_first: bool,
    logger: FsmRunLogger | None,
    reason_suffix: str,
) -> MatchResult | None:
    reachable = _get_reachable_targets(graph, state_id)
    nxt, confidence = _pick_reachable_transition_candidate(matches, state_id, reachable if prefer_reachable_first else set())
    if nxt is not None:
        reason = f"reachable{reason_suffix}"
        _log(logger, f"[fsm][transition][{reason}] from={state_id} action={action_id} to={nxt.state_id} confidence={confidence}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason=reason, confidence=confidence)
    if nxt is not None and nxt.state_id != state_id:
        runtime["last_state_id"] = nxt.state_id
        runtime["last_transition_ok"] = True
        runtime["pending_from_state_id"] = None
        runtime["pending_action_id"] = None
        _save_runtime(runtime)
        return nxt
    return None
