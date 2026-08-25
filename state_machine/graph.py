from __future__ import annotations

from typing import Any

from state_machine import constants
from state_machine.io import _load_json, _now_iso, _save_json
from state_machine.logger import FsmRunLogger


def _append_graph_node(state_id: str, slug: str, logger: FsmRunLogger | None = None) -> None:
    graph = _load_json(constants.FSM_GRAPH_PATH)
    graph["nodes"].append({"state_id": state_id, "slug": slug, "enabled": True})
    graph["updated_at"] = _now_iso()
    _save_json(constants.FSM_GRAPH_PATH, graph)
    if logger is not None:
        logger.event("graph_node_added", state_id=state_id, slug=slug)


def _append_graph_edge(
    from_state_id: str | None,
    action_id: str,
    to_state_id: str,
    logger: FsmRunLogger | None = None,
    reason: str = "",
    confidence: str = "strong",
    transition_kind: str = "normal",
) -> None:
    if not from_state_id:
        return
    graph = _load_json(constants.FSM_GRAPH_PATH)
    for e in graph.get("edges", []):
        if e.get("from_state_id") == from_state_id and e.get("action_id") == action_id and e.get("to_state_id") == to_state_id:
            changed = False
            if confidence in {"strong", "llm_verified"} and e.get("confidence") == "tentative":
                e["confidence"] = confidence
                e["enabled"] = True
                e["reason"] = reason
                e["updated_at"] = _now_iso()
                changed = True
            if transition_kind == "reentry" and e.get("transition_kind") != "reentry":
                e["transition_kind"] = "reentry"
                e["reason"] = reason
                e["updated_at"] = _now_iso()
                changed = True
            if changed:
                graph["updated_at"] = _now_iso()
                _save_json(constants.FSM_GRAPH_PATH, graph)
            return
    graph["edges"].append(
        {
            "from_state_id": from_state_id,
            "action_id": action_id,
            "to_state_id": to_state_id,
            "enabled": confidence != "tentative",
            "weight": 1.0,
            "confidence": confidence,
            "reason": reason,
            "transition_kind": transition_kind,
            "created_at": _now_iso(),
        }
    )
    graph["updated_at"] = _now_iso()
    _save_json(constants.FSM_GRAPH_PATH, graph)
    if logger is not None:
        logger.event(
            "graph_edge_added",
            from_state_id=from_state_id,
            action_id=action_id,
            to_state_id=to_state_id,
            reason=reason,
            confidence=confidence,
            transition_kind=transition_kind,
        )


def _get_reachable_targets(graph: dict[str, Any], from_state_id: str | None) -> set[str]:
    if not from_state_id:
        return set()
    return {
        str(e.get("to_state_id"))
        for e in graph.get("edges", [])
        if e.get("enabled", True) and str(e.get("from_state_id")) == from_state_id
    }
