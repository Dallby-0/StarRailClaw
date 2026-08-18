from __future__ import annotations

import uuid
from typing import Any

from state_machine.time_utils import now_iso as _now_iso


INTENT_SCHEMA_VERSION = "intent.v1"
INTENT_RUNTIME_KEY = "intent_runtime"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "abandoned"}


def ensure_intent_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    value = runtime.get(INTENT_RUNTIME_KEY)
    if not isinstance(value, dict):
        value = {
            "active_intent_id": None,
            "intents": {},
            "intent_stack": [],
        }
        runtime[INTENT_RUNTIME_KEY] = value
    value.setdefault("active_intent_id", None)
    if not isinstance(value.get("intents"), dict):
        value["intents"] = {}
    if not isinstance(value.get("intent_stack"), list):
        value["intent_stack"] = []
    return value


def active_intent(runtime: dict[str, Any]) -> dict[str, Any] | None:
    intent_runtime = ensure_intent_runtime(runtime)
    intent_id = str(intent_runtime.get("active_intent_id") or "")
    if not intent_id:
        return None
    intent = intent_runtime["intents"].get(intent_id)
    if not isinstance(intent, dict) or str(intent.get("status", "running")) in TERMINAL_STATUSES:
        intent_runtime["active_intent_id"] = None
        return None
    return intent


def create_intent(
    runtime: dict[str, Any],
    *,
    kind: str,
    phase: str = "start",
    params: dict[str, Any] | None = None,
    facts: dict[str, Any] | None = None,
    transitions: list[dict[str, Any]] | None = None,
    completion: dict[str, Any] | None = None,
    activate: bool = True,
) -> dict[str, Any]:
    intent_runtime = ensure_intent_runtime(runtime)
    intent_id = f"intent_{uuid.uuid4().hex[:12]}"
    now = _now_iso()
    intent = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "intent_id": intent_id,
        "kind": str(kind).strip(),
        "status": "running",
        "phase": str(phase).strip() or "start",
        "params": dict(params or {}),
        "facts": dict(facts or {}),
        "transitions": list(transitions or []),
        "completion": dict(completion or {}),
        "revision": 1,
        "created_at": now,
        "updated_at": now,
    }
    intent_runtime["intents"][intent_id] = intent
    if activate:
        intent_runtime["active_intent_id"] = intent_id
    return intent


def adopt_intent_proposal(runtime: dict[str, Any], proposal: Any) -> dict[str, Any] | None:
    if active_intent(runtime) is not None or not isinstance(proposal, dict):
        return None
    kind = str(proposal.get("kind") or "").strip()
    if not kind:
        return None
    return create_intent(
        runtime,
        kind=kind,
        phase=str(proposal.get("phase") or "start"),
        params=proposal.get("params") if isinstance(proposal.get("params"), dict) else {},
        facts=proposal.get("facts") if isinstance(proposal.get("facts"), dict) else {},
        transitions=proposal.get("transitions") if isinstance(proposal.get("transitions"), list) else [],
        completion=proposal.get("completion") if isinstance(proposal.get("completion"), dict) else {},
    )


def push_child_intent(
    runtime: dict[str, Any],
    *,
    kind: str,
    phase: str = "start",
    params: dict[str, Any] | None = None,
    transitions: list[dict[str, Any]] | None = None,
    completion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent_runtime = ensure_intent_runtime(runtime)
    parent = active_intent(runtime)
    if isinstance(parent, dict):
        parent["status"] = "suspended"
        parent["updated_at"] = _now_iso()
        intent_runtime["intent_stack"].append(str(parent["intent_id"]))
    return create_intent(runtime, kind=kind, phase=phase, params=params, transitions=transitions, completion=completion, activate=True)


def _resume_parent_intent(runtime: dict[str, Any]) -> dict[str, Any] | None:
    intent_runtime = ensure_intent_runtime(runtime)
    while intent_runtime["intent_stack"]:
        parent_id = str(intent_runtime["intent_stack"].pop())
        parent = intent_runtime["intents"].get(parent_id)
        if not isinstance(parent, dict) or str(parent.get("status")) in TERMINAL_STATUSES:
            continue
        parent["status"] = "running"
        parent["revision"] = int(parent.get("revision", 0) or 0) + 1
        parent["updated_at"] = _now_iso()
        intent_runtime["active_intent_id"] = parent_id
        return parent
    intent_runtime["active_intent_id"] = None
    return None


def activate_intent(runtime: dict[str, Any], intent_id: str | None) -> None:
    intent_runtime = ensure_intent_runtime(runtime)
    if intent_id is None:
        intent_runtime["active_intent_id"] = None
        return
    intent = intent_runtime["intents"].get(str(intent_id))
    if not isinstance(intent, dict):
        raise KeyError(f"unknown intent_id: {intent_id}")
    if str(intent.get("status", "running")) in TERMINAL_STATUSES:
        raise ValueError(f"cannot activate terminal intent: {intent_id}")
    intent["status"] = "running"
    intent["updated_at"] = _now_iso()
    intent_runtime["active_intent_id"] = str(intent_id)


def _event_value(event: dict[str, Any], reference: Any) -> Any:
    if not isinstance(reference, str) or not reference.startswith("$event."):
        return reference
    current: Any = event
    for part in reference[len("$event.") :].split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def reduce_intent_event(runtime: dict[str, Any], event: dict[str, Any] | None) -> dict[str, Any] | None:
    intent = active_intent(runtime)
    if intent is None or not isinstance(event, dict):
        return intent
    event_type = str(event.get("type") or "").strip()
    if not event_type:
        return intent

    facts = intent.get("facts") if isinstance(intent.get("facts"), dict) else {}
    intent["facts"] = facts
    event_facts = event.get("facts")
    if isinstance(event_facts, dict):
        facts.update(event_facts)

    current_phase = str(intent.get("phase", "start"))
    matched = None
    for rule in intent.get("transitions") if isinstance(intent.get("transitions"), list) else []:
        if not isinstance(rule, dict):
            continue
        from_phase = str(rule.get("from_phase", "*"))
        if from_phase not in {"*", current_phase} or str(rule.get("event", "")) != event_type:
            continue
        matched = rule
        break

    if isinstance(matched, dict):
        if matched.get("next_phase") is not None:
            intent["phase"] = str(matched.get("next_phase"))
        if matched.get("status") is not None:
            intent["status"] = str(matched.get("status"))
        fact_patch = matched.get("fact_patch")
        if isinstance(fact_patch, dict):
            for key, value in fact_patch.items():
                facts[str(key)] = _event_value(event, value)

    completion = intent.get("completion") if isinstance(intent.get("completion"), dict) else {}
    if str(completion.get("event") or "") == event_type:
        intent["status"] = "completed"
    if event_type == "intent_failed":
        intent["status"] = "failed"

    intent["last_event"] = dict(event)
    intent["revision"] = int(intent.get("revision", 0) or 0) + 1
    intent["updated_at"] = _now_iso()
    if str(intent.get("status")) in TERMINAL_STATUSES:
        ensure_intent_runtime(runtime)["active_intent_id"] = None
        _resume_parent_intent(runtime)
    return intent


def intent_projection(intent: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(intent, dict):
        return {}
    return {
        "intent_id": intent.get("intent_id"),
        "kind": intent.get("kind"),
        "phase": intent.get("phase"),
        "params": intent.get("params", {}),
        "facts": intent.get("facts", {}),
        "status": intent.get("status"),
        "revision": intent.get("revision"),
        "last_event": intent.get("last_event", {}),
    }
