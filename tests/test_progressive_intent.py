from __future__ import annotations

import json

from state_machine.command import resolve_effective_command
from state_machine.intent import active_intent, adopt_intent_proposal, create_intent, push_child_intent, reduce_intent_event
from state_machine.page_handler.store import handler_from_bootstrap
from state_machine.protocol import parse_state_payload


def _provider(provider_id: str, x: int = 500, y: int = 500) -> dict:
    return {
        "provider_id": provider_id,
        "base_priority": 50,
        "repeat_policy": "once_per_visit",
        "locators": [{"type": "point", "x": x, "y": y, "coordinate_space": "logical", "source": "bootstrap"}],
        "hints": [],
        "deferred_hints": [],
        "effect_hints": [],
        "successors": [],
        "emits_on_success": {"type": "advanced"},
        "brief": "advance",
    }


def _operation(name: str, *, default: bool = False, scope: str = "intent_specific", routes=None) -> dict:
    return {
        "operation": name,
        "is_default": default,
        "intent_scope": scope,
        "intent_effect": "advance",
        "expected_event": "advanced",
        "intent_routes": routes or [],
        "providers": [_provider(f"{name}_provider")],
    }


def _state(handler: dict) -> dict:
    return {"state_id": "001", "page_handler": handler}


def test_default_operation_needs_no_persistent_intent() -> None:
    handler = handler_from_bootstrap([_operation("advance", default=True)])
    command = resolve_effective_command(_state(handler), None)
    assert command is not None
    assert command.operation == "advance"
    assert command.source == "state_default"


def test_active_intent_route_overrides_default() -> None:
    handler = handler_from_bootstrap([
        _operation("advance", default=True, scope="intent_invariant"),
        _operation("inspect", routes=[{"intent_kind": "inspect", "phase": "start"}]),
    ])
    # Routes belong to the operation that declares them.
    handler["intent_routes"] = [{"intent_kinds": ["inspect"], "phases": ["start"], "operation": "inspect", "intent_effect": "advance", "expected_event": None, "params": {}}]
    intent = {"intent_id": "intent_x", "kind": "inspect", "phase": "start", "params": {}, "facts": {}}
    assert resolve_effective_command(_state(handler), intent).operation == "inspect"


def test_active_intent_uses_only_intent_invariant_default_as_fallback() -> None:
    intent = {"intent_id": "intent_x", "kind": "other", "phase": "start", "params": {}, "facts": {}}
    specific = handler_from_bootstrap([_operation("advance", default=True, scope="intent_specific")])
    invariant = handler_from_bootstrap([_operation("advance", default=True, scope="intent_invariant")])
    assert resolve_effective_command(_state(specific), intent) is None
    assert resolve_effective_command(_state(invariant), intent).operation == "advance"


def test_single_operation_becomes_default_but_multiple_are_ambiguous() -> None:
    single = handler_from_bootstrap([_operation("advance")])
    multiple = handler_from_bootstrap([_operation("advance"), _operation("inspect")])
    assert single["default_operation"]["operation"] == "advance"
    assert multiple["default_operation"] is None


def test_intent_survives_events_until_completion() -> None:
    runtime: dict = {}
    intent = create_intent(
        runtime,
        kind="workflow",
        phase="first",
        transitions=[
            {"from_phase": "first", "event": "selected", "next_phase": "second", "fact_patch": {"slot": "$event.slot"}},
            {"from_phase": "second", "event": "confirmed", "next_phase": "done"},
        ],
        completion={"event": "completed"},
    )
    reduce_intent_event(runtime, {"type": "selected", "slot": "x"})
    assert active_intent(runtime)["phase"] == "second"
    assert active_intent(runtime)["facts"]["slot"] == "x"
    reduce_intent_event(runtime, {"type": "confirmed"})
    reduce_intent_event(runtime, {"type": "completed"})
    assert active_intent(runtime) is None
    assert runtime["intent_runtime"]["intents"][intent["intent_id"]]["status"] == "completed"


def test_sparse_intent_proposal_never_replaces_active_intent() -> None:
    runtime: dict = {}
    proposed = adopt_intent_proposal(runtime, {"kind": "inspect", "phase": "start", "completion": {"event": "done"}})
    assert proposed is not None
    assert adopt_intent_proposal(runtime, {"kind": "replacement"}) is None
    assert active_intent(runtime)["kind"] == "inspect"


def test_child_intent_resumes_parent() -> None:
    runtime: dict = {}
    parent = create_intent(runtime, kind="parent", phase="start")
    push_child_intent(runtime, kind="child", completion={"event": "child_done"})
    reduce_intent_event(runtime, {"type": "child_done"})
    assert active_intent(runtime)["intent_id"] == parent["intent_id"]


def test_state_payload_requires_v2_provider_shape() -> None:
    payload = {
        "page_summary": "generic surface",
        "slug": "generic_surface",
        "possible_page_type": "none",
        "page_family": "generic_surface",
        "surface_relation": "uncertain",
        "common_identity": [],
        "elements": [],
        "intent_assessment": {"relation": "unknown", "reason": "none"},
        "intent_proposal": None,
        "bootstrap_operations": [_operation("advance", default=True)],
    }
    assert parse_state_payload(json.dumps(payload)) is not None
    payload["bootstrap_operations"][0]["providers"][0]["locators"][0]["coordinate_space"] = "real"
    assert parse_state_payload(json.dumps(payload)) is None
