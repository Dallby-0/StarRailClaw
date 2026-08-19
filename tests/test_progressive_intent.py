from __future__ import annotations

import json

from state_machine.command import classify_strategy_result, resolve_effective_command, step_preconditions_pass
from state_machine.intent import active_intent, adopt_intent_proposal, create_intent, push_child_intent, reduce_intent_event
from state_machine.page_handler.store import handler_from_bootstrap, mark_strategy_result, select_strategy
from state_machine.protocol import parse_state_payload


def _state(handler: dict) -> dict:
    return {"state_id": "001", "page_handler": handler}


def test_safe_default_needs_no_persistent_intent() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "dismiss_overlay",
            "is_default": True,
            "intent_scope": "intent_invariant",
            "intent_effect": "preserve",
            "safety": "low_risk",
            "steps": [{"resolver": {"type": "fixed_point", "x": 500, "y": 850}, "expected_after": {"exit_likely": True}}],
        }
    ])
    command = resolve_effective_command(_state(handler), None)
    assert command is not None
    assert command.operation == "dismiss_overlay"
    assert command.source == "state_default"
    assert command.intent_id is None


def test_active_intent_route_overrides_default_operation() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "dismiss_overlay",
            "is_default": True,
            "intent_scope": "intent_invariant",
            "intent_effect": "preserve",
            "safety": "low_risk",
            "steps": [{"resolver": {"type": "fixed_point", "x": 500, "y": 850}}],
            "intent_routes": [{"intent_kind": "inspect_reward", "phase": "inspect", "operation": "open_reward_details", "expected_event": "details_opened"}],
        },
        {
            "operation": "open_reward_details",
            "intent_scope": "intent_specific",
            "intent_effect": "advance",
            "safety": "reversible",
            "steps": [{"resolver": {"type": "fixed_point", "x": 500, "y": 420}}],
        },
    ])
    intent = {"intent_id": "intent_x", "kind": "inspect_reward", "phase": "inspect", "params": {"target": "foo"}, "facts": {}}
    command = resolve_effective_command(_state(handler), intent)
    assert command is not None
    assert command.operation == "open_reward_details"
    assert command.source == "active_intent"
    assert command.intent_id == "intent_x"


def test_compact_llm_intent_route_is_normalized_to_executable_route() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "start_run",
            "is_default": True,
            "intent_scope": "intent_specific",
            "intent_effect": "advance",
            "safety": "low_risk",
            "steps": [{"resolver": {"type": "fixed_point", "x": 845, "y": 900}}],
            "intent_routes": [{"kind": "clear_universe", "phase": "start"}],
        }
    ])
    assert handler["intent_routes"] == [{"intent_kinds": ["clear_universe"], "phases": ["start"], "operation": "start_run", "intent_effect": "advance", "expected_event": None, "params": {}}]
    intent = {"intent_id": "intent_x", "kind": "clear_universe", "phase": "start", "params": {}, "facts": {}}
    assert resolve_effective_command(_state(handler), intent).operation == "start_run"


def test_active_intent_only_allows_transparent_default_fallback() -> None:
    unsafe_handler = handler_from_bootstrap([
        {
            "operation": "confirm_purchase",
            "is_default": True,
            "intent_scope": "intent_specific",
            "intent_effect": "advance",
            "safety": "commit",
            "steps": [{"resolver": {"type": "fixed_point", "x": 800, "y": 850}}],
        }
    ])
    intent = {"intent_id": "intent_x", "kind": "inspect_only", "phase": "start", "params": {}, "facts": {}}
    assert resolve_effective_command(_state(unsafe_handler), intent) is None

    transparent_handler = handler_from_bootstrap([
        {
            "operation": "dismiss_tutorial",
            "is_default": True,
            "intent_scope": "intent_invariant",
            "intent_effect": "preserve",
            "safety": "low_risk",
            "steps": [{"resolver": {"type": "fixed_point", "x": 500, "y": 850}}],
        }
    ])
    command = resolve_effective_command(_state(transparent_handler), intent)
    assert command is not None
    assert command.operation == "dismiss_tutorial"
    assert command.intent_effect == "preserve"


def test_explicit_safe_default_from_first_llm_call_executes_without_intent() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "start_run",
            "is_default": True,
            "intent_scope": "intent_specific",
            "intent_effect": "advance",
            "safety": "low_risk",
            "steps": [{"resolver": {"type": "fixed_point", "x": 845, "y": 900}, "expected_after": {"exit_likely": True}}],
        }
    ])
    command = resolve_effective_command(_state(handler), None)
    assert command is not None
    assert command.operation == "start_run"
    assert command.source == "state_default"


def test_single_safe_bootstrap_operation_becomes_default_without_second_llm_call() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "select_candidate",
            "intent_scope": "intent_specific",
            "intent_effect": "advance",
            "safety": "reversible",
            "steps": [{"resolver": {"type": "fixed_point", "x": 400, "y": 500}}],
        }
    ])
    assert handler["default_operation"]["operation"] == "select_candidate"
    assert resolve_effective_command(_state(handler), None).operation == "select_candidate"


def test_ambiguous_or_high_risk_bootstrap_does_not_become_implicit_default() -> None:
    multiple = handler_from_bootstrap([
        {"operation": "inspect", "safety": "low_risk", "steps": [{"resolver": {"type": "fixed_point", "x": 400, "y": 400}}]},
        {"operation": "leave", "safety": "low_risk", "steps": [{"resolver": {"type": "fixed_point", "x": 50, "y": 50}}]},
    ])
    assert multiple["default_operation"] is None
    destructive = handler_from_bootstrap([
        {"operation": "delete", "safety": "destructive", "steps": [{"resolver": {"type": "fixed_point", "x": 800, "y": 850}}]},
    ])
    assert destructive["default_operation"] is None


def test_operation_safety_is_inherited_by_its_strategies() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "confirm_purchase",
            "intent_scope": "intent_specific",
            "intent_effect": "advance",
            "safety": "commit",
            "strategies": [{"strategy_id": "confirm", "level": 0, "resolver": {"type": "fixed_point", "x": 800, "y": 850}}],
        }
    ])
    strategy = handler["operation_policies"]["confirm_purchase"]["strategies"][0]
    assert strategy["safety"] == "commit"


def test_two_step_preconditions_are_explicit_and_fail_closed() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "select_and_confirm",
            "safety": "reversible",
            "steps": [
                {"step_id": "select", "resolver": {"type": "fixed_point", "x": 300, "y": 400}, "emits_on_success": {"type": "candidate_selected"}},
                {"step_id": "confirm", "resolver": {"type": "fixed_point", "x": 800, "y": 850}, "preconditions": {"previous_step_verified": True, "previous_event": "candidate_selected", "current_page_still_matches": True}},
            ],
        }
    ])
    steps = handler["operation_policies"]["select_and_confirm"]["strategies"][0]["steps"]
    assert steps[1]["preconditions"] == {"previous_step_verified": True, "previous_event": "candidate_selected", "current_page_still_matches": True}
    ok, reason = step_preconditions_pass(steps[1]["preconditions"], previous_step_verified=True, previous_event={"type": "candidate_selected"}, current_state_matches=True, intent=None)
    assert ok and reason == "ok"
    ok, reason = step_preconditions_pass({"confirm_enabled": True}, previous_step_verified=True, previous_event=None, current_state_matches=True, intent=None)
    assert not ok and reason.startswith("unsupported_preconditions")


def test_no_visual_change_intermediate_step_continues_to_confirmation() -> None:
    result = classify_strategy_result(
        changed=False,
        left_state=True,  # transient matcher miss must not end an intermediate step
        expected={"screen_should_change": False, "exit_likely": False},
        final_step=False,
    )
    assert result == "partial_progress"


def test_final_exit_step_accepts_changed_screen_before_new_state_is_known() -> None:
    result = classify_strategy_result(
        changed=True,
        left_state=False,
        expected={"screen_should_change": True, "exit_likely": True},
        final_step=True,
    )
    assert result == "verified_success"


def test_intent_survives_page_events_until_completion_event() -> None:
    runtime: dict = {}
    intent = create_intent(
        runtime,
        kind="select_and_confirm",
        phase="select",
        transitions=[
            {"from_phase": "select", "event": "candidate_selected", "next_phase": "confirm", "fact_patch": {"slot": "$event.slot"}},
            {"from_phase": "confirm", "event": "selection_confirmed", "next_phase": "await_reward"},
        ],
        completion={"event": "reward_acquired"},
    )
    intent_id = intent["intent_id"]
    reduce_intent_event(runtime, {"type": "candidate_selected", "slot": "right"})
    assert active_intent(runtime)["intent_id"] == intent_id
    assert active_intent(runtime)["phase"] == "confirm"
    assert active_intent(runtime)["facts"]["slot"] == "right"
    reduce_intent_event(runtime, {"type": "selection_confirmed"})
    assert active_intent(runtime)["phase"] == "await_reward"
    reduce_intent_event(runtime, {"type": "reward_acquired"})
    assert active_intent(runtime) is None
    assert runtime["intent_runtime"]["intents"][intent_id]["status"] == "completed"


def test_llm_intent_proposal_is_sparse_and_never_replaces_active_intent() -> None:
    runtime: dict = {}
    assert adopt_intent_proposal(runtime, None) is None
    proposed = adopt_intent_proposal(runtime, {"kind": "inspect_reward", "phase": "inspect", "completion": {"event": "inspection_done"}})
    assert proposed is not None
    assert active_intent(runtime)["kind"] == "inspect_reward"
    assert adopt_intent_proposal(runtime, {"kind": "replacement"}) is None
    assert active_intent(runtime)["kind"] == "inspect_reward"


def test_child_intent_interrupt_resumes_parent_after_completion() -> None:
    runtime: dict = {}
    parent = create_intent(runtime, kind="equip_relic", phase="select")
    child = push_child_intent(runtime, kind="dismiss_tutorial", completion={"event": "overlay_dismissed"})
    assert active_intent(runtime)["intent_id"] == child["intent_id"]
    assert runtime["intent_runtime"]["intents"][parent["intent_id"]]["status"] == "suspended"
    reduce_intent_event(runtime, {"type": "overlay_dismissed"})
    assert active_intent(runtime)["intent_id"] == parent["intent_id"]
    assert active_intent(runtime)["status"] == "running"


def test_failed_cheap_strategy_escalates_to_stronger_strategy() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "dismiss_overlay",
            "is_default": True,
            "intent_scope": "intent_invariant",
            "intent_effect": "preserve",
            "safety": "low_risk",
            "strategies": [
                {"strategy_id": "fixed", "level": 0, "status": "active", "resolver": {"type": "fixed_point", "x": 500, "y": 850}},
                {"strategy_id": "template", "level": 2, "status": "active", "resolver": {"type": "region_template", "template_path": "button.png", "search_rect": [600, 700, 300, 200]}},
            ],
        }
    ])
    assert select_strategy(handler, "dismiss_overlay")["strategy_id"] == "fixed"
    mark_strategy_result(handler, "dismiss_overlay", "fixed", "no_effect")
    assert select_strategy(handler, "dismiss_overlay")["strategy_id"] == "template"
    mark_strategy_result(handler, "dismiss_overlay", "template", "verified_success")
    assert select_strategy(handler, "dismiss_overlay")["strategy_id"] == "template"


def test_degraded_strategy_gets_one_rehabilitation_canary_before_llm_repair() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "select_and_confirm",
            "safety": "reversible",
            "strategies": [{
                "strategy_id": "two_step",
                "level": 0,
                "status": "degraded",
                "steps": [
                    {"resolver": {"type": "fixed_point", "x": 235, "y": 480}, "expected_after": {"screen_should_change": False, "exit_likely": False}},
                    {"resolver": {"type": "fixed_point", "x": 850, "y": 885}, "expected_after": {"screen_should_change": True, "exit_likely": True}},
                ],
            }],
        }
    ])
    assert select_strategy(handler, "select_and_confirm")["strategy_id"] == "two_step"
    assert select_strategy(handler, "select_and_confirm", excluded={"two_step"}) is None


def test_state_payload_requires_bootstrap_operations_not_legacy_actions() -> None:
    payload = {
        "page_summary": "popup",
        "slug": "popup",
        "possible_page_type": "popup",
        "elements": [],
        "bootstrap_operations": [],
    }
    assert parse_state_payload(json.dumps(payload)) == payload
    legacy = {"page_summary": "popup", "slug": "popup", "elements": [], "actions": []}
    assert parse_state_payload(json.dumps(legacy)) is None


def test_state_payload_locally_repairs_missing_bbox_commas_without_llm_retry() -> None:
    malformed = r'''{
      "page_summary": "blessing select",
      "slug": "blessing_select",
      "possible_page_type": "none",
      "elements": [{"type":"pattern","bbox":[26 25, 62 80]}],
      "bootstrap_operations": [{
        "operation":"select_first",
        "steps":[
          {"resolver":{"type":"fixed_point","x":235,"y":480},"expected_after":{"screen_should_change":false,"exit_likely":false}},
          {"resolver":{"type":"fixed_point","x":850,"y":885},"expected_after":{"screen_should_change":true,"exit_likely":true}}
        ]
      }]
    }'''
    payload = parse_state_payload(malformed)
    assert payload is not None
    assert payload["elements"][0]["bbox"] == [26, 25, 62, 80]
