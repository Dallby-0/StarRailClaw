from __future__ import annotations

from state_machine.page_handler.reactive import (
    advance_observation,
    apply_progress_audit,
    ensure_progress_capacity,
    grant_capacity_after_recovery,
    mark_provider_result,
    merge_controller,
    promote_recovery_provider,
    provider_resolver_signature,
    progress_capacity_exhausted,
    record_provider_attempt,
    select_provider,
)
from state_machine.page_handler.store import apply_handler_patch, ensure_page_handler, handler_from_bootstrap
from state_machine.page_handler.llm import parse_next_action_recovery, parse_progress_audit
from state_machine.progress_guard import capacity_recovery_exhausted, clear_visit_progress, reactive_cursor_for, record_capacity_recovery


def _operation(handler, name="advance_page"):
    return handler["operation_policies"][name]


def test_bootstrap_builds_minimal_reactive_fast_path_without_exploration() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "advance_page",
            "is_default": True,
            "intent_scope": "intent_invariant",
            "intent_effect": "advance",
            "safety": "low_risk",
            "steps": [{
                "step_id": "continue",
                "resolver": {"type": "fixed_point", "x": 840, "y": 880},
                "expected_after": {"state_relation": "must_leave", "reentry_policy": "forbid"},
            }],
        }
    ])

    controller = _operation(handler)["controller"]
    assert controller["type"] == "reactive_local"
    assert len(controller["providers"]) == 1
    assert controller["providers"][0]["provider_id"] == "continue"
    assert controller["providers"][0]["resolver"] == {"type": "fixed_point", "x": 840, "y": 880}
    assert controller["providers"][0]["cost"] == 0
    assert controller["max_actions"] == 20
    assert controller["max_observation_rounds"] == 10


def test_changed_observation_does_not_replay_once_per_visit_direct_action() -> None:
    controller = {
        "type": "reactive_local",
        "providers": [
            {"provider_id": "select", "kind": "action", "cost": 0, "repeat_policy": "once_per_visit", "status": "active"},
            {"provider_id": "confirm", "kind": "action", "cost": 1, "repeat_policy": "once_per_visit", "status": "active"},
        ],
    }
    cursor = {
        "observation_epoch": 1,
        "attempted_visit": [],
        "attempted_by_epoch": {"1": []},
        "total_actions": 0,
    }

    selected = select_provider(controller, cursor)
    assert selected["provider_id"] == "select"
    record_provider_attempt(cursor, selected, action_count=1)
    advance_observation(cursor)
    assert select_provider(controller, cursor)["provider_id"] == "confirm"


def test_once_per_observation_provider_can_run_again_only_after_change() -> None:
    controller = {
        "type": "reactive_local",
        "providers": [
            {"provider_id": "find_confirm", "kind": "action", "cost": 1, "repeat_policy": "once_per_observation", "status": "active"},
        ],
    }
    cursor = {
        "observation_epoch": 1,
        "attempted_visit": [],
        "attempted_by_epoch": {"1": []},
        "total_actions": 0,
    }
    provider = select_provider(controller, cursor)
    record_provider_attempt(cursor, provider)
    assert select_provider(controller, cursor) is None
    advance_observation(cursor)
    assert select_provider(controller, cursor)["provider_id"] == "find_confirm"


def test_repair_patch_appends_generalized_fallback_without_replacing_direct_action() -> None:
    handler = handler_from_bootstrap([
        {
            "operation": "select_station_and_confirm",
            "safety": "reversible",
            "steps": [{"step_id": "select_center", "resolver": {"type": "fixed_point", "x": 640, "y": 400}}],
        }
    ])
    touched = apply_handler_patch(
        handler,
        {
            "operations": [{
                "operation": "select_station_and_confirm",
                "safety": "reversible",
                "controller": {
                    "type": "reactive_local",
                    "providers": [{
                        "provider_id": "find_confirm_text",
                        "kind": "action",
                        "cost": 5,
                        "repeat_policy": "once_per_observation",
                        "resolver": {"type": "fixed_point", "x": 640, "y": 600},
                    }],
                    "fallback_profiles": [{
                        "id": "station_choice_band",
                        "kind": "horizontal_probe",
                        "x_start": 480,
                        "x_end": 800,
                        "y": 400,
                        "samples": 5,
                    }],
                },
            }]
        },
    )

    controller = _operation(handler, "select_station_and_confirm")["controller"]
    ids = {item["provider_id"] for item in controller["providers"]}
    assert ids == {"select_center", "find_confirm_text", "learned_exploration"}
    assert touched == {"find_confirm_text", "learned_exploration"}

    merged = merge_controller(
        controller,
        {
            "type": "reactive_local",
            "fallback_profiles": [{
                "id": "confirm_band",
                "kind": "horizontal_probe",
                "x_start": 500,
                "x_end": 780,
                "y": 600,
                "samples": 4,
            }],
        },
    )
    exploration = next(item for item in merged["providers"] if item["provider_id"] == "learned_exploration")
    assert {item["id"] for item in exploration["profiles"]} == {"station_choice_band", "confirm_band"}


def test_provider_miss_is_not_globally_degraded() -> None:
    controller = {
        "type": "reactive_local",
        "providers": [{
            "provider_id": "layout_specific_point",
            "kind": "action",
            "cost": 0,
            "repeat_policy": "once_per_visit",
            "status": "active",
            "success_count": 2,
            "fail_count": 0,
            "result_counts": {},
        }],
    }
    mark_provider_result(controller, "layout_specific_point", "no_change")
    provider = controller["providers"][0]
    assert provider["status"] == "active"
    assert provider["success_count"] == 2
    assert provider["fail_count"] == 1


def test_legacy_strategy_schema_is_rejected_instead_of_migrated() -> None:
    meta = {
        "page_handler": {
            "schema_version": "progressive_handler.v1",
            "operation_policies": {
                "advance_page": {
                    "operation": "advance_page",
                    "safety": "low_risk",
                    "strategies": [{
                        "strategy_id": "fixed_v1",
                        "status": "active",
                        "level": 0,
                        "steps": [{"step_id": "click", "resolver": {"type": "fixed_point", "x": 500, "y": 800}}],
                    }],
                }
            },
        }
    }
    try:
        ensure_page_handler(meta)
    except ValueError as exc:
        assert "fresh state workspace" in str(exc)
    else:
        raise AssertionError("legacy handler must not be silently migrated")


def test_visit_cleanup_removes_reactive_cursor() -> None:
    runtime: dict = {}
    reactive_cursor_for(runtime, "004:1", "advance_page")
    assert runtime["visit_operation_reactive"]
    clear_visit_progress(runtime, "004:1")
    assert runtime["visit_operation_reactive"] == {}


def test_progress_capacity_expands_by_fixed_runtime_rule() -> None:
    cursor = {
        "total_actions": 12,
        "observation_epoch": 7,
        "action_capacity": 0,
        "observation_capacity": 0,
    }
    ensure_progress_capacity(cursor, {"max_actions": 12, "max_observation_rounds": 6})
    assert progress_capacity_exhausted(cursor) == ["actions", "observations"]

    decision = apply_progress_audit(cursor, "progressing", "mid")
    assert decision == {
        "granted": True,
        "grant_kind": "progress_extension",
        "before": {"actions": 12, "observations": 6},
        "after": {"actions": 20, "observations": 10},
        "extensions": 1,
    }
    assert progress_capacity_exhausted(cursor) == []


def test_progress_capacity_refuses_stall_and_limits_uncertain_probe() -> None:
    cursor = {
        "total_actions": 12,
        "observation_epoch": 7,
        "action_capacity": 12,
        "observation_capacity": 6,
        "capacity_extensions": 0,
        "uncertain_probe_used": False,
    }
    stalled = apply_progress_audit(cursor, "stalled", "high")
    assert stalled["granted"] is False
    assert cursor["action_capacity"] == 12

    probe = apply_progress_audit(cursor, "uncertain", "mid")
    assert probe["grant_kind"] == "uncertain_probe"
    assert probe["after"] == {"actions": 14, "observations": 7}

    second_probe = apply_progress_audit(cursor, "uncertain", "high")
    assert second_probe["granted"] is False
    assert cursor["action_capacity"] == 14


def test_progress_capacity_has_fixed_extension_count_limit() -> None:
    cursor = {
        "total_actions": 12,
        "observation_epoch": 7,
        "action_capacity": 12,
        "observation_capacity": 6,
        "capacity_extensions": 0,
    }
    for _ in range(4):
        assert apply_progress_audit(cursor, "new_instance", "high")["granted"] is True
    final = apply_progress_audit(cursor, "progressing", "high")
    assert final["granted"] is False
    assert cursor["action_capacity"] == 44
    assert cursor["observation_capacity"] == 22


def test_progress_audit_parser_accepts_only_classification_fields() -> None:
    parsed = parse_progress_audit(
        '{"verdict":"progressing","confidence":"high","reason":"new card set",'
        '"evidence":["confirm led to another selection screen"],"grant_actions":999}'
    )
    assert parsed == {
        "verdict": "progressing",
        "confidence": "high",
        "reason": "new card set",
        "evidence": ["confirm led to another selection screen"],
    }
    assert parse_progress_audit('{"verdict":"extend_by_999","confidence":"high"}') is None


def test_capacity_recovery_has_independent_two_attempt_budget() -> None:
    runtime: dict = {}
    assert capacity_recovery_exhausted(runtime, "004:22", "select_station") is False
    assert record_capacity_recovery(runtime, "004:22", "select_station") == 1
    assert capacity_recovery_exhausted(runtime, "004:22", "select_station") is False
    assert record_capacity_recovery(runtime, "004:22", "select_station") == 2
    assert capacity_recovery_exhausted(runtime, "004:22", "select_station") is True
    clear_visit_progress(runtime, "004:22")
    assert capacity_recovery_exhausted(runtime, "004:22", "select_station") is False


def test_recovery_provider_is_promoted_only_after_observed_progress_and_deduplicated() -> None:
    controller = {"type": "reactive_local", "providers": [], "max_actions": 20, "max_observation_rounds": 10}
    raw = {
        "provider_id": "click_visible_confirm",
        "kind": "action",
        "cost": 5,
        "repeat_policy": "once_per_observation",
        "status": "proposed",
        "resolver": {"type": "fixed_point", "x": 640, "y": 720},
        "expected_after": {"state_relation": "may_leave", "reentry_policy": "same_visit"},
        "brief": "click visible confirm",
    }
    assert promote_recovery_provider(controller, raw, outcome="no_change", visit_id="004:22", observation_epoch=8) is None
    assert controller["providers"] == []

    proposed = promote_recovery_provider(controller, raw, outcome="changed_same_state", visit_id="004:22", observation_epoch=8)
    assert proposed["status"] == "proposed"
    assert proposed["source"] == "llm_capacity_recovery"
    assert proposed["result_counts"] == {"changed_same_state": 1}

    same_behavior = {**raw, "provider_id": "different_model_name"}
    active = promote_recovery_provider(controller, same_behavior, outcome="state_left", visit_id="004:22", observation_epoch=9)
    assert len(controller["providers"]) == 1
    assert active["status"] == "active"
    assert active["success_count"] == 1
    assert active["result_counts"] == {"changed_same_state": 1, "state_left": 1}
    assert provider_resolver_signature(active) == provider_resolver_signature(raw)


def test_next_action_recovery_parser_rejects_non_action_payloads() -> None:
    parsed = parse_next_action_recovery(
        '{"decision":"act","reason":"visible confirm","provider":{'
        '"provider_id":"confirm","kind":"action","cost":1,"repeat_policy":"once_per_observation",'
        '"status":"proposed","resolver":{"type":"fixed_point","x":640,"y":720},'
        '"expected_after":{"state_relation":"may_leave","reentry_policy":"same_visit"},"brief":"confirm"}}'
    )
    assert parsed is not None
    assert parsed["provider"]["provider_id"] == "confirm"
    assert parse_next_action_recovery('{"decision":"act","reason":"x","provider":{"kind":"exploration"}}') is None
    assert parse_next_action_recovery('{"decision":"give_up","reason":"no visible action","provider":{}}') == {
        "decision": "give_up",
        "reason": "no visible action",
    }


def test_successful_same_state_recovery_gets_small_fixed_follow_up_window() -> None:
    cursor = {"total_actions": 21, "observation_epoch": 11, "action_capacity": 20, "observation_capacity": 10}
    assert grant_capacity_after_recovery(cursor) == {"actions": 25, "observations": 13}
