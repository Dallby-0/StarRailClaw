from __future__ import annotations

from state_machine.page_handler.reactive import (
    advance_observation,
    mark_provider_result,
    merge_controller,
    record_provider_attempt,
    select_provider,
)
from state_machine.page_handler.store import apply_handler_patch, ensure_page_handler, handler_from_bootstrap
from state_machine.progress_guard import clear_visit_progress, reactive_cursor_for


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


def test_v1_strategy_is_migrated_to_reactive_controller() -> None:
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
    handler = ensure_page_handler(meta)
    controller = _operation(handler)["controller"]
    assert handler["schema_version"] == "progressive_handler.v2"
    assert controller["providers"][0]["provider_id"] == "fixed_v1_click"


def test_visit_cleanup_removes_reactive_cursor() -> None:
    runtime: dict = {}
    reactive_cursor_for(runtime, "004:1", "advance_page")
    assert runtime["visit_operation_reactive"]
    clear_visit_progress(runtime, "004:1")
    assert runtime["visit_operation_reactive"] == {}
