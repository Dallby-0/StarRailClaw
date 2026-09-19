from __future__ import annotations

from pathlib import Path

import numpy as np

from state_machine.page_handler.actions import (
    evaluate_guard,
    evaluate_effect,
    resolve_provider,
)
from state_machine.page_handler.guard_learning import observe_successful_transition
from state_machine.page_handler.llm import parse_repair_response
from state_machine.page_handler.reactive import (
    cursor_for,
    mark_confirmed_effect,
    normalize_locator,
    normalize_provider,
    provider_score,
    rank_providers,
    record_attempt,
    update_successor_context,
)
from state_machine.page_handler.store import (
    HANDLER_SCHEMA_VERSION,
    apply_handler_patch,
    ensure_page_handler,
    handler_from_bootstrap,
    materialize_provider_templates,
    merge_operation,
    record_provider_result,
)


def _point(x: int, y: int) -> dict:
    return {"type": "point", "x": x, "y": y, "coordinate_space": "logical", "source": "bootstrap"}


def _provider(provider_id: str, priority: int = 0, **extra) -> dict:
    return normalize_provider({
        "provider_id": provider_id,
        "base_priority": priority,
        "locators": [_point(500, 500)],
        **extra,
    })


class FakeVision:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def match_template(self, frame, path, rect, threshold=0.82):
        return (path.name == "found.png", (321, 123), 0.91)

    def ocr_blocks(self, frame, rect, white_text=False):
        return [{"text": "target text", "bbox": (100, 200, 80, 20), "center": (140, 210)}]

    def detect_text_lines(self, frame, rect):
        return {"available": True, "line_count": int(frame[0, 0, 0]), "lines": []}

    def save_template_from_rect(self, frame, rect, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"template")
        self.saved.append(path)


def test_bootstrap_persists_v3_providers_without_controller_or_strategy() -> None:
    handler = handler_from_bootstrap([{
        "operation": "advance_surface",
        "is_default": True,
        "intent_scope": "intent_specific",
        "intent_effect": "advance",
        "providers": [
            {"provider_id": "first", "base_priority": 100, "locators": [_point(300, 400)], "successors": ["second"]},
            {"provider_id": "second", "base_priority": 80, "locators": [_point(800, 850)]},
        ],
    }])
    policy = handler["operation_policies"]["advance_surface"]
    assert handler["schema_version"] == HANDLER_SCHEMA_VERSION
    assert "controller" not in policy
    assert "strategies" not in policy
    assert [item["provider_id"] for item in policy["providers"]] == ["first", "second"]
    assert policy["providers"][0]["successors"] == ["second"]


def test_old_or_unversioned_handler_is_rejected() -> None:
    for page_handler in ({"schema_version": "reactive_handler.v1", "operation_policies": {}}, {"operation_policies": {}}):
        try:
            ensure_page_handler({"page_handler": page_handler})
        except ValueError as exc:
            assert "fresh state workspace" in str(exc)
        else:
            raise AssertionError("legacy page handler was accepted")


def test_coordinate_space_is_explicit_and_real_points_are_not_persisted() -> None:
    assert normalize_locator({"type": "point", "x": 10, "y": 20, "coordinate_space": "real"}) is None
    locator = normalize_locator({"type": "point", "x": 10, "y": 20, "coordinate_space": "logical"})
    assert locator["coordinate_space"] == "logical"


def test_visual_hints_rank_pass_above_unknown_above_fail() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    operation = {"providers": [_provider("pass", 0), _provider("unknown", 0), _provider("fail", 0)]}
    ranked = rank_providers(operation, cursor, {"pass": ["pass"], "unknown": ["unknown"], "fail": ["fail"]})
    assert [item["provider_id"] for item in ranked] == ["pass", "unknown", "fail"]


def test_line_count_guard_returns_graded_strength() -> None:
    guard = {"type": "line_count", "rect": [0, 0, 100, 100], "target": 3, "tolerance": 1}
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0, 0] = 2
    detail = evaluate_guard(guard, frame, FakeVision(), {}, allow_expensive=True)
    assert detail == {"result": "pass", "strength": 0.7, "line_count": 2}


def test_explicit_entry_prevents_isolated_terminal_provider_from_starting() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    select = _provider("select", 0, successors=["confirm"])
    confirm = _provider("confirm", 100)
    operation = {"entry_providers": ["select"], "providers": [select, confirm]}
    assert rank_providers(operation, cursor, {})[0] is select


def test_successor_is_a_short_bonus_and_unmentioned_provider_is_neutral() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    cursor["successor_ids"] = ["next"]
    next_provider = _provider("next", 0)
    neutral = _provider("neutral", 0)
    assert provider_score(next_provider, cursor, []) > provider_score(neutral, cursor, [])
    cursor["successor_ids"] = []
    assert provider_score(next_provider, cursor, []) == provider_score(neutral, cursor, [])


def test_successor_chain_bonus_can_override_one_failed_visual_hint() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    cursor["successor_ids"] = ["next"]
    operation = {"providers": [_provider("next", 0), _provider("neutral", 0)]}
    ranked = rank_providers(operation, cursor, {"next": ["fail"], "neutral": []})
    assert ranked[0]["provider_id"] == "next"


def test_fresh_visit_gives_chain_bonus_to_unreferenced_root_provider() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    root = _provider("root", 0, successors=["child"])
    child = _provider("child", 0)
    operation = {"providers": [root, child]}
    ranked = rank_providers(operation, cursor, {"root": ["unknown"], "child": ["pass"]})
    assert ranked[0]["provider_id"] == "root"


def test_successor_bonus_exceeds_stable_visible_unrelated_provider() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    cursor["successor_ids"] = ["next"]
    operation = {"providers": [_provider("next", 0), _provider("visible", 0)]}
    ranked = rank_providers(operation, cursor, {"next": ["unknown"], "visible": ["pass"]})
    assert ranked[0]["provider_id"] == "next"


def test_successor_context_is_not_set_by_failed_attempt() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    provider = _provider("first", successors=["next"], effects=[{
        "id": "effect",
        "probe": {"id": "lines", "type": "line_count", "rect": [0, 0, 100, 100], "target": 1, "tolerance": 0, "coordinate_space": "logical"},
        "expected": "pass",
    }])
    record_attempt(cursor, provider, executed=False)
    assert cursor["successor_ids"] == []
    update_successor_context(cursor, provider, "contradicted")
    assert cursor["successor_ids"] == []
    update_successor_context(cursor, provider, "confirmed")
    assert cursor["successor_ids"] == ["next"]


def test_unknown_effect_probe_does_not_block_successor_context() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    provider = _provider("first", successors=["next"], effects=[{
        "id": "effect",
        "probe": {"id": "missing", "type": "template", "rect": [0, 0, 10, 10], "template_bbox": [0, 0, 5, 5], "coordinate_space": "logical"},
        "expected": "pass",
    }])
    update_successor_context(cursor, provider, "unverified", [{"matched": None}])
    assert cursor["successor_ids"] == ["next"]


def test_provider_is_suppressed_for_visit_without_global_degradation() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    provider = _provider("layout_specific")
    operation = {"providers": [provider]}
    record_attempt(cursor, provider, executed=False)
    assert rank_providers(operation, cursor, {}) == []
    record_provider_result(operation, "layout_specific", "resolver_miss", visit_id="001:1")
    assert provider["status"] == "proposed"
    assert provider["result_counts"] == {"resolver_miss": 1}


def test_repeatable_provider_reopens_only_after_confirmed_effect() -> None:
    cursor = cursor_for({}, "001:1", "advance")
    provider = _provider("repeat", repeat_policy="after_confirmed_effect")
    operation = {"providers": [provider]}
    record_attempt(cursor, provider)
    assert rank_providers(operation, cursor, {}) == []
    mark_confirmed_effect(cursor)
    assert rank_providers(operation, cursor, {}) == [provider]


def test_locator_falls_back_from_missing_template_to_logical_point() -> None:
    provider = normalize_provider({
        "provider_id": "target",
        "locators": [
            {"type": "region_template", "template_path": "missing.png", "search_rect": [0, 0, 500, 500], "coordinate_space": "logical"},
            _point(700, 800),
        ],
    })
    action, info = resolve_provider(provider, np.zeros((2, 2, 3), dtype=np.uint8), FakeVision())
    assert action == {"type": "click", "x": 700, "y": 800, "coordinate_space": "logical", "brief": ""}
    assert info["locator_index"] == 1


def test_text_target_returns_real_coordinate_once() -> None:
    provider = normalize_provider({
        "provider_id": "target",
        "locators": [{"type": "text_target", "rect": [0, 0, 500, 500], "texts": ["target"], "coordinate_space": "logical"}],
    })
    action, _ = resolve_provider(provider, np.zeros((2, 2, 3), dtype=np.uint8), FakeVision())
    assert action["coordinate_space"] == "real"
    assert (action["x"], action["y"]) == (140, 210)


def test_effect_hint_is_an_observable_hypothesis() -> None:
    provider = _provider("target", effects=[{
        "id": "enabled",
        "probe": {"id": "lines", "type": "line_count", "rect": [0, 0, 100, 100], "target": 2, "tolerance": 0, "coordinate_space": "logical"},
        "expected": "becomes_pass",
    }])
    before = np.zeros((2, 2, 3), dtype=np.uint8)
    after = np.zeros((2, 2, 3), dtype=np.uint8)
    after[0, 0, 0] = 2
    result, details = evaluate_effect(provider, {"enabled": "fail"}, after, FakeVision())
    assert result == "confirmed"
    assert details[0]["before"] == "fail"
    assert details[0]["after"] == "pass"


def test_watch_learns_guards_for_current_and_successor(tmp_path: Path) -> None:
    first = _provider("first", successors=["second"], watches=[{
        "id": "button_state",
        "rect": [0, 0, 1000, 1000],
        "modalities": ["appearance"],
        "after_provider": "second",
        "coordinate_space": "logical",
    }])
    second = _provider("second")
    operation = {"providers": [first, second]}
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[30:60, 40:80] = 255

    class LearningVision(FakeVision):
        class Mapper:
            @staticmethod
            def rect_to_real(rect):
                return 0, 0, 100, 100

            @staticmethod
            def point_to_logical(x, y):
                return x * 10, y * 10

        mapper = Mapper()

        def crop_rect(self, frame, rect):
            del rect
            return frame

        def match_template(self, frame, path, rect, threshold=0.82):
            del rect, threshold
            is_before = "before" in path.name
            return (bool(frame.mean() == 0) == is_before, (1, 1), 1.0)

    vision = LearningVision()
    observed = observe_successful_transition(
        operation, first, before, after, visit_id="001:1", state_dir=tmp_path, vision=vision
    )
    assert observed[0]["template_bbox"] is not None
    assert first["guards"][0]["status"] == "provisional"
    assert second["guards"][0]["status"] == "provisional"
    observe_successful_transition(
        operation, first, before, after, visit_id="001:2", state_dir=tmp_path, vision=vision
    )
    assert first["guards"][0]["status"] == "active"
    assert second["guards"][0]["status"] == "active"


def test_repair_patch_keeps_local_and_family_candidates_separate() -> None:
    handler = handler_from_bootstrap([{
        "operation": "advance",
        "providers": [{"provider_id": "old", "locators": [_point(100, 100)]}],
    }])
    touched = apply_handler_patch(handler, {
        "providers": [{"provider_id": "local", "locators": [_point(200, 200)]}],
        "generalization_candidates": [{
            "provider_id": "shared",
            "locators": [{"type": "text_target", "rect": [0, 0, 500, 500], "texts": ["target"], "coordinate_space": "logical"}],
        }],
    }, operation="advance")
    assert touched == {"local", "shared"}
    providers = {item["provider_id"]: item for item in handler["operation_policies"]["advance"]["providers"]}
    assert providers["local"]["scope"] == "instance"
    assert providers["shared"]["scope"] == "family"
    assert providers["old"]["locators"][0]["x"] == 100


def test_repair_provider_with_same_operation_and_overlapping_region_is_archived_and_merged() -> None:
    region_a = {"type": "click_region", "rect": [700, 800, 950, 930], "preferred_point": [850, 880], "coordinate_space": "logical"}
    region_b = {"type": "click_region", "rect": [720, 810, 960, 940], "preferred_point": [820, 890], "coordinate_space": "logical"}
    existing = {
        "operation": "advance",
        "providers": [normalize_provider({"provider_id": "confirm", "operation_key": "confirm_selection", "base_priority": 10, "locators": [region_a]})],
    }
    incoming = {
        "operation": "advance",
        "providers": [normalize_provider({"provider_id": "confirm_second_round", "operation_key": "confirm_selection", "base_priority": 20, "locators": [region_b]})],
    }
    merged, touched = merge_operation(existing, incoming)
    assert [item["provider_id"] for item in merged["providers"]] == ["confirm"]
    assert touched == {"confirm"}
    candidates = merged["providers"][0]["locators"][0]["candidate_points"]
    assert [850, 880] in [item["point"] for item in candidates]
    assert [820, 890] in [item["point"] for item in candidates]
    assert merged["provider_archive"][0]["provider"]["provider_id"] == "confirm_second_round"


def test_click_region_uses_an_untried_candidate() -> None:
    provider = normalize_provider({
        "provider_id": "confirm",
        "locators": [{
            "type": "click_region",
            "rect": [700, 800, 950, 930],
            "preferred_point": [850, 880],
            "candidate_points": [[850, 880], [820, 890]],
            "coordinate_space": "logical",
        }],
    })
    cursor = {"locator_attempts": {"confirm": [[850, 880]]}}
    action, info = resolve_provider(provider, np.zeros((2, 2, 3), dtype=np.uint8), FakeVision(), cursor)
    assert (action["x"], action["y"]) == (820, 890)
    assert info["candidate_point"] == [820, 890]


def test_materialize_provider_templates_includes_effect_probes(tmp_path: Path) -> None:
    provider = _provider("select", effects=[{
        "id": "selected",
        "probe": {
            "id": "selected_probe",
            "type": "template",
            "template_bbox": [0, 0, 1, 1],
            "rect": [0, 0, 2, 2],
            "coordinate_space": "logical",
        },
        "expected": "becomes_pass",
    }])
    handler = {"operation_policies": {"advance": {"providers": [provider]}}}
    materialize_provider_templates(handler, tmp_path, np.zeros((2, 2, 3), dtype=np.uint8), FakeVision(), {"select"})
    path = Path(provider["effects"][0]["probe"]["template_path"])
    assert path.name == "reactive_effect_select_1.png"
    assert path.is_file()


def test_family_provider_requires_success_on_two_visits_to_activate() -> None:
    provider = _provider("shared", scope="family", status="canary")
    operation = {"providers": [provider]}
    record_provider_result(operation, "shared", "confirmed", visit_id="001:1")
    assert provider["status"] == "canary"
    record_provider_result(operation, "shared", "confirmed", visit_id="001:2")
    assert provider["status"] == "active"


def test_repair_parser_supports_one_bounded_query_round() -> None:
    assert parse_repair_response('{"decision":"query","image_queries":[{"cell_id":"a"},{"cell_id":"b"},{"cell_id":"c"}],"reason":"zoom"}') == {
        "decision": "query", "image_queries": ["a", "b"], "reason": "zoom"
    }
    repaired = parse_repair_response('{"decision":"repair","local_patch":{"providers":[]},"generalization_candidates":[],"reason":"done"}')
    assert repaired["decision"] == "repair"
    misidentified = parse_repair_response('{"decision":"state_misidentified","confidence":"high","visible_evidence":["different title"],"reason":"wrong surface"}')
    assert misidentified == {
        "decision": "state_misidentified",
        "confidence": "high",
        "visible_evidence": ["different title"],
        "reason": "wrong surface",
    }
    unsupported = parse_repair_response('{"decision":"state_misidentified","confidence":"high","visible_evidence":[],"reason":"guess"}')
    assert unsupported["confidence"] == "low"
