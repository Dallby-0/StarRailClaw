from __future__ import annotations

from agent.behavior_tree.coord_mapper import CoordinateMapper
from state_machine.matching import _eval_state_match
from state_machine.page_handler.store import handler_from_bootstrap
from state_machine.page_identity import build_match_clauses, normalize_elements, temporal_continuation_evidence


class FakeVision:
    def __init__(self) -> None:
        self.mapper = CoordinateMapper()

    def ocr_blocks(self, frame, rect, white_text=False):
        del frame, rect, white_text
        return [
            {"text": "差分宇宙", "conf": 0.95, "bbox": (80, 14, 90, 22)},
            {"text": "事件", "conf": 0.98, "bbox": (80, 42, 45, 22)},
        ]

    def ocr(self, frame, rect, white_text=False):
        del frame, white_text
        text = "事件" if rect == [62, 58, 98, 90] else ""
        return [{"text": text}] if text else []

    def match_template(self, frame, path, rect, threshold=0.8):
        del frame, path, rect, threshold
        return False, None, 0.0


def test_multiline_llm_text_is_split_into_observable_single_lines() -> None:
    elements = normalize_elements(
        [{
            "type": "text_line",
            "text": "差分宇宙 事件",
            "bbox": [60, 15, 150, 100],
            "brief": "左上角标题",
            "role": "identity",
            "stability": "high",
            "discrimination": "high",
        }],
        FakeVision(),
        object(),
    )
    assert [item["text"] for item in elements] == ["差分宇宙", "事件"]
    assert all(item["normalized_from_multiline"] for item in elements)
    assert all(item["role"] == "identity" for item in elements)


def test_invalid_llm_bbox_is_ignored_instead_of_crashing() -> None:
    elements = normalize_elements(
        [{
            "type": "pattern",
            "bbox": [28, "2022-01-01", 28, "2022-01-01"],
            "brief": "bad model output",
            "role": "identity",
            "stability": "high",
            "discrimination": "high",
        }],
        FakeVision(),
        object(),
    )
    assert elements == []


def test_identity_clauses_are_or_while_supports_are_conjunctive() -> None:
    conditions = [
        {"id": "event", "role": "identity", "condition_status": "active"},
        {"id": "mode", "role": "identity", "condition_status": "active"},
        {"id": "layout", "role": "identity_support", "condition_status": "active"},
    ]
    assert build_match_clauses(conditions) == [{"all": ["event"]}, {"all": ["mode"]}]
    assert build_match_clauses(conditions[2:]) == [{"all": ["layout"]}]


def test_match_succeeds_when_any_identity_clause_matches(tmp_path) -> None:
    meta = {
        "state_id": "015",
        "match_conditions": [
            {"id": "mode", "enabled": True, "kind": "text_line_contains", "params": {"text": "差分宇宙", "rect": [1, 1, 2, 2]}},
            {"id": "event", "enabled": True, "kind": "text_line_contains", "params": {"text": "事件", "rect": [62, 58, 98, 90]}},
        ],
        "match_clauses": [{"all": ["mode"]}, {"all": ["event"]}],
    }
    result = _eval_state_match(meta, tmp_path, FakeVision(), object())
    assert result.success is True
    assert result.matched_clause == 1


def test_high_stability_support_can_continue_temporal_surface() -> None:
    meta = {
        "match_conditions": [{
            "id": "event_support",
            "role": "identity_support",
            "stability": "high",
            "condition_status": "active",
            "kind": "text_line_contains",
            "params": {"text": "事件", "rect": [62, 58, 98, 90]},
        }]
    }
    evidence = temporal_continuation_evidence(meta, FakeVision(), object())
    assert evidence["accepted"] is True
    assert evidence["score"] == 2


def test_new_bootstrap_persists_only_reactive_controller() -> None:
    handler = handler_from_bootstrap([{
        "operation": "advance_event",
        "is_default": True,
        "intent_scope": "intent_specific",
        "intent_effect": "advance",
        "safety": "low_risk",
        "expected_event": "event_advanced",
        "intent_routes": [],
        "steps": [{
            "step_id": "advance",
            "resolver": {"type": "fixed_point", "x": 500, "y": 800},
            "expected_after": {"state_relation": "may_leave", "reentry_policy": "same_visit"},
            "emits_on_success": {"type": "event_advanced"},
            "brief": "推进事件",
        }],
    }])
    policy = handler["operation_policies"]["advance_event"]
    assert policy["controller"]["type"] == "reactive_local"
    assert policy["strategies"] == []
