from __future__ import annotations

from state_machine.progress_guard import blocked_strategies, clear_visit_progress, record_no_progress, record_repair, repair_exhausted
from state_machine.page_handler.review_protocol import parse_same_state_review, result_for_same_state_verdict
from state_machine.settlement import classify_pending_settlement
from state_machine.visit import ensure_page_visit, start_new_visit


def test_same_state_review_maps_semantic_verdicts() -> None:
    assert result_for_same_state_verdict("completed_same_visit", final_step=True) == "verified_success"
    assert result_for_same_state_verdict("completed_new_visit", final_step=True) == "verified_reentry"
    assert result_for_same_state_verdict("partial_needs_continue", final_step=True) == "no_effect"
    assert result_for_same_state_verdict("ineffective", final_step=True) == "no_effect"
    assert result_for_same_state_verdict("wrong_effect", final_step=True) == "wrong_transition"


def test_same_state_review_requires_known_verdict_and_patch_object() -> None:
    parsed = parse_same_state_review('{"verdict":"partial_needs_continue","reason":"selected, confirm remains","handler_patch":{"operations":[]}}')
    assert parsed is not None
    assert parsed["verdict"] == "partial_needs_continue"
    assert parse_same_state_review('{"verdict":"invented"}') is None
    assert parse_same_state_review('{"verdict":"ineffective","handler_patch":[]}') is None


def test_no_progress_is_scoped_to_visit() -> None:
    runtime: dict = {}
    first = start_new_visit(runtime, "001", reason="test")
    record_no_progress(runtime, first["visit_id"], "confirm", "fixed", "no_effect")
    assert blocked_strategies(runtime, first["visit_id"], "confirm") == set()
    record_no_progress(runtime, first["visit_id"], "confirm", "fixed", "no_effect")
    assert blocked_strategies(runtime, first["visit_id"], "confirm") == {"fixed"}
    second = start_new_visit(runtime, "001", reason="reentry")
    assert blocked_strategies(runtime, second["visit_id"], "confirm") == set()
    clear_visit_progress(runtime, first["visit_id"])


def test_llm_repair_budget_is_persistent_per_visit() -> None:
    runtime: dict = {}
    visit = start_new_visit(runtime, "001", reason="test")
    assert not repair_exhausted(runtime, visit["visit_id"], "confirm")
    record_repair(runtime, visit["visit_id"], "confirm")
    assert repair_exhausted(runtime, visit["visit_id"], "confirm")
    next_visit = start_new_visit(runtime, "001", reason="reentry")
    assert not repair_exhausted(runtime, next_visit["visit_id"], "confirm")


def test_ensure_visit_reuses_same_state_but_changes_for_other_state() -> None:
    runtime: dict = {}
    first = ensure_page_visit(runtime, "001")
    assert ensure_page_visit(runtime, "001")["visit_id"] == first["visit_id"]
    assert ensure_page_visit(runtime, "002")["visit_id"] != first["visit_id"]


def test_pending_resolution_back_to_source_without_departure_is_no_effect() -> None:
    pending = {
        "state_id": "001",
        "expected_after": {"state_relation": "must_leave", "reentry_policy": "new_visit"},
        "departure_evidence": "weak",
    }
    assert classify_pending_settlement(pending, "001") == "no_effect"
    pending["departure_evidence"] = "confirmed"
    assert classify_pending_settlement(pending, "001") == "verified_reentry"
