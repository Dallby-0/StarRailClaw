from __future__ import annotations

from state_machine.progress_guard import blocked_strategies, clear_visit_progress, consume_continuation, continuation_for, grant_continuation, record_no_progress, record_repair, record_review, repair_exhausted, reset_run_local_progress, review_exhausted
from state_machine.page_handler.review_protocol import parse_same_state_review, result_for_same_state_verdict
from state_machine.page_handler.store import ensure_page_handler, mark_strategy_result, select_strategy
from state_machine.settlement import classify_pending_settlement
from state_machine.visit import ensure_page_visit, start_new_visit


def test_same_state_review_maps_semantic_verdicts() -> None:
    assert result_for_same_state_verdict("completed_same_visit", final_step=True) == "verified_success"
    assert result_for_same_state_verdict("completed_new_visit", final_step=True) == "verified_reentry"
    assert result_for_same_state_verdict("partial_needs_continue", final_step=True) == "partial_progress"
    assert result_for_same_state_verdict("ineffective", final_step=True) == "no_effect"
    assert result_for_same_state_verdict("wrong_effect", final_step=True) == "wrong_transition"


def test_same_state_review_requires_known_verdict_and_patch_object() -> None:
    parsed = parse_same_state_review('{"verdict":"partial_needs_continue","reason":"selected, confirm remains","handler_patch":{"operations":[]}}')
    assert parsed is not None
    assert parsed["verdict"] == "partial_needs_continue"
    assert parse_same_state_review('{"verdict":"invented"}') is None
    assert parse_same_state_review('{"verdict":"ineffective","handler_patch":[]}') is None
    assert parse_same_state_review('{"verdict":"partial_needs_continue","continuation":[]}') is None


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


def test_same_state_review_budget_is_separate_from_repair() -> None:
    runtime: dict = {}
    visit = start_new_visit(runtime, "001", reason="test")
    record_review(runtime, visit["visit_id"], "advance")
    assert not review_exhausted(runtime, visit["visit_id"], "advance")
    assert not repair_exhausted(runtime, visit["visit_id"], "advance")
    record_review(runtime, visit["visit_id"], "advance")
    assert review_exhausted(runtime, visit["visit_id"], "advance")


def test_continuation_lease_is_bounded_and_visit_scoped() -> None:
    runtime: dict = {}
    visit = start_new_visit(runtime, "015", reason="test")
    grant_continuation(runtime, visit["visit_id"], "advance_event_page", "arrow", max_additional_actions=2)
    assert continuation_for(runtime, visit["visit_id"], "advance_event_page")["strategy_id"] == "arrow"
    assert consume_continuation(runtime, visit["visit_id"], "advance_event_page", "other") is None
    assert consume_continuation(runtime, visit["visit_id"], "advance_event_page", "arrow") == 1
    assert consume_continuation(runtime, visit["visit_id"], "advance_event_page", "arrow") == 0
    assert continuation_for(runtime, visit["visit_id"], "advance_event_page") is None
    clear_visit_progress(runtime, visit["visit_id"])
    assert runtime["visit_operation_continuations"] == {}


def test_progress_unknown_neither_promotes_nor_degrades_strategy() -> None:
    strategy = {"strategy_id": "arrow", "status": "active", "level": 0, "success_count": 3, "fail_count": 1}
    handler = {"operation_policies": {"advance": {"strategies": [strategy]}}}
    mark_strategy_result(handler, "advance", "arrow", "progress_unknown")
    assert strategy["status"] == "active"
    assert strategy["success_count"] == 3
    assert strategy["fail_count"] == 1
    assert strategy["result_counts"]["progress_unknown"] == 1


def test_continuation_can_prefer_its_strategy_over_normal_ranking() -> None:
    first = {"strategy_id": "fixed", "status": "active", "level": 0}
    leased = {"strategy_id": "arrow", "status": "proposed", "level": 2}
    handler = {"operation_policies": {"advance": {"strategies": [first, leased]}}}
    assert select_strategy(handler, "advance")["strategy_id"] == "fixed"
    assert select_strategy(handler, "advance", preferred_id="arrow")["strategy_id"] == "arrow"


def test_new_run_discards_stale_visit_guards() -> None:
    runtime = {
        "page_visit": {"state_id": "015", "visit_id": "015:43"},
        "page_visit_seq": 43,
        "no_progress_guard": {"015:43|advance|fixed": {"count": 2}},
        "visit_operation_attempts": {"015:43|advance": 6},
        "visit_operation_repairs": {"015:43|advance": 1},
        "visit_operation_reviews": {"015:43|advance": 2},
        "visit_operation_continuations": {"015:43|advance": {"remaining_actions": 3}},
        "visit_operation_exploration": {"015:43|advance": True},
    }
    reset_run_local_progress(runtime)
    assert runtime["page_visit"] is None
    assert runtime["page_visit_seq"] == 43
    assert runtime["no_progress_guard"] == {}
    assert runtime["visit_operation_attempts"] == {}
    assert runtime["visit_operation_repairs"] == {}
    assert runtime["visit_operation_reviews"] == {}
    assert runtime["visit_operation_continuations"] == {}
    assert runtime["visit_operation_exploration"] == {}


def test_legacy_partial_review_reactivates_only_its_degraded_strategy() -> None:
    fixed = {
        "strategy_id": "fixed",
        "status": "degraded",
        "success_count": 0,
        "fail_count": 1,
        "result_counts": {"no_effect": 1},
    }
    unrelated = {
        "strategy_id": "other",
        "status": "degraded",
        "success_count": 0,
        "fail_count": 1,
        "result_counts": {"no_effect": 1},
    }
    meta = {
        "page_handler": {
            "operation_policies": {"advance": {"strategies": [fixed, unrelated]}},
            "episode_trace": [{
                "command": {"operation": "advance"},
                "strategy_id": "fixed",
                "result": "no_effect",
                "same_state_review": {"verdict": "partial_needs_continue"},
            }],
        }
    }
    ensure_page_handler(meta)
    assert fixed["status"] == "proposed"
    assert fixed["fail_count"] == 0
    assert fixed["result_counts"] == {"partial_progress": 1}
    assert fixed["legacy_partial_review_migrated"] is True
    assert unrelated["status"] == "degraded"
    # Migration is idempotent when the same state is loaded again.
    ensure_page_handler(meta)
    assert fixed["result_counts"] == {"partial_progress": 1}


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
