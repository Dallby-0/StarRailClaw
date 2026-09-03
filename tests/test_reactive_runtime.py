from state_machine.page_handler.reactive import clear_visit, cursor_for, reset_run_local_progress
from state_machine.visit import ensure_page_visit, start_new_visit


def test_cursor_is_scoped_to_visit_and_operation() -> None:
    runtime: dict = {}
    first = cursor_for(runtime, "001:1", "advance")
    assert cursor_for(runtime, "001:1", "advance") is first
    assert cursor_for(runtime, "001:1", "inspect") is not first
    clear_visit(runtime, "001:1")
    assert runtime["reactive_visits"] == {}


def test_new_run_discards_only_run_local_reactive_state() -> None:
    runtime = {
        "page_visit": {"state_id": "001", "visit_id": "001:9"},
        "page_visit_seq": 9,
        "reactive_visits": {"001:9|advance": {"total_actions": 4}},
        "reactive_total_actions": 20,
        "reactive_total_repairs": 3,
    }
    reset_run_local_progress(runtime)
    assert runtime["page_visit"] is None
    assert runtime["page_visit_seq"] == 9
    assert runtime["reactive_visits"] == {}
    assert runtime["reactive_total_actions"] == 0
    assert runtime["reactive_total_repairs"] == 0


def test_page_visits_reuse_state_and_change_on_reentry() -> None:
    runtime: dict = {}
    first = ensure_page_visit(runtime, "001")
    assert ensure_page_visit(runtime, "001")["visit_id"] == first["visit_id"]
    second = start_new_visit(runtime, "001", reason="reentry")
    assert second["visit_id"] != first["visit_id"]
