from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent.behavior_tree.coord_mapper import CoordinateMapper
from state_machine.page_handler.runner import run_page_handler
from state_machine.page_handler.store import handler_from_bootstrap


def _point(x: int, y: int) -> dict:
    return {"type": "point", "x": x, "y": y, "coordinate_space": "logical", "source": "bootstrap"}


def _line_probe() -> dict:
    return {"id": "ready", "type": "line_count", "rect": [0, 0, 100, 100], "min": 2, "max": 2, "coordinate_space": "logical"}


class FakeVision:
    def detect_text_lines(self, frame, rect):
        del rect
        return {"available": True, "line_count": int(frame[0, 0, 0]), "lines": []}

    def save_template_from_rect(self, frame, rect, path):
        del frame, rect
        Path(path).write_bytes(b"template")

    def match_template(self, frame, path, rect, threshold=0.82):
        del frame, rect, threshold
        return Path(path).exists(), (100, 100), 0.9


class FakeEmulator:
    def __init__(self) -> None:
        self.taps: list[tuple[int, int]] = []

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))


def test_two_provider_chain_runs_locally_and_promotes_deferred_hint(tmp_path: Path, monkeypatch) -> None:
    handler = handler_from_bootstrap([{
        "operation": "advance",
        "is_default": True,
        "intent_scope": "intent_specific",
        "intent_effect": "advance",
        "expected_event": "advanced",
        "providers": [
            {
                "provider_id": "first",
                "base_priority": 80,
                "locators": [_point(250, 500)],
                "effect_hints": [{"id": "first_effect", "probe": _line_probe(), "expected": "becomes_pass"}],
                "successors": ["second"],
            },
            {
                "provider_id": "second",
                "base_priority": 40,
                "locators": [_point(750, 800)],
                "deferred_hints": [{
                    "id": "second_visible",
                    "type": "template",
                    "template_bbox": [100, 100, 200, 200],
                    "rect": [50, 50, 250, 250],
                    "coordinate_space": "logical",
                    "materialize_after": "first",
                }],
                "effect_hints": [{"id": "second_effect", "probe": _line_probe(), "expected": "pass"}],
                "successors": [],
            },
        ],
    }])
    state_meta = {"state_id": "001", "samples": [], "page_handler": handler}
    (tmp_path / "state.json").write_text(json.dumps(state_meta), encoding="utf-8")
    start = np.zeros((4, 4, 3), dtype=np.uint8)
    after_first = start.copy()
    after_first[0, 0, 0] = 2
    after_second = after_first.copy()
    frames = iter([after_first, after_second])
    monkeypatch.setattr("state_machine.page_handler.runner._wait_for_screen_stable", lambda *args, **kwargs: next(frames))
    monkeypatch.setattr("state_machine.page_handler.runner._save_runtime", lambda runtime: None)

    emulator = FakeEmulator()
    current_match = SimpleNamespace(state_id="001", success=True)
    llm = SimpleNamespace()
    ok, final_frame = run_page_handler(
        emulator=emulator,
        mapper=CoordinateMapper(real_w=1280, real_h=720),
        vision=FakeVision(),
        state_id="001",
        state_dir=tmp_path,
        action_id="operation_main",
        start_frame=start,
        matches_provider=lambda frame: [current_match],
        runtime={},
        llm=llm,
        llm_session_id="test",
        system_prompt="test",
        graph={"nodes": [], "edges": []},
        prefer_reachable_first=False,
    )

    assert ok is True
    assert final_frame is after_second
    assert emulator.taps == [(320, 360), (960, 576)]
    persisted = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    providers = {item["provider_id"]: item for item in persisted["page_handler"]["operation_policies"]["advance"]["providers"]}
    assert providers["first"]["result_counts"] == {"confirmed": 1}
    assert providers["second"]["result_counts"] == {"confirmed": 1}
    assert providers["second"]["hints"][0]["id"] == "second_visible"
    assert providers["second"]["hints"][0]["status"] == "confirmed"


def test_confirmed_same_state_reentry_starts_fresh_visit(tmp_path: Path, monkeypatch) -> None:
    handler = handler_from_bootstrap([{
        "operation": "advance",
        "is_default": True,
        "intent_scope": "intent_specific",
        "intent_effect": "advance",
        "expected_event": "advanced",
        "providers": [{
            "provider_id": "next",
            "base_priority": 80,
            "locators": [_point(250, 500)],
            "effect_hints": [{"id": "changed", "probe": _line_probe(), "expected": "becomes_pass"}],
        }],
    }])
    state_meta = {"state_id": "001", "samples": [], "page_handler": handler}
    (tmp_path / "state.json").write_text(json.dumps(state_meta), encoding="utf-8")
    before = np.ones((4, 4, 3), dtype=np.uint8)
    after = before.copy()
    after[0, 0, 0] = 2
    monkeypatch.setattr("state_machine.page_handler.runner._wait_for_screen_stable", lambda *args, **kwargs: after)
    monkeypatch.setattr("state_machine.page_handler.runner._save_runtime", lambda runtime: None)

    class Match:
        def __init__(self, text: str):
            self.state_id = "001"
            self.success = True
            self.condition_results = [{"kind": "text_line_contains", "ocr_lines": [text]}]

    emulator = FakeEmulator()
    calls = iter([Match("页面 A"), Match("页面 B")])
    runtime: dict = {}
    ok, final_frame = run_page_handler(
        emulator=emulator,
        mapper=CoordinateMapper(real_w=1280, real_h=720),
        vision=FakeVision(),
        state_id="001",
        state_dir=tmp_path,
        action_id="operation_main",
        start_frame=before,
        matches_provider=lambda frame: [next(calls)],
        runtime=runtime,
        llm=SimpleNamespace(),
        llm_session_id="test",
        system_prompt="test",
        graph={"nodes": [], "edges": []},
        prefer_reachable_first=False,
    )

    assert ok is True
    assert final_frame is after
    assert runtime["page_visit"]["visit_id"] == "001:2"
    assert runtime["last_transition_ok"] is True
    assert runtime["reactive_visits"] == {}


def test_scene_3d_rejects_non_preset_provider(tmp_path: Path, monkeypatch) -> None:
    handler = handler_from_bootstrap([{
        "operation": "advance",
        "is_default": True,
        "intent_scope": "intent_specific",
        "intent_effect": "advance",
        "expected_event": "advanced",
        "providers": [{"provider_id": "point", "locators": [_point(250, 500)]}],
    }])
    state_meta = {"state_id": "001", "scene_mode": "scene_3d", "samples": [], "page_handler": handler}
    (tmp_path / "state.json").write_text(json.dumps(state_meta), encoding="utf-8")
    emulator = FakeEmulator()
    ok, _ = run_page_handler(
        emulator=emulator,
        mapper=CoordinateMapper(real_w=1280, real_h=720),
        vision=FakeVision(),
        state_id="001",
        state_dir=tmp_path,
        action_id="operation_main",
        start_frame=np.zeros((4, 4, 3), dtype=np.uint8),
        matches_provider=lambda frame: [],
        runtime={},
        llm=SimpleNamespace(),
        llm_session_id="test",
        system_prompt="test",
        graph={"nodes": [], "edges": []},
        prefer_reachable_first=False,
    )
    assert ok is False
    assert emulator.taps == []
