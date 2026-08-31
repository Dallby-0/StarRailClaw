from __future__ import annotations

import json
from pathlib import Path

from agent.fsm_cytoscape_gui import _state_occurrences
from state_machine.logger import FsmRunLogger


def _write_events(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_logger_adds_sequence_and_observation_context(tmp_path: Path) -> None:
    logger = FsmRunLogger("session-test", "run1", debug_dir=tmp_path / "debug")
    logger.set_context(loop_id="loop-000001", observation_id="obs-000001")
    logger.event("loop_start")
    logger.set_context(frame_id="frame-000001", artifact_path="frames/frame-000001.png")
    logger.event("state_selected", state_id="015")

    rows = [json.loads(line) for line in logger.events_path.read_text(encoding="utf-8").splitlines()]
    assert [row["seq"] for row in rows] == [1, 2]
    assert rows[0]["loop_id"] == "loop-000001"
    assert "frame_id" not in rows[0]
    assert rows[1]["observation_id"] == "obs-000001"
    assert rows[1]["frame_id"] == "frame-000001"
    assert rows[1]["artifact_path"] == "frames/frame-000001.png"
    assert logger.frames_dir.is_dir()


def test_state_occurrences_groups_loop_and_builds_artifact_url(tmp_path: Path) -> None:
    workspace = tmp_path / "task"
    graph_path = workspace / "state_graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text("{}", encoding="utf-8")
    run_dir = workspace / "debug" / "sessions" / "session-a" / "runs" / "run_abc"
    frame_path = run_dir / "frames" / "frame-000001.png"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"png")
    common = {
        "session_id": "session-a",
        "run_id": "abc",
        "loop_id": "loop-000001",
        "observation_id": "obs-000001",
        "frame_id": "frame-000001",
        "artifact_path": "frames/frame-000001.png",
    }
    _write_events(
        run_dir / "events.jsonl",
        [
            {**common, "ts": "2026-09-01T00:00:00.000Z", "seq": 1, "event": "loop_start"},
            {**common, "ts": "2026-09-01T00:00:00.100Z", "seq": 2, "event": "frame_captured"},
            {
                **common,
                "ts": "2026-09-01T00:00:00.200Z",
                "seq": 3,
                "event": "state_selected",
                "state_id": "015",
                "match": {"passed_enabled": 1, "total_enabled": 1, "success": True},
            },
            {
                **common,
                "ts": "2026-09-01T00:00:00.300Z",
                "seq": 4,
                "event": "controller_exhausted",
                "state_id": "015",
                "visit_id": "015:1",
            },
            {
                **common,
                "ts": "2026-09-01T00:00:01.000Z",
                "seq": 5,
                "event": "state_selected",
                "state_id": "999",
            },
        ],
    )

    occurrences = _state_occurrences(graph_path, "015", limit=10)

    assert len(occurrences) == 1
    occurrence = occurrences[0]
    assert occurrence["session_id"] == "session-a"
    assert occurrence["run_id"] == "abc"
    assert occurrence["outcome_event"] == "controller_exhausted"
    assert occurrence["visit_ids"] == ["015:1"]
    assert occurrence["match"]["success"] is True
    assert occurrence["image_url"].startswith("/api/run-artifact?")
    assert "session_id=session-a" in occurrence["image_url"]
    assert [event["seq"] for event in occurrence["events"]] == [1, 2, 3, 4, 5]


def test_state_occurrences_supports_legacy_loop_boundaries(tmp_path: Path) -> None:
    workspace = tmp_path / "task"
    graph_path = workspace / "state_graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text("{}", encoding="utf-8")
    events_path = workspace / "debug" / "sessions" / "session-old" / "runs" / "run_old" / "events.jsonl"
    _write_events(
        events_path,
        [
            {"ts": "1", "event": "loop_start", "session_id": "session-old", "run_id": "old"},
            {"ts": "2", "event": "state_selected", "session_id": "session-old", "run_id": "old", "state_id": "015"},
            {"ts": "3", "event": "page_handler_no_strategy", "session_id": "session-old", "run_id": "old", "state_id": "015"},
            {"ts": "4", "event": "loop_start", "session_id": "session-old", "run_id": "old"},
        ],
    )

    occurrences = _state_occurrences(graph_path, "015", limit=10)

    assert len(occurrences) == 1
    assert occurrences[0]["outcome_event"] == "page_handler_no_strategy"
    assert occurrences[0]["image_url"] == ""
    assert [event["event"] for event in occurrences[0]["events"]] == [
        "loop_start",
        "state_selected",
        "page_handler_no_strategy",
    ]
