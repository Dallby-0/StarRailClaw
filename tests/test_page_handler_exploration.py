from __future__ import annotations

import numpy as np

from state_machine.page_handler.exploration import run_exploration


class _Mapper:
    def point_to_real(self, x, y):
        return x, y


class _Emulator:
    def __init__(self):
        self.taps = []
        self.index = 0

    def tap(self, x, y):
        self.taps.append((x, y))
        self.index += 1

    def screenshot(self, prefer_png=True):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :, :] = min(self.index, 255)
        return frame


class _Match:
    def __init__(self, state_id, success):
        self.state_id = state_id
        self.success = success


def test_horizontal_probe_stops_when_page_leaves_state():
    emulator = _Emulator()
    start = np.zeros((8, 8, 3), dtype=np.uint8)

    def matches(frame):
        return [_Match("004", emulator.index != 1)]

    ok, frame, info = run_exploration(
        emulator=emulator,
        mapper=_Mapper(),
        vision=None,
        state_id="004",
        operation="select_battle_station_and_confirm",
        frame=start,
        matches_provider=matches,
        profiles=[{"id": "probe", "kind": "horizontal_probe", "y": 10, "x_start": 1, "x_end": 9, "samples": 5}],
        pause_s=0,
    )

    assert ok is True
    assert len(emulator.taps) == 1
    assert info["result"] == "left_state"
    assert frame is not start


def test_probe_reports_partial_progress_without_leaving_state():
    emulator = _Emulator()
    start = np.zeros((8, 8, 3), dtype=np.uint8)

    def matches(frame):
        return [_Match("004", True)]

    ok, _, info = run_exploration(
        emulator=emulator,
        mapper=_Mapper(),
        vision=None,
        state_id="004",
        operation="select_battle_station_and_confirm",
        frame=start,
        matches_provider=matches,
        profiles=[{"id": "probe", "kind": "horizontal_probe", "y": 10, "x_start": 1, "x_end": 9, "samples": 2}],
        pause_s=0,
    )

    assert ok is True
    assert info["result"] == "partial_progress"
    assert len(emulator.taps) == 2
