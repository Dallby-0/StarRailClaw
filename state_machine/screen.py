from __future__ import annotations

import time

import cv2

from state_machine.constants import (
    SCREEN_CHANGE_DIFF_THRESHOLD,
    UNKNOWN_STABILITY_DIFF_THRESHOLD,
    UNKNOWN_STABILITY_SAMPLE_INTERVAL_S,
)
from state_machine.logger import FsmRunLogger


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _screen_changed(before_rgb, after_rgb, threshold: float = SCREEN_CHANGE_DIFF_THRESHOLD) -> tuple[bool, float]:
    if before_rgb is None or after_rgb is None:
        return False, 0.0
    if before_rgb.shape != after_rgb.shape:
        return True, 1.0
    diff = cv2.absdiff(before_rgb, after_rgb)
    score = float(diff.mean()) / 255.0
    return score >= threshold, score


def _normalize_frame(frame_rgb):
    if frame_rgb.shape[1] != 1280 or frame_rgb.shape[0] != 720:
        return cv2.resize(frame_rgb, (1280, 720), interpolation=cv2.INTER_LINEAR)
    return frame_rgb


def _capture_frame(emulator):
    return _normalize_frame(emulator.screenshot(prefer_png=True))


def _wait_for_screen_stable(
    emulator,
    first_frame=None,
    *,
    threshold: float = UNKNOWN_STABILITY_DIFF_THRESHOLD,
    interval_s: float = UNKNOWN_STABILITY_SAMPLE_INTERVAL_S,
    logger: FsmRunLogger | None = None,
    label: str = "screen",
    event: str = "screen_stability_check",
    return_unstable_latest: bool = True,
    max_checks: int = 1,
):
    first_frame = _capture_frame(emulator) if first_frame is None else _normalize_frame(first_frame)
    latest = first_frame
    for check_idx in range(1, max(1, max_checks) + 1):
        time.sleep(interval_s)
        second = _capture_frame(emulator)
        time.sleep(interval_s)
        third = _capture_frame(emulator)

        changed_12, diff_12 = _screen_changed(first_frame, second, threshold)
        changed_23, diff_23 = _screen_changed(second, third, threshold)
        changed_13, diff_13 = _screen_changed(first_frame, third, threshold)
        stable = not (changed_12 or changed_23 or changed_13)
        latest = third
        _log(
            logger,
            (
                f"[fsm][{label}][stability] "
                f"stable={stable} check={check_idx}/{max(1, max_checks)} "
                f"diff12={diff_12:.4f} diff23={diff_23:.4f} diff13={diff_13:.4f} threshold={threshold:.4f}"
            ),
            event,
            stable=stable,
            check=check_idx,
            max_checks=max(1, max_checks),
            diff12=diff_12,
            diff23=diff_23,
            diff13=diff_13,
            threshold=threshold,
            interval_s=interval_s,
        )
        if stable:
            return latest
        first_frame = latest
    if return_unstable_latest:
        return latest
    return None


def _wait_for_unknown_screen_stable(
    emulator,
    first_frame,
    *,
    threshold: float = UNKNOWN_STABILITY_DIFF_THRESHOLD,
    interval_s: float = UNKNOWN_STABILITY_SAMPLE_INTERVAL_S,
    logger: FsmRunLogger | None = None,
):
    return _wait_for_screen_stable(
        emulator,
        first_frame,
        threshold=threshold,
        interval_s=interval_s,
        logger=logger,
        label="unknown",
        event="unknown_stability_check",
        return_unstable_latest=False,
    )
