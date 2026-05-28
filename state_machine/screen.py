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


def _wait_for_unknown_screen_stable(
    emulator,
    first_frame,
    *,
    threshold: float = UNKNOWN_STABILITY_DIFF_THRESHOLD,
    interval_s: float = UNKNOWN_STABILITY_SAMPLE_INTERVAL_S,
    logger: FsmRunLogger | None = None,
):
    time.sleep(interval_s)
    second = emulator.screenshot(prefer_png=True)
    if second.shape[1] != 1280 or second.shape[0] != 720:
        second = cv2.resize(second, (1280, 720), interpolation=cv2.INTER_LINEAR)
    time.sleep(interval_s)
    third = emulator.screenshot(prefer_png=True)
    if third.shape[1] != 1280 or third.shape[0] != 720:
        third = cv2.resize(third, (1280, 720), interpolation=cv2.INTER_LINEAR)

    changed_12, diff_12 = _screen_changed(first_frame, second, threshold)
    changed_23, diff_23 = _screen_changed(second, third, threshold)
    changed_13, diff_13 = _screen_changed(first_frame, third, threshold)
    stable = not (changed_12 or changed_23 or changed_13)
    _log(
        logger,
        (
            "[fsm][unknown][stability] "
            f"stable={stable} diff12={diff_12:.4f} diff23={diff_23:.4f} diff13={diff_13:.4f} threshold={threshold:.4f}"
        ),
        "unknown_stability_check",
        stable=stable,
        diff12=diff_12,
        diff23=diff_23,
        diff13=diff_13,
        threshold=threshold,
        interval_s=interval_s,
    )
    return third if stable else None
