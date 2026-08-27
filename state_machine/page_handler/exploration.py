"""Bounded, feedback-driven local exploration for low-risk page handlers.

Exploration is deliberately separate from learned strategies.  It is a local
fallback for pages that belong to a known operation but whose exact geometry
does not match the stored strategy.  Every probe is followed by an observation
and the profile stops as soon as the page leaves the current state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from state_machine.screen import _normalize_frame, _screen_changed


DEFAULT_PROFILES: dict[str, list[dict[str, Any]]] = {
    "station_select": [
        {
            "id": "station_card_horizontal_probe",
            "kind": "horizontal_probe",
            "y": 400,
            "x_start": 500,
            "x_end": 780,
            "samples": 5,
            "stop_on_change": True,
        },
        {
            "id": "confirm_text_or_horizontal_probe",
            "kind": "confirm_search",
            "region": [430, 500, 420, 180],
            "keywords": ["确定", "确认", "前往", "出发"],
            "x_start": 500,
            "x_end": 780,
            "y": 600,
            "samples": 5,
        },
    ],
}


def _log(logger, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
    else:
        logger.text(message, event=event, **fields)


def _points(profile: dict[str, Any]) -> list[tuple[int, int]]:
    samples = max(1, int(profile.get("samples", 1) or 1))
    x_start = int(profile.get("x_start", 0) or 0)
    x_end = int(profile.get("x_end", x_start) or x_start)
    y = int(profile.get("y", 0) or 0)
    if samples == 1:
        return [(x_start, y)]
    return [
        (round(x_start + (x_end - x_start) * index / (samples - 1)), y)
        for index in range(samples)
    ]


def _contains_keyword(entries: list[dict[str, Any]], keywords: list[str]) -> tuple[int, int] | None:
    lowered = [str(item).lower() for item in keywords if str(item)]
    for entry in entries:
        text = str(entry.get("text") or "").lower()
        if any(keyword.lower() in text for keyword in lowered):
            center = entry.get("center")
            if isinstance(center, (tuple, list)) and len(center) >= 2:
                return int(center[0]), int(center[1])
    return None


def _left_state(matches_provider, frame, state_id: str) -> bool:
    try:
        matches = matches_provider(frame)
    except Exception:
        return False
    for match in matches:
        if str(getattr(match, "state_id", "")) == str(state_id):
            return not bool(getattr(match, "success", False))
    return True


def _capture(emulator):
    return _normalize_frame(emulator.screenshot(prefer_png=True))


def run_exploration(
    *,
    emulator,
    mapper,
    vision,
    state_id: str,
    operation: str,
    frame,
    matches_provider,
    profiles: list[dict[str, Any]] | None = None,
    logger=None,
    max_actions: int = 10,
    pause_s: float = 0.12,
) -> tuple[bool, Any, dict[str, Any]]:
    """Run at most ``max_actions`` local probes and return progress/frame/info.

    A changed frame is considered local progress, but the probe continues into
    the next profile so a card-selection change can be followed by confirm.
    Leaving the current state is an immediate success and stops all probing.
    """
    chosen = profiles
    if not isinstance(chosen, list) or not chosen:
        family = "station_select" if any(token in operation.lower() for token in ("station", "next", "confirm")) else ""
        chosen = DEFAULT_PROFILES.get(family, [])
    current = frame
    actions = 0
    changed_any = False
    attempts: list[dict[str, Any]] = []
    for profile in chosen:
        if actions >= max_actions or not isinstance(profile, dict):
            break
        kind = str(profile.get("kind") or "horizontal_probe")
        candidates: list[tuple[int, int]] = []
        if kind == "confirm_search":
            region = profile.get("region") if isinstance(profile.get("region"), list) else None
            keywords = profile.get("keywords") if isinstance(profile.get("keywords"), list) else []
            if region:
                point = _contains_keyword(vision.ocr(current, region), [str(x) for x in keywords])
                if point is not None:
                    offset = profile.get("click_offset") if isinstance(profile.get("click_offset"), list) else [0, 0]
                    candidates.append((point[0] + int(offset[0] or 0), point[1] + int(offset[1] or 0)))
            candidates.extend(_points(profile))
        elif kind == "template_offset":
            # The profile may refer to a runtime template/anchor.  A missing
            # anchor simply makes this branch a no-op; it must not trigger a
            # broad screen scan.
            template = profile.get("template_path")
            region = profile.get("region") if isinstance(profile.get("region"), list) else None
            if template:
                ok, point, similarity = vision.match_template(current, Path(str(template)), region, threshold=float(profile.get("threshold", 0.0) or 0.0))
                _log(logger, f"[fsm][handler][explore] template={template} similarity={similarity:.3f} found={ok}", "page_handler_exploration_template", similarity=similarity, found=ok)
                if ok and point is not None:
                    offset = profile.get("click_offset") if isinstance(profile.get("click_offset"), list) else [0, 0]
                    candidates.append((point[0] + int(offset[0] or 0), point[1] + int(offset[1] or 0)))
        else:
            candidates = _points(profile)
        for x, y in candidates:
            if actions >= max_actions:
                break
            before = current
            rx, ry = mapper.point_to_real(int(x), int(y))
            emulator.tap(rx, ry)
            actions += 1
            time.sleep(pause_s)
            current = _capture(emulator)
            changed, diff = _screen_changed(before, current)
            left = _left_state(matches_provider, current, state_id)
            attempt = {"profile": str(profile.get("id") or kind), "point": [int(x), int(y)], "changed": changed, "diff_score": diff, "left_state": left}
            attempts.append(attempt)
            _log(logger, f"[fsm][handler][explore] profile={attempt['profile']} point=({x},{y}) changed={changed} diff={diff:.4f} left_state={left}", "page_handler_exploration_probe", **attempt)
            if left:
                return True, current, {"actions": actions, "attempts": attempts, "profile": attempt["profile"], "result": "left_state"}
            if changed:
                changed_any = True
                if bool(profile.get("stop_on_change", False)):
                    break
    result = "partial_progress" if changed_any else "no_effect"
    return changed_any, current, {"actions": actions, "attempts": attempts, "result": result}
