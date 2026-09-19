from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from sr_tools.emulator import EmulatorClient
from state_machine.logger import FsmRunLogger


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _probe_key(probe: dict[str, Any]) -> str:
    payload = {key: value for key, value in probe.items() if key not in {"id", "status"}}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def evaluate_guard(
    guard: dict[str, Any],
    frame_rgb,
    vision: VisionEngine,
    cache: dict[str, dict[str, Any]],
    *,
    allow_expensive: bool,
) -> dict[str, Any]:
    key = _probe_key(guard)
    if key in cache:
        return cache[key]
    kind = str(guard.get("type") or "")
    if kind == "text" and not allow_expensive:
        return {"result": "unknown", "reason": "expensive_probe_deferred"}
    try:
        if kind == "template":
            path = Path(str(guard.get("template_path") or ""))
            if not path.is_file():
                result = {"result": "unknown", "reason": "template_not_materialized"}
            else:
                matcher = (
                    getattr(vision, "match_appearance_template", vision.match_template)
                    if guard.get("learned_from_watch")
                    else vision.match_template
                )
                ok, point, similarity = matcher(
                    frame_rgb,
                    path,
                    guard.get("rect") if isinstance(guard.get("rect"), list) else None,
                    threshold=float(guard.get("threshold", 0.82)),
                )
                result = {"result": "pass" if ok else "fail", "point": point, "similarity": similarity}
        elif kind == "line_count":
            detected = vision.detect_text_lines(frame_rgb, guard.get("rect"))
            if not detected.get("available"):
                result = {"result": "unknown", "reason": "detector_unavailable"}
            else:
                count = int(detected.get("line_count", 0) or 0)
                target = int(guard.get("target", 1) or 1)
                tolerance = max(0, int(guard.get("tolerance", 1) or 0))
                distance = abs(count - target)
                if count <= 0:
                    result = {"result": "fail", "strength": 1.0, "line_count": count}
                elif distance == 0:
                    result = {"result": "pass", "strength": 1.0, "line_count": count}
                elif distance <= tolerance:
                    result = {"result": "pass", "strength": 0.7, "line_count": count}
                else:
                    result = {"result": "pass", "strength": 0.25, "line_count": count}
        elif kind == "text":
            entries = vision.ocr_blocks(frame_rgb, guard.get("rect"), white_text=False)
            needles = [str(value) for value in guard.get("texts", [])]
            matched = next((entry for entry in entries if any(needle in str(entry.get("text") or "") for needle in needles)), None)
            result = {
                "result": "pass" if matched is not None else "fail",
                "texts": [str(entry.get("text") or "") for entry in entries],
            }
        else:
            result = {"result": "unknown", "reason": "unsupported_hint"}
    except Exception as exc:  # noqa: BLE001 - a soft guard must not abort an action
        result = {"result": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    cache[key] = result
    return result


def provider_guards(provider: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in provider.get("guards", []) if isinstance(item, dict)]


def _effect_probe_result(detail: dict[str, Any]) -> str:
    result = str(detail.get("result") or "unknown")
    if result == "pass" and "strength" in detail and float(detail.get("strength", 0.0) or 0.0) < 0.7:
        return "fail"
    return result


def evaluate_provider_guards(
    provider: dict[str, Any],
    cursor: dict[str, Any],
    frame_rgb,
    vision: VisionEngine,
    cache: dict[str, dict[str, Any]],
    *,
    allow_expensive: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    del cursor
    details = [evaluate_guard(guard, frame_rgb, vision, cache, allow_expensive=allow_expensive) for guard in provider_guards(provider)]
    return [str(item.get("result") or "unknown") for item in details], details


def resolve_provider(provider: dict[str, Any], frame_rgb, vision: VisionEngine, cursor: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, locator in enumerate(provider.get("locators", [])):
        if not isinstance(locator, dict):
            continue
        kind = str(locator.get("type") or "")
        if kind == "click_region":
            attempted = (cursor or {}).get("locator_attempts", {}).get(str(provider.get("provider_id") or ""), [])
            if len(attempted) >= 2:
                failures.append({"locator_index": index, "locator_type": kind, "reason": "visit_point_limit"})
                continue
            candidates = [item for item in locator.get("candidate_points", []) if isinstance(item, dict)]
            candidates = sorted(candidates, key=lambda item: (-int(item.get("successes", 0) or 0), int(item.get("no_effects", 0) or 0)))
            chosen = next((item for item in candidates if item.get("point") not in attempted), None)
            if chosen is None:
                failures.append({"locator_index": index, "locator_type": kind, "reason": "all_points_attempted"})
                continue
            point = chosen["point"]
            return {"type": "click", "x": int(point[0]), "y": int(point[1]), "coordinate_space": "logical", "brief": str(provider.get("brief") or "")}, {"locator_index": index, "locator_type": kind, "candidate_point": list(point)}
        if kind == "point":
            action = {
                "type": "click",
                "x": int(locator["x"]),
                "y": int(locator["y"]),
                "coordinate_space": "logical",
                "brief": str(provider.get("brief") or ""),
            }
            return action, {"locator_index": index, "locator_type": kind, "locator_source": locator.get("source")}
        if kind == "region_template":
            path = Path(str(locator.get("template_path") or ""))
            if not path.is_file():
                failures.append({"locator_index": index, "locator_type": kind, "reason": "template_not_materialized"})
                continue
            ok, point, similarity = vision.match_template(
                frame_rgb,
                path,
                locator.get("search_rect") if isinstance(locator.get("search_rect"), list) else None,
                threshold=float(locator.get("threshold", 0.82)),
            )
            if ok and point is not None:
                return {
                    "type": "click",
                    "x": int(point[0]),
                    "y": int(point[1]),
                    "coordinate_space": "real",
                    "brief": str(provider.get("brief") or ""),
                }, {"locator_index": index, "locator_type": kind, "similarity": similarity}
            failures.append({"locator_index": index, "locator_type": kind, "reason": "template_not_found", "similarity": similarity})
            continue
        if kind == "text_target":
            try:
                entries = vision.ocr_blocks(frame_rgb, locator.get("rect"), white_text=False)
            except Exception as exc:  # noqa: BLE001
                failures.append({"locator_index": index, "locator_type": kind, "reason": f"{type(exc).__name__}: {exc}"})
                continue
            needles = [str(value) for value in locator.get("texts", [])]
            matched = next((entry for entry in entries if any(needle in str(entry.get("text") or "") for needle in needles)), None)
            if matched is not None:
                bbox = matched.get("bbox", (0, 0, 0, 0))
                center = matched.get("center")
                if not (isinstance(center, (list, tuple)) and len(center) == 2):
                    center = (int(bbox[0]) + int(bbox[2]) // 2, int(bbox[1]) + int(bbox[3]) // 2)
                return {
                    "type": "click",
                    "x": int(center[0]),
                    "y": int(center[1]),
                    "coordinate_space": "real",
                    "brief": str(provider.get("brief") or ""),
                }, {"locator_index": index, "locator_type": kind, "matched_text": matched.get("text")}
            failures.append({"locator_index": index, "locator_type": kind, "reason": "text_not_found"})
    return None, {"reason": "all_locators_failed", "failures": failures}


def capture_effect_baseline(provider: dict[str, Any], frame_rgb, vision: VisionEngine) -> dict[str, str]:
    cache: dict[str, dict[str, Any]] = {}
    baseline: dict[str, str] = {}
    for effect in provider.get("effects", []):
        if not isinstance(effect, dict) or not isinstance(effect.get("probe"), dict):
            continue
        detail = evaluate_guard(effect["probe"], frame_rgb, vision, cache, allow_expensive=True)
        baseline[str(effect.get("id") or "")] = _effect_probe_result(detail)
    return baseline


def evaluate_effect(
    provider: dict[str, Any],
    baseline: dict[str, str],
    frame_rgb,
    vision: VisionEngine,
) -> tuple[str, list[dict[str, Any]]]:
    effects = [item for item in provider.get("effects", []) if isinstance(item, dict) and isinstance(item.get("probe"), dict)]
    if not effects:
        return "unverified", []
    cache: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []
    outcomes: list[bool] = []
    for effect in effects:
        effect_id = str(effect.get("id") or "")
        before = baseline.get(effect_id, "unknown")
        after_detail = evaluate_guard(effect["probe"], frame_rgb, vision, cache, allow_expensive=True)
        after = _effect_probe_result(after_detail)
        expected = str(effect.get("expected") or "pass")
        if before == "unknown" or after == "unknown":
            matched: bool | None = None
        elif expected == "pass":
            matched = after == "pass"
        elif expected == "becomes_pass":
            matched = before != "pass" and after == "pass"
        else:
            matched = before != "fail" and after == "fail"
        details.append({"effect_id": effect_id, "expected": expected, "before": before, "after": after, "matched": matched})
        if matched is not None:
            outcomes.append(matched)
    if not outcomes:
        return "unverified", details
    return ("confirmed" if all(outcomes) else "contradicted"), details


def execute_action(
    *,
    emulator: EmulatorClient,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    state_id: str,
    action_id: str,
    action: dict[str, Any],
    action_info: dict[str, Any],
    matches_provider,
    logger: FsmRunLogger | None,
    attempt: str,
) -> bool:
    atype = str(action.get("type", "click"))
    x = int(action.get("x", 500))
    y = int(action.get("y", 500))
    coordinate_space = str(action.get("coordinate_space") or "")
    if coordinate_space == "real":
        rx, ry = x, y
        logical = None
    elif coordinate_space == "logical":
        rx, ry = mapper.point_to_real(x, y)
        logical = [x, y]
    else:
        raise ValueError(f"click coordinate_space must be explicit, got {coordinate_space!r}")
    brief = str(action.get("brief", "")).strip()
    _log(logger, f"[fsm][handler][click] real=({rx},{ry}) brief={brief}", "page_handler_click", state_id=state_id, action_id=action_id, attempt=attempt, logical=logical, real=[rx, ry], brief=brief, action_info=action_info)
    emulator.tap(rx, ry)
    return True
