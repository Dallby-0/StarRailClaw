from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from sr_tools.emulator import EmulatorClient
from state_machine.logger import FsmRunLogger
from state_machine.matching import _find_match_by_state
from state_machine.presets import run_preset


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _probe_key(probe: dict[str, Any]) -> str:
    payload = {key: value for key, value in probe.items() if key not in {"id", "status"}}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def evaluate_hint(
    hint: dict[str, Any],
    frame_rgb,
    vision: VisionEngine,
    cache: dict[str, dict[str, Any]],
    *,
    allow_expensive: bool,
) -> dict[str, Any]:
    key = _probe_key(hint)
    if key in cache:
        return cache[key]
    kind = str(hint.get("type") or "")
    if kind == "text" and not allow_expensive:
        return {"result": "unknown", "reason": "expensive_probe_deferred"}
    try:
        if kind == "template":
            path = Path(str(hint.get("template_path") or ""))
            if not path.exists():
                result = {"result": "unknown", "reason": "template_not_materialized"}
            else:
                ok, point, similarity = vision.match_template(
                    frame_rgb,
                    path,
                    hint.get("rect") if isinstance(hint.get("rect"), list) else None,
                    threshold=float(hint.get("threshold", 0.82)),
                )
                result = {"result": "pass" if ok else "fail", "point": point, "similarity": similarity}
        elif kind == "line_count":
            detected = vision.detect_text_lines(frame_rgb, hint.get("rect"))
            if not detected.get("available"):
                result = {"result": "unknown", "reason": "detector_unavailable"}
            else:
                count = int(detected.get("line_count", 0) or 0)
                passed = int(hint.get("min", 0) or 0) <= count <= int(hint.get("max", 0) or 0)
                result = {"result": "pass" if passed else "fail", "line_count": count}
        elif kind == "text":
            entries = vision.ocr_blocks(frame_rgb, hint.get("rect"), white_text=False)
            needles = [str(value) for value in hint.get("texts", [])]
            matched = next((entry for entry in entries if any(needle in str(entry.get("text") or "") for needle in needles)), None)
            result = {
                "result": "pass" if matched is not None else "fail",
                "texts": [str(entry.get("text") or "") for entry in entries],
            }
        else:
            result = {"result": "unknown", "reason": "unsupported_hint"}
    except Exception as exc:  # noqa: BLE001 - a soft hint must not abort an action
        result = {"result": "unknown", "reason": f"{type(exc).__name__}: {exc}"}
    cache[key] = result
    return result


def provider_hints(provider: dict[str, Any], cursor: dict[str, Any]) -> list[dict[str, Any]]:
    hints = [dict(item) for item in provider.get("hints", []) if isinstance(item, dict)]
    materialized = cursor.get("materialized_hints") if isinstance(cursor.get("materialized_hints"), dict) else {}
    hints.extend(dict(item) for item in materialized.get(str(provider.get("provider_id")), []) if isinstance(item, dict))
    return hints


def evaluate_provider_hints(
    provider: dict[str, Any],
    cursor: dict[str, Any],
    frame_rgb,
    vision: VisionEngine,
    cache: dict[str, dict[str, Any]],
    *,
    allow_expensive: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    details = [evaluate_hint(hint, frame_rgb, vision, cache, allow_expensive=allow_expensive) for hint in provider_hints(provider, cursor)]
    return [str(item.get("result") or "unknown") for item in details], details


def resolve_provider(provider: dict[str, Any], frame_rgb, vision: VisionEngine) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for index, locator in enumerate(provider.get("locators", [])):
        if not isinstance(locator, dict):
            continue
        kind = str(locator.get("type") or "")
        if kind == "point":
            action = {
                "type": "click",
                "x": int(locator["x"]),
                "y": int(locator["y"]),
                "coordinate_space": "logical",
                "brief": str(provider.get("brief") or ""),
            }
            return action, {"locator_index": index, "locator_type": kind, "locator_source": locator.get("source")}
        if kind == "run_preset":
            return {
                "type": "run_preset",
                "name": str(locator.get("name") or ""),
                "brief": str(provider.get("brief") or ""),
            }, {"locator_index": index, "locator_type": kind}
        if kind == "region_template":
            path = Path(str(locator.get("template_path") or ""))
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
    for effect in provider.get("effect_hints", []):
        if not isinstance(effect, dict) or not isinstance(effect.get("probe"), dict):
            continue
        detail = evaluate_hint(effect["probe"], frame_rgb, vision, cache, allow_expensive=True)
        baseline[str(effect.get("id") or "")] = str(detail.get("result") or "unknown")
    return baseline


def evaluate_effect(
    provider: dict[str, Any],
    baseline: dict[str, str],
    frame_rgb,
    vision: VisionEngine,
) -> tuple[str, list[dict[str, Any]]]:
    effects = [item for item in provider.get("effect_hints", []) if isinstance(item, dict) and isinstance(item.get("probe"), dict)]
    if not effects:
        return "unverified", []
    cache: dict[str, dict[str, Any]] = {}
    details: list[dict[str, Any]] = []
    outcomes: list[bool] = []
    for effect in effects:
        effect_id = str(effect.get("id") or "")
        before = baseline.get(effect_id, "unknown")
        after_detail = evaluate_hint(effect["probe"], frame_rgb, vision, cache, allow_expensive=True)
        after = str(after_detail.get("result") or "unknown")
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


def materialize_deferred_hints(
    operation: dict[str, Any],
    predecessor_id: str,
    frame_rgb,
    state_dir: Path,
    vision: VisionEngine,
    cursor: dict[str, Any],
) -> list[dict[str, Any]]:
    materialized = cursor.get("materialized_hints") if isinstance(cursor.get("materialized_hints"), dict) else {}
    cursor["materialized_hints"] = materialized
    created: list[dict[str, Any]] = []
    safe_visit = "".join(ch if ch.isalnum() else "_" for ch in str(cursor.get("visit_id") or "visit"))
    for provider in operation.get("providers", []):
        if not isinstance(provider, dict):
            continue
        provider_id = str(provider.get("provider_id") or "")
        target = materialized.setdefault(provider_id, [])
        known = {str(item.get("id") or "") for item in target if isinstance(item, dict)}
        for hint in provider.get("deferred_hints", []):
            if not isinstance(hint, dict) or str(hint.get("materialize_after") or "") != predecessor_id:
                continue
            if str(hint.get("id") or "") in known:
                continue
            provisional = dict(hint)
            provisional["status"] = "provisional"
            provisional["source_provider_id"] = predecessor_id
            if provisional.get("type") == "template" and not provisional.get("template_path"):
                bbox = provisional.get("template_bbox")
                if not isinstance(bbox, list):
                    continue
                path = state_dir / f"reactive_provisional_{safe_visit}_{provider_id}_{provisional['id']}.png"
                vision.save_template_from_rect(frame_rgb, bbox, path)
                provisional["template_path"] = str(path)
            target.append(provisional)
            created.append({"provider_id": provider_id, "hint": provisional})
    return created


def promote_materialized_hints(provider: dict[str, Any], cursor: dict[str, Any]) -> list[str]:
    materialized = cursor.get("materialized_hints") if isinstance(cursor.get("materialized_hints"), dict) else {}
    provider_id = str(provider.get("provider_id") or "")
    pending = materialized.get(provider_id) if isinstance(materialized.get(provider_id), list) else []
    hints = provider.get("hints") if isinstance(provider.get("hints"), list) else []
    provider["hints"] = hints
    known = {str(item.get("id") or "") for item in hints if isinstance(item, dict)}
    promoted: list[str] = []
    for raw in pending:
        if not isinstance(raw, dict) or str(raw.get("id") or "") in known:
            continue
        hint = {key: value for key, value in raw.items() if key not in {"materialize_after", "source_provider_id"}}
        hint["status"] = "confirmed"
        hints.append(hint)
        known.add(str(hint.get("id") or ""))
        promoted.append(str(hint.get("id") or ""))
    if promoted:
        materialized.pop(provider_id, None)
    return promoted


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
    if atype == "run_preset":
        name = str(action.get("name", "")).strip()
        _log(logger, f"[fsm][handler][preset] name={name}", "page_handler_preset", state_id=state_id, action_id=action_id, attempt=attempt, name=name, action_info=action_info)
        try:
            return run_preset(name, emulator=emulator, mapper=mapper, vision=vision, state_id=state_id, matches_provider=matches_provider, find_match_by_state=_find_match_by_state)
        except Exception as exc:  # noqa: BLE001
            _log(logger, f"[fsm][handler][preset] failed name={name} error={exc}", "page_handler_preset_exception", state_id=state_id, action_id=action_id, attempt=attempt, name=name, error=str(exc))
            return False
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
