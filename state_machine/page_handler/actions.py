from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from sr_tools.emulator import EmulatorClient
from state_machine.constants import ACTION_CLICK_WAIT_S
from state_machine.logger import FsmRunLogger
from state_machine.matching import _find_match_by_state
from state_machine.presets import run_preset


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def resolve_strategy_step(strategy: dict[str, Any], step_index: int, frame_rgb, vision: VisionEngine) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    steps = strategy.get("steps") if isinstance(strategy.get("steps"), list) else []
    if step_index < 0 or step_index >= len(steps) or not isinstance(steps[step_index], dict):
        return None, {"reason": "step_out_of_range"}
    step = steps[step_index]
    resolver = step.get("resolver") if isinstance(step.get("resolver"), dict) else {}
    rtype = str(resolver.get("type") or "")
    info = {
        "strategy_id": str(strategy.get("strategy_id") or ""),
        "strategy_level": int(strategy.get("level", 0) or 0),
        "step_id": str(step.get("step_id") or f"step_{step_index + 1}"),
        "step_index": step_index,
        "resolver_type": rtype,
        "expected_after": dict(step.get("expected_after") or {}),
        "emits_on_success": dict(step.get("emits_on_success") or {}),
    }
    brief = str(step.get("brief") or strategy.get("strategy_id") or "")
    if rtype == "fixed_point":
        return {"type": "click", "x": int(resolver.get("x")), "y": int(resolver.get("y")), "coordinate_space": "logical", "brief": brief}, info
    if rtype == "run_preset":
        name = str(resolver.get("name") or "").strip()
        return ({"type": "run_preset", "name": name, "brief": brief}, info) if name else (None, {**info, "reason": "empty_preset"})
    if rtype == "region_template":
        path = Path(str(resolver.get("template_path") or ""))
        rect = resolver.get("search_rect") if isinstance(resolver.get("search_rect"), list) else None
        ok, point, similarity = vision.match_template(frame_rgb, path, rect, threshold=float(resolver.get("threshold", 0.82)))
        info["template_path"] = str(path)
        info["template_similarity"] = similarity
        if not ok or point is None:
            return None, {**info, "reason": "template_not_found"}
        return {"type": "click", "x": int(point[0]), "y": int(point[1]), "coordinate_space": "real", "brief": brief}, info
    return None, {**info, "reason": "unsupported_resolver"}


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
        except Exception as exc:
            _log(logger, f"[fsm][handler][preset][exception] name={name} error={type(exc).__name__}: {exc}", "page_handler_preset_exception", state_id=state_id, action_id=action_id, attempt=attempt, name=name, error_type=type(exc).__name__, error=str(exc), treat_as_success=False)
            return False
    x = int(action.get("x", 500))
    y = int(action.get("y", 500))
    if str(action.get("coordinate_space", "logical")) == "real":
        rx, ry = x, y
        logical = None
    else:
        rx, ry = mapper.point_to_real(x, y)
        logical = [x, y]
    brief = str(action.get("brief", "")).strip()
    _log(logger, f"[fsm][handler][click] real=({rx},{ry}) brief={brief}", "page_handler_click", state_id=state_id, action_id=action_id, attempt=attempt, logical=logical, real=[rx, ry], brief=brief, action_info=action_info)
    emulator.tap(rx, ry)
    _log(logger, f"[fsm][handler][wait] sleep={ACTION_CLICK_WAIT_S}s", "page_handler_wait", state_id=state_id, action_id=action_id, sleep_s=ACTION_CLICK_WAIT_S, action_info=action_info)
    time.sleep(ACTION_CLICK_WAIT_S)
    return True
