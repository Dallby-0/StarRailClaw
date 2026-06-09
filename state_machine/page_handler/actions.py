from __future__ import annotations

import time
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from sr_tools.emulator import EmulatorClient
from state_machine.constants import ACTION_CLICK_WAIT_S
from state_machine.logger import FsmRunLogger
from state_machine.presets import run_preset
from state_machine.matching import _find_match_by_state
from state_machine.page_handler.store import _slug


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _rect_center(rect: list[Any]) -> tuple[int, int] | None:
    if not (isinstance(rect, list) and len(rect) == 4):
        return None
    a, b, c, d = [int(v) for v in rect]
    if c > a and d > b:
        return max(0, min(999, (a + c) // 2)), max(0, min(999, (b + d) // 2))
    return max(0, min(999, a + max(1, c) // 2)), max(0, min(999, b + max(1, d) // 2))


def _normalize_click_action(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    atype = str(raw.get("type", "click")).strip()
    if atype == "run_preset":
        name = str(raw.get("name", "")).strip()
        if not name:
            return None
        return {"type": "run_preset", "name": name, "brief": str(raw.get("brief", ""))}
    if atype != "click":
        return None
    try:
        x = int(raw.get("x"))
        y = int(raw.get("y"))
    except Exception:
        return None
    return {"type": "click", "x": x, "y": y, "brief": str(raw.get("brief", ""))}


def resolve_action(handler: dict[str, Any], decision: Any, *, fallback_action: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(decision, dict):
        decision = {}
    direct = _normalize_click_action(decision.get("action"))
    if direct is not None:
        return direct, {"source": "direct", "template_id": "", "reason": str(decision.get("reason", ""))}

    template_id = _slug(decision.get("template_id") or decision.get("action_template_id"), "")
    templates = handler.get("action_templates") if isinstance(handler.get("action_templates"), dict) else {}
    template = templates.get(template_id) if template_id else None
    if isinstance(template, dict) and template.get("status", "active") == "active":
        kind = str(template.get("kind", ""))
        brief = str(decision.get("brief") or decision.get("reason") or template.get("label") or template_id)
        if kind == "run_preset":
            name = str(template.get("name", "")).strip()
            if name:
                return {"type": "run_preset", "name": name, "brief": brief}, {"source": "template", "template_id": template_id, "kind": kind}
        if kind in {"click", "click_button", "click_active_confirm", "click_blank_to_dismiss"}:
            if template.get("x") is not None and template.get("y") is not None:
                return {"type": "click", "x": int(template.get("x")), "y": int(template.get("y")), "brief": brief}, {"source": "template", "template_id": template_id, "kind": kind}
            center = _rect_center(template.get("bbox"))
            if center is not None:
                x, y = center
                return {"type": "click", "x": x, "y": y, "brief": brief}, {"source": "template", "template_id": template_id, "kind": kind}
        if kind == "select_from_slots":
            slots = template.get("slots") if isinstance(template.get("slots"), list) else []
            slot_id = _slug(decision.get("slot_id") or decision.get("target_slot"), "")
            selected = None
            for slot in slots:
                if isinstance(slot, dict) and slot_id and _slug(slot.get("slot_id") or slot.get("id"), "") == slot_id:
                    selected = slot
                    break
            if selected is None:
                selected = next((slot for slot in slots if isinstance(slot, dict)), None)
            if isinstance(selected, dict):
                if selected.get("x") is not None and selected.get("y") is not None:
                    x, y = int(selected.get("x")), int(selected.get("y"))
                else:
                    center = _rect_center(selected.get("bbox"))
                    if center is None:
                        return None, {"source": "template", "template_id": template_id, "kind": kind, "reason": "slot_has_no_click_point"}
                    x, y = center
                label = str(selected.get("label") or selected.get("slot_id") or "")
                return {"type": "click", "x": x, "y": y, "brief": brief or label}, {
                    "source": "template",
                    "template_id": template_id,
                    "kind": kind,
                    "slot_id": str(selected.get("slot_id") or ""),
                }

    fallback = _normalize_click_action(fallback_action)
    if fallback is not None:
        return fallback, {"source": "fallback_seed", "template_id": "", "reason": "no_resolved_template_action"}
    return None, {"source": "none", "template_id": template_id, "reason": "no_action_resolved"}


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
            return run_preset(
                name,
                emulator=emulator,
                mapper=mapper,
                vision=vision,
                state_id=state_id,
                matches_provider=matches_provider,
                find_match_by_state=_find_match_by_state,
            )
        except Exception as exc:
            _log(
                logger,
                f"[fsm][handler][preset][exception] name={name} error={type(exc).__name__}: {exc}; idle=10s treat_as_success",
                "page_handler_preset_exception",
                state_id=state_id,
                action_id=action_id,
                attempt=attempt,
                name=name,
                error_type=type(exc).__name__,
                error=str(exc),
                sleep_s=10.0,
                treat_as_success=True,
            )
            time.sleep(10.0)
            return True
    x = int(action.get("x", 500))
    y = int(action.get("y", 500))
    rx, ry = mapper.point_to_real(x, y)
    brief = str(action.get("brief", "")).strip()
    _log(
        logger,
        f"[fsm][handler][click] logical=({x},{y}) real=({rx},{ry}) brief={brief}",
        "page_handler_click",
        state_id=state_id,
        action_id=action_id,
        attempt=attempt,
        logical=[x, y],
        real=[rx, ry],
        brief=brief,
        action_info=action_info,
    )
    emulator.tap(rx, ry)
    _log(logger, f"[fsm][handler][wait] sleep={ACTION_CLICK_WAIT_S}s", "page_handler_wait", state_id=state_id, action_id=action_id, sleep_s=ACTION_CLICK_WAIT_S, action_info=action_info)
    time.sleep(ACTION_CLICK_WAIT_S)
    return True
