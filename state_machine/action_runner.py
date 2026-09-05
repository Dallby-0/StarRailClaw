from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.io import _load_json, _save_runtime
from state_machine.logger import FsmRunLogger
from state_machine.matching import _find_match_by_state
from state_machine.presets import get_tool, invoke_tool


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _execute_state_action(
    *,
    emulator: EmulatorClient,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    state_id: str,
    state_dir: Path,
    frame_before,
    matches_provider,
    runtime: dict[str, Any],
    llm: DoubaoClient,
    llm_session_id: str,
    system_prompt: str,
    graph: dict[str, Any],
    prefer_reachable_first: bool,
    entry_context: dict[str, Any] | None = None,
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    from state_machine.page_handler import run_page_handler

    state_meta = _load_json(state_dir / "state.json")
    execution = state_meta.get("execution") if isinstance(state_meta.get("execution"), dict) else {}
    execution_kind = str(execution.get("kind") or "")
    scene_mode = str(state_meta.get("scene_mode") or "unknown")
    action_id = "operation_main"

    if execution_kind == "invoke_tool":
        tool_name = str(execution.get("tool_name") or "")
        tool = get_tool(tool_name)
        controller = {"type": "tool", "name": tool_name}
        if tool is None or scene_mode not in tool.supported_scene_modes:
            runtime["force_state_resolution"] = True
            runtime["force_exclude_state_id"] = state_id
            runtime["last_transition_ok"] = False
            _save_runtime(runtime)
            _log(logger, f"[fsm][controller] unavailable tool={tool_name} scene={scene_mode}", "controller_tool_unavailable", state_id=state_id, action_id=action_id, controller=controller, scene_mode=scene_mode)
            return False, frame_before
        _log(logger, f"[fsm][controller] state={state_id} type=tool name={tool_name}", "controller_start", state_id=state_id, action_id=action_id, controller=controller)
        try:
            result = invoke_tool(
                tool_name,
                emulator=emulator,
                mapper=mapper,
                vision=vision,
                state_id=state_id,
                matches_provider=matches_provider,
                find_match_by_state=_find_match_by_state,
            )
        except Exception as exc:  # noqa: BLE001 - a user-registered tool is an external executor
            result = None
            _log(logger, f"[fsm][controller] tool exception name={tool_name} error={exc}", "controller_tool_exception", state_id=state_id, action_id=action_id, controller=controller, error=str(exc))
        post_frame = emulator.screenshot(prefer_png=True)
        if result is not None and result.succeeded:
            runtime["pending_from_state_id"] = state_id
            runtime["pending_action_id"] = f"tool:{tool_name}"
            _save_runtime(runtime)
            _log(logger, f"[fsm][controller] tool completed name={tool_name} status={result.status}", "controller_tool_completed", state_id=state_id, action_id=action_id, controller=controller, status=result.status, reason=result.reason)
            return True, post_frame
        runtime["force_state_resolution"] = True
        runtime["force_exclude_state_id"] = state_id
        runtime["last_transition_ok"] = False
        _save_runtime(runtime)
        _log(logger, f"[fsm][controller] tool failed name={tool_name}", "controller_tool_failed", state_id=state_id, action_id=action_id, controller=controller, status=result.status if result is not None else "exception", reason=result.reason if result is not None else "")
        return False, post_frame

    if execution_kind == "cannot_handle":
        controller = {"type": "cannot_handle", "reason": str(execution.get("reason") or "")}
        _log(logger, "[fsm][controller] no applicable executor", "controller_cannot_handle", state_id=state_id, action_id=action_id, controller=controller)
        raise SystemExit(1)

    if execution_kind != "reactive_2d" or scene_mode != "ui_2d":
        _log(logger, f"[fsm][controller] invalid execution kind={execution_kind} scene={scene_mode}", "controller_invalid_execution", state_id=state_id, action_id=action_id, execution=execution, scene_mode=scene_mode)
        raise SystemExit(1)

    controller = {"type": "page_handler", "source": "reactive_handler.v2"}
    _log(logger, f"[fsm][controller] state={state_id} type=page_handler", "controller_start", state_id=state_id, action_id=action_id, controller=controller)
    local_ok, local_frame = run_page_handler(
        emulator=emulator,
        mapper=mapper,
        vision=vision,
        state_id=state_id,
        state_dir=state_dir,
        action_id=action_id,
        start_frame=frame_before,
        matches_provider=matches_provider,
        runtime=runtime,
        llm=llm,
        llm_session_id=llm_session_id,
        system_prompt=system_prompt,
        graph=graph,
        prefer_reachable_first=prefer_reachable_first,
        entry_context=entry_context,
        logger=logger,
    )
    if local_ok:
        return True, local_frame
    runtime["repair_fail_count"] = int(runtime.get("repair_fail_count", 0)) + 1
    runtime["last_transition_ok"] = False
    _save_runtime(runtime)
    _log(
        logger,
        "[fsm][controller] exhausted type=page_handler",
        "controller_exhausted",
        state_id=state_id,
        action_id=action_id,
        controller=controller,
        repair_fail_count=runtime["repair_fail_count"],
        diagnostic_frame_available=local_frame is not None,
    )
    raise SystemExit(1)
