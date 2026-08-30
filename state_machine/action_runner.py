from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.io import _load_json, _save_runtime
from state_machine.logger import FsmRunLogger


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
    action_id = "operation_main"
    controller = {"type": "page_handler", "source": "progressive_handler.v2"}
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
        state_slug=str(state_meta.get("slug", state_id)),
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
