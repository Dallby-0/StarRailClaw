from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.io import _load_json, _save_runtime
from state_machine.llm_tasks import (
    _request_llm_failure_diagnosis,
)
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import MatchResult


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _diagnose_before_repair(
    *,
    llm: DoubaoClient,
    llm_session_id: str,
    frame_rgb,
    system_prompt: str,
    state_meta: dict[str, Any],
    steps: list[dict[str, Any]],
    matches: list[MatchResult],
    runtime: dict[str, Any],
    logger: FsmRunLogger | None = None,
) -> dict[str, Any] | None:
    believed = {
        "state_id": state_meta.get("state_id"),
        "slug": state_meta.get("slug"),
        "page_type": state_meta.get("page_type"),
        "description": str(state_meta.get("description", ""))[:200],
        "enabled_conditions": [c for c in state_meta.get("match_conditions", []) if isinstance(c, dict) and c.get("enabled", False)],
    }
    obs = {
        "matching_candidates": [summarize_match(m) for m in matches if m.success or m.state_id == state_meta.get("state_id")],
        "note": "screen change is only a hint; decide whether believed state may be wrong before page-op catalog repair continues",
    }
    result = _request_llm_failure_diagnosis(
        llm,
        llm_session_id,
        frame_rgb,
        system_prompt,
        believed_state=believed,
        action_steps=steps,
        runtime_observation=obs,
        raw_debug_dir=logger.llm_raw_dir if logger is not None else None,
    )
    runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
    _save_runtime(runtime)
    if logger is not None:
        logger.event("failure_diagnosis", result=result)
    return result


def _force_state_resolution(
    *,
    runtime: dict[str, Any],
    state_id: str,
    action_id: str,
    logger: FsmRunLogger | None,
    diagnosis: dict[str, Any] | None,
    frame,
) -> tuple[bool, Any]:
    runtime["last_state_id"] = None
    runtime["last_transition_ok"] = True
    runtime["pending_from_state_id"] = state_id
    runtime["pending_action_id"] = action_id
    runtime["force_state_resolution"] = True
    runtime["force_exclude_state_id"] = state_id
    _save_runtime(runtime)
    _log(logger, f"[fsm][diagnosis] state_misidentified; force state resolution", "state_misidentified", state_id=state_id, action_id=action_id, diagnosis=diagnosis)
    return True, frame


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
    controller = {"type": "page_handler", "source": "progressive_handler.v1"}
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
    last_failure_frame = local_frame
    last_failure_matches = matches_provider(last_failure_frame)

    diagnosis = _diagnose_before_repair(
        llm=llm,
        llm_session_id=llm_session_id,
        frame_rgb=last_failure_frame,
        system_prompt=system_prompt,
        state_meta=state_meta,
        steps=[{"type": "controller", "controller": controller}],
        matches=last_failure_matches,
        runtime=runtime,
        logger=logger,
    )
    if diagnosis and diagnosis.get("diagnosis") == "state_misidentified":
        return _force_state_resolution(runtime=runtime, state_id=state_id, action_id=action_id, logger=logger, diagnosis=diagnosis, frame=last_failure_frame)

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
    )
    raise SystemExit(1)
