from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.constants import ACTION_CLICK_WAIT_S
from state_machine.graph import _get_reachable_targets
from state_machine.io import _load_json, _save_runtime
from state_machine.llm_tasks import (
    _request_llm_failure_diagnosis,
)
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import MatchResult, _find_match_by_state
from state_machine.presets import run_preset
from state_machine.screen import _wait_for_screen_stable
from state_machine.transition_policy import _defer_unknown_transition, _pick_reachable_transition_candidate


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


def _execute_action_steps(
    emulator: EmulatorClient,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    state_dir: Path,
    steps: list[dict[str, Any]],
    state_id: str,
    matches_provider,
    logger: FsmRunLogger | None = None,
    action_id: str = "",
    attempt: int | str = "",
) -> bool:
    for idx, step in enumerate(steps, start=1):
        stype = str(step.get("type", "click"))
        if stype == "run_preset":
            name = str(step.get("name", ""))
            _log(logger, f"[fsm][action][preset] step={idx} name={name}", "action_preset", state_id=state_id, action_id=action_id, attempt=attempt, step=idx, name=name)
            try:
                ok = run_preset(
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
                    f"[fsm][action][preset][exception] step={idx} name={name} error={type(exc).__name__}: {exc}; idle=10s treat_as_success",
                    "action_preset_exception",
                    state_id=state_id,
                    action_id=action_id,
                    attempt=attempt,
                    step=idx,
                    name=name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    sleep_s=10.0,
                    treat_as_success=True,
                )
                time.sleep(10.0)
                continue
            if not ok:
                return False
            continue

        x = int(step.get("x", 500))
        y = int(step.get("y", 500))
        brief = str(step.get("brief", "")).strip()
        rx, ry = mapper.point_to_real(x, y)
        _log(
            logger,
            f"[fsm][action][click] step={idx} logical=({x},{y}) real=({rx},{ry}) brief={brief}",
            "action_click",
            state_id=state_id,
            action_id=action_id,
            attempt=attempt,
            step=idx,
            logical=[x, y],
            real=[rx, ry],
            brief=brief,
        )
        emulator.tap(rx, ry)
        _log(logger, f"[fsm][action][wait] sleep={ACTION_CLICK_WAIT_S}s after step={idx}", "action_wait", state_id=state_id, action_id=action_id, attempt=attempt, step=idx, sleep_s=ACTION_CLICK_WAIT_S)
        time.sleep(ACTION_CLICK_WAIT_S)
    return True


def _reset_same_external_state_counter(runtime: dict[str, Any]) -> None:
    runtime["same_external_state_id"] = None
    runtime["same_external_state_count"] = 0


def _first_enabled_action_steps(state_meta: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    actions = [a for a in state_meta.get("actions", []) if isinstance(a, dict) and a.get("enabled", True)]
    if not actions:
        return "action_main", []
    action = actions[0]
    steps = action.get("steps", [])
    return str(action.get("action_id", "action_main")), steps if isinstance(steps, list) else []


def _controller_from_state(state_meta: dict[str, Any]) -> dict[str, Any]:
    controller = state_meta.get("controller")
    if isinstance(controller, dict) and controller.get("type"):
        return controller
    _, steps = _first_enabled_action_steps(state_meta)
    first = steps[0] if steps and isinstance(steps[0], dict) else None
    if first and str(first.get("type", "click")) == "run_preset":
        return {"type": "preset", "name": str(first.get("name", "")), "source": "inferred_from_action"}
    if first:
        return {"type": "page_op_flow", "seed_action": first, "source": "inferred_from_action"}
    return {"type": "page_op_flow", "source": "default"}


def _has_active_page_ops(state_meta: dict[str, Any]) -> bool:
    flow = state_meta.get("page_op_flow")
    if not isinstance(flow, dict):
        return False
    catalog = flow.get("op_catalog")
    if not isinstance(catalog, dict):
        return False
    ops = catalog.get("ops")
    if not isinstance(ops, list):
        return False
    return any(isinstance(op, dict) and op.get("status", "active") == "active" and not bool(op.get("effectless", False)) for op in ops)


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


def _resolve_controller_post_frame(
    *,
    state_id: str,
    action_id: str,
    frame,
    matches_provider,
    runtime: dict[str, Any],
    graph: dict[str, Any],
    prefer_reachable_first: bool,
    logger: FsmRunLogger | None,
    reason_suffix: str,
) -> tuple[str, Any]:
    matches = matches_provider(frame)
    reachable = _get_reachable_targets(graph, state_id)
    if logger is not None:
        logger.event(
            "post_controller_match",
            state_id=state_id,
            action_id=action_id,
            reachable=sorted(reachable),
            candidates=[summarize_match(m) for m in matches],
            reason_suffix=reason_suffix,
        )
    nxt, confidence = _pick_reachable_transition_candidate(matches, state_id, reachable if prefer_reachable_first else set())
    if nxt is not None and nxt.state_id != state_id:
        _log(logger, f"[fsm][transition][reachable{reason_suffix}] from={state_id} action={action_id} to={nxt.state_id} confidence={confidence}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason=f"reachable{reason_suffix}", confidence=confidence)
        runtime["last_state_id"] = nxt.state_id
        runtime["last_transition_ok"] = True
        runtime["pending_from_state_id"] = None
        runtime["pending_action_id"] = None
        _reset_same_external_state_counter(runtime)
        _save_runtime(runtime)
        return "transition", frame
    curr = _find_match_by_state(matches, state_id)
    if curr is None or not curr.success:
        _reset_same_external_state_counter(runtime)
        ok, out = _defer_unknown_transition(
            runtime=runtime,
            state_id=state_id,
            action_id=action_id,
            logger=logger,
            frame=frame,
            reason=f"controller-original-state-no-longer-matched{reason_suffix}",
        )
        return "unknown" if ok else "failed", out
    return "same_state", frame


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
    from state_machine.page_op_flow import _run_page_op_flow

    state_meta = _load_json(state_dir / "state.json")
    action_id, action_steps = _first_enabled_action_steps(state_meta)
    controller = _controller_from_state(state_meta)
    ctype = str(controller.get("type", "page_op_flow"))
    _log(logger, f"[fsm][controller] state={state_id} type={ctype}", "controller_start", state_id=state_id, action_id=action_id, controller=controller)

    if ctype == "preset":
        name = str(controller.get("name", ""))
        if not name:
            return False, frame_before
        if not _execute_action_steps(emulator, mapper, vision, state_dir, [{"type": "run_preset", "name": name, "brief": str(controller.get("brief", ""))}], state_id, matches_provider, logger=logger, action_id=action_id, attempt="controller:preset"):
            return False, frame_before
        post = _wait_for_screen_stable(emulator, logger=logger, label="action-post", event="action_post_stability_check", max_checks=3)
        status, out = _resolve_controller_post_frame(
            state_id=state_id,
            action_id=action_id,
            frame=post,
            matches_provider=matches_provider,
            runtime=runtime,
            graph=graph,
            prefer_reachable_first=prefer_reachable_first,
            logger=logger,
            reason_suffix="-preset",
        )
        if status in {"transition", "unknown"}:
            return True, out
        last_failure_frame = out
        last_failure_matches = matches_provider(last_failure_frame)
    elif ctype == "page_op_flow":
        seed_action = controller.get("seed_action") if isinstance(controller.get("seed_action"), dict) else None
        if seed_action is None and action_steps and isinstance(action_steps[0], dict) and str(action_steps[0].get("type", "click")) != "run_preset":
            seed_action = action_steps[0]
        if _has_active_page_ops(state_meta):
            seed_action = None
        start_frame = frame_before
        if isinstance(seed_action, dict):
            _log(logger, f"[fsm][controller][seed] state={state_id}", "controller_seed_action", state_id=state_id, action_id=action_id, action=seed_action)
            if not _execute_action_steps(emulator, mapper, vision, state_dir, [seed_action], state_id, matches_provider, logger=logger, action_id=action_id, attempt="controller:seed"):
                return False, frame_before
            post_seed = _wait_for_screen_stable(emulator, logger=logger, label="action-seed", event="action_seed_stability_check", max_checks=3)
            status, out = _resolve_controller_post_frame(
                state_id=state_id,
                action_id=action_id,
                frame=post_seed,
                matches_provider=matches_provider,
                runtime=runtime,
                graph=graph,
                prefer_reachable_first=prefer_reachable_first,
                logger=logger,
                reason_suffix="-seed",
            )
            if status in {"transition", "unknown"}:
                return True, out
            start_frame = out
        local_ok, local_frame = _run_page_op_flow(
            emulator=emulator,
            mapper=mapper,
            vision=vision,
            state_id=state_id,
            state_dir=state_dir,
            action_id=action_id,
            start_frame=start_frame,
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
    else:
        _log(logger, f"[fsm][controller] unknown type={ctype}", "controller_unknown", state_id=state_id, action_id=action_id, controller=controller)
        return False, frame_before

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
        f"[fsm][controller] exhausted type={ctype}",
        "controller_exhausted",
        state_id=state_id,
        action_id=action_id,
        controller=controller,
        repair_fail_count=runtime["repair_fail_count"],
    )
    raise SystemExit(1)
