from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.io import _load_json, _now_iso, _save_runtime
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import _find_match_by_state
from state_machine.page_handler import (
    action_for_edge,
    add_page_handler_edge,
    edge_conditions_pass,
    ensure_page_handler,
    get_default_node,
    match_page_handler_edge,
    materialize_page_handler_edge,
    request_page_handler_edge,
    summarize_page_handler,
    update_edge_stats,
)
from state_machine.screen import _screen_changed
from state_machine.transition_policy import _defer_unknown_transition, _resolve_transition_after_progress
from state_machine.constants import LOCAL_FLOW_MAX_STEPS, LOCAL_FLOW_NO_PROGRESS_LIMIT


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _run_page_local_flow(
    *,
    emulator: EmulatorClient,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    state_id: str,
    state_dir: Path,
    action_id: str,
    start_frame,
    matches_provider,
    runtime: dict[str, Any],
    llm: DoubaoClient,
    llm_session_id: str,
    system_prompt: str,
    graph: dict[str, Any],
    prefer_reachable_first: bool,
    state_slug: str,
    seed_step: dict[str, Any] | None = None,
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    from state_machine.action_runner import _execute_action_steps

    current = start_frame
    state_path = state_dir / "state.json"
    state_meta = _load_json(state_path)
    handler = ensure_page_handler(state_meta)
    current_node = get_default_node(handler)
    previous_steps: list[dict[str, Any]] = []
    if isinstance(seed_step, dict):
        previous_steps.append(seed_step)
        seed_action = seed_step.get("action")
        if isinstance(seed_action, dict):
            seed_edge = {
                "id": f"edge_seed_{uuid.uuid4().hex[:12]}",
                "enabled": True,
                "from_node": current_node,
                "to_node": f"{current_node}_seed_action",
                "condition_policy": "default_if_no_edge_matches",
                "conditions": [],
                "action": seed_action,
                "expected_after_action": {
                    "same_page_likely": True,
                    "exit_likely": False,
                    "screen_should_change": True,
                    "reason": "seed action captured when entering page mode",
                },
                "priority": 1,
                "brief": str(seed_step.get("brief", "entry action before page mode")),
                "success_count": 1,
                "fail_count": 0,
                "status": "probation",
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
            add_page_handler_edge(state_meta, seed_edge, state_path=state_path)
            handler = ensure_page_handler(state_meta)
            current_node = str(seed_edge["to_node"])
    no_progress_count = 0
    made_progress = False
    for step_idx in range(1, LOCAL_FLOW_MAX_STEPS + 1):
        edge, diagnostics = match_page_handler_edge(handler, current_node, vision, current)
        learned_edge = False
        if edge is None:
            _log(
                logger,
                f"[fsm][handler] miss node={current_node}; request edge from llm",
                "page_handler_miss",
                state_id=state_id,
                node=current_node,
                diagnostics=diagnostics,
            )
            edge = request_page_handler_edge(
                llm,
                llm_session_id,
                current,
                system_prompt,
                state_slug=state_slug,
                handler_summary=summarize_page_handler(handler, current_node),
                previous_steps=previous_steps,
                logger=logger,
            )
            runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
            _save_runtime(runtime)
            if edge is None:
                _log(logger, f"[fsm][handler] no valid edge step={step_idx}", "page_handler_invalid", state_id=state_id, step=step_idx, node=current_node)
                return made_progress, current
            edge = materialize_page_handler_edge(edge, state_dir=state_dir, frame_rgb=current, mapper=mapper)
            if edge is None or not edge_conditions_pass(edge, vision, current):
                _log(logger, f"[fsm][handler] generated edge does not match current screen step={step_idx}", "page_handler_rejected", state_id=state_id, step=step_idx, node=current_node, edge=edge)
                return made_progress, current
            learned_edge = True
        else:
            _log(
                logger,
                f"[fsm][handler] hit node={current_node} edge={edge.get('id')}",
                "page_handler_hit",
                state_id=state_id,
                step=step_idx,
                node=current_node,
                edge_id=edge.get("id"),
            )

        action = action_for_edge(handler, edge)
        if not isinstance(action, dict):
            return made_progress, current
        previous_steps.append(
            {
                "node": current_node,
                "edge_id": edge.get("id"),
                "source": "llm_new_edge" if learned_edge else "handler",
                "action": action,
                "expected_after_action": edge.get("expected_after_action", {}),
                "brief": edge.get("brief", ""),
            }
        )
        if not _execute_action_steps(
            emulator,
            mapper,
            vision,
            state_dir,
            [action],
            state_id,
            matches_provider,
            logger=logger,
            action_id=action_id,
            attempt=f"local:{step_idx}",
        ):
            if not learned_edge:
                update_edge_stats(state_meta, str(edge.get("id")), success=False, state_path=state_path)
            return made_progress, current
        post = emulator.screenshot(prefer_png=True)
        changed, diff_score = _screen_changed(current, post)
        matches = matches_provider(post)
        if logger is not None:
            logger.event(
                "page_local_step_result",
                state_id=state_id,
                action_id=action_id,
                step=step_idx,
                changed=changed,
                diff_score=diff_score,
                candidates=[summarize_match(m) for m in matches],
            )
        nxt = _resolve_transition_after_progress(
            state_id=state_id,
            action_id=action_id,
            matches=matches,
            graph=graph,
            runtime=runtime,
            prefer_reachable_first=prefer_reachable_first,
            logger=logger,
            reason_suffix="-page-local",
        )
        if nxt is not None:
            if learned_edge:
                edge["success_count"] = 1
                add_page_handler_edge(state_meta, edge, state_path=state_path)
                handler = ensure_page_handler(state_meta)
            else:
                update_edge_stats(state_meta, str(edge.get("id")), success=True, state_path=state_path)
            return True, post
        curr_match = _find_match_by_state(matches, state_id)
        if curr_match is None or not curr_match.success:
            if learned_edge:
                edge["success_count"] = 1
                add_page_handler_edge(state_meta, edge, state_path=state_path)
            else:
                update_edge_stats(state_meta, str(edge.get("id")), success=True, state_path=state_path)
            return _defer_unknown_transition(
                runtime=runtime,
                state_id=state_id,
                action_id=action_id,
                logger=logger,
                frame=post,
                reason="page-local-original-state-no-longer-matched",
            )
        if changed:
            if learned_edge:
                edge["success_count"] = 1
                add_page_handler_edge(state_meta, edge, state_path=state_path)
                handler = ensure_page_handler(state_meta)
            else:
                update_edge_stats(state_meta, str(edge.get("id")), success=True, state_path=state_path)
                state_meta = _load_json(state_path)
                handler = ensure_page_handler(state_meta)
            made_progress = True
            no_progress_count = 0
            current_node = str(edge.get("to_node") or current_node or "root")
            current = post
            continue
        if not learned_edge:
            update_edge_stats(state_meta, str(edge.get("id")), success=False, state_path=state_path)
        no_progress_count += 1
        current = post
        if no_progress_count >= LOCAL_FLOW_NO_PROGRESS_LIMIT:
            _log(logger, f"[fsm][local] no progress limit hit state={state_id}", "page_local_no_progress", state_id=state_id, limit=LOCAL_FLOW_NO_PROGRESS_LIMIT)
            return made_progress, current
    _log(logger, f"[fsm][local] max steps reached state={state_id} progress={made_progress}", "page_local_max_steps", state_id=state_id, max_steps=LOCAL_FLOW_MAX_STEPS, made_progress=made_progress)
    return made_progress, current
