from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.constants import LOCAL_FLOW_MAX_STEPS, LOCAL_FLOW_NO_PROGRESS_LIMIT
from state_machine.io import _load_json, _now_iso, _save_json, _save_runtime
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import _find_match_by_state
from state_machine.page_handler.actions import execute_action, resolve_action
from state_machine.page_handler.llm import request_handler_step
from state_machine.page_handler.store import (
    active_intent,
    append_episode,
    apply_handler_patch,
    ensure_page_handler,
    mark_template_result,
    merge_intent_patch,
    set_active_intent,
)
from state_machine.screen import _screen_changed, _wait_for_screen_stable
from state_machine.transition_policy import _defer_unknown_transition, _resolve_transition_after_progress


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _template_count(handler: dict[str, Any]) -> int:
    templates = handler.get("action_templates") if isinstance(handler.get("action_templates"), dict) else {}
    return sum(1 for item in templates.values() if isinstance(item, dict) and item.get("status", "active") == "active")


def _update_intent_from_response(runtime: dict[str, Any], state_id: str, current: dict[str, Any] | None, response: dict[str, Any]) -> dict[str, Any] | None:
    intent = response.get("intent") if isinstance(response.get("intent"), dict) else None
    if intent is not None:
        current = merge_intent_patch(None, intent)
    current = merge_intent_patch(current, response.get("intent_patch"))
    if isinstance(current, dict):
        set_active_intent(runtime, state_id, current)
    return current


def _maybe_clear_satisfied_intent(runtime: dict[str, Any], state_id: str, intent: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(intent, dict):
        return intent
    if str(intent.get("status", "")) in {"satisfied", "abandoned"}:
        set_active_intent(runtime, state_id, None)
        return None
    return intent


def run_page_handler(
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
    controller: dict[str, Any] | None = None,
    seed_action: dict[str, Any] | None = None,
    entry_context: dict[str, Any] | None = None,
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    current = start_frame
    previous_frame = None
    state_path = state_dir / "state.json"
    state_meta = _load_json(state_path)
    handler = ensure_page_handler(state_meta)
    state_meta["updated_at"] = _now_iso()
    _save_json(state_path, state_meta)

    made_progress = False
    no_progress_count = 0
    last_action: dict[str, Any] | None = None
    last_action_info: dict[str, Any] | None = None
    intent = active_intent(runtime, state_id)
    entry_context = entry_context if isinstance(entry_context, dict) else {"mode": "normal"}

    _log(
        logger,
        f"[fsm][handler] enter state={state_id} templates={_template_count(handler)} intent={(intent or {}).get('kind', '')}",
        "page_handler_enter",
        state_id=state_id,
        action_id=action_id,
        templates=_template_count(handler),
        intent=intent,
        entry_context=entry_context,
    )

    for step_idx in range(1, LOCAL_FLOW_MAX_STEPS + 1):
        request = {
            "step": step_idx,
            "reason": "enter_or_continue_page_handler" if last_action is None else "judge_previous_and_continue",
            "need_intent": intent is None or no_progress_count > 0,
            "need_handler_patch": _template_count(handler) == 0 or no_progress_count > 0,
            "need_decision": True,
            "entry_context": entry_context,
            "seed_action": seed_action if step_idx == 1 and isinstance(seed_action, dict) else None,
            "previous_action": last_action,
            "previous_action_info": last_action_info,
            "runtime_signals": {
                "no_progress_count": no_progress_count,
                "has_seed_action": isinstance(seed_action, dict),
                "made_progress": made_progress,
            },
        }
        response = request_handler_step(
            llm=llm,
            session_id=llm_session_id,
            frame_rgb=current,
            previous_frame_rgb=previous_frame,
            system_prompt=system_prompt,
            state_meta=state_meta,
            handler=handler,
            active_intent=intent,
            request=request,
            logger=logger,
            effort="high" if no_progress_count > 0 else None,
        )
        runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0) or 0) + 1
        _save_runtime(runtime)
        if response is None:
            response = {"decision": {}}
        apply_handler_patch(handler, response.get("handler_patch"))
        intent = _update_intent_from_response(runtime, state_id, intent, response)
        intent = _maybe_clear_satisfied_intent(runtime, state_id, intent)
        state_meta["updated_at"] = _now_iso()
        _save_json(state_path, state_meta)

        fallback = seed_action if step_idx == 1 else None
        action, action_info = resolve_action(handler, response.get("decision"), fallback_action=fallback)
        if action is None:
            _log(
                logger,
                f"[fsm][handler] no decision state={state_id}",
                "page_handler_no_decision",
                state_id=state_id,
                action_id=action_id,
                decision=response.get("decision", {}),
                action_info=action_info,
                intent=intent,
            )
            return made_progress, current

        if not execute_action(
            emulator=emulator,
            mapper=mapper,
            vision=vision,
            state_id=state_id,
            action_id=action_id,
            action=action,
            action_info=action_info,
            matches_provider=matches_provider,
            logger=logger,
            attempt=f"handler:{step_idx}",
        ):
            return made_progress, current

        post = _wait_for_screen_stable(emulator, logger=logger, label="page-handler", event="page_handler_stability_check", max_checks=3)
        matches = matches_provider(post)
        changed, diff_score = _screen_changed(current, post)
        if logger is not None:
            logger.event(
                "page_handler_step_result",
                state_id=state_id,
                action_id=action_id,
                step=step_idx,
                changed=changed,
                diff_score=diff_score,
                action=action,
                action_info=action_info,
                candidates=[summarize_match(m) for m in matches],
            )

        append_episode(
            handler,
            {
                "state_id": state_id,
                "action": action,
                "action_info": action_info,
                "intent": intent,
                "changed": changed,
                "diff_score": diff_score,
            },
        )

        nxt = _resolve_transition_after_progress(
            state_id=state_id,
            action_id=action_id,
            matches=matches,
            graph=graph,
            runtime=runtime,
            prefer_reachable_first=prefer_reachable_first,
            logger=logger,
            reason_suffix="-page-handler",
        )
        if nxt is not None:
            mark_template_result(handler, action_info.get("template_id"), True)
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)
            return True, post

        curr_match = _find_match_by_state(matches, state_id)
        if curr_match is None or not curr_match.success:
            mark_template_result(handler, action_info.get("template_id"), True)
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)
            return _defer_unknown_transition(
                runtime=runtime,
                state_id=state_id,
                action_id=action_id,
                logger=logger,
                frame=post,
                reason="page-handler-original-state-no-longer-matched",
            )

        if changed:
            made_progress = True
            no_progress_count = 0
            mark_template_result(handler, action_info.get("template_id"), True)
        else:
            no_progress_count += 1
            mark_template_result(handler, action_info.get("template_id"), False)

        previous_frame = current
        current = post
        last_action = action
        last_action_info = action_info
        state_meta["updated_at"] = _now_iso()
        _save_json(state_path, state_meta)

        if no_progress_count >= LOCAL_FLOW_NO_PROGRESS_LIMIT:
            _log(
                logger,
                f"[fsm][handler] no progress limit hit state={state_id}",
                "page_handler_no_progress",
                state_id=state_id,
                action_id=action_id,
                limit=LOCAL_FLOW_NO_PROGRESS_LIMIT,
                intent=intent,
            )
            return made_progress, current

    _log(
        logger,
        f"[fsm][handler] max steps reached state={state_id} progress={made_progress}",
        "page_handler_max_steps",
        state_id=state_id,
        action_id=action_id,
        max_steps=LOCAL_FLOW_MAX_STEPS,
        made_progress=made_progress,
        intent=intent,
    )
    state_meta["updated_at"] = _now_iso()
    _save_json(state_path, state_meta)
    return made_progress, current
