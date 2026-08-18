from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.command import EffectiveCommand, resolve_effective_command, step_preconditions_pass
from state_machine.constants import LOCAL_FLOW_MAX_STEPS, LOCAL_FLOW_NO_PROGRESS_LIMIT
from state_machine.intent import active_intent, intent_projection, reduce_intent_event
from state_machine.io import _load_json, _now_iso, _save_json, _save_runtime
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import _find_match_by_state
from state_machine.page_handler.actions import execute_action, resolve_strategy_step
from state_machine.page_handler.llm import request_handler_repair
from state_machine.page_handler.store import apply_handler_patch, append_episode, ensure_page_handler, mark_strategy_result, materialize_strategy_templates, promote_operation_to_default, select_strategy
from state_machine.screen import _screen_changed, _wait_for_screen_stable
from state_machine.transition_policy import _defer_unknown_transition, _resolve_transition_after_progress


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _fallback_command(intent: dict[str, Any] | None) -> EffectiveCommand:
    if isinstance(intent, dict):
        return EffectiveCommand(
            operation=str(intent.get("kind") or "intent_operation"),
            source="unresolved_intent",
            intent_id=str(intent.get("intent_id") or "") or None,
            intent_kind=str(intent.get("kind") or "") or None,
            intent_phase=str(intent.get("phase") or "") or None,
            params=dict(intent.get("params") or {}),
            persistent=True,
            intent_effect="advance",
        )
    return EffectiveCommand(operation="advance", source="unresolved_default")


def _event_for_success(command: EffectiveCommand, step_info: dict[str, Any], *, final_step: bool) -> dict[str, Any] | None:
    event = step_info.get("emits_on_success")
    if isinstance(event, dict) and event.get("type"):
        return dict(event)
    if final_step and command.expected_event:
        return {"type": command.expected_event}
    return None


def _strategy_result(*, changed: bool, left_state: bool, expected: dict[str, Any], final_step: bool) -> str:
    if bool(expected.get("exit_likely", False)):
        return "verified_success" if left_state else "no_effect"
    if bool(expected.get("same_page_likely", False)):
        return ("verified_success" if final_step else "partial_progress") if changed and not left_state else "wrong_transition" if left_state else "no_effect"
    if left_state or changed:
        return "verified_success" if final_step or left_state else "partial_progress"
    return "no_effect"


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
    del state_slug, controller, seed_action
    current = start_frame
    state_path = state_dir / "state.json"
    state_meta = _load_json(state_path)
    handler = ensure_page_handler(state_meta)
    intent = active_intent(runtime)
    command = resolve_effective_command(state_meta, intent) or _fallback_command(intent)
    failed_attempts: list[dict[str, Any]] = []
    excluded: set[str] = set()
    total_actions = 0
    made_progress = False
    repair_calls = 0

    _log(logger, f"[fsm][handler] enter state={state_id} operation={command.operation} source={command.source}", "page_handler_enter", state_id=state_id, action_id=action_id, command=command.to_dict(), intent=intent_projection(intent), entry_context=entry_context or {"mode": "normal"})

    while total_actions < LOCAL_FLOW_MAX_STEPS:
        strategy = select_strategy(handler, command.operation, excluded=excluded)
        if strategy is None:
            if repair_calls >= LOCAL_FLOW_NO_PROGRESS_LIMIT:
                _log(logger, f"[fsm][handler] no strategy state={state_id} operation={command.operation}", "page_handler_no_strategy", state_id=state_id, operation=command.operation, failed_attempts=failed_attempts)
                return made_progress, current
            response = request_handler_repair(
                llm=llm,
                session_id=llm_session_id,
                frame_rgb=current,
                previous_frame_rgb=start_frame if current is not start_frame else None,
                system_prompt=system_prompt,
                state_meta=state_meta,
                handler=handler,
                command=command.to_dict(),
                failed_attempts=failed_attempts,
                logger=logger,
            )
            runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0) or 0) + 1
            _save_runtime(runtime)
            repair_calls += 1
            if response is None:
                continue
            raw_patch = response.get("handler_patch") if isinstance(response.get("handler_patch"), dict) else {}
            filtered_patch = dict(raw_patch)
            filtered_patch["operations"] = [
                item
                for item in raw_patch.get("operations", [])
                if isinstance(item, dict) and str(item.get("operation") or "").strip() == command.operation
            ]
            touched = apply_handler_patch(handler, filtered_patch)
            materialize_strategy_templates(handler, state_dir, current, vision, touched)
            excluded.difference_update(touched)
            reduce_intent_event(runtime, response.get("event"))
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)
            _save_runtime(runtime)
            continue

        strategy_id = str(strategy.get("strategy_id") or "")
        if command.intent_id is None and str(strategy.get("safety", "unknown")) not in {"low_risk", "reversible"}:
            failed_attempts.append({"strategy_id": strategy_id, "result": "unsafe_effect", "reason": "explicit_intent_required"})
            mark_strategy_result(handler, command.operation, strategy_id, "unsafe_effect")
            excluded.add(strategy_id)
            continue
        steps = strategy.get("steps") if isinstance(strategy.get("steps"), list) else []
        strategy_failed = False
        previous_step_verified = False
        previous_event: dict[str, Any] | None = None
        current_state_matches = True
        for step_index in range(len(steps)):
            if total_actions >= LOCAL_FLOW_MAX_STEPS:
                break
            step = steps[step_index] if isinstance(steps[step_index], dict) else {}
            preconditions = step.get("preconditions") if isinstance(step.get("preconditions"), dict) else {}
            applicable, precondition_reason = step_preconditions_pass(
                preconditions,
                previous_step_verified=previous_step_verified,
                previous_event=previous_event,
                current_state_matches=current_state_matches,
                intent=active_intent(runtime),
            )
            if not applicable:
                if made_progress:
                    return True, current
                failed_attempts.append({"strategy_id": strategy_id, "step_index": step_index, "result": "no_effect", "reason": precondition_reason})
                mark_strategy_result(handler, command.operation, strategy_id, "no_effect")
                excluded.add(strategy_id)
                strategy_failed = True
                break
            action, action_info = resolve_strategy_step(strategy, step_index, current, vision)
            if action is None:
                failed = {"strategy_id": strategy_id, "step_index": step_index, "result": "no_effect", "reason": action_info.get("reason")}
                failed_attempts.append(failed)
                mark_strategy_result(handler, command.operation, strategy_id, "no_effect")
                excluded.add(strategy_id)
                strategy_failed = True
                break

            runtime["pending_operation"] = {
                "intent_id": command.intent_id,
                "intent_revision": intent.get("revision") if isinstance(intent, dict) else None,
                "state_id": state_id,
                "operation": command.operation,
                "strategy_id": strategy_id,
                "step_id": action_info.get("step_id"),
                "status": "issued",
                "created_at": _now_iso(),
            }
            _save_runtime(runtime)
            if not execute_action(emulator=emulator, mapper=mapper, vision=vision, state_id=state_id, action_id=action_id, action=action, action_info=action_info, matches_provider=matches_provider, logger=logger, attempt=f"strategy:{strategy_id}:{step_index + 1}"):
                runtime["pending_operation"] = None
                _save_runtime(runtime)
                mark_strategy_result(handler, command.operation, strategy_id, "no_effect")
                excluded.add(strategy_id)
                strategy_failed = True
                break

            total_actions += 1
            post = _wait_for_screen_stable(emulator, logger=logger, label="page-handler", event="page_handler_stability_check", max_checks=3)
            matches = matches_provider(post)
            changed, diff_score = _screen_changed(current, post)
            current_match = _find_match_by_state(matches, state_id)
            left_state = current_match is None or not current_match.success
            current_state_matches = not left_state
            final_step = step_index == len(steps) - 1
            result = _strategy_result(changed=changed, left_state=left_state, expected=action_info.get("expected_after", {}), final_step=final_step)
            allowed = action_info.get("expected_after", {}).get("allowed_state_ids")
            if left_state and isinstance(allowed, list):
                observed = {m.state_id for m in matches if m.success}
                if observed and not observed.intersection({str(v) for v in allowed}):
                    result = "wrong_transition"

            runtime["pending_operation"] = None
            mark_strategy_result(handler, command.operation, strategy_id, result)
            if result in {"verified_success", "partial_progress"}:
                made_progress = True
            if result == "verified_success" and command.source == "unresolved_default":
                promote_operation_to_default(handler, command.operation)
            event = _event_for_success(command, action_info, final_step=final_step) if result in {"verified_success", "partial_progress"} else None
            reduce_intent_event(runtime, event)
            previous_step_verified = result in {"verified_success", "partial_progress"}
            previous_event = event
            append_episode(handler, {"state_id": state_id, "command": command.to_dict(), "strategy_id": strategy_id, "step_id": action_info.get("step_id"), "result": result, "event": event, "changed": changed, "diff_score": diff_score})
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)
            _save_runtime(runtime)
            if logger is not None:
                logger.event("page_handler_step_result", state_id=state_id, action_id=action_id, operation=command.operation, strategy_id=strategy_id, step=step_index + 1, result=result, changed=changed, diff_score=diff_score, event=event, candidates=[summarize_match(m) for m in matches])

            if result in {"no_effect", "wrong_transition", "unsafe_effect"}:
                failed_attempts.append({"strategy_id": strategy_id, "step_index": step_index, "result": result, "diff_score": diff_score})
                excluded.add(strategy_id)
                strategy_failed = True
                current = post
                if left_state:
                    nxt = _resolve_transition_after_progress(state_id=state_id, action_id=action_id, matches=matches, graph=graph, runtime=runtime, prefer_reachable_first=prefer_reachable_first, logger=logger, reason_suffix="-page-handler-failed-policy")
                    if nxt is not None:
                        return True, post
                    return _defer_unknown_transition(runtime=runtime, state_id=state_id, action_id=action_id, logger=logger, frame=post, reason="page-handler-failed-policy-left-state")
                break

            nxt = _resolve_transition_after_progress(state_id=state_id, action_id=action_id, matches=matches, graph=graph, runtime=runtime, prefer_reachable_first=prefer_reachable_first, logger=logger, reason_suffix="-page-handler")
            if nxt is not None:
                return True, post
            if left_state:
                return _defer_unknown_transition(runtime=runtime, state_id=state_id, action_id=action_id, logger=logger, frame=post, reason="page-handler-original-state-no-longer-matched")
            current = post

        if strategy_failed:
            continue
        if steps:
            return True, current

    _log(logger, f"[fsm][handler] max steps reached state={state_id}", "page_handler_max_steps", state_id=state_id, action_id=action_id, operation=command.operation, max_steps=LOCAL_FLOW_MAX_STEPS)
    return made_progress, current
