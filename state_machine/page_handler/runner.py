from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.command import EffectiveCommand, resolve_effective_command, step_preconditions_pass
from state_machine.constants import LOCAL_FLOW_MAX_STEPS
from state_machine.graph import _append_graph_edge
from state_machine.intent import active_intent, intent_projection, reduce_intent_event
from state_machine.io import _load_json, _now_iso, _save_json, _save_runtime
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import _find_match_by_state
from state_machine.page_handler.actions import execute_action, resolve_strategy_step
from state_machine.page_handler.exploration import run_exploration
from state_machine.page_handler.llm import request_handler_repair, request_same_state_review
from state_machine.page_handler.review_protocol import result_for_same_state_verdict
from state_machine.page_handler.store import apply_handler_patch, append_episode, continuation_patch_from_sibling, ensure_page_handler, mark_strategy_result, materialize_strategy_templates, promote_operation_to_default, select_strategy
from state_machine.progress_guard import blocked_strategies, clear_continuation, clear_visit_progress, consume_continuation, continuation_for, grant_continuation, operation_exhausted, record_attempt, record_no_progress, record_repair, record_review, repair_exhausted, review_exhausted
from state_machine.screen import _screen_changed, _wait_for_screen_stable
from state_machine.transition_policy import _defer_unknown_transition, _resolve_transition_after_progress
from state_machine.visit import ensure_page_visit, start_new_visit


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _fallback_command(intent: dict[str, Any] | None) -> EffectiveCommand:
    if isinstance(intent, dict):
        return EffectiveCommand(
            # A task-level intent (for example "clear_universe") is not a
            # page operation. Keep the repair namespace page-local instead of
            # duplicating a known handler under the global task name.
            operation="advance_page",
            source="unresolved_default",
            intent_id=str(intent.get("intent_id") or "") or None,
            intent_kind=str(intent.get("kind") or "") or None,
            intent_phase=str(intent.get("phase") or "") or None,
            params=dict(intent.get("params") or {}),
            persistent=False,
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
    visit = ensure_page_visit(runtime, state_id)
    visit_id = str(visit["visit_id"])
    failed_attempts: list[dict[str, Any]] = []
    excluded: set[str] = blocked_strategies(runtime, visit_id, command.operation)
    total_actions = 0
    made_progress = False

    _log(logger, f"[fsm][handler] enter state={state_id} visit={visit_id} operation={command.operation} source={command.source}", "page_handler_enter", state_id=state_id, visit_id=visit_id, action_id=action_id, command=command.to_dict(), intent=intent_projection(intent), entry_context=entry_context or {"mode": "normal"})

    while total_actions < LOCAL_FLOW_MAX_STEPS:
        continuation = continuation_for(runtime, visit_id, command.operation)
        if operation_exhausted(runtime, visit_id, command.operation) and continuation is None:
            _log(logger, f"[fsm][circuit-breaker] state={state_id} visit={visit_id} operation={command.operation} reason=attempt-limit", "page_handler_circuit_breaker", state_id=state_id, visit_id=visit_id, operation=command.operation)
            return False, current
        preferred_strategy_id = str(continuation.get("strategy_id") or "") if continuation is not None else None
        strategy = select_strategy(handler, command.operation, excluded=excluded, preferred_id=preferred_strategy_id)
        if strategy is None:
            # Before spending the operation's LLM repair budget, try one
            # bounded, feedback-driven local exploration.  This is useful for
            # same-family layouts whose geometry is different but whose
            # operation surface is still known (for example 004 station
            # selection with one vs. three cards).
            exploration_key = f"{visit_id}|{command.operation}"
            explored = runtime.get("visit_operation_exploration")
            explored = explored if isinstance(explored, dict) else {}
            if not explored.get(exploration_key):
                explored[exploration_key] = True
                runtime["visit_operation_exploration"] = explored
                _save_runtime(runtime)
                policy = handler.get("operation_policies", {}).get(command.operation, {}) if isinstance(handler.get("operation_policies"), dict) else {}
                profiles = policy.get("exploration") if isinstance(policy, dict) else None
                policy_safety = str(policy.get("safety", "unknown")) if isinstance(policy, dict) else "unknown"
                allow_exploration = policy_safety in {"low_risk", "reversible"} or bool(policy.get("allow_exploration")) if isinstance(policy, dict) else False
                exploration_ok, explored_frame, exploration_info = (False, current, {"result": "not_allowed", "actions": 0, "attempts": []})
                if allow_exploration:
                    exploration_ok, explored_frame, exploration_info = run_exploration(
                        emulator=emulator,
                        mapper=mapper,
                        vision=vision,
                        state_id=state_id,
                        operation=command.operation,
                        frame=current,
                        matches_provider=matches_provider,
                        profiles=profiles if isinstance(profiles, list) else None,
                        logger=logger,
                    )
                current = explored_frame
                if logger is not None:
                    logger.event(
                        "page_handler_exploration_result",
                        state_id=state_id,
                        visit_id=visit_id,
                        operation=command.operation,
                        result=exploration_info.get("result"),
                        actions=exploration_info.get("actions", 0),
                        attempts=exploration_info.get("attempts", []),
                    )
                if exploration_ok:
                    made_progress = True
                    return True, current
            if repair_exhausted(runtime, visit_id, command.operation):
                _log(logger, f"[fsm][handler] no strategy state={state_id} operation={command.operation}", "page_handler_no_strategy", state_id=state_id, operation=command.operation, failed_attempts=failed_attempts)
                return made_progress, current
            record_repair(runtime, visit_id, command.operation)
            _save_runtime(runtime)
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
        using_continuation = continuation is not None and str(continuation.get("strategy_id") or "") == strategy_id
        if not using_continuation:
            record_attempt(runtime, visit_id, command.operation)
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
                if using_continuation:
                    clear_continuation(runtime, visit_id, command.operation)
                failed = {"strategy_id": strategy_id, "step_index": step_index, "result": "no_effect", "reason": action_info.get("reason")}
                failed_attempts.append(failed)
                mark_strategy_result(handler, command.operation, strategy_id, "no_effect")
                excluded.add(strategy_id)
                strategy_failed = True
                break

            runtime["pending_operation"] = {
                "attempt_id": f"{visit_id}:{command.operation}:{strategy_id}:{total_actions + 1}",
                "intent_id": command.intent_id,
                "intent_revision": intent.get("revision") if isinstance(intent, dict) else None,
                "state_id": state_id,
                "state_dir": str(state_dir),
                "visit_id": visit_id,
                "operation": command.operation,
                "strategy_id": strategy_id,
                "step_id": action_info.get("step_id"),
                "expected_after": dict(action_info.get("expected_after") or {}),
                "success_event": _event_for_success(command, action_info, final_step=step_index == len(steps) - 1),
                "status": "issued",
                "created_at": _now_iso(),
            }
            _save_runtime(runtime)
            if not execute_action(emulator=emulator, mapper=mapper, vision=vision, state_id=state_id, action_id=action_id, action=action, action_info=action_info, matches_provider=matches_provider, logger=logger, attempt=f"strategy:{strategy_id}:{step_index + 1}"):
                if using_continuation:
                    clear_continuation(runtime, visit_id, command.operation)
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
            successful_targets = [match for match in matches if match.success and str(match.state_id) != state_id]
            review = None
            review_patch: dict[str, Any] | None = None
            switch_strategy_after_progress = False
            relation = str(action_info.get("expected_after", {}).get("state_relation") or "may_leave")
            if not left_state and not final_step and relation == "must_remain":
                # A declared page-local intermediate step is already guarded by
                # the following step; avoid paying for an LLM review between
                # known steps of the same strategy.
                result = "partial_progress" if changed else "no_effect"
            elif not left_state:
                active_continuation = continuation_for(runtime, visit_id, command.operation)
                if active_continuation is not None and str(active_continuation.get("strategy_id") or "") != strategy_id:
                    # The leased strategy is no longer selectable (for example
                    # after an external handler edit). Do not let a stale lease
                    # suppress review of a different strategy.
                    clear_continuation(runtime, visit_id, command.operation)
                    active_continuation = None
                if active_continuation is not None and str(active_continuation.get("strategy_id") or "") == strategy_id:
                    remaining = consume_continuation(runtime, visit_id, command.operation, strategy_id)
                    result = "progress_unknown"
                    _log(
                        logger,
                        f"[fsm][handler][continuation] state={state_id} visit={visit_id} operation={command.operation} strategy={strategy_id} remaining={remaining}",
                        "page_handler_continuation_consumed",
                        state_id=state_id,
                        visit_id=visit_id,
                        operation=command.operation,
                        strategy_id=strategy_id,
                        remaining_actions=remaining,
                    )
                elif not review_exhausted(runtime, visit_id, command.operation):
                    record_review(runtime, visit_id, command.operation)
                    _save_runtime(runtime)
                    review = request_same_state_review(
                        llm=llm,
                        session_id=llm_session_id,
                        before_frame_rgb=current,
                        after_frame_rgb=post,
                        system_prompt=system_prompt,
                        state_meta=state_meta,
                        handler=handler,
                        command=command.to_dict(),
                        executed_action={"strategy_id": strategy_id, "step": step, "action_info": action_info},
                        logger=logger,
                    )
                    runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0) or 0) + 1
                if active_continuation is None:
                    verdict = str((review or {}).get("verdict") or "uncertain")
                    # A default operation that claims completion while leaving the
                    # same page would simply be selected again by the outer FSM.
                    # Treat that as an incomplete page advance.
                    if verdict == "completed_same_visit" and command.source == "state_default" and final_step:
                        verdict = "partial_needs_continue"
                    result = result_for_same_state_verdict(verdict, final_step=final_step)
                    if verdict == "partial_needs_continue":
                        continuation_spec = review.get("continuation") if isinstance(review, dict) and isinstance(review.get("continuation"), dict) else {}
                        reuse_strategy = continuation_spec.get("reuse_strategy", True) is not False
                        if reuse_strategy:
                            try:
                                continuation_actions = int(continuation_spec.get("max_additional_actions", 6) or 6)
                            except (TypeError, ValueError):
                                continuation_actions = 6
                            lease = grant_continuation(
                                runtime,
                                visit_id,
                                command.operation,
                                strategy_id,
                                max_additional_actions=continuation_actions,
                            )
                            _log(
                                logger,
                                f"[fsm][handler][continuation] granted state={state_id} visit={visit_id} operation={command.operation} strategy={strategy_id} actions={lease['remaining_actions']}",
                                "page_handler_continuation_granted",
                                state_id=state_id,
                                visit_id=visit_id,
                                operation=command.operation,
                                strategy_id=strategy_id,
                                remaining_actions=lease["remaining_actions"],
                            )
                        else:
                            raw_patch = review.get("handler_patch") if isinstance(review, dict) and isinstance(review.get("handler_patch"), dict) else {}
                            filtered_patch = dict(raw_patch)
                            filtered_patch["operations"] = [
                                item for item in raw_patch.get("operations", [])
                                if isinstance(item, dict) and str(item.get("operation") or "").strip() == command.operation
                            ]
                            review_patch = filtered_patch
                            switch_strategy_after_progress = True
                    if result in {"no_effect", "wrong_transition"}:
                        raw_patch = review.get("handler_patch") if isinstance(review, dict) and isinstance(review.get("handler_patch"), dict) else {}
                        filtered_patch = dict(raw_patch)
                        filtered_patch["operations"] = [
                            item for item in raw_patch.get("operations", [])
                            if isinstance(item, dict) and str(item.get("operation") or "").strip() == command.operation
                        ]
                        review_patch = filtered_patch
                        sibling_patch = continuation_patch_from_sibling(handler, command.operation)
                        if sibling_patch is not None:
                            review_patch = sibling_patch
                            _log(logger, f"[fsm][handler][same-state-review] reuse sibling continuation operation={command.operation}", "same_state_sibling_continuation", operation=command.operation)
            elif successful_targets:
                clear_continuation(runtime, visit_id, command.operation)
                result = "verified_success" if final_step else "partial_progress"
            else:
                clear_continuation(runtime, visit_id, command.operation)
                result = "transient_unknown"
            allowed = action_info.get("expected_after", {}).get("allowed_state_ids")
            if left_state and isinstance(allowed, list):
                observed = {m.state_id for m in matches if m.success}
                if observed and not observed.intersection({str(v) for v in allowed}):
                    result = "wrong_transition"

            if result == "transient_unknown":
                runtime["pending_operation"].update({
                    "status": "awaiting_resolution",
                    "departure_evidence": "unknown",
                })
            else:
                runtime["pending_operation"] = None
                mark_strategy_result(handler, command.operation, strategy_id, result)
            # Apply the review correction only after settling the executed
            # strategy. If the model reuses its id, the replacement must remain
            # proposed rather than inheriting this attempt's failure.
            if review_patch is not None:
                touched = apply_handler_patch(handler, review_patch)
                if touched:
                    materialize_strategy_templates(handler, state_dir, post, vision, touched)
                    excluded.difference_update(touched)
                    _log(logger, f"[fsm][handler][same-state-review] verdict={verdict} patched={sorted(touched)}", "same_state_review_patch", verdict=verdict, touched=sorted(touched))
            if result in {"verified_success", "verified_reentry", "partial_progress"}:
                made_progress = True
            if result == "verified_success" and not current_state_matches:
                clear_visit_progress(runtime, visit_id)
            if result in {"verified_success", "verified_reentry"} and command.source == "unresolved_default":
                promote_operation_to_default(handler, command.operation)
            reviewed_event = review.get("event") if isinstance(review, dict) and isinstance(review.get("event"), dict) else None
            event = (reviewed_event or _event_for_success(command, action_info, final_step=final_step)) if result in {"verified_success", "verified_reentry", "partial_progress"} else None
            reduce_intent_event(runtime, event)
            previous_step_verified = result in {"verified_success", "verified_reentry", "partial_progress"}
            previous_event = event
            append_episode(handler, {"state_id": state_id, "visit_id": visit_id, "command": command.to_dict(), "strategy_id": strategy_id, "step_id": action_info.get("step_id"), "result": result, "event": event, "changed": changed, "diff_score": diff_score, "same_state_review": review})
            state_meta["updated_at"] = _now_iso()
            _save_json(state_path, state_meta)
            _save_runtime(runtime)
            if logger is not None:
                logger.event("page_handler_step_result", state_id=state_id, visit_id=visit_id, action_id=action_id, operation=command.operation, strategy_id=strategy_id, step=step_index + 1, result=result, changed=changed, diff_score=diff_score, same_state_review=review, semantic_event=event, candidates=[summarize_match(m) for m in matches])

            if result in {"no_effect", "wrong_transition", "unsafe_effect"}:
                count = record_no_progress(runtime, visit_id, command.operation, strategy_id, result)
                failed_attempts.append({"strategy_id": strategy_id, "step_index": step_index, "result": result, "diff_score": diff_score})
                excluded.add(strategy_id)
                strategy_failed = True
                current = post
                _log(logger, f"[fsm][loop-guard] state={state_id} visit={visit_id} operation={command.operation} strategy={strategy_id} result={result} count={count}", "page_handler_no_progress", state_id=state_id, visit_id=visit_id, operation=command.operation, strategy_id=strategy_id, result=result, count=count)
                if left_state:
                    nxt = _resolve_transition_after_progress(state_id=state_id, action_id=action_id, matches=matches, graph=graph, runtime=runtime, prefer_reachable_first=prefer_reachable_first, logger=logger, reason_suffix="-page-handler-failed-policy")
                    if nxt is not None:
                        return True, post
                    return _defer_unknown_transition(runtime=runtime, state_id=state_id, action_id=action_id, logger=logger, frame=post, reason="page-handler-failed-policy-left-state")
                break

            if switch_strategy_after_progress:
                # The executed action was useful, but the next page-local step
                # requires a different resolver/action. Continue immediately
                # with the patched strategy without globally degrading the
                # strategy that produced the partial progress.
                excluded.add(strategy_id)
                strategy_failed = True
                current = post
                break

            if result == "verified_reentry":
                clear_continuation(runtime, visit_id, command.operation)
                clear_visit_progress(runtime, visit_id)
                new_visit = start_new_visit(runtime, state_id, reason="confirmed_reentry")
                _append_graph_edge(state_id, action_id, state_id, logger=logger, reason="page-handler-confirmed-reentry", confidence="strong", transition_kind="reentry")
                _log(logger, f"[fsm][visit][reentry] state={state_id} from={visit_id} to={new_visit['visit_id']}", "page_visit_reentry", state_id=state_id, old_visit_id=visit_id, new_visit_id=new_visit["visit_id"])
                _save_runtime(runtime)
                return True, post

            expected_exit = str(action_info.get("expected_after", {}).get("state_relation")) == "must_leave"
            if result == "transient_unknown":
                return _defer_unknown_transition(runtime=runtime, state_id=state_id, action_id=action_id, logger=logger, frame=post, reason="page-handler-awaiting-settlement")
            if final_step or expected_exit:
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
