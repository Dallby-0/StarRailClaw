from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.constants import ACTION_CLICK_WAIT_S, LLM_PARSE_RETRY, MAX_RETRY_PER_ACTION, REPAIR_EFFORTS, SAME_EXTERNAL_STATE_REPAIR_THRESHOLD
from state_machine.graph import _get_reachable_targets
from state_machine.io import _append_experience, _load_json, _now_iso, _save_json, _save_runtime
from state_machine.llm_tasks import (
    _apply_reasoning_effort,
    _build_user_message_from_frame,
    _normalize_assistant_text,
    _parse_llm_repair,
    _request_llm_failure_diagnosis,
    _save_llm_raw_debug,
)
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import MatchResult, _find_match_by_state
from state_machine.presets import run_preset
from state_machine.screen import _screen_changed
from state_machine.state_builder import _normalize_actions
from state_machine.transition_policy import _defer_unknown_transition, _pick_reachable_transition_candidate


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _request_llm_repair(
    llm: DoubaoClient,
    session_id: str,
    frame_rgb,
    system_prompt: str,
    effort: str,
    state_slug: str,
    last_actions: list[dict[str, Any]],
    fail_index: int,
    logger: FsmRunLogger | None = None,
) -> dict[str, Any] | None:
    old_effort = _apply_reasoning_effort(llm, effort)
    text = (
        "MODE=REPAIR\n"
        "输出严格 JSON：{\"judgement\":\"invalid|partial\",\"mode\":\"override|append\",\"actions\":[...],\"note\":\"...\"}\n"
        f"当前状态: {state_slug}\n"
        f"已执行动作: {json.dumps(last_actions, ensure_ascii=False)}\n"
        f"结果: 执行后仍停留原状态（第{fail_index}次纠错）\n"
        "请判断上一步动作是无效还是部分有效，并给出纠正动作。"
    )
    try:
        for attempt in range(1, LLM_PARSE_RETRY + 2):
            msg = _build_user_message_from_frame(frame_rgb, text)
            resp = llm.chat_with_session(
                session_id=session_id,
                system_prompt=system_prompt,
                user_message=msg,
                tools=[],
                tool_choice="none",
            )
            raw = _normalize_assistant_text(resp["choices"][0]["message"].get("content"))
            _save_llm_raw_debug(session_id, attempt, raw, f"repair_{effort}", logger.llm_raw_dir if logger is not None else None)
            _log(logger, f"[fsm][repair][llm] effort={effort} attempt={attempt} raw={raw[:600]}", "llm_repair_raw", effort=effort, attempt=attempt, raw_preview=raw[:600])
            parsed = _parse_llm_repair(raw)
            if parsed is not None:
                parsed["actions"] = _normalize_actions(parsed.get("actions", []))
                if logger is not None:
                    logger.event(
                        "llm_repair_parsed",
                        effort=effort,
                        attempt=attempt,
                        judgement=parsed.get("judgement"),
                        mode=parsed.get("mode"),
                        actions=parsed.get("actions", []),
                    )
                return parsed
    finally:
        llm.reasoning_effort = old_effort
    return None


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
        "note": "screen change is only a hint; decide whether believed state may be wrong before repairing action",
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


def _apply_repair_result(action_steps: list[dict[str, Any]], repair: dict[str, Any]) -> list[dict[str, Any]]:
    new_steps = repair.get("actions", [])
    if repair.get("mode") == "override":
        return list(new_steps)
    return list(action_steps) + list(new_steps)


def _reset_same_external_state_counter(runtime: dict[str, Any]) -> None:
    runtime["same_external_state_id"] = None
    runtime["same_external_state_count"] = 0


def _bump_same_external_state_counter(runtime: dict[str, Any], state_id: str) -> int:
    if runtime.get("same_external_state_id") == state_id:
        count = int(runtime.get("same_external_state_count", 0) or 0) + 1
    else:
        count = 1
    runtime["same_external_state_id"] = state_id
    runtime["same_external_state_count"] = count
    return count


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
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    from state_machine.page_op_flow import _run_page_op_flow

    state_meta = _load_json(state_dir / "state.json")
    actions = [a for a in state_meta.get("actions", []) if isinstance(a, dict) and a.get("enabled", True)]
    if not actions:
        _log(logger, f"[fsm][reason] action not triggered: state={state_id} has no enabled actions", "action_skipped", state_id=state_id, reason="no_enabled_actions")
        return False, frame_before
    action = actions[0]
    action_id = str(action.get("action_id", "action_main"))
    steps = action.get("steps", [])
    if not isinstance(steps, list) or not steps:
        _log(logger, f"[fsm][reason] action not triggered: state={state_id} action={action_id} has empty steps", "action_skipped", state_id=state_id, action_id=action_id, reason="empty_steps")
        return False, frame_before

    last_attempt_frame = frame_before
    last_failure_frame = frame_before
    last_failure_matches: list[MatchResult] = []
    same_state_threshold_reached = False
    for attempt in range(1, MAX_RETRY_PER_ACTION + 1):
        _log(logger, f"[fsm][action] state={state_id} action={action_id} attempt={attempt}", "action_attempt", state_id=state_id, action_id=action_id, attempt=attempt, steps=steps)
        if not _execute_action_steps(emulator, mapper, vision, state_dir, steps, state_id, matches_provider, logger=logger, action_id=action_id, attempt=attempt):
            return False, frame_before
        post = emulator.screenshot(prefer_png=True)
        matches = matches_provider(post)
        last_failure_frame = post
        last_failure_matches = matches
        reachable = _get_reachable_targets(graph, state_id)
        curr_match = _find_match_by_state(matches, state_id)
        if logger is not None:
            logger.event(
                "post_action_match",
                state_id=state_id,
                action_id=action_id,
                attempt=attempt,
                reachable=sorted(reachable),
                candidates=[summarize_match(m) for m in matches],
            )
        nxt, confidence = _pick_reachable_transition_candidate(matches, state_id, reachable if prefer_reachable_first else set())
        if nxt is not None:
            reason = "reachable"
            _log(logger, f"[fsm][transition][{reason}] from={state_id} action={action_id} to={nxt.state_id} confidence={confidence}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason=reason, confidence=confidence)
        elif reachable:
            _log(logger, f"[fsm][transition][reachable-miss] from={state_id} action={action_id} reachable={sorted(reachable)} result={confidence}", "transition_reachable_miss", from_state=state_id, action_id=action_id, reachable=sorted(reachable), result=confidence)
        if nxt is not None and nxt.state_id != state_id:
            runtime["last_state_id"] = nxt.state_id
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
            _reset_same_external_state_counter(runtime)
            _save_runtime(runtime)
            return True, post
        if curr_match is None or not curr_match.success:
            _reset_same_external_state_counter(runtime)
            return _defer_unknown_transition(
                runtime=runtime,
                state_id=state_id,
                action_id=action_id,
                logger=logger,
                frame=post,
                reason="original-state-no-longer-matched",
            )
        same_count = _bump_same_external_state_counter(runtime, state_id)
        _save_runtime(runtime)
        if same_count >= SAME_EXTERNAL_STATE_REPAIR_THRESHOLD:
            same_state_threshold_reached = True
            _log(
                logger,
                f"[fsm][same-state] state={state_id} count={same_count}; trigger diagnosis",
                "same_external_state_repair_threshold",
                state_id=state_id,
                action_id=action_id,
                count=same_count,
                threshold=SAME_EXTERNAL_STATE_REPAIR_THRESHOLD,
            )
            break
        changed, diff_score = _screen_changed(last_attempt_frame, post)
        _log(
            logger,
            f"[fsm][local] self-loop state={state_id} diff={diff_score:.4f}; enter page-op flow",
            "page_local_enter",
            state_id=state_id,
            action_id=action_id,
            attempt=attempt,
            diff_score=diff_score,
            changed=changed,
        )
        local_ok, local_frame = _run_page_op_flow(
            emulator=emulator,
            mapper=mapper,
            vision=vision,
            state_id=state_id,
            state_dir=state_dir,
            action_id=action_id,
            start_frame=post,
            matches_provider=matches_provider,
            runtime=runtime,
            llm=llm,
            llm_session_id=llm_session_id,
            system_prompt=system_prompt,
            graph=graph,
            prefer_reachable_first=prefer_reachable_first,
            state_slug=str(state_meta.get("slug", state_id)),
            seed_step={
                "node": "root",
                "source": "entry_action_self_loop",
                "action": steps[0],
                "brief": f"entry action attempt={attempt}",
                "runtime_observation": {"same_state_still_matches": True, "screen_changed_hint": changed, "diff_score": diff_score},
            } if len(steps) == 1 and isinstance(steps[0], dict) else None,
            seed_before_frame=last_attempt_frame,
            logger=logger,
        )
        if local_ok:
            return True, local_frame
        break

    diagnosis = _diagnose_before_repair(
        llm=llm,
        llm_session_id=llm_session_id,
        frame_rgb=last_failure_frame,
        system_prompt=system_prompt,
        state_meta=state_meta,
        steps=steps,
        matches=last_failure_matches,
        runtime=runtime,
        logger=logger,
    )
    if diagnosis and diagnosis.get("diagnosis") == "state_misidentified":
        runtime["last_state_id"] = None
        runtime["last_transition_ok"] = True
        runtime["pending_from_state_id"] = state_id
        runtime["pending_action_id"] = action_id
        runtime["force_state_resolution"] = True
        runtime["force_exclude_state_id"] = state_id
        _save_runtime(runtime)
        _log(logger, f"[fsm][diagnosis] state_misidentified; force state resolution", "state_misidentified", state_id=state_id, action_id=action_id, diagnosis=diagnosis)
        return True, last_failure_frame
    if same_state_threshold_reached and diagnosis:
        _log(
            logger,
            f"[fsm][same-state] diagnosis={diagnosis.get('diagnosis')} did not request forced resolution; continue repair",
            "same_external_state_continue_repair",
            state_id=state_id,
            action_id=action_id,
            diagnosis=diagnosis,
        )

    for idx, effort in enumerate(REPAIR_EFFORTS, start=1):
        post = emulator.screenshot(prefer_png=True)
        repair = _request_llm_repair(
            llm=llm,
            session_id=llm_session_id,
            frame_rgb=post,
            system_prompt=system_prompt,
            effort=effort,
            state_slug=str(state_meta.get("slug", state_id)),
            last_actions=steps,
            fail_index=idx,
            logger=logger,
        )
        runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
        _save_runtime(runtime)
        if repair is None:
            continue
        if not repair.get("actions"):
            _log(logger, f"[fsm][repair] skip empty actions effort={effort}", "repair_skipped", effort=effort, reason="empty_actions")
            continue
        steps = _apply_repair_result(steps, repair)
        action["version"] = int(action.get("version", 1)) + 1
        action["steps"] = steps
        state_meta["updated_at"] = _now_iso()
        _save_json(state_dir / "state.json", state_meta)

        _log(logger, f"[fsm][repair] effort={effort} mode={repair.get('mode')} judgement={repair.get('judgement')}", "repair_applied", effort=effort, mode=repair.get("mode"), judgement=repair.get("judgement"), steps=steps)
        if not _execute_action_steps(emulator, mapper, vision, state_dir, steps, state_id, matches_provider, logger=logger, action_id=action_id, attempt=f"repair:{effort}"):
            continue
        post2 = emulator.screenshot(prefer_png=True)
        matches2 = matches_provider(post2)
        reachable2 = _get_reachable_targets(graph, state_id)
        curr_match2 = _find_match_by_state(matches2, state_id)
        if logger is not None:
            logger.event(
                "post_repair_match",
                state_id=state_id,
                action_id=action_id,
                effort=effort,
                reachable=sorted(reachable2),
                candidates=[summarize_match(m) for m in matches2],
            )
        nxt2, confidence2 = _pick_reachable_transition_candidate(matches2, state_id, reachable2 if prefer_reachable_first else set())
        if nxt2 is not None:
            reason2 = "reachable-after-repair"
            _log(logger, f"[fsm][transition][{reason2}] from={state_id} action={action_id} to={nxt2.state_id} effort={effort} confidence={confidence2}", "transition", from_state=state_id, action_id=action_id, to_state=nxt2.state_id, reason=reason2, effort=effort, confidence=confidence2)
        elif reachable2:
            _log(logger, f"[fsm][transition][reachable-miss] from={state_id} action={action_id} after_repair={effort} result={confidence2}", "transition_reachable_miss", from_state=state_id, action_id=action_id, reachable=sorted(reachable2), effort=effort, result=confidence2)
        if nxt2 is not None and nxt2.state_id != state_id:
            runtime["last_state_id"] = nxt2.state_id
            runtime["repair_fail_count"] = 0
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
            _reset_same_external_state_counter(runtime)
            _save_runtime(runtime)
            _append_experience(f"repair success: {state_meta.get('slug', state_id)} -> {nxt2.state_id}; mode={repair.get('mode')} effort={effort}")
            return True, post2
        if curr_match2 is None or not curr_match2.success:
            _reset_same_external_state_counter(runtime)
            _append_experience(f"repair success(to-unknown): {state_meta.get('slug', state_id)}; mode={repair.get('mode')} effort={effort}")
            return _defer_unknown_transition(
                runtime=runtime,
                state_id=state_id,
                action_id=action_id,
                logger=logger,
                frame=post2,
                reason="to-unknown-after-repair",
                effort=effort,
            )
        changed2, diff_score2 = _screen_changed(post, post2)
        _log(
            logger,
            f"[fsm][local] repair self-loop state={state_id} diff={diff_score2:.4f}; enter page-op flow",
            "page_local_enter_after_repair",
            state_id=state_id,
            action_id=action_id,
            effort=effort,
            diff_score=diff_score2,
            changed=changed2,
        )
        local_ok2, local_frame2 = _run_page_op_flow(
            emulator=emulator,
            mapper=mapper,
            vision=vision,
            state_id=state_id,
            state_dir=state_dir,
            action_id=action_id,
            start_frame=post2,
            matches_provider=matches_provider,
            runtime=runtime,
            llm=llm,
            llm_session_id=llm_session_id,
            system_prompt=system_prompt,
            graph=graph,
            prefer_reachable_first=prefer_reachable_first,
            state_slug=str(state_meta.get("slug", state_id)),
            seed_step={
                "node": "root",
                "source": f"repair_entry_action_self_loop:{effort}",
                "action": steps[0],
                "brief": f"repair entry action effort={effort}",
                "runtime_observation": {"same_state_still_matches": True, "screen_changed_hint": changed2, "diff_score": diff_score2},
            } if len(steps) == 1 and isinstance(steps[0], dict) else None,
            seed_before_frame=post,
            logger=logger,
        )
        if local_ok2:
            runtime["repair_fail_count"] = 0
            _save_runtime(runtime)
            return True, local_frame2

    runtime["repair_fail_count"] = int(runtime.get("repair_fail_count", 0)) + 1
    runtime["last_transition_ok"] = False
    _save_runtime(runtime)
    _log(logger, "[fsm][repair] failed for low/mid/high, exiting", "repair_failed", state_id=state_id, action_id=action_id, repair_fail_count=runtime["repair_fail_count"])
    raise SystemExit(1)
