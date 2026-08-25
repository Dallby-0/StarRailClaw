from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.adb import resolve_target_serial
from sr_tools.emulator import EmulatorClient
from state_machine.action_runner import _execute_state_action
from state_machine.constants import FSM_GRAPH_PATH, UNKNOWN_STABILITY_RETRY_WAIT_S
from state_machine.disambiguation import _disambiguate_matches, _successful_matches
from state_machine.graph import _append_graph_edge, _append_graph_node
from state_machine.io import (
    _build_system_prompt_with_experience,
    _ensure_fsm_resources,
    _load_json,
    _load_runtime,
    _save_runtime,
)
from state_machine.intent import active_intent, adopt_intent_proposal, ensure_intent_runtime, intent_projection
from state_machine.llm_tasks import _request_llm_payload
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import _eval_state_match, _select_best_for_unknown
from state_machine.merge import _try_merge_page_type
from state_machine.screen import _wait_for_unknown_screen_stable
from state_machine.settlement import settle_pending_operation
from state_machine.session_runtime import _maybe_rotate_session, _refresh_session_if_needed
from state_machine.state_builder import _create_state_from_llm
from state_machine.state_store import (
    _add_state_sample,
    _iter_state_meta,
    _meta_for_state,
    _page_type_summaries,
)
from state_machine.tasks import configure_fsm_workspace, create_task_workspace, task_workspace_path


def _consume_resume_hint(runtime: dict, state_id: str, logger: FsmRunLogger | None = None) -> dict:
    stack = runtime.get("resume_stack")
    if not isinstance(stack, list) or not stack:
        return {"mode": "normal"}
    normalized: list[dict] = []
    matched_index: int | None = None
    for idx, raw in enumerate(stack):
        if not isinstance(raw, dict):
            continue
        try:
            ttl = int(raw.get("ttl", 0) or 0) - 1
        except Exception:
            ttl = 0
        if ttl < 0:
            continue
        hint = dict(raw)
        hint["ttl"] = ttl
        normalized.append(hint)
        if str(hint.get("return_state_id") or "") == state_id:
            matched_index = len(normalized) - 1
    if matched_index is None:
        runtime["resume_stack"] = normalized
        _save_runtime(runtime)
        return {"mode": "normal"}
    hint = normalized[matched_index]
    runtime["resume_stack"] = normalized[:matched_index]
    _save_runtime(runtime)
    if logger is not None:
        logger.event("resume_hint_consumed", state_id=state_id, hint=hint, remaining_depth=len(runtime["resume_stack"]))
    return {"mode": "resume", "resume_hint": hint}


def run_agent_loop_fsm(
    *,
    session_id: str,
    serial: str | None = None,
    adb_path: str | None = None,
    interval_s: float = 5.0,
    task: str | None = None,
    task_dir: str | Path | None = None,
) -> None:
    workspace = task_workspace_path(task, task_dir)
    if task and task_dir is None and not workspace.exists():
        workspace = create_task_workspace(task)
        print(f"[fsm][task] created missing task workspace dir={workspace}")
    workspace = configure_fsm_workspace(workspace)
    _ensure_fsm_resources()
    target_serial = resolve_target_serial(serial=serial, adb_path=adb_path, auto_connect=True)
    emulator = EmulatorClient(serial=target_serial, adb_path=adb_path)
    llm = DoubaoClient()
    mapper = CoordinateMapper(logical_w=1000, logical_h=1000, real_w=1280, real_h=720)
    vision = VisionEngine(mapper=mapper, log_ocr_calls=False)

    runtime = _load_runtime()
    runtime["run_id"] = uuid.uuid4().hex[:8]
    runtime["last_state_id"] = None
    runtime["pending_refresh"] = True
    runtime.setdefault("pending_from_state_id", None)
    runtime.setdefault("pending_action_id", None)
    runtime.setdefault("same_external_state_id", None)
    runtime.setdefault("same_external_state_count", 0)
    runtime.setdefault("force_state_resolution", False)
    runtime.setdefault("force_exclude_state_id", None)
    ensure_intent_runtime(runtime)
    _save_runtime(runtime)
    logger = FsmRunLogger(session_id, str(runtime["run_id"]))
    logger.event(
        "run_start",
        session_id=session_id,
        serial=target_serial,
        adb_path=adb_path,
        interval_s=interval_s,
        task=task,
        task_dir=str(workspace),
        run_dir=logger.run_dir,
        events_path=logger.events_path,
        summary_path=logger.summary_path,
        report_path=logger.report_path,
        latest_report_path=logger.latest_report_path,
        llm_raw_dir=logger.llm_raw_dir,
    )
    logger.text(f"[fsm][log] dir={logger.run_dir}", "log_path", run_dir=logger.run_dir, report_path=logger.report_path, latest_report_path=logger.latest_report_path)

    prev_frame = None

    while True:
        llm_session_id = _refresh_session_if_needed(runtime, session_id)
        system_prompt = _build_system_prompt_with_experience()
        logger.event(
            "loop_start",
            llm_session_id=llm_session_id,
            llm_turn_count=runtime.get("llm_turn_count", 0),
            session_tier=runtime.get("session_tier"),
            last_state_id=runtime.get("last_state_id"),
            pending_from_state_id=runtime.get("pending_from_state_id"),
            pending_action_id=runtime.get("pending_action_id"),
        )

        frame = emulator.screenshot(prefer_png=True)
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        logger.event("frame_captured", width=int(frame.shape[1]), height=int(frame.shape[0]))

        metas = _iter_state_meta()

        def matches_provider(img):
            vision.reset_ocr_stats()
            # State creation/merge may happen before this outer iteration ends.
            # Reload metadata so the first bootstrap action can observe the
            # state that was just persisted instead of matching against a
            # stale pre-creation snapshot.
            live_metas = _iter_state_meta()
            matches = [_eval_state_match(meta, sdir, vision, img) for sdir, meta in live_metas]
            ocr_stats = vision.consume_ocr_stats()
            calls = int(ocr_stats["calls"])
            if calls > 0:
                elapsed_s = float(ocr_stats["elapsed_s"])
                logger.text(
                    f"[vision][ocr] summary calls={calls} successes={int(ocr_stats['successes'])} errors={int(ocr_stats['errors'])} elapsed={elapsed_s:.2f}s",
                    "ocr_summary",
                    calls=calls,
                    successes=int(ocr_stats["successes"]),
                    errors=int(ocr_stats["errors"]),
                    elapsed_s=elapsed_s,
                )
            return matches

        matches = matches_provider(frame)
        dbg = [f"{m.state_id}:{m.passed_enabled}/{m.total_enabled}|all={m.passed_all}/{m.total_all}|ok={m.success}" for m in matches]
        logger.text(f"[fsm][match] candidates={dbg}", "match_candidates", candidates=[summarize_match(m) for m in matches])

        matches_for_selection = matches
        metas_for_resolution = metas
        successes = _successful_matches(matches)
        force_state_resolution = bool(runtime.get("force_state_resolution", False))
        force_exclude_state_id = str(runtime.get("force_exclude_state_id") or "")
        if force_state_resolution and force_exclude_state_id:
            original_successes = successes
            matches_for_selection = [m for m in matches if m.state_id != force_exclude_state_id]
            metas_for_resolution = [(sdir, meta) for sdir, meta in metas if str(meta.get("state_id", "")) != force_exclude_state_id]
            successes = _successful_matches(matches_for_selection)
            logger.text(
                f"[fsm][forced-resolution] exclude={force_exclude_state_id} "
                f"before={[m.state_id for m in original_successes]} after={[m.state_id for m in successes]}",
                "forced_state_resolution",
                excluded_state_id=force_exclude_state_id,
                candidates_before=[summarize_match(m) for m in original_successes],
                candidates_after=[summarize_match(m) for m in successes],
            )
        pending_resolution_confidence = "matcher_confirmed"
        transition_hint_id = str(runtime.get("last_state_id") or "")
        transition_hint_ok = bool(runtime.get("last_transition_ok", False))
        hinted_best = None
        if transition_hint_ok and transition_hint_id and not force_state_resolution:
            hinted_best = next((m for m in successes if m.state_id == transition_hint_id), None)
        if hinted_best is not None:
            best = hinted_best
            pending_resolution_confidence = "transition_hint"
            logger.text(
                f"[fsm][selection][transition-hint] state={best.state_id}",
                "transition_hint_selected",
                state_id=best.state_id,
                candidates=[summarize_match(m) for m in successes],
            )
        elif len(successes) > 1:
            logger.text(
                f"[fsm][disambiguation] candidates={[m.state_id for m in successes]}",
                "disambiguation_started",
                candidates=[summarize_match(m) for m in successes],
            )
            pending_resolution_confidence = "llm_disambiguated"
            best = _disambiguate_matches(
                llm=llm,
                llm_session_id=llm_session_id,
                frame_rgb=frame,
                system_prompt=system_prompt,
                matches=matches_for_selection,
                metas=metas_for_resolution,
                vision=vision,
                runtime=runtime,
                logger=logger,
            )
        else:
            best = successes[0] if force_state_resolution and len(successes) == 1 else _select_best_for_unknown(matches_for_selection)
        if best is None:
            stable_frame = _wait_for_unknown_screen_stable(emulator, frame, logger=logger)
            if stable_frame is None:
                logger.text(
                    f"[fsm][unknown] screen unstable, wait {UNKNOWN_STABILITY_RETRY_WAIT_S}s before retry",
                    "unknown_unstable_wait",
                    wait_s=UNKNOWN_STABILITY_RETRY_WAIT_S,
                )
                prev_frame = frame
                time.sleep(UNKNOWN_STABILITY_RETRY_WAIT_S)
                continue
            frame = stable_frame
            logger.text("[fsm] unknown state, requesting llm", "unknown_state")
            payload = _request_llm_payload(
                llm,
                llm_session_id,
                frame,
                system_prompt,
                _page_type_summaries(metas_for_resolution),
                active_intent=intent_projection(active_intent(runtime)),
                raw_debug_dir=logger.llm_raw_dir,
            )
            runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
            _save_runtime(runtime)
            if payload is None:
                logger.text("[fsm][llm] invalid payload", "llm_payload_invalid", llm_session_id=llm_session_id)
                prev_frame = frame
                time.sleep(interval_s)
                continue
            logger.event(
                "llm_payload_parsed",
                slug=payload.get("slug"),
                possible_page_type=payload.get("possible_page_type"),
                elements_count=len(payload.get("elements", [])),
                bootstrap_operations_count=len(payload.get("bootstrap_operations", [])),
            )
            proposed_intent = adopt_intent_proposal(runtime, payload.get("intent_proposal"))
            if proposed_intent is not None:
                _save_runtime(runtime)
                logger.event("intent_created_from_state_assessment", intent=intent_projection(proposed_intent))
            merged = _try_merge_page_type(
                llm=llm,
                session_id=llm_session_id,
                system_prompt=system_prompt,
                frame_rgb=frame,
                llm_payload=payload,
                mapper=mapper,
                vision=vision,
                metas=metas_for_resolution,
                logger=logger,
            )
            if merged is None:
                new_state_id, new_state_dir = _create_state_from_llm(payload, frame, mapper, vision, logger=logger)
                _append_graph_node(new_state_id, str(payload.get("slug", "state")), logger=logger)
                logger.text(f"[fsm] new_state state_id={new_state_id} dir={new_state_dir}", "state_created_console", state_id=new_state_id, state_dir=new_state_dir)
            else:
                new_state_id, new_state_dir = merged
                logger.text(f"[fsm] merged_state state_id={new_state_id} dir={new_state_dir}", "state_merged_console", state_id=new_state_id, state_dir=new_state_dir)
            pending_from = runtime.get("pending_from_state_id")
            pending_action = runtime.get("pending_action_id")
            if isinstance(pending_from, str) and pending_from and isinstance(pending_action, str) and pending_action:
                settlement = settle_pending_operation(runtime, new_state_id)
                allow_edge = pending_from != new_state_id or bool(settlement and settlement.get("allow_self_edge"))
                if allow_edge:
                    transition_kind = "reentry" if pending_from == new_state_id else "normal"
                    _append_graph_edge(pending_from, pending_action, new_state_id, logger=logger, reason="pending-unknown-resolution", confidence="llm_verified", transition_kind=transition_kind)
                else:
                    logger.text(f"[fsm][edge][skipped-self] state={new_state_id} settlement={settlement}", "edge_self_skipped", state_id=new_state_id, settlement=settlement)
                runtime["pending_from_state_id"] = None
                runtime["pending_action_id"] = None
            runtime["force_state_resolution"] = False
            runtime["force_exclude_state_id"] = None
            runtime["last_state_id"] = new_state_id
            runtime["last_transition_ok"] = False
            _save_runtime(runtime)
            ok, post_frame = _execute_state_action(
                emulator=emulator,
                mapper=mapper,
                vision=vision,
                state_id=new_state_id,
                state_dir=new_state_dir,
                frame_before=frame,
                matches_provider=matches_provider,
                runtime=runtime,
                llm=llm,
                llm_session_id=llm_session_id,
                system_prompt=system_prompt,
                graph=_load_json(FSM_GRAPH_PATH),
                prefer_reachable_first=False,
                logger=logger,
            )
            prev_frame = post_frame if ok else frame
            _maybe_rotate_session(runtime)
            _save_runtime(runtime)
            time.sleep(interval_s)
            continue

        runtime["last_state_id"] = best.state_id
        runtime["last_transition_ok"] = False
        logger.event("state_selected", state_id=best.state_id, state_dir=best.state_dir, match=summarize_match(best))
        pending_from2 = runtime.get("pending_from_state_id")
        pending_action2 = runtime.get("pending_action_id")
        if isinstance(pending_from2, str) and pending_from2 and isinstance(pending_action2, str) and pending_action2:
            settlement = settle_pending_operation(runtime, best.state_id)
            allow_edge = pending_from2 != best.state_id or bool(settlement and settlement.get("allow_self_edge"))
            if allow_edge:
                transition_kind = "reentry" if pending_from2 == best.state_id else "normal"
                _append_graph_edge(pending_from2, pending_action2, best.state_id, logger=logger, reason="pending-unknown-resolved-to-existing", confidence=pending_resolution_confidence, transition_kind=transition_kind)
                best_meta_item = _meta_for_state(metas, best.state_id)
                if best_meta_item is not None:
                    _add_state_sample(best_meta_item[0], best_meta_item[1], frame, role="positive", source="pending_existing", confidence=0.8, logger=logger)
            else:
                logger.text(f"[fsm][edge][skipped-self] state={best.state_id} settlement={settlement}", "edge_self_skipped", state_id=best.state_id, settlement=settlement)
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
        if force_state_resolution:
            runtime["force_state_resolution"] = False
            runtime["force_exclude_state_id"] = None
        _save_runtime(runtime)

        entry_context = _consume_resume_hint(runtime, best.state_id, logger=logger)
        ok, post_frame = _execute_state_action(
            emulator=emulator,
            mapper=mapper,
            vision=vision,
            state_id=best.state_id,
            state_dir=best.state_dir,
            frame_before=frame,
            matches_provider=matches_provider,
            runtime=runtime,
            llm=llm,
            llm_session_id=llm_session_id,
            system_prompt=system_prompt,
            graph=_load_json(FSM_GRAPH_PATH),
            prefer_reachable_first=True,
            entry_context=entry_context,
            logger=logger,
        )
        prev_frame = post_frame if ok else frame
        _maybe_rotate_session(runtime)
        _save_runtime(runtime)
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="FSM driven emulator agent loop")
    parser.add_argument("--session-id", default=f"session-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--adb-path", default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--task", default=None, help="task name under StateMachineTasks")
    parser.add_argument("--task-dir", default=None, help="explicit task workspace directory")
    parser.add_argument("--create-task", default=None, help="create a task workspace and exit")
    parser.add_argument("--task-summary", default="", help="English summary written when creating a task")
    args = parser.parse_args()
    if args.create_task:
        workspace = create_task_workspace(args.create_task, summary=args.task_summary)
        print(f"[fsm][task] created name={args.create_task} dir={workspace}")
        return
    run_agent_loop_fsm(
        session_id=args.session_id,
        serial=args.serial,
        adb_path=args.adb_path,
        interval_s=args.interval,
        task=args.task,
        task_dir=args.task_dir,
    )


if __name__ == "__main__":
    main()
