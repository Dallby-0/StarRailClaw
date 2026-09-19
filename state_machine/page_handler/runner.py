from __future__ import annotations

import time
import unicodedata
from pathlib import Path
from typing import Any

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.emulator import EmulatorClient
from state_machine.command import EffectiveCommand, resolve_effective_command
from state_machine.graph import _append_graph_edge
from state_machine.intent import active_intent, intent_projection, reduce_intent_event
from state_machine.io import _load_frame, _load_json, _save_json, _save_runtime
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.page_handler.actions import (
    capture_effect_baseline,
    evaluate_effect,
    evaluate_provider_guards,
    execute_action,
    resolve_provider,
)
from state_machine.page_handler.guard_learning import observe_successful_transition
from state_machine.page_handler.llm import request_handler_repair
from state_machine.page_handler.reactive import (
    GLOBAL_HARD_ACTIONS,
    GLOBAL_HARD_REPAIRS,
    cursor_for,
    effective_repair_limit,
    mark_confirmed_effect,
    normalized_patch_hash,
    normalize_budgets,
    rank_provider_details,
    rank_providers,
    record_attempt,
    update_successor_context,
)
from state_machine.page_handler.store import (
    apply_handler_patch,
    append_episode,
    ensure_page_handler,
    materialize_provider_templates,
    record_provider_result,
)
from state_machine.screen import _screen_changed, _wait_for_screen_stable
from state_machine.transition_policy import _defer_unknown_transition, _resolve_transition_after_progress
from state_machine.visit import ensure_page_visit, start_new_visit
from state_machine.page_handler.reactive import clear_visit


MAX_CONSECUTIVE_TEXT_REENTRIES = 2


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)


def _fallback_command(intent: dict[str, Any] | None) -> EffectiveCommand:
    if isinstance(intent, dict):
        return EffectiveCommand(
            operation="advance_page",
            source="unresolved_default",
            intent_id=str(intent.get("intent_id") or "") or None,
            intent_kind=str(intent.get("kind") or "") or None,
            intent_phase=str(intent.get("phase") or "") or None,
            params=dict(intent.get("params") or {}),
            intent_effect="advance",
        )
    return EffectiveCommand(operation="advance", source="unresolved_default")


def _history_item(frame, cell_id: str, **fields: Any) -> dict[str, Any]:
    return {"frame": frame, "cell_id": cell_id, **fields}


def _cursor_summary(cursor: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cursor.items() if key != "history"}


def _provider_guard_results(
    operation: dict[str, Any],
    cursor: dict[str, Any],
    frame,
    vision: VisionEngine,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
    cache: dict[str, dict[str, Any]] = {}
    results: dict[str, list[str]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    for provider in operation.get("providers", []):
        if not isinstance(provider, dict):
            continue
        provider_id = str(provider.get("provider_id") or "")
        values, info = evaluate_provider_guards(provider, cursor, frame, vision, cache, allow_expensive=False)
        results[provider_id] = values
        details[provider_id] = info

    preliminary = rank_providers(operation, cursor, details)
    expensive_limit = normalize_budgets(operation.get("budgets"))["max_expensive_probes"]
    for provider in preliminary[:expensive_limit]:
        provider_id = str(provider.get("provider_id") or "")
        values, info = evaluate_provider_guards(provider, cursor, frame, vision, cache, allow_expensive=True)
        results[provider_id] = values
        details[provider_id] = info
    return results, details


def _text_fingerprint(match: Any) -> tuple[str, ...]:
    """Return only recognized text from state match conditions.

    Empty/unknown OCR is deliberately represented by an empty tuple; it must
    never be mistaken for evidence that a self-loop advanced.
    """
    details = getattr(match, "condition_results", None)
    if not isinstance(details, list):
        return ()
    values: list[str] = []
    for detail in details:
        if not isinstance(detail, dict) or detail.get("kind") != "text_line_contains":
            continue
        lines = detail.get("ocr_lines")
        if isinstance(lines, list):
            values.extend(str(line).strip() for line in lines if str(line).strip())
    return _normalize_fingerprint(values)


def _normalize_fingerprint(values: list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        text = "".join(
            char
            for char in unicodedata.normalize("NFKC", str(value)).strip().casefold()
            if not unicodedata.category(char).startswith(("P", "S", "Z"))
        )
        if text:
            normalized.append(text)
    return tuple(normalized)


def _effect_has_observable_change(effect_details: list[dict[str, Any]]) -> bool:
    for detail in effect_details:
        if not isinstance(detail, dict) or detail.get("matched") is not True:
            continue
        # ``expected=pass`` is a level assertion, not evidence of progress;
        # transitions such as becomes_pass/becomes_fail are.
        if str(detail.get("expected") or "pass") != "pass":
            return True
        if detail.get("before") != detail.get("after"):
            return True
    return False


def _element_text_fingerprint(state_meta: dict[str, Any], frame: Any, vision: VisionEngine) -> tuple[str, ...]:
    """Read a few model-labeled text regions for same-state instance changes.

    This is intentionally a fallback: normal matching already provides OCR
    for identity conditions.  It runs only when a terminal provider has no
    other progress evidence, and remains bounded to small regions rather than
    triggering a full-screen OCR pass.
    """
    ocr = getattr(vision, "ocr", None)
    if not callable(ocr):
        return ()
    values: list[str] = []
    elements = state_meta.get("elements") if isinstance(state_meta.get("elements"), list) else []
    for element in elements:
        if not isinstance(element, dict) or element.get("type") != "text_line":
            continue
        role = str(element.get("role") or "")
        instance_like_diagnostic = role == "diagnostic" and str(element.get("stability") or "mid") != "high"
        if role != "instance" and not instance_like_diagnostic:
            continue
        rect = element.get("bbox")
        if not (isinstance(rect, list) and len(rect) == 4):
            continue
        try:
            entries = ocr(frame, rect, white_text=False)
        except Exception:
            continue
        values.extend(str(item.get("text") or "").strip() for item in entries if isinstance(item, dict) and str(item.get("text") or "").strip())
        if len(values) >= 4:
            break
    return _normalize_fingerprint(values)


def _record_text_only_reentry(runtime: dict[str, Any], state_id: str) -> int:
    previous = str(runtime.get("text_only_reentry_state_id") or "")
    count = int(runtime.get("text_only_reentry_count", 0) or 0) + 1 if previous == state_id else 1
    runtime["text_only_reentry_state_id"] = state_id
    runtime["text_only_reentry_count"] = count
    return count


def _clear_text_only_reentry(runtime: dict[str, Any]) -> None:
    runtime["text_only_reentry_state_id"] = None
    runtime["text_only_reentry_count"] = 0


def _sample_frames(state_meta: dict[str, Any], current) -> list[Any]:
    frames = [current]
    for sample in state_meta.get("samples", []):
        if not isinstance(sample, dict):
            continue
        path = Path(str(sample.get("path") or ""))
        frame = _load_frame(path) if path.is_file() else None
        if frame is not None:
            frames.append(frame)
            break
    return frames


def _validate_family_candidates(
    operation: dict[str, Any],
    touched: set[str],
    frames: list[Any],
    vision: VisionEngine,
) -> dict[str, str]:
    results: dict[str, str] = {}
    providers = operation.get("providers") if isinstance(operation.get("providers"), list) else []
    has_variant_evidence = len(frames) >= 2 and _screen_changed(frames[0], frames[1])[0]
    for provider in list(providers):
        provider_id = str(provider.get("provider_id") or "") if isinstance(provider, dict) else ""
        if provider_id not in touched or str(provider.get("scope")) != "family":
            continue
        if not has_variant_evidence:
            providers.remove(provider)
            results[provider_id] = "rejected_no_variant_evidence"
            continue
        dynamic_locators = [
            locator for locator in provider.get("locators", [])
            if isinstance(locator, dict) and locator.get("type") != "point"
        ]
        if not dynamic_locators:
            providers.remove(provider)
            results[provider_id] = "rejected_point_only"
            continue
        dynamic_provider = {**provider, "locators": dynamic_locators}
        resolved_all = all(resolve_provider(dynamic_provider, frame, vision)[0] is not None for frame in frames)
        if not resolved_all:
            providers.remove(provider)
            results[provider_id] = "rejected_dry_run"
            continue
        provider["status"] = "canary"
        provider["generalization_evidence"] = {"dry_run_frames": len(frames)}
        results[provider_id] = "canary"
    return results


def _repair(
    *,
    llm: DoubaoClient,
    llm_session_id: str,
    system_prompt: str,
    current,
    history: list[dict[str, Any]],
    state_meta: dict[str, Any],
    handler: dict[str, Any],
    operation: dict[str, Any] | None,
    command: EffectiveCommand,
    cursor: dict[str, Any],
    failures: list[dict[str, Any]],
    state_dir: Path,
    vision: VisionEngine,
    logger: FsmRunLogger | None,
) -> tuple[dict[str, Any] | None, str]:
    response = request_handler_repair(
        llm=llm,
        session_id=llm_session_id,
        current_frame=current,
        history=history,
        system_prompt=system_prompt,
        state_meta=state_meta,
        handler=handler,
        command=command.to_dict(),
        failed_attempts=failures,
        cursor_summary=_cursor_summary(cursor),
        logger=logger,
    )
    if response is None:
        return operation, "invalid_response"
    cursor["progress_assessment"] = str(response.get("progress_assessment") or "unknown")
    if response.get("decision") == "give_up":
        return operation, "give_up"
    if response.get("decision") == "state_misidentified":
        if response.get("confidence") == "high":
            return operation, "state_misidentified"
        return operation, "state_misidentified_unconfirmed"
    patch = {
        "providers": response.get("local_patch", {}).get("providers", []),
        "generalization_candidates": response.get("generalization_candidates", []),
    }
    patch_hash = normalized_patch_hash(patch)
    hashes = cursor.get("repair_hashes") if isinstance(cursor.get("repair_hashes"), list) else []
    cursor["repair_hashes"] = hashes
    if not patch_hash or patch_hash in hashes:
        return operation, "duplicate_patch"
    hashes.append(patch_hash)
    touched = apply_handler_patch(handler, patch, operation=command.operation)
    if not touched:
        return operation, "empty_patch"
    operation = handler["operation_policies"].get(command.operation)
    materialize_provider_templates(handler, state_dir, current, vision, touched)
    validation = _validate_family_candidates(operation, touched, _sample_frames(state_meta, current), vision)
    valid_ids = {str(item.get("provider_id")) for item in operation.get("providers", []) if isinstance(item, dict)}
    attempts = cursor.get("attempt_counts") if isinstance(cursor.get("attempt_counts"), dict) else {}
    for provider_id in touched & valid_ids:
        attempts.pop(provider_id, None)
    cursor["attempt_counts"] = attempts
    cursor["actions_since_repair"] = 0
    _log(logger, f"[fsm][handler][repair] added={sorted(touched & valid_ids)}", "page_handler_repair_applied", operation=command.operation, touched=sorted(touched), validation=validation)
    return operation, "applied" if touched & valid_ids else "no_valid_provider"


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
    entry_context: dict[str, Any] | None = None,
    logger: FsmRunLogger | None = None,
) -> tuple[bool, Any]:
    current = start_frame
    state_path = state_dir / "state.json"
    state_meta = _load_json(state_path)
    execution = state_meta.get("execution") if isinstance(state_meta.get("execution"), dict) else {}
    if str(state_meta.get("scene_mode") or "") != "ui_2d" or execution.get("kind") != "reactive_2d":
        _log(logger, "[fsm][handler] rejected non-reactive state", "page_handler_invalid_route", state_id=state_id, scene_mode=state_meta.get("scene_mode"), execution=execution)
        return False, current
    handler = ensure_page_handler(state_meta)
    intent = active_intent(runtime)
    command = resolve_effective_command(state_meta, intent) or _fallback_command(intent)
    visit_id = str(ensure_page_visit(runtime, state_id)["visit_id"])
    cursor = cursor_for(runtime, visit_id, command.operation, now_monotonic=time.monotonic())
    if float(cursor.get("started_monotonic", 0.0) or 0.0) <= 0:
        cursor["started_monotonic"] = time.monotonic()
    operation = handler.get("operation_policies", {}).get(command.operation)
    budgets = normalize_budgets(operation.get("budgets") if isinstance(operation, dict) else None)
    failures: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    prior_frames = _sample_frames(state_meta, current)[1:]
    if prior_frames:
        history.append(_history_item(prior_frames[0], "family_sample", kind="stored_family_sample", provider_id=None))
    history.append(_history_item(current, "entry", kind="entry", provider_id=None))
    pending_transition: dict[str, Any] | None = None

    _log(logger, f"[fsm][handler] enter state={state_id} visit={visit_id} operation={command.operation}", "page_handler_enter", state_id=state_id, visit_id=visit_id, action_id=action_id, command=command.to_dict(), intent=intent_projection(intent), entry_context=entry_context or {"mode": "normal"}, schema="reactive_handler.v3.1")

    while True:
        elapsed = time.monotonic() - float(cursor.get("started_monotonic", 0.0) or 0.0)
        hard_reason = None
        if int(cursor.get("total_actions", 0) or 0) >= budgets["hard_actions"]:
            hard_reason = "visit_action_limit"
        elif elapsed >= budgets["max_seconds"]:
            hard_reason = "visit_time_limit"
        elif int(runtime.get("reactive_total_actions", 0) or 0) >= GLOBAL_HARD_ACTIONS:
            hard_reason = "run_action_limit"
        elif int(runtime.get("reactive_total_repairs", 0) or 0) >= GLOBAL_HARD_REPAIRS:
            hard_reason = "run_repair_limit"
        if hard_reason:
            _log(logger, f"[fsm][handler] hard stop reason={hard_reason}", "page_handler_hard_limit", state_id=state_id, visit_id=visit_id, operation=command.operation, reason=hard_reason, cursor=_cursor_summary(cursor))
            return False, current

        guard_results: dict[str, list[str]] = {}
        guard_details: dict[str, list[dict[str, Any]]] = {}
        ranked: list[dict[str, Any]] = []
        if isinstance(operation, dict):
            guard_results, guard_details = _provider_guard_results(operation, cursor, current, vision)
            ranking = rank_provider_details(operation, cursor, guard_details)
            ranked = [item["provider"] for item in ranking]
            _log(logger, "[fsm][handler] provider ranking", "page_handler_provider_ranking", ranking=[{key: value for key, value in item.items() if key != "provider"} for item in ranking])
        needs_repair = not ranked or int(cursor.get("actions_since_repair", 0) or 0) >= budgets["soft_actions"]
        if needs_repair:
            repair_limit = effective_repair_limit(cursor, budgets["max_repairs"])
            if int(cursor.get("repair_count", 0) or 0) >= repair_limit:
                _log(logger, "[fsm][handler] repair budget exhausted", "page_handler_no_strategy", state_id=state_id, visit_id=visit_id, operation=command.operation, repair_limit=repair_limit, progress_assessment=cursor.get("progress_assessment"), failed_attempts=failures[-12:])
                return False, current
            cursor["repair_count"] = int(cursor.get("repair_count", 0) or 0) + 1
            runtime["reactive_total_repairs"] = int(runtime.get("reactive_total_repairs", 0) or 0) + 1
            _save_runtime(runtime)
            operation, repair_result = _repair(
                llm=llm,
                llm_session_id=llm_session_id,
                system_prompt=system_prompt,
                current=current,
                history=history,
                state_meta=state_meta,
                handler=handler,
                operation=operation,
                command=command,
                cursor=cursor,
                failures=failures,
                state_dir=state_dir,
                vision=vision,
                logger=logger,
            )
            runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0) or 0) + 1
            _save_json(state_path, state_meta)
            _save_runtime(runtime)
            if repair_result == "state_misidentified":
                runtime["last_state_id"] = None
                runtime["last_transition_ok"] = True
                runtime["pending_from_state_id"] = state_id
                runtime["pending_action_id"] = action_id
                runtime["force_state_resolution"] = True
                runtime["force_exclude_state_id"] = state_id
                _save_runtime(runtime)
                _log(
                    logger,
                    f"[fsm][handler] state misidentified state={state_id}; force state resolution",
                    "state_misidentified",
                    state_id=state_id,
                    visit_id=visit_id,
                    action_id=action_id,
                )
                return True, current
            if repair_result != "applied":
                failures.append({"result": "repair_failed", "reason": repair_result})
                if repair_result in {"give_up", "duplicate_patch", "no_valid_provider"}:
                    return False, current
            budgets = normalize_budgets(operation.get("budgets") if isinstance(operation, dict) else None)
            continue

        provider = ranked[0]
        provider_id = str(provider.get("provider_id") or "")
        before_matches = matches_provider(current)
        before_match = next((item for item in before_matches if item.state_id == state_id), None)
        before_text_fp = _text_fingerprint(before_match)
        baseline = capture_effect_baseline(provider, current, vision)
        action, action_info = resolve_provider(provider, current, vision, cursor)
        if action is None:
            record_attempt(cursor, provider, executed=False)
            record_provider_result(operation, provider_id, "resolver_miss", visit_id=visit_id)
            failure = {"provider_id": provider_id, "result": "resolver_miss", "guard_results": guard_details.get(provider_id, []), "resolver": action_info}
            failures.append(failure)
            append_episode(handler, {"state_id": state_id, "visit_id": visit_id, "operation": command.operation, **failure})
            _save_json(state_path, state_meta)
            continue

        record_attempt(cursor, provider)
        if action_info.get("locator_type") == "click_region" and isinstance(action_info.get("candidate_point"), list):
            locator_attempts = cursor.setdefault("locator_attempts", {})
            locator_attempts.setdefault(provider_id, []).append(action_info["candidate_point"])
        runtime["reactive_total_actions"] = int(runtime.get("reactive_total_actions", 0) or 0) + 1
        executed = execute_action(
            emulator=emulator,
            mapper=mapper,
            vision=vision,
            state_id=state_id,
            action_id=action_id,
            action=action,
            action_info={"provider_id": provider_id, **action_info},
            matches_provider=matches_provider,
            logger=logger,
            attempt=f"reactive:{provider_id}",
        )
        if not executed:
            record_provider_result(operation, provider_id, "action_error", visit_id=visit_id)
            failures.append({"provider_id": provider_id, "result": "action_error"})
            _save_runtime(runtime)
            continue

        post = _wait_for_screen_stable(emulator, logger=logger, label="page-handler-reactive", event="page_handler_stability_check", max_checks=3)
        changed, diff_score = _screen_changed(current, post)
        effect_result, effect_details = evaluate_effect(provider, baseline, post, vision)
        matches = matches_provider(post)
        current_match = next((item for item in matches if item.state_id == state_id), None)
        still_current = bool(current_match and current_match.success)
        after_text_fp = _text_fingerprint(current_match)
        text_progress = bool(changed and before_text_fp and after_text_fp and before_text_fp != after_text_fp)
        if text_progress:
            confirmation_matches = matches_provider(post)
            confirmation_match = next((item for item in confirmation_matches if item.state_id == state_id), None)
            text_progress = _text_fingerprint(confirmation_match) == after_text_fp
        # Identity conditions intentionally omit volatile instance labels. If
        # the action otherwise looks unverified and leaves us on the same
        # state, use bounded local OCR as a final self-reentry signal.
        if (
            still_current
            and not provider.get("successors")
            and not text_progress
            and (effect_result == "unverified" or not _effect_has_observable_change(effect_details))
        ):
            before_instance_fp = _element_text_fingerprint(state_meta, current, vision)
            after_instance_fp = _element_text_fingerprint(state_meta, post, vision)
            confirmed_instance_fp = _element_text_fingerprint(state_meta, post, vision) if after_instance_fp else ()
            if (
                changed
                and before_instance_fp
                and after_instance_fp
                and before_instance_fp != after_instance_fp
                and confirmed_instance_fp == after_instance_fp
            ):
                before_text_fp = before_instance_fp
                after_text_fp = after_instance_fp
                text_progress = True
        # A second successful matcher is ambiguity, not evidence of a
        # transition.  The current state must first stop matching; otherwise
        # an ineffective click can be recorded as a successful transition and
        # poison the graph with alternating self/other edges.
        known_other = next(
            (item for item in matches if item.success and item.state_id != state_id),
            None,
        ) if not still_current else None
        result = "transitioned" if known_other is not None else effect_result
        effect_progress = _effect_has_observable_change(effect_details)
        if result == "unverified" and still_current and text_progress:
            reentry_count = _record_text_only_reentry(runtime, state_id)
            if reentry_count <= MAX_CONSECUTIVE_TEXT_REENTRIES:
                # OCR provides bounded local evidence for a fresh instance.
                result = "confirmed"
            else:
                text_progress = False
                failures.append({
                    "provider_id": provider_id,
                    "result": "suspect_text_only_reentry",
                    "count": reentry_count,
                })
                _log(
                    logger,
                    f"[fsm][handler] text-only reentry limit reached state={state_id} count={reentry_count}",
                    "page_handler_suspect_reentry",
                    state_id=state_id,
                    visit_id=visit_id,
                    provider_id=provider_id,
                    count=reentry_count,
                )
        elif result in {"confirmed", "transitioned"} and effect_progress:
            _clear_text_only_reentry(runtime)
        update_successor_context(cursor, provider, result, effect_details)
        record_provider_result(operation, provider_id, result, visit_id=visit_id)
        learned_guards: list[dict[str, Any]] = []
        if result in {"confirmed", "transitioned"}:
            cleared_repairs = int(cursor.get("repair_count", 0) or 0)
            mark_confirmed_effect(cursor)
            if cleared_repairs:
                runtime["reactive_total_repairs"] = max(0, int(runtime.get("reactive_total_repairs", 0) or 0) - cleared_repairs)
            try:
                learned_guards = observe_successful_transition(
                    operation,
                    provider,
                    current,
                    post,
                    visit_id=visit_id,
                    state_dir=state_dir,
                    vision=vision,
                )
            except Exception as exc:  # noqa: BLE001 - learning must not invalidate a successful action
                learned_guards = [{"error": f"{type(exc).__name__}: {exc}"}]
                _log(
                    logger,
                    f"[fsm][handler] guard learning failed provider={provider_id}: {exc}",
                    "page_handler_guard_learning_failed",
                    provider_id=provider_id,
                    error_type=type(exc).__name__,
                )
            if pending_transition is not None and provider_id in pending_transition["successor_ids"]:
                predecessor = pending_transition["provider"]
                try:
                    learned_guards.extend(observe_successful_transition(
                        operation,
                        predecessor,
                        pending_transition["before"],
                        pending_transition["after"],
                        visit_id=visit_id,
                        state_dir=state_dir,
                        vision=vision,
                    ))
                    record_provider_result(operation, str(predecessor.get("provider_id") or ""), "confirmed_by_successor", visit_id=visit_id)
                    if isinstance(pending_transition.get("candidate"), dict):
                        pending_transition["candidate"]["successes"] = int(pending_transition["candidate"].get("successes", 0) or 0) + 1
                except Exception as exc:  # noqa: BLE001
                    learned_guards.append({"backfill_error": f"{type(exc).__name__}: {exc}"})
                pending_transition = None
            event = provider.get("emits_on_success") if isinstance(provider.get("emits_on_success"), dict) else None
            if not event and not provider.get("successors") and command.expected_event:
                event = {"type": command.expected_event}
            reduce_intent_event(runtime, event)
        elif result == "unverified" and changed and provider.get("watches") and provider.get("successors"):
            pending_transition = {
                "provider": provider,
                "provider_id": provider_id,
                "successor_ids": [str(value) for value in provider.get("successors", [])],
                "before": current,
                "after": post,
            }

        if action_info.get("locator_type") == "click_region" and isinstance(action_info.get("candidate_point"), list):
            locator_index = int(action_info.get("locator_index", 0) or 0)
            locators = provider.get("locators", [])
            locator = locators[locator_index] if 0 <= locator_index < len(locators) else None
            candidate = next((item for item in (locator or {}).get("candidate_points", []) if item.get("point") == action_info["candidate_point"]), None)
            if candidate is not None:
                if pending_transition is not None and pending_transition.get("provider_id") == provider_id:
                    pending_transition["candidate"] = candidate
                if result in {"confirmed", "transitioned"}:
                    candidate["successes"] = int(candidate.get("successes", 0) or 0) + 1
                elif not changed:
                    candidate["no_effects"] = int(candidate.get("no_effects", 0) or 0) + 1
                    remaining = [item for item in locator.get("candidate_points", []) if item.get("point") not in cursor.get("locator_attempts", {}).get(provider_id, [])]
                    if remaining and len(cursor.get("locator_attempts", {}).get(provider_id, [])) < 2:
                        attempts = cursor.get("attempt_counts", {})
                        attempts[provider_id] = max(0, int(attempts.get(provider_id, 1) or 1) - 1)
        episode = {
            "state_id": state_id,
            "visit_id": visit_id,
            "operation": command.operation,
            "provider_id": provider_id,
            "result": result,
            "effect_details": effect_details,
            "changed": changed,
            "diff_score": diff_score,
            "still_current": still_current,
            "learned_guards": learned_guards,
            "text_fingerprint_before": before_text_fp,
            "text_fingerprint_after": after_text_fp,
            "text_progress": text_progress,
        }
        append_episode(handler, episode)
        history.append(_history_item(post, f"after_{len(history)}", kind="post_action", provider_id=provider_id, result=result, changed=changed, diff_score=diff_score))
        if len(history) > 7:
            del history[1 : len(history) - 6]
        _save_json(state_path, state_meta)
        _save_runtime(runtime)
        if logger is not None:
            logger.event("page_handler_step_result", action_id=action_id, candidates=[summarize_match(item) for item in matches], **episode)

        if known_other is not None:
            _clear_text_only_reentry(runtime)
            nxt = _resolve_transition_after_progress(state_id=state_id, action_id=action_id, matches=matches, graph=graph, runtime=runtime, prefer_reachable_first=prefer_reachable_first, logger=logger, reason_suffix="-reactive-v3.1")
            if nxt is None:
                _append_graph_edge(state_id, action_id, known_other.state_id, logger=logger, reason="reactive-v3.1-observed", confidence="strong")
                runtime["last_state_id"] = known_other.state_id
                runtime["last_transition_ok"] = True
            return True, post
        if not still_current:
            _clear_text_only_reentry(runtime)
            runtime["pending_from_state_id"] = state_id
            runtime["pending_action_id"] = action_id
            return _defer_unknown_transition(runtime=runtime, state_id=state_id, action_id=action_id, logger=logger, frame=post, reason="reactive-v3.1-unmatched-stable-frame")
        if result == "confirmed" and not provider.get("successors"):
            # A confirmed observable effect with the same matcher is a fresh
            # instance (e.g. repeated cards/dialogs), not a reason to reuse
            # exhausted once-per-visit providers.
            if text_progress or effect_progress:
                clear_visit(runtime, visit_id)
                new_visit = start_new_visit(runtime, state_id, reason="confirmed_self_reentry")
                runtime["last_state_id"] = state_id
                runtime["last_transition_ok"] = True
                _save_runtime(runtime)
                _log(logger, f"[fsm][handler] confirmed self-reentry old_visit={visit_id} new_visit={new_visit['visit_id']}", "page_handler_self_reentry", state_id=state_id, old_visit_id=visit_id, new_visit_id=new_visit["visit_id"], text_progress=text_progress, effect_progress=effect_progress)
            return True, post
        if result == "contradicted":
            failures.append({"provider_id": provider_id, "result": result, "effect_details": effect_details})
        current = post
