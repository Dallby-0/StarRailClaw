from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from state_machine.constants import (
    ACTION_CLICK_WAIT_S,
    FORCE_RESET_AT,
    FSM_GRAPH_PATH,
    FSM_TEMPLATES_DIR,
    HARD_LIMIT,
    LOCAL_FLOW_MAX_STEPS,
    LOCAL_FLOW_NO_PROGRESS_LIMIT,
    LLM_PARSE_RETRY,
    MAX_RETRY_PER_ACTION,
    MID_LIMIT,
    REPAIR_EFFORTS,
    SCHEMA_VERSION,
    SCREEN_CHANGE_DIFF_THRESHOLD,
    SOFT_LIMIT,
    UNKNOWN_STABILITY_DIFF_THRESHOLD,
    UNKNOWN_STABILITY_RETRY_WAIT_S,
    UNKNOWN_STABILITY_SAMPLE_INTERVAL_S,
)
from state_machine.io import (
    _append_experience,
    _backup_json,
    _build_system_prompt_with_experience,
    _ensure_fsm_resources,
    _load_frame,
    _load_json,
    _load_runtime,
    _normalize_page_type,
    _now_iso,
    _save_frame,
    _save_json,
    _save_runtime,
    _slugify,
    _state_page_type,
)
from state_machine.llm_tasks import (
    _apply_reasoning_effort,
    _build_user_message_from_frame,
    _normalize_assistant_text,
    _parse_llm_repair,
    _request_llm_disambiguation,
    _request_llm_condition_revision,
    _request_llm_failure_diagnosis,
    _request_llm_payload,
    _save_llm_raw_debug,
)
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import (
    MatchResult,
    _condition_eval,
    _condition_passed,
    _find_match_by_state,
    _level,
    _select_best_for_unknown,
    _select_enabled_conditions,
    _eval_state_match,
)
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
from agent.llm_client import DoubaoClient
from state_machine.presets import run_preset
from sr_tools.adb import resolve_target_serial
from sr_tools.emulator import EmulatorClient


def _log(logger: FsmRunLogger | None, message: str, event: str = "console", **fields: Any) -> None:
    if logger is None:
        print(message)
        return
    logger.text(message, event=event, **fields)

def _iter_state_meta() -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    if not FSM_TEMPLATES_DIR.exists():
        return out
    for d in sorted(FSM_TEMPLATES_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "state.json"
        if not meta_path.exists():
            continue
        try:
            out.append((d, _load_json(meta_path)))
        except Exception:
            continue
    return out


def _page_type_summaries(metas: list[tuple[Path, dict[str, Any]]], limit: int = 40) -> list[dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for state_dir, meta in metas:
        ptype = _state_page_type(meta)
        entry = by_type.setdefault(
            ptype,
            {
                "page_type": ptype,
                "example_slug": str(meta.get("slug", "")),
                "description": str(meta.get("description", ""))[:160],
                "sample_count": 0,
                "latest_dir": str(state_dir),
            },
        )
        entry["sample_count"] = int(entry.get("sample_count", 0)) + 1
        try:
            if state_dir.stat().st_mtime >= Path(str(entry["latest_dir"])).stat().st_mtime:
                entry["example_slug"] = str(meta.get("slug", ""))
                entry["description"] = str(meta.get("description", ""))[:160]
                entry["latest_dir"] = str(state_dir)
        except Exception:
            pass
    return sorted(by_type.values(), key=lambda x: str(x.get("page_type", "")))[:limit]


def _states_for_page_type(metas: list[tuple[Path, dict[str, Any]]], page_type: str) -> list[tuple[Path, dict[str, Any]]]:
    norm = _normalize_page_type(page_type)
    if not norm:
        return []
    return [(d, m) for d, m in metas if _state_page_type(m) == norm]


def _latest_state_for_page_type(metas: list[tuple[Path, dict[str, Any]]], page_type: str) -> tuple[Path, dict[str, Any]] | None:
    states = _states_for_page_type(metas, page_type)
    if not states:
        return None
    return max(states, key=lambda item: item[0].stat().st_mtime)


def _sample_screenshot_paths(state_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for d in state_dirs:
        shots = sorted(d.glob("screenshot_*.png"))
        if shots:
            out.extend(shots)
    return out


def _latest_screenshot_path(state_dir: Path) -> Path | None:
    shots = sorted(state_dir.glob("screenshot_*.png"))
    return shots[-1] if shots else None


def _add_state_sample(
    state_dir: Path,
    meta: dict[str, Any],
    frame_rgb,
    *,
    role: str,
    source: str,
    confidence: float = 0.5,
    logger: FsmRunLogger | None = None,
) -> None:
    samples = meta.setdefault("samples", [])
    if not isinstance(samples, list):
        samples = []
        meta["samples"] = samples
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = state_dir / f"sample_{role}_{source}_{ts}.png"
    _save_frame(path, frame_rgb)
    samples.append(
        {
            "path": str(path),
            "role": role,
            "source": source,
            "confidence": max(0.0, min(1.0, float(confidence))),
            "created_at": _now_iso(),
        }
    )
    meta["updated_at"] = _now_iso()
    _save_json(state_dir / "state.json", meta)
    if logger is not None:
        logger.event("state_sample_added", state_id=meta.get("state_id"), role=role, source=source, confidence=confidence, path=path)


def _sample_paths_from_meta(state_dir: Path, meta: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    samples = meta.get("samples")
    if isinstance(samples, list):
        for s in samples:
            if not isinstance(s, dict):
                continue
            p = Path(str(s.get("path", "")))
            if p.exists():
                out.append(p)
    latest = _latest_screenshot_path(state_dir)
    if latest is not None:
        out.append(latest)
    return out


def _conditions_from_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, e in enumerate(elements):
        if not isinstance(e, dict):
            continue
        etype = e.get("type")
        bbox = e.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if etype == "text_line":
            text = str(e.get("text", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "id": f"elem_text_{i+1}",
                    "enabled": True,
                    "kind": "text_line_contains",
                    "params": {"text": text, "rect": bbox},
                    "weight": 1.0,
                    "brief": str(e.get("brief", "")),
                    "role": str(e.get("role", "")),
                    "stability": _level(e.get("stability"), "mid"),
                    "discrimination": _level(e.get("discrimination"), "mid"),
                    "condition_status": "active",
                    "bbox": bbox,
                }
            )
        elif etype == "pattern":
            out.append(
                {
                    "id": f"elem_pattern_{i+1}",
                    "enabled": True,
                    "kind": "region_template",
                    "params": {"rect": bbox, "threshold": 0.8},
                    "weight": 1.0,
                    "brief": str(e.get("brief", "")),
                    "role": str(e.get("role", "")),
                    "stability": _level(e.get("stability"), "mid"),
                    "discrimination": _level(e.get("discrimination"), "mid"),
                    "condition_status": "active",
                    "bbox": bbox,
                }
            )
    return out


def _extract_region_templates(frame_rgb, mapper: CoordinateMapper, state_dir: Path, conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    t_idx = 1
    for c in conditions:
        cc = dict(c)
        if cc.get("kind") == "region_template":
            rect = cc.get("params", {}).get("rect")
            if isinstance(rect, list) and len(rect) == 4:
                x, y, w, h = mapper.rect_to_real(rect)
                crop = frame_rgb[y : y + h, x : x + w]
                tpath = state_dir / f"template_{t_idx}.png"
                t_idx += 1
                cv2.imwrite(str(tpath), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                cc["params"]["template_path"] = str(tpath)
                cc["source"] = {
                    "screenshot_path": str(state_dir / "screenshot_1.png"),
                    "bbox_logical": rect,
                    "template_path": str(tpath),
                }
        out.append(cc)
    return out


def _limit_enable_conditions(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enabled_count = 0
    for c in conditions:
        valid = c.get("kind") in {"text_line_contains", "region_template"}
        if valid and enabled_count < 2:
            c["enabled"] = True
            enabled_count += 1
        else:
            c["enabled"] = False
    return conditions


def _ensure_unique_state_dir(slug: str) -> Path:
    base = _slugify(slug)
    d = FSM_TEMPLATES_DIR / base
    n = 1
    while d.exists():
        n += 1
        d = FSM_TEMPLATES_DIR / f"{base}_{n}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _normalize_actions(raw_actions: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for a in raw_actions:
        if not isinstance(a, dict):
            continue
        atype = str(a.get("type", "click"))
        brief = str(a.get("brief", ""))
        if atype == "run_preset":
            out.append({"type": "run_preset", "name": str(a.get("name", "")), "brief": brief})
            continue
        x = int(a.get("x", 500))
        y = int(a.get("y", 500))
        out.append({"type": "click", "x": x, "y": y, "brief": brief})
    return out


def _create_state_from_llm(
    llm_payload: dict[str, Any],
    frame_rgb,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    logger: FsmRunLogger | None = None,
) -> tuple[str, Path]:
    state_id = uuid.uuid4().hex
    state_dir = _ensure_unique_state_dir(str(llm_payload.get("slug", "state")))
    _save_frame(state_dir / "screenshot_1.png", frame_rgb)
    conds = _conditions_from_elements(llm_payload.get("elements", []))
    conds = _extract_region_templates(frame_rgb, mapper, state_dir, conds)
    conds, weak_match = _select_enabled_conditions(conds, vision, frame_rgb)

    state_meta = {
        "schema_version": SCHEMA_VERSION,
        "state_id": state_id,
        "slug": _slugify(str(llm_payload.get("slug", "state"))),
        "page_type": _normalize_page_type(llm_payload.get("possible_page_type") or llm_payload.get("page_type")),
        "display_name": str(llm_payload.get("slug", "state")),
        "description": str(llm_payload.get("page_summary", "")),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "model_info": {"source": "llm", "weak_match": weak_match},
        "elements": llm_payload.get("elements", []),
        "samples": [
            {
                "path": str(state_dir / "screenshot_1.png"),
                "role": "prototype",
                "source": "created",
                "confidence": 1.0,
                "created_at": _now_iso(),
            }
        ],
        "match_conditions": conds,
        "actions": [
            {
                "action_id": "action_main",
                "version": 1,
                "enabled": True,
                "cooldown_ms": 0,
                "steps": _normalize_actions(llm_payload.get("actions", [])),
            }
        ],
    }
    _save_json(state_dir / "state.json", state_meta)
    if logger is not None:
        logger.event(
            "state_created",
            state_id=state_id,
            state_dir=state_dir,
            slug=state_meta["slug"],
            page_type=state_meta["page_type"],
            weak_match=weak_match,
            elements_count=len(llm_payload.get("elements", [])),
            actions_count=len(llm_payload.get("actions", [])),
            conditions=conds,
            screenshot_path=state_dir / "screenshot_1.png",
        )
    return state_id, state_dir


def _append_graph_node(state_id: str, slug: str, logger: FsmRunLogger | None = None) -> None:
    graph = _load_json(FSM_GRAPH_PATH)
    graph["nodes"].append({"state_id": state_id, "slug": slug, "enabled": True})
    graph["updated_at"] = _now_iso()
    _save_json(FSM_GRAPH_PATH, graph)
    if logger is not None:
        logger.event("graph_node_added", state_id=state_id, slug=slug)


def _append_graph_edge(
    from_state_id: str | None,
    action_id: str,
    to_state_id: str,
    logger: FsmRunLogger | None = None,
    reason: str = "",
    confidence: str = "strong",
) -> None:
    if not from_state_id:
        return
    graph = _load_json(FSM_GRAPH_PATH)
    for e in graph.get("edges", []):
        if e.get("from_state_id") == from_state_id and e.get("action_id") == action_id and e.get("to_state_id") == to_state_id:
            if confidence in {"strong", "llm_verified"} and e.get("confidence") == "tentative":
                e["confidence"] = confidence
                e["enabled"] = True
                e["reason"] = reason
                e["updated_at"] = _now_iso()
                graph["updated_at"] = _now_iso()
                _save_json(FSM_GRAPH_PATH, graph)
            return
    graph["edges"].append(
        {
            "from_state_id": from_state_id,
            "action_id": action_id,
            "to_state_id": to_state_id,
            "enabled": confidence != "tentative",
            "weight": 1.0,
            "confidence": confidence,
            "reason": reason,
            "created_at": _now_iso(),
        }
    )
    graph["updated_at"] = _now_iso()
    _save_json(FSM_GRAPH_PATH, graph)
    if logger is not None:
        logger.event(
            "graph_edge_added",
            from_state_id=from_state_id,
            action_id=action_id,
            to_state_id=to_state_id,
            reason=reason,
            confidence=confidence,
        )


def _get_reachable_targets(graph: dict[str, Any], from_state_id: str | None) -> set[str]:
    if not from_state_id:
        return set()
    return {
        str(e.get("to_state_id"))
        for e in graph.get("edges", [])
        if e.get("enabled", True) and str(e.get("from_state_id")) == from_state_id
    }


def _successful_matches(matches: list[MatchResult]) -> list[MatchResult]:
    return [m for m in matches if m.success]


def _strong_match(meta: dict[str, Any], match: MatchResult) -> bool:
    weak = bool(meta.get("model_info", {}).get("weak_match", False)) if isinstance(meta.get("model_info"), dict) else False
    return (not weak and match.total_enabled >= 1) or match.total_enabled >= 2


def _meta_for_state(metas: list[tuple[Path, dict[str, Any]]], state_id: str) -> tuple[Path, dict[str, Any]] | None:
    for d, m in metas:
        if str(m.get("state_id", "")) == state_id:
            return d, m
    return None


def _candidate_summary_for_llm(match: MatchResult, meta: dict[str, Any]) -> dict[str, Any]:
    matched_briefs: list[str] = []
    for detail in match.condition_results or []:
        if detail.get("enabled") and detail.get("passed"):
            matched_briefs.append(str(detail.get("brief", "")))
    return {
        "state_id": match.state_id,
        "slug": meta.get("slug"),
        "page_type": meta.get("page_type"),
        "description": str(meta.get("description", ""))[:180],
        "enabled_match": f"{match.passed_enabled}/{match.total_enabled}",
        "weak_match": bool(meta.get("model_info", {}).get("weak_match", False)) if isinstance(meta.get("model_info"), dict) else False,
        "matched_enabled_briefs": matched_briefs,
    }


def _try_strengthen_winner_conditions(
    *,
    winner: MatchResult,
    losers: list[MatchResult],
    metas: list[tuple[Path, dict[str, Any]]],
    vision: VisionEngine,
    frame_rgb,
    logger: FsmRunLogger | None = None,
) -> bool:
    found = _meta_for_state(metas, winner.state_id)
    if found is None:
        return False
    winner_dir, winner_meta = found
    conds = [c for c in winner_meta.get("match_conditions", []) if isinstance(c, dict)]
    current_enabled = [c for c in conds if c.get("enabled", False)]
    disabled = [c for c in conds if not c.get("enabled", False) and c.get("condition_status", "active") == "active"]
    passing_disabled: list[dict[str, Any]] = []
    for cond in disabled:
        ok, _ = _condition_eval(cond, vision, frame_rgb)
        if ok:
            passing_disabled.append(cond)
    if not passing_disabled:
        return False

    loser_samples: dict[str, list[Any]] = {}
    for loser in losers:
        item = _meta_for_state(metas, loser.state_id)
        if item is None:
            continue
        loser_dir, loser_meta = item
        frames: list[Any] = []
        for p in _sample_paths_from_meta(loser_dir, loser_meta)[:3]:
            img = _load_frame(p)
            if img is not None:
                frames.append(img)
        loser_samples[loser.state_id] = frames

    selected: list[dict[str, Any]] = []
    remaining = {l.state_id for l in losers}
    while remaining:
        best_cond = None
        best_excluded: set[str] = set()
        for cond in passing_disabled:
            if cond in selected:
                continue
            excluded: set[str] = set()
            for loser_id in remaining:
                frames = loser_samples.get(loser_id) or []
                if frames and not any(_condition_eval(cond, vision, img)[0] for img in frames):
                    excluded.add(loser_id)
            if len(excluded) > len(best_excluded):
                best_cond = cond
                best_excluded = excluded
        if best_cond is None or not best_excluded:
            break
        selected.append(best_cond)
        remaining -= best_excluded
    if not selected:
        return False

    for c in conds:
        c["enabled"] = c in current_enabled or c in selected
    winner_meta.setdefault("model_info", {})
    if isinstance(winner_meta["model_info"], dict):
        winner_meta["model_info"]["weak_match"] = False
        winner_meta["model_info"]["last_enabled_reason"] = {
            "source": "runtime_disambiguation",
            "winner_against": [l.state_id for l in losers],
            "added_condition_ids": [str(c.get("id", "")) for c in selected],
            "created_at": _now_iso(),
        }
    winner_meta["updated_at"] = _now_iso()
    _save_json(winner_dir / "state.json", winner_meta)
    if logger is not None:
        logger.event("disambiguation_strengthened_winner", state_id=winner.state_id, added_condition_ids=[c.get("id") for c in selected], remaining_losers=sorted(remaining))
    return True


def _disambiguate_matches(
    *,
    llm: DoubaoClient,
    llm_session_id: str,
    frame_rgb,
    system_prompt: str,
    matches: list[MatchResult],
    metas: list[tuple[Path, dict[str, Any]]],
    vision: VisionEngine,
    runtime: dict[str, Any],
    logger: FsmRunLogger | None = None,
) -> MatchResult | None:
    successes = _successful_matches(matches)
    if len(successes) <= 1:
        return successes[0] if successes else None
    summaries: list[dict[str, Any]] = []
    by_id = {m.state_id: m for m in successes}
    for m in successes:
        item = _meta_for_state(metas, m.state_id)
        if item is None:
            continue
        summaries.append(_candidate_summary_for_llm(m, item[1]))
    if not summaries:
        return None
    result = _request_llm_disambiguation(
        llm,
        llm_session_id,
        frame_rgb,
        system_prompt,
        candidates=summaries,
        raw_debug_dir=logger.llm_raw_dir if logger is not None else None,
    )
    runtime["llm_turn_count"] = int(runtime.get("llm_turn_count", 0)) + 1
    _save_runtime(runtime)
    if logger is not None:
        logger.event("disambiguation_result", result=result, candidates=summaries)
    if not result:
        return None
    winner_id = str(result.get("winner_state_id", "")).strip()
    if winner_id == "none" or winner_id not in by_id:
        return None
    winner = by_id[winner_id]
    losers = [m for m in successes if m.state_id != winner_id]
    _try_strengthen_winner_conditions(winner=winner, losers=losers, metas=metas, vision=vision, frame_rgb=frame_rgb, logger=logger)
    return winner


def _pick_transition_candidate(matches: list[MatchResult], state_id: str, reachable: set[str]) -> tuple[MatchResult | None, str]:
    reachable_hits = [m for m in matches if m.success and m.state_id in reachable and m.state_id != state_id]
    if len(reachable_hits) == 1:
        return reachable_hits[0], "strong"
    if len(reachable_hits) > 1:
        return None, "ambiguous_reachable"
    fallback_hits = [m for m in matches if m.success and m.state_id != state_id]
    if len(fallback_hits) == 1:
        return fallback_hits[0], "tentative"
    if len(fallback_hits) > 1:
        return None, "ambiguous_fallback"
    return None, "none"


def _condition_from_revision(raw: dict[str, Any], base: dict[str, Any], state_dir: Path, frame_rgb, mapper: CoordinateMapper) -> dict[str, Any] | None:
    decision = str(raw.get("decision", "keep")).strip().lower()
    if decision == "deprecate":
        cc = dict(base)
        cc["condition_status"] = "deprecated"
        cc["enabled"] = False
        return cc
    if decision not in {"keep", "revise"}:
        return None
    cc = dict(base)
    cc["condition_status"] = "active"
    cc["enabled"] = False
    cc["stability"] = _level(raw.get("stability", cc.get("stability")), "mid")
    cc["discrimination"] = _level(raw.get("discrimination", cc.get("discrimination")), "mid")
    cc["brief"] = str(raw.get("brief", cc.get("brief", "")))
    if decision == "revise":
        kind = str(raw.get("kind", cc.get("kind", ""))).strip()
        if kind == "line_contains_text":
            kind = "text_line_contains"
        params = raw.get("params", {})
        if not isinstance(params, dict):
            params = {}
        if kind == "text_line_contains":
            text = str(params.get("text") or params.get("contains") or raw.get("text") or "").strip()
            rect = params.get("rect") or raw.get("bbox") or cc.get("params", {}).get("rect") or cc.get("bbox")
            if not text or not (isinstance(rect, list) and len(rect) == 4):
                return None
            cc["kind"] = kind
            cc["params"] = {"text": text, "rect": rect}
            cc["bbox"] = rect
        elif kind == "region_template":
            rect = params.get("rect") or raw.get("bbox") or cc.get("params", {}).get("rect") or cc.get("bbox")
            if not (isinstance(rect, list) and len(rect) == 4):
                return None
            x, y, w, h = mapper.rect_to_real(rect)
            crop = frame_rgb[y : y + h, x : x + w]
            tpath = state_dir / f"template_revised_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{cc.get('id', 'cond')}.png"
            cv2.imwrite(str(tpath), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            cc["kind"] = kind
            cc["params"] = {
                "rect": rect,
                "threshold": float(params.get("threshold", cc.get("params", {}).get("threshold", 0.8))),
                "template_path": str(tpath),
            }
            cc["bbox"] = rect
            cc["source"] = {
                "screenshot_path": str(state_dir / "screenshot_latest_merge.png"),
                "bbox_logical": rect,
                "template_path": str(tpath),
            }
        else:
            return None
    return cc


def _conditions_pass_all(conditions: list[dict[str, Any]], vision: VisionEngine, frames: list[Any]) -> bool:
    active = [c for c in conditions if c.get("condition_status", "active") == "active"]
    if not active:
        return False
    for frame in frames:
        for cond in active:
            if not _condition_passed(cond, vision, frame):
                return False
    return True


def _selected_conditions_match_other_page(
    conditions: list[dict[str, Any]],
    vision: VisionEngine,
    metas: list[tuple[Path, dict[str, Any]]],
    page_type: str,
) -> str | None:
    enabled = [c for c in conditions if c.get("enabled", False)]
    if not enabled:
        return None
    for state_dir, meta in metas:
        if _state_page_type(meta) == page_type:
            continue
        shot = _latest_screenshot_path(state_dir)
        if shot is None:
            continue
        frame = _load_frame(shot)
        if frame is None:
            continue
        if all(_condition_passed(c, vision, frame) for c in enabled):
            return f"{_state_page_type(meta)}:{meta.get('slug', '')}"
    return None


def _try_merge_page_type(
    *,
    llm: DoubaoClient,
    session_id: str,
    system_prompt: str,
    frame_rgb,
    llm_payload: dict[str, Any],
    mapper: CoordinateMapper,
    vision: VisionEngine,
    metas: list[tuple[Path, dict[str, Any]]],
    logger: FsmRunLogger | None = None,
) -> tuple[str, Path] | None:
    page_type = _normalize_page_type(llm_payload.get("possible_page_type"))
    if not page_type:
        return None
    latest = _latest_state_for_page_type(metas, page_type)
    if latest is None:
        return None
    latest_dir, latest_meta = latest
    sample_path = _latest_screenshot_path(latest_dir)
    if sample_path is None:
        return None
    latest_rgb = _load_frame(sample_path)
    if latest_rgb is None:
        return None

    existing_conds = [c for c in latest_meta.get("match_conditions", []) if isinstance(c, dict)]
    failed = [c for c in existing_conds if c.get("condition_status", "active") != "deprecated" and not _condition_passed(c, vision, frame_rgb)]
    rev = _request_llm_condition_revision(
        llm,
        session_id,
        system_prompt,
        latest_sample_rgb=latest_rgb,
        current_rgb=frame_rgb,
        page_type=page_type,
        latest_meta=latest_meta,
        failed_conditions=failed,
        new_payload=llm_payload,
        raw_debug_dir=logger.llm_raw_dir if logger is not None else None,
    )
    if rev is None or not rev.get("same_page_type", False):
        _log(logger, f"[fsm][merge] reject page_type={page_type} reason=llm_not_same_or_invalid", "merge_rejected", page_type=page_type, reason="llm_not_same_or_invalid")
        return None

    by_id = {str(c.get("id", "")): c for c in existing_conds}
    revised: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in rev.get("conditions", []):
        if not isinstance(item, dict):
            continue
        cid = str(item.get("condition_id") or item.get("id") or "").strip()
        base = by_id.get(cid)
        if base is None:
            continue
        new_cond = _condition_from_revision(item, base, latest_dir, frame_rgb, mapper)
        if new_cond is not None:
            revised.append(new_cond)
            seen_ids.add(cid)
    for cond in existing_conds:
        if str(cond.get("id", "")) not in seen_ids:
            cc = dict(cond)
            if not cc.get("enabled", False):
                cc["condition_status"] = "deprecated"
            revised.append(cc)

    sample_frames: list[Any] = []
    for shot in _sample_screenshot_paths([d for d, _ in _states_for_page_type(metas, page_type)]):
        img = _load_frame(shot)
        if img is not None:
            sample_frames.append(img)
    sample_frames.append(frame_rgb)
    if not _conditions_pass_all(revised, vision, sample_frames):
        _log(logger, f"[fsm][merge] reject page_type={page_type} reason=revised_conditions_do_not_cover_positive_samples", "merge_rejected", page_type=page_type, reason="revised_conditions_do_not_cover_positive_samples")
        return None

    revised, weak_match = _select_enabled_conditions(revised, vision, frame_rgb)
    false_positive = _selected_conditions_match_other_page(revised, vision, metas, page_type)
    if false_positive:
        _log(logger, f"[fsm][merge] reject page_type={page_type} reason=false_positive other={false_positive}", "merge_rejected", page_type=page_type, reason="false_positive", other=false_positive)
        return None

    _backup_json(latest_dir / "state.json")
    next_idx = len(list(latest_dir.glob("screenshot_*.png"))) + 1
    _save_frame(latest_dir / f"screenshot_{next_idx}.png", frame_rgb)
    samples = latest_meta.setdefault("samples", [])
    if isinstance(samples, list):
        samples.append(
            {
                "path": str(latest_dir / f"screenshot_{next_idx}.png"),
                "role": "positive",
                "source": "merge",
                "confidence": 0.8,
                "created_at": _now_iso(),
            }
        )
    latest_meta["match_conditions"] = revised
    latest_meta["updated_at"] = _now_iso()
    latest_meta["page_type"] = page_type
    latest_meta.setdefault("model_info", {})
    if isinstance(latest_meta["model_info"], dict):
        latest_meta["model_info"]["weak_match"] = weak_match
        latest_meta["model_info"]["last_merge_note"] = str(rev.get("note", ""))
    _save_json(latest_dir / "state.json", latest_meta)
    _log(
        logger,
        f"[fsm][merge] accepted page_type={page_type} state_id={latest_meta.get('state_id')} weak={weak_match}",
        "merge_accepted",
        page_type=page_type,
        state_id=latest_meta.get("state_id"),
        weak_match=weak_match,
        conditions=revised,
        screenshot_path=latest_dir / f"screenshot_{next_idx}.png",
    )
    return str(latest_meta.get("state_id", "")), latest_dir


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


def _screen_changed(before_rgb, after_rgb, threshold: float = SCREEN_CHANGE_DIFF_THRESHOLD) -> tuple[bool, float]:
    if before_rgb is None or after_rgb is None:
        return False, 0.0
    if before_rgb.shape != after_rgb.shape:
        return True, 1.0
    diff = cv2.absdiff(before_rgb, after_rgb)
    score = float(diff.mean()) / 255.0
    return score >= threshold, score


def _wait_for_unknown_screen_stable(
    emulator: EmulatorClient,
    first_frame,
    *,
    threshold: float = UNKNOWN_STABILITY_DIFF_THRESHOLD,
    interval_s: float = UNKNOWN_STABILITY_SAMPLE_INTERVAL_S,
    logger: FsmRunLogger | None = None,
):
    time.sleep(interval_s)
    second = emulator.screenshot(prefer_png=True)
    if second.shape[1] != 1280 or second.shape[0] != 720:
        second = cv2.resize(second, (1280, 720), interpolation=cv2.INTER_LINEAR)
    time.sleep(interval_s)
    third = emulator.screenshot(prefer_png=True)
    if third.shape[1] != 1280 or third.shape[0] != 720:
        third = cv2.resize(third, (1280, 720), interpolation=cv2.INTER_LINEAR)

    changed_12, diff_12 = _screen_changed(first_frame, second, threshold)
    changed_23, diff_23 = _screen_changed(second, third, threshold)
    changed_13, diff_13 = _screen_changed(first_frame, third, threshold)
    stable = not (changed_12 or changed_23 or changed_13)
    _log(
        logger,
        (
            "[fsm][unknown][stability] "
            f"stable={stable} diff12={diff_12:.4f} diff23={diff_23:.4f} diff13={diff_13:.4f} threshold={threshold:.4f}"
        ),
        "unknown_stability_check",
        stable=stable,
        diff12=diff_12,
        diff23=diff_23,
        diff13=diff_13,
        threshold=threshold,
        interval_s=interval_s,
    )
    return third if stable else None


def _resolve_transition_after_progress(
    *,
    state_id: str,
    action_id: str,
    matches: list[MatchResult],
    graph: dict[str, Any],
    runtime: dict[str, Any],
    prefer_reachable_first: bool,
    logger: FsmRunLogger | None,
    reason_suffix: str,
) -> MatchResult | None:
    reachable = _get_reachable_targets(graph, state_id)
    nxt, confidence = _pick_transition_candidate(matches, state_id, reachable if prefer_reachable_first else set())
    if nxt is not None:
        reason = f"{'reachable' if confidence == 'strong' else 'unknown-fallback'}{reason_suffix}"
        _log(logger, f"[fsm][transition][{reason}] from={state_id} action={action_id} to={nxt.state_id} confidence={confidence}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason=reason, confidence=confidence)
        if confidence != "strong":
            _append_graph_edge(state_id, action_id, nxt.state_id, logger=logger, reason=reason, confidence=confidence)
    if nxt is not None and nxt.state_id != state_id:
        runtime["last_state_id"] = nxt.state_id
        runtime["last_transition_ok"] = True
        runtime["pending_from_state_id"] = None
        runtime["pending_action_id"] = None
        _save_runtime(runtime)
        return nxt
    return None


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
            runtime["last_state_id"] = None
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = state_id
            runtime["pending_action_id"] = action_id
            _save_runtime(runtime)
            _log(logger, f"[fsm][local][transition][to-unknown] from={state_id} action={action_id}", "transition", from_state=state_id, action_id=action_id, to_state=None, reason="page-local-original-state-no-longer-matched")
            return True, post
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


def _maybe_rotate_session(runtime: dict[str, Any]) -> bool:
    turns = int(runtime.get("llm_turn_count", 0))
    tier = str(runtime.get("session_tier", "soft"))
    if turns >= FORCE_RESET_AT:
        runtime["pending_refresh"] = True
        return True
    if tier == "soft" and turns >= SOFT_LIMIT and runtime.get("last_transition_ok", False):
        runtime["session_tier"] = "mid"
        runtime["pending_refresh"] = True
        return True
    if tier == "mid" and turns >= MID_LIMIT and runtime.get("last_transition_ok", False):
        runtime["session_tier"] = "hard"
        runtime["pending_refresh"] = True
        return True
    if tier == "hard" and turns >= HARD_LIMIT:
        runtime["pending_refresh"] = True
        return True
    return False


def _refresh_session_if_needed(runtime: dict[str, Any], base_session_id: str) -> str:
    if not runtime.get("llm_session_id") or runtime.get("pending_refresh", False):
        runtime["llm_session_id"] = f"{base_session_id}-fsm-{uuid.uuid4().hex[:8]}"
        runtime["llm_turn_count"] = 0
        runtime["pending_refresh"] = False
        runtime["last_transition_ok"] = False
    return str(runtime["llm_session_id"])


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
    # normal attempts
    for attempt in range(1, MAX_RETRY_PER_ACTION + 1):
        _log(logger, f"[fsm][action] state={state_id} action={action_id} attempt={attempt}", "action_attempt", state_id=state_id, action_id=action_id, attempt=attempt, steps=steps)
        if not _execute_action_steps(emulator, mapper, vision, state_dir, steps, state_id, matches_provider, logger=logger, action_id=action_id, attempt=attempt):
            return False, frame_before
        post = emulator.screenshot(prefer_png=True)
        matches = matches_provider(post)
        last_failure_frame = post
        last_failure_matches = matches
        reachable = _get_reachable_targets(graph, state_id)
        if logger is not None:
            logger.event(
                "post_action_match",
                state_id=state_id,
                action_id=action_id,
                attempt=attempt,
                reachable=sorted(reachable),
                candidates=[summarize_match(m) for m in matches],
            )
        nxt, confidence = _pick_transition_candidate(matches, state_id, reachable if prefer_reachable_first else set())
        if nxt is not None:
            reason = "reachable" if confidence == "strong" else "unknown-fallback"
            _log(logger, f"[fsm][transition][{reason}] from={state_id} action={action_id} to={nxt.state_id} confidence={confidence}", "transition", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason=reason, confidence=confidence)
            if confidence != "strong":
                _append_graph_edge(state_id, action_id, nxt.state_id, logger=logger, reason=reason, confidence=confidence)
                _log(logger, f"[fsm][edge][added] from={state_id} action={action_id} to={nxt.state_id} reason={reason} confidence={confidence}", "edge_added", from_state=state_id, action_id=action_id, to_state=nxt.state_id, reason=reason, confidence=confidence)
        elif reachable:
            _log(logger, f"[fsm][transition][reachable-miss] from={state_id} action={action_id} reachable={sorted(reachable)} result={confidence}", "transition_reachable_miss", from_state=state_id, action_id=action_id, reachable=sorted(reachable), result=confidence)
        if nxt is not None and nxt.state_id != state_id:
            runtime["last_state_id"] = nxt.state_id
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
            _save_runtime(runtime)
            return True, post
        curr_match = _find_match_by_state(matches, state_id)
        if curr_match is None or not curr_match.success:
            # Leave original state but no known target: treat as unknown transition, skip repair.
            runtime["last_state_id"] = None
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = state_id
            runtime["pending_action_id"] = action_id
            _save_runtime(runtime)
            _log(logger, f"[fsm][transition][to-unknown] from={state_id} action={action_id} reason=original-state-no-longer-matched", "transition", from_state=state_id, action_id=action_id, to_state=None, reason="original-state-no-longer-matched")
            return True, post
        changed, diff_score = _screen_changed(last_attempt_frame, post)
        if changed:
            _log(
                logger,
                f"[fsm][local] same state but screen changed diff={diff_score:.4f}; enter page-local flow",
                "page_local_enter",
                state_id=state_id,
                action_id=action_id,
                attempt=attempt,
                diff_score=diff_score,
            )
            local_ok, local_frame = _run_page_local_flow(
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
                    "source": "entry_action",
                    "action": steps[0],
                    "brief": f"entry action attempt={attempt}",
                    "runtime_observation": {"same_state_still_matches": True, "screen_changed_hint": True, "diff_score": diff_score},
                } if len(steps) == 1 and isinstance(steps[0], dict) else None,
                logger=logger,
            )
            if local_ok:
                return True, local_frame
            break
        last_attempt_frame = post

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
        _save_runtime(runtime)
        _log(logger, f"[fsm][diagnosis] state_misidentified; defer to unknown resolution", "state_misidentified", state_id=state_id, action_id=action_id, diagnosis=diagnosis)
        return True, last_failure_frame

    # repair loop
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
        if logger is not None:
            logger.event(
                "post_repair_match",
                state_id=state_id,
                action_id=action_id,
                effort=effort,
                reachable=sorted(reachable2),
                candidates=[summarize_match(m) for m in matches2],
            )
        nxt2, confidence2 = _pick_transition_candidate(matches2, state_id, reachable2 if prefer_reachable_first else set())
        if nxt2 is not None:
            reason2 = "reachable-after-repair" if confidence2 == "strong" else "unknown-fallback-after-repair"
            _log(logger, f"[fsm][transition][{reason2}] from={state_id} action={action_id} to={nxt2.state_id} effort={effort} confidence={confidence2}", "transition", from_state=state_id, action_id=action_id, to_state=nxt2.state_id, reason=reason2, effort=effort, confidence=confidence2)
            if confidence2 != "strong":
                _append_graph_edge(state_id, action_id, nxt2.state_id, logger=logger, reason=reason2, confidence=confidence2)
                _log(logger, f"[fsm][edge][added] from={state_id} action={action_id} to={nxt2.state_id} reason={reason2} confidence={confidence2}", "edge_added", from_state=state_id, action_id=action_id, to_state=nxt2.state_id, reason=reason2, confidence=confidence2)
        elif reachable2:
            _log(logger, f"[fsm][transition][reachable-miss] from={state_id} action={action_id} after_repair={effort} result={confidence2}", "transition_reachable_miss", from_state=state_id, action_id=action_id, reachable=sorted(reachable2), effort=effort, result=confidence2)
        if nxt2 is not None and nxt2.state_id != state_id:
            runtime["last_state_id"] = nxt2.state_id
            runtime["repair_fail_count"] = 0
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
            _save_runtime(runtime)
            _append_experience(f"repair success: {state_meta.get('slug', state_id)} -> {nxt2.state_id}; mode={repair.get('mode')} effort={effort}")
            return True, post2
        curr_match2 = _find_match_by_state(matches2, state_id)
        if curr_match2 is None or not curr_match2.success:
            runtime["last_state_id"] = None
            runtime["repair_fail_count"] = 0
            runtime["last_transition_ok"] = True
            runtime["pending_from_state_id"] = state_id
            runtime["pending_action_id"] = action_id
            _save_runtime(runtime)
            _append_experience(f"repair success(to-unknown): {state_meta.get('slug', state_id)}; mode={repair.get('mode')} effort={effort}")
            _log(logger, f"[fsm][transition][to-unknown] from={state_id} action={action_id} after_repair={effort}", "transition", from_state=state_id, action_id=action_id, to_state=None, reason="to-unknown-after-repair", effort=effort)
            return True, post2
        changed2, diff_score2 = _screen_changed(post, post2)
        if changed2:
            _log(
                logger,
                f"[fsm][local] repair kept same state but screen changed diff={diff_score2:.4f}; enter page-local flow",
                "page_local_enter_after_repair",
                state_id=state_id,
                action_id=action_id,
                effort=effort,
                diff_score=diff_score2,
            )
            local_ok2, local_frame2 = _run_page_local_flow(
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
                    "source": f"repair_entry_action:{effort}",
                    "action": steps[0],
                    "brief": f"repair entry action effort={effort}",
                    "runtime_observation": {"same_state_still_matches": True, "screen_changed_hint": True, "diff_score": diff_score2},
                } if len(steps) == 1 and isinstance(steps[0], dict) else None,
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


def run_agent_loop_fsm(*, session_id: str, serial: str | None = None, adb_path: str | None = None, interval_s: float = 5.0) -> None:
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
    _save_runtime(runtime)
    logger = FsmRunLogger(session_id, str(runtime["run_id"]))
    logger.event(
        "run_start",
        session_id=session_id,
        serial=target_serial,
        adb_path=adb_path,
        interval_s=interval_s,
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
            matches = [_eval_state_match(meta, sdir, vision, img) for sdir, meta in metas]
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

        successes = _successful_matches(matches)
        if len(successes) > 1:
            logger.text(
                f"[fsm][disambiguation] candidates={[m.state_id for m in successes]}",
                "disambiguation_started",
                candidates=[summarize_match(m) for m in successes],
            )
            best = _disambiguate_matches(
                llm=llm,
                llm_session_id=llm_session_id,
                frame_rgb=frame,
                system_prompt=system_prompt,
                matches=matches,
                metas=metas,
                vision=vision,
                runtime=runtime,
                logger=logger,
            )
        else:
            best = _select_best_for_unknown(matches)
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
            payload = _request_llm_payload(llm, llm_session_id, frame, system_prompt, _page_type_summaries(metas), raw_debug_dir=logger.llm_raw_dir)
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
                actions_count=len(payload.get("actions", [])),
            )
            merged = _try_merge_page_type(
                llm=llm,
                session_id=llm_session_id,
                system_prompt=system_prompt,
                frame_rgb=frame,
                llm_payload=payload,
                mapper=mapper,
                vision=vision,
                metas=metas,
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
                _append_graph_edge(pending_from, pending_action, new_state_id, logger=logger, reason="pending-unknown-resolution", confidence="llm_verified")
                logger.text(
                    f"[fsm][edge][added] from={pending_from} action={pending_action} "
                    f"to={new_state_id} reason=pending-unknown-resolution",
                    "edge_added",
                    from_state=pending_from,
                    action_id=pending_action,
                    to_state=new_state_id,
                    reason="pending-unknown-resolution",
                )
                runtime["pending_from_state_id"] = None
                runtime["pending_action_id"] = None
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
            _append_graph_edge(pending_from2, pending_action2, best.state_id, logger=logger, reason="pending-unknown-resolved-to-existing", confidence="llm_verified")
            best_meta_item = _meta_for_state(metas, best.state_id)
            if best_meta_item is not None:
                _add_state_sample(best_meta_item[0], best_meta_item[1], frame, role="positive", source="pending_existing", confidence=0.8, logger=logger)
            logger.text(
                f"[fsm][edge][added] from={pending_from2} action={pending_action2} "
                f"to={best.state_id} reason=pending-unknown-resolved-to-existing",
                "edge_added",
                from_state=pending_from2,
                action_id=pending_action2,
                to_state=best.state_id,
                reason="pending-unknown-resolved-to-existing",
            )
            runtime["pending_from_state_id"] = None
            runtime["pending_action_id"] = None
        _save_runtime(runtime)

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
    args = parser.parse_args()
    run_agent_loop_fsm(session_id=args.session_id, serial=args.serial, adb_path=args.adb_path, interval_s=args.interval)


if __name__ == "__main__":
    main()
