from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from state_machine.io import _load_frame, _now_iso, _save_json, _save_runtime
from state_machine.llm_tasks import _request_llm_disambiguation
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import MatchResult, _condition_eval
from state_machine.state_store import _meta_for_state, _sample_paths_from_meta


def _successful_matches(matches: list[MatchResult]) -> list[MatchResult]:
    return [m for m in matches if m.success]


def _strong_match(meta: dict[str, Any], match: MatchResult) -> bool:
    weak = bool(meta.get("model_info", {}).get("weak_match", False)) if isinstance(meta.get("model_info"), dict) else False
    return (not weak and match.total_enabled >= 1) or match.total_enabled >= 2


def _candidate_summary_for_llm(match: MatchResult, meta: dict[str, Any]) -> dict[str, Any]:
    matched_briefs: list[str] = []
    for detail in match.condition_results or []:
        if detail.get("passed"):
            brief = str(detail.get("brief", "")).strip()
            if brief:
                matched_briefs.append(brief)
    return {
        "state_id": match.state_id,
        "slug": meta.get("slug"),
        "page_type": meta.get("page_type"),
        "description": meta.get("description"),
        "matched_briefs": matched_briefs[:8],
        "match": summarize_match(match),
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
