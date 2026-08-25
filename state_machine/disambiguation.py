from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from state_machine.io import _load_frame, _now_iso, _save_json, _save_runtime
from state_machine.llm_tasks import _request_llm_disambiguation
from state_machine.logger import FsmRunLogger, summarize_match
from state_machine.matching import MatchResult, _condition_eval, _eval_state_match
from state_machine.merge import _try_merge_ambiguous_states
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


def _condition_rank(cond: dict[str, Any]) -> tuple[int, int]:
    score = {"low": 1, "mid": 2, "high": 3}
    stability = score.get(str(cond.get("stability", "mid")).lower(), 2)
    discrimination = score.get(str(cond.get("discrimination", "mid")).lower(), 2)
    return discrimination, stability


def _sample_frames(state_dir: Path, meta: dict[str, Any], *, limit: int = 3) -> list[Any]:
    frames: list[Any] = []
    for p in _sample_paths_from_meta(state_dir, meta)[:limit]:
        img = _load_frame(p)
        if img is not None:
            frames.append(img)
    return frames


def _try_exclude_current_from_losers(
    *,
    winner: MatchResult,
    losers: list[MatchResult],
    metas: list[tuple[Path, dict[str, Any]]],
    vision: VisionEngine,
    frame_rgb,
    logger: FsmRunLogger | None = None,
) -> dict[str, str]:
    strengthened: dict[str, str] = {}
    failures: dict[str, str] = {}
    for loser in losers:
        item = _meta_for_state(metas, loser.state_id)
        if item is None:
            failures[loser.state_id] = "meta_not_found"
            continue
        loser_dir, loser_meta = item
        frames = _sample_frames(loser_dir, loser_meta)
        if not frames:
            failures[loser.state_id] = "no_positive_samples"
            continue
        conds = [c for c in loser_meta.get("match_conditions", []) if isinstance(c, dict)]
        disabled = [c for c in conds if not c.get("enabled", False) and c.get("condition_status", "active") == "active"]
        candidates: list[dict[str, Any]] = []
        for cond in disabled:
            current_ok, _ = _condition_eval(cond, vision, frame_rgb)
            if current_ok:
                continue
            if all(_condition_eval(cond, vision, sample)[0] for sample in frames):
                candidates.append(cond)
        if not candidates:
            failures[loser.state_id] = "no_condition_passes_loser_samples_and_fails_current"
            continue
        selected = None
        # Conditions are conjunctive. Enable candidates one at a time and
        # re-run the actual matcher after every mutation; stop only when this
        # loser no longer matches the disambiguated frame.
        for candidate in sorted(candidates, key=_condition_rank, reverse=True):
            candidate["enabled"] = True
            selected = candidate
            check = _eval_state_match(loser_meta, loser_dir, vision, frame_rgb)
            if not check.success:
                break
        if selected is None or _eval_state_match(loser_meta, loser_dir, vision, frame_rgb).success:
            failures[loser.state_id] = "enabled_candidates_did_not_exclude_current"
            continue
        loser_meta.setdefault("model_info", {})
        if isinstance(loser_meta["model_info"], dict):
            loser_meta["model_info"]["last_exclusion_reason"] = {
                "source": "runtime_disambiguation",
                "winner_state_id": winner.state_id,
                "enabled_condition_id": str(selected.get("id", "")),
                "reason": "condition passes loser positive samples but fails current disambiguated screen",
                "created_at": _now_iso(),
            }
        loser_meta["updated_at"] = _now_iso()
        _save_json(loser_dir / "state.json", loser_meta)
        strengthened[loser.state_id] = str(selected.get("id", ""))
    if logger is not None:
        logger.event(
            "disambiguation_losers_strengthened",
            winner_state_id=winner.state_id,
            strengthened=strengthened,
            failures=failures,
        )
        if strengthened:
            logger.text(
                f"[fsm][disambiguation][losers] winner={winner.state_id} strengthened={strengthened}",
                "disambiguation_losers_strengthened_console",
                winner_state_id=winner.state_id,
                strengthened=strengthened,
                failures=failures,
            )
        elif failures:
            logger.text(
                f"[fsm][disambiguation][losers] winner={winner.state_id} no exclusion conditions failures={failures}",
                "disambiguation_losers_strengthen_failed",
                winner_state_id=winner.state_id,
                failures=failures,
            )
    return strengthened


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
    strengthened = _try_exclude_current_from_losers(
        winner=winner,
        losers=losers,
        metas=metas,
        vision=vision,
        frame_rgb=frame_rgb,
        logger=logger,
    )
    if len(strengthened) < len(losers):
        remaining_losers = [m for m in losers if m.state_id not in strengthened]
        _try_merge_ambiguous_states(
            winner=winner,
            losers=remaining_losers,
            metas=metas,
            vision=vision,
            frame_rgb=frame_rgb,
            logger=logger,
        )
    return winner
