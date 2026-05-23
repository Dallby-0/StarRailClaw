from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.behavior_tree.vision import VisionEngine

from .constants import DISCRIMINATION_SCORE, MATCH_DISCRIMINATION_TARGET, STABILITY_SCORE


@dataclass
class MatchResult:
    state_id: str
    state_dir: Path
    passed_enabled: int
    total_enabled: int
    success: bool
    passed_all: int
    total_all: int


def _ocr_lines(vision: VisionEngine, frame_rgb, rect: list[int] | None) -> list[str]:
    entries = vision.ocr(frame_rgb, rect, white_text=False)
    return [str(e.get("text", "")).strip() for e in entries if str(e.get("text", "")).strip()]


def _condition_passed(cond: dict[str, Any], vision: VisionEngine, frame_rgb) -> bool:
    kind = str(cond.get("kind", ""))
    params = cond.get("params", {})
    if not isinstance(params, dict):
        params = {}
    rect = params.get("rect")
    if not (isinstance(rect, list) and len(rect) == 4):
        bbox = cond.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            rect = bbox
    if kind == "text_line_contains":
        needle = str(params.get("text", "")).strip()
        if not needle:
            return False
        lines = _ocr_lines(vision, frame_rgb, rect)
        return any(needle in line for line in lines)
    if kind == "region_template":
        tpl_path = str(params.get("template_path", "")).strip()
        if not tpl_path:
            return False
        threshold = float(params.get("threshold", 0.8))
        ok, _, _ = vision.match_template(frame_rgb, Path(tpl_path), rect, threshold=threshold)
        return ok
    return False


def _level(raw: Any, default: str = "mid") -> str:
    value = str(raw or default).strip().lower()
    return value if value in {"high", "mid", "low"} else default


def _discrimination_score(cond: dict[str, Any]) -> int:
    return DISCRIMINATION_SCORE[_level(cond.get("discrimination"), "mid")]


def _stability_score(cond: dict[str, Any]) -> int:
    return STABILITY_SCORE[_level(cond.get("stability"), "mid")]


def _condition_sort_key(cond: dict[str, Any]) -> tuple[int, int]:
    return (_stability_score(cond), _discrimination_score(cond))


def _select_enabled_conditions(
    conditions: list[dict[str, Any]],
    vision: VisionEngine,
    frame_rgb,
) -> tuple[list[dict[str, Any]], bool]:
    valid = [c for c in conditions if c.get("kind") in {"text_line_contains", "region_template"}]
    passed = [c for c in valid if _condition_passed(c, vision, frame_rgb)]
    for c in conditions:
        c["enabled"] = False
    if not passed:
        return conditions, False

    ordered = sorted(passed, key=_condition_sort_key, reverse=True)
    selected: list[dict[str, Any]] = []
    score_sum = 0
    for c in ordered:
        selected.append(c)
        score_sum += _discrimination_score(c)
        if score_sum >= MATCH_DISCRIMINATION_TARGET:
            break
    if score_sum < MATCH_DISCRIMINATION_TARGET:
        selected = ordered

    weak = len(selected) == 1 and score_sum >= MATCH_DISCRIMINATION_TARGET
    selected_ids = {id(c) for c in selected}
    for c in conditions:
        c["enabled"] = id(c) in selected_ids
    return conditions, weak


def _eval_state_match(meta: dict[str, Any], state_dir: Path, vision: VisionEngine, frame_rgb) -> MatchResult:
    conds = [c for c in meta.get("match_conditions", []) if isinstance(c, dict)]
    enabled_conds = [c for c in conds if c.get("enabled", False)]
    passed_enabled = sum(1 for c in enabled_conds if _condition_passed(c, vision, frame_rgb))
    passed_all = sum(1 for c in conds if _condition_passed(c, vision, frame_rgb))
    total_enabled = len(enabled_conds)
    total_all = len(conds)
    success = (passed_enabled == total_enabled) if total_enabled > 0 else False
    return MatchResult(
        state_id=str(meta.get("state_id", "")),
        state_dir=state_dir,
        passed_enabled=passed_enabled,
        total_enabled=total_enabled,
        success=success,
        passed_all=passed_all,
        total_all=total_all,
    )


def _select_best_for_unknown(matches: list[MatchResult]) -> MatchResult | None:
    cands = [m for m in matches if m.success]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    cands_sorted = sorted(cands, key=lambda m: (-m.passed_all, -m.passed_enabled))
    return cands_sorted[0]


def _find_match_by_state(matches: list[MatchResult], state_id: str) -> MatchResult | None:
    for m in matches:
        if m.state_id == state_id:
            return m
    return None
