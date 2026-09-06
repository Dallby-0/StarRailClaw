from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np

# The isolated unit-test runtime need not install the HTTP client because this
# test exercises pure matching logic and never constructs an emulator or LLM.
sys.modules.setdefault("requests", types.ModuleType("requests"))

from state_machine.disambiguation import _try_exclude_current_from_losers, refine_state_pair
from state_machine.matching import MatchResult, _eval_state_match
from state_machine.merge import _try_merge_ambiguous_states


class _PixelVision:
    def ocr(self, frame, rect, white_text=False):
        del rect, white_text
        text = "common loser-only" if int(frame[0, 0, 0]) == 10 else "common winner"
        return [{"text": text}]


def _text_condition(condition_id: str, text: str, *, enabled: bool) -> dict:
    return {
        "id": condition_id,
        "kind": "text_line_contains",
        "enabled": enabled,
        "condition_status": "active",
        "role": "identity_support",
        "params": {"text": text, "rect": [0, 0, 1, 1]},
        "stability": "high",
        "discrimination": "high",
    }


def test_loser_strengthening_is_verified_with_real_matcher(tmp_path: Path) -> None:
    winner_dir = tmp_path / "winner"
    loser_dir = tmp_path / "loser"
    winner_dir.mkdir()
    loser_dir.mkdir()
    loser_sample = np.full((2, 2, 3), 10, dtype=np.uint8)
    current = np.full((2, 2, 3), 20, dtype=np.uint8)
    sample_path = loser_dir / "screenshot_1.png"
    cv2.imwrite(str(sample_path), cv2.cvtColor(loser_sample, cv2.COLOR_RGB2BGR))

    winner_meta = {"state_id": "winner", "match_conditions": [_text_condition("winner", "winner", enabled=True)]}
    loser_meta = {
        "state_id": "loser",
        "samples": [{"path": str(sample_path)}],
        "match_clauses": [{"all": ["broad"]}],
        "match_conditions": [
            _text_condition("broad", "common", enabled=True),
            _text_condition("loser-exclusive", "loser-only", enabled=False),
        ],
    }
    # Keep the initial loser match explicit; the strengthening helper only
    # mutates conditions and verifies the post-mutation state itself.
    winner = MatchResult("winner", winner_dir, 1, 1, True, 1, 1)
    loser = MatchResult("loser", loser_dir, 1, 1, True, 1, 2)
    strengthened = _try_exclude_current_from_losers(
        winner=winner,
        losers=[loser],
        metas=[(winner_dir, winner_meta), (loser_dir, loser_meta)],
        vision=_PixelVision(),
        frame_rgb=current,
    )

    assert strengthened == {"loser": "loser-exclusive"}
    assert not _eval_state_match(loser_meta, loser_dir, _PixelVision(), current).success


def test_pair_refinement_promotes_runtime_verified_text_element(tmp_path: Path) -> None:
    correct_dir = tmp_path / "correct"
    excluded_dir = tmp_path / "excluded"
    correct_dir.mkdir()
    excluded_dir.mkdir()
    old_frame = np.full((2, 2, 3), 10, dtype=np.uint8)
    current = np.full((2, 2, 3), 20, dtype=np.uint8)
    old_path = excluded_dir / "screenshot_1.png"
    new_path = correct_dir / "screenshot_1.png"
    cv2.imwrite(str(old_path), cv2.cvtColor(old_frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(new_path), cv2.cvtColor(current, cv2.COLOR_RGB2BGR))
    correct_meta = {
        "state_id": "correct",
        "match_conditions": [_text_condition("common", "common", enabled=True)],
        "match_clauses": [{"all": ["common"]}],
        "elements": [],
        "samples": [{"path": str(new_path)}],
    }
    excluded_meta = {
        "state_id": "excluded",
        "match_conditions": [_text_condition("common", "common", enabled=True)],
        "match_clauses": [{"all": ["common"]}],
        "elements": [{
            "type": "text_line",
            "text": "loser-only",
            "bbox": [0, 0, 1, 1],
            "role": "diagnostic",
            "stability": "high",
            "discrimination": "high",
            "observable": True,
        }],
        "samples": [{"path": str(old_path)}],
    }
    (correct_dir / "state.json").write_text("{}", encoding="utf-8")
    (excluded_dir / "state.json").write_text("{}", encoding="utf-8")
    metas = [(correct_dir, correct_meta), (excluded_dir, excluded_meta)]
    result = refine_state_pair(
        correct_state_id="correct",
        excluded_state_id="excluded",
        metas=metas,
        vision=_PixelVision(),
        correct_frame_rgb=current,
    )
    assert result["excluded"] == {"excluded": "runtime_text_1"}
    persisted = json.loads((excluded_dir / "state.json").read_text(encoding="utf-8"))
    assert persisted["match_clauses"] == [{"all": ["common", "runtime_text_1"]}]


def test_ambiguous_merge_rejects_different_page_types(tmp_path: Path) -> None:
    winner_dir = tmp_path / "winner"
    loser_dir = tmp_path / "loser"
    winner_dir.mkdir()
    loser_dir.mkdir()
    winner_meta = {"state_id": "winner", "page_type": "story", "match_conditions": []}
    loser_meta = {"state_id": "loser", "page_type": "story_with_options", "match_conditions": []}
    winner = MatchResult("winner", winner_dir, 1, 1, True, 1, 1)
    loser = MatchResult("loser", loser_dir, 1, 1, True, 1, 1)
    merged = _try_merge_ambiguous_states(
        winner=winner,
        losers=[loser],
        metas=[(winner_dir, winner_meta), (loser_dir, loser_meta)],
        vision=_PixelVision(),
        frame_rgb=np.full((2, 2, 3), 20, dtype=np.uint8),
    )
    assert merged == set()
