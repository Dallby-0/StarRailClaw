from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

# Preset parameters live in code (no local json config).
PRESET_WAIT_TIMEOUT_S = 600.0
WAIT_TILL_COMBAT_END_TEMPLATE_PATH = Path(
    "StateMachineResources/preset_assets/wait_till_combat_end/combat_ongoing_marker.png"
)
WAIT_TILL_COMBAT_END_LEGACY_TEMPLATE_PATH = Path(
    "StateMachineResources/templates/wait_till_combat_end/assets/combat_ongoing_marker.png"
)
WAIT_TILL_COMBAT_END_RECT = [38, 14, 63, 38]
WAIT_TILL_COMBAT_END_THRESHOLD = 0.8

FIND_NEXT_MOVE_PRESS_POS = (191, 676)
FIND_NEXT_INTERACT_POS = (639, 562)
FIND_NEXT_ATTACK_POS = (818, 735)


def _still_in_state(*, emulator, state_id: str, matches_provider, find_match_by_state) -> bool:
    """Return whether the latest frame still matches the preset's entry state.

    This probe deliberately resolves the specific state rather than merely
    checking that *some* state matched.  A successful match for another state
    means the interaction completed and lets the outer FSM settle the
    transition normally.
    """
    frame = emulator.screenshot(prefer_png=True)
    matches = matches_provider(frame)
    current = find_match_by_state(matches, state_id)
    return current is not None and bool(current.success)


def _run_wait_till_combat_end(emulator, vision) -> bool:
    template_path = WAIT_TILL_COMBAT_END_TEMPLATE_PATH
    if not template_path.exists() and WAIT_TILL_COMBAT_END_LEGACY_TEMPLATE_PATH.exists():
        template_path = WAIT_TILL_COMBAT_END_LEGACY_TEMPLATE_PATH
    if not template_path.exists():
        print(f"[preset] missing template: {template_path}")
        return False

    started_at = time.monotonic()
    while True:
        if time.monotonic() - started_at >= PRESET_WAIT_TIMEOUT_S:
            print(f"[preset][wait_till_combat_end] hit global timeout={PRESET_WAIT_TIMEOUT_S}s; treat as done")
            return True
        time.sleep(10.0)
        frame = emulator.screenshot(prefer_png=True)
        ok, _, sim = vision.match_template(frame, template_path, WAIT_TILL_COMBAT_END_RECT, threshold=WAIT_TILL_COMBAT_END_THRESHOLD)
        print(f"[preset][wait_till_combat_end] probe10s ok={ok} sim={sim:.3f}")
        if ok:
            continue
        burst_ok = False
        for _ in range(10):
            if time.monotonic() - started_at >= PRESET_WAIT_TIMEOUT_S:
                print(f"[preset][wait_till_combat_end] hit global timeout={PRESET_WAIT_TIMEOUT_S}s during burst; treat as done")
                return True
            time.sleep(0.5)
            frame2 = emulator.screenshot(prefer_png=True)
            ok2, _, sim2 = vision.match_template(frame2, template_path, WAIT_TILL_COMBAT_END_RECT, threshold=WAIT_TILL_COMBAT_END_THRESHOLD)
            print(f"[preset][wait_till_combat_end] probe0.5s ok={ok2} sim={sim2:.3f}")
            if ok2:
                burst_ok = True
                break
        if burst_ok:
            continue
        print("[preset][wait_till_combat_end] timeout; end preset")
        return True


def _run_find_and_interact_with_next_object(
    *,
    emulator,
    mapper,
    state_id: str,
    matches_provider: Callable[[Any], list[Any]],
    find_match_by_state: Callable[[list[Any], str], Any],
) -> bool:
    move = FIND_NEXT_MOVE_PRESS_POS
    interact = FIND_NEXT_INTERACT_POS
    attack = FIND_NEXT_ATTACK_POS
    cycle_idx = 0
    while True:
        cycle_idx += 1
        print(f"[preset][find_and_interact_with_next_object] cycle={cycle_idx} start")
        mx, my = mapper.point_to_real(int(move[0]), int(move[1]))
        ix, iy = mapper.point_to_real(int(interact[0]), int(interact[1]))
        ax, ay = mapper.point_to_real(int(attack[0]), int(attack[1]))
        for step_idx in range(10):
            # A previous interaction may have taken us to another GUI.  Probe
            # immediately before each movement so neither the swipe nor the
            # following interaction is sent to stale coordinates.
            if not _still_in_state(
                emulator=emulator,
                state_id=state_id,
                matches_provider=matches_provider,
                find_match_by_state=find_match_by_state,
            ):
                print(f"[preset][find_and_interact_with_next_object] state_changed before_action from={state_id}, treat as done")
                return True
            emulator.swipe(mx, my, mx, my, duration_ms=800)
            time.sleep(1)
            emulator.tap(ix, iy)
            time.sleep(1)
            print(f"[preset][find_and_interact_with_next_object] action_step={step_idx + 1}/10 move=({mx},{my}) interact=({ix},{iy})")

        print("[preset][find_and_interact_with_next_object] idle=5s")
        time.sleep(5.0)
        print(f"[preset][find_and_interact_with_next_object] cycle={cycle_idx} complete; next state check is before movement")


def run_preset(
    name: str,
    *,
    emulator,
    mapper,
    vision,
    state_id: str,
    matches_provider: Callable[[Any], list[Any]],
    find_match_by_state: Callable[[list[Any], str], Any],
) -> bool:
    if name == "wait_till_combat_end":
        return _run_wait_till_combat_end(emulator, vision)
    if name == "find_and_interact_with_next_object":
        return _run_find_and_interact_with_next_object(
            emulator=emulator,
            mapper=mapper,
            state_id=state_id,
            matches_provider=matches_provider,
            find_match_by_state=find_match_by_state,
        )
    print(f"[preset] unknown preset name={name}")
    return False
