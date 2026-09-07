from __future__ import annotations

import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ToolResult:
    status: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"progressed", "completed", "no_progress", "failed", "aborted"}:
            raise ValueError(f"invalid tool result status: {self.status!r}")

    @property
    def succeeded(self) -> bool:
        return self.status in {"progressed", "completed"}


ToolHandler = Callable[..., ToolResult | bool]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    supported_scene_modes: tuple[str, ...]
    handler: ToolHandler

    def prompt_entry(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "supported_scene_modes": list(self.supported_scene_modes),
        }


_TOOL_REGISTRY: dict[str, RegisteredTool] = {}


def register_tool(
    name: str,
    *,
    description: str,
    supported_scene_modes: list[str] | tuple[str, ...],
    handler: ToolHandler,
    replace: bool = False,
) -> None:
    normalized_name = str(name).strip()
    modes = tuple(dict.fromkeys(str(mode).strip() for mode in supported_scene_modes if str(mode).strip()))
    if not normalized_name or not description.strip() or not modes:
        raise ValueError("tool name, description, and supported_scene_modes are required")
    if not callable(handler):
        raise TypeError("tool handler must be callable")
    invalid_modes = set(modes) - {"ui_2d", "scene_3d", "unknown"}
    if invalid_modes:
        raise ValueError(f"unsupported scene modes: {sorted(invalid_modes)}")
    if normalized_name in _TOOL_REGISTRY and not replace:
        raise ValueError(f"tool already registered: {normalized_name}")
    _TOOL_REGISTRY[normalized_name] = RegisteredTool(
        name=normalized_name,
        description=str(description).strip(),
        supported_scene_modes=modes,
        handler=handler,
    )


def unregister_tool(name: str) -> None:
    _TOOL_REGISTRY.pop(str(name).strip(), None)


def tool_catalog() -> list[dict[str, Any]]:
    return [_TOOL_REGISTRY[name].prompt_entry() for name in sorted(_TOOL_REGISTRY)]


def get_tool(name: str) -> RegisteredTool | None:
    return _TOOL_REGISTRY.get(str(name).strip())


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
    if not template_path.is_file() and WAIT_TILL_COMBAT_END_LEGACY_TEMPLATE_PATH.is_file():
        template_path = WAIT_TILL_COMBAT_END_LEGACY_TEMPLATE_PATH
    if not template_path.is_file():
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
            time.sleep(0.5)
            emulator.tap(ix, iy)
            time.sleep(0.5)
            emulator.tap(ax, ay)
            time.sleep(0.5)
            print(f"[preset][find_and_interact_with_next_object] action_step={step_idx + 1}/10 move=({mx},{my}) interact=({ix},{iy})")

        print("[preset][find_and_interact_with_next_object] idle=5s")
        time.sleep(5.0)
        print(f"[preset][find_and_interact_with_next_object] cycle={cycle_idx} complete; next state check is before movement")


def invoke_tool(
    name: str,
    *,
    emulator,
    mapper,
    vision,
    state_id: str,
    matches_provider: Callable[[Any], list[Any]],
    find_match_by_state: Callable[[list[Any], str], Any],
) -> ToolResult:
    tool = get_tool(name)
    if tool is None:
        return ToolResult("failed", "tool_not_registered")
    result = tool.handler(
        emulator=emulator,
        mapper=mapper,
        vision=vision,
        state_id=state_id,
        matches_provider=matches_provider,
        find_match_by_state=find_match_by_state,
    )
    if isinstance(result, ToolResult):
        return result
    if isinstance(result, bool):
        return ToolResult("completed" if result else "failed")
    return ToolResult("failed", f"invalid_tool_result:{type(result).__name__}")


def _wait_tool(**context: Any) -> bool:
    return _run_wait_till_combat_end(context["emulator"], context["vision"])


def _find_next_tool(**context: Any) -> bool:
    return _run_find_and_interact_with_next_object(
        emulator=context["emulator"],
        mapper=context["mapper"],
        state_id=context["state_id"],
        matches_provider=context["matches_provider"],
        find_match_by_state=context["find_match_by_state"],
    )


register_tool(
    "wait_till_combat_end",
    description="Wait for an ongoing automated combat sequence to finish.",
    supported_scene_modes=("ui_2d", "scene_3d"),
    handler=_wait_tool,
)
register_tool(
    "find_and_interact_with_next_object",
    description="In a free-camera 3D scene, move forward and interact with the next reachable object.",
    supported_scene_modes=("scene_3d",),
    handler=_find_next_tool,
)
