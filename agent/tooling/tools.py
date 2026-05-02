from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from sr_tools.emulator import EmulatorClient


def build_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "tap",
                "description": "Tap a point on emulator screen",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "description": "x coordinate"},
                        "y": {"type": "integer", "description": "y coordinate"},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
        }
    ]


class ToolExecutor:
    def __init__(self, emulator: EmulatorClient) -> None:
        self.emulator = emulator
        self.viewport_size: tuple[int, int] | None = None
        self._handlers: Dict[str, Callable[..., Any]] = {
            "tap": self._tap,
        }

    def set_viewport_size(self, width: int, height: int) -> None:
        self.viewport_size = (int(width), int(height))

    def _tap(self, x: int, y: int) -> Dict[str, Any]:
        raw_x, raw_y = int(x), int(y)
        # The model currently returns coordinates in a normalized 1000x1000 space.
        # Remap to our agent viewport space (1280x720).
        # We intentionally do not scale with wm size because this emulator can
        # report rotated dimensions that distort otherwise-correct tap points.
        req_x = int(round(raw_x * 1280 / 1000))
        req_y = int(round(raw_y * 720 / 1000))
        dev_size = self.emulator.screen_size()
        real_x, real_y = req_x, req_y
        note = "ignore_wm_size"

        if self.viewport_size:
            view_w, view_h = self.viewport_size
            if view_w > 0 and view_h > 0:
                real_x = max(0, min(view_w - 1, real_x))
                real_y = max(0, min(view_h - 1, real_y))

        self.emulator.tap(real_x, real_y)
        return {
            "ok": True,
            "action": "tap",
            "serial": self.emulator.serial,
            "requested": {"x": raw_x, "y": raw_y},
            "remapped_viewport": {"x": req_x, "y": req_y},
            "actual": {"x": real_x, "y": real_y},
            "device_size": {"w": dev_size[0], "h": dev_size[1]} if dev_size else None,
            "viewport_size": {"w": self.viewport_size[0], "h": self.viewport_size[1]} if self.viewport_size else None,
            "note": note or None,
        }

    def run_tool_call(self, tool_call: Dict[str, Any]) -> str:
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        if name not in self._handlers:
            return json.dumps({"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False)

        args_text = fn.get("arguments", "{}")
        try:
            args = json.loads(args_text) if args_text else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"ok": False, "error": f"invalid arguments: {exc}"}, ensure_ascii=False)

        try:
            result = self._handlers[name](**args)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
