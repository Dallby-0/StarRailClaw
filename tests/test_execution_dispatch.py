from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agent.behavior_tree.coord_mapper import CoordinateMapper
from state_machine.action_runner import _execute_state_action
from state_machine.presets import ToolResult, register_tool, tool_catalog, unregister_tool
from state_machine.protocol import state_bootstrap_json_schema


class FakeEmulator:
    def __init__(self, post_frame) -> None:
        self.post_frame = post_frame

    def screenshot(self, prefer_png=True):
        assert prefer_png is True
        return self.post_frame


def test_dispatcher_invokes_3d_tool_outside_page_handler(tmp_path: Path, monkeypatch) -> None:
    state_meta = {
        "state_id": "001",
        "scene_mode": "scene_3d",
        "execution": {"kind": "invoke_tool", "tool_name": "find_and_interact_with_next_object"},
    }
    (tmp_path / "state.json").write_text(json.dumps(state_meta), encoding="utf-8")
    post = np.ones((2, 2, 3), dtype=np.uint8)
    calls: list[str] = []

    def fake_invoke(name, **context):
        calls.append(name)
        assert context["state_id"] == "001"
        return ToolResult("progressed", "test")

    monkeypatch.setattr("state_machine.action_runner.invoke_tool", fake_invoke)
    monkeypatch.setattr("state_machine.action_runner._save_runtime", lambda runtime: None)
    monkeypatch.setattr("state_machine.page_handler.run_page_handler", lambda **kwargs: (_ for _ in ()).throw(AssertionError("page handler called")))
    runtime: dict = {}
    ok, frame = _execute_state_action(
        emulator=FakeEmulator(post),
        mapper=CoordinateMapper(real_w=1280, real_h=720),
        vision=SimpleNamespace(),
        state_id="001",
        state_dir=tmp_path,
        frame_before=np.zeros((2, 2, 3), dtype=np.uint8),
        matches_provider=lambda image: [],
        runtime=runtime,
        llm=SimpleNamespace(),
        llm_session_id="test",
        system_prompt="test",
        graph={"nodes": [], "edges": []},
        prefer_reachable_first=False,
    )

    assert ok is True
    assert frame is post
    assert calls == ["find_and_interact_with_next_object"]
    assert runtime["pending_action_id"] == "tool:find_and_interact_with_next_object"
    assert "reactive_total_actions" not in runtime


def test_dynamic_tool_registration_updates_prompt_catalog_and_schema() -> None:
    name = "test_dynamic_2d_tool"
    register_tool(
        name,
        description="Handle a known two-dimensional surface.",
        supported_scene_modes=("ui_2d",),
        handler=lambda **context: True,
    )
    try:
        catalog = tool_catalog()
        assert next(item for item in catalog if item["name"] == name)["supported_scene_modes"] == ["ui_2d"]
        schema = state_bootstrap_json_schema(catalog)
        tool_variant = next(
            item for item in schema["properties"]["execution"]["anyOf"]
            if item["properties"]["kind"]["enum"] == ["invoke_tool"]
        )
        assert name in tool_variant["properties"]["tool_name"]["enum"]
    finally:
        unregister_tool(name)
