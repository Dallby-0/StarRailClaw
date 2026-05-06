from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .engine import NodeStatus
from .storage import extract_json_object, node_signature, read_json, write_json


@dataclass
class GrowthManager:
    tree_path: Path
    template_dir: Path
    fail_streak_for_reflect: int = 6
    max_inline_nodes: int = 20
    recent_results: list[bool] = field(default_factory=list)

    def load_tree(self) -> dict[str, Any]:
        return read_json(self.tree_path)

    def save_tree(self, tree: dict[str, Any]) -> None:
        write_json(self.tree_path, tree)

    def record_result(self, ok: bool) -> None:
        self.recent_results.append(ok)
        if len(self.recent_results) > 200:
            self.recent_results = self.recent_results[-200:]

    def need_reflect(self, tree: dict[str, Any]) -> bool:
        if len(self.recent_results) >= self.fail_streak_for_reflect and all(not x for x in self.recent_results[-self.fail_streak_for_reflect :]):
            return True
        selector = self._root_selector(tree)
        if selector is None:
            return False
        return len(selector.get("children") or []) > self.max_inline_nodes

    def insert_experience(self, tree: dict[str, Any], condition: dict[str, Any], actions: dict[str, Any]) -> bool:
        selector = self._root_selector(tree)
        if selector is None:
            return False
        sequence = {
            "type": "Sequence",
            "name": f"经验_{int(time.time())}",
            "comment": "由llm_decide探索成功后自动固化",
            "children": [condition, actions],
        }
        sig = node_signature(sequence["children"][0])
        for child in selector.get("children") or []:
            if not isinstance(child, dict) or child.get("type") != "Sequence":
                continue
            children = child.get("children") or []
            if not children:
                continue
            if node_signature(children[0]) == sig:
                return False

        children = selector.get("children") or []
        llm_idx = 0
        for i, child in enumerate(children):
            if isinstance(child, dict) and child.get("type") == "Action" and child.get("do") == "llm_decide":
                llm_idx = i
                break
        children.insert(llm_idx, sequence)
        selector["children"] = children
        return True

    def _root_selector(self, tree: dict[str, Any]) -> dict[str, Any] | None:
        if tree.get("type") != "Repeat":
            return None
        child = tree.get("child")
        if not isinstance(child, dict) or child.get("type") != "Selector":
            return None
        return child

    def extract_and_replace_templates(self, frame_rgb, condition: dict[str, Any], vision) -> dict[str, Any]:
        clone = self._deep_copy(condition)

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "Condition" and node.get("check") == "template_match":
                    params = node.get("params") or {}
                    rect = params.get("rect")
                    if isinstance(rect, list) and len(rect) == 4:
                        name = f"tpl_{uuid.uuid4().hex[:12]}.png"
                        out = self.template_dir / name
                        vision.save_template_from_rect(frame_rgb, rect, out)
                        params["template"] = name
                        node["params"] = params
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(clone)
        return clone

    def parse_llm_growth_payload(self, text: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        obj = extract_json_object(text)
        if not isinstance(obj, dict):
            return None
        condition = obj.get("condition")
        actions = obj.get("actions")
        if not isinstance(condition, dict) or not isinstance(actions, dict):
            return None
        return condition, actions

    def _deep_copy(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deep_copy(v) for v in obj]
        return obj


@dataclass
class LlmDecisionResult:
    status: NodeStatus
    reason: str
