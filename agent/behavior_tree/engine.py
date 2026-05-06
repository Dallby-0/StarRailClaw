from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sr_tools.emulator import EmulatorClient

from .coord_mapper import CoordinateMapper
from .storage import read_json
from .vision import VisionEngine


class NodeStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


@dataclass
class TickContext:
    frame_rgb: np.ndarray
    prev_frame_rgb: np.ndarray | None
    now_monotonic: float
    state: dict[str, Any]


@dataclass
class BehaviorTreeEngine:
    emulator: EmulatorClient
    tree_dir: Path
    mapper: CoordinateMapper
    llm_decide_cb: Callable[[TickContext], NodeStatus]
    log_fn: Callable[[str], None] = print
    vision: VisionEngine = field(init=False)

    def __post_init__(self) -> None:
        self.vision = VisionEngine(self.mapper, log_fn=self.log_fn)

    def tick(self, tree: dict[str, Any], ctx: TickContext, node_path: str = "root") -> NodeStatus:
        ntype = tree.get("type")
        name = str(tree.get("name", "")).strip()
        if ntype == "Sequence":
            status = self._tick_sequence(tree, ctx, node_path)
            self._log_node(node_path, ntype, name, status)
            return status
        if ntype == "Selector":
            status = self._tick_selector(tree, ctx, node_path)
            self._log_node(node_path, ntype, name, status)
            return status
        if ntype == "Condition":
            status = self._tick_condition(tree, ctx)
            self._log_node(node_path, ntype, name, status)
            return status
        if ntype == "Action":
            status = self._tick_action(tree, ctx, node_path)
            self._log_node(node_path, ntype, name, status)
            return status
        if ntype == "SubTree":
            ref = str(tree.get("ref", "")).strip()
            if not ref:
                return NodeStatus.FAILURE
            sub = read_json(self.tree_dir / ref)
            status = self.tick(sub, ctx, node_path=f"{node_path}/sub:{ref}")
            self._log(f"[bt][subtree] path={node_path} ref={ref} status={status.value}")
            return status
        if ntype == "Repeat":
            child = tree.get("child")
            if not isinstance(child, dict):
                return NodeStatus.FAILURE
            status = self.tick(child, ctx, node_path=f"{node_path}/child")
            if status == NodeStatus.FAILURE:
                self._log_node(node_path, ntype, name, NodeStatus.FAILURE)
                return NodeStatus.FAILURE
            self._log_node(node_path, ntype, name, NodeStatus.RUNNING)
            return NodeStatus.RUNNING
        return NodeStatus.FAILURE

    def _tick_sequence(self, node: dict[str, Any], ctx: TickContext, node_path: str) -> NodeStatus:
        children = node.get("children") or []
        key = f"{node_path}:idx"
        idx = int(ctx.state.get(key, 0))
        while idx < len(children):
            status = self.tick(children[idx], ctx, node_path=f"{node_path}/{idx}")
            if status == NodeStatus.SUCCESS:
                idx += 1
                continue
            if status == NodeStatus.RUNNING:
                ctx.state[key] = idx
                return NodeStatus.RUNNING
            ctx.state.pop(key, None)
            return NodeStatus.FAILURE
        ctx.state.pop(key, None)
        return NodeStatus.SUCCESS

    def _tick_selector(self, node: dict[str, Any], ctx: TickContext, node_path: str) -> NodeStatus:
        children = node.get("children") or []
        key = f"{node_path}:idx"
        idx = int(ctx.state.get(key, 0))
        while idx < len(children):
            status = self.tick(children[idx], ctx, node_path=f"{node_path}/{idx}")
            if status == NodeStatus.SUCCESS:
                ctx.state.pop(key, None)
                return NodeStatus.SUCCESS
            if status == NodeStatus.RUNNING:
                ctx.state[key] = idx
                return NodeStatus.RUNNING
            idx += 1
        ctx.state.pop(key, None)
        return NodeStatus.FAILURE

    def _tick_condition(self, node: dict[str, Any], ctx: TickContext) -> NodeStatus:
        check = node.get("check")
        params = node.get("params") or {}
        ok = self._eval_condition(check, params, ctx)
        return NodeStatus.SUCCESS if ok else NodeStatus.FAILURE

    def _eval_condition(self, check: str, params: dict[str, Any], ctx: TickContext) -> bool:
        if check == "template_match":
            template = str(params.get("template", "")).strip()
            rect = params.get("rect")
            threshold = float(params.get("threshold", 0.8))
            ok, _, _ = self.vision.match_template(ctx.frame_rgb, self.tree_dir / "templates" / template, rect, threshold)
            self._log(
                f"[bt][condition] check=template_match template={template} rect={rect} threshold={threshold} ok={ok}"
            )
            return ok
        if check in {"text_match", "text_match_white"}:
            target = str(params.get("text", "")).strip()
            rect = params.get("rect")
            threshold = float(params.get("threshold", 0.8))
            white = check == "text_match_white"
            words = self.vision.ocr(ctx.frame_rgb, rect, white_text=white)
            self._log_ocr_debug(words, rect, threshold, f"condition:{check}")
            for item in words:
                if item["conf"] >= threshold and target in item["text"]:
                    self._log(
                        f"[bt][condition] check={check} text={target} rect={rect} threshold={threshold} ok=True matched={item['text']}"
                    )
                    return True
            self._log(f"[bt][condition] check={check} text={target} rect={rect} threshold={threshold} ok=False")
            return False
        if check == "stable":
            rect = params.get("rect")
            diff_threshold = float(params.get("diff_threshold", 0.01))
            ok = self.vision.is_stable(ctx.prev_frame_rgb, ctx.frame_rgb, rect, diff_threshold)
            self._log(f"[bt][condition] check=stable rect={rect} diff_threshold={diff_threshold} ok={ok}")
            return ok
        if check == "AND":
            return all(self._eval_condition(c.get("check"), c.get("params") or {}, ctx) for c in (params.get("conditions") or []))
        if check == "OR":
            return any(self._eval_condition(c.get("check"), c.get("params") or {}, ctx) for c in (params.get("conditions") or []))
        if check == "NOT":
            c = params.get("condition") or {}
            return not self._eval_condition(c.get("check"), c.get("params") or {}, ctx)
        return False

    def _tick_action(self, node: dict[str, Any], ctx: TickContext, node_path: str) -> NodeStatus:
        do = node.get("do")
        params = node.get("params") or {}
        if do == "click_template":
            ok, center, _ = self.vision.match_template(
                ctx.frame_rgb,
                self.tree_dir / "templates" / str(params.get("template", "")),
                params.get("rect"),
                float(params.get("threshold", 0.8)),
            )
            if not ok or center is None:
                self._log(
                    f"[bt][action] do=click_template status=failure template={params.get('template')} rect={params.get('rect')}"
                )
                return NodeStatus.FAILURE
            self.emulator.tap(center[0], center[1])
            logical = self.mapper.point_to_logical(center[0], center[1])
            self._log(
                f"[bt][action] do=click_template status=success template={params.get('template')} logical={logical} real={center}"
            )
            return NodeStatus.SUCCESS
        if do == "click_pos":
            pos = params.get("pos") or []
            if len(pos) != 2:
                return NodeStatus.FAILURE
            rx, ry = self.mapper.point_to_real(int(pos[0]), int(pos[1]))
            self.emulator.tap(rx, ry)
            self._log(f"[bt][action] do=click_pos status=success logical=({int(pos[0])},{int(pos[1])}) real=({rx},{ry})")
            return NodeStatus.SUCCESS
        if do in {"click_text", "click_text_any", "ocr_click_priority"}:
            rect = params.get("rect")
            threshold = float(params.get("threshold", 0.8))
            words = self.vision.ocr(ctx.frame_rgb, rect, white_text=False)
            self._log_ocr_debug(words, rect, threshold, f"action:{do}")
            if do == "click_text":
                target = str(params.get("text", "")).strip()
                for w in words:
                    if w["conf"] >= threshold and target in w["text"]:
                        self.emulator.tap(w["center"][0], w["center"][1])
                        logical = self.mapper.point_to_logical(w["center"][0], w["center"][1])
                        self._log(
                            f"[bt][action] do=click_text status=success target={target} matched={w['text']} logical={logical} real={w['center']}"
                        )
                        return NodeStatus.SUCCESS
                self._log(f"[bt][action] do=click_text status=failure target={target} rect={rect}")
                return NodeStatus.FAILURE
            if do == "click_text_any":
                if not words:
                    self._log(f"[bt][action] do=click_text_any status=failure rect={rect}")
                    return NodeStatus.FAILURE
                self.emulator.tap(words[0]["center"][0], words[0]["center"][1])
                logical = self.mapper.point_to_logical(words[0]["center"][0], words[0]["center"][1])
                self._log(
                    f"[bt][action] do=click_text_any status=success text={words[0]['text']} logical={logical} real={words[0]['center']}"
                )
                return NodeStatus.SUCCESS
            keywords = [str(k) for k in (params.get("keywords") or [])]
            for kw in keywords:
                for w in words:
                    if w["conf"] >= threshold and kw in w["text"]:
                        self.emulator.tap(w["center"][0], w["center"][1])
                        logical = self.mapper.point_to_logical(w["center"][0], w["center"][1])
                        self._log(
                            f"[bt][action] do=ocr_click_priority status=success keyword={kw} matched={w['text']} logical={logical} real={w['center']}"
                        )
                        return NodeStatus.SUCCESS
            self._log(
                f"[bt][action] do=ocr_click_priority status=failure keywords={keywords} rect={rect} threshold={threshold}"
            )
            return NodeStatus.FAILURE
        if do == "wait":
            ms = int(params.get("milliseconds", 0))
            return self._wait_until(node_path, ctx, ms)
        if do == "wait_stable":
            timeout = int(params.get("timeout", 10000))
            rect = params.get("rect")
            diff_threshold = float(params.get("diff_threshold", 0.01))
            if self.vision.is_stable(ctx.prev_frame_rgb, ctx.frame_rgb, rect, diff_threshold):
                self._clear_wait(node_path, ctx)
                self._log(f"[bt][action] do=wait_stable status=success rect={rect}")
                return NodeStatus.SUCCESS
            return self._wait_until(node_path, ctx, timeout, fail_on_timeout=True)
        if do in {"wait_for_template", "wait_for_template_disappear"}:
            template = str(params.get("template", "")).strip()
            rect = params.get("rect")
            threshold = float(params.get("threshold", 0.8))
            timeout = int(params.get("timeout", 10000))
            found, _, _ = self.vision.match_template(ctx.frame_rgb, self.tree_dir / "templates" / template, rect, threshold)
            if do == "wait_for_template" and found:
                self._clear_wait(node_path, ctx)
                self._log(f"[bt][action] do=wait_for_template status=success template={template}")
                return NodeStatus.SUCCESS
            if do == "wait_for_template_disappear" and not found:
                self._clear_wait(node_path, ctx)
                self._log(f"[bt][action] do=wait_for_template_disappear status=success template={template}")
                return NodeStatus.SUCCESS
            return self._wait_until(node_path, ctx, timeout, fail_on_timeout=True)
        if do == "llm_decide":
            self._log("[bt][action] do=llm_decide invoke")
            return self.llm_decide_cb(ctx)
        return NodeStatus.FAILURE

    def _wait_until(self, node_path: str, ctx: TickContext, milliseconds: int, fail_on_timeout: bool = False) -> NodeStatus:
        key = f"{node_path}:wait_start"
        start = float(ctx.state.get(key, 0.0))
        if start <= 0.0:
            ctx.state[key] = ctx.now_monotonic
            self._log(f"[bt][action] wait_start path={node_path} timeout_ms={milliseconds}")
            return NodeStatus.RUNNING
        elapsed_ms = int((ctx.now_monotonic - start) * 1000)
        if elapsed_ms >= milliseconds:
            ctx.state.pop(key, None)
            self._log(
                f"[bt][action] wait_end path={node_path} elapsed_ms={elapsed_ms} status={'failure' if fail_on_timeout else 'success'}"
            )
            return NodeStatus.FAILURE if fail_on_timeout else NodeStatus.SUCCESS
        self._log(f"[bt][action] waiting path={node_path} elapsed_ms={elapsed_ms}/{milliseconds}")
        return NodeStatus.RUNNING

    def _clear_wait(self, node_path: str, ctx: TickContext) -> None:
        ctx.state.pop(f"{node_path}:wait_start", None)

    def _log(self, message: str) -> None:
        self.log_fn(message)

    def _log_node(self, node_path: str, ntype: str, name: str, status: NodeStatus) -> None:
        short_name = name if name else "-"
        self._log(f"[bt][node] path={node_path} type={ntype} name={short_name} status={status.value}")

    def _log_ocr_debug(self, words: list[dict[str, Any]], rect: Any, threshold: float, source: str) -> None:
        self._log(f"[bt][ocr] source={source} rect={rect} threshold={threshold} candidates={len(words)}")
        preview = words[:8]
        for i, w in enumerate(preview, start=1):
            text = str(w.get("text", "")).replace("\n", " ").strip()
            conf = float(w.get("conf", 0.0))
            bbox = w.get("bbox")
            center = w.get("center")
            self._log(
                f"[bt][ocr] #{i} text={text!r} conf={conf:.3f} bbox={bbox} center={center}"
            )
