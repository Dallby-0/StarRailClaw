from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.vision import VisionEngine
from agent.llm_client import DoubaoClient
from sr_tools.adb import resolve_target_serial
from sr_tools.emulator import EmulatorClient

FSM_DIR = Path("StateMachineResources")
FSM_TEMPLATES_DIR = FSM_DIR / "templates"
FSM_GRAPH_PATH = FSM_DIR / "state_graph.json"
FSM_SCHEMA_PATH = FSM_DIR / "llm_protocol_schema.json"
FSM_RUNTIME_PATH = FSM_DIR / "runtime_state.json"
FSM_DEBUG_DIR = FSM_DIR / "debug"

SCHEMA_VERSION = "1.0.0"
ALLOWED_CONDITION_KINDS = {"text_line_contains", "region_template"}
MAX_CONDITION_CANDIDATES = 6
MAX_CONDITION_COMBO = 3
MIN_COMBO_SIZE = 2
MAX_RETRY_PER_ACTION = 2
REPLAN_COOLDOWN_S = 60.0
LLM_PARSE_RETRY = 2
ACTION_CLICK_WAIT_S = 5.0

LLM_FSM_PROMPT = """你是视觉驱动游戏自动化的状态标注器和动作规划器,当前任务是通关崩坏星穹铁道差分宇宙。

你将收到一张游戏截图。你的任务：
1) 识别并输出用于区分该页面的关键信息，按重要性排序。
2) 输出推进流程的点击动作序列（1000x1000逻辑坐标）。

强约束：
- 必须输出严格 JSON 对象，字段必须符合约定。
- elements: 每项必须有 type。
  - type=text_line: 必须给 text, bbox[x1,y1,x2,y2], brief。
  - type=pattern: 必须给 bbox[x1,y1,x2,y2], brief。
- 注意选择每次进入页面时100%会出现的元素，而不是随时变动的数字或很可能随游戏进度变化而在同一页面发生变化的图标。
- slug: 英文小写+下划线，简短可读。
- actions: 仅点击序列，每项包含 x, y, brief。
- 不要输出 candidate_conditions，不要输出权重、kind、代码指令。

输出字段：
{
  "page_summary": "...",
  "slug": "...",
  "elements": [...],
  "actions": [...]
}
"""


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _slugify(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", raw.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "state"


def _normalize_assistant_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in {"text", "output_text"}]
        return "".join(parts).strip()
    return ""


def _build_user_message_from_frame(image_rgb) -> dict[str, Any]:
    data_url = DoubaoClient.encode_image_to_data_url(image_rgb)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "请按约定输出JSON"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }


def _ensure_fsm_resources() -> None:
    FSM_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    FSM_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    if not FSM_GRAPH_PATH.exists():
        FSM_GRAPH_PATH.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "nodes": [],
                    "edges": [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if not FSM_RUNTIME_PATH.exists():
        FSM_RUNTIME_PATH.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "last_state_id": None,
                    "last_replan_at": 0.0,
                    "consecutive_failures": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    FSM_SCHEMA_PATH.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "required_fields": ["page_summary", "slug", "elements", "actions"],
                "condition_kinds": ["text_line_contains", "region_template"],
                "action_fields": ["x", "y", "brief"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_state_meta() -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for d in sorted(FSM_TEMPLATES_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "state.json"
        if not meta_path.exists():
            continue
        try:
            out.append((d, _load_json(meta_path)))
        except Exception:
            continue
    return out


@dataclass
class MatchResult:
    state_id: str
    state_dir: Path
    score: float
    passed_ids: list[str]
    failed_ids: list[str]


def _ocr_lines(vision: VisionEngine, frame_rgb, rect: list[int] | None) -> list[str]:
    entries = vision.ocr(frame_rgb, rect, white_text=False)
    return [str(e.get("text", "")).strip() for e in entries if str(e.get("text", "")).strip()]


def _condition_passed(cond: dict[str, Any], vision: VisionEngine, frame_rgb, prev_frame_rgb) -> bool:
    if not cond.get("enabled", True):
        return False
    kind = cond.get("kind")
    raw_params = cond.get("params", {})
    if isinstance(raw_params, str):
        try:
            raw_params = json.loads(raw_params)
        except Exception:
            raw_params = {}
    params = raw_params if isinstance(raw_params, dict) else {}
    rect = params.get("rect")
    if not (isinstance(rect, list) and len(rect) == 4):
        bbox = cond.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            rect = bbox
    if kind == "stable":
        return vision.is_stable(prev_frame_rgb, frame_rgb, rect, float(params.get("diff_threshold", 0.01)))
    if kind == "text_line_equals":
        expected = str(params.get("text", "")).strip()
        lines = _ocr_lines(vision, frame_rgb, rect)
        return expected in lines
    if kind == "text_line_contains":
        needle = str(params.get("text", "")).strip()
        lines = _ocr_lines(vision, frame_rgb, rect)
        return any(needle in x for x in lines)
    if kind == "region_template":
        tpl_path = params.get("template_path")
        threshold = float(params.get("threshold", 0.8))
        if not tpl_path:
            return False
        ok, _, _ = vision.match_template(frame_rgb, Path(tpl_path), rect, threshold=threshold)
        return ok
    return False


def _match_state(meta: dict[str, Any], state_dir: Path, vision: VisionEngine, frame_rgb, prev_frame_rgb) -> MatchResult:
    conditions = [c for c in meta.get("match_conditions", []) if isinstance(c, dict) and c.get("enabled", True)]
    passed: list[str] = []
    failed: list[str] = []
    score = 0.0
    for cond in conditions:
        cid = str(cond.get("id", "")) or f"cond_{len(passed)+len(failed)+1}"
        if _condition_passed(cond, vision, frame_rgb, prev_frame_rgb):
            passed.append(cid)
            score += float(cond.get("weight", 1.0))
        else:
            failed.append(cid)
    return MatchResult(state_id=str(meta.get("state_id", "")), state_dir=state_dir, score=score, passed_ids=passed, failed_ids=failed)


def _load_runtime() -> dict[str, Any]:
    return _load_json(FSM_RUNTIME_PATH)


def _save_runtime(runtime: dict[str, Any]) -> None:
    _save_json(FSM_RUNTIME_PATH, runtime)


def _reachable_targets(graph: dict[str, Any], state_id: str | None) -> set[str] | None:
    if not state_id:
        return None
    reachable = {str(e.get("to_state_id")) for e in graph.get("edges", []) if str(e.get("from_state_id")) == state_id and e.get("enabled", True)}
    return reachable


def _select_best_match(matches: list[MatchResult], reachable: set[str] | None) -> MatchResult | None:
    filtered = matches
    if reachable is not None:
        filtered = [m for m in matches if m.state_id in reachable]
    if not filtered:
        return None
    return sorted(filtered, key=lambda m: (-m.score, m.state_id))[0]


def _save_frame(path: Path, frame_rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


def _parse_llm_payload(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    required = {"page_summary", "slug", "elements", "actions"}
    if not isinstance(payload, dict) or not required.issubset(payload.keys()):
        return None
    return payload


def _normalize_condition(cond: dict[str, Any], idx: int) -> dict[str, Any] | None:
    if not isinstance(cond, dict):
        return None
    kind = cond.get("kind")
    if kind not in ALLOWED_CONDITION_KINDS:
        return None
    raw_params = cond.get("params", {})
    if isinstance(raw_params, str):
        try:
            raw_params = json.loads(raw_params)
        except Exception:
            raw_params = {}
    params = raw_params if isinstance(raw_params, dict) else {}
    if "rect" not in params:
        bbox = cond.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            params["rect"] = bbox
    out = {
        "id": str(cond.get("id", f"cond_{idx+1}")),
        "enabled": bool(cond.get("enabled", True)),
        "kind": kind,
        "params": params,
        "weight": float(cond.get("weight", 1.0)),
        "brief": str(cond.get("brief", "")),
    }
    return out


def _conditions_from_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, e in enumerate(elements):
        if not isinstance(e, dict):
            continue
        etype = e.get("type")
        bbox = e.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        if etype == "text_line":
            text = str(e.get("text", "")).strip()
            if not text:
                continue
            out.append(
                {
                    "id": f"elem_text_{i+1}",
                    "enabled": True,
                    "kind": "text_line_contains",
                    "params": {"text": text, "rect": bbox},
                    "weight": 1.0,
                    "brief": str(e.get("brief", "")),
                    "bbox": bbox,
                }
            )
        elif etype == "pattern":
            out.append(
                {
                    "id": f"elem_pattern_{i+1}",
                    "enabled": True,
                    "kind": "region_template",
                    "params": {"rect": bbox, "threshold": 0.8},
                    "weight": 1.0,
                    "brief": str(e.get("brief", "")),
                    "bbox": bbox,
                }
            )
    return out


def _save_llm_raw_debug(session_id: str, attempt: int, text: str) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = FSM_DEBUG_DIR / f"llm_raw_{session_id}_attempt{attempt}_{ts}.txt"
    path.write_text(text, encoding="utf-8")


def _extract_region_templates(frame_rgb, mapper: CoordinateMapper, state_dir: Path, conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = 1
    for cond in conditions:
        c = dict(cond)
        if c.get("kind") == "region_template":
            rect = c.get("params", {}).get("rect")
            if not isinstance(rect, list) or len(rect) != 4:
                continue
            x, y, w, h = mapper.rect_to_real(rect)
            crop = frame_rgb[y : y + h, x : x + w]
            tpl_path = state_dir / f"template_{idx}.png"
            idx += 1
            cv2.imwrite(str(tpl_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            c.setdefault("params", {})["template_path"] = str(tpl_path)
            c["source"] = {
                "screenshot_path": str(state_dir / "screenshot_1.png"),
                "bbox_logical": rect,
                "template_path": str(tpl_path),
            }
        out.append(c)
    return out


def _conditions_passing_current(conditions: list[dict[str, Any]], vision: VisionEngine, frame_rgb, prev_frame_rgb) -> list[dict[str, Any]]:
    passed = []
    for cond in conditions[:MAX_CONDITION_CANDIDATES]:
        if _condition_passed(cond, vision, frame_rgb, prev_frame_rgb):
            passed.append(cond)
    return passed


def _condition_set_distinguishes(conds: list[dict[str, Any]], vision: VisionEngine, state_samples: list[tuple[Path, dict[str, Any]]], current_state_id: str) -> bool:
    if not any(c.get("kind") != "stable" for c in conds):
        return False
    for state_dir, meta in state_samples:
        if str(meta.get("state_id")) == current_state_id:
            continue
        sample_path = state_dir / "screenshot_1.png"
        if not sample_path.exists():
            continue
        bgr = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        other = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        effective = [c for c in conds if c.get("kind") != "stable"]
        if all(_condition_passed(c, vision, other, None) for c in effective):
            return False
    return True


def _pick_minimal_condition_combo(
    passed_conds: list[dict[str, Any]],
    vision: VisionEngine,
    state_samples: list[tuple[Path, dict[str, Any]]],
    current_state_id: str,
) -> list[dict[str, Any]]:
    if len(passed_conds) <= 1:
        return passed_conds[:1]
    max_k = min(MAX_CONDITION_COMBO, len(passed_conds))
    for k in range(MIN_COMBO_SIZE, max_k + 1):
        for combo in combinations(passed_conds, k):
            if _condition_set_distinguishes(list(combo), vision, state_samples, current_state_id):
                return list(combo)
    return passed_conds[:max(MIN_COMBO_SIZE, 1)]


def _ensure_unique_state_dir(slug: str) -> Path:
    base = _slugify(slug)
    d = FSM_TEMPLATES_DIR / base
    n = 1
    while d.exists():
        n += 1
        d = FSM_TEMPLATES_DIR / f"{base}_{n}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _create_state_from_llm(
    llm_payload: dict[str, Any],
    frame_rgb,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    prev_frame_rgb,
) -> tuple[str, Path]:
    state_id = uuid.uuid4().hex
    state_dir = _ensure_unique_state_dir(str(llm_payload.get("slug", "state")))
    _save_frame(state_dir / "screenshot_1.png", frame_rgb)

    raw_conditions = _conditions_from_elements(llm_payload.get("elements", []))
    conditions = _extract_region_templates(frame_rgb, mapper, state_dir, raw_conditions)
    current_passed = _conditions_passing_current(conditions, vision, frame_rgb, prev_frame_rgb)
    minimal = _pick_minimal_condition_combo(current_passed, vision, _iter_state_meta(), state_id)

    state_meta = {
        "schema_version": SCHEMA_VERSION,
        "state_id": state_id,
        "slug": _slugify(str(llm_payload.get("slug", "state"))),
        "display_name": str(llm_payload.get("slug", "state")),
        "description": str(llm_payload.get("page_summary", "")),
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "model_info": {"source": "llm"},
        "elements": llm_payload.get("elements", []),
        "match_conditions": minimal,
        "actions": [
            {
                "action_id": "action_main",
                "version": 1,
                "enabled": True,
                "cooldown_ms": 0,
                "steps": llm_payload.get("actions", []),
            }
        ],
    }
    _save_json(state_dir / "state.json", state_meta)
    return state_id, state_dir


def _append_graph_node(state_id: str, slug: str) -> None:
    graph = _load_json(FSM_GRAPH_PATH)
    graph["nodes"].append({"state_id": state_id, "slug": slug, "enabled": True})
    graph["updated_at"] = _now_iso()
    _save_json(FSM_GRAPH_PATH, graph)


def _append_graph_edge(from_state_id: str | None, action_id: str, to_state_id: str) -> None:
    if not from_state_id:
        return
    graph = _load_json(FSM_GRAPH_PATH)
    for e in graph.get("edges", []):
        if e.get("from_state_id") == from_state_id and e.get("action_id") == action_id and e.get("to_state_id") == to_state_id:
            return
    graph["edges"].append(
        {
            "from_state_id": from_state_id,
            "action_id": action_id,
            "to_state_id": to_state_id,
            "enabled": True,
            "weight": 1.0,
            "created_at": _now_iso(),
        }
    )
    graph["updated_at"] = _now_iso()
    _save_json(FSM_GRAPH_PATH, graph)


def _execute_action_steps(emulator: EmulatorClient, mapper: CoordinateMapper, steps: list[dict[str, Any]]) -> None:
    for idx, step in enumerate(steps, start=1):
        x = int(step.get("x", 500))
        y = int(step.get("y", 500))
        brief = str(step.get("brief", "")).strip()
        rx, ry = mapper.point_to_real(x, y)
        print(f"[fsm][action][click] step={idx} logical=({x},{y}) real=({rx},{ry}) brief={brief}")
        emulator.tap(rx, ry)
        print(f"[fsm][action][wait] sleep={ACTION_CLICK_WAIT_S}s after step={idx}")
        time.sleep(ACTION_CLICK_WAIT_S)


def _request_llm_payload(llm: DoubaoClient, session_id: str, frame_rgb) -> dict[str, Any] | None:
    msg = _build_user_message_from_frame(frame_rgb)
    prompt_fix = (
        "上次JSON无效或不完整。只输出一个完整JSON对象，不要省略字段，不要解释文字。"
    )
    for attempt in range(1, LLM_PARSE_RETRY + 2):
        if attempt == 1:
            resp = llm.chat_with_session(
                session_id=session_id,
                system_prompt=LLM_FSM_PROMPT,
                user_message=msg,
                tools=[],
                tool_choice="none",
            )
        else:
            fix_msg = {
                "role": "user",
                "content": [{"type": "text", "text": prompt_fix}],
            }
            resp = llm.chat_with_session(
                session_id=session_id,
                system_prompt=LLM_FSM_PROMPT,
                user_message=fix_msg,
                tools=[],
                tool_choice="none",
            )
        assistant = resp["choices"][0]["message"]
        text = _normalize_assistant_text(assistant.get("content"))
        _save_llm_raw_debug(session_id, attempt, text)
        print(f"[fsm][llm] raw(attempt={attempt})={text[:600]}")
        payload = _parse_llm_payload(text)
        if payload is not None:
            return payload
    return None


def _execute_state_action_once(
    *,
    emulator: EmulatorClient,
    mapper: CoordinateMapper,
    vision: VisionEngine,
    graph: dict[str, Any],
    metas: list[tuple[Path, dict[str, Any]]],
    runtime: dict[str, Any],
    state_id: str,
    state_dir: Path,
    pre_action_frame,
) -> bool:
    state_meta = _load_json(state_dir / "state.json")
    actions = [a for a in state_meta.get("actions", []) if isinstance(a, dict) and a.get("enabled", True)]
    if not actions:
        print(f"[fsm][reason] action not triggered: state={state_id} has no enabled actions")
        return False

    action = actions[0]
    action_id = str(action.get("action_id", "action_main"))
    steps = action.get("steps", [])
    if not isinstance(steps, list) or not steps:
        print(f"[fsm][reason] action not triggered: state={state_id} action={action_id} has empty steps")
        return False

    for attempt in range(1, MAX_RETRY_PER_ACTION + 1):
        print(f"[fsm][action] state={state_id} action={action_id} attempt={attempt}")
        _execute_action_steps(emulator, mapper, steps)
        time.sleep(0.8)
        post = emulator.screenshot(prefer_png=True)
        post_matches = [_match_state(meta, sdir, vision, post, pre_action_frame) for sdir, meta in metas]
        post_best = _select_best_match(post_matches, _reachable_targets(graph, state_id))
        if post_best is not None and post_best.state_id != state_id and post_best.score > 0.0:
            print(f"[fsm][transition] {state_id} --{action_id}--> {post_best.state_id}")
            _append_graph_edge(state_id, action_id, post_best.state_id)
            runtime["last_state_id"] = post_best.state_id
            runtime["consecutive_failures"] = 0
            _save_runtime(runtime)
            return True
        time.sleep(0.6)
    return False


def run_agent_loop_fsm(
    *,
    session_id: str,
    serial: str | None = None,
    adb_path: str | None = None,
    interval_s: float = 5.0,
) -> None:
    _ensure_fsm_resources()
    target_serial = resolve_target_serial(serial=serial, adb_path=adb_path, auto_connect=True)
    emulator = EmulatorClient(serial=target_serial, adb_path=adb_path)
    llm = DoubaoClient()
    mapper = CoordinateMapper(logical_w=1000, logical_h=1000, real_w=1280, real_h=720)
    vision = VisionEngine(mapper=mapper)

    prev_frame = None
    runtime = _load_runtime()
    runtime["last_state_id"] = None
    runtime["last_replan_at"] = 0.0
    runtime["consecutive_failures"] = 0
    runtime["run_id"] = uuid.uuid4().hex[:8]
    _save_runtime(runtime)
    llm_session_id = f"{session_id}-fsm-{runtime['run_id']}"
    print(f"[fsm] started serial={target_serial} session={llm_session_id}")

    while True:
        frame = emulator.screenshot(prefer_png=True)
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        graph = _load_json(FSM_GRAPH_PATH)
        metas = _iter_state_meta()
        matches = [_match_state(meta, sdir, vision, frame, prev_frame) for sdir, meta in metas]
        reachable = _reachable_targets(graph, runtime.get("last_state_id"))

        debug_line = [f"{m.state_id}:{m.score:.2f}" for m in sorted(matches, key=lambda x: -x.score)]
        reachable_dbg = "ALL(cold-start)" if reachable is None else sorted(reachable)
        print(f"[fsm][match] candidates={debug_line} reachable={reachable_dbg}")
        hit_states = [m for m in matches if m.score > 0.0]
        if len(hit_states) > 1:
            hit_debug = [f"{m.state_id} score={m.score:.2f} pass={m.passed_ids} fail={m.failed_ids}" for m in sorted(hit_states, key=lambda x: -x.score)]
            print(f"[fsm][match][multi-hit] {hit_debug}")

        best = _select_best_match(matches, reachable)
        if best is None or best.score <= 0.0:
            positive_matches = [m for m in matches if m.score > 0.0]
            if positive_matches and reachable == set():
                print(
                    "[fsm][reason] action not triggered: last_state has no reachable edges, "
                    "strict policy treats as unknown state"
                )
            elif positive_matches:
                pm = [f"{m.state_id}:{m.score:.2f}" for m in sorted(positive_matches, key=lambda x: -x.score)]
                print(f"[fsm][reason] action not triggered: matches exist but filtered by reachable={sorted(reachable)} positives={pm}")
            else:
                print("[fsm][reason] action not triggered: no state reached positive score")
            print("[fsm] unknown state, requesting llm")
            payload = _request_llm_payload(llm, llm_session_id, frame)
            if payload is None:
                print("[fsm][llm] invalid payload")
                time.sleep(interval_s)
                prev_frame = frame
                continue
            new_state_id, new_state_dir = _create_state_from_llm(payload, frame, mapper, vision, prev_frame)
            _append_graph_node(new_state_id, str(payload.get("slug", "state")))
            print(f"[fsm] new_state state_id={new_state_id} dir={new_state_dir}")
            runtime["last_state_id"] = new_state_id
            runtime["consecutive_failures"] = 0
            _save_runtime(runtime)
            print(f"[fsm] immediate action execution for new_state={new_state_id}")
            transitioned = _execute_state_action_once(
                emulator=emulator,
                mapper=mapper,
                vision=vision,
                graph=graph,
                metas=_iter_state_meta(),
                runtime=runtime,
                state_id=new_state_id,
                state_dir=new_state_dir,
                pre_action_frame=frame,
            )
            if not transitioned:
                print(f"[fsm][reason] new_state action did not reach a reachable target state: state={new_state_id}")
            prev_frame = frame
            time.sleep(interval_s)
            continue

        state_meta = _load_json(best.state_dir / "state.json")
        actions = [a for a in state_meta.get("actions", []) if isinstance(a, dict) and a.get("enabled", True)]
        if not actions:
            print(f"[fsm][reason] action not triggered: state={best.state_id} has no enabled actions")
            runtime["last_state_id"] = best.state_id
            _save_runtime(runtime)
            prev_frame = frame
            time.sleep(interval_s)
            continue

        action = actions[0]
        action_id = str(action.get("action_id", "action_main"))
        steps = action.get("steps", [])
        if not isinstance(steps, list) or not steps:
            print(f"[fsm][reason] action not triggered: state={best.state_id} action={action_id} has empty steps")
            runtime["last_state_id"] = best.state_id
            _save_runtime(runtime)
            prev_frame = frame
            time.sleep(interval_s)
            continue
        success = False
        for attempt in range(1, MAX_RETRY_PER_ACTION + 1):
            print(f"[fsm][action] state={best.state_id} action={action_id} attempt={attempt}")
            _execute_action_steps(emulator, mapper, steps)
            time.sleep(0.8)
            post = emulator.screenshot(prefer_png=True)
            post_matches = [_match_state(meta, sdir, vision, post, frame) for sdir, meta in metas]
            post_best = _select_best_match(post_matches, _reachable_targets(graph, best.state_id))
            if post_best is not None and post_best.state_id != best.state_id and post_best.score > 0.0:
                print(f"[fsm][transition] {best.state_id} --{action_id}--> {post_best.state_id}")
                _append_graph_edge(best.state_id, action_id, post_best.state_id)
                runtime["last_state_id"] = post_best.state_id
                runtime["consecutive_failures"] = 0
                _save_runtime(runtime)
                prev_frame = post
                success = True
                break
            time.sleep(0.6)

        if success:
            time.sleep(interval_s)
            continue

        runtime["consecutive_failures"] = int(runtime.get("consecutive_failures", 0)) + 1
        now = time.time()
        last_replan = float(runtime.get("last_replan_at", 0.0))
        if now - last_replan >= REPLAN_COOLDOWN_S:
            print("[fsm][repair] action failed continuously, requesting llm repair")
            payload = _request_llm_payload(llm, llm_session_id, frame)
            if payload is not None:
                screenshot_idx = 1
                while (best.state_dir / f"screenshot_{screenshot_idx}.png").exists():
                    screenshot_idx += 1
                _save_frame(best.state_dir / f"screenshot_{screenshot_idx}.png", frame)
                new_actions = payload.get("actions", [])
                if isinstance(new_actions, list) and new_actions:
                    action["version"] = int(action.get("version", 1)) + 1
                    action["steps"] = new_actions
                    state_meta["updated_at"] = _now_iso()
                    _save_json(best.state_dir / "state.json", state_meta)
                runtime["last_replan_at"] = now
                runtime["consecutive_failures"] = 0
            _save_runtime(runtime)

        prev_frame = frame
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="FSM driven emulator agent loop")
    parser.add_argument("--session-id", default=f"session-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--adb-path", default=None)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    run_agent_loop_fsm(
        session_id=args.session_id,
        serial=args.serial,
        adb_path=args.adb_path,
        interval_s=args.interval,
    )


if __name__ == "__main__":
    main()
