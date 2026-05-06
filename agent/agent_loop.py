from __future__ import annotations

import argparse
import base64
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import requests

from agent.behavior_tree.coord_mapper import CoordinateMapper
from agent.behavior_tree.engine import BehaviorTreeEngine, NodeStatus, TickContext
from agent.behavior_tree.growth import GrowthManager
from agent.behavior_tree.storage import ensure_main_tree
from agent.llm_client import DoubaoClient
from sr_tools.adb import resolve_target_serial
from sr_tools.emulator import EmulatorClient

DEFAULT_SYSTEM_PROMPT = (
    "# 目标\n"
    "你是一个安卓agent，拥有在安卓模拟器上模拟点击的能力，正在尝试通关模拟宇宙-差分宇宙。\n"
    "你当前只在行为树兜底阶段工作：当已有脚本无法覆盖当前画面时，给出可复用的场景定义和动作序列。\n"
)

LLM_GROWTH_PROMPT = """你是一个视觉驱动游戏自动化脚本架构师。

## 任务背景（可替换）
{你正在控制一个游戏角色，目标是在当前游戏模式下自动推进流程，尽可能高效地完成关卡。}

你将收到一张来自当前游戏的实时画面截图。请基于画面内容判断当前所处的场景，并给出最优的操作序列。

## 核心约束
- 输出严格 JSON 对象，仅包含两个字段：condition 和 actions。
- condition 必须是完整的 Condition 节点，用于识别当前场景。
- actions 必须是完整的 Action、Sequence 或 Selector 节点，用于执行操作。
- 所有参数必须放入 params 字段，禁止简写或挂在节点外层。
- 不要使用 markdown 代码块，不要输出任何额外文字。

## 节点类型（仅限以下）
组合节点：Sequence、Selector、Repeat
条件节点：Condition
动作节点：Action
引用节点：SubTree

## Condition 设计原则（关键）
- **condition 的唯一目标：精准确认当前是哪一个页面/场景。**
- 必须选择该页面**独有的、标志性的视觉元素**进行 template_match。
- 优先匹配固定图标、独特背景、专属边框等不会与其他页面混淆的元素。
- 不要选择通用按钮（如“确认”、“返回”）或变化频繁的文字作为主条件——它们容易在不同页面重复出现。
- 若单一模板不足以区分当前页面，使用 AND 组合多个特征。
- 配合 stable 条件，确保画面已完全渲染、无转场动画。

## Condition check 允许的函数
- template_match：在 rect 区域内匹配模板图片
- stable：画面稳定检测（帧间差异小于阈值）
- AND / OR / NOT：逻辑组合
  · AND 和 OR 的参数为 conditions（Condition 对象数组）
  · NOT 的参数为 condition（单个 Condition 对象）

## Action do 允许的函数
- click_template：找到模板并点击其中心
- click_pos：点击指定坐标
- wait：等待固定毫秒数
- wait_stable：等待画面稳定（须指定 timeout）
- wait_for_template：循环等待模板出现（须指定 timeout）
- wait_for_template_disappear：循环等待模板消失（须指定 timeout）

## 坐标与区域
- 统一使用 1000x1000 逻辑坐标空间，左上角为 (0,0)，右下角为 (1000,1000)。
- rect 格式为 [x1, y1, x2, y2]（左上角横坐标、左上角纵坐标、右下角横坐标、右下角纵坐标）。
- 必须基于画面中 UI 元素的实际位置精确估算 rect，不允许猜测。

## 画面稳定性处理
- 游戏画面可能处于转场动画、加载中或动态特效等不稳定状态。
- 在 condition 中，应结合 stable 条件确认画面已稳定。
- 在 actions 中，点击前使用 wait_stable 等待稳定，点击后使用 wait_for_template_disappear 并设置 timeout 等待操作生效。

## 典型操作流程模式
推荐顺序：
1. wait_stable（等待界面稳定）
2. click_template（点击目标元素）
3. wait_for_template_disappear（等待该元素消失，确认流程推进）

## 容错与回退
- 等待类动作必须设置 timeout，超时后应有备选方案。
- 使用 Selector 节点实现 fallback：优先尝试精确模板点击，若失败则回退到 click_pos 点击固定区域，或进行短暂 wait 后重试。

## 注释要求
- 每个节点可包含可选的 comment 字段（字符串）。
- 强烈建议为关键节点添加简短的中文注释，说明节点用途。

## 节点格式参考

Condition 示例：
{"type":"Condition","comment":"确认战斗界面标志图标存在且画面稳定","check":"AND","params":{"conditions":[{"type":"Condition","check":"template_match","params":{"template":"combat_indicator","rect":[20,20,80,60],"threshold":0.8}},{"type":"Condition","check":"stable","params":{"rect":[0,0,1000,1000],"diff_threshold":0.01}}]}}

Action 示例：
{"type":"Action","comment":"点击自动战斗按钮","do":"click_template","params":{"template":"btn_auto","rect":[850,620,980,700],"threshold":0.8}}

Selector 示例（带 fallback）：
{"type":"Selector","comment":"优先点击模板，若失败则点击固定坐标","children":[{"type":"Action","do":"click_template","params":{"template":"btn_main","rect":[400,600,600,680],"threshold":0.8}},{"type":"Action","do":"click_pos","params":{"pos":[500,640]}}]}

## 最终输出格式
{"condition":{ /* Condition 节点 */ },"actions":{ /* Action / Sequence / Selector 节点 */ }}"""

ANSWER_IMAGE_DIR = Path("debug") / "answers"
TREE_DIR = Path("BehaviorTree")
TREE_PATH = TREE_DIR / "main.json"
TEMPLATE_DIR = TREE_DIR / "templates"
ENABLE_REFLECT = False


def _save_answer_image_data_url(data_url: str, out_dir: Path) -> str | None:
    prefix = "data:image/"
    if not data_url.startswith(prefix):
        return None
    try:
        header, b64 = data_url.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header:
        return None
    ext = "png"
    mime = header[len("data:") :].split(";")[0]
    if "/" in mime:
        ext = mime.split("/", 1)[1] or "png"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"llm_annotated.{ext}"
    raw = base64.b64decode(b64)
    out.write_bytes(raw)
    return str(out)


def _save_answer_image_http_url(image_url: str, out_dir: Path) -> str | None:
    if not (image_url.startswith("http://") or image_url.startswith("https://")):
        return None
    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
    except Exception:
        return None
    ctype = (resp.headers.get("Content-Type") or "").lower()
    ext = "png"
    if "jpeg" in ctype or "jpg" in ctype:
        ext = "jpg"
    elif "webp" in ctype:
        ext = "webp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"llm_annotated.{ext}"
    out.write_bytes(resp.content)
    return str(out)


def _save_answer_image_base64(raw_b64: str, out_dir: Path, ext: str = "png") -> str | None:
    if not raw_b64:
        return None
    try:
        raw = base64.b64decode(raw_b64)
    except Exception:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"llm_annotated.{ext}"
    out.write_bytes(raw)
    return str(out)


def _extract_and_save_llm_image(assistant_msg: dict, out_dir: Path) -> str | None:
    content = assistant_msg.get("content")
    if not isinstance(content, list):
        return None
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "image_url":
            image_obj = part.get("image_url")
            if isinstance(image_obj, dict):
                url = image_obj.get("url", "")
            elif isinstance(image_obj, str):
                url = image_obj
            else:
                url = ""
            if isinstance(url, str) and url:
                saved = _save_answer_image_data_url(url, out_dir) or _save_answer_image_http_url(url, out_dir)
                if saved:
                    return saved
        if ptype in {"output_image", "image"}:
            b64 = part.get("b64_json") or part.get("image_base64") or ""
            if isinstance(b64, str) and b64:
                saved = _save_answer_image_base64(b64, out_dir)
                if saved:
                    return saved
    return None


def _build_user_message_from_frame(image_rgb) -> dict:
    data_url = DoubaoClient.encode_image_to_data_url(image_rgb)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": ""},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }


def _normalize_assistant_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in {"text", "output_text"}]
        return "".join(parts).strip()
    return ""


def _truncate_text(text: str, max_len: int = 1200) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...<truncated>"


def run_agent_loop(
    *,
    system_prompt: str,
    session_id: str,
    serial: str | None = None,
    adb_path: str | None = None,
    interval_s: float = 1.0,
) -> None:
    ensure_main_tree(TREE_PATH)
    target_serial = resolve_target_serial(serial=serial, adb_path=adb_path, auto_connect=True)
    emulator = EmulatorClient(serial=target_serial, adb_path=adb_path)
    llm = DoubaoClient()
    mapper = CoordinateMapper(logical_w=1000, logical_h=1000, real_w=1280, real_h=720)
    growth = GrowthManager(tree_path=TREE_PATH, template_dir=TEMPLATE_DIR)

    runtime_state: dict[str, object] = {}
    prev_frame = None
    latest_round_dir: Path | None = None

    def llm_decide_cb(ctx: TickContext) -> NodeStatus:
        nonlocal latest_round_dir
        user_message = _build_user_message_from_frame(ctx.frame_rgb)
        try:
            resp = llm.chat_with_session(
                session_id=f"{session_id}-growth",
                system_prompt=LLM_GROWTH_PROMPT,
                user_message=user_message,
                tools=[],
                tool_choice="none",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[llm_decide] call failed: {exc}")
            return NodeStatus.FAILURE

        assistant_msg = resp["choices"][0]["message"]
        payload_text = _normalize_assistant_text(assistant_msg.get("content"))
        print(f"[llm_decide] raw_output={_truncate_text(payload_text)}")
        if latest_round_dir is not None:
            saved = _extract_and_save_llm_image(assistant_msg, latest_round_dir)
            if saved:
                print(f"[llm_decide] annotated saved: {saved}")

        parsed = growth.parse_llm_growth_payload(payload_text)
        if parsed is None:
            print("[llm_decide] invalid JSON payload")
            return NodeStatus.FAILURE

        condition, actions = parsed
        print(f"[llm_decide] parsed_condition={json.dumps(condition, ensure_ascii=False)}")
        print(f"[llm_decide] parsed_actions={json.dumps(actions, ensure_ascii=False)}")
        condition = growth.extract_and_replace_templates(ctx.frame_rgb, condition, engine.vision)
        print(f"[llm_decide] condition_after_template_extract={json.dumps(condition, ensure_ascii=False)}")

        candidate = {"type": "Sequence", "children": [condition, actions]}
        dry_ctx = TickContext(
            frame_rgb=ctx.frame_rgb,
            prev_frame_rgb=ctx.prev_frame_rgb,
            now_monotonic=ctx.now_monotonic,
            state={},
        )
        dry_status = engine.tick(candidate, dry_ctx, node_path="dryrun")
        print(f"[llm_decide] dry_run_status={dry_status.value}")
        if dry_status == NodeStatus.FAILURE:
            print("[llm_decide] dry-run failed, skip solidify")
            return NodeStatus.FAILURE

        tree = growth.load_tree()
        inserted = growth.insert_experience(tree, condition, actions)
        if inserted:
            growth.save_tree(tree)
            print("[llm_decide] experience solidified into main.json")
        else:
            print("[llm_decide] duplicate experience, skipped")
        return NodeStatus.SUCCESS

    engine = BehaviorTreeEngine(
        emulator=emulator,
        tree_dir=TREE_DIR,
        mapper=mapper,
        llm_decide_cb=llm_decide_cb,
    )

    if llm.reasoning is None and (llm.reasoning_effort is None or str(llm.reasoning_effort).strip() == ""):
        print("[agent] reasoning=OFF (no reasoning payload)")
    else:
        print(f"[agent] reasoning=ON reasoning={llm.reasoning} reasoning_effort={llm.reasoning_effort}")
    print(f"[agent] started, serial={target_serial}, session_id={session_id}")
    ANSWER_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    round_idx = 1
    while True:
        round_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        round_dir = ANSWER_IMAGE_DIR / f"round{round_idx}_{round_ts}"
        round_dir.mkdir(parents=True, exist_ok=True)
        latest_round_dir = round_dir

        frame = emulator.screenshot(prefer_png=True)
        fh, fw = frame.shape[:2]
        if (fw, fh) != (1280, 720):
            print(f"[round {round_idx}] resize screenshot {fw}x{fh} -> 1280x720")
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        tree = growth.load_tree()
        tick_ctx = TickContext(
            frame_rgb=frame,
            prev_frame_rgb=prev_frame,
            now_monotonic=time.monotonic(),
            state=runtime_state,
        )

        status = engine.tick(tree, tick_ctx)
        growth.record_result(status == NodeStatus.SUCCESS)
        print(f"[round {round_idx}] tree_status={status.value}")

        if ENABLE_REFLECT and growth.need_reflect(tree):
            print("[reflect] trigger reached (fail streak or node threshold), pending architect step")

        prev_frame = frame.copy()
        time.sleep(interval_s)
        round_idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior tree driven emulator agent loop")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for task policy")
    parser.add_argument("--session-id", default=f"session-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--adb-path", default=None)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    run_agent_loop(
        system_prompt=args.system,
        session_id=args.session_id,
        serial=args.serial,
        adb_path=args.adb_path,
        interval_s=args.interval,
    )


if __name__ == "__main__":
    main()
