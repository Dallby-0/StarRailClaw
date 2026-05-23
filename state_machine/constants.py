from __future__ import annotations

from pathlib import Path

FSM_DIR = Path("StateMachineResources")
FSM_TEMPLATES_DIR = FSM_DIR / "templates"
FSM_GRAPH_PATH = FSM_DIR / "state_graph.json"
FSM_SCHEMA_PATH = FSM_DIR / "llm_protocol_schema.json"
FSM_RUNTIME_PATH = FSM_DIR / "runtime_state.json"
FSM_DEBUG_DIR = FSM_DIR / "debug"
EXPERIENCE_PATH = FSM_DIR / "experience.md"

SCHEMA_VERSION = "1.1.0"
ACTION_CLICK_WAIT_S = 5.0
LLM_PARSE_RETRY = 2
MAX_RETRY_PER_ACTION = 2
PRESET_WAIT_TIMEOUT_S = 600.0

SOFT_LIMIT = 5
MID_LIMIT = 10
HARD_LIMIT = 15
FORCE_RESET_AT = 16

REPAIR_EFFORTS = ["low", "mid", "high"]
DISCRIMINATION_SCORE = {"high": 4, "mid": 2, "low": 1}
STABILITY_SCORE = {"high": 3, "mid": 2, "low": 1}
MATCH_DISCRIMINATION_TARGET = 4

LLM_FSM_PROMPT_BASE = """你是视觉驱动游戏自动化的状态标注器和动作规划器,当前任务是通关崩坏星穹铁道差分宇宙。

你将收到一张游戏截图。你的任务：
1) 识别并输出用于区分该页面的关键信息，按重要性排序。
2) 输出推进流程的动作序列（1000x1000逻辑坐标）。

强约束：
- 必须输出严格 JSON 对象，字段必须符合约定。
- elements: 每项必须有 type。
  - type=text_line: 必须给 text, bbox[x1,y1,x2,y2], brief。
    - runtime 对 text_line 使用单行 OCR；一个 bbox 必须只框住一行文本，不要框多行、段落或包含上下相邻文字。
    - text 字段会作为该单行 OCR 结果的 contains 子串匹配，不是精确等于，也不是正则。
    - 因此 text 应该是该单行里稳定、短、可区分的子串；不要直接复制长句或跨行内容。
    - 当一行文字同时包含可变实例信息和稳定页面类型词时，优先只抽取稳定页面类型词作为 text。
      例如遇到“名称/编号/进度 + 类型词”的单行结构时，应倾向于丢弃名称、编号、进度、计数、序号、轮次、等级、数量等可变部分，只保留能代表页面类别或交互类别的稳定词。
    - 如果不确定某个子串是否稳定，应降低 stability；如果某个子串过于通用，应降低 discrimination。
    - 如果目标信息分布在多行，请拆成多个 text_line 元素，每个元素只对应一行。
    - 优先选择页面标题、固定标签、固定按钮文字、固定栏目名、稳定类别词；避免选择长正文、叙事描述、实例名称、对象名称、奖励名称、数值、编号、进度、计数等容易变化的内容。
  - type=pattern: 必须给 bbox[x1,y1,x2,y2], brief。
    - pattern 的 bbox 区域将直接作为模板图像用于匹配，请尽量紧贴图案边缘裁剪，避免包含大面积背景，否则背景变化会导致匹配失败。
    - 如果一个图案内部包含可独立识别的子元素（例如按钮上的文字、图标、数字等），且该子元素具有稳定的视觉特征，优先选择该子元素的 bbox，而不是整个父级图案。这能提高匹配的鲁棒性
  - 每项必须额外给 stability 和 discrimination。
    - stability 表示该元素在同类型页面中不变化的程度，只能是 high/mid/low。
    - discrimination 表示该元素能区分当前页面的能力，只能是 high/mid/low。
    - 具体实例名称、对象名称、奖励名称、长正文、数值、编号、进度、计数通常 stability=low 或 mid，不要高估。
    - 页面标题、固定图标、固定按钮、固定交互控件通常更稳定。
- slug: 英文小写+下划线，简短可读。
- possible_page_type: 如果当前页面可能属于已知页面类型，输出该类型英文名；否则输出 "none"。不要把具体实例名称当作页面类型。
- actions: 数组，每项仅允许两类：
  - 点击：{"type":"click","x":整数,"y":整数,"brief":"..."}
  - 预置动作：{"type":"run_preset","name":"wait_till_combat_end","brief":"..."}
    - 预置动作 wait_till_combat_end 若当前是战斗状态，则调用该动作，会挂起至战斗结束，此时自动战斗 
    - 预制动作 find_and_interact_with_next_object 会在当前场景寻找并移动至下一个可交互对象并与其交互,只要是在场景中需要与物体交互，都调用这个，包括与前方怪物战斗、与NPC、机关、门互动等
- 注意在3D场景中不要尝试点击物体触发交互，这没有任何效果，若发现需要在3D场景中需要与物体交互，请使用预置动作 find_and_interact_with_next_object。

输出字段：
{
  "page_summary": "...",
  "slug": "...",
  "possible_page_type": "none 或 已知/候选页面类型英文名",
  "elements": [...],
  "actions": [...]
}
"""
