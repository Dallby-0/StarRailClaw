from __future__ import annotations

from pathlib import Path

FSM_DIR = Path("StateMachineResources")
FSM_TEMPLATES_DIR = FSM_DIR / "templates"
FSM_GRAPH_PATH = FSM_DIR / "state_graph.json"
FSM_SCHEMA_PATH = FSM_DIR / "llm_protocol_schema.json"
FSM_RUNTIME_PATH = FSM_DIR / "runtime_state.json"
FSM_DEBUG_DIR = FSM_DIR / "debug"
EXPERIENCE_PATH = FSM_DIR / "experience.md"
TASK_SUMMARY_PATH = FSM_DIR / "task_summary.md"
EXPERIENCE_WRITE_ENABLED = False

SCHEMA_VERSION = "2.0.0"
ACTION_CLICK_WAIT_S = 1.0
UNKNOWN_STABILITY_DIFF_THRESHOLD = 0.1
UNKNOWN_STABILITY_SAMPLE_INTERVAL_S = 0.5
UNKNOWN_STABILITY_RETRY_WAIT_S = 5.0
LLM_PARSE_RETRY = 2
PRESET_WAIT_TIMEOUT_S = 600.0
LOCAL_FLOW_MAX_STEPS = 12
LOCAL_FLOW_NO_PROGRESS_LIMIT = 2
SCREEN_CHANGE_DIFF_THRESHOLD = 0.01
SAME_EXTERNAL_STATE_REPAIR_THRESHOLD = 5

SOFT_LIMIT = 5
MID_LIMIT = 10
HARD_LIMIT = 15
FORCE_RESET_AT = 16

DISCRIMINATION_SCORE = {"high": 4, "mid": 2, "low": 1}
STABILITY_SCORE = {"high": 3, "mid": 2, "low": 1}
MATCH_DISCRIMINATION_TARGET = 4

LLM_FSM_PROMPT_BASE = """你是视觉驱动游戏自动化的状态标注器和动作规划器。

当前任务目标由任务工作区中的 task_summary.md 提供；如果没有任务概要，则仅按当前画面推进通用游戏自动化流程。

你将收到一张游戏截图。你的任务：
1) 识别并输出用于区分该页面的关键信息，按匹配价值排序：优先高 stability 且高 discrimination 的 pattern，其次是稳定 text_line。
2) 在同一次响应中输出最多两个 bootstrap operation，用于低成本推进当前页面。

强约束：
- 必须输出严格 JSON 对象，字段必须符合约定。
- elements: 每项必须有 type。
  - elements 应优先服务于“页面匹配”，不是描述画面。优先输出最能稳定区分当前页面类型的元素。
  - 请尽量标注当前画面中潜在可用于匹配的 OCR 区域和模板区域；即使稳定性或区分度不高，也可以输出，但必须如实把 stability/discrimination 标为 mid 或 low，不要为了让元素看起来有用而虚高评分。
  - 低 stability 或低 discrimination 的元素仍有诊断、消歧、后续修复价值；不要因为它们不是最佳匹配条件就完全省略。
  - 除非当前画面几乎没有稳定视觉图案，否则 elements 中至少应包含 1 个 type=pattern。
  - 如果页面上同时存在文字和稳定视觉图案，应优先选择“1 个高区分度 pattern + 1 个稳定 text_line”的组合，而不是只输出 text_line。
  - pattern 用于抵抗 OCR 失败和区分页布局；不要因为页面上有文字就省略 pattern。
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
    - pattern 是首选页面区分信号之一；只要画面中存在稳定且有区分度的视觉区域，就应该输出 pattern。
    - pattern 的 bbox 区域将直接作为模板图像用于匹配，请尽量紧贴图案边缘裁剪，避免包含大面积背景，否则背景变化会导致匹配失败。
    - pattern/模板只能来自看起来稳定的 2D UI 元素或控件状态，不要从 3D 场景中的物体、角色、怪物、地面、墙面、背景、光效、可移动目标或摄像机视角相关区域取模板。
    - 如果当前画面主要是 3D 场景，且没有明显稳定 UI 图标/控件可作为 pattern，应优先使用稳定 text_line 或省略 pattern，不要为了满足 pattern 数量而裁剪场景物体。
    - 优先选择高稳定、高区分度的小区域：固定图标、选中态页签、弹窗固定装饰、页面专属徽标、固定按钮图标、固定控件边缘/形状。
    - 避免选择会变化的区域：角色立绘、奖励物品、随机名称、数值、进度、列表内容、动态特效、大面积背景、3D 场景内容。
    - 如果一个图案内部包含可独立识别的子元素（例如按钮上的图标、页签标识、固定角标等），且该子元素具有稳定的视觉特征，优先选择该子元素的 bbox，而不是整个父级图案。这能提高匹配的鲁棒性。
    - 如果 pattern 很可能能单独区分页面类型，应给 discrimination=high；如果同类页面中基本不变化，应给 stability=high。
  - 每项必须额外给 stability 和 discrimination。
    - stability 表示该元素在同类型页面中不变化的程度，只能是 high/mid/low。
    - discrimination 表示该元素能区分当前页面的能力，只能是 high/mid/low。
    - 具体实例名称、对象名称、奖励名称、长正文、数值、编号、进度、计数通常 stability=low 或 mid，不要高估。
    - 页面标题、固定图标、固定按钮、固定交互控件通常更稳定。
- slug: 英文小写+下划线，简短可读。
- possible_page_type: 如果当前页面可能属于已知页面类型，输出该类型英文名；否则输出 "none"。不要把具体实例名称当作页面类型。
- bootstrap_operations: 数组。每项表示一种操作语义及其初始渐进策略，而不是无条件 state action。
  - operation 必须是稳定的短语义名，例如 dismiss_overlay、confirm、advance、select_candidate。
  - intent_scope 只能是 intent_invariant 或 intent_specific。只有纯提示/透明阻塞弹窗才使用 intent_invariant。
  - intent_effect 只能是 preserve、advance、complete 或 none。
  - safety 只能是 low_risk、reversible、commit 或 destructive。commit/destructive 不可作为无条件默认操作。
  - 最多输出两个 steps；每一步都必须提供 expected_after，runtime 会在每步后重新截图，禁止设计无条件连点宏。
  - 初始策略优先 resolver.type=fixed_point，坐标为 1000x1000 逻辑坐标。
  - 只有明显需要预置动作时才使用 resolver.type=run_preset。
  - 若 active_intent 存在，intent_routes 应说明哪些 intent kind/phase 映射到这个 operation。
  - 预置动作：resolver={"type":"run_preset","name":"wait_till_combat_end"}
    - 预置动作 wait_till_combat_end 若当前是战斗状态，则调用该动作，会挂起至战斗结束，此时自动战斗 
    - 预制动作 find_and_interact_with_next_object 会在当前场景寻找并移动至下一个可交互对象并与其交互,只要是在场景中需要与物体交互，都调用这个，包括与前方怪物战斗、与NPC、机关、门互动等
- 注意在3D场景中不要尝试点击物体触发交互，这没有任何效果，若发现需要在3D场景中需要与物体交互，请使用预置动作 find_and_interact_with_next_object。

字段、枚举、必填项和嵌套结构由 Responses API 的 JSON Schema 提供，不要输出 schema 之外的字段。
"""
