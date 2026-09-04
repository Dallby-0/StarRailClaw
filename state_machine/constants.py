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

SCHEMA_VERSION = "3.0.0"
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
2) 在同一次响应中输出最多两个 bootstrap operation；每个 operation 优先只给当前画面上最简单的直接动作。

强约束：
- 必须输出严格 JSON 对象，字段必须符合约定。
- elements: 每项必须有 type。
  - 每项还必须标注 role：identity=跨该交互表面各步骤都稳定存在、可单独确认页面身份；identity_support=稳定但过于通用、只能与其他锚点组合；interaction=可操作控件；instance=仅当前实例存在的名称、数值或正文；diagnostic=仅供解释和排错。
  - 只有 identity/identity_support 会进入全局 state matcher。interaction 只供 page handler 定位，instance/diagnostic 不得成为 state 身份。不要把具体事件名、选项文字、当前卡面或步骤按钮标成 identity。
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
- scene_mode: 必须判断为 ui_2d、scene_3d 或 unknown。scene_3d 表示当前主要是可自由移动/寻找交互物体的 3D 场景；仅有 3D 背景但当前有明确稳定 UI 控件时仍可使用 ui_2d。
- page_family: 英文小写+下划线，表示可共享同类操作经验的稳定页面族；不知道时使用与 slug 相同的值。
- surface_relation: 若提供了 previous_surface，上下两图仍是同一稳定交互表面、只是页内步骤不同，必须输出 same_surface_step；同族但应独立处理的阻塞层/结果层输出 same_family_new_surface；否则输出 different_surface；没有前图时输出 uncertain。
- common_identity: 仅列出前后步骤共同保留的稳定身份元素，禁止填写具体事件名、选项或仅当前步骤存在的按钮。
- bootstrap_operations: 数组。每项表示一种稳定操作语义及其 reactive providers，而不是无条件宏。
  - operation 使用简短、稳定、领域无关的语义名；同一 page_family 的连续页内步骤复用同一个完整目标 operation。
  - 每个 provider 表达一个可独立观察和结算的动作。明显的短链可在一次输出中给出多个 provider，并用 successors 表达短期后继先验。
  - runtime 在每个动作后重新截图和评估，不会把 providers 当作无条件连点序列。
  - locators 按顺序 fallback。point 必须显式声明 coordinate_space=logical，所有点和矩形均为 1000x1000 逻辑坐标。
  - hints 是软排序证据，通常满足 > 无法判断/无 hint > 不满足；hint 失败不能直接证明动作无效。
  - 文本识别成本高，text hint/locator 应限制在较小区域；只需要文字行数时优先使用区域 line_count hint。
  - effect_hints 只描述动作后的可观测假设；没有稳定可观测效果时保持空数组。
  - 只有前驱动作后才会出现的特征放入 deferred_hints，并通过 materialize_after 引用前驱 provider_id。
  - 禁止在 core provider 中写入特定业务对象、固定选项数量或其他只适用于单一场景的规则。
  - intent_scope 只能是 intent_invariant 或 intent_specific。只有纯提示/透明阻塞弹窗才使用 intent_invariant。
  - intent_effect 只能是 preserve、advance、complete 或 none。
  - 初见页面可优先使用 point locator；同时存在稳定视觉定位方式时，可将其放在 point 之前作为更通用的 locator。
  - 只有明显需要已注册预置动作时才使用 run_preset locator。
  - 若 active_intent 存在，intent_routes 应说明哪些 intent kind/phase 映射到这个 operation。
  - run_preset.name 只能引用 runtime 已注册的预置动作名称，不要自行创造领域规则。
  - scene_mode=scene_3d 时，bootstrap operation 必须提供 run_preset locator，name 必须为 find_and_interact_with_next_object；不要输出固定坐标、模板或 OCR 点击来操作场景物体。scene_mode=ui_2d 使用 reactive providers；unknown 时保持谨慎，不要假设 3D 物体可点击。

字段、枚举、必填项和嵌套结构由 Responses API 的 JSON Schema 提供，不要输出 schema 之外的字段。
"""
