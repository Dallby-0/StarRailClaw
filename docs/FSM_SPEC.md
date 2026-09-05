# FSM Spec

> 注意：本文的 action、preset 和 page-handler 章节描述已删除的旧实现，不再是运行时契约。当前状态使用顶层 `execution` 在注册工具与仅支持 2D 的 `reactive_handler.v2` 之间分发；权威规范见 `REACTIVE_HANDLER_V2.md`。旧状态文件不会迁移。

本文档描述当前 `state_machine/` 实现所支持的 FSM 运行时规范。范围只包含状态机核心：资源文件、状态模型、匹配、动作、状态图、状态学习、合并、修复、同页局部流、预置动作和日志。GUI、具体 LLM client、ADB/模拟器底层实现不在本文档范围内。

## 1. 目标和边界

FSM 负责把游戏截图映射为已知状态，执行该状态的动作，并根据执行后的截图推进状态图。当遇到未知页面时，FSM 可请求外部视觉/文本推理服务生成状态描述；当动作无效时，FSM 可请求外部推理服务修复动作；当同一页面内发生局部推进时，FSM 可学习并复用 page-local handler edge。

核心实现入口：

- `state_machine/loop.py`：主循环、状态创建、合并、动作执行、转移、修复、page-local flow。
- `state_machine/matching.py`：条件求值、状态匹配、启用条件选择。
- `state_machine/page_handler.py`：同页局部流的节点/边学习与匹配。
- `state_machine/io.py`：资源初始化、JSON 读写、截图读写、经验文件。
- `state_machine/presets.py`：内置动作宏。
- `state_machine/logger.py`：运行日志与摘要。

固定 schema version 为 `1.1.0`。

## 2. 坐标和截图约定

所有持久化条件 bbox、点击坐标和 LLM 输出坐标均使用 `1000x1000` 逻辑坐标。

运行时使用 `CoordinateMapper(logical_w=1000, logical_h=1000, real_w=1280, real_h=720)` 把逻辑坐标映射到实际模拟器截图/点击坐标。主循环截图若不是 `1280x720`，会先 resize 到 `1280x720`。

矩形字段统一形态为：

```json
[x1, y1, x2, y2]
```

其中 `x1/y1/x2/y2` 是逻辑坐标。运行时通过 `mapper.rect_to_real(rect)` 转成实际像素区域。条件中的 `params.rect` 优先级高于顶层 `bbox`；缺失 `params.rect` 时可回退到 `bbox`。

## 3. 资源目录

FSM 资源根目录固定为：

```text
StateMachineResources/
```

运行时会确保以下文件或目录存在：

```text
StateMachineResources/
  state_graph.json
  runtime_state.json
  llm_protocol_schema.json
  experience.md
  debug/
  templates/
```

### 3.1 `state_graph.json`

状态图文件，保存全局状态节点和动作转移边。

```json
{
  "schema_version": "1.1.0",
  "created_at": "2026-05-23T00:00:00Z",
  "updated_at": "2026-05-23T00:00:00Z",
  "nodes": [
    {
      "state_id": "hex-string",
      "slug": "state_slug",
      "enabled": true
    }
  ],
  "edges": [
    {
      "from_state_id": "state-id",
      "action_id": "action_main",
      "to_state_id": "state-id",
      "enabled": true,
      "weight": 1.0,
      "created_at": "2026-05-23T00:00:00Z"
    }
  ]
}
```

边语义：当 `from_state_id` 的 `action_id` 执行后，若后续截图匹配 `to_state_id`，该转移成立。图边用于优先选择 reachable target；未知 fallback 仍可发现图外目标并自动补边。

### 3.2 `runtime_state.json`

运行时游标和会话状态。

```json
{
  "schema_version": "1.1.0",
  "last_state_id": null,
  "run_id": null,
  "llm_session_id": null,
  "llm_turn_count": 0,
  "session_tier": "soft",
  "repair_fail_count": 0,
  "last_transition_ok": false,
  "pending_refresh": false,
  "pending_from_state_id": null,
  "pending_action_id": null
}
```

字段语义：

- `last_state_id`：最近选中的状态。
- `run_id`：本轮运行日志 ID。
- `llm_session_id` / `llm_turn_count` / `session_tier` / `pending_refresh`：外部推理会话轮转状态。FSM 只依赖这些计数与刷新标记，不规定 client 实现。
- `repair_fail_count`：连续修复失败次数。
- `last_transition_ok`：上一轮是否成功离开当前状态或完成转移。
- `pending_from_state_id` / `pending_action_id`：动作后离开原状态但未匹配到已知目标时暂存，之后解析到新状态或已有状态时补全图边。

### 3.3 `templates/<state_slug>/`

每个状态一个目录，包含：

```text
StateMachineResources/templates/<state_slug>/
  state.json
  screenshot_1.png
  screenshot_2.png
  template_1.png
  template_revised_*.png
  page_handler_template_*.png
```

`screenshot_*.png` 是状态样本截图。`template_*.png` 是从截图区域裁剪出的模板条件资源。

## 4. 状态文件 `state.json`

状态文件是 FSM 的主要持久化单元。

```json
{
  "schema_version": "1.1.0",
  "state_id": "hex-string",
  "slug": "english_lower_snake_case",
  "page_type": "normalized_page_type_or_null",
  "display_name": "display string",
  "description": "human readable summary",
  "created_at": "ISO-8601 UTC",
  "updated_at": "ISO-8601 UTC",
  "model_info": {
    "source": "llm",
    "weak_match": false,
    "last_merge_note": "optional"
  },
  "elements": [],
  "match_conditions": [],
  "actions": [],
  "page_handler": {}
}
```

### 4.1 `state_id`

创建状态时由 `uuid.uuid4().hex` 生成。状态图和 runtime 均以 `state_id` 作为稳定引用。

### 4.2 `slug`

由外部生成 payload 的 `slug` 经 `_slugify` 归一化得到：非字母数字字符转 `_`，小写，折叠重复 `_`，去除首尾 `_`，空值回退为 `state`。创建目录时若重名，会追加 `_2`、`_3` 等后缀；状态文件内 `slug` 保持归一化后的基础名。

### 4.3 `page_type`

由 `possible_page_type` 或 `page_type` 经 `_normalize_page_type` 得到。`none`、`null`、`unknown`、`n/a` 或空字符串视为 `null`。同类页面合并以归一化后的 `page_type` 为 key；若 `page_type` 为空，匹配摘要会回退使用 `slug`。

### 4.4 `elements`

`elements` 是外部状态生成 payload 的原始页面元素记录，主要用于审计和后续条件生成。当前 runtime 只把两类元素转成匹配条件：

```json
{
  "type": "text_line",
  "text": "短稳定子串",
  "bbox": [0, 0, 100, 50],
  "brief": "说明",
  "stability": "high",
  "discrimination": "mid"
}
```

```json
{
  "type": "pattern",
  "bbox": [0, 0, 100, 50],
  "brief": "说明",
  "stability": "high",
  "discrimination": "high"
}
```

`stability` 和 `discrimination` 只允许 `high`、`mid`、`low`，非法值按 `mid` 处理。

## 5. 匹配条件

`match_conditions` 是状态匹配的正式条件集合。

通用字段：

```json
{
  "id": "elem_text_1",
  "enabled": true,
  "kind": "text_line_contains",
  "params": {},
  "weight": 1.0,
  "brief": "说明",
  "role": "",
  "stability": "high",
  "discrimination": "high",
  "condition_status": "active",
  "bbox": [0, 0, 100, 50],
  "source": {}
}
```

字段语义：

- `enabled`：只有启用条件参与默认状态匹配。`include_disabled=True` 时可用于诊断。
- `kind`：当前只支持 `text_line_contains` 和 `region_template`。
- `params`：条件类型参数。
- `weight`：当前实现保留但不参与匹配计分。
- `stability`：稳定性等级，用于启用条件选择。
- `discrimination`：区分度等级，用于启用条件选择。
- `condition_status`：`active` 或 `deprecated`。deprecated 条件不会在合并正样本校验中作为 active 条件使用；默认匹配仍只看 `enabled`。
- `bbox`：条件原始逻辑区域。
- `source`：模板条件可记录截图和模板来源。

### 5.1 `text_line_contains`

```json
{
  "kind": "text_line_contains",
  "params": {
    "text": "方程",
    "rect": [65, 54, 102, 87]
  }
}
```

求值流程：

1. 对 `rect` 区域执行 OCR，参数 `white_text=false`。
2. 提取每个 OCR entry 的非空 `text`。
3. 若任意单行 OCR 文本包含 `params.text` 子串，则条件通过。

注意：这是 contains 匹配，不是精确匹配，也不是正则。`rect` 应只覆盖一行稳定文本。

### 5.2 `region_template`

```json
{
  "kind": "region_template",
  "params": {
    "rect": [100, 100, 200, 200],
    "threshold": 0.8,
    "template_path": "StateMachineResources/templates/state/template_1.png"
  }
}
```

求值流程：

1. 必须存在 `template_path`，否则失败。
2. 使用 `VisionEngine.match_template(frame_rgb, template_path, rect, threshold)` 在 `rect` 内模板匹配。
3. 默认 `threshold=0.8`。
4. 返回匹配是否成功、匹配位置和相似度，写入诊断 detail。

创建状态或 materialize page handler edge 时，runtime 会从当前截图的 `rect` 裁剪模板并写入 `template_path`。

## 6. 条件启用选择

新建状态或合并状态后，FSM 会调用 `_select_enabled_conditions` 自动选择启用条件。

等级分数：

```text
discrimination: high=4, mid=2, low=1
stability:      high=3, mid=2, low=1
target discrimination score = 4
```

选择流程：

1. 只考虑 `kind` 为 `text_line_contains` 或 `region_template` 的条件。
2. 先在当前截图上求值，过滤出通过的条件。
3. 所有条件先置为 `enabled=false`。
4. 将通过条件按 `(stability_score, discrimination_score)` 降序排序。
5. 依次选择条件，直到累计 discrimination 分数达到 `4`。
6. 如果只选中一个文本条件，会额外尝试加入一个稳定性至少 `mid` 的通过条件。
7. 如果累计分数仍低于 `4`，启用所有通过条件。
8. 若最终只启用一个条件且累计分数达到 `4`，`weak_match=true`。

状态匹配成功条件：启用条件数量大于 0，且所有启用条件均通过。

## 7. 状态匹配和选择

主循环每轮对所有 `templates/*/state.json` 执行 `_eval_state_match`。

`MatchResult` 结构：

```json
{
  "state_id": "state-id",
  "state_dir": "path",
  "passed_enabled": 2,
  "total_enabled": 2,
  "success": true,
  "passed_all": 2,
  "total_all": 2,
  "condition_results": []
}
```

默认匹配只评估 `enabled=true` 的条件。`success=true` 当且仅当：

```text
total_enabled > 0 && passed_enabled == total_enabled
```

未知场景选择 `_select_best_for_unknown(matches)`：

1. 过滤 `success=true` 的候选。
2. 无候选返回 `None`。
3. 单候选直接返回。
4. 多候选按 `passed_enabled` 降序、`total_enabled` 降序排序，取第一项。

动作后转移优先级：

1. 若 `prefer_reachable_first=true` 且图中存在从当前状态可达的目标，优先选择成功匹配且在 reachable 集合中的候选，并按 `passed_all`、`passed_enabled` 降序排序。
2. 否则使用 unknown fallback，选择最佳成功匹配候选。
3. fallback 候选必须不是当前状态，且 `passed_enabled == total_enabled`。
4. fallback 目标不在 reachable 集合时，自动向 `state_graph.json` 追加图边。

## 8. 动作模型

每个状态可包含多个 action，但当前执行器只取第一个 `enabled=true` 的 action。

```json
{
  "action_id": "action_main",
  "version": 1,
  "enabled": true,
  "cooldown_ms": 0,
  "steps": []
}
```

`cooldown_ms` 当前保留但未参与执行节奏控制。

### 8.1 `click`

```json
{
  "type": "click",
  "x": 849,
  "y": 903,
  "brief": "点击确认按钮"
}
```

执行流程：

1. 将逻辑坐标映射为实际坐标。
2. 调用 emulator tap。
3. 每次点击后固定等待 `ACTION_CLICK_WAIT_S=5.0` 秒。

### 8.2 `run_preset`

```json
{
  "type": "run_preset",
  "name": "wait_till_combat_end",
  "brief": "等待战斗结束"
}
```

执行流程：

1. 调用 `state_machine.presets.run_preset`。
2. preset 返回 `false` 时动作执行失败。
3. preset 抛异常时，runtime 记录异常，sleep 10 秒，并把该 step 当作成功继续后续 step。

当前内置 preset：

- `wait_till_combat_end`
- `find_and_interact_with_next_object`

## 9. 主循环语义

主入口 `run_agent_loop_fsm(session_id, serial=None, adb_path=None, interval_s=5.0)`。

初始化：

1. 确保 FSM 资源文件存在。
2. 初始化 emulator、坐标映射、视觉引擎、runtime 和 logger。
3. 重置 `runtime.run_id`、`last_state_id=null`、`pending_refresh=true`。

每轮循环：

1. 如有必要刷新外部推理会话 ID。
2. 构建系统提示，附加 `experience.md` 最近 5000 字。
3. 截图并 resize 到 `1280x720`。
4. 加载所有状态 meta。
5. 对当前截图匹配所有状态。
6. 如果没有匹配状态：
   - 请求外部推理生成状态 payload。
   - 优先尝试按 `possible_page_type` 合并到已有同类状态。
   - 合并失败则创建新状态目录和图节点。
   - 若 runtime 存在 pending transition，则补边到新状态。
   - 立即执行新状态动作，且 `prefer_reachable_first=false`。
7. 如果匹配到状态：
   - 写入 `runtime.last_state_id`。
   - 若 runtime 存在 pending transition，则补边到该已有状态。
   - 执行状态动作，且 `prefer_reachable_first=true`。
8. 根据外部推理轮数和成功转移状态决定是否轮转会话。
9. sleep `interval_s`。

## 10. 未知状态创建

外部状态生成 payload 最小结构：

```json
{
  "page_summary": "页面说明",
  "slug": "state_slug",
  "possible_page_type": "none",
  "elements": [],
  "actions": []
}
```

创建流程：

1. 生成 `state_id`。
2. 创建唯一状态目录。
3. 保存 `screenshot_1.png`。
4. 从 `elements` 转换 `match_conditions`：
   - `text_line` -> `text_line_contains`
   - `pattern` -> `region_template`
5. 对 `region_template` 裁剪模板文件并写入 `template_path`。
6. 自动选择启用条件。
7. 写入 `state.json`。
8. 向 `state_graph.json` 追加节点。

`actions` 会归一化为一个 `action_main`，非法 action 项会被跳过。非 `run_preset` 类型默认归一化为 `click`，缺省坐标为 `(500,500)`。

## 11. 同类页面合并

当未知截图 payload 带有有效 `possible_page_type` 且已有同 `page_type` 状态时，FSM 尝试合并而不是创建新状态。

合并前提：

1. 找到同 `page_type` 最新状态。
2. 最新状态存在样本截图。
3. 外部推理服务判断当前截图仍属于同一 `page_type`。

合并流程：

1. 找出当前截图上失败的 existing active conditions。
2. 请求外部推理服务对已有条件逐个 `keep`、`revise` 或 `deprecate`。
3. `revise` 支持改为 `text_line_contains` 或 `region_template`。
4. region template 修订会从当前截图裁剪 `template_revised_*.png`。
5. 未被返回且原本未启用的条件会标记为 `deprecated`。
6. 校验修订后的 active 条件必须覆盖所有同 page type 样本截图和当前截图。
7. 自动重新选择启用条件。
8. 校验启用条件不能误匹配其它 page type 的最新截图。
9. 通过后备份原 `state.json`，追加新样本截图，更新 `match_conditions`、`updated_at`、`page_type`、`model_info.weak_match` 和 `last_merge_note`。

任一校验失败则拒绝合并，转为创建新状态。

## 12. 动作执行和转移

普通动作最多尝试 `MAX_RETRY_PER_ACTION=2` 次。

每次尝试：

1. 执行动作 steps。
2. 截图。
3. 匹配所有状态。
4. 先按 reachable 目标尝试转移；失败再 unknown fallback。
5. 若转移到不同状态：
   - `runtime.last_state_id=target`
   - `runtime.last_transition_ok=true`
   - 清空 pending transition
   - 返回成功
6. 若当前状态不再匹配，但没有已知目标：
   - `runtime.last_state_id=null`
   - `runtime.last_transition_ok=true`
   - 记录 `pending_from_state_id=current`
   - 记录 `pending_action_id=action_id`
   - 返回成功，等待后续循环解析未知目标
7. 若仍匹配当前状态，但截图变化幅度达到 `SCREEN_CHANGE_DIFF_THRESHOLD=0.01`：
   - 进入 page-local flow
8. 若无变化，则进入下一次普通尝试。

截图变化检测使用两帧绝对差均值：

```text
diff_score = mean(abs(before - after)) / 255.0
changed = diff_score >= 0.01
```

## 13. 修复流程

普通动作尝试耗尽且没有成功转移后，进入修复。

修复 effort 顺序：

```text
low -> mid -> high
```

外部修复 payload：

```json
{
  "judgement": "invalid",
  "mode": "override",
  "actions": [],
  "note": "说明"
}
```

约束：

- `judgement` 只能是 `invalid` 或 `partial`。
- `mode` 只能是 `override` 或 `append`。
- `actions` 必须是数组，并按普通 action step 规则归一化。

应用规则：

- `override`：用修复动作替换原 steps。
- `append`：把修复动作追加到原 steps 后。

修复动作会立即持久化到当前 `state.json`：

- `action.version += 1`
- `action.steps = repaired_steps`
- `state.updated_at = now`

修复后按普通转移流程再次判断。如果离开原状态但目标未知，记录 pending transition。若修复后仍在当前状态但截图变化，进入 page-local flow。

全部 effort 失败时：

1. `runtime.repair_fail_count += 1`
2. `runtime.last_transition_ok=false`
3. 记录 `repair_failed`
4. 进程 `SystemExit(1)`

修复成功会追加一条 `experience.md` 经验记录。

## 14. Page Handler 同页局部流

Page handler 用于处理“状态仍匹配同一个页面，但页面内部内容或阶段变化”的场景，例如连续对话、选择确认、同类页面内多个临时子页。

状态文件中的结构：

```json
{
  "page_handler": {
    "enabled": true,
    "current_node_default": "root",
    "nodes": [
      {
        "id": "root",
        "brief": "默认局部阶段",
        "action": {
          "type": "click",
          "x": 500,
          "y": 500,
          "brief": "optional"
        }
      }
    ],
    "edges": []
  }
}
```

`ensure_page_handler` 会保证：

- `enabled=true`
- `current_node_default="root"`
- 至少存在 root node
- `edges` 为数组

### 14.1 Page Handler Edge

```json
{
  "id": "edge_xxx",
  "enabled": true,
  "from_node": "root",
  "to_node": "next_node",
  "conditions": [],
  "action": {},
  "expected_after_action": {
    "same_page_likely": true,
    "exit_likely": false,
    "screen_should_change": true,
    "reason": "说明"
  },
  "priority": 100,
  "brief": "说明",
  "success_count": 0,
  "fail_count": 0,
  "created_at": "ISO-8601 UTC",
  "updated_at": "ISO-8601 UTC"
}
```

edge conditions 与主状态条件一致，只支持 `text_line_contains` 和 `region_template`。edge action 与普通 action step 一致，只支持单个 `click` 或 `run_preset`。

### 14.2 Edge 匹配

在当前 `current_node` 下，只考虑：

- `enabled=true`
- `from_node == current_node`
- conditions 非空

所有 conditions 通过才命中。多条 edge 同时命中时按以下 key 降序选择：

```text
(priority, success_count - fail_count, len(conditions))
```

### 14.3 Edge 学习

如果当前 node 没有命中的 edge：

1. 请求外部推理服务生成一条 edge。
2. 将 edge 条件归一化。
3. 对 `region_template` 裁剪 `page_handler_template_*.png`。
4. 立即验证 edge conditions 必须匹配当前截图。
5. 验证失败则拒绝。

学习到的 edge 不会在执行前立即持久化；只有执行后产生进展或成功离开原状态时才写入状态文件。

### 14.4 Page-local 执行

最大步数 `LOCAL_FLOW_MAX_STEPS=12`。连续无进展上限 `LOCAL_FLOW_NO_PROGRESS_LIMIT=2`。

每步流程：

1. 在当前 node 匹配 edge；无 edge 则学习 edge。
2. 执行 edge 对应 action。
3. 截图并检测是否变化。
4. 匹配全局状态。
5. 若转移到其它已知状态，记录成功并返回。
6. 若原状态不再匹配但无已知目标，记录 pending transition 并返回。
7. 若仍是原状态且截图变化：
   - 认为 page-local 有进展。
   - learned edge 持久化并记 `success_count=1`；已有 edge 增加 `success_count`。
   - `current_node = edge.to_node`
   - 继续下一步。
8. 若无变化：
   - 已有 edge 增加 `fail_count`。
   - `no_progress_count += 1`
   - 达到上限则退出 page-local。

## 15. 预置动作

### 15.1 `wait_till_combat_end`

用途：等待战斗结束。

配置：

```text
template_path = StateMachineResources/templates/wait_till_combat_end/assets/combat_ongoing_marker.png
rect = [38, 14, 63, 38]
threshold = 0.8
timeout = 600s
```

流程：

1. 若模板文件不存在，返回 `false`。
2. 每 10 秒截图并匹配战斗中标记。
3. 若仍匹配，继续等待。
4. 若不匹配，进入 10 次 0.5 秒 burst 复查。
5. burst 期间重新匹配到战斗标记则继续等待。
6. burst 结束仍未匹配，返回 `true`。
7. 达到 600 秒全局超时也视为 done，返回 `true`。

### 15.2 `find_and_interact_with_next_object`

用途：在 3D 场景中循环移动并尝试交互，直到当前状态不再匹配。

逻辑坐标：

```text
move press: (191, 676)
interact:   (639, 562)
attack:     (818, 735)  # 当前实现定义但未使用
```

流程：

1. 每个 cycle 执行 10 次：
   - 在 move 坐标执行持续 800ms 的原地 swipe。
   - 等待 1 秒。
   - 点击 interact。
   - 等待 1 秒。
2. cycle 后 idle 5 秒。
3. 截图并匹配当前 state。
4. 若当前 state 不再成功匹配，返回 `true`。
5. 否则继续下一 cycle。

未知 preset 名称返回 `false`。

## 16. 外部推理接口边界

FSM 不规定外部推理 client 的实现，但规定运行时消费的 JSON 结构。

### 16.1 状态生成 payload

```json
{
  "page_summary": "string",
  "slug": "string",
  "possible_page_type": "none or known page_type",
  "elements": [
    {
      "type": "text_line",
      "text": "string",
      "bbox": [0, 0, 0, 0],
      "brief": "string",
      "stability": "high",
      "discrimination": "high"
    },
    {
      "type": "pattern",
      "bbox": [0, 0, 0, 0],
      "brief": "string",
      "stability": "mid",
      "discrimination": "mid"
    }
  ],
  "actions": [
    {
      "type": "click",
      "x": 500,
      "y": 500,
      "brief": "string"
    },
    {
      "type": "run_preset",
      "name": "wait_till_combat_end",
      "brief": "string"
    }
  ]
}
```

### 16.2 条件修订 payload

```json
{
  "same_page_type": true,
  "conditions": [
    {
      "condition_id": "existing-condition-id",
      "decision": "keep",
      "kind": "text_line_contains",
      "params": {
        "text": "string",
        "rect": [0, 0, 0, 0],
        "threshold": 0.8
      },
      "brief": "string",
      "stability": "high",
      "discrimination": "mid"
    }
  ],
  "note": "string"
}
```

`decision`：

- `keep`：保留原条件，可更新 brief/stability/discrimination。
- `revise`：按给定 kind/params 改写条件。
- `deprecate`：标记废弃并禁用。

兼容别名：`line_contains_text` 会被转换为 `text_line_contains`。

### 16.3 修复 payload

见第 13 节。

### 16.4 Page Handler Edge payload

见第 14 节。解析时会生成 edge id、补齐 condition id、时间戳和统计字段。

## 17. 会话轮转

FSM 维护外部推理会话计数：

```text
SOFT_LIMIT = 5
MID_LIMIT = 10
HARD_LIMIT = 15
FORCE_RESET_AT = 16
```

轮转规则：

- `llm_turn_count >= 16`：强制 `pending_refresh=true`。
- `session_tier=soft` 且 turn >= 5 且上一转移成功：升为 `mid` 并刷新。
- `session_tier=mid` 且 turn >= 10 且上一转移成功：升为 `hard` 并刷新。
- `session_tier=hard` 且 turn >= 15：刷新。

刷新时：

- 生成新的 `llm_session_id = "{base_session_id}-fsm-{8hex}"`。
- `llm_turn_count=0`
- `pending_refresh=false`
- `last_transition_ok=false`

## 18. 日志

每次运行会创建：

```text
StateMachineResources/debug/sessions/<session_id>/runs/run_<run_id>/
  events.jsonl
  summary.json
  session_report.txt
  llm_raw/

StateMachineResources/debug/sessions/<session_id>/latest_session_report.txt
```

`events.jsonl` 每行一个 JSON event。`summary.json` 聚合计数：

- loops
- llm_turns
- unknown_states
- states_created
- states_merged
- merge_rejected
- actions
- clicks
- presets
- transitions
- edges_added
- repairs
- repair_failures
- ocr_calls
- ocr_errors
- last_state_id
- last_transition

主要事件名包括：

- `run_start`
- `loop_start`
- `frame_captured`
- `match_candidates`
- `unknown_state`
- `llm_payload_parsed`
- `state_created`
- `merge_accepted`
- `merge_rejected`
- `graph_node_added`
- `graph_edge_added`
- `state_selected`
- `action_attempt`
- `action_click`
- `action_preset`
- `transition`
- `transition_reachable_miss`
- `transition_rejected`
- `page_handler_hit`
- `page_handler_miss`
- `page_local_step_result`
- `repair_applied`
- `repair_failed`
- `ocr_summary`

## 19. 持久化和备份规则

- `state_graph.json` 新增节点/边时直接覆盖写入。
- `runtime_state.json` 每次游标或会话状态变化后覆盖写入。
- 新建状态会保存 `state.json` 和 `screenshot_1.png`。
- 合并状态前会对原 `state.json` 创建同目录备份：

```text
state.json.bak_<YYYYMMDD_HHMMSS_microseconds>
```

- 修复动作会直接更新当前 `state.json`，当前实现不为修复写备份。
- page handler edge 统计更新会直接更新当前 `state.json`。

## 20. 失败和退出语义

可恢复失败：

- 未知状态外部 payload 无效：本轮跳过，等待下一轮。
- 合并失败：回退为创建新状态。
- page handler 学习 edge 无效：退出 page-local flow，回到动作修复或主流程判断。
- preset 返回 `false`：当前动作执行失败。

不可恢复失败：

- 普通动作重试和 low/mid/high 修复全部失败：记录 `repair_failed`，进程退出 `SystemExit(1)`。

特殊容错：

- preset 抛异常时，runtime 记录异常、等待 10 秒，并将该 preset step 当作成功继续。

## 21. 当前实现限制

- 每个状态当前只执行第一个启用 action。
- `cooldown_ms`、`weight` 当前不参与决策。
- 状态匹配是启用条件全通过语义，没有部分匹配阈值。
- OCR 文本条件依赖单行 OCR contains，长文本、跨行 bbox、动态名称和数字会显著降低稳定性。
- `region_template` 条件依赖创建时裁剪的固定模板，背景变化或 bbox 过大可能导致误差。
- page handler 的 `enabled` 字段目前只保证存在，主流程进入 page-local 时没有显式检查 `enabled=false` 跳过。
- `find_and_interact_with_next_object` 定义了 attack 坐标，但当前实现未使用。
