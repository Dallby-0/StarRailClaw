# Intent-aware Progressive Page Handler v2

本规范覆盖旧的 state-scoped intent、`actions`、`controller.seed_action` 和 `action_templates` 设计。v2 面向全新状态机资源，不兼容旧状态文件。

## 0. Unknown-state LLM protocol

新状态的首次识别使用 Volcano Ark `POST /api/v3/responses`，并在同一个请求中完成页面特征、intent 判断和 1–2 步 bootstrap operation 的生成。请求使用 `text.format.type=json_schema`；所有对象关闭额外字段，关键枚举、四元坐标和步骤数量由 schema 约束。

该调用是无会话的一次性请求，不携带 Chat Completions 历史，也不会因 JSON 解析失败再次上传同一截图。底层 HTTP 客户端仍可对超时、429 或 5xx 做传输级重试；这与再次让模型分析页面的语义重试不同。handler 已有策略全部失效后触发的 repair LLM 是另一个、按需发生的升级阶段。

## 1. 职责边界

- 外部 FSM 识别当前页面状态并确认页面转移。
- 可选的全局 intent 保存少量需要跨页面维持的业务语义。
- `EffectiveCommand` 是每次 page handler 执行的统一输入；所有操作都有 command，但并非所有 command 都有 intent。
- page handler 按 operation 保存渐进式策略，负责“怎么做”，不负责创建或任意改写 intent。
- intent reducer 消费 handler 的结构化事件，更新 `phase/facts/status`。

## 2. Runtime intent

```json
{
  "intent_runtime": {
    "active_intent_id": null,
    "intents": {},
    "intent_stack": []
  },
  "pending_operation": null
}
```

普通唯一操作页面保持 `active_intent_id=null`。只有同页异操作、跨页连续任务、参数化选择或中断恢复需要 intent。

Intent 示例：

```json
{
  "schema_version": "intent.v1",
  "intent_id": "intent_x",
  "kind": "select_and_confirm_blessing",
  "status": "running",
  "phase": "select_candidate",
  "params": {"prefer_rarity": 3},
  "facts": {},
  "transitions": [
    {
      "from_phase": "select_candidate",
      "event": "candidate_selected",
      "next_phase": "confirm_selection",
      "fact_patch": {"selected_slot": "$event.slot"}
    }
  ],
  "completion": {"event": "blessing_acquired"},
  "revision": 1
}
```

页面切换不会清除 intent。只有 completion event、失败、取消或放弃令 intent 终止。

## 3. EffectiveCommand 解析顺序

1. active intent 的 `kind/phase` 命中当前 handler 的 `intent_routes`。
2. 若当前页面是透明阻塞层，允许执行 `intent_scope=intent_invariant` 且 `intent_effect=preserve` 的默认 operation，原 intent 保持不变。
3. 无 active intent 时，允许执行显式声明为 `intent_invariant` 且安全等级为 `low_risk/reversible` 的默认 operation。
4. 没有可解析 command 时，生成临时 unresolved command，只允许经 LLM 学得的低风险策略；危险操作必须有显式 intent。

绝不在 active intent 无法投影时盲目使用 intent-specific 默认动作。

## 4. State page handler schema

```json
{
  "page_handler": {
    "schema_version": "progressive_handler.v1",
    "default_operation": {
      "operation": "dismiss_overlay",
      "intent_scope": "intent_invariant",
      "intent_effect": "preserve",
      "safety": "low_risk",
      "expected_event": "overlay_dismissed"
    },
    "intent_routes": [
      {
        "intent_kind": "inspect_reward",
        "phase": "inspect",
        "operation": "open_reward_details",
        "expected_event": "details_opened"
      }
    ],
    "operation_policies": {
      "dismiss_overlay": {
        "operation": "dismiss_overlay",
        "intent_scope": "intent_invariant",
        "intent_effect": "preserve",
        "safety": "low_risk",
        "strategies": []
      }
    },
    "episode_trace": []
  }
}
```

## 5. Strategy lifecycle and levels

Strategy 状态：

```text
proposed -> active -> degraded -> quarantined
```

成功 canary 会将 proposed strategy 激活。确认失败会降级策略；危险副作用会立即隔离。旧策略不会被更强策略破坏性覆盖。

推荐等级：

```text
level 0: fixed_point
level 1: guarded fixed point（后续扩展）
level 2: region_template
level 3: OCR / detector（后续扩展）
level 4: LLM repair proposal
```

策略最多包含两个条件步骤。每一步执行后必须重新截图、等待稳定、匹配状态并验证 `expected_after`，不得盲目连续点击。

中间步骤若声明 `screen_should_change=false`，无明显画面变化是合法结果；瞬时 matcher miss 也不代表已经离开页面。runner 应继续执行同一 bootstrap strategy 的下一条件步骤。只有最终步骤或明确声明 `exit_likely=true` 的步骤才能触发全局状态交接。

## 6. 一次看图同时建立状态和 bootstrap policy

未知状态 LLM payload 必须包含：

同一张未知状态截图只允许一次网络 LLM 调用。响应 JSON 的轻微格式错误（当前支持数字 bbox 坐标之间漏逗号）在本地修复；不得以 `attempt=2` 再次上传图片。无法本地解析时本次 one-shot 识别失败，由调用方记录失败，而不是在同一次请求函数中重试。

```json
{
  "page_summary": "...",
  "slug": "...",
  "possible_page_type": "...",
  "elements": [],
  "intent_assessment": {
    "relation": "expected_step|blocking_overlay|completion_evidence|unrelated|contradiction|unknown",
    "reason": "..."
  },
  "intent_proposal": null,
  "bootstrap_operations": []
}
```

仅当无 active intent 且页面确实存在同页异操作或跨页语义时，LLM 才能提出 `intent_proposal`。普通唯一推进页面必须保持为 null。

Bootstrap operation 显式设置 `is_default=true` 时会成为默认操作。若响应中只有一个 `low_risk/reversible` operation，系统也会把这个无歧义结果作为默认操作，避免在刚完成“识别 + 行动建议”的同一轮之后再次调用 LLM。多个候选以及 `commit/destructive` operation 不会被隐式提升。

无 active intent 时，显式或无歧义推导出的安全 default 可以直接执行；`intent_scope` 只控制 active intent 已存在时该 default 能否插入并保持原 intent。换言之，`intent_specific` 不会让首次响应里已经选定的普通 default 失效。

## 7. 执行与验证

```text
识别页面
  -> 读取可选 active intent
  -> 解析 EffectiveCommand
  -> 选择 operation 下最低 level 的可用 strategy
  -> 写 pending_operation
  -> 执行一步
  -> 等待稳定并重新识别
  -> 分类结果
  -> 提交 semantic event
  -> reducer 更新 intent
  -> 清除 pending_operation
```

策略结果枚举：

```text
verified_success
partial_progress
no_effect
wrong_transition
unsafe_effect
state_mismatch
transient_unknown
```

截图变化本身不是语义成功。离开原页面时必须立即把控制权交回外部 FSM；旧 handler 不得在新页面继续重试。

## 8. Failure escalation

1. 最便宜 active/proposed strategy 执行并验证。
2. 确认失败后标为 degraded，尝试现有更高等级策略。
3. 所有物化策略耗尽后才调用 LLM repair。
4. Repair 只能给当前 `EffectiveCommand.operation` 添加 strategy patch，不能创建或改写 intent。
5. `region_template` proposal 的 `template_bbox` 会从当前截图物化为本地模板。
6. 新策略经 canary 验证后转为 active。

## 9. Safety invariants

- 无 intent 时禁止执行 `commit/destructive` strategy。
- active intent 无匹配 route 时，只允许透明、intent-preserving 默认操作。
- LLM repair 返回其他 operation 的 patch 会被丢弃。
- 两步策略必须逐步观察和验证。
- 动作导致离开原状态时立即停止当前 handler。
- pending operation 在点击前持久化，结果处理后清除，为后续崩溃恢复提供审计依据。
