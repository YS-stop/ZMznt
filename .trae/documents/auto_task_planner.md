# 自动任务规划器实现计划

## 背景与目标

用户希望助手能够：

1. 自动检测**多步操作请求**并拆分为子任务
2. 按**阶段**（准备/信息/动作/验证）和**优先级**组织子任务
3. 自动连续执行所有子任务
4. 在 UI 中显示任务进度

触发原因：之前的"打开抖音→搜索→进主页→点视频"复杂指令导致 Agent 死循环（GraphRecursionError），需要结构化任务执行来避免 LLM 在超长工具链中迷失。

## 推荐方案：外部规划器 + 复用现有 ReAct 子图

不将 planner 插入现有 LangGraph，而是在 `AssistantAgent` 入口层增加规划路径：

* 单步请求：直接走原有 `_execute_internal`

* 多步请求：先由 LLM 生成 `TaskPlan`，再循环执行每个子任务

* 每个子任务使用独立 `thread_id` 复用现有 ReAct 图，保留工具调用、高危确认、Checkpoint 等全部现有能力

## 数据结构设计

### 新建 `src/core/task_plan.py`

```python
TaskPhase(IntEnum):
    PREP = 0      # 准备：打开浏览器/应用/文件
    INFO = 1      # 信息：搜索、列出元素、读取内容
    ACTION = 2    # 动作：点击、输入、提交
    VERIFY = 3    # 验证：提取结果、总结

TaskStatus(Literal):
    "pending" | "running" | "success" | "failed" | "skipped"

TaskItem:
    - id: str
    - description: str
    - tool_hint: str | None
    - phase: TaskPhase
    - priority: int
    - dependencies: list[str]
    - status: TaskStatus
    - result_summary: str
    - retry_count: int
    - max_retries: int

TaskPlan:
    - goal: str
    - tasks: list[TaskItem]
```

工具函数：

* `is_multi_step_request(text: str) -> bool`：启发式多步检测

* `topological_sort(tasks) -> list[TaskItem]`：按依赖排序

* `get_ready_tasks(tasks, done_ids)`：获取可执行任务

### 修改 `src/core/state.py`

`AgentState` 新增可选字段：

```python
task_plan: "dict[str, Any] | None"
```

## Agent 层改造

### 修改 `src/services/agent_service.py`

新增方法：

1. `_detect_multi_step(user_input: str) -> bool`

   * 基于顺序连接词（先/然后/接着/再/最后）、步骤编号、多动作动词、多工具关键词、请求长度等加权打分

2. `_generate_plan(user_input: str) -> TaskPlan`

   * 调用 `get_main_llm()`，使用 Pydantic 结构化输出

   * Prompt 注入可用工具列表、阶段定义、安全规则

   * 解析失败时返回 `None`

3. `_execute_plan(plan: TaskPlan, parent_thread_id: str, stream_cb) -> str`

   * 拓扑排序后循环执行

   * 每个子任务调用 `_execute_subtask`

   * 失败时自动重试 1 次，两次失败后标记 `failed` 并跳过所有传递依赖为 `skipped`

   * 通过 `stream_cb` 发送 `"plan"` 和 `"task_update"` 事件

4. `_execute_subtask(task, parent_thread_id) -> tuple[bool, str]`

   * 构造聚焦式系统提示词："当前执行整体任务的第 X 步：..."

   * 子任务 thread\_id：`f"{parent_thread_id}__{task.id}"`

   * 调用现有 `_execute_internal` 执行

   * 返回 (成功/失败, 结果摘要)

修改入口：

* `stream_events` / `run` 中先检测多步，是多步则生成并执行计划，否则走单步

* 规划失败时降级为单步执行并发送警告日志

StreamCallback 扩展文档：

* `"plan"`：附加 `TaskPlan` 字典

* `"task_update"`：附加 `{task_id, status, result_summary, progress_pct}`

## 桥接层与 UI 进度

### 修改 `src/services/ui_bridge_service.py`

`_AgentWorker` 新增信号：

```python
plan_ready = Signal(object)
task_changed = Signal(str, str, str, int)  # task_id, status, summary, progress_pct
```

`_on_stream` 中增加分支处理 `"plan"` 和 `"task_update"`。

`UIBridgeService` 新增 UI 控制信号：

```python
sig_show_task_plan = Signal(object)
sig_update_task_progress = Signal(str, str, str, int)
sig_hide_task_plan = Signal()
```

`_start_task` 中连接 Worker 信号到 Bridge 信号。

### 新建 `src/ui/widgets/task_progress.py`

实现 `TaskProgressWidget`：

* 顶部 `QProgressBar` 显示总进度

* 下方任务列表，每行显示：状态图标 + 任务描述 + 阶段标签 + 结果摘要

* 方法：`set_plan(plan)`、`update_task(...)`、`clear()`

* 样式沿用现有设计系统

### 修改 `src/ui/drawer_main_panel.py`

在 `_build_chat_tab` 的气泡列表上方嵌入 `TaskProgressWidget`（默认隐藏）。

新增对外 API：

```python
show_task_progress(plan)
update_task_progress(task_id, status, summary, pct)
hide_task_progress()
```

### 修改 `src/ui/application.py`

在 `AppController` 中新增槽方法，并在 `bind_app` 中连接 Bridge 信号：

```python
show_task_plan -> panel.show_task_progress
update_task_progress -> panel.update_task_progress
hide_task_plan -> panel.hide_task_progress
```

## 多步检测启发式

`is_multi_step_request` 加权打分（超过阈值视为多步）：

| 维度     | 示例              | 权重 |
| ------ | --------------- | -- |
| 顺序连接词  | 先/然后/接着/再/最后/之后 | 高  |
| 步骤编号   | 第一步/1./①        | 高  |
| 多动作动词  | 创建+搜索+打开+点击     | 中  |
| 多工具关键词 | 浏览器+文件+搜索       | 中  |
| 请求长度   | >30 字符          | 低  |

单步请求直接排除：「今天天气」「打开微信」。

## 失败回退策略

* LLM 规划失败 / 解析失败 / 空计划 → 发送 `"AGENT"` 警告日志，降级为单步 `_execute_internal`

* 子任务连续失败 2 次 → 标记失败，跳过依赖任务，最终总结中说明哪些步骤未完成

## 关键文件

* `src/core/task_plan.py`（新建）

* `src/core/state.py`（扩展字段）

* `src/services/agent_service.py`（规划器+执行器）

* `src/services/ui_bridge_service.py`（信号传递）

* `src/ui/widgets/task_progress.py`（新建 UI 组件）

* `src/ui/drawer_main_panel.py`（嵌入进度组件）

* `src/ui/application.py`（信号连接）

* `tests/test_task_planner.py`（新建测试）

## 验证计划

1. 单元测试：多步检测启发式、拓扑排序、循环依赖检测
2. Mock 执行测试：3 任务计划顺序执行、中间任务失败导致下游跳过、重试后成功
3. 集成测试：模拟 LLM 返回 plan，验证工具按 plan 调用，进度事件正确触发
4. UI 测试：TaskProgressWidget 列表行数、状态更新、进度条变化
5. 回归测试：跑通 `tests/test_M9_web_automation.py` 和 `tests/test_M8_advanced_listen.py`，确保单步链路未被破坏

