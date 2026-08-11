"""LangGraph StateGraph 状态定义。
根据经验 1289479：只要底层 agent 要读 state["messages"]，状态 schema 就**必须包含 messages 字段**。
我们在包含 messages 的前提下扩展以下字段：
    - thread_id:    会话 ID，用于 LangGraph Checkpoint 断点恢复
    - pending_confirm: 高危操作挂起信息 {tool_calls:[...], hint:str} / None
    - debug_log:    调试日志列表（推送给 P-03 调试面板）
    - last_tools:   上一步工具调用的元数据，供高危确认边读取
"""
from __future__ import annotations

# 模块顶层显式 import 所有会出现在类型注解中的符号，彻底解决 get_type_hints / Annotated NameError
# （对齐 M0 pytest 遇到的 NameError: Annotated not defined 根因：TypedDict 注解延迟求值时 namespace 缺符号）
import operator
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypedDict

# 把项目根加到 sys.path，import src.* 不报错
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import BaseMessage  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402  LangGraph 官方 add_messages reducer（合并消息/工具返回）


# ——— 为 get_type_hints 提供全局 namespace（StateGraph 构造时读不到就会 NameError）———
STATE_GLOBALS: dict[str, Any] = {
    "Annotated": Annotated,
    "TypedDict": TypedDict,
    "NotRequired": NotRequired,
    "Literal": Literal,
    "BaseMessage": BaseMessage,
    "Any": Any,
    "list": list,
    "dict": dict,
    "str": str,
    "int": int,
    "None": type(None),
    "add_messages": add_messages,
    "operator": operator,
}


class AgentState(TypedDict, total=False):
    """LangGraph ReAct 状态图使用的状态 Schema。
    - 使用 total=False：除 messages 外，其他字段都可以缺失（NotRequired 语义）。
    - messages 用 Annotated 绑定 LangGraph add_messages reducer：状态合并时自动做消息追加，而不是覆盖。
    """

    # —— 必需字段：只要用 ReAct Agent + ToolNode，这一个字段必须有（对齐经验 1289479）——
    messages: Annotated[list[BaseMessage], add_messages]

    # —— 扩展字段：业务/调试所需 ——
    thread_id: str
    """会话唯一标识，LangGraph Checkpoint 用 thread_id 恢复上一步状态"""

    pending_confirm: "dict[str, Any] | None"
    """高危操作挂起容器：
    {
      "tool_calls": [ { "id":..., "name":"delete_file", "args":{...} } ],
      "hint": "本次操作涉及 4 个文件永久删除，需要用户确认 DELETE"
    }
    None = 当前没有待确认操作。
    """

    last_tool_results: "list[dict[str, Any]]"
    """上一步工具调用结果摘要（调试面板显示用），格式：
    [ { "tool_name":"...", "args":{...}, "ok":True, "result_preview":"..." , "elapsed_ms":123 } ]
    """

    debug_log: "list[tuple[str, str, int]]"
    """调试面板日志队列 (时间戳_str, level[ASR|AGENT|TOOL|OBS|ERROR], message_str)，对话服务按时间推送到 P-03。"""

    user_id: str
    """预留多用户字段，当前单用户版固定 'default'"""

    step_count: int
    """Agent 已执行的 agent→tool 轮次计数，用于防止死循环"""

    runtime_system: str
    """运行时系统提示：调用方注入（通常含长期记忆），优先于默认 system_prompt。
    每次用户输入都会刷新；agent_node 读取它构造 SystemMessage。"""


# 单用户默认值，Graph 初始化 state 时用
DEFAULT_STATE: dict[str, Any] = {
    "messages": [],
    "thread_id": "default-thread-001",
    "pending_confirm": None,
    "last_tool_results": [],
    "debug_log": [],
    "user_id": "default",
    "step_count": 0,
    "runtime_system": "",
}


__all__ = ["AgentState", "DEFAULT_STATE", "STATE_GLOBALS", "add_messages"]
