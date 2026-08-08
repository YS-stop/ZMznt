"""Agent 服务：把 6 个工具 + LLM + LangGraph Checkpointer 组装成 ReAct Agent。

对外暴露：
    - AssistantAgent 类（可注入 llm / tools / checkpointer，便于测试 Mock）
    - get_agent() 全局单例
    - .run(user_input, thread_id) -> str  最后一段 AI 文本
    - .run_and_get_state(...) -> dict  完整 messages + debug_log

Graph 结构（标准 ReAct 循环）：
    START → [agent 节点：LLM.bind_tools 推理 → 输出 AIMessage 或 AIMessage+tool_calls]
                ↓  should_continue
             有 tool_calls  → END?   — 有 → [tools 节点：LangGraph ToolNode(6 个工具)]
                                            ↓
                                          agent（循环）
                ↓  无 tool_calls
            END（返回最终 AI 文本）
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# 确保 import src.*
_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from src.core.state import AgentState, DEFAULT_STATE  # noqa: E402
from src.infra.llm_client import get_qwen_llm  # noqa: E402
from src.services.checkpoint_service import get_checkpointer  # noqa: E402
from src.tools import get_all_tools  # noqa: E402


# ============================================================
# System Prompt：给 LLM 说清楚工具用法 + 高危操作二次确认
# ============================================================
DEFAULT_SYSTEM_PROMPT = """你是「桌面语音小助手」，是运行在用户本地电脑上的全能助手。你必须严格使用提供的工具完成任务，不要凭空猜测，不要编造结果。

## 一、工具列表（按需要选择，一次最多可并行调用多个）

1. **create_file(file_path, content, overwrite=False)**
   - 在白名单目录创建 UTF-8 文本文件。路径建议以中文别名开头，如：
     桌面/xxx.txt、文档/xxx.md、下载/xxx、数据根/xxx、项目根/xxx。
   - overwrite=False（默认）时文件已存在会拒绝，需要 LLM 显式再调一次 + overwrite=True。
   - 严禁写「项目根/src/...」代码目录，会被安全层拒绝。

2. **search_files(query, search_root="白名单默认", max_results=20)**
   - 按文件名搜索：query 支持关键字（不分大小写）或通配符(*.md/*2026*.txt)。
   - search_root 可选：白名单默认 / 桌面 / 文档 / 下载 / 数据根 / 项目根 / 家目录。

3. **open_browser(target, new_tab=True, autoraise=True)**
   - 打开系统默认浏览器：
     - target 可以是快捷站点名（知乎/B站/GitHub/淘宝/百度等 100+ 个）
     - 也可以是 URL（example.com 会自动补 https://）
     - 也可以是纯关键词（会打开「百度搜该关键词」的结果页）

3.5 **close_browser_tab(target)**
   - 关闭浏览器标签页或整个浏览器，与 open_browser 配对使用：
     - 「关闭抖音标签页」「把抖音关了」→ target=抖音（按中文名+域名同时匹配标签标题/网址）
     - 「关闭浏览器」「把浏览器都关了」→ target=全部（给所有浏览器窗口发 WM_CLOSE 优雅关闭，不是杀进程）
   - 找不到匹配目标时工具会列出当前打开的浏览器窗口，如实转告用户，**不要编造已关闭**。

3.6 **open_app(name, list_only=False, refresh=False)**
   - 打开本机已安装的桌面应用（微信/QQ/WPS/记事本/计算器等）：
     - 「打开微信」→ name=微信；名称模糊匹配，命中多个自动选最精确的
     - 「我装了哪些应用」→ list_only=True；新装软件识别不到 → refresh=True
   - 找不到时会返回候选列表，如实转告用户，不要编造已打开。

3.7 **list_active_apps(filter_keyword="")** 和 **recognize_screen(question)**
   - 桌面监控两件套：
     - 「我桌面上开了哪些应用」「微信开着吗」→ list_active_apps（快、不联网，只列窗口+进程）
     - 「看看屏幕上显示什么」「读一下当前窗口里的报错」→ recognize_screen（截图 + Qwen-VL 视觉识别，需联网，较慢）

4. **⚠️ delete_file（高危！必须两次调用）**
   - 参数：target, search_root="白名单默认", recursive=False, **dry_run=True**, confirm_keyword=None, max_items=100
   - **严格流程（LLM 必须遵循，否则一定失败）**：
       ① 第 1 次调用：**必须 dry_run=True（默认就是 True，别改）** → 只会返回「待删除 N 个文件/目录 + 总空间」预览列表，不会真删任何东西。
       ② LLM 先把预览结果回复给用户，明确询问：「以下 N 个文件/目录将被删除，共释放 XXX MB，确认要永久删除吗？（删除后不可恢复，确认请回复确认删除/是的/DELETE）」。
       ③ 用户真的确认后（例如用户说「确认删除」「删吧」「是的」等明确肯定语气），LLM 才进行第 2 次调用，并传：
          - dry_run=False
          - confirm_keyword="DELETE"（必须全大写 DELETE，其他任何值都会被拒绝）
          - 如果是非空目录，还要 recursive=True（否则会被拦）
   - 三层安全拦截（即便按流程也可能跳过部分项）：
     ① 路径黑名单（C:\\Windows / Program Files 等） ② 白名单前缀（家/桌面/文档/下载/数据/项目） ③ 受保护对象（src/venv/.git/.env/app.db 等）
   - 删除策略：若安装了 send2trash（推荐）→ 进回收站（可恢复），否则真删不可恢复。

5. **recognize_file(file_path, with_preview_lines=20, with_sha256=False, with_image_info=True)**
   - 识别单个文件信息（不修改任何内容）：大类（图片/视频/音频/文档/代码/压缩包/可执行等）、大小、创建/修改时间、文本预览前 N 行、可选 SHA256 指纹、可选图片尺寸。
   - 在用户说「看看这个文件是什么」「帮我看看这个文件前几行」「delete_file 之前先确认是不是这个文件」时一定要先调 recognize_file 再做决定。

6. **search_news(query, engine="auto", max_results=10, hours=0)**
   - 搜索最新新闻资讯：返回结构化列表（标题/摘要/来源/时间/URL）。
   - engine=auto（默认）：依次尝试必应 → 百度 → 离线模拟，保证一定有结果。
   - engine=mock：直接返回模拟数据，避免网络请求。
   - hours>0：按最近 N 小时过滤（24=一天内，168=一周内）。
   - **注意**：当前版本只返回原文列表，不会替你总结；如需总结请 LLM 自己读列表后用自然语言总结给用户。

## 二、路径别名（LLM 不要再硬编码绝对路径！）

中文前缀：桌面/文档/下载/数据根/项目根/家(~)
英文前缀：Desktop/Documents/Downloads/data/project/home
例：「数据根/会议记录/2026-08.md」「桌面/周报 W32.docx」

## 三、回答风格

- 中文、简洁、步骤清晰、结果结构化（适当用 ① ② ③ / 📝 📎 🔗 ⚠️ 📰 等 emoji 分点）。
- 工具调用成功就明确说「已执行成功」，失败就把工具返回的错误原文复述并给出解决建议（例如「C:\\Windows 被安全层拒绝，请换桌面或数据根路径下新建」）。
- 遇到用户要求超出白名单范围或高危操作时，不要嘴硬，解释风险后请用户确认或换更安全的方式。
- delete_file 的两次调用流程是硬规则，**LLM 不得省略第一步 dry_run 直接跳到第二步真删**。如果 LLM 直接真删将被二次确认门槛直接挡住返回红色错误，届时请回退一步先执行 dry_run 预览。
"""


StreamCallback = Callable[[str, "dict[str, Any] | BaseMessage | str | None"], None]
"""流式回调签名：(阶段标识, 附加数据) -> None
阶段标识有：
    "human":    用户输入进入图（附加：HumanMessage）
    "ai":       AI 返回一段 AIMessage（无 tool_calls 就是最终回答，有 tool_calls 就是准备调工具）
    "tool_pre": AI 即将调 N 个工具（附加：list[dict] [{name,args}]）
    "tool":     单个工具返回 observation（附加：ToolMessage）
    "done":     图执行结束（附加：最终回答 str）
"""


# ============================================================
# 类：AssistantAgent（可复用实例，编译一次多次调用）
# ============================================================

class AssistantAgent:
    """ReAct Agent：LLM + 6 Tools + Checkpointer（MemorySaver / SQLiteSaver）。

    典型用法：
        agent = AssistantAgent()  # 用默认 Qwen + 全部 6 工具 + 全局 Checkpointer
        answer = agent.run("在数据根建一份 2026 周报模板，然后搜一下 AI 领域最近新闻", thread_id="u1-t1")
        # 同一 thread_id 下次再调，Checkpointer 自动带回历史 messages
        answer2 = agent.run("刚才建的文件路径再发我一次", thread_id="u1-t1")
    """

    def __init__(
        self,
        *,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[list[Any]] = None,
        checkpointer: Any = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 20,
    ) -> None:
        self._lock = threading.Lock()
        self.system_prompt: str = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps: int = max(1, int(max_steps))

        # 1. LLM（不指定就用 Qwen 默认单例；bind_tools 后生成 llm_with_tools 给 agent_node 用）
        self.llm: BaseChatModel = llm if llm is not None else get_qwen_llm()
        self.tools: list[Any] = list(tools) if tools else get_all_tools()
        if not self.tools:
            raise ValueError("AssistantAgent 至少需要 1 个工具，当前 get_all_tools() 为空")
        try:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        except Exception:  # noqa: BLE001 - Mock LLM 可能不实现 bind_tools，降级直接用原始 llm
            self.llm_with_tools = self.llm

        # 2. Checkpointer（不指定用全局服务：MemorySaver 或 SQLite）
        self.checkpointer = checkpointer if checkpointer is not None else get_checkpointer()

        # 3. 构建并编译 StateGraph（一次构建反复用）
        self._graph: CompiledStateGraph | None = None
        self._build_graph()

    # —————————————————————————————————————————————————————————————
    # Graph 构建
    # —————————————————————————————————————————————————————————————

    def _build_graph(self) -> None:
        """构建 LangGraph ReAct 图（START→agent⇄tools→END）。"""
        # 闭包引用 self，避免每次调用时重新 bind
        llm_with_tools = self.llm_with_tools
        system_prompt = self.system_prompt

        def _agent_node(state: AgentState) -> dict[str, Any]:
            """Agent 节点：SystemMessage + 历史 messages → LLM → 返回新 AIMessage。"""
            messages: list[BaseMessage] = list(state.get("messages") or [])
            # 在最前面插 SystemMessage（如果还没插过——简单起见每次都插到最前面，同内容 LLM 不敏感）
            full = [SystemMessage(content=system_prompt)] + messages
            try:
                ai_msg: BaseMessage = llm_with_tools.invoke(full)
            except Exception as e:  # noqa: BLE001
                # LLM 抛错（网络断开/403/Key 错）：包装成 AIMessage 告诉用户，别让图崩
                ai_msg = AIMessage(
                    content=f"⚠️ LLM 调用失败：{type(e).__name__}: {e}\n建议检查网络或 .env 中 QWEN_API_KEY 配置。"
                )
            return {"messages": [ai_msg]}

        def _should_continue(state: AgentState) -> str:
            """条件边：最后一条是 AIMessage 且有 tool_calls → 去 tools；否则结束。"""
            last = (state.get("messages") or [])[-1] if (state.get("messages") or []) else None
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return END

        builder = StateGraph(AgentState)
        builder.add_node("agent", _agent_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", _should_continue, path_map={"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        self._graph = builder.compile(
            checkpointer=self.checkpointer,
            # 循环上限（硬保护），比 max_steps 多一步，给上层留余地
            interrupt_before=[],
            interrupt_after=[],
        )

    # —————————————————————————————————————————————————————————————
    # 对外：同步 run / run_and_get_state / stream_events
    # —————————————————————————————————————————————————————————————

    def run(
        self,
        user_input: str,
        thread_id: str = "default-thread-001",
        *,
        extra_state_fields: Optional[dict[str, Any]] = None,
    ) -> str:
        """最简 API：输入用户文本 + thread_id，返回最后一条 AI 文本。"""
        _, final_text, _err = self._execute_internal(
            user_input=user_input,
            thread_id=thread_id,
            extra_state_fields=extra_state_fields,
            stream_cb=None,
        )
        return final_text

    def run_and_get_state(
        self,
        user_input: str,
        thread_id: str = "default-thread-001",
        *,
        extra_state_fields: Optional[dict[str, Any]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """返回 (final_text, final_state_dict)，调试面板需要完整 state 时用。"""
        final_state, final_text, _err = self._execute_internal(
            user_input=user_input,
            thread_id=thread_id,
            extra_state_fields=extra_state_fields,
            stream_cb=None,
        )
        return final_text, final_state

    def stream_events(
        self,
        user_input: str,
        thread_id: str = "default-thread-001",
        *,
        stream_cb: Optional[StreamCallback] = None,
        extra_state_fields: Optional[dict[str, Any]] = None,
    ) -> str:
        """带流式回调版（UI 调试面板显示 7 色日志时用）。"""
        _final_state, final_text, _err = self._execute_internal(
            user_input=user_input,
            thread_id=thread_id,
            extra_state_fields=extra_state_fields,
            stream_cb=stream_cb,
        )
        return final_text

    # —————————————————————————————————————————————————————————————
    # 内部：统一执行入口
    # —————————————————————————————————————————————————————————————

    def _execute_internal(
        self,
        *,
        user_input: str,
        thread_id: str,
        extra_state_fields: Optional[dict[str, Any]],
        stream_cb: Optional[StreamCallback],
    ) -> tuple[dict[str, Any], str, Optional[Exception]]:
        """invoke 一次图，返回 (final_state, final_text, error_or_None)。"""
        t0 = time.perf_counter_ns()
        err: Optional[Exception] = None
        final_state: dict[str, Any] = dict(DEFAULT_STATE)
        final_text: str = ""

        if self._graph is None:  # 理论 __init__ 已 build
            with self._lock:
                if self._graph is None:
                    self._build_graph()
        graph = self._graph
        assert graph is not None

        # 1. 准备输入 state：新增一条 HumanMessage；extra_state_fields 允许调用方覆盖 thread_id/pending_confirm
        user_msg = HumanMessage(content=(user_input or "").strip() or "(用户未输入内容)")
        input_state: dict[str, Any] = dict(DEFAULT_STATE)
        input_state["messages"] = [user_msg]
        input_state["thread_id"] = str(thread_id or "default-thread-001")
        if extra_state_fields:
            # 允许额外字段（但 messages 不能覆盖，已经是 list 形式 reducer）
            for k, v in extra_state_fields.items():
                if k == "messages":
                    continue
                input_state[k] = v

        if stream_cb:
            try:
                stream_cb("human", user_msg)
            except Exception:  # noqa: BLE001 - 回调出错不影响主流程
                pass

        config: dict[str, Any] = {
            "configurable": {"thread_id": input_state["thread_id"]},
            "recursion_limit": max(3, self.max_steps * 2 + 4),
        }

        try:
            # 2. 调 stream(values)，实时按事件推给 stream_cb
            last_chunk: dict[str, Any] = {}
            for chunk in graph.stream(input_state, config=config, stream_mode="values"):
                last_chunk = chunk if isinstance(chunk, dict) else {}
                msgs: list[BaseMessage] = list(chunk.get("messages") or []) if isinstance(chunk, dict) else []
                last_msg: BaseMessage | None = msgs[-1] if msgs else None
                if last_msg is None:
                    continue
                # 回调：按消息类型分流
                try:
                    if isinstance(last_msg, AIMessage):
                        tcs = getattr(last_msg, "tool_calls", None) or []
                        if tcs and stream_cb:
                            pre_list = [{"name": tc.get("name"), "args": tc.get("args")} for tc in tcs]
                            stream_cb("tool_pre", pre_list)
                        if stream_cb:
                            stream_cb("ai", last_msg)
                    elif isinstance(last_msg, ToolMessage):
                        if stream_cb:
                            stream_cb("tool", last_msg)
                except Exception:  # noqa: BLE001
                    pass
            final_state = dict(last_chunk) if last_chunk else dict(final_state)
            # 3. 从最终 state 里找最后一段「非 tool_calls AIMessage.content」做 final_text
            msgs_final = list(final_state.get("messages") or [])
            for m in reversed(msgs_final):
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                    final_text = (m.content or "").strip()
                    break
            if not final_text and msgs_final:
                # 兜底：最后一条不管是什么直接转字符串
                final_text = str(getattr(msgs_final[-1], "content", msgs_final[-1]))
        except Exception as e:  # noqa: BLE001 - LangGraph 层任何异常都兜住返回用户可读字符串
            err = e
            final_text = f"❌ Agent 执行异常：{type(e).__name__}: {e}\n(总耗时 {(time.perf_counter_ns() - t0)//1_000_000} ms)"

        if not final_text:
            final_text = "(Agent 未返回任何回答)"
        if stream_cb:
            try:
                stream_cb("done", final_text)
            except Exception:  # noqa: BLE001
                pass
        return final_state, final_text, err


# ============================================================
# 模块级单例（懒加载，首次 get_agent 时构建）
# ============================================================
_AGENT_SINGLETON: Optional[AssistantAgent] = None
_AGENT_LOCK = threading.Lock()


def get_agent(
    *,
    llm: Optional[BaseChatModel] = None,
    tools: Optional[list[Any]] = None,
    checkpointer: Any = None,
    system_prompt: Optional[str] = None,
    force_rebuild: bool = False,
) -> AssistantAgent:
    """拿全局 Agent 单例。传入 llm/tools/checkpointer 会强制重建（force_rebuild=True 更明确）。"""
    global _AGENT_SINGLETON
    needs_new = force_rebuild or _AGENT_SINGLETON is None or any(
        x is not None for x in (llm, tools, checkpointer, system_prompt)
    )
    if not needs_new and _AGENT_SINGLETON is not None:
        return _AGENT_SINGLETON
    with _AGENT_LOCK:
        if not needs_new and _AGENT_SINGLETON is not None:
            return _AGENT_SINGLETON
        _AGENT_SINGLETON = AssistantAgent(
            llm=llm,
            tools=tools,
            checkpointer=checkpointer,
            system_prompt=system_prompt,
        )
        return _AGENT_SINGLETON


def reset_agent_singleton() -> None:
    """测试用：清掉单例（避免不同用例间污染 thread/checkpointer）。"""
    global _AGENT_SINGLETON
    with _AGENT_LOCK:
        _AGENT_SINGLETON = None


# ---------------------------------------------------------------
# 兼容别名：UI 层 ui_bridge_service.py 调用 get_assistant_agent()
# ---------------------------------------------------------------
get_assistant_agent = get_agent


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AssistantAgent",
    "StreamCallback",
    "get_agent",
    "get_assistant_agent",
    "reset_agent_singleton",
]
