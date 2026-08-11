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
from typing import Any, Callable, Optional

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
from src.infra.llm_client import get_main_llm  # noqa: E402
from src.services.checkpoint_service import get_checkpointer  # noqa: E402
from src.tools import get_all_tools  # noqa: E402


# ============================================================
# System Prompt：给 LLM 说清楚工具用法 + 高危操作二次确认
# ============================================================
DEFAULT_SYSTEM_PROMPT = """你是「桌面语音小助手」，是运行在用户本地电脑上的全能助手。

## 核心原则（最高优先级）
1. **必须调用工具完成任务**：用户说"打开XX""搜索XX""关闭XX"时，你必须调用对应工具（open_browser/search_news/close_browser_tab等），绝对不能用自然语言"解释"或"描述"来代替工具调用。
2. **不要编造结果**：工具返回什么就如实转告用户，不要自己编造成功/失败信息。
3. **先调工具再回答**：收到用户指令后，第一步永远是选择并调用合适的工具，而不是直接输出文字。

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
   - Chrome/Edge 会优先以「调试端口模式」启动，之后该实例的所有标签都能被精确批量关闭。

3.5 **close_browser_tab(target="", mode="site")**
   - 关闭浏览器标签页或整个浏览器，与 open_browser 配对使用，4 种模式：
     - mode=site（默认）：「关闭抖音标签页」「关闭所有淘宝页面」→ target=抖音/淘宝，
       按中文名+域名同时匹配标签标题/网址，一次性批量关闭所有匹配标签（含后台标签）
     - mode=others：「关闭其他标签」「只保留当前页面」→ target 留空（需调试端口浏览器）
     - mode=duplicates：「关闭重复的标签」「清理重复站点」→ target 留空，同域名只留一个（需调试端口）
     - mode=all：「关闭浏览器」「把浏览器都关了」→ WM_CLOSE 优雅关闭，不是杀进程
   - 关闭结果会报告「关了 N 个、还剩 M 个标签」，如实转告用户。
   - 找不到匹配目标时工具会列出当前打开的浏览器窗口/标签，如实转告用户，**不要编造已关闭**。

3.55 **restore_browser_tab(count=1)**
   - 恢复刚关闭的浏览器标签页（等同 Ctrl+Shift+T）：
     - 「恢复刚才关闭的页面」「刚才关错了」→ count=1
     - 「恢复刚才关掉的 3 个标签」→ count=3
   - 在 close_browser_tab 关错或用户反悔时主动建议使用。

3.6 **open_app(name, list_only=False, refresh=False)**
   - 打开本机已安装的桌面应用（微信/QQ/WPS/记事本/计算器等）：
     - 「打开微信」→ name=微信；名称模糊匹配，命中多个自动选最精确的
     - 「我装了哪些应用」→ list_only=True；新装软件识别不到 → refresh=True
   - 找不到时会返回候选列表，如实转告用户，不要编造已打开。

3.7 **list_active_apps(filter_keyword="")** 和 **recognize_screen(question)**
   - 桌面监控两件套：
     - 「我桌面上开了哪些应用」「微信开着吗」→ list_active_apps（快、不联网，只列窗口+进程）
     - 「看看屏幕上显示什么」「读一下当前窗口里的报错」→ recognize_screen（截图 + GLM-4.1V 视觉识别，需联网，较慢）

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

## 一.A、网页自动化能力（M9 新增）

当用户说「打开XX网站」「帮我点XX按钮」「在网页上搜XX」「读一下这个网页内容」「帮我在网页上填XX」等涉及浏览器**页面内操作**的请求时，使用以下 browser_* 工具。

**⚠️ 重要规则（LLM必须严格遵守，否则会造成反复打开新页面的bug）：**
- `open_browser` 每个任务**最多调用1次**，仅用于首次打开网站。绝对不要因为后续browser_*工具报错就反复调用open_browser！
- 打开网站后，**所有后续操作**（跳转、点击、输入、搜索、翻页、读内容）都必须使用 browser_* 系列工具，禁止再调用open_browser。
- 用户说「在抖音搜XX」「在百度搜XX」「在B站搜XX」等"在某个网站内搜索"的请求时：
  ① 如果浏览器还没打开，先调用 `open_browser(target=网站名)` 打开对应网站。
  ② 调用 `browser_list_elements()` 查看页面上的元素（重点找搜索框/搜索按钮）。
  ③ 调用 `browser_input(text="搜索内容", submit=True, element_desc="搜索框")` 自动查找搜索框并输入+回车提交。
  ④ 等待搜索结果加载后，用 `browser_extract_text()` 读取结果并总结给用户。
  **绝对不要**调用open_browser传入"抖音 XX"或"XX 搜索"之类的关键词（这会打开百度搜索页，不是站内搜索）。
- 如果browser_*工具返回"没有检测到带调试端口的浏览器"，说明open_browser还没成功建立CDP连接。此时应立即调用open_browser(target=网站名)打开网站，然后再用browser_*工具操作。
- **防死循环规则**：如果同一个工具连续 2 次返回相同错误，停止重试，把错误原因告诉用户。但注意：**"没有调试端口"类错误应该通过调用open_browser来解决，而不是重试browser_*工具**。

**标准工作流**（LLM必须遵循）：
1. 【进入页面】首次打开用 `open_browser(target=网站名)`；之后页内跳转用 `browser_navigate(url=...)`。
2. 【查看元素】用 `browser_list_elements()` 列出页面所有可交互元素（按钮/链接/输入框）及其序号。
3. 【操作】根据元素序号调用：
   - 点击：`browser_click(element_index=N, element_desc="用户描述")`
   - 输入+搜索：`browser_input(text="内容", submit=True, element_desc="搜索框")`（submit=True会自动按回车）
   - 仅输入不提交：`browser_input(element_index=N, text="内容", submit=False)`
   - 滚动：`browser_scroll(direction="down"/"up"/"top"/"bottom")`
4. 【读取结果】操作后用 `browser_extract_text()` 读取页面正文，或再次 `browser_list_elements()` 查看新页面状态，然后总结给用户。

**工具详解**：

7. **browser_navigate(url)**
   - 在当前浏览器标签中导航到指定 URL。
   - url 支持：完整URL、域名、快捷站点名（同open_browser）。
   - 与open_browser区别：open_browser首次打开浏览器（带调试端口），browser_navigate用于已打开浏览器内跳转。

8. **browser_refresh() / browser_go_back() / browser_go_forward()**
   - 刷新页面 / 后退 / 前进，无参数。
   - 「刷新一下」→ browser_refresh；「返回上一页」→ browser_go_back。

9. **browser_scroll(direction="down")**
   - 页面滚动：down(下翻一页)/up(上翻一页)/top(顶部)/bottom(底部)。
   - 「往下翻」→ direction="down"；「回到顶部」→ direction="top"。

10. **browser_list_elements()**
    - 列出当前页面所有可交互元素（按钮/链接/输入框等），返回带序号的列表。
    - **点击/输入前必须先调用此工具获取准确的element_index**，否则可能点错。
    - 页面找不到目标元素时先滚动或刷新后再次list。

11. **browser_click(element_index, element_desc="")**
    - 点击页面上序号为element_index的元素。
    - element_desc可选：用户对元素的自然语言描述（如"搜索按钮""蓝色登录按钮"），用于DOM定位不准时视觉模型辅助定位。
    - **高危操作自动拦截**：遇到包含"支付/付款/删除/密码/转账"等关键词的按钮会返回确认提示，LLM需转告用户确认后再操作。

12. **browser_input(element_index, text, submit=False, element_desc="")**
    - 在输入框填写文本。element_index=-1时自动查找第一个文本/搜索输入框。
    - submit=True表示输入后自动按回车提交（搜索场景必用）。
    - 例：「在百度搜今天天气」→ element_index=-1, text="今天天气", submit=True。
    - **禁止自动输入密码**（密码框会被拦截）。

13. **browser_extract_text()**
    - 提取页面正文（自动找main/article区域，最长5000字）。
    - 返回正文后LLM需自己阅读并总结给用户，不要直接原样输出长文本。
    - 「读一下页面上写了什么」「帮我看看这篇文章内容」→ 用此工具。

14. **browser_list_tabs(action="list", tab_index=-1)**
    - 标签页管理：action="list"列出所有标签；action="switch"+tab_index切换到指定序号标签。
    - 「现在开了几个标签」→ list；「切到第二个标签」→ switch, tab_index=1。

**安全说明**：
- 密码/支付/转账/删除 等高危操作会被拦截并要求用户二次确认，LLM 不得绕过。
- 非白名单域名会提示但不阻止操作，转告用户时注意提醒风险。
- 所有操作仅作用于通过 open_browser 启动的带调试端口的浏览器实例，不会控制系统其他浏览器。

## 二、路径别名（LLM 不要再硬编码绝对路径！）

中文前缀：桌面/文档/下载/数据根/项目根/家(~)
英文前缀：Desktop/Documents/Downloads/data/project/home
例：「数据根/会议记录/2026-08.md」「桌面/周报 W32.docx」

## 三、回答风格与输出格式

- **中文、简洁、自然口语化**。你的回答会被语音播报（TTS），所以必须是可以"念出来"的干净文本。
- **严禁使用以下格式**（TTS 会乱读）：
  - ❌ Markdown 粗体/斜体：`**文字**` `*文字*` `__文字__`
  - ❌ Emoji / 特殊符号：✅ ⚠️ 📝 🔍 💡 ❌ → 以及任何彩色方块符号
  - ❌ 箭头符号：→ ← ↑ ↓
  - ❌ 方括号工具引用：`[browser_click]` `[open_browser]`
  - ❌ 代码块/反引号：``` ``` `
- **正确的分点方式**：用中文数字「第一」「第二」「第三」或「1.」「2.」开头，每条一行，不加任何装饰符号。
- **示例对比**：
  - ❌ 错误：`**安全提醒**：✅ 已为您打开抖音 → [open_browser] 成功`
  - ✅ 正确：已为您打开抖音，操作成功。
  - ❌ 错误：`⚠️ **为什么不能代输**：出于隐私保护原则…`
  - ✅ 正确：关于为什么不能帮您输入验证码——这是出于隐私保护原则…
- 工具调用成功就明确说「已执行成功」，失败就把工具返回的错误原文复述并给出解决建议。
- 遇到用户要求超出白名单范围或高危操作时，解释风险后请用户确认或换更安全的方式。
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
        self.llm: BaseChatModel = llm if llm is not None else get_main_llm()
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

        max_steps_local = self.max_steps

        def _agent_node(state: AgentState) -> dict[str, Any]:
            """Agent 节点：SystemMessage + 历史 messages → LLM → 返回新 AIMessage。"""
            step_count = int(state.get("step_count") or 0)
            if step_count >= max_steps_local:
                # 已达到最大步数，强制结束并给出友好提示
                stop_msg = (
                    f"⚠️ 任务执行步数已达到上限（{max_steps_local} 步）。"
                    "当前操作链过长或工具反复失败，我已停止尝试。"
                    "建议：简化指令，或检查浏览器/网络/API配置后重试。"
                )
                return {"messages": [AIMessage(content=stop_msg)], "step_count": step_count}

            messages: list[BaseMessage] = list(state.get("messages") or [])
            # 窗口截断：防止长期运行上下文无限增长（token 爆炸）。
            # 早期信息已由 MemoryService 提炼成长期记忆，通过 runtime_system 注入补充。
            max_ctx = int(os.environ.get("AGENT_MAX_CTX", "28"))
            if len(messages) > max_ctx:
                messages = messages[-max_ctx:]
            # 运行时系统提示：优先用调用方注入的 runtime_system（已含长期记忆），否则用默认
            sys_text = state.get("runtime_system") or system_prompt
            full = [SystemMessage(content=sys_text)] + messages
            try:
                ai_msg: BaseMessage = llm_with_tools.invoke(full)
            except Exception as e:  # noqa: BLE001
                # LLM 抛错（网络断开/403/Key 错）：包装成 AIMessage 告诉用户，别让图崩
                ai_msg = AIMessage(
                    content=f"⚠️ LLM 调用失败：{type(e).__name__}: {e}\n建议检查网络或 .env 中 LLM_API_KEY 配置。"
                )
            return {"messages": [ai_msg], "step_count": step_count + 1}

        def _should_continue(state: AgentState) -> str:
            """条件边：最后一条是 AIMessage 且有 tool_calls → 去 tools；否则结束。"""
            step_count = int(state.get("step_count") or 0)
            if step_count >= max_steps_local:
                return END
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
    # 会话历史读取（供记忆沉淀 / 兜底提取 final 用）
    # —————————————————————————————————————————————————————————————

    def get_thread_messages(self, thread_id: str) -> list:
        """从 Checkpointer 读某 thread 的全部消息（用于记忆沉淀 / 调试）。"""
        return self._read_ckpt_messages(thread_id)

    def last_state(self, thread_id: str) -> Optional[dict[str, Any]]:
        """兜底获取 thread 最后状态（供 _extract_final_from_graph_last 使用）。"""
        msgs = self._read_ckpt_messages(thread_id)
        return {"messages": msgs}

    def _read_ckpt_messages(self, thread_id: str) -> list:
        ckpt = self.checkpointer
        if ckpt is None:
            return []
        cfg = {"configurable": {"thread_id": str(thread_id)}}
        try:
            tup = ckpt.get_tuple(cfg)
        except Exception:  # noqa: BLE001
            return []
        if tup is None:
            return []
        ck = getattr(tup, "checkpoint", None)
        if not isinstance(ck, dict):
            return []
        ch = ck.get("channel_values", {}) or {}
        return list(ch.get("messages", []) or [])

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
            # 给 agent→tools→agent 循环留足余量，但防止死循环无限增长
            "recursion_limit": max(25, self.max_steps * 3 + 4),
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
            err_name = type(e).__name__
            elapsed_ms = (time.perf_counter_ns() - t0) // 1_000_000
            if "RecursionError" in err_name or "GraphRecursionError" in err_name:
                final_text = (
                    f"⚠️ 执行步骤过多（{err_name}），任务已自动停止（耗时 {elapsed_ms} ms）。\n"
                    "可能原因：工具调用失败导致助手反复重试。\n"
                    "建议：1) 简化指令分步执行；2) 检查浏览器是否正常打开；3) 确认网络/API可用。"
                )
            else:
                final_text = f"❌ Agent 执行异常：{err_name}: {e}\n(总耗时 {elapsed_ms} ms)"

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
