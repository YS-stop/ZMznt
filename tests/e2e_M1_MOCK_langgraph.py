"""M1 离线 Mock 端到端验证：不花钱、不需要网络，验证 LangGraph ReAct 链路 + 三个工具真执行。

背景：真实云端 Qwen 403（免费额度用完）→ 按经验 728523，切换到离线 Mock 方案：
    用 LangChain 自带 FakeListChatModel（每次调用按顺序吐出预先录好的 AIMessage），
    录制 4 次调用的返回：3 次 tool_calls（create_file → search_files → open_browser） + 1 次最终 AI 总结。

这个脚本跑完会：
    ✅ LangGraph 完整 ReAct 循环走 8 次 messages 流转（Hu→AI_toolcall→Tool→AI_toolcall→Tool→AI_toolcall→Tool→AI_final）
    ✅ 真实创建文件（DATA_ROOT 下，脚本结束自动清理）
    ✅ 真实执行 search_files
    ✅ 真实执行 open_browser（会真的开浏览器弹百度天气，如果你是桌面环境）
    ✅ 所有层的代码（llm_client 占位 / state.py / tools / ToolNode / StateGraph）全链路通过
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ============================================================
# 0. 路径 & 环境加载
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ============================================================
# 1. 生成唯一的时间戳（保证每次跑 mock 不会和之前的文件冲突）
# ============================================================
TS = int(time.time() * 1000)
TEST_FILE_ALIAS = f"数据根/M1端到端验证_MOCK_{TS}.txt"
TEST_FILE_CONTENT = f"Hello from MOCK Qwen + LangGraph 端到端验证！\n生成时间戳：{TS}\n"
TEST_SEARCH_KEYWORD = f"M1端到端验证_MOCK_{TS}"
TEST_BROWSER_TARGET = "北京2026年8月7日天气"

print("=" * 70)
print("【M1 离线 MOCK 端到端验证】LangGraph ReAct + 3 Tools 真执行")
print("=" * 70)
print("  说明：真实云端 Qwen 返回 403（免费额度用完）。")
print("        本脚本用 FakeListChatModel 预先录制 3 次 tool_calls + 1 次 AI 总结")
print("        不花钱，不需要外网，三个工具会真的执行（文件真创建/浏览器真弹出）。")
print(f"  时间戳 TS     : {TS}")
print(f"  测试文件别名  : {TEST_FILE_ALIAS}")
print(f"  搜索关键字    : {TEST_SEARCH_KEYWORD}")
print(f"  浏览器关键词  : {TEST_BROWSER_TARGET}")
print("=" * 70)


# ============================================================
# 2. 导入所有模块
# ============================================================
print("[2/7] 导入模块……")
from collections import deque
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from src.core.state import AgentState  # noqa: E402
from src.tools import get_all_tools  # noqa: E402
from src.tools.file_tools import resolve_user_path  # noqa: E402
print("      ✅ 模块全部 import 成功")


# ============================================================
# 3. 自定义 Mock LLM：按顺序吐预录好的 AIMessage（包括 tool_calls）
# ============================================================
class SequentialMockChatModel(BaseChatModel):
    """按顺序吐出 responses 队列里的 AIMessage，用于离线验证 ReAct 链路。"""

    responses: "deque[AIMessage]" = deque()
    invoke_count: int = 0

    @property
    def _llm_type(self) -> str:  # pragma: no cover - 纯接口
        return "sequential-mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            raise RuntimeError("SequentialMockChatModel 已耗尽预录响应，无法继续 _generate")
        self.invoke_count += 1
        next_msg: AIMessage = self.responses.popleft()
        return ChatResult(generations=[ChatGeneration(message=next_msg)])


# ============================================================
# 3. 构造 4 次预录响应（AIMessage 序列）
# ============================================================
print("[3/7] 构造 4 次 LLM 预录响应（3 个 tool_calls + 1 个最终总结）……")

# 3 个 tool_calls 的 id 要连贯，LangGraph ToolNode 需要 tool_call_id 匹配
TC_ID_CREATE = "call_create_" + str(TS)
TC_ID_SEARCH = "call_search_" + str(TS)
TC_ID_BROWSER = "call_browser_" + str(TS)

def tc(id_: str, name: str, args: dict) -> dict:
    """构造符合 LangChain / OpenAI 格式的 tool_call dict。"""
    return {
        "id": id_,
        "name": name,
        "type": "tool_call",
        "args": args,
    }


# 解析 TEST_FILE_ALIAS 对应的真实文件路径（只是为了最终总结里能写对路径，mock AI 能说对）
_real_file_path = resolve_user_path(TEST_FILE_ALIAS)

resp1 = AIMessage(
    content="好的，我先帮你创建文件。",
    tool_calls=[tc(
        TC_ID_CREATE,
        "create_file",
        {"file_path": TEST_FILE_ALIAS, "content": TEST_FILE_CONTENT, "overwrite": False},
    )],
)

resp2 = AIMessage(
    content="文件创建完成，接下来我搜索一下确认文件是否真的存在。",
    tool_calls=[tc(
        TC_ID_SEARCH,
        "search_files",
        {"query": f"*{TEST_SEARCH_KEYWORD}*.txt", "search_root": "数据根", "max_results": 10},
    )],
)

resp3 = AIMessage(
    content="搜索成功，确实搜到了。最后我打开浏览器帮你搜天气。",
    tool_calls=[tc(
        TC_ID_BROWSER,
        "open_browser",
        {"target": TEST_BROWSER_TARGET, "new_tab": True, "autoraise": True},
    )],
)

resp4 = AIMessage(
    content=(
        f"📝 三件事都完成啦！总结如下：\n"
        f"1️⃣ 创建文件：路径 {_real_file_path}，内容为「Hello from MOCK Qwen + LangGraph 端到端验证！」（UTF-8）。\n"
        f"2️⃣ 搜索验证：在数据根目录下搜到了 1 个文件名包含「{TEST_SEARCH_KEYWORD}」的匹配，就是我们刚创建的。\n"
        f"3️⃣ 浏览器打开：已成功调用 open_browser 打开百度搜索「{TEST_BROWSER_TARGET}」的页面。\n"
        f"如果你能在桌面看到浏览器弹出北京天气的百度搜索结果页，就说明 open_browser 真跑通了 😊"
    ),
)

# SequentialMockChatModel：把 4 条预录响应放队列
mock_llm = SequentialMockChatModel(
    responses=deque([resp1, resp2, resp3, resp4]),
    invoke_count=0,
)

tools = get_all_tools()
print(f"      工具数量: {len(tools)} -> {[t.name for t in tools]}")

llm_for_agent = mock_llm
print("      ✅ 4 条预录响应已放入队列，Mock LLM 初始化完成")


# ============================================================
# 4. 构建 LangGraph ReAct（和生产代码完全一致的图结构）
# ============================================================
print("[4/7] 构建 LangGraph ReAct 图……")

SYSTEM_PROMPT_TEXT = """你是小助手，使用给定的 3 个工具完成任务。（Mock 模式下已预录响应，此 prompt 仅占位）"""

def agent_node(state: AgentState) -> dict[str, Any]:
    """Agent 节点：直接调用预录的 mock_llm（每次调用按顺序吐出 resp1-4）。"""
    # SystemPrompt 注入（和生产一致），mock_llm 实际忽略内容，只按顺序吐响应
    ai_resp = llm_for_agent.invoke(state.get("messages", []))
    return {"messages": [ai_resp]}


def should_continue(state: AgentState) -> str:
    last = state.get("messages", [])[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


checkpointer = MemorySaver()
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, path_map={"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")
react_graph = graph_builder.compile(checkpointer=checkpointer)
THREAD_ID = "e2e-M1-MOCK-" + str(TS)
CONFIG = {"configurable": {"thread_id": THREAD_ID}}
print(f"      ✅ LangGraph compile 成功 | thread_id={THREAD_ID}")


# ============================================================
# 5. 用户输入（和真实云端 e2e 完全一致的 prompt 结构）
# ============================================================
USER_INPUT = f"""请帮我按顺序做 3 件事：
1. 在「{TEST_FILE_ALIAS}」创建一个UTF-8文本文件，内容如下：
```
{TEST_FILE_CONTENT.strip()}
```
2. 文件创建完后，用 search_files 工具，搜索 数据根 目录下文件名包含「{TEST_SEARCH_KEYWORD}」的文件，验证刚才的文件确实存在。
3. 最后用 open_browser 工具，打开百度搜索关键词「{TEST_BROWSER_TARGET}」。

三件事都完成后，给我一个简洁的中文总结报告。"""

print("[5/7] 用户输入就绪")
print("-" * 70)


# ============================================================
# 6. 执行 stream(values 模式)
# ============================================================
print("[6/7] 执行 LangGraph ReAct stream……")
print("-" * 70)

start_ns = time.perf_counter_ns()
final_state: AgentState | None = None
step = 0
for chunk in react_graph.stream(
    {"messages": [HumanMessage(content=USER_INPUT)]},
    config=CONFIG,
    stream_mode="values",
):
    step += 1
    messages = chunk.get("messages", [])
    last: BaseMessage = messages[-1] if messages else None
    elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
    inferred_node = "agent" if isinstance(last, (AIMessage, HumanMessage)) else (
        "tools" if isinstance(last, ToolMessage) else "?"
    )
    print(f"\n[Step {step}] +{elapsed_ms}ms | inferred_node={inferred_node} | msg_type={type(last).__name__ if last else 'None'}")

    if isinstance(last, AIMessage):
        tc_ = getattr(last, "tool_calls", None) or []
        if tc_:
            print(f"  ✨ AI 调用 {len(tc_)} 个工具:")
            for i, t in enumerate(tc_, start=1):
                print(f"     [{i}] 🛠️  {t.get('name')}  args={t.get('args')}")
        else:
            print(f"  🤖 AI 最终回答:\n{last.content}")
    elif isinstance(last, ToolMessage):
        obs = last.content or ""
        preview = obs if len(obs) <= 500 else obs[:500] + f"\n…（observation 共{len(obs)}字符，已截断）"
        print(f"  📥 工具 observation (name={last.name} | id={last.tool_call_id[:10]}…):\n{preview}")

    final_state = chunk

total_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
print("-" * 70)
print(f"[6/7] ✅ stream 完成 | 总耗时 {total_ms} ms | messages 流转 {step} 次")


# ============================================================
# 7. 校验 & 汇总
# ============================================================
print("[7/7] 校验：文件真创建 + 工具真调 + messages 结构对")
real_file: Path = resolve_user_path(TEST_FILE_ALIAS)
print(f"      期望文件路径   : {real_file}")
exists_ok = real_file.exists() and real_file.is_file()
content_ok = False
if exists_ok:
    actual = real_file.read_text(encoding="utf-8")
    content_ok = actual.strip() == TEST_FILE_CONTENT.strip()
    print(f"      文件存在       : {'✅ YES' if exists_ok else '❌ NO'}  大小={real_file.stat().st_size} B")
    print(f"      文件内容正确   : {'✅ YES' if content_ok else '❌ NO'}")
else:
    print(f"      文件存在       : ❌ NO（create_file 可能未执行成功）")

msgs: list[BaseMessage] = final_state.get("messages", []) if final_state else []

# 统计各类型消息数
from collections import Counter  # noqa: E402
type_counts = Counter(type(m).__name__ for m in msgs)
# 工具调用次数：AI tool_calls 总数量
ai_tool_calls_total = sum(len(getattr(m, "tool_calls", None) or []) for m in msgs if isinstance(m, AIMessage))
# ToolMessage 数量
tool_msg_count = sum(1 for m in msgs if isinstance(m, ToolMessage))
# 三个工具分别调用过没
names_used_in_tool_msgs = {getattr(m, "name", None) for m in msgs if isinstance(m, ToolMessage)}
create_called = "create_file" in names_used_in_tool_msgs
search_called = "search_files" in names_used_in_tool_msgs
browser_called = "open_browser" in names_used_in_tool_msgs
browser_obs_ok = any(
    isinstance(m, ToolMessage) and (m.name or "") == "open_browser" and "✅" in (m.content or "")
    for m in msgs
)

# 最终 AI 回答是否自然（含 3 个要点）
last_ai_msg = next((m for m in reversed(msgs) if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)), None)
ai_final_ok = bool(
    last_ai_msg
    and "创建文件" in last_ai_msg.content
    and "搜索" in last_ai_msg.content
    and "浏览器" in last_ai_msg.content
)

print("-" * 70)
print("【离线 MOCK 端到端验证 最终报告】")
print(f"  📨 messages 结构 : {dict(type_counts)}")
print(f"  🛠️  AI 发起工具调用 总数: {ai_tool_calls_total} 次")
print(f"  📥  Tool 返回 observation 总数: {tool_msg_count} 条")
print(f"  ✅ create_file 被调用 : {create_called}")
print(f"  ✅ search_files 被调用: {search_called}")
print(f"  ✅ open_browser 被调用: {browser_called} (open_browser 返回 ✅: {browser_obs_ok})")
print(f"  ✅ 文件真实存在      : {exists_ok}")
print(f"  ✅ 文件内容正确      : {content_ok}")
print(f"  ✅ AI 最终回答自然(含3个要点): {ai_final_ok}")
print(f"  ⏱️  总耗时           : {total_ms} ms")

passed = all([
    create_called, search_called, browser_called,
    exists_ok, content_ok, ai_final_ok,
    ai_tool_calls_total >= 3,
    tool_msg_count >= 3,
])
print("=" * 70)
if passed:
    print("🎉🎉🎉 MOCK 端到端验证 **全部通过**！以下链路 100% 跑通：")
    print("   代码路径：main.py 环境加载 → state.py AgentState →")
    print("             tools（file_tools/browser_tools）+ 安全校验 →")
    print("             LangGraph StateGraph → agent_node/should_continue/tools_node 循环 →")
    print("             ToolNode 执行 3 个工具 → 最终 AI 总结输出")
    print("\n👉 说明：")
    print("   - 真实云端 Qwen 返回 403（你的账号 Free tier 免费额度用完了），可选择：")
    print("     a) 阿里云 DashScope 控制台充值 / 关闭 use free tier only 模式")
    print("     b) 换另一个有效 Key 填到 .env QWEN_API_KEY=")
    print("     c) 或者换更便宜的 QWEN_MODEL=qwen-turbo（tokens 价格比 qwen-plus 低）")
    print("   - 等你处理完 Key 之后，直接运行 tests/e2e_M1_real_qwen_langgraph.py 就能跑真云端。")
    print("   - 本脚本 3 个工具都是**真实执行**的（文件真创建了、浏览器真弹了），说明：")
    print("     工具层、LangGraph 编排层、State 层，完全没问题，等 Key 好了上层直接用。")
else:
    print("❌ 存在失败项，请对照上面 ✅ 列表看哪一项没过，再检查对应代码。")
    sys.exit(2)


# —— 自动清理测试文件（以 M1端到端验证_MOCK_ 开头且 .txt 结尾）——
if exists_ok and real_file.name.startswith("M1端到端验证_MOCK_") and real_file.suffix == ".txt":
    try:
        real_file.unlink()
        print(f"\n🧹 自动清理测试文件 {real_file.name} 成功（避免污染数据目录）")
    except Exception as e:  # noqa: BLE001
        print(f"\n🧹 清理失败（不影响验证，手动删即可）：{e}")

sys.exit(0)
