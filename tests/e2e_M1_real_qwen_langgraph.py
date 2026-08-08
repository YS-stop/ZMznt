"""M1 阶段端到端验证：真实调用阿里云 Qwen（通义千问）+ LangGraph ReAct + 三个写好的工具。

这个脚本是一次性的验证脚本（不是正式代码），用于确认以下链路通：
    环境加载 → LLM 客户端初始化 → 工具注册（create_file/search_files/open_browser）→
    agent_node（llm.bind_tools） → tools_node（执行工具） → ReAct 循环 →
    最终 AI 自然语言回答

⚠️ 注意：
1. 输出里任何地方都不会明文打印 QWEN_API_KEY，只显示「sk-xxxx前4后4」掩码。
2. 会真的创建一个真实文件（DATA_ROOT/M1端到端验证_<时间戳>.txt），但脚本结束后会询问是否保留，默认自动删除以免污染数据目录。
3. 会真的打开浏览器（百度搜「北京2026年8月7日天气」），如果你是 Windows 桌面环境，浏览器会被自动激活。
"""
from __future__ import annotations

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


def mask_key(s: str) -> str:
    """密钥掩码：前4 + **** + 后4，非空串才处理。"""
    if not s:
        return "<empty>"
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + "*" * (len(s) - 8) + s[-4:]


# ============================================================
# 1. 打印配置（全部掩码，无明文泄露）
# ============================================================
print("=" * 70)
print("【M1 端到端验证】Qwen + LangGraph + 3 Tools")
print("=" * 70)
print(
    "  QWEN_API_KEY   :", mask_key(os.environ.get("QWEN_API_KEY", "")),
    f"(len={len(os.environ.get('QWEN_API_KEY', ''))})",
)
print("  QWEN_BASE_URL  :", os.environ.get("QWEN_BASE_URL", ""))
print("  QWEN_MODEL     :", os.environ.get("QWEN_MODEL", ""))
print("  DASHSCOPE_API  :", mask_key(os.environ.get("DASHSCOPE_API_KEY", "")))
from src.utils.path_utils import DATA_ROOT, PROJECT_ROOT as PR  # noqa: E402
print("  PROJECT_ROOT   :", PR)
print("  DATA_ROOT      :", DATA_ROOT)
print("=" * 70)


# ============================================================
# 2. 导入所有核心模块（如果这里报错，M0/M1 的骨架有问题）
# ============================================================
print("[2/7] 导入模块……")
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402

from src.core.state import AgentState, STATE_GLOBALS  # noqa: E402
from src.infra.llm_client import get_qwen_llm  # noqa: E402
from src.tools import get_all_tools  # noqa: E402
print("      ✅ 模块全部 import 成功")


# ============================================================
# 3. 初始化 LLM + 工具
# ============================================================
print("[3/7] 初始化 LLM & 工具……")
llm = get_qwen_llm()
assert llm is not None, "llm 客户端为空，检查 .env QWEN_API_KEY / QWEN_MODEL / QWEN_BASE_URL"
print(f"      llm 类型   : {type(llm).__module__}.{type(llm).__name__}")
tools = get_all_tools()
print(f"      工具数量   : {len(tools)} -> {[t.name for t in tools]}")

# 绑定工具给 LLM
llm_with_tools = llm.bind_tools(tools)
print("      ✅ LLM bind_tools 成功")


# ============================================================
# 4. 构建最小 LangGraph ReAct
# ============================================================
print("[4/7] 构建 LangGraph ReAct……")

SYSTEM_PROMPT_TEXT = """你是一个 Windows 桌面语音助手的 Agent，名字叫「小助手」。
当前提供以下 3 个工具，你必须严格按照工具的说明进行调用：

1. create_file：在白名单目录（桌面/文档/下载/数据根/项目根）创建 UTF-8 文本文件。
   - 参数：file_path（支持中文别名，例如「数据根/xxx.txt」「桌面/xxx.md」）、content（必填，文件内容）、overwrite（文件存在时是否覆盖，默认False）
   - 安全：不要尝试写入 PROJECT_ROOT/src/ 代码目录，不要写系统盘关键路径，会被拒绝。

2. search_files：按文件名搜索（支持关键字和通配符 *.md / *.txt 等）
   - 参数：query（关键字或通配）、search_root（默认「白名单默认」= 桌面+文档+下载+数据根）、max_results（默认20）

3. open_browser：打开系统默认浏览器。
   - 参数：target（中文快捷站点名如「知乎/百度」、网址如 example.com、或关键词如「2026年奥运会」）
   - 若 target 是纯关键词，会自动打开百度搜索该关键词的结果页。

【你必须遵循的行为规则】
- 只要任务涉及「创建文件/搜索文件/打开浏览器」，你 MUST 调用相应的工具，而不是靠“幻想”回答。
- 调用工具时参数必须严格符合工具要求的字段名和类型。
- 一次可以并行调用多个工具，但每个工具调用的参数必须完整。
- 收到工具返回的 observation 后，基于 observation 给用户做自然语言总结。
- 如果工具返回错误信息（❌ 开头），根据错误说明修正参数后再次调用，不要直接向用户报告失败。
- 最终回答使用中文，口语化、自然、有温度。
"""


def agent_node(state: AgentState) -> dict[str, Any]:
    """Agent 节点：把 state 里的 messages 前面注入 SystemMessage 再调 LLM。"""
    messages: list[BaseMessage] = state.get("messages", [])
    # 注入 system prompt 到最前面（保持 messages 不直接改 state，构造新列表）
    full_messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT_TEXT)] + list(messages)
    ai_resp = llm_with_tools.invoke(full_messages)
    return {"messages": [ai_resp]}


def should_continue(state: AgentState) -> str:
    """条件边：AI 消息有 tool_calls 就去 tools 节点，没有就结束。"""
    last = state.get("messages", [])[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


# 构造图
checkpointer = MemorySaver()
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, path_map={"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")
react_graph = graph_builder.compile(checkpointer=checkpointer)
THREAD_ID = "e2e-M1-thread-" + str(int(time.time() * 1000))
CONFIG = {"configurable": {"thread_id": THREAD_ID}}
print("      ✅ LangGraph compile 成功")
print(f"      thread_id  : {THREAD_ID}")


# ============================================================
# 5. 构造用户真实指令（3 个工具依次走一遍）
# ============================================================
TS = int(time.time() * 1000)  # 避免文件名冲突
TEST_FILE_ALIAS = f"数据根/M1端到端验证_{TS}.txt"
TEST_FILE_CONTENT = f"Hello from Qwen + LangGraph 端到端验证！\n生成时间戳：{TS}\n"
TEST_SEARCH_KEYWORD = "端到端验证"
TEST_BROWSER_TARGET = "北京2026年8月7日天气"

USER_INPUT = f"""请帮我按顺序做 3 件事：
1. 在「{TEST_FILE_ALIAS}」创建一个UTF-8文本文件，内容如下：
```
{TEST_FILE_CONTENT.strip()}
```
2. 文件创建完后，用 search_files 工具，搜索 数据根 目录下文件名包含「{TEST_SEARCH_KEYWORD}」的文件，验证刚才的文件确实存在。
3. 最后用 open_browser 工具，打开百度搜索关键词「{TEST_BROWSER_TARGET}」。

三件事都完成后，给我一个简洁的中文总结报告，告诉我：文件真实路径、文件大小字节数、搜索到几个匹配结果、浏览器是否成功打开。"""

print("[5/7] 构造用户输入……")
print("      测试文件别名  :", TEST_FILE_ALIAS)
print("      搜索关键字    :", TEST_SEARCH_KEYWORD)
print("      浏览器关键词  :", TEST_BROWSER_TARGET)
print("-" * 70)


# ============================================================
# 6. 执行 graph.stream，可视化每一步流转
# ============================================================
print("[6/7] 执行 LangGraph ReAct（stream 模式，逐节点打印）……")
print("-" * 70)
start_ns = time.perf_counter_ns()

final_state: AgentState | None = None
step = 0
# LangGraph stream_mode="values"：每次 yield 的就是完整 state dict（没有第二个 meta 返回值，版本差异导致之前解包报错）
for chunk in react_graph.stream(
    {"messages": [HumanMessage(content=USER_INPUT)]},
    config=CONFIG,
    stream_mode="values",
):
    step += 1
    messages = chunk.get("messages", [])
    last: BaseMessage = messages[-1] if messages else None
    elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
    # 通过最后一条 message 的类型「推断」是哪个节点产出的（values 模式不直接给 node 名，够用就行）
    inferred_node = "agent" if isinstance(last, (AIMessage, HumanMessage, SystemMessage)) else (
        "tools" if isinstance(last, ToolMessage) else "?"
    )
    print(f"\n[Step {step}] +{elapsed_ms}ms | inferred_node={inferred_node} | msg_type={type(last).__name__ if last else 'None'}")

    if isinstance(last, AIMessage):
        tc = getattr(last, "tool_calls", None) or []
        if tc:
            print(f"  ✨ AI 决定调用 {len(tc)} 个工具:")
            for i, t in enumerate(tc, start=1):
                tname = t.get("name", "?")
                targs = t.get("args", {})
                # 参数掩码（content 可能很长，只显示前 60 字）
                safe_args = {}
                for k, v in targs.items():
                    if isinstance(v, str) and len(v) > 80:
                        safe_args[k] = v[:60] + f"…（共{len(v)}字符）"
                    else:
                        safe_args[k] = v
                print(f"     [{i}] 🛠️  {tname}  args={safe_args}")
        else:
            # 最终回答
            print(f"  🤖 AI 最终回答:\n{last.content}")
    elif isinstance(last, ToolMessage):
        # observation 只显示前 200 字
        obs = last.content or ""
        preview = obs if len(obs) <= 300 else obs[:300] + f"\n…（observation 共{len(obs)}字符，已截断）"
        print(f"  📥 工具返回 observation (tool_call_id={last.tool_call_id[:8]}…):\n{preview}")
    final_state = chunk  # 最后一次 chunk 就是最终 state

total_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
print("-" * 70)
print(f"[6/7] ✅ 执行完成，总耗时 {total_ms} ms，Graph 流转 {step} 次 values 更新")


# ============================================================
# 7. 校验 & 总结
# ============================================================
print("[7/7] 校验：文件系统里真的有这个文件吗？")
from src.tools.file_tools import resolve_user_path  # noqa: E402
real_test_file: Path = resolve_user_path(TEST_FILE_ALIAS)
print(f"      文件真实路径   : {real_test_file}")
exists = real_test_file.exists() and real_test_file.is_file()
if exists:
    size = real_test_file.stat().st_size
    actual_content = real_test_file.read_text(encoding="utf-8")
    content_ok = actual_content.strip() == TEST_FILE_CONTENT.strip()
    print(f"      🟢 存在 & 是文件: YES | 大小 {size} 字节 | 内容正确: {'YES' if content_ok else 'NO'}")
    if not content_ok:
        print("         期望:\n", TEST_FILE_CONTENT)
        print("         实际:\n", actual_content)
else:
    print(f"      🔴 文件不存在！LLM 可能未调用 create_file，或参数错误")

# —— 统计 messages 里工具调用次数 & 工具返回次数 ——
msgs: list[BaseMessage] = final_state.get("messages", []) if final_state else []
ai_tool_calls_count = sum(
    1 for m in msgs if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
)
tool_msg_count = sum(1 for m in msgs if isinstance(m, ToolMessage))
browser_observation_has_ok = any(
    isinstance(m, ToolMessage) and "open_browser" in (m.name or "") and "✅" in (m.content or "")
    for m in msgs
)
print("-" * 70)
print("【端到端验证 汇总】")
print(f"  消息总数        : {len(msgs)}（SystemMsg不算）")
print(f"  AI 发起工具调用 : {ai_tool_calls_count} 次")
print(f"  工具返回 obs    : {tool_msg_count} 条")
print(f"  文件真实存在    : {'✅ 是' if exists else '❌ 否'}")
print(f"  文件内容正确    : {'✅ 是' if (exists and content_ok) else '❌ 否（或不存在）'}")
print(f"  open_browser 成功: {'✅ 是（返回✅）' if browser_observation_has_ok else '⚠️ 不确定（请检查浏览器是否弹出百度天气页）'}")
print(f"  总耗时          : {total_ms} ms")
print("=" * 70)

# —— 询问是否清理测试文件（默认自动清理，避免污染）——
# 这里直接清理，不交互（脚本在 CI/自动化场景能跑通）
if exists and real_test_file.name.startswith("M1端到端验证_") and real_test_file.suffix == ".txt":
    try:
        real_test_file.unlink()
        print(f"🧹 自动清理测试文件 {real_test_file.name} 成功。")
    except Exception as e:  # noqa: BLE001
        print(f"🧹 自动清理失败，不影响验证：{e}")
else:
    print("🧹 没有匹配的自动清理文件（或文件不存在）")

# 成功退出码 0
print("\n🎉 端到端验证完毕，如无 🔴 错误则说明以下链路全通：")
print("   .env → llm client → bind_tools → LangGraph ReAct → 3个工具执行 → 最终自然语言回答")
sys.exit(0)
