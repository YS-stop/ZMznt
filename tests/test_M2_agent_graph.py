"""M2 阶段单测：CheckpointService + AssistantAgent 离线 Mock（不花钱，不需要外网）。

覆盖：
    1. CheckpointService 两种显式 backend：memory 必通；sqlite 不可用时降级不抛。
    2. AssistantAgent（注入 SequentialMockChatModel）完整 ReAct 链路，6 个工具各调一次：
         create_file → recognize_file → search_files → delete_file(dry_run=True) → search_news → open_browser
       最终 AI 总结出现。
    3. 同一 thread_id 追加一句新 query：Checkpointer 确实把历史 messages 带入了第二轮。
    4. 工具路由正确性：ToolMessage.name 集合 = 6 个工具名，说明 LangGraph ToolNode 路由对。
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# 先加载 .env（即便没 key 也 OK，全程用 Mock LLM）
from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJ / ".env", override=False)


# ============================================================
# 1. CheckpointService 冒烟
# ============================================================

def test_M2_01_checkpoint_service_memory_and_sqlite_fallback():
    from src.services.checkpoint_service import (
        CheckpointService,
        get_checkpointer,
        get_checkpoint_info,
        start_checkpoint_service,
        stop_checkpoint_service,
    )

    # 显式 memory 后端
    svc = CheckpointService()
    try:
        svc.start(force_backend="memory")
        info = svc.get_info()
        assert info["backend"] == "memory", info
        assert svc.started is True
        saver = svc.get()
        # MemorySaver 至少有 get_tuple / put_tuple 属性（接口上鸭式类型）
        assert hasattr(saver, "get_tuple") or "Memory" in type(saver).__name__

        # info 里包含 env_hint
        assert "env_hint" in info

        # 再调函数式入口（不同实例但接口一致）
        start_checkpoint_service(force_backend="memory")
        assert get_checkpoint_info()["backend"] == "memory"
        s2 = get_checkpointer()
        assert s2 is not None
    finally:
        svc.stop()
        stop_checkpoint_service()

    # sqlite 不可用时降级 memory（当前环境没装 langgraph-checkpoint-sqlite，预期降级）
    svc2 = CheckpointService()
    try:
        svc2.start(force_backend="sqlite")
        info = svc2.get_info()
        # 要么是 sqlite 要么降级 memory，两者都算过；但一定不能抛错
        assert info["backend"] in ("sqlite", "memory"), info
        if info["backend"] == "memory":
            assert "失败" in info.get("note", "") or "降级" in info.get("note", "")
        assert svc2.get() is not None
    finally:
        svc2.stop()


# ============================================================
# 2. Mock LLM（同 M1 的 SequentialMockChatModel）
# ============================================================

def _build_sequential_mock(ts: int, file_alias: str, file_content: str,
                           news_query: str, browser_keyword: str,
                           ) -> "SequentialMockChatModel":
    """录制 7 次 AIMessage：6 次 tool_calls + 1 次最终总结。"""
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from src.tools.file_tools import resolve_user_path

    class SequentialMockChatModel(BaseChatModel):
        responses: "deque[AIMessage]" = deque()
        invoke_count: int = 0

        @property
        def _llm_type(self) -> str:
            return "sequential-mock-m2"

        def _generate(self, messages, stop=None, run_manager=None, **kw):  # type: ignore[override]
            if not self.responses:
                raise RuntimeError("SequentialMockChatModel for M2 已耗尽预录响应")
            self.invoke_count += 1
            return ChatResult(generations=[ChatGeneration(message=self.responses.popleft())])

    def tc(id_: str, name: str, args: dict) -> dict:
        return {"id": id_, "name": name, "type": "tool_call", "args": args}

    real_file_path = resolve_user_path(file_alias)

    # 1. create_file
    r1 = AIMessage(
        content="好的，我先在数据根下创建你要求的文件。",
        tool_calls=[tc(f"tc_create_{ts}", "create_file", {
            "file_path": file_alias, "content": file_content, "overwrite": False,
        })],
    )
    # 2. recognize_file
    r2 = AIMessage(
        content="文件创建完成，先识别一下这个文件看看基本信息对不对。",
        tool_calls=[tc(f"tc_rec_{ts}", "recognize_file", {
            "file_path": file_alias, "with_preview_lines": 10, "with_sha256": True,
        })],
    )
    # 3. search_files（确认创建成功）
    r3 = AIMessage(
        content="识别信息对的。再用 search_files 搜文件名确认一下。",
        tool_calls=[tc(f"tc_sf_{ts}", "search_files", {
            "query": Path(file_alias).name, "search_root": "数据根", "max_results": 10,
        })],
    )
    # 4. delete_file 只做 dry_run 预览（M2 不真删）
    r4 = AIMessage(
        content="搜到了。接下来按流程先对同一路径做 delete_file dry_run 预览（不真删）。",
        tool_calls=[tc(f"tc_del_{ts}", "delete_file", {
            "target": file_alias, "dry_run": True,  # True 只预览
        })],
    )
    # 5. search_news（engine=mock 保证离线不出网）
    r5 = AIMessage(
        content="预览没问题（只是预览不会真删，文件继续保留）。接下来搜一下你关心的最新资讯。",
        tool_calls=[tc(f"tc_news_{ts}", "search_news", {
            "query": news_query, "engine": "mock", "max_results": 5, "hours": 0,
        })],
    )
    # 6. open_browser
    r6 = AIMessage(
        content="资讯拿到了。最后打开浏览器帮你搜索天气信息。",
        tool_calls=[tc(f"tc_br_{ts}", "open_browser", {
            "target": browser_keyword, "new_tab": True, "autoraise": False,  # autoraise=False 避免测试弹浏览器抢焦点
        })],
    )
    # 7. 最终总结
    r7 = AIMessage(
        content=(
            f"📝 6 项任务全部完成，简要总结：\n"
            f"1️⃣ 创建文件：路径 {real_file_path}，内容为「{file_content.strip()}」。\n"
            f"2️⃣ 识别文件：recognize_file 已确认是文本文件，带 SHA256 指纹。\n"
            f"3️⃣ 搜索确认：search_files 在数据根下命中了刚刚创建的文件。\n"
            f"4️⃣ 删除预览：delete_file dry_run 给出了待删除清单（未真删，文件继续保留）。\n"
            f"5️⃣ 资讯搜索：search_news 用 engine=mock 返回了 5 条关于「{news_query}」的资讯。\n"
            f"6️⃣ 浏览器打开：open_browser 已发起关键词「{browser_keyword}」的百度搜索。\n"
            f"✅ 以上 6 个工具均通过 LangGraph ReAct 循环真实执行。"
        )
    )
    return SequentialMockChatModel(
        responses=deque([r1, r2, r3, r4, r5, r6, r7]),
        invoke_count=0,
    )


# ============================================================
# 3. M2 主测试：6 工具链路 + Checkpointer 历史保留
# ============================================================

def test_M2_02_assistant_agent_6tools_and_checkpoint_history():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.memory import MemorySaver
    from src.services.agent_service import AssistantAgent, reset_agent_singleton
    from src.tools.file_tools import resolve_user_path

    ts = int(time.time() * 1_000_000)
    file_alias = f"数据根/M2验证_{ts}.md"
    file_content = f"# M2 验证文件\n这是 M2 阶段生成的测试文件\n时间戳={ts}\n"
    news_q = "人工智能大模型"
    browser_kw = "上海今天天气"
    thread_id = f"t_m2_{ts}"

    reset_agent_singleton()
    # 用独立 MemorySaver 避免和全局单例交叉污染（测试之间隔离）
    ckpt = MemorySaver()
    mock_llm = _build_sequential_mock(ts, file_alias, file_content, news_q, browser_kw)
    agent = AssistantAgent(llm=mock_llm, checkpointer=ckpt, system_prompt="MOCK 模式占位")

    try:
        # ============== 第一轮 ==============
        user1 = (
            f"请按顺序完成：\n"
            f"1. 在「{file_alias}」创建 Markdown，内容：\n```\n{file_content.strip()}\n```\n"
            f"2. 用 recognize_file 识别该文件（前 10 行预览 + SHA256）。\n"
            f"3. 用 search_files 在数据根下搜索文件名确认存在。\n"
            f"4. 对同一路径做 delete_file dry_run 预览（不真删）。\n"
            f"5. 用 search_news engine=mock 搜「{news_q}」取 5 条。\n"
            f"6. 用 open_browser 搜「{browser_kw}」（autoraise=False 不抢焦点）。\n"
            f"做完给一个 6 点小结。"
        )
        ans1, state1 = agent.run_and_get_state(user1, thread_id=thread_id)

        # 最终回答里要点齐
        for kw in ["创建文件", "识别文件", "搜索确认", "删除预览", "资讯搜索", "浏览器打开", "6 个工具均通过"]:
            assert kw in ans1, f"最终总结缺关键词 {kw}：\n{ans1}"

        # messages 结构：Human1 + (AI_toolcall + Tool) × 6 + AI_final = 14 条
        msgs1 = list(state1.get("messages") or [])
        type_names = [type(m).__name__ for m in msgs1]
        # 至少 1 Human + 7 AIMessage + 6 ToolMessage = 14
        assert len(msgs1) >= 14, f"messages 不够 14 条：{len(msgs1)} -> {type_names}"

        tool_names_used = sorted({
            m.name for m in msgs1 if isinstance(m, ToolMessage)
        })
        assert tool_names_used == sorted([
            "create_file", "recognize_file", "search_files",
            "delete_file", "search_news", "open_browser",
        ]), f"6 个工具没全调用：{tool_names_used}"

        # 文件真实存在（create_file 真执行了）
        real_p = resolve_user_path(file_alias)
        assert real_p.exists() and real_p.is_file(), "create_file 应真的创建文件"

        # ============== 第二轮（同一 thread_id，测试 Checkpointer 历史保留） ==============
        # 注入一个「只有 1 条 response」的 Mock：拿到历史直接回答引用
        from langchain_core.messages import AIMessage
        from langchain_core.callbacks import CallbackManagerForLLMRun
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.outputs import ChatGeneration, ChatResult

        class EchoMockLLM(BaseChatModel):
            """检查 messages 里是否带了上一轮的 Human + 工具历史，然后输出最终回答。"""
            saw_history_ok: bool = False

            @property
            def _llm_type(self) -> str:
                return "echo-mock-m2-round2"

            def _generate(self, messages, stop=None, run_manager=None, **kw):  # type: ignore[override]
                names = [type(m).__name__ for m in messages]
                # 如果历史被带入，应至少含 Human + Tool + AIMessage 若干
                has_human = any(isinstance(m, HumanMessage) for m in messages)
                has_tool = any(isinstance(m, ToolMessage) for m in messages)
                has_ai = any(isinstance(m, AIMessage) for m in messages)
                self.saw_history_ok = bool(has_human and has_tool and has_ai)
                proof = (
                    f"历史带入证明：看到了 {len(messages)} 条 messages，"
                    f"类型有 {sorted(set(names))}，Human={has_human} Tool={has_tool} AI={has_ai}"
                )
                return ChatResult(generations=[
                    ChatGeneration(message=AIMessage(
                        content=f"✅ 第二轮收到你的追问！同时 Checkpointer 把上一轮的消息历史带进来了：\n{proof}"
                    ))
                ])

        # 替换 agent 内部的 llm_with_tools（简单起见直接新构造一个 Agent，但共用同一个 ckpt + thread_id）
        round2_mock = EchoMockLLM()
        agent2 = AssistantAgent(
            llm=round2_mock,
            tools=agent.tools,  # 复用工具集（第二轮不真调工具，Mock 直接出最终 AIMessage）
            checkpointer=ckpt,
            system_prompt="MOCK R2",
        )
        ans2, _state2 = agent2.run_and_get_state(
            "再问问：刚才第一轮到第 6 个工具分别是什么？（不要调工具，直接看历史回答我）",
            thread_id=thread_id,
        )
        assert "Checkpointer 把上一轮的消息历史带进来了" in ans2, ans2
        # 关键断言：round2_mock 在 _generate 里确实看到了上一轮的 ToolMessage / HumanMessage / AIMessage
        assert round2_mock.saw_history_ok is True, (
            "Checkpointer 历史没带入第二轮："
            "EchoMockLLM._generate 看到的 messages 里缺少 Human/Tool/AI 其中一种。"
        )
    finally:
        # 清理：真的删掉测试文件（通过 DeleteFileTool 真实路径也可，这里直接 unlink 避免二次确认）
        real_p = resolve_user_path(file_alias)
        try:
            if real_p.exists():
                real_p.unlink()
        except Exception:
            pass
        reset_agent_singleton()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
