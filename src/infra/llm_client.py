"""LLM 客户端封装：通义千问 Qwen，使用 OpenAI 兼容模式（LangChain ChatOpenAI）。
M0 阶段只封装统一入口函数；M1 阶段开始在 LangGraph agent_node 里调用。
"""
from __future__ import annotations
import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

_llm_instance: Optional[BaseChatModel] = None


def get_qwen_llm(
    *,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    timeout: int = 60,
) -> BaseChatModel:
    """全局单例 Qwen Chat 实例。M1 开始绑 tools 用 llm.bind_tools(tools)。

    Args:
        model: 指定模型系列，默认读 .env QWEN_MODEL，再兜底 qwen-plus
        temperature: 默认 0，工具调用更稳定；创意类写作场景可传 0.7
    """
    global _llm_instance
    if _llm_instance is not None and model is None and temperature is None:
        return _llm_instance

    api_key = os.getenv("QWEN_API_KEY", "").strip()
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    model_name = (model or os.getenv("QWEN_MODEL") or "qwen-plus").strip()
    temp = 0.0 if temperature is None else temperature

    # M0 无 Key 不抛错，给用户友好提示；M1 真调用时再校验
    if not api_key:
        # 返回一个假实例占位，避免启动报错；M1 验收阶段再启用真实 Key
        try:
            from langchain_core.messages import AIMessage
            from langchain_core.runnables import RunnableLambda

            def _fake_invoke(messages, *args, **kwargs):  # type: ignore[no-untyped-def]
                return AIMessage(content="[QWEN_API_KEY 未配置] 请在项目根目录 .env 文件里填入 QWEN_API_KEY 后重启程序。")

            return RunnableLambda(_fake_invoke)  # type: ignore[return-value]
        except Exception:
            raise RuntimeError(
                "未配置 QWEN_API_KEY！请在项目根目录 .env 文件中填入 QWEN_API_KEY=sk-..."
            )

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temp,
        timeout=timeout,
    )
    if model is None and temperature is None:
        _llm_instance = llm
    return llm
