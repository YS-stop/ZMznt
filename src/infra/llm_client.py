"""LLM 客户端封装：主模型统一使用通义千问 Qwen（OpenAI 兼容模式）。

注意：
    - 本文件只负责「主对话 LLM」（Qwen）。
    - 视觉模型（GLM-4.1V / Qwen-VL）由各自的工具直接调用，不在此处，
      见 src/tools/app_tools.py（RecognizeScreenTool）与
      src/tools/web_automation_tools.py（_call_vl_model）。
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
    """获取通义千问 Qwen Chat 实例（OpenAI 兼容模式）。

    Args:
        model: 指定模型系列，默认读 .env QWEN_MODEL，再兜底 qwen-plus
        temperature: 默认 0，工具调用更稳定；创意类写作场景可传 0.7
    """
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    model_name = (model or os.getenv("QWEN_MODEL") or "qwen-plus").strip()
    temp = 0.0 if temperature is None else temperature

    if not api_key:
        try:
            from langchain_core.messages import AIMessage
            from langchain_core.runnables import RunnableLambda

            def _fake_invoke(messages, *args, **kwargs):
                return AIMessage(content="[LLM API Key 未配置] 请在 .env 文件中填入 QWEN_API_KEY 后重启程序。")

            return RunnableLambda(_fake_invoke)  # type: ignore[return-value]
        except Exception:
            raise RuntimeError("未配置 QWEN_API_KEY！请在 .env 中填入后重启程序。")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        temperature=temp,
        timeout=timeout,
    )


def get_main_llm(
    *,
    temperature: Optional[float] = None,
    timeout: int = 60,
) -> BaseChatModel:
    """获取主 LLM 实例（单例）。

    主模型固定使用通义千问 Qwen（由 .env 的 QWEN_MODEL 指定，默认 qwen-plus）。
    仅当 temperature 为 None 时使用模块级单例；传 temperature 时每次新建实例。
    """
    global _llm_instance
    if _llm_instance is not None and temperature is None:
        return _llm_instance

    llm = get_qwen_llm(temperature=temperature, timeout=timeout)
    if temperature is None:
        _llm_instance = llm
    return llm
