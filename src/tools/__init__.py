"""工具包统一入口。

推荐用法：
    from src.tools import get_all_tools, TOOL_MAP

    tools = get_all_tools()                 # 拿到 list[BaseTool]，给 agent.bind_tools()
    tool_node = ToolNode(list(TOOL_MAP.values()))  # 给 LangGraph ToolNode

设计说明：
    * 用「函数内懒加载」避免模块级循环 import（某个子工具未来若 import 对话服务/LLM，就不会和上层形成环）。
    * TOOL_MAP 是 { tool.name : tool_instance }，便于 LangGraph 路由或根据名字查找工具元信息。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


__all__ = ["get_all_tools", "TOOL_MAP", "AVAILABLE_TOOL_NAMES"]

# —— 缓存：单例工具实例（BaseTool 是无状态的，一次初始化反复用即可）——
_TOOL_CACHE: dict[str, "BaseTool"] = {}
_INIT_DONE = False


def _build_all() -> dict[str, "BaseTool"]:
    """懒加载所有 Tool 类并实例化 → 返回 {name: instance}。
    内部 import 避免模块级循环依赖。
    """
    # —— 子工具 ——
    from src.tools.file_tools import (
        CreateFileTool,
        SearchFilesTool,
        DeleteFileTool,
        RecognizeFileTool,
    )
    from src.tools.browser_tools import (
        OpenBrowserTool,
        CloseBrowserTabTool,
        RestoreClosedTabTool,
    )
    from src.tools.news_tools import SearchNewsTool
    from src.tools.voice_tools import VoiceInputTool, VoiceOutputTool
    from src.tools.system_tools import get_all_system_tools  # M6.2 新增
    from src.tools.app_tools import (  # 桌面应用 + 屏幕监控
        OpenAppTool,
        ListActiveAppsTool,
        RecognizeScreenTool,
    )

    instances: list[BaseTool] = [
        CreateFileTool(),
        SearchFilesTool(),
        OpenBrowserTool(),
        CloseBrowserTabTool(),
        RestoreClosedTabTool(),
        # M1.5 新增
        DeleteFileTool(),
        RecognizeFileTool(),
        SearchNewsTool(),
        # M3 新增（VoiceOutput 是占位，先注册保证 7 工具完整，M4 替换实现）
        VoiceInputTool(),
        VoiceOutputTool(),
        # —— M6.2 新增 4 个系统工具：音量/截图/电源/翻译 ——
        *get_all_system_tools(),
        # —— 桌面应用 + 屏幕监控 ——
        OpenAppTool(),
        ListActiveAppsTool(),
        RecognizeScreenTool(),
        # —— 后续阶段新增的工具在这里 append ——
    ]
    out: dict[str, BaseTool] = {}
    for inst in instances:
        if getattr(inst, "name", None):
            out[inst.name] = inst
    return out


def get_all_tools() -> "list[BaseTool]":
    """拿到所有工具实例（list 形式），给 LangChain agent.bind_tools() 用。"""
    global _INIT_DONE, _TOOL_CACHE
    if not _INIT_DONE:
        _TOOL_CACHE = _build_all()
        _INIT_DONE = True
    return list(_TOOL_CACHE.values())


def TOOL_MAP() -> dict[str, "BaseTool"]:  # noqa: N802 —— 对外暴露成常量风格 API
    """拿到 {tool.name -> instance} 字典，给 LangGraph ToolNode / 按名字查找工具。"""
    global _INIT_DONE, _TOOL_CACHE
    if not _INIT_DONE:
        _TOOL_CACHE = _build_all()
        _INIT_DONE = True
    return dict(_TOOL_CACHE)  # 返回浅拷贝，防止外部误改缓存


def AVAILABLE_TOOL_NAMES() -> list[str]:  # noqa: N802
    """拿到当前已注册的工具名称列表（调试 / UI 展示用）。"""
    return sorted(TOOL_MAP().keys())
