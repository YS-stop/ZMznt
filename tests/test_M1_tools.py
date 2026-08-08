"""M1 阶段验收：Tool 层（不依赖 LLM，不启动 GUI，纯工具直接 invoke）。

覆盖内容：
    1. path_utils 路径别名解析（resolve_user_path）
    2. CreateFileTool：正常创建 / 父目录自动建立 / overwrite=False 拒覆盖 / 越界 src 被拒 / 黑名单路径被拒
    3. SearchFilesTool：在 数据根 创建一堆文件，按关键字和通配搜索能命中，结果排序正确
    4. OpenBrowserTool + 快捷站点 / 自动补 https / 关键词默认百度搜索 + mock webbrowser.open 真被调用
    5. 导入 src.core.state.AgentState，并传给 StateGraph 不 NameError（对齐 M0 问题）
    6. src.tools.get_all_tools / TOOL_MAP / AVAILABLE_TOOL_NAMES 三 API 正常
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# 加项目根到 sys.path
PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


def test_00_env_ready():
    """基础 import。"""
    from src.utils.path_utils import DATA_ROOT, PROJECT_ROOT  # noqa: F401
    from src.tools import get_all_tools, TOOL_MAP, AVAILABLE_TOOL_NAMES  # noqa: F401
    from src.tools.file_tools import CreateFileTool, SearchFilesTool, resolve_user_path  # noqa: F401
    from src.tools.browser_tools import OpenBrowserTool, resolve_target_to_url  # noqa: F401
    from src.core.state import AgentState, DEFAULT_STATE, STATE_GLOBALS  # noqa: F401


def test_01_tools_registry_ok():
    """get_all_tools / TOOL_MAP / AVAILABLE_TOOL_NAMES 包含 M3 8 个必选工具（M6 起允许 ≥8）。"""
    from src.tools import get_all_tools, TOOL_MAP, AVAILABLE_TOOL_NAMES

    tools = get_all_tools()
    # M0=3 → M1.5=6 → M3=8（新增 voice_input / voice_output） → M6=12（+4 系统工具）
    names_required_M3 = {
        "create_file",
        "search_files",
        "open_browser",
        "delete_file",
        "recognize_file",
        "search_news",
        "voice_input",
        "voice_output",
    }
    actual = {t.name for t in tools}
    assert names_required_M3.issubset(actual), (
        f"M3 必需工具缺失，实际={sorted(actual)}，缺少={names_required_M3 - actual}"
    )
    assert len(actual) >= 8, f"工具数应 ≥ 8，实际={len(actual)}: {sorted(actual)}"
    mp = TOOL_MAP()
    assert names_required_M3.issubset(set(mp.keys())), "TOOL_MAP 缺 M3 必需工具"
    assert names_required_M3.issubset(set(AVAILABLE_TOOL_NAMES())), "AVAILABLE_TOOL_NAMES 缺 M3 必需工具"
    # 名称列表本身是排好序的，顺序校验
    assert AVAILABLE_TOOL_NAMES() == sorted(AVAILABLE_TOOL_NAMES())


def test_02_agent_state_pass_to_langgraph():
    """AgentState 传给 LangGraph StateGraph 构造不抛 NameError（解决 M0 Annotated 问题）。"""
    from typing import get_type_hints

    from langgraph.graph import StateGraph, START, END

    from src.core.state import AgentState, STATE_GLOBALS

    # 能拿到类型注解
    hints = get_type_hints(AgentState, globalns=STATE_GLOBALS, localns=STATE_GLOBALS)
    assert "messages" in hints, "AgentState 缺必需的 messages 字段"

    g = StateGraph(AgentState)
    g.add_node("a", lambda s: {})
    g.add_edge(START, "a")
    g.add_edge("a", END)
    compiled = g.compile()
    assert compiled is not None
    # 调一次（空 messages）不崩
    out = compiled.invoke({"messages": []})
    assert "messages" in out


def test_03_resolve_user_path_aliases():
    """路径别名：桌面/文档/下载/数据根/项目根/家/相对路径 解析正确。"""
    from src.tools.file_tools import resolve_user_path
    from src.utils.path_utils import DATA_ROOT, PROJECT_ROOT

    home = Path(os.path.expanduser("~"))
    tests = {
        "桌面": home / "Desktop",
        "Documents": home / "Documents",
        "文档/工作": home / "Documents" / "工作",
        "下载/aaa.pdf": home / "Downloads" / "aaa.pdf",
        "数据根/notes/2026.md": DATA_ROOT / "notes" / "2026.md",
        "项目根/README.md": PROJECT_ROOT / "README.md",
        "~/hello.txt": home / "hello.txt",
        "相对路径.log": (DATA_ROOT / "相对路径.log").resolve(),
        "C:\\Windows\\SysWOW64\\calc.exe": Path("C:\\Windows\\SysWOW64\\calc.exe"),
    }
    for raw, expect in tests.items():
        got = resolve_user_path(raw)
        assert got.resolve() == expect.resolve(), f"别名解析错 raw={raw}  got={got}  expect={expect}"


def test_04_create_file_normal_and_denials(tmp_path: Path):
    """CreateFileTool：正常创建 + 自动建父目录 + 拒覆盖 + 越界写 src 拒。"""
    from src.tools.file_tools import CreateFileTool

    t = CreateFileTool()
    # —— 1. 数据根下正常创建（路径用别名）——
    sub = f"M1_test_{int(time.time()*1000)}"
    rel = f"数据根/{sub}/一级/二级/你好世界.md"
    content = "# Hello\n这是 M1 验收脚本写入的文件\n" + "".join([f"line {i}\n" for i in range(1, 11)])
    out_ok = t.invoke({"file_path": rel, "content": content})
    assert "✅ create_file 成功" in out_ok, out_ok
    assert "你好世界.md" in out_ok

    # 实际真写进去了
    from src.tools.file_tools import resolve_user_path
    real = resolve_user_path(rel)
    assert real.exists() and real.is_file()
    assert real.read_text(encoding="utf-8") == content

    # —— 2. overwrite=False（默认）时已存在应拒绝 ——
    out_exists = t.invoke({"file_path": rel, "content": "xxx"})
    assert "❌ create_file 失败" in out_exists and "已存在" in out_exists, out_exists
    # 文件没被改
    assert real.read_text(encoding="utf-8") == content

    # —— 3. overwrite=True 时真覆盖 ——
    new_content = "已被覆盖了"
    out_ow = t.invoke({"file_path": rel, "content": new_content, "overwrite": True})
    assert "✅ create_file 成功" in out_ow, out_ow
    assert real.read_text(encoding="utf-8") == new_content

    # —— 4. 直接写 PROJECT_ROOT/src/main.py 应该被拒 ——
    out_deny_src = t.invoke({"file_path": "项目根/src/main.py", "content": "boom"})
    assert "❌ create_file 失败" in out_deny_src and "禁止直接修改 src/" in out_deny_src, out_deny_src

    # —— 5. 写 C:\Windows\xxx 被黑 ——
    out_deny_blk = t.invoke({"file_path": "C:\\Windows\\M1_evil.txt", "content": "evil"})
    assert "❌ create_file 失败" in out_deny_blk and "黑名单" in out_deny_blk, out_deny_blk

    # cleanup
    real.unlink(missing_ok=True)
    try:
        (real.parent.parent.parent).rmdir()  # 一级
    except OSError:
        pass


def test_05_search_files_by_keyword_and_wildcard(tmp_path: Path):
    """SearchFilesTool：创建 6 个文件，按关键字和 *.md 通配搜索，正确命中 / 截断提示。"""
    from src.tools.file_tools import CreateFileTool, SearchFilesTool, resolve_user_path
    from src.utils.path_utils import DATA_ROOT

    sf = SearchFilesTool()
    cf = CreateFileTool()

    tag = f"M1_search_{int(time.time()*1000)}"
    base = f"数据根/{tag}"
    # 造 6 个文件
    files = [
        (f"{base}/项目周报_2026W31.md", "# W31 周报"),
        (f"{base}/项目周报_2026W32.md", "# W32 周报"),
        (f"{base}/项目规划_2026下半年.docx", "这是个 docx（其实是文本）"),
        (f"{base}/会议记录_8月第一周.md", "# 会议 8月"),
        (f"{base}/todo_个人.txt", "待办"),
        (f"{base}/财务_半年报.xlsx", "假 excel"),
    ]
    for p, c in files:
        r = cf.invoke({"file_path": p, "content": c})
        assert "✅ create_file 成功" in r, f"造文件失败 {p}: {r}"

    # —— 关键字「周报」应该命中 2 个 md ——
    r1 = sf.invoke({"query": "周报", "search_root": "数据根", "max_results": 10})
    assert "匹配总数=2" in r1 or ("匹配总数=" in r1 and "项目周报" in r1), r1
    assert "项目周报_2026W31.md" in r1
    assert "项目周报_2026W32.md" in r1

    # —— 通配 *.md 应该命中 3 个（两个周报+一个会议）——
    r2 = sf.invoke({"query": "*.md", "search_root": f"{base}", "max_results": 10})
    assert "项目周报_2026W31.md" in r2
    assert "项目周报_2026W32.md" in r2
    assert "会议记录_8月第一周.md" in r2
    # todo_个人.txt 不应出现
    assert "todo_个人.txt" not in r2

    # —— max_results=1 且实际有 3 个 md，应出现「截断」提示 ——
    r3 = sf.invoke({"query": "*.md", "search_root": f"{base}", "max_results": 1})
    assert "结果截断" in r3 or ("共" in r3 and "只返回前 1" in r3), r3

    # cleanup 整个子目录
    real_base = resolve_user_path(base)
    import shutil
    shutil.rmtree(real_base, ignore_errors=True)


def test_06_open_browser_shortcuts_and_search():
    """OpenBrowserTool + resolve_target_to_url 解析逻辑 + mock webbrowser.open 验证调用。"""
    from src.tools.browser_tools import OpenBrowserTool, resolve_target_to_url

    # —— 1. 快捷站点 ——
    u1, m1 = resolve_target_to_url("知乎")
    assert u1 == "https://www.zhihu.com" and "快捷站点" in m1

    u2, m2 = resolve_target_to_url("GitHub")
    assert u2 == "https://github.com" and "快捷站点" in m2

    # —— 2. 纯域名补 https ——
    u3, m3 = resolve_target_to_url("example.com/path?x=1")
    assert u3 == "https://example.com/path?x=1" and "自动补协议" in m3

    u3b, m3b = resolve_target_to_url("https://a.b.c:8080/hi")
    assert u3b == "https://a.b.c:8080/hi" and "自动补协议" in m3b

    # —— 3. 纯关键词走百度搜索 ——
    u4, m4 = resolve_target_to_url("2026年奥运会赛程表")
    assert "百度搜索关键词" in m4
    assert "https://www.baidu.com/s?wd=" in u4
    # 关键词被 URL 编码
    from urllib.parse import unquote_plus
    assert "2026年奥运会赛程表" in unquote_plus(u4)

    # —— 4. 实际 invoke 调用时真触发 webbrowser.open ——
    tool = OpenBrowserTool()
    with patch("src.tools.browser_tools.webbrowser.open", return_value=True) as mk:
        out = tool.invoke({"target": "知乎", "new_tab": True, "autoraise": True})
        assert mk.call_count == 1, f"mock webbrowser.open 没被调用"
        call_args, _call_kwargs = mk.call_args
        assert call_args[0] == "https://www.zhihu.com"
    assert "✅ open_browser 成功" in out and "快捷站点" in out, out

    # —— 5. 空 target 报错 ——
    out_bad = tool.invoke({"target": "   "})
    assert "❌ open_browser 失败" in out_bad and "target 不能为空" in out_bad, out_bad


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
