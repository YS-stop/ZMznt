"""M0 非 GUI 验收脚本：不启动 Qt，单独验证所有核心骨架代码是否正常。
运行：在 venv 激活后执行 pytest tests/test_M0_skeleton.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_env_loaded():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    assert (PROJECT_ROOT / ".env").exists(), ".env 文件不存在"


def test_path_utils():
    from src.utils.path_utils import (
        PROJECT_ROOT,
        DATA_ROOT,
        CHECKPOINTS_DIR,
        WHOOSH_INDEX_DIR,
        OCR_TEMP_DIR,
        TTS_CACHE_DIR,
        ASSETS_DIR,
        APP_DB_PATH,
        ensure_data_dirs,
    )
    # 根路径对齐
    assert PROJECT_ROOT.exists(), f"PROJECT_ROOT {PROJECT_ROOT} 不存在"
    assert (PROJECT_ROOT / "src" / "main.py").exists(), "src/main.py 找不到"

    # 自动建目录
    ensure_data_dirs()
    for d in (DATA_ROOT, CHECKPOINTS_DIR, WHOOSH_INDEX_DIR, OCR_TEMP_DIR, TTS_CACHE_DIR, ASSETS_DIR):
        assert d.exists() and d.is_dir(), f"数据目录没创建: {d}"

    # SQLite 路径
    assert str(APP_DB_PATH).endswith("app.db"), "APP_DB_PATH 路径不对"


def test_sqlite_db():
    from src.utils.path_utils import ensure_data_dirs
    ensure_data_dirs()
    from src.services.sqlite_db import init_sqlite, get_app_db, get_setting, set_setting, APP_DB_PATH

    init_sqlite()
    assert APP_DB_PATH.exists(), "app.db 没创建"

    # 连接 & 单例
    c1 = get_app_db()
    c2 = get_app_db()
    assert c1 is c2, "SQLite 没做成单例"

    # 三张表存在
    rows = c1.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('settings','chat_history','langgraph_checkpoints')"
    ).fetchall()
    table_names = {r["name"] for r in rows}
    assert {"settings", "chat_history", "langgraph_checkpoints"}.issubset(table_names), f"缺表: {table_names}"

    # 默认 settings 写入
    wake = get_setting("wake_word")
    assert wake == "小助手", f"默认 settings 缺 wake_word，实际: {wake}"

    # KV 读写正常
    set_setting("m0_test_key", "m0_test_value_12345")
    assert get_setting("m0_test_key") == "m0_test_value_12345"
    # 清掉测试行
    c1.execute("DELETE FROM settings WHERE key='m0_test_key'")


def test_imports_framework():
    """核心框架 import 和 StateGraph 可以正常实例化，不报错。"""
    # LangChain / LangGraph
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage  # noqa: F401
    from langgraph.graph import StateGraph, START, END  # noqa: F401
    from typing import TypedDict, Annotated, get_type_hints
    import operator

    # 把所有类型提前注入 globals 避免 get_type_hints 解析时 NameError
    ns = {
        "TypedDict": TypedDict,
        "Annotated": Annotated,
        "BaseMessage": BaseMessage,
        "operator": operator,
        "list": list,
    }
    S = TypedDict("S", {"messages": Annotated[list[BaseMessage], operator.add]})
    # 显式传 namespace 给 StateGraph （LangGraph 支持 schema 参数里传递 namespace，或直接传类型）
    g = StateGraph(S)
    assert g is not None
    # 可以成功 add_node 不报错
    g.add_node("dummy", lambda state: {})
    g.add_edge(START, "dummy")
    g.add_edge("dummy", END)
    compiled = g.compile()
    assert compiled is not None

    # PySide6 可以 import（不启动 QApplication）
    from PySide6.QtCore import Qt  # noqa: F401
    from PySide6.QtWidgets import QWidget  # noqa: F401


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--color=yes"])
