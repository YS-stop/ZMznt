"""SQLite 统一服务：应用配置 + 对话历史 + LangGraph Checkpoint 都写一个数据库。
设计要点：
1. 全局单例连接（get_app_db 函数返回同一个 sqlite3.Connection，线程安全 check_same_thread=False 配合 QSemaphore 写锁。
2. 建表脚本在 init_sqlite() 里执行，用 CREATE TABLE IF NOT EXISTS 幂等。
3. settings 表 KV 结构：key TEXT PRIMARY KEY, value TEXT，读配置直接 SELECT，写用 REPLACE INTO。
"""
from __future__ import annotations
import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from src.utils.path_utils import APP_DB_PATH, ensure_data_dirs

# 全局单例
_app_db_conn: Optional[sqlite3.Connection] = None
_app_db_lock = threading.Lock()  # 写锁（对齐经验：SQLite database is locked 坑位速查表 #9）


def get_app_db() -> sqlite3.Connection:
    """全局单例 SQLite 连接（线程安全）。"""
    global _app_db_conn
    if _app_db_conn is None:
        with _app_db_lock:
            if _app_db_conn is None:
                ensure_data_dirs()
                _app_db_conn = sqlite3.connect(
                    str(APP_DB_PATH),
                    check_same_thread=False,  # Qt worker 线程会写，必须关闭
                    timeout=30.0,              # 并发等待 30s
                    isolation_level=None,      # 自动提交模式
                )
                _app_db_conn.row_factory = sqlite3.Row
                # 并发写安全 PRAGMA
                _app_db_conn.execute("PRAGMA journal_mode = WAL;")
                _app_db_conn.execute("PRAGMA synchronous = NORMAL;")
                _app_db_conn.execute("PRAGMA busy_timeout = 30000;")
    return _app_db_conn


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """事务上下文：BEGIN IMMEDIATE + 提交/回滚；单例连接 + 全局写锁防 database is locked。"""
    conn = get_app_db()
    with _app_db_lock:
        try:
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


# ============ settings KV 表 ============
_SETTINGS_CREATE = """
CREATE TABLE IF NOT EXISTS settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# ============ chat_history 对话历史表 ============
_HISTORY_CREATE = """
CREATE TABLE IF NOT EXISTS chat_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id  TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    extra      TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""
_HISTORY_INDEX = "CREATE INDEX IF NOT EXISTS idx_chat_history_thread ON chat_history(thread_id)"

# ============ langgraph_checkpoints 表（LangGraph SQLiteCheckpointSaver 用，先占位；M2 才换正式 saver）============
_CHECKPOINTS_CREATE = """
CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
    checkpoint_id  TEXT PRIMARY KEY,
    thread_id      TEXT NOT NULL,
    parent_id      TEXT,
    checkpoint     BLOB NOT NULL,
    metadata       TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""
_CHECKPOINTS_INDEX = "CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON langgraph_checkpoints(thread_id)"

# ============ memories 长期记忆表（记住「之前做过什么 / 用户偏好 / 事实」）============
_MEMORIES_CREATE = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    mem_type     TEXT NOT NULL,                -- fact / preference / task / other
    content      TEXT NOT NULL,
    importance   INTEGER DEFAULT 1,            -- 1(低)~5(高)
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0
)
"""
_MEMORIES_INDEX_IMP = "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);"
_MEMORIES_INDEX_CREATED = "CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);"

# 所有建表 SQL 单条执行（SQLite 限制：conn.execute 一次只能一条语句）
_INIT_SQL_STATEMENTS = (
    _SETTINGS_CREATE,
    _HISTORY_CREATE,
    _HISTORY_INDEX,
    _CHECKPOINTS_CREATE,
    _CHECKPOINTS_INDEX,
    _MEMORIES_CREATE,
    _MEMORIES_INDEX_IMP,
    _MEMORIES_INDEX_CREATED,
)


def init_sqlite() -> None:
    """初始化 SQLite：建三张表+索引，全部单条 execute 避免 SQLite 多语句报错。"""
    conn = get_app_db()
    with transaction():
        for sql in _INIT_SQL_STATEMENTS:
            conn.execute(sql)
    _ensure_default_settings()


# ---------- settings KV helpers ----------
def get_setting(key: str, default: Any = None) -> Any:
    row = get_app_db().execute(
        "SELECT value FROM settings WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row["value"])
    except (json.JSONDecodeError, TypeError):
        return row["value"]


def set_setting(key: str, value: Any) -> None:
    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    with transaction() as c:
        c.execute(
            "REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, serialized),
        )


def _ensure_default_settings() -> None:
    """启动时写默认配置（仅当 key 不存在时写入）。."""
    defaults = {
        "hotkey_show": "Ctrl+Alt+D",
        "hotkey_record": "Ctrl+Alt+V",
        "wake_word": "小助手",
        "wake_word_enabled": True,
        "enable_tts": True,
        "debug_panel_open": True,
        "drawer_side": "right",
        "theme": "light-blue",
        "primary_color": "#3B82F6",
        "default_search_path": "桌面",
        "high_risk_confirm_level": "high",
        # —— 长期记忆（记忆服务开关，默认开）——
        "memory_enabled": True,
        # —— M7 持续监听 / 智能尾点检测（设置中心可视化调整，热生效）——
        "vad_tail_silence_sec": 1.2,   # 尾点静音阈值：连续静音多久判定发言结束
        "vad_max_record_sec": 30,      # 最长录音时长：超时强制提交
        "vad_await_speech_sec": 5,     # 唤醒后等待开口超时（随后语音提示 + 3s 宽限退出）
        "vad_threshold_ratio": 3.0,    # 能量阈值系数（嘈杂环境调高，安静环境调低）
    }
    with transaction() as c:
        for k, v in defaults.items():
            c.execute(
                "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                (k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)),
            )
