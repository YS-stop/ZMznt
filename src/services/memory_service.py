"""长期记忆服务：把每轮对话提炼成结构化记忆存入 SQLite，下次对话前检索注入。

记忆方案分两层（协同工作）：
- 短期（会话连续 + 跨重启）：由 LangGraph SQLite Checkpoint 保证同一 thread_id 的历史
  在进程重启后仍能恢复（见 checkpoint_service）。
- 长期（跨会话「记得做过什么」）：本服务负责沉淀用户偏好 / 已完成任务 / 重要事实，
  避免 LLM 上下文窗口有限导致早期信息被遗忘；也支持切换会话后仍能回忆。

数据流：
  每轮对话结束 → 后台线程 record_turn() 用 LLM 提炼记忆 → 写入 memories 表
  每轮对话开始 → build_context_prompt() 取高重要性 + 最近记忆 → 拼进 system prompt 注入
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Optional

from src.services.sqlite_db import get_app_db, transaction


MEM_TYPE_FACT = "fact"              # 事实 / 信息
MEM_TYPE_PREFERENCE = "preference"  # 用户偏好
MEM_TYPE_TASK = "task"              # 已完成 / 待办任务
MEM_TYPE_OTHER = "other"

_DEFAULT_MAIN_THREAD = "main"
_TYPE_TAGS = {
    MEM_TYPE_FACT: "📌事实",
    MEM_TYPE_PREFERENCE: "⭐偏好",
    MEM_TYPE_TASK: "✅做过",
    MEM_TYPE_OTHER: "📝其他",
}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class MemoryService:
    """长期记忆：SQLite + 后台 LLM 提炼 + 注入检索。单例。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = True

    # ————————————————————————————————
    # 开关
    # ————————————————————————————————
    def set_enabled(self, on: bool) -> None:
        self._enabled = bool(on)

    def is_enabled(self) -> bool:
        return self._enabled

    # ————————————————————————————————
    # 写入
    # ————————————————————————————————
    def add_memory(self, mem_type: str, content: str, importance: int = 2) -> int:
        """写入一条记忆，返回 id（0 表示未写入）。"""
        content = (content or "").strip()
        if not content:
            return 0
        mem_type = str(mem_type or MEM_TYPE_OTHER)
        importance = max(1, min(5, int(importance)))
        try:
            with transaction() as c:
                cur = c.execute(
                    "INSERT INTO memories (mem_type, content, importance, created_at, last_used, access_count) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (mem_type, content, importance, _now_iso(), _now_iso()),
                )
                return int(cur.lastrowid)
        except Exception:  # noqa: BLE001 - 数据库异常不阻断主流程
            return 0

    def touch(self, mem_id: int) -> None:
        """更新访问时间 + 计数（检索注入时调用，便于后续做热度排序）。"""
        try:
            with transaction() as c:
                c.execute(
                    "UPDATE memories SET last_used=?, access_count=access_count+1 WHERE id=?",
                    (_now_iso(), int(mem_id)),
                )
        except Exception:  # noqa: BLE001
            pass

    # ————————————————————————————————
    # 读取（构建注入文本）
    # ————————————————————————————————
    def get_recent(self, limit: int = 18) -> list[dict]:
        rows = get_app_db().execute(
            "SELECT id, mem_type, content, importance, created_at FROM memories "
            "ORDER BY importance DESC, created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]

    def build_context_prompt(self, max_items: int = 18, max_chars: int = 1500) -> str:
        """生成注入 system prompt 的长期记忆文本块（空串=无记忆）。"""
        if not self._enabled:
            return ""
        rows = self.get_recent(limit=max_items)
        if not rows:
            return ""
        for r in rows:
            try:
                self.touch(int(r["id"]))
            except Exception:  # noqa: BLE001
                pass
        lines = []
        for r in rows:
            tag = _TYPE_TAGS.get(r["mem_type"], "📝")
            lines.append(f"- {tag}（重要度{r['importance']}）{r['content']}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…（更多记忆已省略）"
        return text

    def count(self) -> int:
        try:
            row = get_app_db().execute("SELECT COUNT(*) AS n FROM memories").fetchone()
            return int(row["n"]) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    def clear_all(self) -> int:
        """清空全部记忆（返回删除条数）。"""
        try:
            with transaction() as c:
                cur = c.execute("DELETE FROM memories")
                return cur.rowcount
        except Exception:  # noqa: BLE001
            return 0

    # ————————————————————————————————
    # 提炼（LLM，后台线程调用）
    # ————————————————————————————————
    def record_turn(self, user_text: str, final_text: str, thread_id: str = _DEFAULT_MAIN_THREAD) -> int:
        """后台线程调用：用 LLM 从本轮对话提炼记忆并写入，返回新增条数。"""
        if not self._enabled:
            return 0
        user_text = (user_text or "").strip()
        final_text = (final_text or "").strip()
        if not user_text or not final_text:
            return 0
        try:
            items = self._extract_memories_via_llm(user_text, final_text)
        except Exception:  # noqa: BLE001
            return 0
        added = 0
        for it in items:
            try:
                t = str(it.get("type") or MEM_TYPE_OTHER)
                c = str(it.get("content") or "").strip()
                imp = int(it.get("importance") or 2)
                if c:
                    self.add_memory(t, c, imp)
                    added += 1
            except Exception:  # noqa: BLE001
                continue
        return added

    def _extract_memories_via_llm(self, user_text: str, final_text: str) -> list[dict]:
        from langchain_core.messages import HumanMessage, SystemMessage

        from src.infra.llm_client import get_main_llm

        llm = get_main_llm()
        sys_prompt = (
            "你是记忆提取器。阅读用户与助手的本轮对话，提取【值得长期记住】的信息，"
            "用于让助手在未来对话中回忆用户偏好、已完成的任务、重要事实。\n"
            "规则：\n"
            "1. 只提取稳定的、跨会话有用的信息（用户偏好、已完成的任务/打开过的应用/创建过的文件、重要事实）。\n"
            "2. 不要提取一次性闲聊、工具报错细节、或可从最近对话直接推断的临时内容。\n"
            "3. 每条给 type（fact/preference/task/other）和 importance（1~5，5最重要）。\n"
            "4. 必须只输出 JSON 数组，例如："
            "[{\"type\":\"preference\",\"content\":\"用户喜欢用红色标记重点\",\"importance\":3},"
            "{\"type\":\"task\",\"content\":\"2026-08-10 为用户在桌面创建了周报文件\",\"importance\":2}]\n"
            "5. 若没有值得记的，输出空数组 []。"
        )
        user_msg = f"【用户】{user_text}\n\n【助手】{final_text[:1200]}"
        try:
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_msg)])
        except Exception:  # noqa: BLE001
            return []
        text = (getattr(resp, "content", "") or "").strip()
        return self._parse_memory_json(text)

    @staticmethod
    def _parse_memory_json(text: str) -> list[dict]:
        if not text:
            return []
        s = text
        # 去 markdown 代码块包裹
        if "```" in s:
            parts = s.split("```")
            if len(parts) >= 2:
                s = parts[1]
                if s.lstrip().startswith("json"):
                    s = s.lstrip()[4:]
        a, b = s.find("["), s.rfind("]")
        if a == -1 or b == -1:
            return []
        s = s[a : b + 1]
        try:
            data = json.loads(s)
        except Exception:  # noqa: BLE001
            return []
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        return []


# ——————————————————————————————————————————————————————————
# 模块级单例
# ——————————————————————————————————————————————————————————
_MEM_SVC = MemoryService()


def get_memory_service() -> MemoryService:
    return _MEM_SVC


__all__ = [
    "MemoryService",
    "get_memory_service",
    "MEM_TYPE_FACT",
    "MEM_TYPE_PREFERENCE",
    "MEM_TYPE_TASK",
    "MEM_TYPE_OTHER",
]
