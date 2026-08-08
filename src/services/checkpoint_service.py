"""LangGraph Checkpoint 服务：根据环境选择后端并持有单例。

当前 langgraph 包内置：
    ✅ langgraph.checkpoint.memory.MemorySaver（纯内存，随进程丢失，适合测试/开发）
可选增强（需要额外安装独立包）：
    📦 langgraph-checkpoint-sqlite → SqliteSaver（跨进程持久化，SQLite 文件）

选择策略（按优先级）：
    1. 环境变量 CHECKPOINT_BACKEND=sqlite → 尝试启用 SQLite 后端，失败降级 memory 并记录原因。
    2. 其他值或缺省 → MemorySaver。

生命周期：
    本模块暴露一个「CheckpointService」单例，应用启动时调用 .start() 进入上下文，
    退出时调用 .stop() 释放资源（SQLite 后端需关闭连接）。
    不允许用户在函数内手动调 with SqliteSaver.from_conn_string(...)() __enter__，
    避免丢失上下文管理协议（对齐经验 1035072 的失败教训）。
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

# 确保 import src.*
_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.utils.path_utils import APP_DB_PATH, ensure_data_dirs  # noqa: E402


# LangGraph 的 BaseCheckpointSaver 类型（仅用于注解）
try:  # pragma: no cover - 分支保护
    from langgraph.checkpoint.base import BaseCheckpointSaver  # type: ignore
except Exception:  # noqa: BLE001
    class BaseCheckpointSaver:  # type: ignore
        """兜底空类，避免 import 失败时后续类定义崩。"""
        pass


class CheckpointService:
    """Checkpointer 生命周期与单例管理。

    属性：
        backend        : "memory" | "sqlite"（实际生效的后端）
        backend_note   : 说明字符串（降级原因、安装提示等，调试面板可展示）
        saver          : 实际可用的 BaseCheckpointSaver 实例
        started        : 是否已启动（start() 之后才能拿 saver）
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started: bool = False
        self.backend: str = "memory"
        self.backend_note: str = ""
        self.saver: Optional[BaseCheckpointSaver] = None
        # sqlite 专用：持有 with 块内取到的真实 saver + cm 引用
        self._sqlite_cm: Any = None  # SqliteSaver.from_conn_string 返回的 _GeneratorContextManager
        self._sqlite_saver: Any = None  # 通过 __enter__ 拿到的真实实例

    # —————————————————————————————————————————————————————————————
    # 生命周期 API
    # —————————————————————————————————————————————————————————————

    def start(self, force_backend: Optional[str] = None) -> "CheckpointService":
        """启动 Checkpointer（应用启动时调一次，可重入多次幂等）。

        force_backend: 显式指定 "memory" / "sqlite"；None 时读环境变量 CHECKPOINT_BACKEND。
        """
        if self.started:
            return self
        with self._lock:
            if self.started:
                return self
            # 决定期望后端
            desired = (force_backend or "").strip().lower()
            if not desired:
                desired = (os.environ.get("CHECKPOINT_BACKEND") or "memory").strip().lower()
            if desired not in ("memory", "sqlite"):
                desired = "memory"

            if desired == "sqlite":
                ok, note, saver, cm = self._try_start_sqlite()
                if ok:
                    self.backend = "sqlite"
                    self.backend_note = note
                    self.saver = saver
                    self._sqlite_saver = saver
                    self._sqlite_cm = cm
                else:
                    # 降级到 memory
                    self.backend = "memory"
                    self.backend_note = (
                        f"CHECKPOINT_BACKEND=sqlite 但启用失败，已降级 memory。原因：{note}"
                    )
                    self.saver = self._new_memory_saver()
            else:
                self.backend = "memory"
                self.backend_note = "默认 MemorySaver（进程内会话，重启丢失）"
                self.saver = self._new_memory_saver()

            self.started = True
            return self

    def stop(self) -> None:
        """应用关闭时释放资源（SQLite 后端会退出上下文关闭连接）。"""
        with self._lock:
            if not self.started:
                return
            # sqlite: 调 cm.__exit__ 安全关闭
            if self._sqlite_cm is not None:
                try:
                    self._sqlite_cm.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    pass
                self._sqlite_cm = None
                self._sqlite_saver = None
            self.saver = None
            self.backend_note = ""
            self.backend = "memory"
            self.started = False

    # —————————————————————————————————————————————————————————————
    # 对外取 saver（未 start 先自动 start，避免漏调）
    # —————————————————————————————————————————————————————————————

    def get(self) -> BaseCheckpointSaver:
        """获取已就绪的 checkpointer 实例（自动懒启动）。"""
        if not self.started:
            self.start()
        if self.saver is None:  # 理论上 start 后不会 None
            raise RuntimeError("CheckpointService.start() 完成但 saver 仍为 None，请检查安装依赖")
        return self.saver  # type: ignore[return-value]

    def get_info(self) -> dict[str, str]:
        """调试面板用：返回后端摘要字典。"""
        return {
            "backend": self.backend,
            "started": "true" if self.started else "false",
            "note": self.backend_note or "OK",
            "env_hint": (
                "如需持久化：pip install langgraph-checkpoint-sqlite 且 set CHECKPOINT_BACKEND=sqlite，"
                "会写入和 app.db 同目录的 langgraph_checkpoints.db"
            ) if self.backend == "memory" else "",
        }

    # —————————————————————————————————————————————————————————————
    # 内部：memory / sqlite 各自构造
    # —————————————————————————————————————————————————————————————

    @staticmethod
    def _new_memory_saver() -> BaseCheckpointSaver:
        from langgraph.checkpoint.memory import MemorySaver  # type: ignore

        return MemorySaver()

    @staticmethod
    def _try_start_sqlite() -> tuple[bool, str, Any, Any]:
        """尝试 SqliteSaver：返回 (成功?, 说明, saver实例, cm引用)。

        关键：SqliteSaver.from_conn_string(...) 返回的是 **上下文管理器**，
        必须在此处调 __enter__() 持有真实 saver，并在 stop() 时 __exit__ 关连接；
        严禁只取 __enter__() 后丢弃 cm 引用（对齐经验 1035072）。
        """
        # 1) 尝试 import SqliteSaver（来自独立包 langgraph-checkpoint-sqlite，或 langgraph.checkpoint.sqlite）
        sqlitesaver_cls: Any = None
        import_paths_tried: list[str] = []
        for mod_name, cls_name in [
            ("langgraph_checkpoint_sqlite", "SqliteSaver"),
            ("langgraph.checkpoint.sqlite", "SqliteSaver"),
        ]:
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
                sqlitesaver_cls = getattr(mod, cls_name, None)
                if sqlitesaver_cls is not None:
                    break
            except Exception:  # noqa: BLE001
                import_paths_tried.append(mod_name)
                continue
        if sqlitesaver_cls is None:
            return (
                False,
                "未找到 SqliteSaver，已尝试 import 路径：" + "、".join(import_paths_tried) +
                "。解决：pip install langgraph-checkpoint-sqlite（并确保使用同步 invoke 而非 ainvoke，因 SqliteSaver 不支持 async）",
                None,
                None,
            )
        # 2) 确保数据目录存在
        try:
            ensure_data_dirs()
        except Exception as e:  # noqa: BLE001
            return False, f"ensure_data_dirs 失败：{type(e).__name__}: {e}", None, None

        # 3) 构造 SQLite 文件路径（与 app.db 并列，避免和 app.db 单例连接冲突）
        ckpt_db = APP_DB_PATH.with_name("langgraph_checkpoints.db")
        conn_str = f"sqlite:///{ckpt_db}"

        # 4) from_conn_string → cm → 立刻 enter，拿到真 saver + 保留 cm 引用
        try:
            cm = sqlitesaver_cls.from_conn_string(conn_str)
        except Exception as e:  # noqa: BLE001
            return False, f"SqliteSaver.from_conn_string({conn_str}) 抛错：{type(e).__name__}: {e}", None, None
        try:
            saver = cm.__enter__()
        except Exception as e:  # noqa: BLE001
            # 进入失败尝试关闭 cm
            try:
                cm.__exit__(type(e), e, getattr(e, "__traceback__", None))
            except Exception:  # noqa: BLE001
                pass
            return False, f"SqliteSaver cm.__enter__ 抛错：{type(e).__name__}: {e}", None, None
        return (
            True,
            f"SqliteSaver 持久化已启动，DB={ckpt_db}（同步接口，仅 invoke/stream；若用 ainvoke 请换 AsyncSqliteSaver+aiosqlite）",
            saver,
            cm,
        )


# ——————————————————————————————————————————————————————————————————
# 模块级单例（全局共享）
# ——————————————————————————————————————————————————————————————————
_CHECKPOINT_SVC = CheckpointService()


def get_checkpointer() -> BaseCheckpointSaver:
    """推荐：直接函数式调用，懒启动。"""
    return _CHECKPOINT_SVC.get()


def get_checkpoint_info() -> dict[str, str]:
    """调试面板展示摘要。"""
    return _CHECKPOINT_SVC.get_info()


def start_checkpoint_service(force_backend: Optional[str] = None) -> CheckpointService:
    """应用启动时显式调（可用环境变量或 force_backend 指定 memory/sqlite）。"""
    return _CHECKPOINT_SVC.start(force_backend=force_backend)


def stop_checkpoint_service() -> None:
    """应用退出时调一次（多进程/嵌入式要关连接）。"""
    _CHECKPOINT_SVC.stop()


__all__ = [
    "CheckpointService",
    "get_checkpointer",
    "get_checkpoint_info",
    "start_checkpoint_service",
    "stop_checkpoint_service",
]
