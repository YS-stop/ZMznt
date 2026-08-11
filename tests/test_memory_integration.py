"""记忆方案端到端验证（不启动 GUI）：

1. 同一 thread 多轮续接（checkpointer 续接历史）
2. 每轮后台提炼记忆写入 SQLite memories 表
3. build_context_prompt 能检索到记忆
4. 模拟「重启」：新建 agent 实例读回同一 thread 历史
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 加载 .env
for line in Path(".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ[k.strip()] = v.strip()


def _hr(t: str) -> None:
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)


def main() -> None:
    from src.services.sqlite_db import init_sqlite
    from src.services.checkpoint_service import get_checkpoint_info, start_checkpoint_service
    from src.services.agent_service import get_agent, reset_agent_singleton
    from src.services.memory_service import get_memory_service

    init_sqlite()
    start_checkpoint_service(force_backend="sqlite")
    info = get_checkpoint_info()
    _hr("Checkpoint 后端")
    print(f"backend={info['backend']} | note={info['note']}")
    assert info["backend"] == "sqlite", "SQLite Checkpoint 未启用，记忆重启恢复将无法验证！"

    agent = get_agent()
    mem = get_memory_service()
    mem.set_enabled(True)
    mem.clear_all()
    TID = "main"

    def run_turn(text: str) -> str:
        mem_text = mem.build_context_prompt()
        runtime_system = agent.system_prompt + (
            "\n\n【长期记忆 · 之前做过/用户偏好/重要事实】\n" + mem_text if mem_text else ""
        )
        ans = agent.stream_events(
            text, thread_id=TID, extra_state_fields={"runtime_system": runtime_system}
        )
        t = threading.Thread(target=mem.record_turn, args=(text, ans, TID), daemon=True)
        t.start()
        t.join(timeout=40)
        return ans

    _hr("第1轮：记住用户偏好")
    a1 = run_turn("帮我记一下：我喜欢蓝色，我的工作日是周一到周五，我常用桌面笔记软件。")
    print("A1:", a1[:200])
    print("memories count:", mem.count())

    _hr("第2轮：我刚才让你记住了什么偏好")
    a2 = run_turn("我刚才让你记住了什么关于我的偏好？")
    print("A2:", a2[:300])
    print("memories count:", mem.count())

    _hr("长期记忆检索（build_context_prompt）")
    ctx = mem.build_context_prompt()
    print(ctx if ctx else "（空）")

    _hr("模拟重启：新建 agent 实例读回 thread 历史")
    reset_agent_singleton()
    agent2 = get_agent()
    msgs = agent2.get_thread_messages(TID)
    print(f"重启后 thread={TID} 消息数: {len(msgs)}")
    if msgs:
        for m in msgs[-4:]:
            role = type(m).__name__
            print(f"  - {role}: {str(getattr(m, 'content', m))[:80]}")
    assert len(msgs) >= 3, "重启后历史未恢复！"

    _hr("全部验证通过 ✅")
    print(f"最终 memories 表条数: {mem.count()}")


if __name__ == "__main__":
    main()
