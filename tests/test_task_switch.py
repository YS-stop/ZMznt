"""任务打断确认状态机 + 回答分类器 单元测试（offscreen，不联网、不跑真 agent）。"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.services.ui_bridge_service import _classify_switch_answer as cls  # noqa: E402

cases = {
    "确认": "confirm", "好的": "confirm", "可以，换吧": "confirm",
    "取消当前任务": "confirm", "执行新指令": "confirm", "嗯": "confirm",
    "继续": "deny", "不用了": "deny", "不要换": "deny", "取消": "deny",
    "先做完再说": "deny", "别动": "deny",
    "今天天气不错": "unclear", "": "unclear",
}
bad = [(k, cls(k), v) for k, v in cases.items() if cls(k) != v]
print("RESULT classifier:", "ALL PASS" if not bad else f"FAIL {bad}")
assert not bad

from PySide6.QtWidgets import QApplication  # noqa: E402

app = QApplication([])

from src.services.ui_bridge_service import get_ui_bridge  # noqa: E402

b = get_ui_bridge()
started: list[str] = []
b._start_task = lambda content, tid=None: started.append(content)  # type: ignore[assignment]
bubbles: list[str] = []
b.sig_append_system_bubble.connect(lambda t: bubbles.append(t))

# 1. 空闲 → 直接执行
b.submit_text("任务A")
assert started == ["任务A"], started

# 2. 忙碌 → 缓冲 + 询问，不执行
b._task_busy = True
b._current_task_text = "任务A"
b.submit_text("任务B")
assert b._awaiting_switch_answer and b._pending_task_text == "任务B"
assert started == ["任务A"]

# 3. 确认 → 替换执行
b.submit_text("确认")
assert started == ["任务A", "任务B"], started
assert not b._awaiting_switch_answer and b._pending_task_text is None

# 4. 否定 → 保留当前
b._task_busy = True
b._current_task_text = "任务B"
b.submit_text("任务C")
b.submit_text("继续")
assert started == ["任务A", "任务B"], started
assert b._pending_task_text is None and not b._awaiting_switch_answer

# 5. 超时 → 丢弃缓冲
b.submit_text("任务D")
b._on_switch_answer_timeout(b._switch_ask_id)
assert b._pending_task_text is None and not b._awaiting_switch_answer

print("RESULT state machine: ALL PASS")
print("RESULT bubbles:")
for t in bubbles:
    print("  ", t[:60])
