r"""headless 把 M5 UI 三大件渲染成 PNG：ball.png / panel.png / settings.png。

运行方式：
    $env:QT_QPA_PLATFORM="offscreen"
    .\venv_assistant\Scripts\python.exe _render_ui_preview.py

输出到 D:/zhuomZNT/data/ui_preview/*.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 先切 offscreen，必须在 QApplication 创建前
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_OFFSCREEN_NO_GLX", "1")

_PROJECT_ROOT: Path = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

OUT_DIR: Path = _PROJECT_ROOT / "data" / "ui_preview"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. QApplication
# ---------------------------------------------------------------------------
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication

app = QApplication.instance() or QApplication(sys.argv)
screen = QGuiApplication.primaryScreen()
if screen is not None:
    print(f"  [INFO] virtual screen size={screen.size().width()}x{screen.size().height()}")


def _save(widget, name: str, w: int | None = None, h: int | None = None) -> Path:
    if w and h:
        widget.resize(w, h)
    widget.createWinId()
    app.style().polish(widget)
    app.sendPostedEvents()
    pix = widget.grab()
    out = OUT_DIR / name
    if pix.isNull():
        from PySide6.QtGui import QPixmap, QPainter, QColor, QFont
        ww = w or widget.width() or 400
        hh = h or widget.height() or 300
        pix = QPixmap(ww, hh)
        pix.fill(QColor("#F1F5F9"))
        p = QPainter(pix)
        p.setFont(QFont("Microsoft YaHei", 12))
        p.setPen(QColor("#0F172A"))
        p.drawText(10, 30, f"[{name}] 渲染占位（grab() 返回空）")
        p.end()
    ok = pix.save(str(out), "PNG")
    print(f"  [SAVE {'OK' if ok else 'FAIL'}] {out.name:40s}  widget={widget.width()}x{widget.height()}  saved={ok}")
    return out


# ---------------------------------------------------------------------------
# 2. P-01 悬浮球（4 状态各一张）
# ---------------------------------------------------------------------------
print("\n=== 渲染 P-01 FloatingBall 4 状态 ===")
from src.ui.floating_ball_widget import FloatingBallWidget, FloatingBallState

ball_states = [
    (FloatingBallState.IDLE, "ball_idle.png"),
    (FloatingBallState.LISTENING, "ball_listening.png"),
    (FloatingBallState.THINKING, "ball_thinking.png"),
    (FloatingBallState.SPEAKING, "ball_speaking.png"),
]
for s, fn in ball_states:
    b = FloatingBallWidget()
    b.setState(s)
    b.resize(72, 72)
    _save(b, fn)


# ---------------------------------------------------------------------------
# 3. P-02 抽屉主面板（塞几条气泡）
# ---------------------------------------------------------------------------
print("\n=== 渲染 P-02 DrawerMainPanel ===")
from src.ui.drawer_main_panel import DrawerMainPanel

dp = DrawerMainPanel()
dp.append_user("你好，帮我在 D 盘创建一个 todo.txt", timestamp="15:20")
dp.append_ai("好的，已为您创建 D:/todo.txt。接下来您想做什么？", timestamp="15:20")
dp.append_user("查一下今天的 AI 新闻", timestamp="15:21")
dp.append_ai("检索完成，今日 AI 资讯 Top3 如下：1) xxx 2) yyy 3) zzz", timestamp="15:21")
dp.append_system("（系统提示：30 分钟后提醒您喝水）")
_save(dp, "panel_chat.png")


# ---------------------------------------------------------------------------
# 4. P-03 调试面板（塞几条日志）
# ---------------------------------------------------------------------------
print("\n=== 渲染 P-03 DebugPanel ===")
from src.ui.debug_panel import DebugPanel

dbg = DebugPanel()
stages_demo = [
    ("ASR",    "[ASR] 录音 2.1s，识别文本：'你好帮我创建文件'  置信度 0.96"),
    ("AGENT",  "[Agent] 选择工具 create_file，参数 target='D:/new.txt'"),
    ("TOOL",   "[Tool] create_file 执行 → 返回 '成功创建，2 字节'"),
    ("OBS",    "[Obs] Tool output → 加入 messages"),
    ("TTS",    "[TTS] Edge-TTS online 合成 1.8s audio → 播放 OK"),
    ("SYSTEM", "[SYSTEM] 回合结束，总耗时 3.42s"),
]
for s, msg in stages_demo:
    dbg.append_stage(s, msg)
_save(dbg, "panel_debug.png")


# ---------------------------------------------------------------------------
# 5. P-04 设置中心（3 页）
# ---------------------------------------------------------------------------
print("\n=== 渲染 P-04 SettingsWindow（3 页） ===")
from src.ui.settings_window import SettingsWindow

sw = SettingsWindow()
pages = [
    (0, "settings_general.png"),
    (1, "settings_voice.png"),
    (3, "settings_aikey.png"),
]
for idx, fn in pages:
    sw.stack.setCurrentIndex(idx)
    app.sendPostedEvents()
    app.style().polish(sw)
    _save(sw, fn)


# ---------------------------------------------------------------------------
# 6. P-05 历史 Tab
# ---------------------------------------------------------------------------
print("\n=== 渲染 P-05 HistoryTabPage ===")
from src.ui.history_tab import HistoryTabPage

hp = HistoryTabPage()
_save(hp, "panel_history.png")


# ---------------------------------------------------------------------------
# 7. P-07 HighRiskConfirmDialog
# ---------------------------------------------------------------------------
print("\n=== 渲染 P-07 HighRiskConfirmDialog ===")
from src.ui.confirm_dialog import HighRiskConfirmDialog

hd = HighRiskConfirmDialog(
    title="⚠️  请确认删除以下 3 个文件",
    ops=[
        "删除 D:/reports/2024-Q1.xlsx",
        "删除 D:/reports/2024-Q2.xlsx  (1.2MB)",
        "删除 D:/reports/temp/  (空目录)",
    ],
    require_keyword="DELETE",
)
_save(hd, "confirm_delete.png")


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
print("\n=== 全部渲染完成 ===")
for p in sorted(OUT_DIR.glob("*.png")):
    kb = p.stat().st_size / 1024
    print(f"  ✅ {p.name:40s}  {kb:7.1f} KB")

app.quit()
