"""M5.5 桌面应用编排层：AppController。

负责：
    - 初始化 QApplication（若不存在）
    - 装配：悬浮球 / 抽屉 / 设置窗口 / 调试面板 / 历史 Tab / 托盘菜单
    - 所有互连线信号槽：点击悬浮球 → 抽屉开合 / 右键 → 退出 等
    - 暴露 run() / quit_app() 给 main.py

后续 M5-6：把 UIBridgeService 接入到 AppController（submit_text → AgentService.run 异步执行）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
)

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 导入 UI 组件
from src.ui.floating_ball_widget import FloatingBallWidget, FloatingBallState  # noqa: E402
from src.ui.drawer_main_panel import DrawerMainPanel  # noqa: E402
from src.ui.settings_window import SettingsWindow  # noqa: E402
from src.ui.debug_panel import DebugPanel  # noqa: E402
from src.ui.history_tab import HistoryTabPage  # noqa: E402


def _make_color_pixmap(color_hex: str, size: int = 24) -> QPixmap:
    """画一个纯色渐变的圆形图标，作为托盘图标（避免依赖外部 .ico/.png 文件）。"""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    from PySide6.QtGui import QLinearGradient, QBrush
    grd = QLinearGradient(0, 0, size, size)
    col = QColor(color_hex)
    lighter = QColor(col)
    lighter = lighter.lighter(125)
    grd.setColorAt(0.0, lighter)
    grd.setColorAt(1.0, col)
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(grd))
    p.drawEllipse(1, 1, size - 2, size - 2)
    # 内部白色小圆点高光
    p.setBrush(QColor(255, 255, 255, 180))
    p.drawEllipse(4, 4, 7, 7)
    p.end()
    return pm


class AppController(QObject):
    """桌面应用主控：非 QApplication 子类，组合装配所有 UI 组件。"""

    # 对外业务信号（M5-6 UIBridgeService 监听）
    sig_user_submit_text = Signal(str)
    sig_user_voice_start = Signal()
    sig_user_voice_end = Signal()
    sig_user_tts_toggled = Signal(bool)
    sig_request_settings_saved = Signal(str, object)

    # UI 控制信号（发给桥接层）
    sig_app_quit = Signal()

    def __init__(self, app: Optional[QApplication] = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        # QApplication 单例（若调用方未初始化就我们自己初始化）
        self._qt_app: QApplication = app or QApplication.instance() or QApplication(sys.argv)
        self._qt_app.setQuitOnLastWindowClosed(False)  # 隐藏面板时不退出，靠托盘退出

        # 4 种图标（对应悬浮球状态），托盘切换图标用
        self._tray_icons = {
            FloatingBallState.IDLE:      QIcon(_make_color_pixmap("#5FA87C")),   # 叶青绿
            FloatingBallState.LISTENING: QIcon(_make_color_pixmap("#EF4444")),   # 红
            FloatingBallState.THINKING:  QIcon(_make_color_pixmap("#F59E0B")),   # 琥珀
            FloatingBallState.SPEAKING:  QIcon(_make_color_pixmap("#10B981")),   # 绿
        }

        # UI 组件
        self.ball: FloatingBallWidget = FloatingBallWidget()
        self.panel: DrawerMainPanel = DrawerMainPanel()
        self.settings_win: SettingsWindow = SettingsWindow()
        self.settings_win.setWindowIcon(self._tray_icons[FloatingBallState.IDLE])
        self.tray: QSystemTrayIcon = QSystemTrayIcon(self._tray_icons[FloatingBallState.IDLE])
        self.tray.setToolTip("桌面语音助手")

        # 替换 P-02 抽屉的历史 & 调试占位 Tab
        self.history_page = HistoryTabPage(self.panel)
        self.debug_page = DebugPanel(self.panel)
        self._replace_placeholders()

        # 初始化
        self._init_tray_menu()
        self._wire_signals()

        # 先显示悬浮球（抽屉默认隐藏，用户点击球再滑出）
        self.ball.show()
        self.tray.show()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _replace_placeholders(self) -> None:
        """替换 P-02 抽屉里的历史/调试占位页。"""
        # 索引 0:对话 / 1:历史 / 2:调试
        # 先找出历史页(1)和调试页(2)的 widget，调用 removeTab + insertTab
        # History
        if hasattr(self.panel, "history_placeholder"):
            idx = self.panel.tabs.indexOf(self.panel.history_placeholder)
            if idx >= 0:
                self.panel.tabs.removeTab(idx)
                self.panel.tabs.insertTab(idx, self.history_page, "📜 历史")
        # Debug
        if hasattr(self.panel, "debug_placeholder"):
            idx = self.panel.tabs.indexOf(self.panel.debug_placeholder)
            if idx >= 0:
                self.panel.tabs.removeTab(idx)
                self.panel.tabs.insertTab(idx, self.debug_page, "🛠 调试")

    def _init_tray_menu(self) -> None:
        """托盘菜单：显示/隐藏 + 悬浮球开关 + 设置 + TTS + 退出。"""
        menu = QMenu()

        act_show = QAction("显示 / 隐藏主面板", menu)
        act_show.setShortcut("Ctrl+Alt+D")
        act_show.triggered.connect(self.panel.toggle_panel)
        menu.addAction(act_show)

        act_ball = QAction("显示 / 隐藏悬浮球", menu)
        act_ball.triggered.connect(self._toggle_ball)
        menu.addAction(act_ball)

        menu.addSeparator()

        act_tts = QAction("切换 TTS 朗读（同步主面板）", menu)
        act_tts.triggered.connect(self._toggle_tts_from_tray)
        menu.addAction(act_tts)

        menu.addSeparator()

        act_settings = QAction("设置…", menu)
        act_settings.triggered.connect(self.show_settings)
        menu.addAction(act_settings)

        menu.addSeparator()

        act_quit = QAction("退出程序", menu)
        act_quit.triggered.connect(self.quit_app_with_confirm)
        menu.addAction(act_quit)

        self.tray.setContextMenu(menu)

    def _wire_signals(self) -> None:
        # 悬浮球信号
        self.ball.clicked.connect(self.panel.toggle_panel)
        self.ball.request_settings.connect(self.show_settings)
        self.ball.request_quit.connect(self.quit_app_with_confirm)
        self.ball.request_voice_start.connect(self._on_voice_start)
        self.ball.request_voice_end.connect(self._on_voice_end)

        # 主面板信号
        self.panel.user_send_text.connect(self._on_user_send_text)
        self.panel.user_voice_start.connect(self._on_voice_start)
        self.panel.user_voice_end.connect(self._on_voice_end)
        self.panel.user_tts_toggled.connect(self._on_tts_toggled)
        self.panel.request_settings.connect(self.show_settings)
        self.panel.request_hide.connect(lambda: self._push_debug("SYSTEM", "主面板已隐藏，点击悬浮球或托盘「显示」可重新展开。"))
        self.panel.request_show.connect(lambda: self._push_debug("SYSTEM", "主面板已展开。"))

        # 设置窗口信号
        self.settings_win.settings_changed.connect(self.sig_request_settings_saved.emit)

        # 托盘
        self.tray.activated.connect(self._on_tray_activated)

        # 历史页：继续会话
        self.history_page.request_continue.connect(self._on_history_continue)
        self.history_page.request_delete.connect(lambda sid: self._push_debug("SYSTEM", f"已删除会话 {sid}（mock 仅 UI 层移除）。"))

    # ------------------------------------------------------------------
    # 对外 API：主入口 + 退出
    # ------------------------------------------------------------------

    def qt_app(self) -> QApplication:
        return self._qt_app

    def run(self) -> int:
        """阻塞到 QApplication.exec()。返回 exit code。"""
        return self._qt_app.exec()

    def quit_app_with_confirm(self) -> None:
        ret = QMessageBox.question(
            None,
            "退出桌面语音助手",
            "确定要退出桌面语音助手吗？\n\n（正在进行的任务会被中止，Checkpoint 自动保留，下次打开可继续）",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if ret == QMessageBox.Yes:
            self.sig_app_quit.emit()
            self._push_debug("SYSTEM", "用户已确认退出程序… bye 👋")
            # 隐藏 UI（给信号发出时间）
            try:
                self.tray.hide()
            except Exception:
                pass
            QApplication.instance().quit()

    def show_settings(self) -> None:
        self.settings_win.show()
        self.settings_win.raise_()
        self.settings_win.activateWindow()
        self._push_debug("SYSTEM", "打开设置中心。")

    # ------------------------------------------------------------------
    # UI 状态快捷 API（M5-6 桥接层会调用）
    # ------------------------------------------------------------------

    def set_ball_state(self, state: FloatingBallState) -> None:
        self.ball.setState(state)
        if state in self._tray_icons:
            self.tray.setIcon(self._tray_icons[state])

    def append_user_bubble(self, text: str, ts: Optional[str] = None) -> None:
        self.panel.append_user(text, ts)

    def append_ai_bubble(self, text: str, ts: Optional[str] = None) -> None:
        self.panel.append_ai(text, ts)

    def append_system_bubble(self, text: str, ts: Optional[str] = None) -> None:
        self.panel.append_system(text, ts)

    def push_debug(self, stage: str, message: str) -> None:
        self.debug_page.append_stage(stage, message)

    # ------------------------------------------------------------------
    # 内部 UI action handlers（对外转发信号 + 给调试面板打日志）
    # ------------------------------------------------------------------

    def _push_debug(self, stage: str, msg: str) -> None:
        self.debug_page.append_stage(stage, msg)

    def _on_user_send_text(self, text: str) -> None:
        self.append_user_bubble(text)
        self._push_debug("ASR", f"(键盘输入) 用户指令：{text}")
        self.sig_user_submit_text.emit(text)

    def _on_voice_start(self) -> None:
        self.set_ball_state(FloatingBallState.LISTENING)
        self._push_debug("ASR", "开始录音…（按住说话，松开结束）")
        self.sig_user_voice_start.emit()

    def _on_voice_end(self) -> None:
        self.set_ball_state(FloatingBallState.IDLE)
        self._push_debug("ASR", "录音结束，正在离线识别（SenseVoice）。")
        self.sig_user_voice_end.emit()

    def _on_tts_toggled(self, on: bool) -> None:
        self._push_debug("SYSTEM", f"用户切换 TTS：{'开启 🔊' if on else '关闭 🔈'}")
        self.sig_user_tts_toggled.emit(on)

    def _toggle_ball(self) -> None:
        if self.ball.isVisible():
            self.ball.hide()
            self._push_debug("SYSTEM", "悬浮球已隐藏，可通过托盘菜单重新显示。")
        else:
            self.ball.show()
            self._push_debug("SYSTEM", "悬浮球已显示。")

    def _toggle_tts_from_tray(self) -> None:
        self.panel.set_tts_on(not self.panel.is_tts_on())

    def _on_tray_activated(self, reason) -> None:
        # 单击托盘 → 切换面板显示
        if reason == QSystemTrayIcon.Trigger:
            self.panel.toggle_panel()

    def _on_history_continue(self, sid: str) -> None:
        self._push_debug("SYSTEM", f"用户选择继续会话 {sid}（后续 M6 接 Checkpoint 恢复）。")
        self.append_system_bubble(f"📜 恢复会话 {sid}：该功能 M6 阶段接入 CheckpointService 后生效。")


__all__ = ["AppController"]
