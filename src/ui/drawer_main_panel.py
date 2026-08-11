"""P-02 抽屉式主对话面板：右侧滑入滑出 + 对话/历史/调试 3 Tab + 底部输入栏。

形态：
    - 无边框 + StaysOnTop + Tool（不占任务栏）
    - 固定尺寸 W=420 H=720，默认贴着屏幕右侧外 10px
    - 滑入滑出动画 260ms OutCubic
    - 顶栏：Logo 标题 + 「隐藏」按钮（最小化） + 「设置」+ 「关闭面板」（隐藏抽屉）
    - 中部 Tab：
        Tab 1「对话」：BubbleListWidget 气泡
        Tab 2「历史」：占位（后面 M5-5 填充）
        Tab 3「调试」：占位（后面 M5-4 填充）
    - 底部输入栏：文本输入框 + 🎤 语音按钮 + 🔊 TTS 开关 + 🚀 发送

对外信号：
    user_send_text(text: str)
    user_toggle_voice_start()
    user_toggle_voice_end()
    user_toggle_tts(on: bool)
    user_request_settings()
    panel_will_hide()
    panel_will_show()
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtGui import QColor

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# 公共控件：BubbleListWidget / SwitchToggle
from src.ui.widgets.common import BubbleListWidget, SwitchToggle


PANEL_W = 420
PANEL_H = 720
PANEL_OFFSET = 10  # 离屏幕右边间距
PANEL_TOP = 80
ANIM_MS = 260


class DrawerMainPanel(QWidget):
    """P-02 抽屉式主面板：桌面语音助手核心容器。"""

    # 信号
    user_send_text = Signal(str)
    user_voice_start = Signal()
    user_voice_end = Signal()
    user_tts_toggled = Signal(bool)
    request_settings = Signal()
    request_hide = Signal()
    request_show = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # 无边框 + 置顶 + 工具窗口
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setFixedSize(QSize(PANEL_W, PANEL_H))
        self._anim: Optional[QPropertyAnimation] = None
        self._is_visible_state = False

        # 外层容器 + 阴影（因为 WA_TranslucentBackground 为 False，要圆角好看我们用一个内框包起来）
        self._outer = QWidget(self)
        self._outer.setObjectName("DrawerOuter")
        self._outer.setGeometry(0, 0, PANEL_W, PANEL_H)
        self._outer.setStyleSheet(
            "QWidget#DrawerOuter{background:white;border:1px solid #DCE7D6;border-radius:14px;}"
        )
        shadow = QGraphicsDropShadowEffect(self._outer)
        shadow.setBlurRadius(24)
        shadow.setXOffset(0)
        shadow.setYOffset(6)
        shadow.setColor(self._qcolor(15, 23, 42, 32))
        self._outer.setGraphicsEffect(shadow)

        # 整体布局
        layout = QVBoxLayout(self._outer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- 顶部栏 ---
        layout.addWidget(self._build_topbar())

        # --- 中部 Tab ---
        self.tabs = QTabWidget(self._outer)
        self.tabs.setStyleSheet(
            "QTabWidget::pane{border:none;border-top:1px solid #E2ECDE;}"
            "QTabBar::tab{"
            "background:transparent;color:#52614F;font-size:13px;padding:8px 16px;"
            "border:none;border-bottom:2px solid transparent;}"
            "QTabBar::tab:selected{color:#4D9269;border-bottom-color:#5FA87C;font-weight:600;}"
            "QTabBar::tab:hover{color:#24352A;}"
        )
        # 三个 Tab 页
        self._chat_tab = self._build_chat_tab()
        self._history_tab = self._build_history_tab_placeholder()
        self._debug_tab = self._build_debug_tab_placeholder()
        self.tabs.addTab(self._chat_tab, "💬 对话")
        self.tabs.addTab(self._history_tab, "📜 历史")
        self.tabs.addTab(self._debug_tab, "🛠 调试")
        layout.addWidget(self.tabs, 1)

        # --- 底部输入栏 ---
        layout.addWidget(self._build_input_bar())

        # 初始位置：屏幕右侧外
        self._place_offscreen()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _qcolor(self, r: int, g: int, b: int, a: int = 255) -> "QColor":
        from PySide6.QtGui import QColor as _C
        return _C(r, g, b, a)

    def _build_topbar(self) -> QWidget:
        bar = QFrame(self._outer)
        bar.setStyleSheet(
            "QFrame{background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "stop:0 #EFF7EB,stop:1 #FFFFFF);"
            "border-bottom:1px solid #E2ECDE;"
            "border-top-left-radius:14px;border-top-right-radius:14px;}"
        )
        bar.setFixedHeight(52)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 10, 0)
        lay.setSpacing(8)

        title = QLabel("🎙️ 桌面语音助手", bar)
        title.setStyleSheet(
            "color:#24352A;font-size:15px;font-weight:700;letter-spacing:0.2px;"
            "background:transparent;border:none;"
        )
        lay.addWidget(title, 1)

        # 隐藏（最小化）
        btn_hide = self._icon_btn("▾", "隐藏面板")
        btn_hide.clicked.connect(self.hide_panel)
        lay.addWidget(btn_hide)

        # 设置
        btn_set = self._icon_btn("⚙️", "设置中心")
        btn_set.clicked.connect(self.request_settings.emit)
        lay.addWidget(btn_set)

        # 关闭面板（= 隐藏抽屉）
        btn_close = self._icon_btn("✕", "关闭面板")
        btn_close.setStyleSheet(
            "QPushButton{border:none;background:transparent;border-radius:10px;"
            "font-size:14px;color:#6E7F6A;padding:6px 8px;}"
            "QPushButton:hover{background:#FEE2E2;color:#DC2626;}"
        )
        btn_close.clicked.connect(self.hide_panel)
        lay.addWidget(btn_close)
        return bar

    def _build_chat_tab(self) -> QWidget:
        wrap = QWidget(self.tabs)
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(4, 0, 4, 4)
        lay.setSpacing(0)
        # 顶部分隔，避免欢迎气泡与 Tab 栏/顶栏重叠被截断
        from PySide6.QtWidgets import QSpacerItem
        lay.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))
        self.chat_bubbles = BubbleListWidget(wrap)
        lay.addWidget(self.chat_bubbles, 1)
        # 欢迎气泡
        self.chat_bubbles.append_system("👋 欢迎使用桌面语音助手，试试说「创建个文件」或直接打字。")
        return wrap

    def _build_history_tab_placeholder(self) -> QWidget:
        wrap = QWidget(self.tabs)
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(16, 24, 16, 24)
        tip = QLabel("📜 历史会话\n\nM5-5 将接入此 Tab：按日期分组、可回看、可清空、可继续上轮对话。", wrap)
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet(
            "color:#6E7F6A;font-size:13px;padding:24px;background:#F5F9F3;border-radius:12px;"
            "border:1px dashed #B9D4BE;line-height:1.8;"
        )
        tip.setWordWrap(True)
        lay.addStretch(1)
        lay.addWidget(tip, 0, Qt.AlignHCenter)
        lay.addStretch(2)
        self.history_placeholder = wrap
        return wrap

    def _build_debug_tab_placeholder(self) -> QWidget:
        wrap = QWidget(self.tabs)
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(16, 24, 16, 24)
        tip = QLabel(
            "🛠 消息同步调试面板\n\n"
            "M5-4 将接入此 Tab：ASR → Agent → Tool → Observation → Answer 全链路彩色日志，\n"
            "支持分阶段折叠（AccordionSection）、单行高亮、最大行数 5000 自动清理。",
            wrap,
        )
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet(
            "color:#6E7F6A;font-size:13px;padding:24px;background:#F5F9F3;border-radius:12px;"
            "border:1px dashed #A8C4A2;line-height:1.8;"
        )
        tip.setWordWrap(True)
        lay.addStretch(1)
        lay.addWidget(tip, 0, Qt.AlignHCenter)
        lay.addStretch(2)
        self.debug_placeholder = wrap
        return wrap

    def _build_input_bar(self) -> QWidget:
        bar = QFrame(self._outer)
        bar.setStyleSheet(
            "QFrame{background:#F7FAF4;border-top:1px solid #E2ECDE;border-bottom-left-radius:14px;"
            "border-bottom-right-radius:14px;}"
        )
        bar.setFixedHeight(72)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 10, 10, 12)
        lay.setSpacing(8)

        # 🎤 语音按钮（按住说话）
        self.btn_voice = QPushButton("🎤", bar)
        self.btn_voice.setFixedSize(42, 42)
        self.btn_voice.setCursor(Qt.PointingHandCursor)
        self.btn_voice.setStyleSheet(
            "QPushButton{background:#EAF4EC;color:#3F7D58;border:1px solid #C6E2CC;"
            "border-radius:21px;font-size:16px;}"
            "QPushButton:hover{background:#DCEEDF;}"
            "QPushButton:pressed{background:#4D9269;color:white;border-color:#4D9269;}"
        )
        self.btn_voice.setToolTip("按住说话，松开发送；短按切换录音状态。")
        self.btn_voice.pressed.connect(self.user_voice_start.emit)
        self.btn_voice.released.connect(self.user_voice_end.emit)
        lay.addWidget(self.btn_voice)

        # 输入框
        self.input_edit = QLineEdit(bar)
        self.input_edit.setPlaceholderText("说点什么…（Enter 发送，Shift+Enter 换行）")
        self.input_edit.setStyleSheet(
            "QLineEdit{background:white;border:1px solid #E2ECDE;border-radius:20px;padding:9px 14px;"
            "font-size:13px;color:#24352A;selection-background-color:#CDE7D3;}"
            "QLineEdit:focus{border-color:#5FA87C;}"
        )
        self.input_edit.setMinimumHeight(40)
        self.input_edit.returnPressed.connect(self._on_send_clicked)
        lay.addWidget(self.input_edit, 1)

        # 🔊 TTS 开关
        tts_wrap = QWidget(bar)
        tts_lay = QVBoxLayout(tts_wrap)
        tts_lay.setContentsMargins(0, 0, 0, 0)
        tts_lay.setSpacing(2)
        self.tts_switch = SwitchToggle(checked=True, parent=tts_wrap)
        tts_center = QHBoxLayout()
        tts_center.addStretch(1)
        tts_center.addWidget(self.tts_switch)
        tts_center.addStretch(1)
        tts_lay.addLayout(tts_center)
        tts_label = QLabel("🔊 TTS", tts_wrap)
        tts_label.setAlignment(Qt.AlignHCenter)
        tts_label.setStyleSheet("color:#6E7F6A;font-size:10px;")
        tts_lay.addWidget(tts_label)
        self.tts_switch.toggled.connect(self.user_tts_toggled.emit)
        lay.addWidget(tts_wrap)

        # 🚀 发送按钮
        self.btn_send = QPushButton("🚀", bar)
        self.btn_send.setFixedSize(42, 42)
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setStyleSheet(
            "QPushButton{background:#5FA87C;color:white;border-radius:21px;font-size:16px;border:none;}"
            "QPushButton:hover{background:#4D9269;}"
            "QPushButton:pressed{background:#3F7D58;}"
            "QPushButton:disabled{background:#D7E2D3;color:#F4F8F2;}"
        )
        self.btn_send.clicked.connect(self._on_send_clicked)
        lay.addWidget(self.btn_send)
        return bar

    def _icon_btn(self, text: str, tip: str) -> QPushButton:
        btn = QPushButton(text, self._outer)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip(tip)
        btn.setFixedSize(32, 32)
        btn.setStyleSheet(
            "QPushButton{border:none;background:transparent;border-radius:10px;"
            "font-size:14px;color:#52614F;padding:4px;}"
            "QPushButton:hover{background:#EDF4E9;color:#24352A;}"
            "QPushButton:pressed{background:#DEEAD9;}"
        )
        return btn

    # ------------------------------------------------------------------
    # 对外 API：聊天便捷方法（参数名与 BubbleListWidget 对齐：timestamp）
    # ------------------------------------------------------------------

    def append_user(self, text: str, timestamp: Optional[str] = None, is_preview: bool = False) -> None:
        self.chat_bubbles.append_user(text, timestamp=timestamp, is_preview=is_preview)

    def append_ai(self, text: str, timestamp: Optional[str] = None) -> None:
        self.chat_bubbles.append_ai(text, timestamp=timestamp)

    def append_system(self, text: str, timestamp: Optional[str] = None) -> None:
        self.chat_bubbles.append_system(text, timestamp=timestamp)

    def update_user_preview(self, text: str) -> None:
        """M8: 更新用户语音实时预览气泡。"""
        self.chat_bubbles.update_last_user_preview(text)

    def finalize_user_message(self, text: str) -> None:
        """M8: 把预览气泡转为正式用户消息。"""
        self.chat_bubbles.finalize_last_user(text)

    def clear_user_preview(self) -> None:
        """M8: 清除预览气泡（识别失败/超时）。"""
        self.chat_bubbles.clear_last_preview()

    def clear_chat(self) -> None:
        self.chat_bubbles.clear()

    def input_text(self) -> str:
        return self.input_edit.text().strip()

    def set_input_text(self, text: str) -> None:
        self.input_edit.setText(text or "")

    def is_tts_on(self) -> bool:
        return self.tts_switch.isChecked()

    def set_tts_on(self, on: bool) -> None:
        self.tts_switch.setChecked(bool(on))

    def set_input_busy(self, busy: bool) -> None:
        """ASR 录音 / Agent 思考期间，禁用输入栏防连点 3 次重复 warning。

        Args:
            busy: True = 禁用（按钮变灰，连点无效）；False = 恢复可用
        """
        busy = bool(busy)
        self.btn_voice.setDisabled(busy)
        self.btn_send.setDisabled(busy)
        self.input_edit.setDisabled(busy)
        # tts_switch 不禁用：任何状态下都允许用户手动开/关 TTS
        # 视觉提示：录音中 btn_voice 文字从 🎤 变 🔴
        try:
            self.btn_voice.setText("🔴" if busy else "🎤")
            if busy:
                self.btn_voice.setToolTip("（正在录音 / 识别中，请稍候…）")
            else:
                self.btn_voice.setToolTip("按住说话，松开发送；短按切换录音状态。")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 显示/隐藏 滑入滑出动画
    # ------------------------------------------------------------------

    def is_panel_visible(self) -> bool:
        return self._is_visible_state

    def toggle_panel(self) -> None:
        if self._is_visible_state:
            self.hide_panel()
        else:
            self.show_panel()

    def show_panel(self) -> None:
        if self._is_visible_state:
            return
        self._is_visible_state = True
        self.request_show.emit()
        self._place_offscreen()
        self.show()
        self.raise_()
        self.activateWindow()
        rect_start = self.geometry()
        rect_end = self._onscreen_rect()
        self._animate(rect_start, rect_end)

    def hide_panel(self) -> None:
        if not self._is_visible_state:
            return
        self._is_visible_state = False
        self.request_hide.emit()
        rect_start = self.geometry()
        rect_end = self._offscreen_rect()
        self._animate(rect_start, rect_end, on_finished=self.hide)

    # ------------------------------------------------------------------
    # 动画 + 屏幕坐标
    # ------------------------------------------------------------------

    def _screen_geom(self) -> QRect:
        sc = self.screen()
        if sc is None:
            return QRect(0, 0, 1920, 1080)
        return sc.availableGeometry()

    def _onscreen_rect(self) -> QRect:
        g = self._screen_geom()
        x = g.right() - PANEL_W - PANEL_OFFSET
        y = max(g.top() + PANEL_TOP, g.top() + PANEL_OFFSET)
        if y + PANEL_H > g.bottom():
            y = g.bottom() - PANEL_H - PANEL_OFFSET
        return QRect(x, y, PANEL_W, PANEL_H)

    def _offscreen_rect(self) -> QRect:
        g = self._screen_geom()
        y = self._onscreen_rect().y()
        x = g.right() + PANEL_OFFSET
        return QRect(x, y, PANEL_W, PANEL_H)

    def _place_offscreen(self) -> None:
        self.setGeometry(self._offscreen_rect())

    def _animate(self, start: QRect, end: QRect, on_finished=None) -> None:
        self._anim = QPropertyAnimation(self, b"geometry", self)
        self._anim.setDuration(ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        if on_finished is not None:
            self._anim.finished.connect(on_finished)
        self._anim.start(QPropertyAnimation.DeleteWhenStopped)

    # ------------------------------------------------------------------
    # 内部：发送
    # ------------------------------------------------------------------

    def _on_send_clicked(self) -> None:
        text = self.input_text()
        if not text:
            return
        self.input_edit.clear()
        self.user_send_text.emit(text)


__all__ = ["DrawerMainPanel", "PANEL_W", "PANEL_H"]
