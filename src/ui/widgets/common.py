"""M5.1 公共控件库：Accordion / ChipButton / BubbleList / SwitchToggle / LogHighlighter。

设计系统（来自 PRD 5.1 Light Mode）：
    Primary      #3B82F6  Blue-500    主按钮/悬浮球待机
    Primary Hov  #2563EB  Blue-600    悬停
    Accent       #8B5CF6  Violet-500  Agent 气泡/Agent 日志
    Warning      #F59E0B  Amber-500   思考/工具调用
    Danger       #EF4444  Red-500     录音中/错误日志/DELETE
    Success      #10B981  Emerald-500 TTS 播报中
    Text Main    #0F172A  Slate-900   正文
    Text Sub     #64748B  Slate-500   辅助文字
    Bg           #FFFFFF  白
    Surface      #F8FAFC  Slate-50    气泡背景
    Border       #E2E8F0  Slate-200   分割线

所有控件都是纯 QWidget 子类，无任何 Web 组件，可直接被悬浮球/抽屉/设置/调试面板复用。
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import (
    QEasingCurve,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QSyntaxHighlighter, QTextCharFormat, QTextDocument
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# 1. AccordionSection（折叠区块）—— 用于 P-03 调试面板 / P-04 设置
# ============================================================

class AccordionSection(QWidget):
    """可折叠区块：标题栏（左文字+右箭头）→ 点击展开/收起，内容区高度动画平滑过渡。

    用法：
        acc = AccordionSection("🔍 ASR 阶段（语音转文字）", expanded=True)
        acc.setContent(widget_inside)
    """

    toggled = Signal(bool)  # 展开/收起时发出，expanded

    def __init__(self, title: str, expanded: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._expanded = bool(expanded)
        self._content: Optional[QWidget] = None
        self._content_height = 240  # 展开时目标高度（动画结束后自动根据内容调整）

        # 外观
        self.setObjectName("AccordionSection")
        self.setStyleSheet(
            "QWidget#AccordionSection{background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;}"
            "QLabel#AccordionTitle{color:#0F172A;font-size:14px;font-weight:600;padding:4px 0;}"
            "QLabel#AccordionArrow{color:#64748B;font-size:13px;padding-right:6px;}"
        )
        self.setAttribute(Qt.WA_StyledBackground, True)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(15, 23, 42, 25))
        self.setGraphicsEffect(shadow)

        # 标题行
        self._title_bar = QFrame(self)
        self._title_bar.setCursor(Qt.PointingHandCursor)
        title_layout = QHBoxLayout(self._title_bar)
        title_layout.setContentsMargins(14, 10, 12, 10)
        title_layout.setSpacing(8)

        self._title_lbl = QLabel(title, self._title_bar)
        self._title_lbl.setObjectName("AccordionTitle")
        title_layout.addWidget(self._title_lbl, 1)

        self._arrow_lbl = QLabel("▾" if self._expanded else "▸", self._title_bar)
        self._arrow_lbl.setObjectName("AccordionArrow")
        self._arrow_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        title_layout.addWidget(self._arrow_lbl, 0, Qt.AlignRight)

        # 内容容器（动画驱动最大高度）
        self._content_area = QScrollArea(self)
        self._content_area.setWidgetResizable(True)
        self._content_area.setFrameShape(QFrame.NoFrame)
        self._content_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._content_area.setStyleSheet("QScrollArea{background:transparent;}")

        # 根布局
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._title_bar)
        root.addWidget(self._content_area)

        # 初始收起
        self._content_area.setMaximumHeight(0 if not self._expanded else self._content_height)
        self._anim_group: Optional[QParallelAnimationGroup] = None

    # ---------- 对外 API ----------

    def setTitle(self, title: str) -> None:
        self._title_lbl.setText(title)

    def setContent(self, widget: QWidget) -> None:
        self._content = widget
        self._content_area.setWidget(widget)
        # 估算内容高度，避免动画跳变
        widget.adjustSize()
        self._content_height = max(120, min(600, widget.sizeHint().height() + 12))
        if self._expanded:
            self._content_area.setMaximumHeight(self._content_height)

    def contentArea(self) -> QScrollArea:
        return self._content_area

    def isExpanded(self) -> bool:
        return self._expanded

    def setExpanded(self, expanded: bool, animate: bool = True) -> None:
        expanded = bool(expanded)
        if expanded == self._expanded:
            return
        self._expanded = expanded
        self._arrow_lbl.setText("▾" if expanded else "▸")
        self.toggled.emit(expanded)

        if not animate:
            self._content_area.setMaximumHeight(self._content_height if expanded else 0)
            return

        target = self._content_height if expanded else 0
        # 同步把内容 sizeHint 再算一遍，防止折叠时过高
        if expanded and self._content is not None:
            self._content.adjustSize()
            self._content_height = max(120, min(600, self._content.sizeHint().height() + 12))
            target = self._content_height

        anim = QPropertyAnimation(self._content_area, b"maximumHeight", self)
        anim.setDuration(260)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.setStartValue(self._content_area.maximumHeight())
        anim.setEndValue(target)
        self._anim_group = QParallelAnimationGroup(self)
        self._anim_group.addAnimation(anim)
        self._anim_group.start(QParallelAnimationGroup.DeleteWhenStopped)

    def toggle(self, animate: bool = True) -> None:
        self.setExpanded(not self._expanded, animate=animate)

    # ---------- 事件 ----------

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if self._title_bar.geometry().contains(ev.position().toPoint()):
            self.toggle(True)
            return
        super().mousePressEvent(ev)


# ============================================================
# 2. ChipButton（圆角 Chip，用于设置/快捷入口）
# ============================================================

class ChipButton(QPushButton):
    """圆角 Chip 按钮：默认空心灰边框，悬停填充主色。

    用法：
        btn = ChipButton("🔊 开启 TTS", active=True)
        btn.clicked.connect(...)
    """

    def __init__(self, text: str, active: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._active = bool(active)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self._apply_style()

    def setActive(self, active: bool) -> None:
        self._active = bool(active)
        self._apply_style()

    def isActive(self) -> bool:
        return self._active

    def toggle_active(self) -> None:
        self.setActive(not self._active)

    def _apply_style(self) -> None:
        if self._active:
            self.setStyleSheet(
                "QPushButton{background:#3B82F6;color:white;border:1px solid #3B82F6;"
                "border-radius:14px;padding:4px 12px;font-size:12px;font-weight:600;}"
                "QPushButton:hover{background:#2563EB;border-color:#2563EB;}"
                "QPushButton:pressed{background:#1D4ED8;}"
            )
        else:
            self.setStyleSheet(
                "QPushButton{background:white;color:#334155;border:1px solid #E2E8F0;"
                "border-radius:14px;padding:4px 12px;font-size:12px;}"
                "QPushButton:hover{background:#F1F5F9;border-color:#CBD5E1;color:#0F172A;}"
                "QPushButton:pressed{background:#E2E8F0;}"
            )


# ============================================================
# 3. SwitchToggle（iOS 风格开关）
# ============================================================

class SwitchToggle(QWidget):
    """自绘 iOS 风开关：W=42 H=24，打开蓝色，关闭灰。

    用法：
        sw = SwitchToggle(checked=True)
        sw.toggled.connect(lambda b: print("TTS:", b))
    """

    toggled = Signal(bool)

    def __init__(self, checked: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._checked = bool(checked)
        self.setFixedSize(42, 24)
        self.setCursor(Qt.PointingHandCursor)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool, emit_signal: bool = True) -> None:
        checked = bool(checked)
        if checked == self._checked:
            return
        self._checked = checked
        self.update()
        if emit_signal:
            self.toggled.emit(checked)

    def toggle(self, emit_signal: bool = True) -> None:
        self.setChecked(not self._checked, emit_signal=emit_signal)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(42, 24)

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = 42, 24
        radius = h / 2
        track_color = QColor("#3B82F6") if self._checked else QColor("#CBD5E1")
        p.setPen(Qt.NoPen)
        p.setBrush(track_color)
        p.drawRoundedRect(0, 0, w, h, radius, radius)

        # 滑块
        knob_r = 10
        knob_y = 2
        knob_x = 2 if not self._checked else (w - knob_r * 2 - 2)
        p.setBrush(Qt.white)
        path = QPainterPath()
        path.addRoundedRect(knob_x, knob_y, knob_r * 2, h - 4, knob_r, knob_r)
        p.drawPath(path)
        p.end()

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            self.toggle(True)
            return
        super().mousePressEvent(ev)


# ============================================================
# 4. BubbleListWidget（左/右气泡聊天列表）
# ============================================================

class BubbleListWidget(QListWidget):
    """聊天气泡列表：左侧=AI（紫底）/右侧=用户（蓝底）/中间=系统（灰 Chip）。

    用法：
        bl = BubbleListWidget()
        bl.append_user("给我创建个文件")
        bl.append_ai("好，已经帮你在桌面创建了 hello.txt")
        bl.append_system("⚠️ 高危操作：需要二次确认 DELETE")
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            "QListWidget{background:#FFFFFF;border:none;padding:10px 4px;}"
            "QListWidget::item{margin:4px 6px;}"
        )
        self.setSelectionMode(QListWidget.NoSelection)
        self.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.verticalScrollBar().setStyleSheet(
            "QScrollBar:vertical{width:6px;background:transparent;}"
            "QScrollBar::handle:vertical{background:#CBD5E1;border-radius:3px;}"
            "QScrollBar::handle:vertical:hover{background:#94A3B8;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )

    # ---------- 对外 API ----------

    def append_user(self, text: str, timestamp: Optional[str] = None) -> None:
        self._append(text, role="user", ts=timestamp)

    def append_ai(self, text: str, timestamp: Optional[str] = None) -> None:
        self._append(text, role="ai", ts=timestamp)

    def append_system(self, text: str, timestamp: Optional[str] = None) -> None:
        self._append(text, role="system", ts=timestamp)

    # ---------- 内部 ----------

    def _append(self, text: str, role: str, ts: Optional[str]) -> None:
        content = (text or "").strip()
        if not content:
            return

        item = QListWidgetItem(self)
        item.setFlags(Qt.NoItemFlags)

        bubble = self._make_bubble(content, role, ts)
        bubble.adjustSize()
        # 根据 bubble 内容估算高度
        hint_h = max(44, bubble.sizeHint().height())
        item.setSizeHint(QSize(self.viewport().width() - 20, hint_h + 8))
        self.addItem(item)
        self.setItemWidget(item, bubble)
        # 滚动到底
        self.scrollToBottom()

    def _make_bubble(self, content: str, role: str, ts: Optional[str]) -> QWidget:
        wrap = QWidget(self)
        wrap.setStyleSheet("background:transparent;")
        root_layout = QVBoxLayout(wrap)
        root_layout.setContentsMargins(6, 2, 6, 2)
        root_layout.setSpacing(2)

        # 时间戳
        if ts:
            ts_lbl = QLabel(ts, wrap)
            ts_lbl.setStyleSheet("color:#94A3B8;font-size:11px;")
            ts_lbl.setAlignment(Qt.AlignHCenter)
            root_layout.addWidget(ts_lbl, 0, Qt.AlignHCenter)

        if role == "system":
            # 系统提示：居中灰 Chip
            chip = QLabel(content, wrap)
            chip.setWordWrap(True)
            chip.setStyleSheet(
                "background:#F1F5F9;color:#475569;border-radius:10px;padding:5px 10px;"
                "font-size:12px;"
            )
            chip.setAlignment(Qt.AlignCenter)
            chip.setMaximumWidth(int(self.viewport().width() * 0.85))
            root_layout.addWidget(chip, 0, Qt.AlignHCenter)
            return wrap

        is_user = role == "user"
        bubble_wrapper = QHBoxLayout()
        bubble_wrapper.setContentsMargins(0, 0, 0, 0)
        if is_user:
            bubble_wrapper.addStretch(1)

        bubble_lbl = QLabel(content, wrap)
        bubble_lbl.setWordWrap(True)
        bubble_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        if is_user:
            bubble_lbl.setStyleSheet(
                "background:#3B82F6;color:white;border-radius:12px;"
                "border-top-right-radius:3px;padding:8px 12px;"
                "font-size:13px;line-height:1.6;"
            )
        else:  # ai
            bubble_lbl.setStyleSheet(
                "background:#F5F3FF;color:#1E1B4B;border:1px solid #DDD6FE;"
                "border-radius:12px;border-top-left-radius:3px;padding:8px 12px;"
                "font-size:13px;line-height:1.6;"
            )
        max_w = int(self.viewport().width() * 0.85)
        bubble_lbl.setMaximumWidth(max_w)
        bubble_lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        bubble_wrapper.addWidget(bubble_lbl, 0, Qt.AlignTop)
        if not is_user:
            bubble_wrapper.addStretch(1)
        root_layout.addLayout(bubble_wrapper)
        return wrap


# ============================================================
# 5. MessageLogHighlighter（调试面板终端风彩色日志上色）
# ============================================================

class MessageLogHighlighter(QSyntaxHighlighter):
    """给 QPlainTextEdit 的调试日志按阶段上色：

    匹配前缀（大小写不敏感）：
        [ASR]          蓝 Blue-600  语音识别
        [AGENT] | [LLM] 紫 Violet-600  大模型思考
        [TOOL] | [ACT] 橙 Orange-600  工具调用开始
        [OBS] | [RES]  绿 Emerald-600 工具结果 Observation
        [TTS]          翠 Teal-600 语音播报
        [ERR] | [FAIL] 红 Red-600 失败
        [WARN] | ⚠️     琥珀 Amber-600 警告
        其他           黑 Slate-900
    """

    _RULES: list[tuple[str, str]] = [
        (r"^\s*\[(?:ASR)\].*", "#2563EB"),     # Blue-600
        (r"^\s*\[(?:AGENT|LLM|THOUGHT)\].*", "#7C3AED"),  # Violet-700
        (r"^\s*\[(?:TOOL|ACT|ACTION)\].*", "#EA580C"),   # Orange-600
        (r"^\s*\[(?:OBS|RES|RESULT|OBSERVATION)\].*", "#059669"),  # Emerald-600
        (r"^\s*\[(?:TTS|SPEAK)\].*", "#0D9488"),          # Teal-600
        (r"^\s*\[(?:ERR|ERROR|FAIL)\].*", "#DC2626"),     # Red-600
        (r"^\s*\[(?:WARN|WARNING)\].*",  "#D97706"),      # Amber-600
    ]

    def __init__(self, document: Optional[QTextDocument] = None) -> None:
        super().__init__(document)
        self._formats: list[tuple[Any, QTextCharFormat]] = []
        import re
        for pattern, color in self._RULES:
            regex = re.compile(pattern, re.IGNORECASE)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            fmt.setFontWeight(QFont.DemiBold)
            self._formats.append((regex, fmt))
        # 默认正文
        self._default_fmt = QTextCharFormat()
        self._default_fmt.setForeground(QColor("#0F172A"))

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        # 先整个行设默认
        self.setFormat(0, len(text), self._default_fmt)
        for regex, fmt in self._formats:
            for m in regex.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ============================================================
# 6. DebugLogView（调试面板的彩色日志视图，P-03 用）
# ============================================================

class DebugLogView(QPlainTextEdit):
    """调试日志视图：彩色 + 自动滚底 + 行上限 5000。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setFont(QFont("Consolas, Menlo, Cascadia Mono", 10))
        self.setStyleSheet(
            "QPlainTextEdit{background:#F8FAFC;color:#0F172A;"
            "border:1px solid #E2E8F0;border-radius:8px;padding:8px;}"
        )
        self._hl = MessageLogHighlighter(self.document())

    # 便捷 API：按阶段追加一行
    def append_stage(self, stage: str, message: str) -> None:
        stage = (stage or "").strip().upper() or "INFO"
        text = f"[{stage}] {message}"
        self.appendPlainText(text)
        # 自动滚到底
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


__all__ = [
    "AccordionSection",
    "ChipButton",
    "SwitchToggle",
    "BubbleListWidget",
    "MessageLogHighlighter",
    "DebugLogView",
]
