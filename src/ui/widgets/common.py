"""M5.1 公共控件库：Accordion / ChipButton / BubbleList / SwitchToggle / LogHighlighter。

设计系统（淡绿自然 · 优雅 Light Mode）：
    Primary      #5FA87C  叶青绿      主按钮/悬浮球待机/开关
    Primary Hov  #4D9269  深叶绿      悬停
    Primary Press#3F7D58  苔绿        按压
    Primary Soft #EAF4EC  嫩芽底      淡绿底衬
    Accent       #F1F7EE  青瓷底      Agent 气泡
    Warning      #F59E0B  Amber-500   思考/工具调用
    Danger       #EF4444  Red-500     录音中/错误日志/DELETE
    Success      #10B981  Emerald-500 TTS 播报中
    Text Main    #24352A  深松绿      正文
    Text Sub     #6E7F6A  灰绿        辅助文字
    Bg           #FFFFFF  白
    Surface      #F5F9F3  浅芽白      气泡背景/卡片
    Border       #E2ECDE  浅绿灰      分割线

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
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QSyntaxHighlighter, QTextCharFormat, QTextDocument
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
            "QWidget#AccordionSection{background:#FFFFFF;border:1px solid #E2ECDE;border-radius:10px;}"
            "QLabel#AccordionTitle{color:#24352A;font-size:14px;font-weight:600;padding:4px 0;}"
            "QLabel#AccordionArrow{color:#6E7F6A;font-size:13px;padding-right:6px;}"
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
                "QPushButton{background:#5FA87C;color:white;border:1px solid #5FA87C;"
                "border-radius:14px;padding:4px 12px;font-size:12px;font-weight:600;}"
                "QPushButton:hover{background:#4D9269;border-color:#4D9269;}"
                "QPushButton:pressed{background:#3F7D58;}"
            )
        else:
            self.setStyleSheet(
                "QPushButton{background:white;color:#4A5D46;border:1px solid #E2ECDE;"
                "border-radius:14px;padding:4px 12px;font-size:12px;}"
                "QPushButton:hover{background:#F0F6EC;border-color:#C6E2CC;color:#24352A;}"
                "QPushButton:pressed{background:#E2ECDE;}"
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
        track_color = QColor("#5FA87C") if self._checked else QColor("#D3DECD")
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
        bl.update_last_user_preview("正在说话...")  # M8: 实时预览更新
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
            "QScrollBar::handle:vertical{background:#C9DCC5;border-radius:3px;}"
            "QScrollBar::handle:vertical:hover{background:#A8C4A2;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )
        self._last_user_item: Optional[QListWidgetItem] = None
        self._last_user_label: Optional[QLabel] = None
        self._is_preview_mode: bool = False

    # ---------- 对外 API ----------

    def append_user(self, text: str, timestamp: Optional[str] = None, is_preview: bool = False) -> None:
        self._append(text, role="user", ts=timestamp, is_preview=is_preview)

    def append_ai(self, text: str, timestamp: Optional[str] = None) -> None:
        self._append(text, role="ai", ts=timestamp)

    def append_system(self, text: str, timestamp: Optional[str] = None) -> None:
        self._append(text, role="system", ts=timestamp)

    def update_last_user_preview(self, text: str) -> None:
        """M8: 更新最后一条用户气泡（实时预览用，带...提示和浅色样式）。"""
        if self._last_user_item is None or self._last_user_label is None:
            self.append_user(text, is_preview=True)
            return
        preview_text = (text or "").strip()
        if not preview_text:
            preview_text = "🎙️ 正在听..."
        else:
            preview_text = preview_text + " ▍"  # 光标效果
        self._last_user_label.setText(preview_text)
        if self._is_preview_mode:
            self._last_user_label.setStyleSheet(
                "background:#88B99A;color:rgba(255,255,255,0.9);border-radius:12px;"
                "border-top-right-radius:3px;padding:8px 12px;"
                "font-size:13px;line-height:1.6;font-style:italic;"
            )
        self._refresh_item_size(self._last_user_item, self._last_user_label)
        self.scrollToBottom()

    def finalize_last_user(self, text: str) -> None:
        """M8: 把预览气泡转为正式用户气泡（提交时调用）。"""
        final_text = (text or "").strip()
        if self._last_user_item is None or self._last_user_label is None:
            self.append_user(final_text)
            return
        self._last_user_label.setText(final_text)
        self._last_user_label.setStyleSheet(
            "background:#5FA87C;color:white;border-radius:12px;"
            "border-top-right-radius:3px;padding:8px 12px;"
            "font-size:13px;line-height:1.6;"
        )
        self._is_preview_mode = False
        self._refresh_item_size(self._last_user_item, self._last_user_label)
        self._last_user_item = None
        self._last_user_label = None
        self.scrollToBottom()

    def clear_last_preview(self) -> None:
        """M8: 清除预览气泡（识别失败/超时丢弃时调用）。"""
        if self._last_user_item is not None:
            row = self.row(self._last_user_item)
            if row >= 0:
                self.takeItem(row)
        self._last_user_item = None
        self._last_user_label = None
        self._is_preview_mode = False

    # ---------- 内部 ----------

    def _refresh_item_size(self, item: QListWidgetItem, label: QLabel) -> None:
        """重新计算 item 高度（文本变化后调用，避免气泡被截断）。"""
        label.adjustSize()
        wrap = label.parentWidget()
        if wrap is not None:
            wrap.adjustSize()
            hint_h = max(44, wrap.sizeHint().height())
            item.setSizeHint(QSize(self.viewport().width() - 20, hint_h + 8))

    def _append(self, text: str, role: str, ts: Optional[str], is_preview: bool = False) -> None:
        content = (text or "").strip()
        if not content:
            return

        item = QListWidgetItem(self)
        item.setFlags(Qt.NoItemFlags)

        bubble, label = self._make_bubble(content, role, ts, is_preview=is_preview)
        bubble.adjustSize()
        # 根据 bubble 内容估算高度
        hint_h = max(44, bubble.sizeHint().height())
        item.setSizeHint(QSize(self.viewport().width() - 20, hint_h + 8))
        self.addItem(item)
        self.setItemWidget(item, bubble)

        # M8: 跟踪最后一条用户消息（用于预览更新）
        if role == "user":
            self._last_user_item = item
            self._last_user_label = label
            self._is_preview_mode = is_preview

        # 滚动到底
        self.scrollToBottom()

    def _make_bubble(self, content: str, role: str, ts: Optional[str], is_preview: bool = False) -> tuple[QWidget, QLabel]:
        wrap = QWidget(self)
        wrap.setStyleSheet("background:transparent;")
        root_layout = QVBoxLayout(wrap)
        root_layout.setContentsMargins(6, 2, 6, 2)
        root_layout.setSpacing(2)
        content_label: Optional[QLabel] = None

        # 时间戳
        if ts:
            ts_lbl = QLabel(ts, wrap)
            ts_lbl.setStyleSheet("color:#9AAD94;font-size:11px;")
            ts_lbl.setAlignment(Qt.AlignHCenter)
            root_layout.addWidget(ts_lbl, 0, Qt.AlignHCenter)

        if role == "system":
            # 系统提示：居中淡绿灰 Chip
            chip = QLabel(content, wrap)
            chip.setWordWrap(True)
            chip.setStyleSheet(
                "background:#F2F6EE;color:#687863;border-radius:10px;padding:5px 10px;"
                "font-size:12px;"
            )
            chip.setAlignment(Qt.AlignCenter)
            chip.setMaximumWidth(int(self.viewport().width() * 0.85))
            root_layout.addWidget(chip, 0, Qt.AlignHCenter)
            content_label = chip
        else:
            is_user = role == "user"
            bubble_wrapper = QHBoxLayout()
            bubble_wrapper.setContentsMargins(0, 0, 0, 0)
            if is_user:
                bubble_wrapper.addStretch(1)

            bubble_lbl = QLabel(content, wrap)
            bubble_lbl.setWordWrap(True)
            bubble_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            if is_user:
                if is_preview:
                    bubble_lbl.setStyleSheet(
                        "background:#88B99A;color:rgba(255,255,255,0.9);border-radius:12px;"
                        "border-top-right-radius:3px;padding:8px 12px;"
                        "font-size:13px;line-height:1.6;font-style:italic;"
                    )
                else:
                    bubble_lbl.setStyleSheet(
                        "background:#5FA87C;color:white;border-radius:12px;"
                        "border-top-right-radius:3px;padding:8px 12px;"
                        "font-size:13px;line-height:1.6;"
                    )
            else:  # ai
                bubble_lbl.setStyleSheet(
                    "background:#F1F7EE;color:#26402E;border:1px solid #DAE9D2;"
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
            content_label = bubble_lbl

        return wrap, content_label


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
        self._default_fmt.setForeground(QColor("#24352A"))

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
            "QPlainTextEdit{background:#F5F9F3;color:#24352A;"
            "border:1px solid #E2ECDE;border-radius:8px;padding:8px;}"
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
