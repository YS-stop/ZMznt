"""P-01 桌面悬浮球：无边框+始终置顶+可拖拽+4 状态自绘动画。

4 状态（对应 PRD 5.0 悬浮球形态）：
    idle        = 待机：叶青绿渐变静态，显示 🎙️ 图标
    listening   = 录音中：红圈脉冲动画，显示 🔴 录音提示
    thinking    = 思考中：琥珀色转圈动画，显示 ⏳
    speaking    = 播报中：翠绿色呼吸动画，显示 🔊

对外信号（Qt Signal）：
    clicked             左键点击（用于展开/收起 P-02 抽屉）
    request_settings    右键菜单「设置」
    request_quit        右键菜单「退出」
    request_voice       长按左键（按住说话），释放时触发录音结束（可选，当前短按即可）

拖拽：鼠标按住非中心区域可拖到屏幕任意位置，自动吸附屏幕左右边缘（可选，默认开启）。
"""
from __future__ import annotations

import math
from enum import IntEnum
from typing import Optional

from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QConicalGradient, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QMenu, QWidget


class FloatingBallState(IntEnum):
    IDLE = 0       # 待机（蓝渐变）
    LISTENING = 1  # 录音（红脉冲）
    THINKING = 2   # 思考（琥珀旋转）
    SPEAKING = 3   # 播报（绿呼吸）


_STATE_ICON = {
    FloatingBallState.IDLE: "🎙️",
    FloatingBallState.LISTENING: "🔴",
    FloatingBallState.THINKING: "⏳",
    FloatingBallState.SPEAKING: "🔊",
}

_STATE_NAME = {
    FloatingBallState.IDLE: "IDLE",
    FloatingBallState.LISTENING: "LISTENING",
    FloatingBallState.THINKING: "THINKING",
    FloatingBallState.SPEAKING: "SPEAKING",
}


class FloatingBallWidget(QWidget):
    """桌面悬浮球：P-01。"""

    # 对外信号
    clicked = Signal()                        # 左键点击（单击，不拖拽时）
    right_clicked = Signal(QPoint)            # 右键点击，位置
    request_settings = Signal()               # 设置
    request_quit = Signal()                   # 退出
    request_voice_start = Signal()            # 按住说话开始
    request_voice_end = Signal()              # 按住说话结束

    # 尺寸常量
    SIZE = 64
    PULSE_INTERVAL = 40   # ms
    EDGE_MARGIN = 10      # 离屏幕边缘吸附的最小距离

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # 无边框 + 始终置顶 + 工具窗口（不占任务栏）
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)

        self.setFixedSize(QSize(self.SIZE, self.SIZE))
        self.setCursor(Qt.PointingHandCursor)

        # 状态
        self._state = FloatingBallState.IDLE
        self._anim_tick = 0.0            # 动画相位，每个 PULSE_INTERVAL +0.1
        self._drag_offset: Optional[QPoint] = None  # 拖拽用：鼠标在控件内的相对位置
        self._pressed_pos: Optional[QPoint] = None   # 记录按下位置，判断单击 vs 拖拽
        self._pressed_global_pos: Optional[QPoint] = None
        self._moved_during_press = False
        self._edge_snap = True           # 自动吸附左右边缘

        # 动画定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(self.PULSE_INTERVAL)

        # 默认位置：屏幕右下角（离边缘 EDGE_MARGIN px）
        self._place_initial()

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def state(self) -> FloatingBallState:
        return self._state

    def setState(self, state: FloatingBallState) -> None:
        if not isinstance(state, FloatingBallState):
            try:
                state = FloatingBallState(int(state))
            except Exception:
                state = FloatingBallState.IDLE
        if self._state == state:
            return
        self._state = state
        self._anim_tick = 0.0
        self.update()

    def setEdgeSnap(self, enable: bool) -> None:  # noqa: N802
        self._edge_snap = bool(enable)

    # ------------------------------------------------------------------
    # 内部：位置
    # ------------------------------------------------------------------

    def _place_initial(self) -> None:
        screen = self.screen().availableGeometry() if self.screen() else QRectF(0, 0, 1920, 1080)
        x = int(screen.width() - self.width() - self.EDGE_MARGIN)
        y = int(screen.height() - self.height() - 120)
        self.move(x, y)

    # ------------------------------------------------------------------
    # 内部：动画
    # ------------------------------------------------------------------

    def _on_tick(self) -> None:
        self._anim_tick = (self._anim_tick + 0.1) % (math.pi * 200)
        if self._state in (FloatingBallState.LISTENING, FloatingBallState.THINKING, FloatingBallState.SPEAKING):
            self.update()

    # ------------------------------------------------------------------
    # 内部：右键菜单
    # ------------------------------------------------------------------

    def _show_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        act_show = QAction("显示 / 隐藏主面板", self)
        act_show.triggered.connect(self.clicked.emit)
        menu.addAction(act_show)
        menu.addSeparator()
        act_settings = QAction("设置…", self)
        act_settings.triggered.connect(self.request_settings.emit)
        menu.addAction(act_settings)
        act_quit = QAction("退出程序", self)
        act_quit.triggered.connect(self.request_quit.emit)
        menu.addAction(act_quit)
        menu.exec(global_pos)

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------

    def paintEvent(self, ev) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        size = self.SIZE
        cx, cy = size / 2, size / 2
        t = self._anim_tick

        # ----- 外圈：根据状态绘制动画 -----
        state = self._state
        if state == FloatingBallState.IDLE:
            # 外圈柔和阴影
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 30))
            p.drawEllipse(3, 5, size - 6, size - 6)

        elif state == FloatingBallState.LISTENING:
            # 录音中：红脉冲双圈
            pulse = 0.5 + 0.5 * math.sin(t * 2.0)
            for i, (r_base, a) in enumerate(((28, 40 + 60 * pulse), (32, 80 + 100 * pulse))):
                col = QColor(239, 68, 68, int(max(20, min(200, a))))
                pen = QPen(col, max(1, int(2 + 2 * pulse)))
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                r = r_base + pulse * 3
                p.drawEllipse(QPointF(cx, cy), r, r)

        elif state == FloatingBallState.THINKING:
            # 思考：琥珀色锥形旋转
            grd = QConicalGradient(QPointF(cx, cy), -t * 60)
            grd.setColorAt(0.0, QColor(245, 158, 11, 220))
            grd.setColorAt(0.5, QColor(251, 191, 36, 40))
            grd.setColorAt(1.0, QColor(245, 158, 11, 220))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grd))
            p.drawEllipse(2, 2, size - 4, size - 4)

        elif state == FloatingBallState.SPEAKING:
            # 播报：绿呼吸外圈
            breath = 0.5 + 0.5 * math.sin(t * 1.4)
            outer_r = 30 + 2 * breath
            col = QColor(16, 185, 129, int(60 + 140 * breath))
            pen = QPen(col, max(1, int(2 + 2 * breath)))
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx, cy), outer_r, outer_r)

        # ----- 内球渐变 -----
        if state == FloatingBallState.IDLE:
            # 叶青绿渐变（淡绿自然主题）
            grd = QLinearGradient(0, 0, size, size)
            grd.setColorAt(0.0, QColor("#92CDA6"))
            grd.setColorAt(1.0, QColor("#5FA87C"))
        elif state == FloatingBallState.LISTENING:
            grd = QLinearGradient(0, 0, size, size)
            grd.setColorAt(0.0, QColor("#FCA5A5"))
            grd.setColorAt(1.0, QColor("#EF4444"))
        elif state == FloatingBallState.THINKING:
            grd = QLinearGradient(0, 0, size, size)
            grd.setColorAt(0.0, QColor("#FCD34D"))
            grd.setColorAt(1.0, QColor("#F59E0B"))
        else:  # SPEAKING
            grd = QLinearGradient(0, 0, size, size)
            grd.setColorAt(0.0, QColor("#6EE7B7"))
            grd.setColorAt(1.0, QColor("#10B981"))
        p.setPen(Qt.NoPen)
        p.setBrush(grd)
        p.drawEllipse(8, 8, size - 16, size - 16)

        # ----- 高光（左上 1/4 椭圆白透） -----
        hl = QLinearGradient(0, 0, 0, size / 2)
        hl.setColorAt(0.0, QColor(255, 255, 255, 160))
        hl.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setBrush(hl)
        p.drawEllipse(12, 12, size - 36, size // 3)

        # ----- 状态图标 -----
        p.setPen(Qt.NoPen)
        p.setBrush(Qt.white)
        font = QFont("Segoe UI Emoji, Microsoft YaHei UI", 16, QFont.Bold)
        p.setFont(font)
        rect = QRectF(4, 2, size - 8, size - 4)
        p.drawText(rect, Qt.AlignCenter, _STATE_ICON.get(state, "🎙️"))

        p.end()

    # ------------------------------------------------------------------
    # 鼠标：拖拽 + 单击/长按
    # ------------------------------------------------------------------

    def mousePressEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            self._drag_offset = ev.position().toPoint()
            self._pressed_pos = ev.position().toPoint()
            self._pressed_global_pos = ev.globalPosition().toPoint()
            self._moved_during_press = False
            # 长按开始
            QTimer.singleShot(250, self._maybe_voice_start)
            return
        if ev.button() == Qt.RightButton:
            self.right_clicked.emit(ev.globalPosition().toPoint())
            self._show_menu(ev.globalPosition().toPoint())
            return
        super().mousePressEvent(ev)

    def _maybe_voice_start(self) -> None:
        # 250ms 后用户仍然按住左键 + 没有移动 → 视为「按住说话」
        if (
            self._pressed_pos is not None
            and not self._moved_during_press
            and self._drag_offset is not None
        ):
            self.request_voice_start.emit()

    def mouseMoveEvent(self, ev) -> None:  # noqa: N802
        if self._drag_offset is not None and ev.buttons() & Qt.LeftButton:
            pos = ev.globalPosition().toPoint() - self._drag_offset
            # 判定是否移动过（阈值 3px）
            if self._pressed_pos is not None:
                d = ev.position().toPoint() - self._pressed_pos
                if d.manhattanLength() > 3:
                    self._moved_during_press = True
            self.move(pos)
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev) -> None:  # noqa: N802
        if ev.button() == Qt.LeftButton:
            was_moving = bool(self._moved_during_press)
            # 结束录音
            if not was_moving and self._pressed_pos is not None:
                # 判定为短按 → 发送 clicked
                self.clicked.emit()
            else:
                # 拖拽结束 → 吸附左右边缘
                if self._edge_snap:
                    self._snap_to_edge()
            self.request_voice_end.emit()
            self._drag_offset = None
            self._pressed_pos = None
            self._pressed_global_pos = None
            self._moved_during_press = False
            return
        super().mouseReleaseEvent(ev)

    def _snap_to_edge(self) -> None:
        screen = self.screen().availableGeometry() if self.screen() else None
        if screen is None:
            return
        cur = self.geometry()
        if cur.center().x() < (screen.left() + screen.width() / 2):
            # 左半
            x = screen.left() + self.EDGE_MARGIN
        else:
            x = screen.right() - self.width() - self.EDGE_MARGIN
        y = max(screen.top(), min(screen.bottom() - self.height(), cur.y()))
        self.move(x, y)


__all__ = ["FloatingBallWidget", "FloatingBallState"]
