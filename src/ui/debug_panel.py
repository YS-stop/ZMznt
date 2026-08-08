"""P-03 消息同步调试面板：分 5 个 AccordionSection 折叠显示 + 全链路总彩色日志。

P-02 的「调试」Tab 页面直接用本组件替换占位。

对外方法：
    append_stage(stage: str, message: str)  stage 取 ["ASR","AGENT","TOOL","OBS","TTS","SYSTEM","ERR","WARN"]
    set_stage_expanded(stage: str, expanded: bool)
    clear_all()

信号：无（纯展示用）。
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets.common import AccordionSection, DebugLogView, ChipButton


_STAGE_ORDER = ["ASR", "AGENT", "TOOL", "OBS", "TTS", "ERR", "WARN", "SYSTEM"]
# Accordion 标题（图标 + 中文）
_STAGE_TITLES = {
    "ASR":     "🎙️ ASR 阶段（语音转文字）",
    "AGENT":   "🧠 Agent 阶段（LLM 思考 + ReAct）",
    "TOOL":    "🛠 Tool 执行阶段（调用 8 个 LangChain Tool）",
    "OBS":     "📨 Observation 阶段（工具返回结果）",
    "TTS":     "🔊 TTS 播报阶段（语音合成）",
    "SYSTEM":  "🛡 System（系统保护/白名单/黑名单）",
    "ERR":     "❌ Errors（错误/异常）",
    "WARN":    "⚠️  Warnings（告警）",
}

_STAGE_ACCORDION_MAPPING = {
    "ASR": "ASR",
    "AGENT": "AGENT",
    "LLM":   "AGENT",
    "THOUGHT": "AGENT",
    "TOOL": "TOOL",
    "ACT":  "TOOL",
    "ACTION": "TOOL",
    "OBS":  "OBS",
    "RES":  "OBS",
    "RESULT": "OBS",
    "OBSERVATION": "OBS",
    "TTS":  "TTS",
    "SPEAK": "TTS",
    "SYSTEM": "SYSTEM",
    "SECURITY": "SYSTEM",
    "ERR": "ERR",
    "ERROR": "ERR",
    "FAIL": "ERR",
    "WARN": "WARN",
    "WARNING": "WARN",
}


class DebugPanel(QWidget):
    """P-03 调试面板：分阶段 Accordion + 全局总览。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._accordions: dict[str, tuple[AccordionSection, DebugLogView]] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 顶栏：标题 + 快捷操作（全部展开 / 全部收起 / 清空）
        top = self._build_topbar()
        root.addWidget(top)

        # 阶段 Accordions 容器（Scroll）
        stage_wrap = QWidget(self)
        stage_lay = QVBoxLayout(stage_wrap)
        stage_lay.setContentsMargins(0, 0, 0, 0)
        stage_lay.setSpacing(8)

        self._stage_widgets_ordered: list[tuple[str, AccordionSection, DebugLogView]] = []
        for key in _STAGE_ORDER:
            accordion, view = self._make_stage(key)
            self._accordions[key] = (accordion, view)
            self._stage_widgets_ordered.append((key, accordion, view))
            stage_lay.addWidget(accordion)
        stage_lay.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(stage_wrap)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            "QScrollBar::handle:vertical{background:#CBD5E1;border-radius:3px;}"
            "QScrollBar::handle:vertical:hover{background:#94A3B8;}"
        )
        root.addWidget(scroll, 1)

        # 最底部：全局总日志（默认展开）
        total_accord, total_view = self._make_stage("TOTAL", title="📋 全链路总日志（按时间顺序）", expanded=True)
        self._accordions["TOTAL"] = (total_accord, total_view)
        self._stage_widgets_ordered.append(("TOTAL", total_accord, total_view))
        root.addWidget(total_accord, 0)

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_topbar(self) -> QWidget:
        wrap = QWidget(self)
        lay = QHBoxLayout(wrap)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(8)

        title = QLabel("🛠 消息同步调试面板（ASR → Agent → Tool → Obs → TTS）")
        title.setStyleSheet(
            "color:#1E293B;font-size:13px;font-weight:700;padding:4px 0;"
        )
        lay.addWidget(title, 1)

        btn_expand = ChipButton("全部展开", active=False, parent=wrap)
        btn_expand.clicked.connect(lambda: self._set_all_expanded(True))
        lay.addWidget(btn_expand)

        btn_collapse = ChipButton("全部收起", active=False, parent=wrap)
        btn_collapse.clicked.connect(lambda: self._set_all_expanded(False))
        lay.addWidget(btn_collapse)

        btn_clear = ChipButton("清空日志", active=False, parent=wrap)
        btn_clear.clicked.connect(self.clear_all)
        lay.addWidget(btn_clear)
        return wrap

    def _make_stage(self, key: str, title: Optional[str] = None, expanded: bool = False):
        t = title or _STAGE_TITLES.get(key, key)
        accord = AccordionSection(t, expanded=expanded, parent=self)
        view = DebugLogView(accord)
        accord.setContent(view)
        return accord, view

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def append_stage(self, stage: str, message: str) -> None:
        """按阶段写入日志：TOTAL 永远追加；stage 对应的分阶段 Accordion 也追加。"""
        s_raw = (stage or "").strip().upper()
        mapped = _STAGE_ACCORDION_MAPPING.get(s_raw)
        # 1. TOTAL 全局
        if "TOTAL" in self._accordions:
            self._accordions["TOTAL"][1].append_stage(s_raw or "INFO", message)
        # 2. 分阶段
        if mapped and mapped in self._accordions:
            self._accordions[mapped][1].append_stage(s_raw, message)
        # ERR / WARN 同时也写到 ERR / WARN 分组（如果还没去重）
        if s_raw in ("ERR", "ERROR", "FAIL") and mapped != "ERR" and "ERR" in self._accordions:
            self._accordions["ERR"][1].append_stage("ERR", message)
        if s_raw in ("WARN", "WARNING") and mapped != "WARN" and "WARN" in self._accordions:
            self._accordions["WARN"][1].append_stage("WARN", message)

    def set_stage_expanded(self, stage: str, expanded: bool) -> None:
        s = (stage or "").strip().upper()
        mapped = _STAGE_ACCORDION_MAPPING.get(s, s)
        if mapped in self._accordions:
            self._accordions[mapped][0].setExpanded(bool(expanded))

    def clear_all(self) -> None:
        for _, view in self._accordions.values():
            view.clear()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _set_all_expanded(self, expanded: bool) -> None:
        for accord, _ in self._accordions.values():
            accord.setExpanded(bool(expanded))


__all__ = ["DebugPanel"]
