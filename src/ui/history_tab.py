"""P-05 历史会话 Tab：左侧日期分组列表 + 右侧详情气泡回放。

M5 阶段使用 mock 数据（3 天 × 3 条会话），M6 接入 SQLite CheckpointService 后会真实拉取历史。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets.common import BubbleListWidget, ChipButton


@dataclass
class HistoryEntry:
    """一条历史会话。"""
    id: str
    date_str: str          # "2026-08-05"
    time_str: str          # "14:28"
    title: str             # 首句摘要
    message_count: int     # 消息总数
    turns: list[tuple[str, str]]  # (role='user'|'ai'|'system', text)


_MOCK: list[HistoryEntry] = [
    HistoryEntry(
        id="h_20260805_1428",
        date_str="2026-08-05",
        time_str="14:28",
        title="帮我在桌面创建一个项目清单文件",
        message_count=6,
        turns=[
            ("user", "帮我在桌面创建一个项目清单文件，列出 M1~M6 待办。"),
            ("ai", "好的，已经创建：D:/Users/xxx/Desktop/项目清单.txt"),
            ("system", "✅ create_file 成功写入 1.2 KB。"),
            ("user", "再在桌面搜索「会议纪要」。"),
            ("ai", "找到 2 个文件：会议纪要_0801.docx、会议纪要_0804.md。"),
            ("ai", "需要我打开哪个吗？"),
        ],
    ),
    HistoryEntry(
        id="h_20260805_1803",
        date_str="2026-08-05",
        time_str="18:03",
        title="今天 AI 圈有什么新闻？",
        message_count=4,
        turns=[
            ("user", "今天 AI 圈有什么新闻？"),
            ("ai", "正在搜索最新资讯…（调用 search_news）"),
            ("system", "[TOOL] search_news → 命中 8 条 Bing 新闻。"),
            ("ai", "摘要：\n① 多模态推理模型 Benchmark 更新…\n② Agent 框架 LangGraph 发布 0.2.x…"),
        ],
    ),
    HistoryEntry(
        id="h_20260806_0947",
        date_str="2026-08-06",
        time_str="09:47",
        title="删除桌面 temp-xxxx 临时文件",
        message_count=5,
        turns=[
            ("user", "帮我删除桌面 temp-xxxx 开头的临时文件。"),
            ("system", "⚠️ delete_file 高危操作：共 3 个文件 2.4MB，请输入 DELETE 二次确认。"),
            ("user", "DELETE"),
            ("system", "✅ 3 个文件已移动到回收站（send2trash）。"),
            ("ai", "好，已经把 3 个 temp-xxxx 文件送进回收站，可随时还原。"),
        ],
    ),
    HistoryEntry(
        id="h_20260807_1012",
        date_str="2026-08-07",
        time_str="10:12",
        title="M5 阶段 UI 验收：悬浮球+抽屉+设置+调试",
        message_count=3,
        turns=[
            ("user", "M5 阶段 UI 验收：显示悬浮球，展开抽屉，打开设置中心，打开调试面板。"),
            ("system", "🟦 [ASR]  (键盘输入旁路) 识别文本： 58 字"),
            ("ai", "好！已按顺序执行：悬浮球可见 → 抽屉滑出 → 设置中心打开 → 调试面板实时显示本消息链路 ✅"),
        ],
    ),
]


class HistoryTabPage(QWidget):
    """P-05 历史 Tab：替代 P-02 抽屉的 placeholder。"""

    request_continue = Signal(str)    # session_id：「继续该会话」
    request_delete = Signal(str)      # session_id：「删除该会话」

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._entries: list[HistoryEntry] = list(_MOCK)
        self._current_id: Optional[str] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(
            "QSplitter::handle{background:#E2E8F0;}"
        )

        # 左侧：列表
        left = self._build_left(splitter)
        splitter.addWidget(left)

        # 右侧：详情气泡
        right = self._build_right(splitter)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([260, 560])

        root.addWidget(splitter, 1)

        # 默认选中第一个
        if self._entries:
            self.list_widget.setCurrentRow(0)
            self._on_current_changed(0)

    # ------------------------------------------------------------------
    # 构建
    # ------------------------------------------------------------------

    def _build_left(self, parent: QWidget) -> QWidget:
        wrap = QFrame(parent)
        wrap.setStyleSheet(
            "QFrame{background:#FAFBFC;border:1px solid #E2E8F0;border-radius:10px;}"
        )
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("📜 历史会话", wrap)
        title.setStyleSheet("color:#0F172A;font-size:14px;font-weight:700;")
        header.addWidget(title, 1)
        btn_refresh = ChipButton("刷新", active=False, parent=wrap)
        btn_refresh.clicked.connect(self.reload_entries)
        header.addWidget(btn_refresh)
        lay.addLayout(header)

        subtitle = QLabel(f"共 {len(self._entries)} 条，按日期分组（mock 数据预览）", wrap)
        subtitle.setStyleSheet("color:#64748B;font-size:11px;padding-left:2px;")
        lay.addWidget(subtitle)

        self.list_widget = QListWidget(wrap)
        self.list_widget.setStyleSheet(
            "QListWidget{background:white;border:1px solid #E2E8F0;border-radius:8px;padding:2px;}"
            "QListWidget::item{border-radius:6px;padding:8px;margin:2px 2px;}"
            "QListWidget::item:selected{background:#DBEAFE;color:#1E3A8A;}"
            "QListWidget::item:hover:!selected{background:#F8FAFC;}"
            "QScrollBar:vertical{width:5px;}"
            "QScrollBar::handle:vertical{background:#CBD5E1;border-radius:3px;}"
        )
        self.list_widget.currentRowChanged.connect(self._on_current_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._on_list_menu)
        lay.addWidget(self.list_widget, 1)

        self._fill_list()
        return wrap

    def _fill_list(self) -> None:
        self.list_widget.clear()
        last_date = None
        for idx, entry in enumerate(self._entries):
            # 日期分组 header
            if entry.date_str != last_date:
                sep = QListWidgetItem(f"📅  {entry.date_str}")
                sep.setFlags(Qt.NoItemFlags)
                sep.setForeground(Qt.darkGray)
                f = QFont()
                f.setPointSize(9)
                f.setBold(True)
                sep.setFont(f)
                sep.setSizeHint(QSize(200, 22))
                self.list_widget.addItem(sep)
                last_date = entry.date_str
            # 条目
            item = QListWidgetItem()
            item.setData(Qt.UserRole, entry.id)
            item.setSizeHint(QSize(220, 64))
            self.list_widget.addItem(item)
            card = self._make_entry_card(entry, self.list_widget)
            self.list_widget.setItemWidget(item, card)

    def _make_entry_card(self, entry: HistoryEntry, parent: QWidget) -> QWidget:
        card = QWidget(parent)
        card.setStyleSheet("background:transparent;")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)

        top = QHBoxLayout()
        time_lbl = QLabel(f"⏰ {entry.time_str}", card)
        time_lbl.setStyleSheet("color:#64748B;font-size:10px;")
        top.addWidget(time_lbl, 0)
        cnt_lbl = QLabel(f"{entry.message_count} 条", card)
        cnt_lbl.setStyleSheet("color:#94A3B8;font-size:10px;")
        cnt_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top.addWidget(cnt_lbl, 1)
        lay.addLayout(top)

        title_lbl = QLabel(entry.title, card)
        title_lbl.setStyleSheet("color:#0F172A;font-size:12px;font-weight:600;")
        title_lbl.setWordWrap(True)
        lay.addWidget(title_lbl)
        return card

    def _build_right(self, parent: QWidget) -> QWidget:
        wrap = QFrame(parent)
        wrap.setStyleSheet(
            "QFrame{background:white;border:1px solid #E2E8F0;border-radius:10px;}"
        )
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # 顶部 header：会话标题 + 继续/删除
        head = QHBoxLayout()
        self.head_title = QLabel("← 请从左侧选择一条历史会话进行回看", wrap)
        self.head_title.setStyleSheet("color:#0F172A;font-size:13px;font-weight:600;")
        head.addWidget(self.head_title, 1)
        self.btn_continue = ChipButton("➡️  继续该会话", active=True, parent=wrap)
        self.btn_continue.clicked.connect(self._on_continue)
        self.btn_continue.setEnabled(False)
        head.addWidget(self.btn_continue)
        self.btn_delete = ChipButton("🗑 删除", active=False, parent=wrap)
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_delete.setEnabled(False)
        head.addWidget(self.btn_delete)
        lay.addLayout(head)

        # 中间：气泡回放
        self.detail_bubbles = BubbleListWidget(wrap)
        lay.addWidget(self.detail_bubbles, 1)

        # 底部：状态摘要
        self.summary_lbl = QLabel("（暂无内容）", wrap)
        self.summary_lbl.setStyleSheet("color:#64748B;font-size:11px;padding:4px 6px;")
        lay.addWidget(self.summary_lbl)
        return wrap

    # ------------------------------------------------------------------
    # 对外：重载数据
    # ------------------------------------------------------------------

    def set_entries(self, entries: list[HistoryEntry]) -> None:
        self._entries = list(entries)
        self._fill_list()
        summary = self.findChild(QLabel, "", Qt.FindDirectChildrenOnly)

    def reload_entries(self) -> None:
        """M6 接入 SQLite 后会改成读 CheckpointService。"""
        self.set_entries(_MOCK)

    # ------------------------------------------------------------------
    # 内部事件
    # ------------------------------------------------------------------

    def _current_entry(self) -> Optional[HistoryEntry]:
        if self._current_id is None:
            return None
        for e in self._entries:
            if e.id == self._current_id:
                return e
        return None

    def _on_current_changed(self, row: int) -> None:
        if row < 0 or row >= self.list_widget.count():
            return
        item = self.list_widget.item(row)
        if item is None:
            return
        sid = item.data(Qt.UserRole)
        if sid is None:
            return  # 日期分组 header
        self._current_id = str(sid)
        entry = self._current_entry()
        if entry is None:
            return
        # 渲染右侧
        self.head_title.setText(f"📜  {entry.date_str}  {entry.time_str}  ·  {entry.title}")
        self.btn_continue.setEnabled(True)
        self.btn_delete.setEnabled(True)
        self.detail_bubbles.clear()
        for role, text in entry.turns:
            if role == "user":
                self.detail_bubbles.append_user(text, timestamp=entry.time_str)
            elif role == "system":
                self.detail_bubbles.append_system(text, timestamp=entry.time_str)
            else:
                self.detail_bubbles.append_ai(text, timestamp=entry.time_str)
        self.summary_lbl.setText(
            f"共 {len(entry.turns)} 条消息 · 用户 {sum(1 for r,_ in entry.turns if r=='user')} · AI {sum(1 for r,_ in entry.turns if r=='ai')} · 系统 {sum(1 for r,_ in entry.turns if r=='system')}"
        )

    def _on_list_menu(self, pos) -> None:
        item = self.list_widget.itemAt(pos)
        if item is None:
            return
        sid = item.data(Qt.UserRole)
        if sid is None:
            return
        menu = QMenu(self.list_widget)
        act_open = menu.addAction("继续该会话")
        act_del = menu.addAction("删除该会话")
        act = menu.exec(self.list_widget.viewport().mapToGlobal(pos))
        if act == act_open:
            self._current_id = str(sid)
            self._on_continue()
        elif act == act_del:
            self._current_id = str(sid)
            self._on_delete()

    def _on_continue(self) -> None:
        if self._current_id:
            self.request_continue.emit(self._current_id)

    def _on_delete(self) -> None:
        if self._current_id:
            self.request_delete.emit(self._current_id)
            # UI 层立即移除（mock）
            self._entries = [e for e in self._entries if e.id != self._current_id]
            self._fill_list()
            self._current_id = None
            self.head_title.setText("← 请从左侧选择一条历史会话进行回看")
            self.btn_continue.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.detail_bubbles.clear()
            self.summary_lbl.setText("（暂无内容）")


__all__ = ["HistoryTabPage", "HistoryEntry"]
