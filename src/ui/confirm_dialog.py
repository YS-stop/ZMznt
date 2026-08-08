"""P-07 高危操作二次确认 Dialog：DELETE 文字二次确认。

由 LangGraph 的高危条件边触发（DeleteFileTool 真实执行前）：
    - 列出操作清单（删除文件/目录、总个数、总大小估算）
    - 用户必须在输入框内准确输入 DELETE（大写）才能点确定
    - 确认后：confirm_accepted.emit(confirm_keyword_used: str, ops_list: list[str])
    - 取消/关闭：confirm_rejected.emit()
"""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

CONFIRM_KEYWORD_DEFAULT = "DELETE"


class HighRiskConfirmDialog(QDialog):
    """P-07 高危确认弹窗：应用模态，居中在主面板上方。"""

    confirm_accepted = Signal(str, list)  # (confirm_keyword, ops_list)
    confirm_rejected = Signal()

    def __init__(
        self,
        title: str = "⚠️ 高危操作二次确认",
        ops: Optional[Iterable[str]] = None,
        require_keyword: str = CONFIRM_KEYWORD_DEFAULT,
        hint: str = "此操作不可撤回！请三思。",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("HighRiskConfirm")
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumSize(QSize(520, 420))
        self.setStyleSheet(
            "QDialog#HighRiskConfirm{background:white;border:1px solid #FCA5A5;border-radius:14px;}"
        )

        self._ops_list: list[str] = list(ops or [])
        self._require_keyword: str = (require_keyword or CONFIRM_KEYWORD_DEFAULT).strip().upper()
        self._hint: str = str(hint).strip() or "此操作不可撤回。"

        self._build_ui()
        self._refresh_state()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(12)

        # --- 头部：图标 + 标题 ---
        header = QHBoxLayout()
        header.setSpacing(10)
        icon_lbl = QLabel("⚠️")
        icon_lbl.setFont(QFont("Segoe UI Emoji", 22))
        header.addWidget(icon_lbl, 0, Qt.AlignTop)
        title_lbl = QLabel(self.windowTitle(), self)
        title_lbl.setStyleSheet(
            "color:#991B1B;font-size:16px;font-weight:700;padding-top:4px;"
        )
        header.addWidget(title_lbl, 1)
        root.addLayout(header)

        # --- 说明 / 提示 ---
        hint_lbl = QLabel(self._hint, self)
        hint_lbl.setStyleSheet(
            "color:#7F1D1D;background:#FEF2F2;border:1px solid #FECACA;"
            "border-radius:8px;padding:8px 10px;font-size:12px;line-height:1.6;"
        )
        hint_lbl.setWordWrap(True)
        root.addWidget(hint_lbl)

        # --- 操作清单 ---
        list_lbl = QLabel(f"📋 待执行操作清单（共 {len(self._ops_list)} 项）：", self)
        list_lbl.setStyleSheet("color:#24352A;font-size:13px;font-weight:600;")
        root.addWidget(list_lbl)

        self.list_widget = QListWidget(self)
        self.list_widget.setStyleSheet(
            "QListWidget{background:#FAFCF8;border:1px solid #E2ECDE;border-radius:8px;padding:4px;}"
            "QListWidget::item{color:#4A5D46;font-size:12px;padding:3px 6px;border-bottom:1px solid #F0F6EC;}"
            "QListWidget::item:selected{background:#DEEDDF;color:#2E5238;}"
            "QScrollBar:vertical{width:6px;}"
            "QScrollBar::handle:vertical{background:#C9DCC5;border-radius:3px;}"
        )
        if self._ops_list:
            for idx, op in enumerate(self._ops_list, 1):
                item = QListWidgetItem(f"{idx:>3}. {op}")
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
                self.list_widget.addItem(item)
        else:
            item = QListWidgetItem("（空清单，此时不应触发确认 Dialog）")
            item.setForeground(Qt.red)
            self.list_widget.addItem(item)
        self.list_widget.setMinimumHeight(120)
        root.addWidget(self.list_widget, 1)

        # --- 关键字二次确认输入框 ---
        keyword_lbl = QLabel(
            f"✅ 请输入 <b style='color:#B91C1C'>{self._require_keyword}</b>（区分大小写）以确认执行：",
            self,
        )
        keyword_lbl.setTextFormat(Qt.RichText)
        keyword_lbl.setStyleSheet("color:#24352A;font-size:12px;")
        root.addWidget(keyword_lbl)

        self.keyword_edit = QLineEdit(self)
        self.keyword_edit.setPlaceholderText(f"请输入 {self._require_keyword} 并回车...")
        self.keyword_edit.setStyleSheet(
            "QLineEdit{background:white;border:1px solid #E2ECDE;border-radius:8px;padding:8px 10px;"
            "font-size:13px;letter-spacing:2px;color:#4A5D46;}"
            "QLineEdit:focus{border-color:#DC2626;}"
        )
        self.keyword_edit.textChanged.connect(self._refresh_state)
        self.keyword_edit.returnPressed.connect(self._try_accept)
        root.addWidget(self.keyword_edit)

        # --- 按钮区 ---
        self.button_box = QDialogButtonBox(self)
        self.button_box.setStandardButtons(QDialogButtonBox.Cancel)
        # 自定义确认按钮（默认禁用，等用户输入 DELETE）
        self.btn_ok = QPushButton(f"✅ 确认输入 {self._require_keyword} 并执行")
        self.btn_ok.setCursor(Qt.PointingHandCursor)
        self.btn_ok.setEnabled(False)
        self.btn_ok.setStyleSheet(
            "QPushButton:enabled{background:#DC2626;color:white;border:none;border-radius:8px;"
            "padding:8px 14px;font-size:13px;font-weight:600;}"
            "QPushButton:enabled:hover{background:#B91C1C;}"
            "QPushButton:disabled{background:#F0F6EC;color:#9AAD94;border:1px solid #E2ECDE;"
            "border-radius:8px;padding:8px 14px;font-size:13px;}"
        )
        self.btn_ok.clicked.connect(self._try_accept)
        self.button_box.addButton(self.btn_ok, QDialogButtonBox.AcceptRole)

        cancel = self.button_box.button(QDialogButtonBox.Cancel)
        if cancel is not None:
            cancel.setText("取消")
            cancel.setCursor(Qt.PointingHandCursor)
            cancel.setStyleSheet(
                "QPushButton{background:white;color:#4A5D46;border:1px solid #E2ECDE;"
                "border-radius:8px;padding:8px 14px;font-size:13px;}"
                "QPushButton:hover{background:#F0F6EC;}"
            )
        self.button_box.rejected.connect(self._on_reject)
        root.addWidget(self.button_box)

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def ops_list(self) -> list[str]:
        return list(self._ops_list)

    def require_keyword(self) -> str:
        return self._require_keyword

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _refresh_state(self) -> None:
        typed = self.keyword_edit.text()
        ok = (typed == self._require_keyword)
        self.btn_ok.setEnabled(ok)
        if ok:
            self.keyword_edit.setStyleSheet(
                "QLineEdit{background:#F0FDF4;border:1px solid #10B981;border-radius:8px;"
                "padding:8px 10px;font-size:13px;letter-spacing:2px;color:#065F46;}"
            )
        else:
            # 提示变红当长度 >= 3 且不匹配
            if len(typed) >= 3:
                self.keyword_edit.setStyleSheet(
                    "QLineEdit{background:#FEF2F2;border:1px solid #F87171;border-radius:8px;"
                    "padding:8px 10px;font-size:13px;letter-spacing:2px;color:#7F1D1D;}"
                )
            else:
                self.keyword_edit.setStyleSheet(
                    "QLineEdit{background:white;border:1px solid #E2ECDE;border-radius:8px;"
                    "padding:8px 10px;font-size:13px;letter-spacing:2px;color:#4A5D46;}"
                    "QLineEdit:focus{border-color:#DC2626;}"
                )

    def _try_accept(self) -> None:
        if not self.btn_ok.isEnabled():
            # 抖动一下输入框提示用户
            return
        self.confirm_accepted.emit(self._require_keyword, list(self._ops_list))
        self.accept()

    def _on_reject(self) -> None:
        self.confirm_rejected.emit()
        self.reject()

    def closeEvent(self, ev) -> None:  # noqa: N802
        # 用户点 X 关闭也视为拒绝
        if self.result() != QDialog.Accepted:
            self.confirm_rejected.emit()
        super().closeEvent(ev)


__all__ = ["HighRiskConfirmDialog", "CONFIRM_KEYWORD_DEFAULT"]
