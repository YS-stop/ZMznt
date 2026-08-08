"""P-04 设置中心：左侧导航 Chip + 右侧 5 页 Stacked（通用/语音/快捷键/AI Key/隐私）。

保存策略：
    - 实时写入 .env 和 settings.json（通过 settings_service，本阶段 M5 先做成纯 UI 展示，
      真实的 settings_service 等用户需要再接入；此时可直接读 .env 预览当前值）
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.ui.widgets.common import ChipButton, SwitchToggle


_SETTINGS_PAGES = [
    ("general",   "⚙️  通用设置"),
    ("voice",     "🎙️  语音 / ASR & TTS"),
    ("hotkeys",   "⌨️  快捷键"),
    ("ai",        "🤖  AI & API 密钥"),
    ("privacy",   "🛡  隐私 & 权限"),
]


def _form_hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#6E7F6A;font-size:11px;margin-top:-4px;")
    lbl.setWordWrap(True)
    return lbl


class SettingsWindow(QMainWindow):
    """P-04 设置中心：模态显示在主面板之上。"""

    settings_changed = Signal(str, object)  # key -> value
    request_close = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("⚙️  桌面语音助手 · 设置中心")
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumSize(QSize(820, 620))
        self.resize(880, 680)

        self.setAttribute(Qt.WA_DeleteOnClose, False)
        # 先初始化容器，避免 _build_ui 内部引用时 AttributeError
        self._nav_btns: dict[str, ChipButton] = {}
        self._pages: dict[str, QWidget] = {}
        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("SettingsOuter")
        central.setStyleSheet(
            "QWidget#SettingsOuter{background:#F5F9F3;}"
        )
        self.setCentralWidget(central)
        shadow = QGraphicsDropShadowEffect(central)
        shadow.setBlurRadius(20)
        shadow.setColor(central.palette().window().color())

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # 左侧导航栏
        nav = self._build_nav_bar(central)
        root.addWidget(nav, 0, Qt.AlignTop)

        # 右侧页面区（卡片 + 阴影）
        page_card = QFrame(central)
        page_card.setObjectName("PageCard")
        page_card.setStyleSheet(
            "QFrame#PageCard{background:white;border:1px solid #E2ECDE;border-radius:14px;}"
        )
        shadow2 = QGraphicsDropShadowEffect(page_card)
        shadow2.setBlurRadius(20)
        shadow2.setXOffset(0)
        shadow2.setYOffset(4)
        from PySide6.QtGui import QColor
        shadow2.setColor(QColor(15, 23, 42, 20))
        page_card.setGraphicsEffect(shadow2)

        page_root = QVBoxLayout(page_card)
        page_root.setContentsMargins(0, 0, 0, 0)
        page_root.setSpacing(0)

        # 页面卡片头部：标题 + 保存 / 关闭按钮
        head = QFrame(page_card)
        head.setStyleSheet(
            "QFrame{background:white;border-bottom:1px solid #E2ECDE;"
            "border-top-left-radius:14px;border-top-right-radius:14px;}"
        )
        head.setFixedHeight(60)
        head_lay = QHBoxLayout(head)
        head_lay.setContentsMargins(22, 0, 18, 0)
        self.page_title_lbl = QLabel(_SETTINGS_PAGES[0][1], head)
        self.page_title_lbl.setStyleSheet("color:#24352A;font-size:16px;font-weight:700;")
        head_lay.addWidget(self.page_title_lbl, 1)

        btn_save = QPushButton("💾 保存", head)
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setStyleSheet(
            "QPushButton{background:#5FA87C;color:white;border-radius:8px;"
            "padding:7px 14px;font-size:12px;font-weight:600;border:none;}"
            "QPushButton:hover{background:#4D9269;}"
        )
        btn_save.clicked.connect(self._on_save_clicked)
        head_lay.addWidget(btn_save)

        btn_close = QPushButton("关闭", head)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet(
            "QPushButton{background:white;color:#4A5D46;border:1px solid #E2ECDE;"
            "border-radius:8px;padding:7px 14px;font-size:12px;}"
            "QPushButton:hover{background:#F0F6EC;}"
        )
        btn_close.clicked.connect(self.close)
        head_lay.addWidget(btn_close)

        page_root.addWidget(head)

        # Stacked pages
        self.stack = QStackedWidget(page_card)
        self.stack.setStyleSheet(
            "QStackedWidget{background:white;border:none;}"
        )
        self._pages: dict[str, QWidget] = {}
        self._nav_btns: dict[str, ChipButton] = {}
        for idx, (key, title) in enumerate(_SETTINGS_PAGES):
            page = self._make_page(key)
            self._pages[key] = page
            self.stack.addWidget(page)
        page_root.addWidget(self.stack, 1)
        root.addWidget(page_card, 1)

        # 默认第一页
        self._switch_page(_SETTINGS_PAGES[0][0])

    def _build_nav_bar(self, parent: QWidget) -> QWidget:
        nav = QFrame(parent)
        nav.setObjectName("NavBar")
        nav.setFixedWidth(190)
        nav.setStyleSheet(
            "QFrame#NavBar{background:white;border:1px solid #E2ECDE;border-radius:14px;}"
        )
        shadow = QGraphicsDropShadowEffect(nav)
        shadow.setBlurRadius(14)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        from PySide6.QtGui import QColor
        shadow.setColor(QColor(15, 23, 42, 18))
        nav.setGraphicsEffect(shadow)

        lay = QVBoxLayout(nav)
        lay.setContentsMargins(10, 14, 10, 14)
        lay.setSpacing(6)
        title = QLabel("📂 分类", nav)
        title.setStyleSheet("color:#6E7F6A;font-size:11px;font-weight:600;padding:4px 8px;")
        lay.addWidget(title)

        for key, label in _SETTINGS_PAGES:
            btn = ChipButton(label, active=False, parent=nav)
            btn.setFixedHeight(34)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self._switch_page(k))
            self._nav_btns[key] = btn
            lay.addWidget(btn)
        lay.addStretch(1)

        ver = QLabel("版本：v0.5.0 (M5)", nav)
        ver.setStyleSheet("color:#9AAD94;font-size:10px;padding:4px 8px;")
        lay.addWidget(ver)
        return nav

    def _switch_page(self, key: str) -> None:
        if key not in self._pages:
            return
        for k, w in self._pages.items():
            idx = self.stack.indexOf(w)
            if idx >= 0 and k == key:
                self.stack.setCurrentIndex(idx)
                # 更新标题
                title_map = {a: b for a, b in _SETTINGS_PAGES}
                self.page_title_lbl.setText(title_map.get(k, k))
        for k, btn in self._nav_btns.items():
            btn.setActive(k == key)

    # ------------------------------------------------------------------
    # 5 个页面表单（QFormLayout + Scroll）
    # ------------------------------------------------------------------

    def _wrap_scroll(self, page_widget: QWidget, form: QFormLayout) -> None:
        wrap = QWidget(page_widget)
        wrap.setLayout(form)
        scroll = QScrollArea(page_widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:white;}"
            "QScrollBar:vertical{width:6px;background:transparent;}"
            "QScrollBar::handle:vertical{background:#C9DCC5;border-radius:3px;}"
        )
        scroll.setWidget(wrap)
        page_lay = QVBoxLayout(page_widget)
        page_lay.setContentsMargins(22, 18, 22, 18)
        page_lay.setSpacing(0)
        page_lay.addWidget(scroll)

    def _form_row(self, form: QFormLayout, label: str, field: QWidget, hint: Optional[str] = None) -> None:
        lab = QLabel(label)
        lab.setStyleSheet("color:#24352A;font-size:13px;font-weight:500;padding:4px 0;")
        form.addRow(lab, field)
        if hint:
            form.addRow(None, _form_hint(hint))

    def _make_page(self, key: str) -> QWidget:
        page = QWidget(self.stack)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(18)
        form.setContentsMargins(2, 2, 2, 2)

        if key == "general":
            # 通用
            theme = QComboBox(page)
            theme.addItems(["Light 浅色（默认）", "Dark 深色（暂未实现）"])
            self.field_general_theme = theme
            self._form_row(form, "主题模式", theme, "深色模式计划 M6 迭代实现。")

            autostart = SwitchToggle(checked=False, parent=page)
            self.field_general_autostart = autostart
            self._form_row(form, "开机自启动", autostart, "Windows 注册表写入启动项，以管理员身份运行有效。")

            show_ball_on_start = SwitchToggle(checked=True, parent=page)
            self.field_general_ball = show_ball_on_start
            self._form_row(form, "启动后显示桌面悬浮球", show_ball_on_start, "关闭后可通过托盘菜单重新显示。")

            panel_on_start = SwitchToggle(checked=False, parent=page)
            self.field_general_panel = panel_on_start
            self._form_row(form, "启动后自动展开主面板", panel_on_start, "默认 False：仅显示悬浮球，点击再展开。")

            edge_snap = SwitchToggle(checked=True, parent=page)
            self.field_general_snap = edge_snap
            self._form_row(form, "悬浮球自动吸附屏幕左右边缘", edge_snap)

        elif key == "voice":
            asr_engine = QComboBox(page)
            asr_engine.addItems(["SenseVoice 离线（推荐）", "FunASR 离线（备选）", "纯文本模式（调试）"])
            self.field_voice_asr = asr_engine
            self._form_row(form, "ASR 语音识别引擎", asr_engine, "SenseVoice 支持中/英/粤/日，CPU 也能跑。")

            tts_engine = QComboBox(page)
            tts_engine.addItems(["Edge-TTS 在线（免费，当前）", "CosyVoice 本地（需额外装包）", "关闭 TTS（仅文字）"])
            self.field_voice_tts = tts_engine
            self._form_row(form, "TTS 语音合成引擎", tts_engine, "Edge-TTS 免费在线，CosyVoice 本地离线。")

            tts_voice = QComboBox(page)
            tts_voice.addItems(["晓晓女声（default）", "云希男声（yunxi）", "云健新闻（news）", "粤语晓佳（cantonese）", "英文 Aria（en）"])
            self.field_voice_tts_voice = tts_voice
            self._form_row(form, "TTS 默认音色", tts_voice)

            tts_speed = QDoubleSpinBox(page)
            tts_speed.setRange(0.5, 2.0)
            tts_speed.setSingleStep(0.05)
            tts_speed.setDecimals(2)
            tts_speed.setValue(1.0)
            self.field_voice_tts_speed = tts_speed
            self._form_row(form, "TTS 默认语速", tts_speed, "1.0 正常；0.8 慢速；1.2 快速。")

            auto_tts_on_answer = SwitchToggle(checked=True, parent=page)
            self.field_voice_auto_tts = auto_tts_on_answer
            self._form_row(form, "AI 回答后自动 TTS 朗读", auto_tts_on_answer, "可在主面板底部随时关闭。")

            save_asr_recording = SwitchToggle(checked=True, parent=page)
            self.field_voice_save_rec = save_asr_recording
            self._form_row(form, "ASR 自动保存录音", save_asr_recording, "保存到 data/asr_cache/，便于排查识别错误。")

        elif key == "hotkeys":
            hk_panel = QLineEdit("Ctrl + Alt + D", page)
            hk_panel.setReadOnly(True)
            self.field_hot_panel = hk_panel
            self._form_row(form, "展开/收起主面板", hk_panel, "自定义快捷键 LineEdit 占位，M6 接入 QHotkey。")

            hk_ptt = QLineEdit("Ctrl + Space（按住说话）", page)
            hk_ptt.setReadOnly(True)
            self.field_hot_ptt = hk_ptt
            self._form_row(form, "全局按住说话", hk_ptt, "默认 Ctrl+Space，可用于任意应用前台时触发。")

            hk_quit = QLineEdit("Ctrl + Alt + Q（退出程序）", page)
            hk_quit.setReadOnly(True)
            self._form_row(form, "退出程序", hk_quit)

            hint = _form_hint("💡 当前展示为占位值；自定义快捷键需要系统级全局热键库（pynput / keyboard / QHotkey），M6 接入。")
            form.addRow(None, hint)

        elif key == "ai":
            key_in = QLineEdit(page)
            # 从 .env 预览
            from src.utils.path_utils import ENV_FILE_PATH
            try:
                env_text = ENV_FILE_PATH.read_text(encoding="utf-8") if ENV_FILE_PATH.exists() else ""
                import re
                m = re.search(r'^QWEN_API_KEY\s*=\s*"?([^"\n]+)"?', env_text, re.M)
                if m and m.group(1).strip():
                    k = m.group(1).strip()
                    # 脱敏：前 6 + *** + 后 4
                    if len(k) > 10:
                        key_in.setText(k[:6] + "****" + k[-4:])
                    else:
                        key_in.setText("****")
            except Exception:
                pass
            key_in.setPlaceholderText("sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            key_in.setEchoMode(QLineEdit.Password)
            self.field_ai_key = key_in
            self._form_row(form, "QWEN_API_KEY（通义千问）", key_in, "保存时写 .env，重启后生效。支持 qwen-turbo / qwen-plus。")

            model = QComboBox(page)
            model.addItems(["qwen-turbo（性价比，推荐）", "qwen-plus（更强，免费额度易超）", "qwen-max（强模型）"])
            self.field_ai_model = model
            self._form_row(form, "默认模型", model)

            temperature = QDoubleSpinBox(page)
            temperature.setRange(0.0, 2.0)
            temperature.setSingleStep(0.05)
            temperature.setDecimals(2)
            temperature.setValue(0.7)
            self.field_ai_temp = temperature
            self._form_row(form, "采样温度 temperature", temperature, "0 更确定；更高更发散。")

            auto_tool_use = SwitchToggle(checked=True, parent=page)
            self.field_ai_auto_tool = auto_tool_use
            self._form_row(form, "Agent 自动调用工具", auto_tool_use, "关闭后 Agent 不会主动调用 delete_file 等 8 个 LangChain Tool。")

        elif key == "privacy":
            save_history = SwitchToggle(checked=True, parent=page)
            self.field_priv_history = save_history
            self._form_row(form, "保存对话历史到本地 SQLite", save_history, "使用 LangGraph CheckpointService，关闭则仅内存会话。")

            send_telemetry = SwitchToggle(checked=False, parent=page)
            self.field_priv_telemetry = send_telemetry
            self._form_row(form, "发送匿名使用统计（默认关闭）", send_telemetry, "未实现；预留开关保持默认关。")

            allow_clipboard = SwitchToggle(checked=True, parent=page)
            self.field_priv_clip = allow_clipboard
            self._form_row(form, "允许工具读取剪贴板（未来工具）", allow_clipboard, "例如粘贴板一键创建文件等功能。")

            clear_btn = QPushButton("🧹 清空本地对话历史")
            clear_btn.setCursor(Qt.PointingHandCursor)
            clear_btn.setStyleSheet(
                "QPushButton{background:#FEF2F2;color:#991B1B;border:1px solid #FECACA;"
                "border-radius:8px;padding:7px 12px;font-size:12px;}"
                "QPushButton:hover{background:#FEE2E2;}"
            )
            self.field_priv_clear = clear_btn
            self._form_row(form, "会话清理", clear_btn, "仅保留 SQLite 表结构，清空全部历史对话与 Checkpoint。")

        else:
            # unknown
            self._form_row(form, "页面", QLabel("未知页面占位", page))

        self._wrap_scroll(page, form)
        return page

    # ------------------------------------------------------------------
    # 保存 & 关闭
    # ------------------------------------------------------------------

    def _on_save_clicked(self) -> None:
        """保存（M5 暂只弹 toast 占位，后续接 settings_service）。"""
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("设置保存")
        msg.setText("💾 设置已记录（M5 预览版：占位保存）")
        msg.setInformativeText(
            "下一阶段 M6 将接真实 settings_service：写 .env + settings.json + Windows 注册表 + SQLite，\n"
            "届时本窗口各字段都会真正生效。"
        )
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()
        # 同时发一个信号，便于 main/application 连接
        self.settings_changed.emit("_saved_placeholder_", True)

    def closeEvent(self, ev) -> None:  # noqa: N802
        self.request_close.emit()
        super().closeEvent(ev)


__all__ = ["SettingsWindow"]
