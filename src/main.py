"""桌面语音助手启动入口 —— 永远保持精简，业务逻辑全部下沉到子包。

阶段回顾：
    M0 骨架：PySide6 启动 + 自动建目录 + 自动建 SQLite + 弹骨架提示窗口
    M1~M4：AgentState + 8 工具 + LangGraph + Checkpoint + ASR/TTS（全链路后台能力）
    M5：桌面 UI 全面装配 —— 悬浮球 + 抽屉 + 设置 + 历史 + 调试面板 + 托盘
            （通过 --m0-only 参数可回退到 M0 骨架窗口）
"""
from __future__ import annotations
import sys
from pathlib import Path

# ★ 必须先加项目根到 sys.path，才能 import src.*（打包后也生效）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv


def _preload_asr_native_deps() -> None:
    """⚠️ 必须在 import PySide6【之前】预加载 ASR 原生依赖（torch/funasr 等）。

    实测（本机可复现）：PySide6 先加载后 import torch →
    `OSError WinError 1114 DLL 初始化例程失败 (c10.dll)`，在完整 App 中表现为
    进程段错误无声消失（启动几十秒后自动终止）。torch 先 import 则两者共存正常。
    依赖未安装时静默跳过（ASR 自动降级纯文本模式）。
    """
    try:
        import torch  # noqa: F401
        import funasr  # noqa: F401
        import modelscope  # noqa: F401
        import sounddevice  # noqa: F401
        import soundfile  # noqa: F401
        print("[startup] ASR 原生依赖预加载完成（torch 先于 Qt 加载）")
    except Exception as _e:  # noqa: BLE001
        print(f"[startup] ASR 依赖未就绪，语音功能将降级：{type(_e).__name__}: {_e}")


_preload_asr_native_deps()

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.utils.path_utils import ensure_data_dirs, PROJECT_ROOT, DATA_ROOT, APP_DB_PATH
from src.services.sqlite_db import init_sqlite, get_setting


def _load_env() -> None:
    """加载 .env；打包后在 exe 同级目录找 .env，开发态在 PROJECT_ROOT 下找。"""
    env_candidates = [
        PROJECT_ROOT / ".env",
        Path(sys.executable).resolve().parent / ".env",
        Path.cwd() / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return


def _build_m0_skeleton_window() -> QWidget:
    """M0 骨架窗口：只显示启动信息，不做任何业务逻辑。回退用 --m0-only。"""
    win = QWidget()
    win.setWindowTitle("桌面语音助手 — M0 脚手架启动成功")
    win.resize(520, 320)
    win.setWindowFlag(Qt.WindowStaysOnTopHint, True)

    layout = QVBoxLayout(win)
    layout.setContentsMargins(28, 28, 28, 28)
    layout.setSpacing(14)

    title = QLabel("✅ M0 环境与脚手架启动成功！（M5 已启用完整 UI，可去掉 --m0-only 体验）")
    f = QFont()
    f.setPointSize(14)
    f.setBold(True)
    title.setFont(f)
    title.setStyleSheet("color:#1D4ED8;")
    title.setWordWrap(True)
    layout.addWidget(title)

    info_lines = [
        f"📁 项目根目录：{PROJECT_ROOT}",
        f"💾 数据根目录：{DATA_ROOT}",
        f"🗄  SQLite 库路径：{APP_DB_PATH}",
        "",
        "🎯 M5 已启用：默认启动完整桌面 UI（悬浮球 + 抽屉 + 托盘）。",
        "   如需回退：运行时追加 --m0-only 参数即可只显示此骨架。",
    ]
    info_label = QLabel("\n".join(info_lines))
    info_label.setWordWrap(True)
    info_label.setStyleSheet("font-family:Consolas,Microsoft YaHei UI;font-size:12px;color:#0F172A;")
    layout.addWidget(info_label)

    spacer = QWidget()
    spacer.setSizePolicy(
        spacer.sizePolicy().horizontalPolicy(), spacer.sizePolicy().verticalPolicy().Expanding
    )
    layout.addWidget(spacer, 1)

    btn_row = QHBoxLayout()
    btn_row.addStretch(1)

    close_btn = QPushButton("关闭窗口")
    close_btn.setStyleSheet(
        "QPushButton{background:#3B82F6;color:#FFFFFF;padding:8px 22px;border-radius:8px;font-weight:bold;}"
        "QPushButton:hover{background:#1D4ED8;}"
    )
    close_btn.clicked.connect(win.close)
    btn_row.addWidget(close_btn)
    layout.addLayout(btn_row)

    return win


def _run_m5_full_ui(app: QApplication) -> int:
    """装配 M5 完整桌面 UI：AppController + UIBridgeService，阻塞到 exec。"""
    from src.ui.application import AppController
    from src.services.ui_bridge_service import get_ui_bridge

    # 1) 组装应用主控（悬浮球 + 抽屉 + 设置 + 历史 + 调试面板 + 托盘）
    ctrl = AppController(app=app)

    # 2) 桥接层：AppController 信号 → 业务服务 → 信号回到 UI 渲染
    bridge = get_ui_bridge()
    bridge.bind_app(ctrl)

    # 3) 欢迎气泡（演示：调试面板首条彩色日志 + 对话 Tab 欢迎语）
    ctrl.push_debug("SYSTEM", "✅ AppController 已装配：悬浮球 / 抽屉 / 历史 / 调试面板 / 托盘 全部就绪。")
    ctrl.push_debug("SYSTEM", "💡 试试：点击桌面右下角蓝色悬浮球 → 展开主面板 → 打字或按住 🎤 说话。")
    ctrl.push_debug("SYSTEM", "💡 设置：悬浮球右键 / 主面板⚙️ / 托盘右键菜单 均可进入设置中心。")

    # 4) run：阻塞到退出
    return ctrl.run()


def main() -> int:
    # Step 1: 加载 .env
    _load_env()

    # Step 2: 建所有运行期目录（幂等）
    ensure_data_dirs()

    # Step 3: 建 SQLite 三张表 + 默认 settings
    init_sqlite()

    # （ASR 原生依赖已在文件顶部、import PySide6 之前预加载，见 _preload_asr_native_deps）

    # Step 4: 启动 Qt App
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("DesktopVoiceAssistant")
    app.setOrganizationName("zhuomZNT")
    app.setQuitOnLastWindowClosed(False)

    try:
        wake = get_setting("wake_word", "小助手")
        assert wake is not None
    except Exception as exc:
        err = QLabel(f"❌ SQLite 初始化失败：{exc}")
        err.setStyleSheet("color:#991B1B;padding:20px;font-size:14px;")
        err.setWindowTitle("启动失败")
        err.show()
        err.resize(640, 200)
        return app.exec()

    # Step 5: 命令行参数决定启动模式
    if "--m0-only" in sys.argv:
        window = _build_m0_skeleton_window()
        window.show()
        return app.exec()

    # 默认模式：M5 完整 UI
    return _run_m5_full_ui(app)


if __name__ == "__main__":
    sys.exit(main())
