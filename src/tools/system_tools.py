"""M6.2 系统级工具集（共 4 个）：

    1. VolumeControlTool     —— Windows 音量 / 静音控制（pycaw）
    2. ScreenShotTool        —— 整屏截图（Qt QScreen，保存 PNG）
    3. SystemPowerTool       —— 锁屏 / 待机 / 重启 / 关机（ctypes + shutdown.exe）
    4. TranslateTextTool     —— 文本翻译（直接走 Qwen Chat LLM：让 LLM 当翻译引擎）

遵循 LangChain BaseTool 规范，与 M1~M1.5 其他工具同风格；
注册到 src/tools/__init__.py TOOL_MAP 即可被 LangGraph agent_node 自动识别调用。
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from ctypes import windll
from pathlib import Path
from typing import ClassVar, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 共享：路径工具（放顶部避免循环依赖）
# ---------------------------------------------------------------------------
def _data_root() -> Path:
    from src.utils.path_utils import DATA_ROOT, ensure_data_dirs
    ensure_data_dirs()
    return DATA_ROOT


# ==============================================================================
# 1. VolumeControlTool —— Windows 系统音量控制（pycaw）
# ==============================================================================

class VolumeControlArgs(BaseModel):
    mode: str = Field(
        "get",
        description=(
            "操作模式：\n"
            "- 'get'   : 查询当前主音量与静音状态（默认）\n"
            "- 'set'   : 设置主音量，需传 volume=0.0 ~ 1.0\n"
            "- 'mute'  : 静音\n"
            "- 'unmute': 取消静音\n"
            "- 'toggle': 切换静音\n"
        ),
    )
    volume: Optional[float] = Field(
        None,
        description="mode='set' 时必填，0.0=静音 1.0=最大；其他模式可忽略",
    )


class VolumeControlTool(BaseTool):
    """Tool Name: system_volume
    用途：查询 / 设置 Windows 主音量、静音、取消静音、切换静音。
    典型场景：
      - 用户说：「把电脑声音调小一点 / 音量开到 80%」→ mode='set' volume=0.8
      - 用户说：「静音」→ mode='mute'；「打开声音」→ mode='unmute'
    安全边界：
      - 只操作默认音频渲染设备（扬声器/耳机），不碰麦克风
      - volume 自动截断到 [0,1]，越界不会报错
    """
    name: ClassVar[str] = "system_volume"
    description: ClassVar[str] = (
        "Tool Name: system_volume\n"
        "用途：Windows 主音量 / 静音控制，支持查询、设置、静音、取消静音、切换静音。\n"
        "参数：VolumeControlArgs(mode='get|set|mute|unmute|toggle', volume=0.0~1.0)\n"
        "返回：当前音量百分比 + 静音状态的人类可读文本报告。\n"
        "降级：若 pycaw 未安装则返回友好提示（不会崩）。"
    )
    args_schema: ClassVar[type[BaseModel]] = VolumeControlArgs

    def _run(self, mode: str = "get", volume: Optional[float] = None) -> str:
        # ---- 连接扬声器音量接口（兼容新/老两个 pycaw 版本）----
        vol = None
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # noqa: WPS433
            # 新版 pycaw（>=2023）：AudioDevice 对象直接挂 EndpointVolume 属性
            speakers = AudioUtilities.GetSpeakers()
            if hasattr(speakers, "EndpointVolume") and speakers.EndpointVolume is not None:
                vol = speakers.EndpointVolume
            else:
                # 老版 pycaw：走 Activate + cast 流程
                from ctypes import cast, POINTER  # noqa: WPS433
                from comtypes import CLSCTX_ALL  # noqa: WPS433
                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                vol = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as e:  # noqa: BLE001
            return (
                "⚠️ system_volume 降级：pycaw/comtypes 不可用或默认音频设备缺失。\n"
                f"   安装命令：pip install pycaw comtypes  （错误：{type(e).__name__}: {e}）"
            )

        mode_norm = (mode or "get").lower().strip()
        if mode_norm == "set":
            if volume is None:
                return "❌ system_volume mode='set' 必须同时传 volume=0.0~1.0。"
            v = max(0.0, min(1.0, float(volume)))
            vol.SetMasterVolumeLevelScalar(v, None)
        elif mode_norm == "mute":
            vol.SetMute(1, None)
        elif mode_norm == "unmute":
            vol.SetMute(0, None)
        elif mode_norm == "toggle":
            cur_mute = vol.GetMute()
            vol.SetMute(0 if cur_mute else 1, None)
        elif mode_norm != "get":
            return f"❌ system_volume 未知 mode='{mode}'，可选：get | set | mute | unmute | toggle。"

        # 查询当前状态（所有模式最后都报告一下）
        cur_vol = float(vol.GetMasterVolumeLevelScalar())
        cur_mute = bool(vol.GetMute())
        return (
            f"🔊 音量状态\n"
            f"  - 主音量：{round(cur_vol * 100, 1)}%\n"
            f"  - 静音  ：{'是 (MUTE)' if cur_mute else '否'}\n"
            f"  - 操作  ：{mode_norm}"
        )


# ==============================================================================
# 2. ScreenShotTool —— 整屏截图（Qt QScreen，避免依赖 mss/pyautogui）
# ==============================================================================

class ScreenShotArgs(BaseModel):
    save_path: Optional[str] = Field(
        None,
        description="保存路径，可留空（默认放到 <DATA_DIR>/ocr_temp/screenshot_<uuid>.png）",
    )
    display_index: int = Field(
        0,
        description="多显示器时选择第几个屏幕（0=主屏，1=副屏...），默认 0",
    )


class ScreenShotTool(BaseTool):
    """Tool Name: system_screenshot
    用途：截取整屏 PNG 并保存到本地，返回保存路径。
    典型场景：用户说「把当前桌面截个图给我」或配合后续 OCR 工具（M6.3）。
    """
    name: ClassVar[str] = "system_screenshot"
    description: ClassVar[str] = (
        "Tool Name: system_screenshot\n"
        "用途：桌面整屏截图并保存 PNG，返回保存文件路径。\n"
        "参数：ScreenShotArgs(save_path=None 自动存到 data/ocr_temp/, display_index=0 主屏)\n"
        "返回：保存成功的绝对路径（UTF-8 字符串）。\n"
        "降级：若 Qt GUI 未初始化则用 mss 再兜底，最后失败返回友好提示。"
    )
    args_schema: ClassVar[type[BaseModel]] = ScreenShotArgs

    def _run(self, save_path: Optional[str] = None, display_index: int = 0) -> str:
        # ---- 构造保存路径 ----
        out: Path
        if save_path:
            out = Path(save_path).expanduser().resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            root = _data_root()
            d = root / "ocr_temp"
            d.mkdir(parents=True, exist_ok=True)
            out = d / f"screenshot_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"

        # ---- 方案 A：QScreen.grabWindow (Qt 已在桌面 UI 中必然初始化) ----
        try:
            from PySide6.QtGui import QGuiApplication

            app_ok = QGuiApplication.instance() is not None
            screens = QGuiApplication.screens() if app_ok else []
            if screens:
                idx = max(0, min(display_index, len(screens) - 1))
                screen = screens[idx]
                # 0 表示捕获整个屏幕（虚屏窗口句柄 0 在 Qt 语义里是抓整个屏幕）
                pixmap = screen.grabWindow(0)
                if not pixmap.isNull():
                    ok = pixmap.save(str(out), "PNG")
                    if ok and out.exists():
                        return (
                            f"✅ 截图成功\n"
                            f"  - 屏幕  ：#{idx} {screen.name()} ({screen.geometry().width()}x{screen.geometry().height()})\n"
                            f"  - 尺寸  ：{pixmap.width()}x{pixmap.height()}\n"
                            f"  - 保存到：{out}"
                        )
        except Exception:  # noqa: BLE001
            pass

        # ---- 方案 B：mss 兜底（无 GUI 场景） ----
        try:
            import mss  # type: ignore
            with mss.mss() as sct:
                monitors = sct.monitors[1:]  # monitors[0] 是全部屏幕合并
                idx = max(0, min(display_index, max(0, len(monitors) - 1)))
                mon = monitors[idx] if monitors else sct.monitors[0]
                shot = sct.grab(mon)
                try:
                    from PIL import Image  # type: ignore
                    img = Image.frombytes("RGB", shot.size, shot.rgb)
                    img.save(str(out), "PNG")
                except Exception:
                    # 无 PIL 就写 mss 原始 PNG（mss 自带输出 mss.tools.to_png）
                    import mss.tools  # type: ignore
                    mss.tools.to_png(shot.rgb, shot.size, output=str(out))
                if out.exists():
                    return (
                        f"✅ 截图成功（mss 兜底）\n"
                        f"  - 屏幕  ：#{idx} {mon}\n"
                        f"  - 保存到：{out}"
                    )
        except Exception as e:  # noqa: BLE001
            return (
                f"❌ system_screenshot 截图失败：{type(e).__name__}: {e}\n"
                "   建议：安装 mss + Pillow  （pip install mss pillow）"
            )
        return "❌ system_screenshot 截图失败（未知原因）。"


# ==============================================================================
# 3. SystemPowerTool —— 锁屏 / 待机 / 重启 / 关机
# ==============================================================================

class SystemPowerArgs(BaseModel):
    action: str = Field(
        "lock",
        description=(
            "动作：\n"
            "- 'lock'       : 锁定桌面（默认，最安全，无破坏性）\n"
            "- 'sleep'      : 待机睡眠（S3）\n"
            "- 'restart'    : 重启电脑（30 秒倒计时，可调 abort 取消）\n"
            "- 'shutdown'   : 关机（30 秒倒计时）\n"
            "- 'abort'      : 取消正在倒计时的关机 / 重启\n"
        ),
    )
    force: bool = Field(
        False,
        description="True=强制（不提示用户保存）；默认 False=安全模式。⚠️ 仅 shutdown/restart 生效。",
    )


class SystemPowerTool(BaseTool):
    """Tool Name: system_power
    用途：锁屏（默认）、睡眠、重启、关机、取消倒计时关机。
    安全策略：
      - 默认只执行 lock（无破坏性）
      - restart/shutdown 必须显式传 action='restart' 或 'shutdown'
      - restart/shutdown 走 shutdown.exe 带 30 秒倒计时，给用户机会取消
    """
    name: ClassVar[str] = "system_power"
    description: ClassVar[str] = (
        "Tool Name: system_power\n"
        "用途：锁屏 / 睡眠 / 重启 / 关机 / 取消倒计时关机\n"
        "参数：SystemPowerArgs(action='lock|sleep|restart|shutdown|abort', force=False)\n"
        "安全：重启/关机默认 30 秒倒计时，可随时 abort 取消。"
    )
    args_schema: ClassVar[type[BaseModel]] = SystemPowerArgs

    def _run(self, action: str = "lock", force: bool = False) -> str:
        a = (action or "lock").lower().strip()
        if sys.platform != "win32":
            return f"❌ system_power 当前仅支持 Windows（当前平台: {sys.platform}）。"

        # --- 安全操作集合 1：只改会话，不伤数据 ---
        if a == "lock":
            try:
                windll.user32.LockWorkStation()
                return "🔒 Windows 已锁定桌面（欢迎回来后输入密码登录）。"
            except Exception as e:
                return f"❌ 锁屏失败：{type(e).__name__}: {e}"

        if a == "sleep":
            try:
                # rundll32.exe powrprof.dll,SetSuspendState 0,1,0 → S3 睡眠（软关机=false，唤醒=true，紧急=false）
                import subprocess
                subprocess.run(
                    ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                    check=False,
                )
                return "💤 Windows 已进入睡眠（按任意键 / 点鼠标唤醒）。"
            except Exception as e:
                return f"❌ 睡眠失败：{type(e).__name__}: {e}"

        # --- 安全操作集合 2：取消倒计时 ---
        if a == "abort":
            try:
                import subprocess
                r = subprocess.run(["shutdown.exe", "/a"], capture_output=True, text=True, check=False)
                return f"🛑 已调用 shutdown /a 取消关机倒计时：\n{(r.stdout or r.stderr).strip()}"
            except Exception as e:
                return f"❌ 取消关机失败：{type(e).__name__}: {e}"

        # --- 高危操作：restart / shutdown，30 秒倒计时 ---
        if a not in ("restart", "shutdown"):
            return f"❌ system_power 未知 action='{a}'，可选 lock|sleep|restart|shutdown|abort。"

        switch = "/r" if a == "restart" else "/s"
        force_flag = "/f" if force else "/t 30"
        try:
            import subprocess
            cmd = ["shutdown.exe", switch, "/t", "30"] if not force else ["shutdown.exe", switch, "/f"]
            cmd_str = " ".join(cmd)
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return (
                f"⚠️  已启动 {'重启' if a == 'restart' else '关机'}（{'强制，不保存用户文档' if force else '30 秒后执行'}）\n"
                f"  命令：{cmd_str}\n"
                f"  输出：{(r.stdout or r.stderr).strip()}\n"
                f"  取消：system_power(action='abort') 或手动运行  shutdown /a"
            )
        except Exception as e:
            return f"❌ {'重启' if a == 'restart' else '关机'}失败：{type(e).__name__}: {e}"


# ==============================================================================
# 4. TranslateTextTool —— 文本翻译（Qwen LLM 翻译：零额外模型依赖）
# ==============================================================================

class TranslateTextArgs(BaseModel):
    text: str = Field(..., description="要翻译的原文（非空字符串）", min_length=1)
    from_lang: str = Field(
        "auto",
        description="源语言：默认 'auto' 自动识别；常见值：'zh'中文, 'en'英语, 'ja'日语, 'ko'韩语, 'de'德语, 'fr'法语",
    )
    to_lang: str = Field(
        "en",
        description="目标语言：默认 'en' 翻译成英语；如需翻成中文填 'zh'",
    )
    extra_requirements: Optional[str] = Field(
        None,
        description="额外要求，例如：'保持 Markdown 格式' / '正式公文语气' / '编程变量名用驼峰' 等，可留空",
    )


class TranslateTextTool(BaseTool):
    """Tool Name: system_translate
    用途：把任意语言文本翻译成目标语言。
    实现：直接构造翻译 prompt 走 Qwen LLM（get_qwen_llm），无需额外翻译 API。
    降级：当 QWEN_API_KEY 未配置时自动返回占位提示。
    """
    name: ClassVar[str] = "system_translate"
    description: ClassVar[str] = (
        "Tool Name: system_translate\n"
        "用途：通用多语言翻译（中文↔英文、中日韩德法等任意组合）。\n"
        "参数：TranslateTextArgs(text, from_lang='auto', to_lang='en', extra_requirements=None)\n"
        "返回：翻译后的纯文本 + 语言元信息（字数 / 源语言识别结果 / 耗时）。\n"
        "实现：构造翻译专用 prompt 调用 Qwen Chat LLM；无 Key 时给出友好占位。"
    )
    args_schema: ClassVar[type[BaseModel]] = TranslateTextArgs

    def _run(
        self,
        text: str,
        from_lang: str = "auto",
        to_lang: str = "en",
        extra_requirements: Optional[str] = None,
    ) -> str:
        original = (text or "").strip()
        if not original:
            return "❌ system_translate text 不能为空。"

        from_lang_norm = (from_lang or "auto").lower().strip()
        to_lang_norm = (to_lang or "zh").lower().strip() or "en"

        # 常见语言代码 → 人类可读（供 System Prompt 用，LLM 更好理解）
        _LANG_NAME = {
            "zh": "中文（简体）", "cn": "中文（简体）", "zh-cn": "中文（简体）",
            "en": "英语（美式）", "en-us": "英语（美式）", "en-gb": "英语（英式）",
            "ja": "日语", "jp": "日语", "ko": "韩语",
            "de": "德语", "fr": "法语", "es": "西班牙语", "ru": "俄语",
            "auto": "自动识别",
        }
        src_name = _LANG_NAME.get(from_lang_norm, from_lang_norm)
        tgt_name = _LANG_NAME.get(to_lang_norm, to_lang_norm)

        # ---- 构造翻译 System Prompt ----
        sys_prompt = (
            f"你是一个专业翻译专家。任务：把用户提供的原文"
            f"从【{src_name}】翻译成【{tgt_name}】。\n"
            f"要求：\n"
            f"1. 先输出一行「源语言识别: <识别结果>」（当 from_lang=auto 时需自行判断）\n"
            f"2. 再输出一行「字数: {len(original)} 字符 / 词数估计」\n"
            f"3. 然后输出分隔线「---」\n"
            f"4. 最后输出纯译文，不要任何额外解释、注释、括号注音；保留原文 Markdown / 换行 / 格式。\n"
            f"{('5. 额外要求: ' + extra_requirements) if extra_requirements else ''}"
        )

        # ---- 调 Qwen LLM ----
        try:
            from src.infra.llm_client import get_qwen_llm
            from langchain_core.messages import SystemMessage, HumanMessage
            llm = get_qwen_llm(temperature=0.1)
            t0 = time.time()
            out = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=original)])
            dt_ms = int((time.time() - t0) * 1000)
            translated = getattr(out, "content", None) or (out if isinstance(out, str) else "")
            rendered = str(translated).strip()
            if "[QWEN_API_KEY 未配置]" in rendered:
                return (
                    "⚠️ system_translate 降级：QWEN_API_KEY 未配置，无法真实翻译。\n"
                    f"   待翻译原文（{len(original)} 字，{from_lang_norm} → {to_lang_norm}）：\n"
                    f"   {original[:500]}{'…' if len(original) > 500 else ''}"
                )
            return (
                f"🌐 翻译结果  {src_name} → {tgt_name}  （LLM 耗时 {dt_ms} ms）\n"
                f"{rendered}"
            )
        except Exception as e:  # noqa: BLE001
            return (
                f"❌ system_translate 调用 LLM 失败：{type(e).__name__}: {e}\n"
                f"   原文 {len(original)} 字 {from_lang_norm}→{to_lang_norm}：\n"
                f"   {original[:500]}"
            )


# ==============================================================================
# 工具合集：给 tools/__init__.py 的 TOOL_MAP 提供 get_all_system_tools()
# ==============================================================================

def get_all_system_tools() -> list[BaseTool]:
    """M6 新增 4 个系统工具集合（按类实例化好，直接 append 到全局工具清单里）。"""
    return [
        VolumeControlTool(),
        ScreenShotTool(),
        SystemPowerTool(),
        TranslateTextTool(),
    ]
