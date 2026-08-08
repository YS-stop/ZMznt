"""桌面应用工具集（打开本机应用 + 桌面状态监控 + 屏幕视觉识别）。

工具：
    1. OpenAppTool          —— open_app：按名称打开本机桌面应用（应用目录模糊匹配 + os.startfile 启动）
    2. ListActiveAppsTool   —— list_active_apps：实时枚举当前桌面开着的窗口及进程（轻量屏幕监控）
    3. RecognizeScreenTool  —— recognize_screen：截图 + Qwen-VL 视觉问答（真·看懂屏幕）

约定：所有异常包装为中文 observation 返回，不向上抛。
"""
from __future__ import annotations

import base64
import os
import sys
import time
import uuid
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, Field

_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from langchain_core.tools import BaseTool  # noqa: E402


# ============================================================
# 1. OpenAppTool —— 打开本机桌面应用
# ============================================================

class OpenAppArgs(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        description=(
            "【必填】应用名称，支持模糊匹配：如「微信」「QQ」「WPS」「记事本」「Postman」。"
            "也可以传英文名如 WeChat（会按目录里的显示名匹配）。"
        ),
    )
    list_only: bool = Field(
        False,
        description="【选填】True=不启动应用，只列出应用目录（可按 name 过滤），用于「我电脑上装了哪些应用」。",
    )
    refresh: bool = Field(
        False,
        description="【选填】True=先重新扫描应用目录再操作（新装了软件但识别不到时用）。",
    )


class OpenAppTool(BaseTool):
    """打开本机桌面应用（微信、QQ、WPS、记事本等）。

    数据来自 AppCatalogService 扫描的「桌面 + 开始菜单」快捷方式目录（带缓存）。
    匹配不到时返回最接近的候选名，LLM 应如实转告用户，不要编造已打开。
    """

    name: ClassVar[str] = "open_app"
    description: ClassVar[str] = (
        "Tool Name: open_app\n"
        "用途：打开用户本机已安装的桌面应用（.exe / 快捷方式）。\n"
        "典型场景：\n"
        "  - 「打开微信」「启动 QQ」「把 WPS 打开」→ name=微信 / QQ / WPS\n"
        "  - 「打开记事本」「打开计算器」→ 内置系统应用也支持\n"
        "  - 「我电脑上装了哪些应用」→ list_only=True（可加 name 过滤）\n"
        "  - 新装的软件识别不到 → refresh=True 重新扫描一次\n"
        "说明：\n"
        "  - 应用目录来自桌面 + 开始菜单快捷方式扫描（覆盖绝大多数 GUI 应用）。\n"
        "  - 匹配是模糊的（「微信」能匹配「微信.lnk」）；命中多个时会自动选最精确的并把候选列出来。\n"
        "  - 找不到时返回候选列表，请如实告诉用户，不要假装已打开。\n"
        "  - 只负责启动，不操作应用内部功能。"
    )
    args_schema: ClassVar[type[BaseModel]] = OpenAppArgs
    return_direct: ClassVar[bool] = False

    def _run(self, name: str, list_only: bool = False, refresh: bool = False) -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            from src.services.app_catalog_service import get_app_catalog
            cat = get_app_catalog()
            total = cat.build(refresh=bool(refresh))

            q = (name or "").strip()
            # ---- 只列出目录 ----
            if list_only:
                names = cat.all_names()
                if q and q.lower() not in ("all", "全部", "所有"):
                    nq = q.lower()
                    names = [n for n in names if nq in n.lower()]
                ms = (time.perf_counter_ns() - t0) // 1_000_000
                listing = "、".join(names[:80]) + (" ……" if len(names) > 80 else "")
                return (
                    f"📋 应用目录（共 {total} 个，匹配到 {len(names)} 个，{ms} ms）：\n{listing or '（无匹配）'}"
                )

            # ---- 查找并启动 ----
            entry, candidates = cat.find_app(q)
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            if entry is None:
                sample = "、".join(cat.all_names()[:30])
                return (
                    f"⚠️ 应用目录（共 {total} 个）里没有找到匹配「{q}」的应用。\n"
                    f"  部分已收录应用：{sample} ……\n"
                    f"  建议：换个名字（如软件的全称/英文名）再试；或 refresh=True 重新扫描。"
                )
            target = entry.get("path") or entry.get("lnk")
            if not target:
                return f"⚠️ 找到应用「{entry['name']}」但无法定位可执行文件，请手动启动。"

            os.startfile(target)  # type: ignore[attr-defined]  # noqa: S606 - 启动本机应用是本工具的职责
            extra = ""
            if len(candidates) > 1:
                extra = f"\n  其他候选（如需打开的是它们，请说全名）：{'、'.join(candidates[1:6])}"
            return (
                f"✅ 已启动应用「{entry['name']}」（{ms} ms）\n"
                f"  目标：{target}\n"
                f"  来源：{entry.get('source', '?')}{extra}"
            )
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ open_app 失败（{ms} ms）：{type(e).__name__}: {e}"


# ============================================================
# 2. ListActiveAppsTool —— 实时枚举桌面窗口（轻量屏幕监控）
# ============================================================

class ListActiveAppsArgs(BaseModel):
    filter_keyword: str = Field(
        "",
        description="【选填】按窗口标题/进程名过滤（不区分大小写包含匹配），留空返回全部。",
    )


def _enum_windows_with_process() -> list[dict[str, str]]:
    """枚举可见顶层窗口：返回 [{'title':..., 'process':...}]。纯 ctypes。"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    results: list[dict[str, str]] = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def _proc_name(pid: int) -> str:
        try:
            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return ""
            try:
                buf = ctypes.create_unicode_buffer(260)
                size = wintypes.DWORD(260)
                if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    return Path(buf.value).name
            finally:
                kernel32.CloseHandle(h)
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _cb(hwnd: int, _lparam: int) -> bool:
        try:
            if not user32.IsWindowVisible(hwnd):
                return True
            n = user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            title = buf.value.strip()
            if not title:
                return True
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            results.append({"title": title, "process": _proc_name(pid.value)})
        except Exception:  # noqa: BLE001
            pass
        return True

    user32.EnumWindows(WNDENUMPROC(_cb), 0)
    return results


class ListActiveAppsTool(BaseTool):
    """实时查看当前桌面上开着哪些应用窗口（轻量屏幕监控，不截图）。"""

    name: ClassVar[str] = "list_active_apps"
    description: ClassVar[str] = (
        "Tool Name: list_active_apps\n"
        "用途：实时枚举当前桌面上所有可见窗口及其进程名——回答「我现在桌面上开着什么」「xxx 是不是开着」。\n"
        "典型场景：\n"
        "  - 「看看我桌面现在开了哪些应用」→ 直接调用\n"
        "  - 「微信开着吗」→ filter_keyword=微信\n"
        "说明：只读操作，不截图不改动任何窗口；返回进程名 + 窗口标题列表。\n"
        "需要「看懂屏幕内容」（如读某个窗口里的文字）时用 recognize_screen。"
    )
    args_schema: ClassVar[type[BaseModel]] = ListActiveAppsArgs
    return_direct: ClassVar[bool] = False

    def _run(self, filter_keyword: str = "") -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            wins = _enum_windows_with_process()
            kw = (filter_keyword or "").strip().lower()
            if kw:
                wins = [w for w in wins if kw in w["title"].lower() or kw in w["process"].lower()]
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            if not wins:
                return f"🔍 当前桌面没有匹配「{filter_keyword}」的可见窗口（{ms} ms）。"
            lines = [f"🖥️ 当前桌面可见窗口 {len(wins)} 个（{ms} ms）："]
            for w in wins[:40]:
                proc = w["process"] or "?"
                lines.append(f"  • [{proc}] {w['title'][:60]}")
            if len(wins) > 40:
                lines.append(f"  ……（还有 {len(wins) - 40} 个）")
            return "\n".join(lines)
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ list_active_apps 失败（{ms} ms）：{type(e).__name__}: {e}"


# ============================================================
# 3. RecognizeScreenTool —— 截图 + Qwen-VL 视觉问答（真·看懂屏幕）
# ============================================================

class RecognizeScreenArgs(BaseModel):
    question: str = Field(
        "描述一下屏幕上显示的内容，列出能看到的主要窗口和应用。",
        description=(
            "【选填】关于屏幕内容的问题，如「屏幕上开了哪些应用」「微信窗口里最后一条消息是什么」"
            "「当前网页的标题是什么」。留空则默认整体描述屏幕内容。"
        ),
    )


def _capture_screen_png(max_width: int = 1280) -> tuple[Optional[Path], str]:
    """截取主屏 PNG（Qt QScreen），缩放到 max_width 以内省 token。返回 (路径, 错误信息)。"""
    try:
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtCore import Qt

        app_ok = QGuiApplication.instance() is not None
        screens = QGuiApplication.screens() if app_ok else []
        if not screens:
            return None, "Qt GUI 未初始化，无法截图"
        pixmap = screens[0].grabWindow(0)
        if pixmap.isNull():
            return None, "grabWindow 返回空图像"
        if pixmap.width() > max_width:
            pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
        from src.utils.path_utils import DATA_ROOT, ensure_data_dirs
        ensure_data_dirs()
        out = DATA_ROOT / "ocr_temp" / f"screen_recog_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not pixmap.save(str(out), "PNG"):
            return None, "PNG 保存失败"
        return out, ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


class RecognizeScreenTool(BaseTool):
    """截图当前屏幕并用 Qwen-VL 视觉模型回答关于屏幕内容的问题。"""

    name: ClassVar[str] = "recognize_screen"
    description: ClassVar[str] = (
        "Tool Name: recognize_screen\n"
        "用途：截取当前屏幕并用视觉大模型「看懂」屏幕内容，回答相关问题。\n"
        "典型场景：\n"
        "  - 「看看我屏幕上现在显示什么」→ 直接调用\n"
        "  - 「帮我读一下当前窗口里的报错信息」→ question=读出当前窗口里的报错文字\n"
        "  - 「我桌面上哪个应用在放视频」→ question=哪个窗口在播放视频\n"
        "说明：\n"
        "  - 需要联网调用 Qwen-VL（qwen-vl-max），截图会临时保存到 data/ocr_temp。\n"
        "  - 只看当前主屏一帧画面；如果只要「有哪些窗口开着」不需要看内容，优先用 list_active_apps（更快不联网）。"
    )
    args_schema: ClassVar[type[BaseModel]] = RecognizeScreenArgs
    return_direct: ClassVar[bool] = False

    def _run(self, question: str = "") -> str:  # noqa: D401
        t0 = time.perf_counter_ns()
        try:
            img_path, err = _capture_screen_png()
            if img_path is None:
                return f"❌ recognize_screen 截图失败：{err}"

            import requests
            api_key = os.getenv("QWEN_API_KEY", "")
            base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            if not api_key:
                return "❌ recognize_screen：未配置 QWEN_API_KEY，无法调用视觉模型。"

            b64 = base64.b64encode(img_path.read_bytes()).decode()
            q = (question or "").strip() or "描述一下屏幕上显示的内容，列出能看到的主要窗口和应用。"
            r = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "qwen-vl-max",
                    "messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": q},
                    ]}],
                    "max_tokens": 800,
                },
                timeout=60,
            )
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            if r.status_code != 200:
                return f"❌ 视觉模型调用失败（HTTP {r.status_code}）：{r.text[:300]}"
            content = (
                r.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if not content:
                return f"⚠️ 视觉模型返回空内容（{ms} ms）。截图保存在：{img_path}"
            return (
                f"👁️ 屏幕识别结果（{ms} ms，截图 {img_path.name}）：\n{content}"
            )
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ recognize_screen 失败（{ms} ms）：{type(e).__name__}: {e}"


__all__ = [
    "OpenAppTool",
    "ListActiveAppsTool",
    "RecognizeScreenTool",
]
