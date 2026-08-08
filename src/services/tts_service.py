"""TTS 语音播报服务：双后端（Edge-TTS 在线 + CosyVoice 本地离线占位）。

引擎策略：
    1. Edge-TTS 在线（微软免费、无需 Key、中文多音色，edge-tts 包已在环境中安装）
       - voice 常用：zh-CN-XiaoxiaoNeural（晓晓女声，默认）/ zh-CN-YunxiNeural（云希男声）
                     / zh-CN-YunjianNeural（云健新闻）/ zh-HK-HiuGaaiNeural（粤语晓佳）/ en-US-AriaNeural
       - rate 格式：用户给 speed 浮点数（0.5~2.0）→ 换算成 "+50%" / "-20%" 字符串
       - 格式：默认 mp3
    2. CosyVoice 本地离线（阿里 FunAudioLLM，** M4 占位，等用户安装后启用**）：
       - 需额外装 cosyvoice + modelscope + torch 等，当前触发时直接降级为 Edge-TTS，并给出安装提示。

三级同步 API（Edge-TTS 是 async，内部用 asyncio.run 包一层，UI 可在 QThread 里调用避免卡主线程）：
    - synthesize_to_file(text, save_path, voice, speed) -> (ok: bool, path|error: str)：只合成不播放
    - play_audio(path, backend="auto") -> (ok: bool, note: str)：播放已有的 mp3/wav
    - speak(text, voice, speed, save_path, play_now) -> dict：合成 + 可选播放，返回结构化摘要

注意：
    播放依赖（playsound/pygame/simpleaudio/soundfile/sounddevice）如果都没装，play_audio 会返回「未播放」提示，
    但 synthesize_to_file 仍能成功保存 mp3（用户可手动打开），不阻塞整体流程。
"""
from __future__ import annotations

import asyncio
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.utils.path_utils import DATA_ROOT, ensure_data_dirs  # noqa: E402


_INSTALL_HINT_CN = (
    "TTS 已启用 Edge-TTS（免费在线）。如需增强功能可安装：\n"
    "  播放：pip install playsound pygame simpleaudio soundfile sounddevice pydub（任选 1-2 个）\n"
    "  本地离线：pip install cosyvoice modelscope funasr torchaudio（再配合 modelscope 下载 CosyVoice 权重）"
)


# ============================================================
# 内部：Edge-TTS 的同步包装
# ============================================================

def _normalize_voice(voice: Optional[str]) -> str:
    """voice 别名 → 实际 Edge-TTS ShortName。"""
    v = (voice or "default").strip().lower()
    aliases = {
        "default": "zh-CN-XiaoxiaoNeural",
        "xiaoxiao": "zh-CN-XiaoxiaoNeural",
        "female": "zh-CN-XiaoxiaoNeural",
        "girl": "zh-CN-XiaoxiaoNeural",
        "yunxi": "zh-CN-YunxiNeural",
        "male": "zh-CN-YunxiNeural",
        "boy": "zh-CN-YunxiNeural",
        "news": "zh-CN-YunjianNeural",
        "yunjian": "zh-CN-YunjianNeural",
        "cantonese": "zh-HK-HiuGaaiNeural",
        "yue": "zh-HK-HiuGaaiNeural",
        "hk": "zh-HK-HiuGaaiNeural",
        "english": "en-US-AriaNeural",
        "en": "en-US-AriaNeural",
        "aria": "en-US-AriaNeural",
    }
    if v in aliases:
        return aliases[v]
    # 如果用户直接传了 FullName（zh-CN-XxxNeural），原样返回
    if v.startswith("zh-") or v.startswith("en-") or v.startswith("ja-") or "-" in v:
        return voice  # type: ignore[return-value]
    return "zh-CN-XiaoxiaoNeural"


def _speed_to_rate(speed: float) -> str:
    """speed 0.5~2.0 → Edge-TTS rate 字符串，如 1.15 → '+15%'，0.8 → '-20%'。"""
    try:
        s = float(speed)
    except Exception:
        s = 1.0
    s = max(0.5, min(2.0, s))
    percent = int(round((s - 1.0) * 100))
    if percent >= 0:
        return f"+{percent}%"
    return f"{percent}%"


def _run_async(coro):
    """执行 asyncio 协程：优先 asyncio.run，已运行事件循环/Win 下 fallback 到显式 loop。"""
    try:
        # 优先标准 asyncio.run（3.7+ 正式 API）
        return asyncio.run(coro)
    except RuntimeError:
        pass
    # Fallback：某些 Windows 场景 asyncio.run 失败时，手动 loop 管理
    if sys.platform.startswith("win"):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        except RuntimeError:
            # nested/closed → 最后一招：新建独立 loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
    # 非 Win：最后一招新建 loop
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _edge_synthesize(text: str, out_path: Path, voice: str, speed: float) -> Tuple[bool, str]:
    """Edge-TTS 真实合成：返回 (是否成功, 错误信息或空串)。"""
    try:
        import edge_tts  # type: ignore
    except Exception as e:
        return False, f"edge-tts 包未安装或导入失败：{type(e).__name__}: {e}"

    async def _do() -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=_speed_to_rate(speed),
        )
        await communicate.save(str(out_path))

    try:
        _run_async(_do())
    except Exception as e:
        return False, f"Edge-TTS 合成失败（可能网络问题/音色不存在）：{type(e).__name__}: {e}"
    # 校验文件真的写出来了
    if not out_path.exists() or out_path.stat().st_size < 256:
        return False, f"Edge-TTS 合成输出文件异常：大小 {getattr(out_path.stat(), 'st_size', 0)}"
    return True, ""


# ============================================================
# 内部：播放（多后端降级）
# ============================================================

def _try_play_with_simple_solution(path: Path) -> Optional[Tuple[bool, str]]:
    """尝试 Windows/Mac/Linux 系统命令快速播放。成功返回 (True, note)，失败返回 None 让上层试下一个。"""
    sysname = platform.system()
    try:
        if sysname == "Windows":
            # Windows 自带：os.startfile 是官方最简单播放 mp3 的方式（shell 关联默认播放器）
            if hasattr(os, "startfile"):
                os.startfile(str(path))  # type: ignore[attr-defined]
                return True, "已调用系统默认播放器（Windows os.startfile）"
            # 兜底：start 命令
            subprocess.Popen(
                ["cmd", "/c", "start", "", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, "已调用 cmd /c start 打开默认播放器"
        if sysname == "Darwin":
            subprocess.Popen(
                ["afplay", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, "已调用 macOS afplay"
        # Linux
        for cmd in (["paplay", str(path)], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True, f"已调用 {cmd[0]}"
            except FileNotFoundError:
                continue
    except Exception:
        return None
    return None


def _try_play_sounddevice(path: Path) -> Optional[Tuple[bool, str]]:
    """进程内扬声器直播：soundfile 解码（含 mp3，需 libsndfile>=1.1）+ sounddevice 输出。

    不弹任何外部播放器窗口，阻塞到播完。依赖缺失返回 None 让上层降级。
    """
    try:
        import sounddevice as sd  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception:
        return None
    try:
        data, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
        sd.play(data, samplerate)
        sd.wait()  # 阻塞到播完，防止函数返回后音频被截断
        return True, "sounddevice 扬声器直播完成（未打开外部播放器）"
    except Exception as e:
        return False, f"sounddevice 播放失败：{type(e).__name__}: {e}"


def _try_play_playsound(path: Path) -> Optional[Tuple[bool, str]]:
    try:
        from playsound import playsound  # type: ignore
    except Exception:
        return None
    try:
        playsound(str(path))
        return True, "playsound 播放完成"
    except Exception as e:
        return False, f"playsound 失败：{type(e).__name__}: {e}"


def _try_play_pygame(path: Path) -> Optional[Tuple[bool, str]]:
    try:
        import pygame  # type: ignore
    except Exception:
        return None
    try:
        if not pygame.get_init():
            pygame.init()
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        # 阻塞到播完，防止服务退出音乐停（最长 600s 保护）
        t0 = time.time()
        while pygame.mixer.music.get_busy():
            if time.time() - t0 > 600:
                break
            time.sleep(0.1)
        return True, "pygame.mixer 播放完成"
    except Exception as e:
        return False, f"pygame 失败：{type(e).__name__}: {e}"


def stop_playback() -> bool:
    """立即停止进程内扬声器播放（sounddevice）。无播放时调用无害，返回是否成功调用。"""
    try:
        import sounddevice as sd  # type: ignore
        sd.stop()
        return True
    except Exception:  # noqa: BLE001
        return False


def play_audio(path: str | Path, backend: str = "auto") -> Tuple[bool, str]:
    """播放音频（mp3/wav）。返回 (是否启动/播放成功, 说明)。

    播放策略（backend="auto" 默认）：优先进程内扬声器直播（sounddevice → playsound → pygame），
    **不会**自动打开外部媒体播放器；确需系统默认播放器时显式传 backend="system"。
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return False, f"音频文件不存在：{p}"
    if p.stat().st_size == 0:
        return False, "音频文件为空"

    b = (backend or "auto").strip().lower()
    # 按优先级尝试：进程内直播优先，外部系统播放器仅在显式指定时使用
    attempts: list[Tuple[str, Any]] = []
    if b in ("auto", "sounddevice", "speaker"):
        attempts.append(("sounddevice", _try_play_sounddevice))
    if b in ("auto", "playsound"):
        attempts.append(("playsound", _try_play_playsound))
    if b in ("auto", "pygame"):
        attempts.append(("pygame", _try_play_pygame))
    if b == "system":
        attempts.append(("system", _try_play_with_simple_solution))
    notes: list[str] = []
    for name, fn in attempts:
        res = fn(p)
        if res is None:
            notes.append(f"{name}：未安装依赖")
            continue
        ok, note = res
        if ok:
            return True, note
        notes.append(f"{name}：{note}")
    return False, "所有进程内播放后端都未成功：" + "；".join(notes) + "。可安装：pip install sounddevice soundfile（或 playsound pygame）"


# ============================================================
# 服务类
# ============================================================

class TTSService:
    """TTS 服务：单例，Edge-TTS 合成 + 多播放后端降级。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: str = "edge-tts"  # edge-tts / cosyvoice(占位) / text-mock
        self._note: str = ""
        # 依赖探测
        self._probe()

    def _probe(self) -> None:
        try:
            import edge_tts  # noqa: F401
            self._engine = "edge-tts"
            self._note = "Edge-TTS 在线引擎就绪（微软免费、中文多音色，需外网）"
        except Exception as e:
            self._engine = "text-mock"
            self._note = f"edge-tts 包不可用：{type(e).__name__}，已降级为文字播报模式。{_INSTALL_HINT_CN}"

    def get_info(self) -> dict[str, str]:
        return {
            "engine": self._engine,
            "note": self._note,
            "install_hint": _INSTALL_HINT_CN if self._engine != "edge-tts" else "",
        }

    # -------------------- 核心 API --------------------

    def synthesize_to_file(
        self,
        text: str,
        save_path: Optional[str | Path] = None,
        voice: str = "default",
        speed: float = 1.0,
        max_len: int = 5000,
    ) -> Tuple[bool, Path | str]:
        """把文本合成 mp3 文件。
        返回：(成功?, 成功=保存路径 Path，失败=错误字符串)。
        """
        content = (text or "").strip()
        if not content:
            return False, "text 不能为空"
        if len(content) > max_len:
            content = content[: max_len - 1] + "…"
        # 路径
        ensure_data_dirs()
        out_path: Path
        if save_path is None or str(save_path).strip() == "":
            ts = int(time.time() * 1000)
            out_dir = DATA_ROOT / "tts_cache"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"tts_{ts}.mp3"
        else:
            out_path = Path(str(save_path))
            if not out_path.is_absolute():
                out_path = (DATA_ROOT / "tts_cache" / out_path).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)

        # 引擎分派
        if self._engine in ("edge-tts", "cosyvoice"):
            # CosyVoice 占位：当前统一调 Edge-TTS，后面装了 CosyVoice 再切
            v = _normalize_voice(voice)
            ok, err = _edge_synthesize(content, out_path, v, speed)
            if ok:
                return True, out_path
            # Edge-TTS 失败（网络断了）：降级 text-mock
            return False, (err or "Edge-TTS 合成失败")
        return False, f"TTS 引擎不可用（engine={self._engine}）：{self._note}"

    def speak(
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        save_path: Optional[str | Path] = None,
        play_now: bool = True,
    ) -> dict[str, Any]:
        """合成 + 可选播放，返回结构化 dict（调试面板/工具返回直接可用）。"""
        result: dict[str, Any] = {
            "ok": False,
            "engine": self._engine,
            "text_len": len((text or "").strip()),
            "voice": voice,
            "speed": speed,
            "audio_path": None,
            "audio_size_bytes": 0,
            "play_ok": False,
            "play_note": "",
            "error": "",
        }
        # 1. 合成
        ok, ret = self.synthesize_to_file(text, save_path=save_path, voice=voice, speed=speed)
        if not ok:
            result["error"] = str(ret)
            return result
        result["ok"] = True
        result["audio_path"] = str(ret)
        try:
            result["audio_size_bytes"] = Path(ret).stat().st_size
        except Exception:
            pass
        # 2. 可选播放
        if play_now and isinstance(ret, Path):
            play_ok, play_note = play_audio(ret, backend="auto")
            result["play_ok"] = bool(play_ok)
            result["play_note"] = play_note
        else:
            result["play_note"] = "play_now=False，未启动播放"
        return result


# ============================================================
# 模块级单例
# ============================================================

_TTS_SVC = TTSService()


def get_tts_service() -> TTSService:
    return _TTS_SVC


def get_tts_info() -> dict[str, str]:
    return _TTS_SVC.get_info()


def tts_synthesize_to_file(
    text: str,
    save_path: Optional[str | Path] = None,
    voice: str = "default",
    speed: float = 1.0,
) -> Tuple[bool, Path | str]:
    return _TTS_SVC.synthesize_to_file(text, save_path=save_path, voice=voice, speed=speed)


def tts_play_audio(path: str | Path, backend: str = "auto") -> Tuple[bool, str]:
    return play_audio(path, backend=backend)


def tts_stop_playback() -> bool:
    """停止当前播报（任务被打断时掐断上一任务的语音）。"""
    return stop_playback()


def tts_speak(
    text: str,
    voice: str = "default",
    speed: float = 1.0,
    save_path: Optional[str | Path] = None,
    play_now: bool = True,
) -> dict[str, Any]:
    return _TTS_SVC.speak(text, voice=voice, speed=speed, save_path=save_path, play_now=play_now)


__all__ = [
    "TTSService",
    "play_audio",
    "stop_playback",
    "get_tts_service",
    "get_tts_info",
    "tts_synthesize_to_file",
    "tts_play_audio",
    "tts_stop_playback",
    "tts_speak",
]
