"""ASR 语音识别服务：SenseVoice 方案（阿里 FunASR + ModelScope）。

三级降级策略（任一依赖缺失都会自动跳到下一级，不影响启动）：
    Level 1: SenseVoiceSmall via FunASR + ModelScope  —— 本地离线高精度（中文/英文/粤语/日语），需 GPU/CPU 均可
    Level 2: 缺 FunASR / ModelScope / torch 任意一个 → 降级为「纯文本输入模式」，返回安装提示
        （语音 UI 按钮此时会变为「请打字」的文本输入框占位）

对外 API（同步）：
    - get_asr_info() -> dict：后端信息 + 安装提示（调试面板展示）
    - transcribe_file(audio_path, lang='auto') -> str：识别一段已保存的 wav/mp3/flac
    - record_and_transcribe(duration=5, samplerate=16000, device=None, save_path=None, force_text=None) -> str：
        录音 duration 秒后转写；force_text 不为空则直接返回该文本（测试/无麦克风场景旁路）

注意：
    1. 模型首次加载会从 ModelScope 自动下载「iic/SenseVoiceSmall」，约 200~500MB，建议在稳定网络下首次调用
    2. 本服务是同步接口，UI 调用时应放到 QThread 中（Qt 主线程阻塞会无响应）
    3. 依赖安装命令（打包给用户看）：
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install modelscope funasr soundfile sounddevice librosa
    4. Windows 上 sounddevice 需要 PortAudio，pip 安装 sounddevice 会自带；如仍缺可 pip install pyaudio 当备选
"""
from __future__ import annotations

import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Optional

_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.utils.path_utils import DATA_ROOT, ensure_data_dirs  # noqa: E402


_INSTALL_HINT_CN = (
    "如需启用离线语音识别，请在虚拟环境中执行以下命令（CPU 版 PyTorch + SenseVoice + FunASR）：\n"
    "  1) .\\venv_assistant\\Scripts\\activate\n"
    "  2) pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
    "  3) pip install modelscope funasr soundfile sounddevice librosa scipy\n"
    "首次调用模型会自动从 ModelScope 下载 ~300MB 权重，需稳定网络。\n"
    "GPU 用户可把第 2 步换成 CUDA 版本（参考 PyTorch 官网），识别速度会显著更快。"
)


class ASRService:
    """ASR 服务：单例懒加载模型。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine: str = "text-mock"  # 实际生效后端：sensevoice-small / funasr-fallback / text-mock
        self._note: str = ""
        self._model: Any = None  # FunASR 的 AutoModel 对象
        self._model_ready: bool = False
        self._startup_errors: list[str] = []
        # 探测 Level1 依赖
        self._probe_dependencies()

    # —————————————————————————————————————————————————————————————
    # 依赖探测
    # —————————————————————————————————————————————————————————————
    def _probe_dependencies(self) -> None:
        errs: list[str] = []
        for mod in ("torch", "funasr", "modelscope"):
            try:
                __import__(mod)
            except Exception as e:  # noqa: BLE001
                errs.append(f"import {mod} 失败：{type(e).__name__}")
        if errs:
            self._startup_errors = errs
            self._engine = "text-mock"
            self._note = "缺少 torch / funasr / modelscope，已降级为纯文本输入模式。"
        else:
            self._engine = "sensevoice-small"  # 后续实际 load 失败再降级
            self._note = "依赖齐全，首次调用 transcribe 时会懒加载 SenseVoiceSmall 模型。"

    # —————————————————————————————————————————————————————————————
    # 对外：信息查询
    # —————————————————————————————————————————————————————————————
    def get_info(self) -> dict[str, str]:
        return {
            "engine": self._engine,
            "model_ready": "true" if self._model_ready else "false",
            "note": self._note or "OK",
            "startup_errors": " | ".join(self._startup_errors) if self._startup_errors else "",
            "install_hint": _INSTALL_HINT_CN if self._engine != "sensevoice-small" else "",
        }

    # —————————————————————————————————————————————————————————————
    # 对外：转写文件
    # —————————————————————————————————————————————————————————————
    def transcribe_file(self, audio_path: str | Path, lang: str = "auto") -> str:
        """识别本地音频文件。失败或降级时返回中文友好提示。"""
        p = Path(audio_path) if isinstance(audio_path, (str, Path)) else Path(str(audio_path))
        if not p.exists() or not p.is_file():
            return f"❌ transcribe_file 失败：音频文件不存在：{p}"

        # 降级：纯文本模式直接返回提示（UI 侧可以弹输入框）
        if self._engine != "sensevoice-small" or not self._ensure_model():
            return (
                f"⚠️ ASR 当前处于纯文本模式（未启用离线识别）。\n"
                f"原因：{self._note or '; '.join(self._startup_errors) or '未知'}\n"
                f"{_INSTALL_HINT_CN}"
            )

        try:
            # 调用 FunASR SenseVoice：实际返回 list[dict]（[{"key":..., "text":...}]），
            # 兼容个别版本返回 dict 的情况
            result = self._model.generate(input=str(p), cache={}, language=lang or "auto", use_itn=True)
            texts: list[str] = []
            items: list[Any] = []
            if isinstance(result, list):
                items.extend(result)
            elif isinstance(result, dict):
                for _, arr in result.items():
                    if isinstance(arr, list):
                        items.extend(arr)
                    elif isinstance(arr, str):
                        texts.append(arr.strip())
            for it in items:
                t = it.get("text") if isinstance(it, dict) else str(it)
                if t:
                    texts.append(str(t).strip())
            if not texts and not items:
                # 兜底：整个 result 转字符串取一段
                import json as _json
                raw = _json.dumps(result, ensure_ascii=False)[:2000]
                return f"⚠️ ASR 已识别但未解析到文本字段，原始返回：{raw}"
            combined = " ".join(t for t in texts if t).strip()
            # 清洗 SenseVoice 特殊标记（<|zh|><|HAPPY|><|Speech|><|withitn|> 等）
            import re as _re
            combined = _re.sub(r"<\|[^|]*\|>", "", combined).strip()
            if not combined:
                return "⚠️ ASR 未识别到任何语音内容，请确认录音是否有声音、采样率 16kHz。"
            return combined
        except Exception as e:  # noqa: BLE001
            return f"❌ SenseVoice 识别失败：{type(e).__name__}: {e}\n建议：重试或换成文字输入。"

    # —————————————————————————————————————————————————————————————
    # 对外：录音 + 转写
    # —————————————————————————————————————————————————————————————
    def record_and_transcribe(
        self,
        duration: float = 5.0,
        samplerate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
        save_path: Optional[str | Path] = None,
        force_text: Optional[str] = None,
    ) -> str:
        """录音 duration 秒 → 保存 WAV → 调 transcribe_file。

        force_text 不为空：**旁路**（不录音、不调模型），直接返回该文本。
            用于：测试场景、缺麦克风环境、用户临时想用文字代替语音的场景。
        """
        # 1. 旁路：指定了 force_text 就直接返回
        if force_text is not None and str(force_text).strip() != "":
            return str(force_text).strip()

        # 2. 录音
        ensure_data_dirs()
        save_p: Path
        if save_path is None:
            ts = int(time.time() * 1000)
            save_p = DATA_ROOT / "asr_cache" / f"record_{ts}.wav"
        else:
            save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)

        ok, err = self._record_wav(
            out_path=save_p,
            duration=float(max(0.5, duration)),
            samplerate=int(max(8000, samplerate)),
            channels=int(max(1, channels)),
            device=device,
        )
        if not ok:
            return (
                f"⚠️ 录音失败：{err or '未知原因'}。\n"
                f"建议：\n"
                f"  a) 检查系统是否有麦克风并已授权应用录音权限；\n"
                f"  b) 安装 sounddevice：pip install sounddevice；\n"
                f"  c) 临时用语音转文字的话，直接传 force_text 参数（不录音直接返回文字）。"
            )

        # 3. 转写
        text = self.transcribe_file(save_p, lang="zh")
        # 附带上录音路径（调试面板可显示，文本里不污染 user_input）
        return text

    # —————————————————————————————————————————————————————————————
    # 内部：确保模型加载（单例懒加载 + 失败二次降级）
    # —————————————————————————————————————————————————————————————
    def _ensure_model(self) -> bool:
        if self._model_ready:
            return True
        with self._lock:
            if self._model_ready:
                return True
            if self._engine != "sensevoice-small":
                return False
            try:
                import torch  # noqa: F401
                from funasr import AutoModel  # type: ignore
                from modelscope import snapshot_download  # type: ignore

                # 首次：ModelScope 下载 iic/SenseVoiceSmall
                model_dir = snapshot_download("iic/SenseVoiceSmall", cache_dir=str(DATA_ROOT / "model_cache"))
                # 用 FunASR 加载（SenseVoice 推荐 AutoModel，disable_update=True 避免重下）
                self._model = AutoModel(
                    model=model_dir,
                    trust_remote_code=True,
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    disable_update=True,
                    device="cuda:0" if _torch_cuda_available_safe() else "cpu",
                )
                self._model_ready = True
                self._note = (
                    f"SenseVoiceSmall 加载成功，device={'cuda' if _torch_cuda_available_safe() else 'cpu'}，"
                    f"model_dir={model_dir}"
                )
                return True
            except Exception as e:  # noqa: BLE001
                # 加载失败：降级到 text-mock，不再重试
                self._engine = "text-mock"
                self._note = f"SenseVoiceSmall 加载失败：{type(e).__name__}: {e}，已降级为纯文本模式。"
                self._startup_errors.append(f"model load: {type(e).__name__}")
                return False

    # —————————————————————————————————————————————————————————————
    # 内部：录音（sounddevice 优先，缺则 pyaudio 备选，都缺就失败）
    # —————————————————————————————————————————————————————————————
    def _record_wav(
        self,
        out_path: Path,
        duration: float,
        samplerate: int,
        channels: int,
        device: Optional[int],
    ) -> tuple[bool, str]:
        """返回 (是否成功, 错误原因)。"""
        # 1) 先试 sounddevice
        sd = None
        try:
            import sounddevice as sd_safe  # type: ignore
            sd = sd_safe
        except Exception:
            sd = None

        if sd is not None:
            try:
                pass  # type: ignore
            except Exception as e:  # noqa: BLE001
                return False, f"numpy 未安装（sounddevice 需要 numpy 存数组）：{type(e).__name__}"
            try:
                frames = int(samplerate * duration)
                recording = sd.rec(
                    frames=frames,
                    samplerate=samplerate,
                    channels=channels,
                    dtype="int16",
                    device=device,
                )
                sd.wait()  # 阻塞直到录完
                # 存成 WAV
                import soundfile as sf_safe  # type: ignore

                sf_safe.write(str(out_path), recording, samplerate)
                return True, ""
            except Exception as e:  # noqa: BLE001
                # sounddevice 失败，尝试 pyaudio 备选
                err_sd = f"sounddevice 录音失败：{type(e).__name__}: {e}"
            # 继续：下面试 pyaudio
        # 2) 备选 pyaudio
        pa = None
        try:
            import pyaudio as pa_safe  # type: ignore
            pa = pa_safe
        except Exception:
            pa = None
        if pa is None:
            last_err = "sounddevice 和 pyaudio 都未安装，无法录音。"
            if sd is not None:
                last_err = err_sd + "；同时 pyaudio 也未安装。"
            return False, last_err
        try:
            audio = pa.PyAudio()
            try:
                fmt = pa.paInt16
                chunk = 1024
                stream = audio.open(
                    format=fmt,
                    channels=channels,
                    rate=samplerate,
                    input=True,
                    input_device_index=device,
                    frames_per_buffer=chunk,
                )
                try:
                    bufs: list[bytes] = []
                    total_chunks = int(samplerate * duration / chunk) + 1
                    for _ in range(total_chunks):
                        bufs.append(stream.read(chunk, exception_on_overflow=False))
                    data = b"".join(bufs)
                finally:
                    try:
                        stream.stop_stream()
                    except Exception:
                        pass
                    try:
                        stream.close()
                    except Exception:
                        pass
            finally:
                try:
                    audio.terminate()
                except Exception:
                    pass
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(2)
                wf.setframerate(samplerate)
                wf.writeframes(data)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, f"pyaudio 录音失败：{type(e).__name__}: {e}"


def _torch_cuda_available_safe() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# —————————————————————————————————————————————————————————————
# 模块级单例
# —————————————————————————————————————————————————————————————
_ASR_SVC = ASRService()


def get_asr_service() -> ASRService:
    return _ASR_SVC


def get_asr_info() -> dict[str, str]:
    return _ASR_SVC.get_info()


def asr_transcribe_file(audio_path: str | Path, lang: str = "auto") -> str:
    return _ASR_SVC.transcribe_file(audio_path, lang=lang)


def asr_record_and_transcribe(
    duration: float = 5.0,
    samplerate: int = 16000,
    channels: int = 1,
    device: Optional[int] = None,
    save_path: Optional[str | Path] = None,
    force_text: Optional[str] = None,
) -> str:
    return _ASR_SVC.record_and_transcribe(
        duration=duration,
        samplerate=samplerate,
        channels=channels,
        device=device,
        save_path=save_path,
        force_text=force_text,
    )


__all__ = [
    "ASRService",
    "get_asr_service",
    "get_asr_info",
    "asr_transcribe_file",
    "asr_record_and_transcribe",
    "_INSTALL_HINT_CN",
]
