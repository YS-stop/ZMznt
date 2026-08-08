"""唤醒词常驻监听服务（免手触发语音输入）。

工作原理（零新增依赖，复用已装的 SenseVoice）：
    1. sounddevice 常驻打开麦克风（16kHz 单声道，30ms 一帧）
    2. 能量 VAD（自适应噪声底）把连续音频切成「语音段」，静音不占 CPU
    3. 每个语音段写临时 wav → 交给 asr_service 的 SenseVoice 离线识别
    4. 识别文本含唤醒词（默认「小助手」）→ 触发 on_wake(command) 回调：
       - 唤醒词后面跟了内容 → command 就是指令（如「小助手打开抖音」→「打开抖音」）
       - 只喊了唤醒词 → command 为空，进入 8 秒「等指令」窗口，下一句话直接当指令

防回声自触发（关键！）：
    TTS 扬声器播报会被自家麦克风收进去 → 可能再次识别出唤醒词死循环。
    上层（UIBridgeService）在 Agent 推理 + TTS 播报期间调 pause()，结束调 resume()；
    pause 期间照常读流但直接丢弃（防缓冲区溢出），resume 后有 0.4s 冷却期。

线程模型：纯 threading.Thread + 回调，不依赖 Qt；回调里发 Qt 信号即可安全跨线程。
CPU 开销：静音时只有能量计算（<0.5%）；有人说话时 SenseVoice 识别 RTF≈0.09（2 秒语音约 0.2s CPU）。
"""
from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.utils.path_utils import DATA_ROOT, ensure_data_dirs  # noqa: E402

# ---------------- 常量 ----------------
SAMPLE_RATE = 16000
FRAME_SAMPLES = 480          # 30ms 一帧
PRE_ROLL_FRAMES = 10         # 0.3s 预滚缓冲，防止切掉语音开头
START_HOT_FRAMES = 2         # 连续 2 帧超阈值判定为说话开始
END_SILENCE_FRAMES = 40      # 连续 ~1.2s 低于阈值判定为说话结束（正常语速的中间停顿不会误切）
MIN_SEGMENT_SEC = 0.35       # 短于此判为噪声丢弃
MAX_SEGMENT_SEC = 15.0       # 最长一段，超时强制切段（防长背景音乐耗尽识别）
AWAIT_COMMAND_SEC = 8.0      # 只喊唤醒词后，等指令的时间窗
RESUME_COOLDOWN_SEC = 0.4    # resume 后冷却期，丢弃残留声音
NOISE_FLOOR_INIT = 0.008     # 初始噪声底（float32 RMS）
NOISE_FLOOR_MIN = 0.003      # 噪声底下限（安静房间）
THRESHOLD_RATIO = 3.0        # 触发阈值 = 噪声底 × 3

# 文本归一化：去空白和标点（唤醒匹配用）
_NORMALIZE_RE = re.compile(r"[\s，。,.!！?？、~～…·:：;；\"'“”‘’]+")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text or "")


class WakeWordService:
    """常驻唤醒监听。start() 后后台线程运行；事件全部走回调。"""

    def __init__(
        self,
        wake_words: Optional[list[str]] = None,
        on_wake: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Args:
            wake_words: 唤醒词列表（默认 ["小助手"]）。命中任意一个即触发。
            on_wake(command): 唤醒回调。command 为唤醒词之后的指令文本（可能为空串）。
            on_event(stage, message): 调试日志回调（stage: WAKE / WAKE_ERR / ASR）。
        """
        self._wake_words: list[str] = [w for w in (wake_words or ["小助手"]) if w]
        self._on_wake: Callable[[str], None] = on_wake or (lambda cmd: None)
        self._on_event: Callable[[str, str]] = on_event or (lambda stage, msg: None)

        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._awaiting_command_until: float = 0.0
        self._resume_at: float = 0.0

    # ------------------------------------------------------------------
    # 对外控制
    # ------------------------------------------------------------------
    def start(self) -> bool:
        """启动后台监听线程。已运行则直接返回 True。"""
        if self._running:
            return True
        self._stop_evt.clear()
        self._pause_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, name="WakeWordListener", daemon=True)
        self._thread.start()
        self._running = True
        return True

    def stop(self) -> None:
        """停止监听（线程退出，麦克风释放）。"""
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        self._thread = None
        self._running = False

    def pause(self) -> None:
        """暂停识别（麦克风照常读取但丢弃）。Agent 推理 / TTS 播报期间必须调用。"""
        self._pause_evt.set()

    def resume(self) -> None:
        """恢复监听，带 0.4s 冷却期（把喇叭余音丢完再识别）。"""
        self._resume_at = time.monotonic() + RESUME_COOLDOWN_SEC
        self._pause_evt.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    def expect_command(self, seconds: float = AWAIT_COMMAND_SEC) -> None:
        """开启「等指令」窗口：窗口内下一段语音无需唤醒词，直接作为指令回调。

        典型用途：系统语音询问用户后（如「要中止当前任务吗？」），用户的回答
        不需要再喊唤醒词。
        """
        self._awaiting_command_until = time.monotonic() + max(1.0, float(seconds))

    # ------------------------------------------------------------------
    # 主循环（后台线程）
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:  # noqa: BLE001
            self._event("WAKE_ERR", f"sounddevice 不可用，唤醒监听未启动：{type(e).__name__}: {e}")
            self._running = False
            return

        # ASR 引擎预检：text-mock 模式没法做唤醒
        try:
            from src.services.asr_service import get_asr_service
            svc = get_asr_service()
            if svc._engine != "sensevoice-small":
                self._event(
                    "WAKE_ERR",
                    f"ASR 引擎为 {svc._engine}（非离线识别），唤醒监听不可用。"
                    "请确认 torch/funasr/modelscope 已安装后重启应用。",
                )
                self._running = False
                return
            svc._ensure_model()  # 提前加载模型，避免第一次唤醒时卡顿 10s+
        except Exception as e:  # noqa: BLE001
            self._event("WAKE_ERR", f"ASR 引擎初始化失败，唤醒监听未启动：{type(e).__name__}: {e}")
            self._running = False
            return

        self._event(
            "WAKE",
            f"🎧 唤醒监听已启动：对麦克风说「{' / '.join(self._wake_words)}」即可唤醒（暂停时自动丢弃声音）。",
        )

        # VAD 状态
        noise_floor = NOISE_FLOOR_INIT
        in_speech = False
        hot_run = 0
        silence_run = 0
        seg_frames: list[np.ndarray] = []
        pre_roll: list[np.ndarray] = []
        seg_start = 0.0

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
            ) as stream:
                while not self._stop_evt.is_set():
                    try:
                        data, _overflowed = stream.read(FRAME_SAMPLES)
                    except Exception as e:  # noqa: BLE001
                        self._event("WAKE_ERR", f"麦克风读取异常：{type(e).__name__}: {e}，1s 后重试")
                        time.sleep(1.0)
                        continue
                    frame = np.asarray(data, dtype=np.float32).reshape(-1)

                    # 暂停 / 冷却期：丢弃声音并重置 VAD 状态
                    if self._pause_evt.is_set() or time.monotonic() < self._resume_at:
                        in_speech = False
                        hot_run = 0
                        silence_run = 0
                        seg_frames = []
                        pre_roll = []
                        continue

                    rms = float(np.sqrt(np.mean(frame * frame)) + 1e-12)
                    threshold = max(noise_floor * THRESHOLD_RATIO, NOISE_FLOOR_MIN * THRESHOLD_RATIO)

                    if not in_speech:
                        # 缓慢更新噪声底（只在非语音段）
                        noise_floor = 0.98 * noise_floor + 0.02 * min(rms, threshold)
                        pre_roll.append(frame)
                        if len(pre_roll) > PRE_ROLL_FRAMES:
                            pre_roll.pop(0)
                        if rms > threshold:
                            hot_run += 1
                            if hot_run >= START_HOT_FRAMES:
                                in_speech = True
                                silence_run = 0
                                seg_start = time.monotonic()
                                seg_frames = list(pre_roll)  # 带上预滚
                                pre_roll = []
                        else:
                            hot_run = 0
                    else:
                        seg_frames.append(frame)
                        seg_dur = time.monotonic() - seg_start
                        if rms > threshold:
                            silence_run = 0
                        else:
                            silence_run += 1
                        if silence_run >= END_SILENCE_FRAMES or seg_dur >= MAX_SEGMENT_SEC:
                            audio = np.concatenate(seg_frames)
                            in_speech = False
                            hot_run = 0
                            silence_run = 0
                            seg_frames = []
                            if len(audio) >= int(MIN_SEGMENT_SEC * SAMPLE_RATE):
                                self._process_segment(audio)
        except Exception as e:  # noqa: BLE001
            self._event("WAKE_ERR", f"麦克风打开失败（可能被占用）：{type(e).__name__}: {e}")
        finally:
            self._running = False
            self._event("WAKE", "唤醒监听已停止。")

    # ------------------------------------------------------------------
    # 语音段处理（独立方法，方便离线测试直接喂音频）
    # ------------------------------------------------------------------
    def _process_segment(self, audio: np.ndarray) -> None:
        """识别一段语音并做唤醒匹配。audio: float32 16kHz 单声道。"""
        ensure_data_dirs()
        ts = int(time.time() * 1000)
        wav_path = DATA_ROOT / "asr_cache" / f"wake_seg_{ts}.wav"
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(wav_path), audio, SAMPLE_RATE)

            from src.services.asr_service import get_asr_service
            text = get_asr_service().transcribe_file(wav_path, lang="zh")
        except Exception as e:  # noqa: BLE001
            self._event("WAKE_ERR", f"语音段识别异常：{type(e).__name__}: {e}")
            return
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

        if not text or text.startswith("⚠️") or text.startswith("❌"):
            # 识别失败/无人说话：静默忽略（等指令窗口内也不报错）
            return

        self._event("ASR", f"环境语音：{text[:80]}")
        self.handle_transcript(text)

    def handle_transcript(self, text: str) -> Optional[str]:
        """对已识别文本做唤醒匹配（独立出来便于单元测试）。

        Returns:
            触发了唤醒 → 指令文本（可能为空串表示只喊了唤醒词）；未触发 → None。
        """
        norm = _normalize(text)
        if not norm:
            return None

        # 等指令窗口内：任何语音都直接当指令
        if time.monotonic() < self._awaiting_command_until:
            self._awaiting_command_until = 0.0
            self._event("WAKE", f"收到唤醒后的指令：{norm[:60]}")
            self._on_wake(norm)
            return norm

        for word in self._wake_words:
            w = _normalize(word)
            idx = norm.find(w)
            if idx < 0:
                continue
            command = norm[idx + len(w):].strip()
            if command:
                self._event("WAKE", f"✅ 命中唤醒词「{word}」，指令：{command[:60]}")
                self._on_wake(command)
            else:
                # 只喊了唤醒词 → 开 8 秒等指令窗口
                self._awaiting_command_until = time.monotonic() + AWAIT_COMMAND_SEC
                self._event("WAKE", f"✅ 命中唤醒词「{word}」，{AWAIT_COMMAND_SEC:.0f}s 内说出指令即可。")
                self._on_wake("")
            return command
        return None

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _event(self, stage: str, message: str) -> None:
        try:
            self._on_event(stage, message)
        except Exception:  # noqa: BLE001 - 回调异常不影响监听线程
            pass


__all__ = ["WakeWordService"]
