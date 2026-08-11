"""唤醒词常驻监听 + 持续监听（M8 进阶版：实时转写预览 + 收尾词 + 追加合并 + 语义断句）。

M8 新增能力（2026-08-10）：
    1. ✅ 实时转写预览：开口后每 ~0.8s 增量识别一次当前累积音频 → UI 气泡实时显示 ▍光标
    2. ✅ 收尾词提前结束：识别到"执行/开始/去吧/就这样/好了/可以了/..."等强收尾词立即提交
    3. ✅ 追加说话合并：尾点静音触发后给 1.0s 追加窗口，用户再开口自动继续收音拼接
    4. ✅ 语义级智能断句：LLM 判断语义完整性，完整则提前提交（高级版，默认开启）
    5. ✅ 所有新功能可通过 settings 表开关/调参，热生效

架构（帧驱动，内部时钟 30ms/帧，pause 期间时钟冻结 = 计时器天然冻结）：

    ┌───────────────────────────── WAKE 模式（环境监听）─────────────────────────────┐
    │  能量 VAD（自适应基线）切语音段 → SenseVoice 识别 → 唤醒词匹配：                │
    │    · 段内含「唤醒词+指令」（小助手打开抖音）→ 直接回调 on_wake(command)         │
    │    · 段内只含唤醒词 → 进入 COMMAND 模式 + on_wake("") 让 UI 提示「我在」        │
    └───────────────────────────────────────────────────────────────────────────────┘
        │ 只喊了唤醒词 / expect_command()（换任务确认等答案）
        ▼
    ┌───────────────────────────── COMMAND 模式（M8 进阶持续监听）────────────────────┐
    │  等待开口态：5s 未开口 → LISTEN_PROMPT 事件（TTS「在呢，请说」），             │
    │              再等 3s 仍无语音 → LISTEN_TIMEOUT 退出回 WAKE                      │
    │  语音进行态(SPEECH)：全程入缓冲（含短暂静音帧，不截断句尾）                      │
    │    · 每 0.8s 触发增量 ASR → LISTEN_PREVIEW 事件（UI 实时预览 ▍）               │
    │    · 增量结果检查收尾词 → 命中则 LISTEN_FINISH_WORD → 立即提交                  │
    │    · 增量结果走语义断句（LLM）→ COMPLETE 则提前提交                            │
    │  尾点追加态(TAIL_WINDOW)：连续静音达到尾点阈值 → 不立即提交，开 1.0s 窗口       │
    │    · 窗口内再次检测到语音 → 回到 SPEECH 继续收音（追加合并）                    │
    │    · 窗口超时无语音 → 整段缓冲一次性提交 ASR → LISTEN_SUBMIT                   │
    │  最长录音保护：总长 ≥ max_record_sec 强制提交（防无限收音）                     │
    │  监听中重复喊唤醒词：识别文本取「最后一次唤醒词出现」之后的内容作为指令，        │
    │    等效于清空缓冲重新监听，无需额外确认                                        │
    └───────────────────────────────────────────────────────────────────────────────┘

防回声自触发（保留原机制）：
    TTS 播报 / 手动按住说话期间上层调 pause()：帧直接丢弃、内部时钟冻结
    （COMMAND 模式的开口超时/尾点计时/增量ASR计时全部暂停），resume() 后 0.4s 冷却再收音。

可配置参数（SQLite settings 表，设置中心可视化调整，reload_config() 实时生效）：
    vad_tail_silence_sec      尾点静音阈值（默认 1.0s，比旧版略短）
    vad_max_record_sec        最长录音时长（默认 30s，超时强制提交）
    vad_await_speech_sec      唤醒后等待开口超时（默认 5s，随后语音提示 + 3s 宽限退出）
    vad_threshold_ratio       能量阈值系数（默认 3.0，嘈杂环境调高）
    vad_incremental_interval  增量ASR间隔秒（默认 0.8s，设0禁用预览）
    vad_append_window_sec     追加说话窗口秒（默认 1.0s，设0禁用追加合并）
    vad_semantic_check        是否启用语义断句（true/false，默认 true）

线程模型：纯 threading.Thread + 回调，不依赖 Qt；回调里发 Qt 信号即可安全跨线程。
"""
from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from src.utils.path_utils import DATA_ROOT, ensure_data_dirs  # noqa: E402

# ---------------- 音频常量 ----------------
SAMPLE_RATE = 16000
FRAME_SAMPLES = 480          # 30ms 一帧
FRAME_SEC = FRAME_SAMPLES / SAMPLE_RATE
PRE_ROLL_FRAMES = 10         # 0.3s 预滚缓冲，防止切掉语音开头
TAIL_KEEP_FRAMES = 10        # 提交时保留 0.3s 句尾静音（防截断）
START_HOT_FRAMES = 2         # 连续 2 帧超阈值判定为说话开始
MIN_SEGMENT_SEC = 0.35       # 有效语音短于此判为噪声丢弃
MAX_SEGMENT_SEC = 15.0       # WAKE 模式环境语音段最长（防长背景音乐耗尽识别）
RESUME_COOLDOWN_FRAMES = 13  # resume 后冷却 ~0.4s，丢弃喇叭余音
NOISE_FLOOR_INIT = 0.008     # 初始噪声底（float32 RMS）
NOISE_FLOOR_MIN = 0.003      # 噪声底下限（安静房间）
NOISE_WINDOW_FRAMES = 33     # 自适应基线窗口：过去 ~1s

# ---------------- M8 新增常量 ----------------
DEFAULT_INCREMENTAL_INTERVAL = 0.8   # 增量ASR间隔：0.8秒（平衡延迟与CPU）
DEFAULT_APPEND_WINDOW_SEC = 1.0      # 追加说话窗口：尾点后1秒内继续说就合并
MIN_CHARS_FOR_SEMANTIC = 4           # 语义断句最少字符数（太短不调LLM）
INCREMENTAL_MIN_AUDIO_SEC = 0.6      # 增量ASR最短音频（太短识别没意义）

# ---------------- 默认可调参数（被 settings 表覆盖） ----------------
DEFAULT_TAIL_SILENCE_SEC = 1.0       # M8: 尾点默认1.0s（旧1.2s，配合追加窗口体验更好）
DEFAULT_MAX_RECORD_SEC = 30.0
DEFAULT_AWAIT_SPEECH_SEC = 5.0
AWAIT_GRACE_SEC = 3.0
DEFAULT_THRESHOLD_RATIO = 3.0

# 模式
MODE_WAKE = "WAKE"
MODE_COMMAND = "COMMAND"

# COMMAND 子状态
CMD_WAIT = "WAIT"               # 等待开口
CMD_SPEECH = "SPEECH"           # 正在收音
CMD_TAIL_WINDOW = "TAIL_WINDOW" # M8: 尾点后追加窗口

# 文本归一化：去空白和标点（唤醒匹配用）
_NORMALIZE_RE = re.compile(r"[\s，。,.!！?？、~～…·:：;；\"'“”‘’]+")


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text or "")


@dataclass
class ListenConfig:
    """持续监听可调参数（从 SQLite settings 读取，见 reload_config）。"""

    tail_silence_sec: float = DEFAULT_TAIL_SILENCE_SEC
    max_record_sec: float = DEFAULT_MAX_RECORD_SEC
    await_speech_sec: float = DEFAULT_AWAIT_SPEECH_SEC
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO
    incremental_interval_sec: float = DEFAULT_INCREMENTAL_INTERVAL  # M8
    append_window_sec: float = DEFAULT_APPEND_WINDOW_SEC            # M8
    semantic_check_enabled: bool = True                              # M8

    @property
    def tail_frames(self) -> int:
        return max(10, int(self.tail_silence_sec / FRAME_SEC))

    @property
    def append_frames(self) -> int:
        return max(0, int(self.append_window_sec / FRAME_SEC))

    @property
    def incremental_frames(self) -> int:
        return max(0, int(self.incremental_interval_sec / FRAME_SEC))


class WakeWordService:
    """常驻唤醒监听 + 唤醒后持续监听（M8 进阶版）。start() 后后台线程运行；事件全部走回调。"""

    def __init__(
        self,
        wake_words: Optional[list[str]] = None,
        on_wake: Optional[Callable[[str], None]] = None,
        on_event: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Args:
            wake_words: 唤醒词列表（默认 ["小助手"]）。命中任意一个即触发。
            on_wake(command): 唤醒回调。command 为指令文本（空串 = 只喊了唤醒词，进入持续监听）。
            on_event(stage, message): 调试/状态回调。stage：
                WAKE / WAKE_ERR / ASR   —— 与原有一致的日志
                LISTEN_PROMPT           —— 持续监听 5s 未开口（上层可 TTS「在呢，请说」）
                LISTEN_TIMEOUT          —— 等待开口超时，已退回环境监听
                LISTEN_SUBMIT           —— 尾点触发，缓冲已提交 ASR（上层可切 THINKING）
                LISTEN_PREVIEW          —— M8: 增量识别结果预览（text 为识别文本）
                LISTEN_FINISH_WORD      —— M8: 命中收尾词提前结束
                LISTEN_APPEND_MERGE     —— M8: 追加说话合并成功
        """
        self._wake_words: list[str] = [w for w in (wake_words or ["小助手"]) if w]
        self._on_wake: Callable[[str], None] = on_wake or (lambda cmd: None)
        self._on_event: Callable[[str, str]] = on_event or (lambda stage, msg: None)

        self._stop_evt = threading.Event()
        self._pause_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._cooldown_frames = 0

        # —— 配置 ——
        self._cfg = ListenConfig()
        self._load_config()

        # —— 内部时钟（秒）：每处理一帧 +FRAME_SEC；pause/冷却期间冻结 ——
        self._clock = 0.0

        # —— 自适应噪声基线：过去 ~1s 非语音帧能量，取最低 20% 均值 ——
        self._noise_window: deque[float] = deque(maxlen=NOISE_WINDOW_FRAMES)

        # —— WAKE 模式 VAD 状态 ——
        self._wk_in_speech = False
        self._wk_hot = 0
        self._wk_silence = 0
        self._wk_frames: list[np.ndarray] = []
        self._wk_pre: list[np.ndarray] = []
        self._wk_start = 0.0

        # —— COMMAND 模式状态 ——
        self._mode = MODE_WAKE
        self._cmd_state = CMD_WAIT
        self._cmd_prompt_enabled = True
        self._cmd_prompt_at = 0.0
        self._cmd_deadline = 0.0
        self._cmd_prompted = False
        self._cmd_hot = 0
        self._cmd_silence = 0
        self._cmd_frames: list[np.ndarray] = []
        self._cmd_pre: list[np.ndarray] = []
        self._cmd_speech_start = 0.0
        # M8: 增量ASR相关
        self._cmd_last_incremental_frame_idx: int = 0  # 上次增量识别时的帧计数
        self._cmd_last_preview_text: str = ""           # 上次预览文本（去重）
        self._cmd_tail_window_start: float = 0.0        # 追加窗口开始时间
        self._cmd_tail_frames_count: int = 0            # 追加窗口内已计时帧数
        self._cmd_submitting: bool = False              # 提交中防重入
        # expect_command() 跨线程传入的待处理请求：(wait_sec, prompt_enabled)
        self._pending_cmd_req: Optional[tuple[float, bool]] = None

        # —— 语义断句器（懒加载）——
        self._segmenter = None

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        """从 SQLite settings 读取可调参数（服务独立运行/测试时静默用默认值）。"""
        try:
            from src.services.sqlite_db import get_setting

            self._cfg = ListenConfig(
                tail_silence_sec=float(get_setting("vad_tail_silence_sec", DEFAULT_TAIL_SILENCE_SEC)),
                max_record_sec=float(get_setting("vad_max_record_sec", DEFAULT_MAX_RECORD_SEC)),
                await_speech_sec=float(get_setting("vad_await_speech_sec", DEFAULT_AWAIT_SPEECH_SEC)),
                threshold_ratio=float(get_setting("vad_threshold_ratio", DEFAULT_THRESHOLD_RATIO)),
                incremental_interval_sec=float(get_setting("vad_incremental_interval", DEFAULT_INCREMENTAL_INTERVAL)),
                append_window_sec=float(get_setting("vad_append_window_sec", DEFAULT_APPEND_WINDOW_SEC)),
                semantic_check_enabled=bool(get_setting("vad_semantic_check", True)),
            )
        except Exception:  # noqa: BLE001 - 无 DB 环境用默认值
            self._cfg = ListenConfig()

    def reload_config(self) -> None:
        """设置中心保存后调用：热加载监听参数，无需重启服务。"""
        old = self._cfg
        self._load_config()
        self._event(
            "WAKE",
            f"监听参数已更新：尾点 {old.tail_silence_sec}s→{self._cfg.tail_silence_sec}s，"
            f"最长录音 {self._cfg.max_record_sec:.0f}s，等待开口 {self._cfg.await_speech_sec:.0f}s，"
            f"阈值系数 {self._cfg.threshold_ratio}，增量ASR {self._cfg.incremental_interval_sec}s，"
            f"追加窗口 {self._cfg.append_window_sec}s，语义断句 {'开' if self._cfg.semantic_check_enabled else '关'}。",
        )

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
        """暂停收音（帧丢弃 + 内部时钟冻结）。Agent 推理 / TTS 播报 / 手动录音期间必须调用。"""
        self._pause_evt.set()

    def resume(self) -> None:
        """恢复监听，带 ~0.4s 冷却期（把喇叭余音丢完再识别）。"""
        self._cooldown_frames = RESUME_COOLDOWN_FRAMES
        self._pause_evt.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    def is_in_command_mode(self) -> bool:
        """当前是否处于「唤醒后持续监听」状态（UI 据此维持 LISTENING 球态）。"""
        return self._mode == MODE_COMMAND

    def expect_command(self, seconds: float = 10.0) -> None:
        """开启「等指令/等答案」持续监听：窗口内语音无需唤醒词，直接作为指令回调。

        典型用途：系统语音询问用户后（如「要中止当前任务吗？」），用户的回答
        不需要再喊唤醒词。seconds 内未开口则自动退回环境监听。
        """
        self._pending_cmd_req = (max(3.0, float(seconds)), False)

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
            f"🎧 唤醒监听已启动（M8进阶持续监听版）：对麦克风说「{' / '.join(self._wake_words)}」即可唤醒；"
            f"尾点 {self._cfg.tail_silence_sec}s + 追加窗口 {self._cfg.append_window_sec}s / "
            f"增量预览 {self._cfg.incremental_interval_sec}s / 语义断句 {'开' if self._cfg.semantic_check_enabled else '关'}。",
        )

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
                    self._feed_frame(frame)
        except Exception as e:  # noqa: BLE001
            self._event("WAKE_ERR", f"麦克风打开失败（可能被占用）：{type(e).__name__}: {e}")
        finally:
            self._running = False
            self._event("WAKE", "唤醒监听已停止。")

    # ------------------------------------------------------------------
    # 帧驱动状态机（独立方法，测试可直接喂合成音频帧）
    # ------------------------------------------------------------------
    def _feed_frame(self, frame: np.ndarray) -> None:
        """喂入一帧 30ms 音频。pause/冷却期间丢弃且时钟冻结（计时器天然暂停）。"""
        if self._pause_evt.is_set():
            # WAKE 模式的半截语音段直接作废（暂停前的大概率是自家 TTS 余音）；
            # COMMAND 模式的缓冲区保留，计时冻结，恢复后继续收音
            if self._mode == MODE_WAKE:
                self._reset_wake_vad()
            return
        if self._cooldown_frames > 0:
            self._cooldown_frames -= 1
            return

        self._clock += FRAME_SEC

        # expect_command 跨线程请求：在帧循环里安全消费
        if self._pending_cmd_req is not None and self._mode == MODE_WAKE:
            wait_sec, prompt = self._pending_cmd_req
            self._pending_cmd_req = None
            self._enter_command_mode(wait_sec, prompt)

        rms = float(np.sqrt(np.mean(frame * frame)) + 1e-12)
        threshold = self._threshold()

        if self._mode == MODE_WAKE:
            self._feed_wake(frame, rms, threshold)
        else:
            self._feed_command(frame, rms, threshold)

    # ---------------- WAKE 模式：环境监听 + 唤醒词检测 ----------------
    def _feed_wake(self, frame: np.ndarray, rms: float, threshold: float) -> None:
        if not self._wk_in_speech:
            self._update_noise(rms, threshold)
            self._wk_pre.append(frame)
            if len(self._wk_pre) > PRE_ROLL_FRAMES:
                self._wk_pre.pop(0)
            if rms > threshold:
                self._wk_hot += 1
                if self._wk_hot >= START_HOT_FRAMES:
                    self._wk_in_speech = True
                    self._wk_silence = 0
                    self._wk_start = self._clock
                    self._wk_frames = list(self._wk_pre)
                    self._wk_pre = []
            else:
                self._wk_hot = 0
            return

        # 语音段进行中
        self._wk_frames.append(frame)
        if rms > threshold:
            self._wk_silence = 0
        else:
            self._wk_silence += 1
        seg_dur = self._clock - self._wk_start
        if self._wk_silence >= self._cfg.tail_frames or seg_dur >= MAX_SEGMENT_SEC:
            audio = self._collect_buffer(self._wk_frames, self._wk_silence)
            self._reset_wake_vad()
            if audio is not None and len(audio) >= int(MIN_SEGMENT_SEC * SAMPLE_RATE):
                self._process_wake_segment(audio)

    def _reset_wake_vad(self) -> None:
        self._wk_in_speech = False
        self._wk_hot = 0
        self._wk_silence = 0
        self._wk_frames = []
        self._wk_pre = []

    def _process_wake_segment(self, audio: np.ndarray) -> None:
        """识别环境语音段并做唤醒匹配（WAKE 模式）。"""
        text = self._transcribe_audio(audio)
        if not text or text.startswith("⚠️") or text.startswith("❌"):
            return
        self._event("ASR", f"环境语音：{text[:80]}")
        self.handle_transcript(text)

    # ---------------- COMMAND 模式：M8 进阶持续监听 ----------------
    def _enter_command_mode(self, wait_speech_sec: float, prompt_enabled: bool) -> None:
        """进入持续监听：等待开口 → 收音 → (增量预览/收尾词/追加窗口/语义断句) → 提交。"""
        self._mode = MODE_COMMAND
        self._cmd_state = CMD_WAIT
        self._cmd_prompt_enabled = bool(prompt_enabled)
        self._cmd_prompt_at = self._clock + max(1.0, wait_speech_sec)
        # 提示开启时给 AWAIT_GRACE_SEC 宽限；expect_command 场景 deadline 就是 wait 本身
        self._cmd_deadline = self._cmd_prompt_at + (AWAIT_GRACE_SEC if prompt_enabled else 0.0)
        self._cmd_prompted = False
        self._cmd_hot = 0
        self._cmd_silence = 0
        self._cmd_frames = []
        self._cmd_pre = []
        # M8: 重置增量/追加状态
        self._cmd_last_incremental_frame_idx = 0
        self._cmd_last_preview_text = ""
        self._cmd_tail_window_start = 0.0
        self._cmd_tail_frames_count = 0
        self._cmd_submitting = False
        # 重置语义断句缓存
        if self._segmenter is not None:
            try:
                self._segmenter.reset_cache()
            except Exception:
                pass
        self._event("LISTEN", f"进入持续监听(M8)：{wait_speech_sec:.0f}s 内开口。")

    def _exit_command_mode(self) -> None:
        self._mode = MODE_WAKE
        self._cmd_state = CMD_WAIT
        self._cmd_frames = []
        self._cmd_pre = []
        self._cmd_hot = 0
        self._cmd_silence = 0
        self._cmd_submitting = False

    def _feed_command(self, frame: np.ndarray, rms: float, threshold: float) -> None:
        if self._cmd_submitting:
            return  # 已在提交流程，不处理新帧

        if self._cmd_state == CMD_WAIT:
            self._update_noise(rms, threshold)
            self._cmd_pre.append(frame)
            if len(self._cmd_pre) > PRE_ROLL_FRAMES:
                self._cmd_pre.pop(0)
            if rms > threshold:
                self._cmd_hot += 1
                if self._cmd_hot >= START_HOT_FRAMES:
                    self._enter_speech_state()
            else:
                self._cmd_hot = 0
            # 等待开口超时管理
            if self._cmd_state == CMD_WAIT:
                if (
                    self._cmd_prompt_enabled
                    and not self._cmd_prompted
                    and self._clock >= self._cmd_prompt_at
                ):
                    self._cmd_prompted = True
                    self._event("LISTEN_PROMPT", "等待开口超时，提示用户。")
                if self._clock >= self._cmd_deadline:
                    self._event("LISTEN_TIMEOUT", "未检测到发言，退出持续监听。")
                    self._exit_command_mode()
            return

        if self._cmd_state == CMD_TAIL_WINDOW:
            # M8: 追加窗口 —— 等一下看用户是否继续说
            self._cmd_frames.append(frame)
            if rms > threshold:
                # 检测到新语音 → 追加合并！回到SPEECH继续收音
                self._cmd_hot += 1
                if self._cmd_hot >= START_HOT_FRAMES:
                    self._cmd_state = CMD_SPEECH
                    self._cmd_silence = 0
                    self._cmd_hot = 0
                    self._cmd_tail_frames_count = 0
                    self._event("LISTEN_APPEND_MERGE", "追加说话检测到语音，继续收音合并。")
            else:
                self._cmd_hot = 0
                self._cmd_tail_frames_count += 1
                if self._cmd_tail_frames_count >= self._cfg.append_frames:
                    # 追加窗口超时 → 真正提交
                    self._finalize_and_submit("追加窗口超时，提交识别")
            return

        # CMD_SPEECH：全程入缓冲（含短暂静音帧，不截断句尾）
        self._cmd_frames.append(frame)
        if rms > threshold:
            self._cmd_silence = 0
        else:
            self._cmd_silence += 1

        speech_dur = self._clock - self._cmd_speech_start

        # M8: 1. 检查是否到增量ASR时间
        total_frames = len(self._cmd_frames)
        inc_interval = self._cfg.incremental_frames
        if inc_interval > 0 and (total_frames - self._cmd_last_incremental_frame_idx) >= inc_interval:
            audio_dur_sec = total_frames * FRAME_SEC
            if audio_dur_sec >= INCREMENTAL_MIN_AUDIO_SEC:
                self._cmd_last_incremental_frame_idx = total_frames
                self._do_incremental_asr()

        # M8: 2. 尾点静音 → 进入追加窗口（不立即提交）
        if self._cmd_silence >= self._cfg.tail_frames:
            if self._cfg.append_frames > 0:
                self._cmd_state = CMD_TAIL_WINDOW
                self._cmd_tail_window_start = self._clock
                self._cmd_tail_frames_count = 0
                self._cmd_hot = 0
                self._event("LISTEN", f"尾点静音达标，进入追加窗口({self._cfg.append_window_sec}s)...")
            else:
                # 追加窗口禁用，直接提交
                self._finalize_and_submit("尾点静音达标")
            return

        # 3. 最长录音保护
        if speech_dur >= self._cfg.max_record_sec:
            self._finalize_and_submit("达到最长录音上限，强制提交")

    def _enter_speech_state(self) -> None:
        """从WAIT进入SPEECH状态的初始化。"""
        self._cmd_state = CMD_SPEECH
        self._cmd_silence = 0
        self._cmd_speech_start = self._clock
        self._cmd_frames = list(self._cmd_pre)
        self._cmd_pre = []
        self._cmd_last_incremental_frame_idx = 0
        self._cmd_last_preview_text = ""
        self._event("LISTEN", "检测到开口，持续收音中…")
        # 立即发一个空预览，让UI显示"正在听..."
        self._event("LISTEN_PREVIEW", "")

    def _do_incremental_asr(self) -> None:
        """M8: 对当前累积音频做增量识别，检查收尾词和语义完整性。"""
        try:
            audio = self._collect_buffer(self._cmd_frames, self._cmd_silence)
            if audio is None or len(audio) < int(INCREMENTAL_MIN_AUDIO_SEC * SAMPLE_RATE):
                return
            text = self._transcribe_audio_fast(audio)
            if not text or text.startswith("⚠️") or text.startswith("❌"):
                return
            norm = _normalize(text)
            if not norm:
                return
            # 去重：和上次预览一样就不发
            if norm == self._cmd_last_preview_text:
                return
            self._cmd_last_preview_text = norm
            # 发送预览事件
            display_text = text.strip()
            self._event("LISTEN_PREVIEW", display_text)

            # 检查强收尾词（快速路径）
            if self._check_finish_words(norm):
                self._event("LISTEN_FINISH_WORD", f"命中收尾词：{display_text[:50]}")
                self._finalize_and_submit("命中收尾词，提前结束")
                return

            # 语义断句（LLM）
            if self._cfg.semantic_check_enabled and len(norm) >= MIN_CHARS_FOR_SEMANTIC:
                self._check_semantic_complete(norm, display_text)
        except Exception as e:  # noqa: BLE001
            # 增量识别失败不影响主流程，等下一次
            pass

    def _check_finish_words(self, normalized_text: str) -> bool:
        """M8: 快速路径检查强收尾词（执行/开始/去吧/就这样/好了等）。"""
        # 强收尾关键词/短语（用户明确表示指令说完了）
        strong_finish = (
            "执行", "开始", "去吧", "就这样", "好了", "可以了", "行了",
            "确定", "确认", "提交", "发送", "搞定", "完事",
            "吧", "呗", "啦",
        )
        t = normalized_text.rstrip("。！？!?，, ")
        # 必须满足最小长度，避免单个字误判
        if len(t) < 2:
            return False
        # 检查强收尾词
        for kw in strong_finish:
            if t.endswith(kw):
                # 单字语气词要求前面至少有2个字符（避免"吧"单独出现误判）
                if len(kw) == 1 and len(t) < 3:
                    continue
                return True
        return False

    def _check_semantic_complete(self, normalized_text: str, display_text: str) -> None:
        """M8: 语义级断句（LLM判断）。"""
        try:
            from src.services.semantic_segment_service import get_semantic_segmenter, SegmentJudgment
            seg = get_semantic_segmenter()
            result = seg.judge(normalized_text, min_chars=MIN_CHARS_FOR_SEMANTIC)
            if result.judgment == SegmentJudgment.COMPLETE and result.confidence >= 0.6:
                self._event("LISTEN", f"语义判定完整({result.confidence:.2f})：{result.reason}")
                self._finalize_and_submit("语义完整，提前提交")
        except Exception:
            pass

    def _finalize_and_submit(self, reason: str) -> None:
        """M8: 统一提交流程（尾点/收尾词/语义/最长录音都走这里）。"""
        if self._cmd_submitting:
            return
        self._cmd_submitting = True

        audio = self._collect_buffer(self._cmd_frames, self._cmd_silence)
        self._exit_command_mode()
        if audio is None or len(audio) < int(MIN_SEGMENT_SEC * SAMPLE_RATE):
            self._event("LISTEN", "语音过短，按噪声丢弃。")
            self._event("LISTEN_PREVIEW_CLEAR", "")
            return

        self._event("LISTEN_SUBMIT", f"{reason}（{len(audio) / SAMPLE_RATE:.1f}s 音频）。")
        text = self._transcribe_audio(audio)
        self._handle_command_text(text)

    # ---------------- COMMAND 文本处理（保留原逻辑） ----------------
    def _handle_command_text(self, text: str) -> None:
        """处理持续监听的识别结果：重复唤醒取最后一次出现后的内容。"""
        if not text or text.startswith("⚠️") or text.startswith("❌"):
            self._event("LISTEN_TIMEOUT", "未识别到有效内容，退回环境监听。")
            self._event("LISTEN_PREVIEW_CLEAR", "")
            return
        norm = _normalize(text)
        if not norm:
            self._event("LISTEN_TIMEOUT", "识别结果为空，退回环境监听。")
            self._event("LISTEN_PREVIEW_CLEAR", "")
            return

        # 监听中重复喊唤醒词：等效「清空缓冲、重新监听」——取最后一次唤醒词之后的内容
        last_idx, last_wlen = -1, 0
        for word in self._wake_words:
            w = _normalize(word)
            if not w:
                continue
            idx = norm.rfind(w)
            if idx > last_idx:
                last_idx, last_wlen = idx, len(w)
        if last_idx >= 0:
            command = norm[last_idx + last_wlen:].strip()
            if command:
                self._event("WAKE", f"✅ 持续监听收到指令：{command[:60]}")
                self._on_wake(command)
            else:
                # 又只喊了一遍唤醒词 → 重新开一轮持续监听
                self._event("WAKE", "再次听到唤醒词，重新进入持续监听。")
                self._on_wake("")
                self._enter_command_mode(self._cfg.await_speech_sec, prompt_enabled=True)
            return

        self._event("WAKE", f"✅ 持续监听收到指令：{norm[:60]}")
        self._on_wake(norm)

    # ------------------------------------------------------------------
    # 唤醒匹配（WAKE 模式识别结果处理；独立出来便于单元测试）
    # ------------------------------------------------------------------
    def handle_transcript(self, text: str) -> Optional[str]:
        """对 WAKE 模式识别文本做唤醒匹配。

        Returns:
            触发了唤醒 → 指令文本（空串表示只喊了唤醒词，已进入持续监听）；未触发 → None。
        """
        norm = _normalize(text)
        if not norm:
            return None

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
                # 只喊了唤醒词 → 进入持续监听
                self._enter_command_mode(self._cfg.await_speech_sec, prompt_enabled=True)
                self._event("WAKE", f"✅ 命中唤醒词「{word}」，进入持续监听，请说指令。")
                self._on_wake("")
            return command
        return None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _threshold(self) -> float:
        """自适应阈值：过去 ~1s 非语音帧能量最低 20% 的均值 × 系数（带下限保护）。"""
        if self._noise_window:
            low = sorted(self._noise_window)
            k = max(1, len(low) // 5)
            baseline = sum(low[:k]) / k
        else:
            baseline = NOISE_FLOOR_INIT
        ratio = self._cfg.threshold_ratio
        return max(baseline * ratio, NOISE_FLOOR_MIN * ratio)

    def _update_noise(self, rms: float, threshold: float) -> None:
        """只在非语音帧更新基线窗口；明显超阈值的响度不入窗（防污染）。"""
        if rms < threshold * 1.5:
            self._noise_window.append(rms)

    @staticmethod
    def _collect_buffer(frames: list[np.ndarray], trailing_silence: int) -> Optional[np.ndarray]:
        """拼接缓冲，并把句尾静音裁剪到 TAIL_KEEP_FRAMES（保留一点防爆音截断）。"""
        if not frames:
            return None
        cut = max(0, trailing_silence - TAIL_KEEP_FRAMES)
        keep = frames[:-cut] if cut > 0 else frames
        if not keep:
            return None
        return np.concatenate(keep)

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        """缓冲音频写临时 wav → SenseVoice 离线识别（最终提交用，完整识别）。"""
        return self._transcribe_audio_impl(audio, is_incremental=False)

    def _transcribe_audio_fast(self, audio: np.ndarray) -> str:
        """M8: 增量识别用（复用相同逻辑，但可后续优化为更快速率）。"""
        return self._transcribe_audio_impl(audio, is_incremental=True)

    def _transcribe_audio_impl(self, audio: np.ndarray, is_incremental: bool = False) -> str:
        """实际识别实现。"""
        ensure_data_dirs()
        ts = int(time.time() * 1000)
        prefix = "inc_" if is_incremental else "wake_seg_"
        wav_path = DATA_ROOT / "asr_cache" / f"{prefix}{ts}.wav"
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(wav_path), audio, SAMPLE_RATE)

            from src.services.asr_service import get_asr_service
            return get_asr_service().transcribe_file(wav_path, lang="zh")
        except Exception as e:  # noqa: BLE001
            if not is_incremental:
                self._event("WAKE_ERR", f"语音段识别异常：{type(e).__name__}: {e}")
            return ""
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    def _event(self, stage: str, message: str) -> None:
        try:
            self._on_event(stage, message)
        except Exception:  # noqa: BLE001 - 回调异常不影响监听线程
            pass


__all__ = ["WakeWordService", "ListenConfig", "MODE_WAKE", "MODE_COMMAND",
           "CMD_WAIT", "CMD_SPEECH", "CMD_TAIL_WINDOW"]
