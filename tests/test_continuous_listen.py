"""持续监听 + 智能尾点检测（M7 基础版）验收：合成音频帧直接喂状态机，无需麦克风。

覆盖场景：
1. 唤醒词+指令一句话直达（WAKE 模式原有能力不回归）
2. 只喊唤醒词 → 进入持续监听 → 中途停顿 0.6s 不截断 → 尾点 1.2s 提交完整指令
3. 唤醒后 5s 未开口 → LISTEN_PROMPT；再等 3s 无语音 → LISTEN_TIMEOUT 退回环境监听
4. 持续监听中重复喊唤醒词 → 取最后一次出现后的内容（等效清空缓冲重新监听）
5. pause 期间时钟冻结：不计超时、缓冲保留，resume 后继续
6. 最长录音上限强制提交
7. expect_command 等答案窗口（换任务确认链路兼容）

运行：venv_assistant/Scripts/python.exe tests/test_continuous_listen.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.services.wake_word_service import (  # noqa: E402
    FRAME_SAMPLES,
    FRAME_SEC,
    MODE_COMMAND,
    MODE_WAKE,
    WakeWordService,
)

# ---------------- 合成帧 ----------------
RNG = np.random.default_rng(42)


def silence_frames(n: int) -> list[np.ndarray]:
    return [RNG.normal(0, 0.001, FRAME_SAMPLES).astype(np.float32) for _ in range(n)]


def speech_frames(n: int) -> list[np.ndarray]:
    return [RNG.normal(0, 0.08, FRAME_SAMPLES).astype(np.float32) for _ in range(n)]


def sec(sec_: float) -> int:
    return int(sec_ / FRAME_SEC)


class Harness:
    """收集回调 + 脚本化 ASR 返回。"""

    def __init__(self, transcripts: list[str] | None = None) -> None:
        self.wakes: list[str] = []
        self.events: list[tuple[str, str]] = []
        self.transcripts = list(transcripts or [])
        self.submit_count = 0
        self.svc = WakeWordService(
            wake_words=["小助手"],
            on_wake=lambda cmd: self.wakes.append(cmd),
            on_event=lambda stage, msg: self.events.append((stage, msg)),
        )
        self.svc._transcribe_audio = self._fake_transcribe  # type: ignore[assignment]

    def _fake_transcribe(self, audio: np.ndarray) -> str:
        self.submit_count += 1
        return self.transcripts.pop(0) if self.transcripts else ""

    def feed(self, frames: list[np.ndarray]) -> None:
        for f in frames:
            self.svc._feed_frame(f)

    def stages(self) -> list[str]:
        return [s for s, _ in self.events]


# ================= 用例 =================

# ---------- 1. 一句话直达（不回归） ----------
h = Harness(transcripts=["小助手打开抖音"])
h.feed(silence_frames(sec(1.0)))
h.feed(speech_frames(sec(0.8)))
h.feed(silence_frames(sec(1.5)))  # 超过尾点 1.2s → 切段识别
print("RESULT 用例1 wakes:", h.wakes)
assert h.wakes == ["打开抖音"], h.wakes
assert h.svc._mode == MODE_WAKE
print("RESULT 用例1 一句话直达: OK")

# ---------- 2. 只喊唤醒词 → 持续监听 → 中途停顿不截断 → 尾点提交 ----------
h = Harness(transcripts=["小助手", "打开抖音然后顺便再打开知乎"])
h.feed(silence_frames(sec(0.5)))
h.feed(speech_frames(sec(0.5)))   # “小助手”
h.feed(silence_frames(sec(1.5)))  # 尾点切段 → 识别“小助手” → 进入 COMMAND
print("RESULT 用例2 mode after wake-only:", h.svc._mode)
assert h.svc._mode == MODE_COMMAND
assert h.wakes == [""]  # UI 收到“我在”提示

# 用户组织语言 1s（未超 5s 开口超时，不提示）
h.feed(silence_frames(sec(1.0)))
assert "LISTEN_PROMPT" not in h.stages()
# 开口说指令，中间停顿 0.6s（< 尾点 1.2s，不得截断）
h.feed(speech_frames(sec(0.8)))
h.feed(silence_frames(sec(0.6)))
assert h.submit_count == 1, "停顿 0.6s 不应触发提交"
h.feed(speech_frames(sec(0.8)))
# 说完，静音 1.5s ≥ 尾点 → 提交
h.feed(silence_frames(sec(1.5)))
print("RESULT 用例2 wakes:", h.wakes, "| submits:", h.submit_count)
assert h.submit_count == 2
assert h.wakes == ["", "打开抖音然后顺便再打开知乎"], h.wakes
assert h.svc._mode == MODE_WAKE
assert "LISTEN_SUBMIT" in h.stages()
print("RESULT 用例2 中途停顿不截断 + 尾点提交: OK")

# ---------- 3. 5s 未开口提示 + 3s 宽限后超时退出 ----------
h = Harness(transcripts=["小助手"])
h.feed(speech_frames(sec(0.5)))
h.feed(silence_frames(sec(1.5)))  # 触发唤醒，进入 COMMAND
assert h.svc._mode == MODE_COMMAND
h.feed(silence_frames(sec(5.2)))  # 超过 5s 未开口
print("RESULT 用例3 stages:", [s for s in h.stages() if s.startswith("LISTEN")])
assert "LISTEN_PROMPT" in h.stages()
assert h.svc._mode == MODE_COMMAND  # 提示后还有 3s 宽限
h.feed(silence_frames(sec(3.2)))  # 宽限也过了
assert "LISTEN_TIMEOUT" in h.stages()
assert h.svc._mode == MODE_WAKE
print("RESULT 用例3 开口超时提示+退出: OK")

# ---------- 4. 监听中重复喊唤醒词 → 取最后一次之后的内容 ----------
h = Harness(transcripts=["小助手", "小助手小助手打开微博"])
h.feed(speech_frames(sec(0.5)))
h.feed(silence_frames(sec(1.5)))  # 唤醒 → COMMAND
h.feed(speech_frames(sec(1.0)))
h.feed(silence_frames(sec(1.5)))  # 尾点提交“小助手小助手打开微博”
print("RESULT 用例4 wakes:", h.wakes)
assert h.wakes == ["", "打开微博"], h.wakes
assert h.svc._mode == MODE_WAKE
print("RESULT 用例4 重复唤醒取最后: OK")

# ---------- 5. pause 冻结：不计超时、缓冲保留 ----------
h = Harness(transcripts=["小助手", "音量调到百分之五十"])
h.feed(speech_frames(sec(0.5)))
h.feed(silence_frames(sec(1.5)))  # 唤醒 → COMMAND
h.feed(speech_frames(sec(0.4)))   # 开始说
h.svc.pause()                      # TTS/手动场景暂停
h.feed(speech_frames(sec(10.0)))   # 暂停 10 秒（远超各种超时）
assert h.svc._mode == MODE_COMMAND, "pause 期间不应超时退出"
h.svc.resume()
h.feed(silence_frames(sec(0.5)))   # 冷却期 0.4s 丢弃
h.feed(speech_frames(sec(0.6)))    # 继续说
h.feed(silence_frames(sec(1.5)))   # 尾点提交
print("RESULT 用例5 wakes:", h.wakes)
assert h.wakes == ["", "音量调到百分之五十"], h.wakes
print("RESULT 用例5 pause 冻结 + resume 继续: OK")

# ---------- 6. 最长录音上限强制提交 ----------
h = Harness(transcripts=["小助手", "这是一段特别特别长的指令"])
h.feed(speech_frames(sec(0.5)))
h.feed(silence_frames(sec(1.5)))  # 唤醒 → COMMAND
h.feed(speech_frames(sec(31.0)))   # 一直说，超过 30s 上限
print("RESULT 用例6 submits:", h.submit_count, "| wakes:", h.wakes)
assert h.submit_count == 2, "30s 上限应强制提交"
assert h.wakes[-1] == "这是一段特别特别长的指令"
print("RESULT 用例6 最长录音强制提交: OK")

# ---------- 7. expect_command 等答案窗口 ----------
h = Harness(transcripts=["确认"])
h.svc.expect_command(10.0)
h.feed(silence_frames(sec(0.1)))   # 帧循环消费 pending 请求
assert h.svc._mode == MODE_COMMAND
h.feed(silence_frames(sec(2.0)))   # 10s 窗口内，不提示不超时
assert "LISTEN_PROMPT" not in h.stages()
h.feed(speech_frames(sec(0.5)))
h.feed(silence_frames(sec(1.5)))
print("RESULT 用例7 wakes:", h.wakes)
assert h.wakes == ["确认"], h.wakes
# 超时路径：10s 无语音自动退出
h.svc.expect_command(10.0)
h.feed(silence_frames(sec(0.1)))
assert h.svc._mode == MODE_COMMAND
h.feed(silence_frames(sec(10.5)))
assert h.svc._mode == MODE_WAKE
assert h.stages().count("LISTEN_TIMEOUT") == 1
print("RESULT 用例7 expect_command 窗口 + 超时退出: OK")

# ---------- 8. 可调参数生效（尾点 2.0s） ----------
h = Harness(transcripts=["小助手", "先打开抖音"])
h.svc._cfg.tail_silence_sec = 2.0
h.feed(speech_frames(sec(0.5)))
h.feed(silence_frames(sec(2.5)))  # WAKE 模式切段也吃新尾点
assert h.svc._mode == MODE_COMMAND
h.feed(speech_frames(sec(0.5)))
h.feed(silence_frames(sec(1.5)))  # 1.5s < 2.0s 不提交
assert h.submit_count == 1
h.feed(speech_frames(sec(0.3)))
h.feed(silence_frames(sec(2.5)))  # ≥ 2.0s 才提交
assert h.submit_count == 2
assert h.wakes == ["", "先打开抖音"]
print("RESULT 用例8 可调尾点参数生效: OK")

print("ALL PASSED")
