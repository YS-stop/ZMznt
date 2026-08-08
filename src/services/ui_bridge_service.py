"""M5.6 UI 线程桥接层：UIBridgeService（UI ↔ 业务服务解耦，信号槽 + QThread 异步）。

线程模型：
    UI 主线程     → AppController / 所有 QWidget
                  → UIBridgeService（QObject，本文件） → 监听 UI 信号
                  → Worker Thread 调用 AssistantAgent / ASRService / TTSService
                  → 信号回 UI 线程写气泡 + 调试面板 + 弹 P-07 确认 Dialog

关键信号：
    sig_append_ai(text) / sig_append_system(text) / sig_push_debug(stage, msg)
    sig_set_ball_state(state_int)
    sig_tts_speak(text)
    sig_highrisk_request(ops: list[str], required_keyword: str)  → UI 弹确认，线程用 BlockingQueuedConnection 等结果

对外 API：
    bind_app(controller: AppController) 把桥接层贴到 UI 层
    submit_text(text, thread_id) 触发 Agent 推理 + 工具 + 气泡
"""
from __future__ import annotations

import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal, QMetaObject, Q_ARG, Q_RETURN_ARG, Slot, QTimer

import sys
from pathlib import Path

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ui.floating_ball_widget import FloatingBallState  # noqa: E402


@dataclass
class AgentTaskRequest:
    text: str
    thread_id: str
    tts_enabled: bool = True


@dataclass
class AgentTaskResult:
    ok: bool
    final_answer: str = ""
    stages: list[tuple[str, str]] = field(default_factory=list)
    error: str = ""
    ms: int = 0


class _AgentWorker(QObject):
    """QThread Worker：跑 AssistantAgent 任务。"""

    task_started = Signal(str)
    stage = Signal(str, str)            # stage, message → 调试面板
    tool_start = Signal(str, str)       # tool_name, args
    tool_end = Signal(str, str)         # tool_name, observation
    final = Signal(str, object)         # final_answer, AgentTaskResult

    def __init__(self, req: AgentTaskRequest) -> None:
        super().__init__()
        self.req = req

    @Slot()
    def run(self) -> None:
        t0 = time.perf_counter_ns()
        req = self.req
        self.task_started.emit(req.thread_id)
        self.stage.emit("AGENT", f"开始处理用户指令（thread_id={req.thread_id}）：{req.text[:80]}")
        try:
            # 调用 AgentService：单轮推理（内部用 LangGraph + 8 Tools + Checkpoint）
            from src.services.agent_service import get_assistant_agent  # 延迟导入避免循环
            agent = get_assistant_agent()
            self.stage.emit("AGENT", f"构建 LangGraph ReAct…")

            def _on_stream(stage_name: str, data: Any) -> None:
                """stream_events 的流式回调（本 worker 线程内执行，经 Qt 信号跨线程投递到 UI）。

                阶段：human / ai / tool_pre(list[dict]) / tool(ToolMessage) / done(str)
                """
                try:
                    if stage_name == "tool_pre" and isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                self.tool_start.emit(
                                    str(item.get("name", "?")),
                                    str(item.get("args", ""))[:300],
                                )
                    elif stage_name == "tool":
                        name = str(getattr(data, "name", None) or "?")
                        obs = str(getattr(data, "content", data) or "")
                        self.tool_end.emit(name, obs[:600])
                    elif stage_name == "ai":
                        tcs = getattr(data, "tool_calls", None) or []
                        if tcs:
                            self.stage.emit("AGENT", f"AI 决定调用 {len(tcs)} 个工具，等待执行…")
                except Exception:  # noqa: BLE001 - 回调出错不影响主流程
                    pass

            try:
                # stream_events 是「回调式」API：传入 stream_cb 接收事件，返回值即最终回答
                final_txt = agent.stream_events(
                    req.text,
                    thread_id=req.thread_id,
                    stream_cb=_on_stream,
                ) or ""
                # 兜底：没有拿到最终回答就从图的最后状态里取
                if not final_txt.strip():
                    final_txt = _extract_final_from_graph_last(agent, req.thread_id)
            except Exception as e:  # 内层 agent 异常：捕获转友好
                self.stage.emit("ERR", f"Agent 异常：{type(e).__name__}: {e}")
                final_txt = (
                    f"抱歉，处理时出现异常：{type(e).__name__}。\n"
                    f"提示：{e}\n"
                    "可以检查 API Key / 网络 / 工具权限。"
                )
            # 结果
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            res = AgentTaskResult(ok=True, final_answer=final_txt, stages=[], ms=ms)
            self.stage.emit("AGENT", f"Agent 完成，耗时 {ms} ms。")
            self.final.emit(final_txt, res)
        except Exception as outer:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            err = f"{type(outer).__name__}: {outer}\n{traceback.format_exc(limit=6)}"
            self.stage.emit("ERR", err)
            final_txt = f"抱歉，出现未捕获异常：{type(outer).__name__}：{outer}"
            res = AgentTaskResult(ok=False, final_answer=final_txt, error=err, ms=ms)
            self.final.emit(final_txt, res)


def _extract_final_from_graph_last(agent, thread_id: str) -> str:
    """兜底从 AgentService 获取 thread 最后一条消息。"""
    try:
        state = agent.last_state(thread_id=thread_id)
        if isinstance(state, dict):
            msgs = state.get("messages") if isinstance(state, dict) else None
            if isinstance(msgs, list) and msgs:
                from langchain_core.messages import AIMessage
                last = msgs[-1]
                if isinstance(last, AIMessage):
                    return str(getattr(last, "content", "") or "")
    except Exception:
        return ""
    return ""


class _VoiceRecordWorker(QObject):
    """QThread Worker：ASR 录音+转写。完成后返回 str 文本。"""

    stage = Signal(str, str)
    finished = Signal(str)  # 识别结果（空表示未识别，含 ❌/⚠️ 前缀）

    def __init__(self, duration: float = 5.0, save_path: Optional[str] = None) -> None:
        super().__init__()
        self.duration = float(duration)
        self.save_path = save_path

    @Slot()
    def run(self) -> None:
        self.stage.emit("ASR", f"启动录音（{self.duration:.1f}s）+ SenseVoice 离线识别…")
        try:
            from src.services.asr_service import asr_record_and_transcribe
            text = asr_record_and_transcribe(
                duration=self.duration,
                samplerate=16000,
                channels=1,
                device=None,
                save_path=self.save_path,
            )
            self.stage.emit("ASR", f"识别完成：{text[:160]}")
            self.finished.emit(text or "")
        except Exception as e:
            self.stage.emit("ERR", f"ASR Worker 异常：{type(e).__name__}: {e}")
            self.finished.emit(f"⚠️ ASR 异常：{type(e).__name__}: {e}")


class _TTSWorker(QObject):
    speak_started = Signal(str)
    stage = Signal(str, str)
    finished = Signal(str)  # 完成 note

    def __init__(self, text: str, voice: str = "default", speed: float = 1.0, play_now: bool = True) -> None:
        super().__init__()
        self.text = text
        self.voice = voice
        self.speed = float(speed)
        self.play_now = bool(play_now)

    @Slot()
    def run(self) -> None:
        if not self.text.strip():
            self.finished.emit("(空文本，已跳过 TTS)")
            return
        self.speak_started.emit(self.text)
        try:
            from src.services.tts_service import tts_speak
            res = tts_speak(text=self.text, voice=self.voice, speed=self.speed, save_path=None, play_now=self.play_now)
            ok = bool(res.get("ok"))
            note = res.get("play_note", "") or str(res.get("error", ""))
            path = res.get("audio_path") or ""
            msg = f"TTS：{'合成成功' if ok else '失败'}，大小 {res.get('audio_size_bytes', 0)} 字节，保存：{path or '-'}"
            self.stage.emit("TTS", msg)
            self.finished.emit(note)
        except Exception as e:
            self.stage.emit("ERR", f"TTS 异常：{type(e).__name__}: {e}")
            self.finished.emit(f"⚠️ TTS 异常：{type(e).__name__}: {e}")


# ============================================================
# UIBridgeService 主类
# ============================================================

class UIBridgeService(QObject):
    """UI ↔ 业务的解耦桥梁（单例）。"""

    # --- UI 控制信号（发给主线程） ---
    sig_append_user_bubble = Signal(str)
    sig_append_ai_bubble = Signal(str)
    sig_append_system_bubble = Signal(str)
    sig_push_debug = Signal(str, str)
    sig_set_ball_state = Signal(int)
    sig_open_settings = Signal()

    # --- 高危确认（跨线程阻塞式） ---
    sig_request_highrisk_confirm = Signal(list, str)   # ops_list, required_keyword；UI 返回 (bool, keyword)

    # --- 唤醒词（常驻监听线程 → UI 线程） ---
    sig_wake_command = Signal(str)                     # 唤醒成功后的指令文本（空串=只喊了唤醒词）

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._controller = None
        self._threads: list[QThread] = []    # 保留引用避免 GC
        self._current_tts_enabled: bool = True
        self._current_tts_voice: str = "default"
        self._current_tts_speed: float = 1.0
        self._pending_voice_duration: float = 5.0
        self._wake_svc = None                # WakeWordService | None

        # —— 任务打断确认状态机 ——
        self._task_busy: bool = False              # 有 Agent 任务正在执行
        self._current_task_text: str = ""          # 当前任务原文（用于询问时复述）
        self._task_generation: int = 0             # 代际号：新任务 +1，旧任务结果按代际丢弃
        self._pending_task_text: Optional[str] = None   # 缓冲的新指令（等待用户确认）
        self._switch_ask_id: int = 0               # 每次询问 +1，防超时回调串台
        self._awaiting_switch_answer: bool = False # 正在等「换不换任务」的回答
        self._expect_answer_after_tts: bool = False    # TTS 询问播完后开启免唤醒词等答案窗口

    # ------------------------------------------------------------------
    # 绑定 UI 层
    # ------------------------------------------------------------------

    def bind_app(self, controller) -> None:
        """绑定 AppController，信号槽全部接好。"""
        if controller is None:
            return
        self._controller = controller

        # ← UI 输入信号
        controller.sig_user_submit_text.connect(self.submit_text)
        controller.sig_user_voice_start.connect(self.on_voice_start)
        controller.sig_user_voice_end.connect(self.on_voice_end)
        controller.sig_user_tts_toggled.connect(self.on_tts_toggled)
        controller.sig_request_settings_saved.connect(self.on_settings_saved)
        controller.sig_app_quit.connect(self.on_app_quit)

        # → UI 渲染信号
        self.sig_append_user_bubble.connect(lambda t: controller.append_user_bubble(t))
        self.sig_append_ai_bubble.connect(lambda t: controller.append_ai_bubble(t))
        self.sig_append_system_bubble.connect(lambda t: controller.append_system_bubble(t))
        self.sig_push_debug.connect(lambda stage, msg: controller.push_debug(stage, msg))
        self.sig_set_ball_state.connect(lambda s: controller.set_ball_state(FloatingBallState(int(s))))

        # 初始状态
        self._set_input_busy(False)
        self.sig_push_debug.emit("SYSTEM", "✅ UIBridgeService 已挂载：文本 / 语音 / TTS / Agent 链路就绪。")

        # 唤醒词常驻监听（settings.wake_word_enabled=false 可关闭）
        self.sig_wake_command.connect(self._on_wake_command)
        try:
            from src.services.sqlite_db import get_setting
            if bool(get_setting("wake_word_enabled", True)):
                self._start_wake_service()
            else:
                self.sig_push_debug.emit("WAKE", "唤醒词监听已禁用（settings.wake_word_enabled=false）。")
        except Exception as e:  # noqa: BLE001 - 唤醒启动失败不影响主功能
            self.sig_push_debug.emit("ERR", f"唤醒服务启动失败：{type(e).__name__}: {e}")

    def _set_input_busy(self, busy: bool) -> None:
        """禁用输入栏（防连点 3 次重复 warning），controller 未绑定时静默跳过。"""
        c = self._controller
        if c is None:
            return
        try:
            panel = getattr(c, "panel", None)
            if panel is not None and hasattr(panel, "set_input_busy"):
                panel.set_input_busy(bool(busy))
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # 对外：提交文本 → Agent
    # ------------------------------------------------------------------

    def submit_text(self, text: str, thread_id: Optional[str] = None) -> None:
        content = (text or "").strip()
        if not content:
            return
        # 0. 正在等「换不换任务」的回答 → 本次输入是回答，不是新任务
        if self._awaiting_switch_answer:
            self._handle_switch_answer(content)
            return
        # 1. 有任务在跑 → 缓冲新指令，先询问用户是否中止当前任务
        if self._task_busy:
            self._pending_task_text = content
            self._ask_task_switch(content)
            return
        # 2. 空闲 → 正常启动
        self._start_task(content, thread_id)

    def _start_task(self, content: str, thread_id: Optional[str] = None) -> None:
        """真正启动一个 Agent 任务（submit_text 路由后的唯一入口）。"""
        # 🛡 防连点：开始提交 → 先禁用输入栏（即使之前因 ASR 被禁用了也没关系）
        self._set_input_busy(True)
        self._task_busy = True
        self._current_task_text = content
        self._task_generation += 1
        gen = self._task_generation
        # 🔊 掐断上一任务可能还在播放的 TTS 播报
        try:
            from src.services.tts_service import tts_stop_playback
            tts_stop_playback()
        except Exception:  # noqa: BLE001
            pass
        tid = thread_id or ("th_" + uuid.uuid4().hex[:8])
        self.sig_set_ball_state.emit(int(FloatingBallState.THINKING))
        self.sig_push_debug.emit("AGENT", f"收到文本指令（thread_id={tid}，gen={gen}）")

        # QThread Worker
        thread = QThread(self)
        worker = _AgentWorker(AgentTaskRequest(text=content, thread_id=tid, tts_enabled=self._current_tts_enabled))
        worker.moveToThread(thread)

        # 工作线程生命周期
        thread.started.connect(worker.run)
        worker.task_started.connect(lambda _id, w=worker: self.sig_push_debug.emit("AGENT", f"任务线程已启动（tid={_id}）"))
        worker.stage.connect(lambda stage, msg, w=worker: self.sig_push_debug.emit(stage, msg))
        worker.tool_start.connect(lambda name, args, w=worker: self.sig_push_debug.emit("TOOL", f"▶ {name}({args[:200]})"))
        worker.tool_end.connect(lambda name, obs, w=worker: self.sig_push_debug.emit("OBS", f"◀ {name}: {obs[:400]}"))
        worker.final.connect(lambda ans, res, g=gen: self._on_agent_final(ans, res, g))
        worker.final.connect(lambda _a, _r, t=thread, w=worker: _cleanup_thread(t, w))
        self._threads.append(thread)
        thread.start()

    # ------------------------------------------------------------------
    # 对外：语音开始/结束
    # ------------------------------------------------------------------

    def on_voice_start(self) -> None:
        # 手动按住说话期间也暂停唤醒监听（避免两个录音流抢麦克风）
        self._wake_pause()
        # 由 AppController 直接切 LISTENING 状态并写入调试面板，这里不重复做
        self.sig_push_debug.emit("ASR", "（UIBridge）检测到语音开始信号，等待松开…")

    def on_voice_end(self) -> None:
        # 🛡 防连点：ASR 开始前 → 禁用输入栏（录音+识别过程不可连点）
        self._set_input_busy(True)
        self.sig_push_debug.emit("ASR", "（UIBridge）开始录音+识别工作线程…")
        # 启动 ASR worker
        thread = QThread(self)
        worker = _VoiceRecordWorker(duration=self._pending_voice_duration, save_path=None)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.stage.connect(lambda stage, msg: self.sig_push_debug.emit(stage, msg))
        worker.finished.connect(self._on_asr_finished)
        worker.finished.connect(lambda _t, th=thread, w=worker: _cleanup_thread(th, w))
        self._threads.append(thread)
        thread.start()

    def on_tts_toggled(self, on: bool) -> None:
        self._current_tts_enabled = bool(on)
        self.sig_push_debug.emit("TTS", f"用户设置 TTS = {on}")

    def on_settings_saved(self, key: str, val: object) -> None:
        self.sig_push_debug.emit("SYSTEM", f"设置保存：{key}={val}")

    def on_app_quit(self) -> None:
        self.sig_push_debug.emit("SYSTEM", "收到 app 退出信号。桥接层清理线程…")
        # 停掉唤醒监听线程（释放麦克风）
        svc = self._wake_svc
        if svc is not None:
            try:
                svc.stop()
            except Exception:  # noqa: BLE001
                pass
            self._wake_svc = None

    # ------------------------------------------------------------------
    # 内部回调
    # ------------------------------------------------------------------

    def _on_agent_final(self, answer: str, result: AgentTaskResult, generation: int = -1) -> None:
        # 🛡 代际守卫：被取代的旧任务结果直接丢弃（不弹气泡、不播报、不清 busy）
        if generation >= 0 and generation != self._task_generation:
            self.sig_push_debug.emit(
                "AGENT",
                f"丢弃已被取代任务的结果（gen={generation}，当前 gen={self._task_generation}）：{(answer or '')[:60]}",
            )
            return
        # 🛡 任务结束：恢复输入栏（即使 TTS 仍在播报，也允许用户继续输入）
        self._set_input_busy(False)
        self._task_busy = False
        self._current_task_text = ""
        # 回到 IDLE
        self.sig_set_ball_state.emit(int(FloatingBallState.IDLE))
        # 气泡
        final = (answer or "").strip()
        if not final:
            final = "(Agent 未返回有效答案，可能是工具链还在执行或模型超时。)"
            self.sig_push_debug.emit("WARN", "Agent 最终答案为空，已用占位气泡提示。")
        self.sig_append_ai_bubble.emit(final)
        self.sig_push_debug.emit("AGENT", f"最终回答（{len(final)} 字符，{result.ms} ms）：{final[:220]}")

        # TTS
        if self._current_tts_enabled:
            self._run_tts(final, self._current_tts_voice, self._current_tts_speed)
        else:
            # 不播报 → 立即恢复唤醒监听
            self._wake_resume()

    def _on_asr_finished(self, text: str) -> None:
        content = (text or "").strip()
        if not content or content.startswith("❌") or content.startswith("⚠️"):
            # ❌ ASR 失败：先恢复输入栏可用 → 再提示用户重试（否则按钮一直禁用）
            self._set_input_busy(False)
            self._wake_resume()
            self.sig_append_system_bubble.emit(content or "（未识别到任何语音，请重试）")
            return
        # ✅ 识别成功 → submit_text 会把 busy 再拉成 True，最终 _on_agent_final 恢复
        self.sig_append_user_bubble.emit(content)
        self.submit_text(content)

    # ------------------------------------------------------------------
    # 唤醒词常驻监听（WakeWordService 生命周期管理）
    # ------------------------------------------------------------------

    def _start_wake_service(self) -> None:
        """按设置启动唤醒监听（ASR 引擎不可用时服务会自己打日志并退出）。

        注：torch/funasr 等原生依赖已在 main.py 创建 QApplication 之前预加载
        （PySide6 与 torch 的 DLL 加载顺序冲突会段错误），此处只是拿已就绪的单例。
        """
        from src.services.asr_service import get_asr_service
        get_asr_service()  # 主线程构造 ASR 单例（依赖探测在此完成）

        from src.services.wake_word_service import WakeWordService
        from src.services.sqlite_db import get_setting

        word = str(get_setting("wake_word", "小助手") or "小助手")
        svc = WakeWordService(
            wake_words=[word],
            on_wake=lambda cmd: self.sig_wake_command.emit(cmd),
            on_event=lambda stage, msg: self.sig_push_debug.emit(stage, msg),
        )
        svc.start()
        self._wake_svc = svc

    def _wake_pause(self) -> None:
        svc = self._wake_svc
        if svc is not None:
            try:
                svc.pause()
            except Exception:  # noqa: BLE001
                pass

    def _wake_resume(self) -> None:
        svc = self._wake_svc
        if svc is not None:
            try:
                svc.resume()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # 任务打断确认状态机（执行中来新命令 → 缓冲 → 询问 → 确认后才替换）
    # ------------------------------------------------------------------

    def _ask_task_switch(self, new_cmd: str) -> None:
        """任务执行中收到新指令：缓冲 + 气泡/语音询问用户是否中止当前任务。"""
        self._switch_ask_id += 1
        ask_id = self._switch_ask_id
        self._awaiting_switch_answer = True
        cur = self._short(self._current_task_text or "当前任务")
        new = self._short(new_cmd)
        question = f"正在执行：{cur}。收到新指令：{new}。要中止当前任务、执行新指令吗？请说「确认」或「继续」。"
        self.sig_push_debug.emit("AGENT", f"任务冲突：缓冲新指令「{new_cmd[:40]}」，等待用户确认（ask_id={ask_id}）。")
        self.sig_append_system_bubble.emit(f"🔀 {question}")
        # 询问期间允许用户在输入框打字回答（任务运行中输入栏默认是禁用的）
        self._set_input_busy(False)
        if self._current_tts_enabled:
            self._expect_answer_after_tts = True
            self._run_tts(
                f"正在执行{cur}。收到新指令{new}。要中止当前任务，执行新指令吗？请说确认，或继续。",
                self._current_tts_voice,
                self._current_tts_speed,
            )
        else:
            self._open_answer_window()
        # 12 秒无回答 → 默认保留当前任务
        QTimer.singleShot(12000, lambda aid=ask_id: self._on_switch_answer_timeout(aid))

    def _handle_switch_answer(self, text: str) -> None:
        """处理「换不换任务」的回答：确认→替换任务；否定/不明确→保留当前任务。"""
        self._awaiting_switch_answer = False
        verdict = _classify_switch_answer(text)
        pending = self._pending_task_text
        self.sig_push_debug.emit("AGENT", f"任务切换回答「{text[:40]}」→ 判定：{verdict}")
        if verdict == "confirm" and pending:
            self._pending_task_text = None
            self.sig_append_system_bubble.emit(f"✅ 收到，中止当前任务，开始执行：{self._short(pending, 40)}")
            # 旧任务还在跑也没关系：结果会被代际守卫丢弃，播报会被 _start_task 掐断
            self._start_task(pending)
        elif verdict == "deny":
            self._pending_task_text = None
            self.sig_append_system_bubble.emit("👌 好的，继续执行当前任务，新指令已忽略。")
        else:
            # 没听清/不明确：保守策略——不换任务（用户可随时再说一次新指令重新触发询问）
            self._pending_task_text = None
            self.sig_append_system_bubble.emit(
                "🤔 没听清您的选择，默认继续执行当前任务。如需中断，请直接说出新指令。"
            )

    def _on_switch_answer_timeout(self, ask_id: int) -> None:
        """确认询问 12 秒超时：默认保留当前任务，丢弃缓冲指令。"""
        if not self._awaiting_switch_answer or ask_id != self._switch_ask_id:
            return
        self._awaiting_switch_answer = False
        dropped = self._pending_task_text or ""
        self._pending_task_text = None
        self.sig_push_debug.emit("AGENT", f"任务切换确认超时（12s 无回答），丢弃缓冲指令：{dropped[:40]}")
        self.sig_append_system_bubble.emit("⏱️ 等待确认超时，继续执行当前任务。")

    def _maybe_open_answer_window(self) -> None:
        """TTS 播完后的钩子：若刚播的是切换询问 → 开启免唤醒词等答案窗口。"""
        if not self._expect_answer_after_tts:
            return
        self._expect_answer_after_tts = False
        self._open_answer_window()

    def _open_answer_window(self) -> None:
        """开启 10 秒免唤醒词等答案窗口（用户的回答直接说「确认/继续」即可）。"""
        svc = self._wake_svc
        if svc is not None:
            try:
                svc.expect_command(10.0)
                self.sig_push_debug.emit("WAKE", "已开启等答案窗口（10s 内直接说「确认/继续」，无需唤醒词）。")
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _short(text: str, n: int = 20) -> str:
        t = (text or "").strip()
        return t if len(t) <= n else t[:n] + "…"

    def _on_wake_command(self, command: str) -> None:
        """唤醒成功（UI 线程槽函数）：空指令只提示，带指令直接走 Agent。"""
        cmd = (command or "").strip()
        self.sig_push_debug.emit("WAKE", f"唤醒回调收到指令：{cmd or '(空)'}")
        if not cmd:
            # 只喊了「小助手」→ 提示用户在听（服务侧 8s 窗口内下一句话会自动再来一次本回调）
            self.sig_set_ball_state.emit(int(FloatingBallState.LISTENING))
            self.sig_append_system_bubble.emit("🎙️ 我在，请说指令…")
            return
        self.sig_append_user_bubble.emit(f"🎙️ {cmd}")
        self.submit_text(cmd)

    def _run_tts(self, text: str, voice: str, speed: float) -> None:
        # 切播报状态 + 播报期间暂停唤醒监听（防自家喇叭回声自触发）
        self.sig_set_ball_state.emit(int(FloatingBallState.SPEAKING))
        self._wake_pause()
        self.sig_push_debug.emit("TTS", f"播报 {len(text)} 字符，voice={voice}, speed={speed}x")
        thread = QThread(self)
        worker = _TTSWorker(text=text, voice=voice, speed=speed, play_now=True)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.speak_started.connect(lambda t: self.sig_push_debug.emit("TTS", f"开始播放：{t[:120]}"))
        worker.stage.connect(lambda stage, msg: self.sig_push_debug.emit(stage, msg))
        worker.finished.connect(lambda note: self.sig_push_debug.emit("TTS", f"播报结束：{note}"))
        worker.finished.connect(lambda _n, th=thread, w=worker: _cleanup_thread(th, w))
        # 播报完回 IDLE + 恢复唤醒监听（此时喇叭已安静，不会自触发）
        worker.finished.connect(lambda _n, self_ref=self: self_ref.sig_set_ball_state.emit(int(FloatingBallState.IDLE)))
        worker.finished.connect(lambda _n: self._wake_resume())
        # 若这次播报是「换任务确认询问」→ 播完开启免唤醒词等答案窗口
        worker.finished.connect(lambda _n: self._maybe_open_answer_window())
        self._threads.append(thread)
        thread.start()


# ------------------------------------------------------------------
# 任务切换回答分类器（纯函数，便于单测）
# ------------------------------------------------------------------
_SWITCH_STRONG_CONFIRM = ("取消当前", "中止当前", "终止当前", "停掉当前", "放弃当前", "执行新", "换任务", "换指令")
_SWITCH_CONFIRM_WORDS = ("确认", "是的", "确定", "可以", "好", "行", "嗯", "对", "换", "停", "要")
_SWITCH_DENY_WORDS = ("继续", "不用", "不要", "否", "保持", "算了", "取消", "别换", "不换", "忽略", "按原", "接着做", "先做完")


def _classify_switch_answer(text: str) -> str:
    """把用户对「要中止当前任务执行新指令吗」的回答分类为 confirm / deny / unclear。

    判定顺序（防歧义）：
        1. 强确认短语（「取消当前任务」这类，含"取消"但意思是换任务）
        2. 「不/别/没」开头一律否定
        3. 否定词命中 → deny
        4. 确认词命中 → confirm
        5. 都不沾 → unclear（上层按「不换」保守处理）
    """
    t = re.sub(r"[\s，。,.!！?？、~～…]+", "", text or "")
    if not t:
        return "unclear"
    for w in _SWITCH_STRONG_CONFIRM:
        if w in t:
            return "confirm"
    if t.startswith(("不", "别", "没")):
        return "deny"
    for w in _SWITCH_DENY_WORDS:
        if w in t:
            return "deny"
    for w in _SWITCH_CONFIRM_WORDS:
        if w in t:
            return "confirm"
    return "unclear"


def _cleanup_thread(thread: QThread, worker: QObject) -> None:
    """QThread + Worker 清理标准流程。"""
    try:
        worker.deleteLater()
    except Exception:
        pass
    try:
        thread.quit()
        if not thread.wait(300):
            thread.terminate()
            thread.wait(200)
    except Exception:
        pass
    try:
        thread.deleteLater()
    except Exception:
        pass


# 模块级单例
_BRIDGE = UIBridgeService()


def get_ui_bridge() -> UIBridgeService:
    return _BRIDGE


__all__ = [
    "UIBridgeService",
    "AgentTaskRequest",
    "AgentTaskResult",
    "get_ui_bridge",
]
