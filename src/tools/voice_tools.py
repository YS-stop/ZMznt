"""语音工具（LangChain BaseTool 子类）。

工具：
    1. VoiceInputTool  —— 「ASR 语音输入」：录制 N 秒语音 → SenseVoice 转文字 → 返回识别结果
                           （无麦克风/缺 ASR 依赖时可通过 force_text 旁路，直接返回文本）
    2. VoiceOutputTool —— 「TTS 语音播报」：**M4 占位**，当前返回文字提示，暂不发声；M4 阶段接入 Edge-TTS/CosyVoice。

设计要点：
    - 所有异常都包装成中文 observation 返回，不向上抛（LangGraph 循环继续）。
    - VoiceInputTool 参数含 force_text：提供后直接跳过录音/模型，等价于打字输入，便于测试与离线演示。
    - VoiceOutputTool 目前是占位接口（保证对外工具签名稳定），后续 M4 内部替换实现即可。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

_SRC_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from langchain_core.tools import BaseTool  # noqa: E402


# ============================================================
# 参数 Schema
# ============================================================

class VoiceInputArgs(BaseModel):
    duration: float = Field(
        5.0,
        ge=0.5,
        le=60.0,
        description="【选填】录音时长，单位秒（0.5~60，默认 5 秒）。",
    )
    samplerate: int = Field(
        16000,
        ge=8000,
        le=48000,
        description="【选填】采样率，默认 16000（SenseVoice 最佳），可选 8000/16000/22050/44100/48000。",
    )
    channels: int = Field(
        1,
        ge=1,
        le=2,
        description="【选填】声道数：1=单声道（默认），2=立体声。单声道识别更稳且速度更快。",
    )
    lang: str = Field(
        "auto",
        description=(
            "【选填】识别语言：auto（自动识别，默认）/ zh（中文）/ en（英文）/ yue（粤语）/ ja（日语）。"
            "若明确知道用户说哪种语言，指定该字段可提高准确率。"
        ),
    )
    save_recording: bool = Field(
        True,
        description="【选填】True=把录音保存为 data/asr_cache/record_xxx.wav（便于回溯），默认 True。",
    )
    device: int | None = Field(
        None,
        description=(
            "【选填】麦克风设备索引（None=系统默认）。"
            "可用 python -m sounddevice 查询本机麦克风列表，再把整数 index 传进来。"
        ),
    )
    force_text: str | None = Field(
        None,
        description=(
            "【选填】旁路文字：不为空时直接跳过录音+ASR，直接返回该文本。"
            "典型场景：① 测试/无依赖的演示环境；② 用户没装好麦克风时打字代替语音。"
            "高级场景：桌面 UI 层已通过 ASR 线程提前识别好文字，直接通过本参数传给 Agent 工具链。"
        ),
    )


class VoiceOutputArgs(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="【必填】要朗读/播报的文本，最多 5000 字符。",
    )
    voice: str = Field(
        "default",
        description=(
            "【选填】TTS 音色：default=晓晓女声（Edge-TTS zh-CN-XiaoxiaoNeural），"
            "也可传别名：xiaoxiao/yunxi(男)/news(云健新闻)/cantonese(粤语)/en(英文Aria)，"
            "或直接传 Edge-TTS ShortName 如 zh-CN-YunjianNeural。"
        ),
    )
    speed: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="【选填】语速：0.5~2.0，默认 1.0（Edge-TTS 会换算成 +50%/-50% 档位）。",
    )
    save_path: str | None = Field(
        None,
        description=(
            "【选填】音频保存路径（留痕/二次播放）："
            "None 默认自动保存到 data/tts_cache/tts_时间戳.mp3；"
            "指定路径时会新建父目录，建议 mp3 后缀。"
        ),
    )
    play_now: bool = Field(
        True,
        description=(
            "【选填】True=合成后立即调用系统播放器播报（默认）；"
            "False=只合成保存音频不播放（UI 层留痕或稍后再播放时用）。"
        ),
    )


# ============================================================
# 1. VoiceInputTool —— ASR
# ============================================================

class VoiceInputTool(BaseTool):
    """语音转文字工具：录音 N 秒 → SenseVoiceSmall（CPU/GPU 均可离线识别 中/英/粤/日）。

    无依赖或无麦克风时可通过 force_text 打字旁路，避免把 LangGraph 卡死在音频 IO。
    """

    name: ClassVar[str] = "voice_input"
    description: ClassVar[str] = (
        "Tool Name: voice_input\n"
        "用途：录音后做离线语音识别（ASR），把用户说的话转为文字，再交给 Agent 继续处理。\n"
        "典型场景：\n"
        "  - 用户点击「按住说话」后松开：调 duration=3~5，保存录音并识别。\n"
        "  - 无麦克风环境或 ASR 依赖未装：传 force_text=「用户实际打字内容」跳过音频阶段。\n"
        "  - UI 层已经提前完成 ASR：把识别结果通过 force_text 直接传进来复用。\n"
        "引擎策略（asr_service 内部三级降级）：\n"
        "  ① SenseVoiceSmall 离线本地识别（推荐，免费无外网）\n"
        "  ② 缺 torch/funasr/modelscope → 降级「纯文本模式」并返回安装提示\n"
        "说明：\n"
        "  - duration 默认 5 秒，足够一句话指令；如需长句建议拆成多轮。\n"
        "  - lang=auto 会自动判别语种，识别不出来可手动指定 zh/en/yue/ja。\n"
        "  - save_recording=True 会把音频保存到 data/asr_cache，后续可回放排查识别错误。\n"
        "  - force_text 不为空时，以上所有参数都被忽略，直接返回该字符串（零耗时）。\n"
    )
    args_schema: type[BaseModel] = VoiceInputArgs
    return_direct: ClassVar[bool] = False

    def _run(  # noqa: D401
        self,
        duration: float = 5.0,
        samplerate: int = 16000,
        channels: int = 1,
        lang: str = "auto",
        save_recording: bool = True,
        device: int | None = None,
        force_text: str | None = None,
    ) -> str:
        t0 = time.perf_counter_ns()
        try:
            # 0. 旁路：force_text 直接返回
            if force_text is not None and str(force_text).strip() != "":
                text = str(force_text).strip()
                ms = (time.perf_counter_ns() - t0) // 1_000_000
                return (
                    f"🎙️ voice_input 完成（force_text 旁路，未录音未调用 ASR，{ms} ms）\n"
                    f"  识别文本：{text}"
                )

            # 1. 导入 asr_service（此处延迟导入避免循环依赖）
            from src.services.asr_service import asr_record_and_transcribe, get_asr_info

            # 2. 如果要保存录音 → 生成路径；否则临时文件后不保存
            save_p: str | None = None
            if save_recording:
                from src.utils.path_utils import DATA_ROOT  # noqa: E402
                ts = int(time.time() * 1000)
                target_dir = DATA_ROOT / "asr_cache"
                target_dir.mkdir(parents=True, exist_ok=True)
                save_p = str(target_dir / f"voice_input_{ts}.wav")

            # 3. 调用服务：录音 + 转写
            text = asr_record_and_transcribe(
                duration=duration,
                samplerate=samplerate,
                channels=channels,
                device=device,
                save_path=save_p,
                force_text=None,  # 上面已处理过 force_text 分支，这里必须 None
            )

            ms = (time.perf_counter_ns() - t0) // 1_000_000
            # 根据内容包装成 observation
            if text.startswith("❌") or text.startswith("⚠️"):
                # 已经是错误/提示前缀，直接加个 head
                info = get_asr_info()
                engine = info.get("backend", info.get("engine", "?"))
                return f"🎙️ voice_input（{ms} ms，engine={engine}）\n{text}"

            # 正常识别
            info = get_asr_info()
            engine = info.get("backend", info.get("engine", "?"))
            return (
                f"🎙️ voice_input 完成（{ms} ms，engine={engine}，duration={duration}s，sr={samplerate}Hz）\n"
                f"  识别文本：{text}\n"
                + (f"  录音保存：{save_p}" if save_p else "")
            )
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ voice_input 失败（{ms} ms）：{type(e).__name__}: {e}"


# ============================================================
# 2. VoiceOutputTool —— TTS（M4 占位，不发声，只返回结构化提示）
# ============================================================

class VoiceOutputTool(BaseTool):
    """文字转语音工具：Edge-TTS 在线（微软免费、无需 Key），本地 CosyVoice 占位。

    真实行为：
        - 合成 mp3 落盘（默认保存到 data/tts_cache/tts_时间戳.mp3，留痕可回放）
        - play_now=True（默认）会调用系统播放器播报，失败就告诉用户手动打开文件
        - CosyVoice 本地离线：M4 占位，后续装 cosyvoice 包即可启用，签名不变
    """

    name: ClassVar[str] = "voice_output"
    description: ClassVar[str] = (
        "Tool Name: voice_output\n"
        "用途：把 Agent 的文本回答朗读出来（TTS，Text To Speech），方便用户不用看屏幕。\n"
        "引擎策略（tts_service 内部降级）：\n"
        "  ① Edge-TTS（微软免费在线、无需 Key、中文多音色 ✅ M4 已启用\n"
        "      默认 voice=default 内置 xiaoxiao/yunxi/news 等 8 个常用别名，自动识别 Edge-TTS ShortName\n"
        "  ② CosyVoice（阿里本地离线，**占位，需额外装 cosyvoice+modelscope+torch）\n"
        "典型场景：\n"
        "  - 用户说「读出来」「播报一下」「用女声读」→ 调用本工具朗读。\n"
        "  - 语音助手完整闭环：voice_input → Agent 推理 → voice_output。\n"
        "参数：\n"
        "  - text（必填）：要播报的文字，最多 5000 字符（超长自动截断）。\n"
        "  - voice：default/xiaoxiao(晓晓女声/yunxi(男)/news(新闻男声)/cantonese(粤语)/en(英文)，或 Edge-TTS ShortName。\n"
        "  - speed：语速 0.5~2.0，默认 1.0。\n"
        "  - save_path：音频落盘路径（留痕/稍后播放），None 时自动保存到 data/tts_cache/tts_xxx.mp3。\n"
        "  - play_now：True=合成后立即调用系统播放器播报（默认），False=只合成落盘。\n"
        "失败策略：\n"
        "  * 合成失败（无外网/edge-tts 抛错）→ 返回中文友好提示 + 原文字，不抛异常，LangGraph 循环继续。\n"
        "  * 播放失败（没装 playsound/pygame 等）→ 合成仍成功，音频在本地，手动打开即可。\n"
    )
    args_schema: type[BaseModel] = VoiceOutputArgs
    return_direct: ClassVar[bool] = False

    def _run(  # noqa: D401
        self,
        text: str,
        voice: str = "default",
        speed: float = 1.0,
        save_path: str | None = None,
        play_now: bool = True,
    ) -> str:
        t0 = time.perf_counter_ns()
        try:
            content = (text or "").strip()
            if not content:
                raise ValueError("text 不能为空")

            # 延迟导入避免循环依赖
            from src.services.tts_service import tts_speak, get_tts_info

            res = tts_speak(
                text=content,
                voice=voice,
                speed=speed,
                save_path=save_path,
                play_now=bool(play_now),
            )
            ms = (time.perf_counter_ns() - t0) // 1_000_000

            # 结构化返回（Observation 格式）
            engine = res.get("engine", "?")
            ok = bool(res.get("ok"))
            audio_path = res.get("audio_path") or ""
            size = int(res.get("audio_size_bytes") or 0)
            play_ok = bool(res.get("play_ok"))
            play_note = res.get("play_note") or ""
            err = res.get("error") or ""
            info = get_tts_info()
            engine_display = info.get("engine", engine)

            if ok:
                # 成功分支：合成 OK（播放不管成功与否都要告诉用户「音频在哪里
                if play_ok:
                    play_part = f"已成功启动播放器（{play_note}）"
                else:
                    play_part = (
                        f"未播放（{play_note or '未启动播放器失败'}。请手动打开上面的 mp3）。可手动：{audio_path}"
                    if audio_path else "未播放，安装：{play_note}"
                    )
                head = f"🔊 voice_output 成功（{ms} ms，engine={engine_display}，voice={voice}，speed={speed}x）"
                lines = [
                    head,
                    f"  文本长度：{len(content)} 字符（超过 5000 自动截断）",
                    f"  音频保存：{audio_path}（大小 {size} 字节）",
                    f"  播放状态：{play_part}",
                ]
                if size > 0 and audio_path:
                    # 文字预览（首行显示 80 字符
                    preview = content if len(content) <= 120 else content[:116] + "…"
                    lines.append(f"  文本预览：{preview}")
                return "\n".join(lines)
            # 失败分支
            head = f"⚠️ voice_output 合成失败（{ms} ms，engine={engine_display}）"
            return head + "\n" + (
                f"  原因：{err}\n" + (
                "  建议：\n"
                "    ① 检查外网联通（Edge-TTS 需要访问 microsoft.com）；\n"
                "    ② 稍后重试或换语音输入/输出打字；\n"
                "    ③ 如需本地离线识别，安装：pip install cosyvoice modelscope funasr torchaudio\n"
                )
            )
        except Exception as e:  # noqa: BLE001
            ms = (time.perf_counter_ns() - t0) // 1_000_000
            return f"❌ voice_output 失败（{ms} ms）：{type(e).__name__}: {e}"


__all__ = ["VoiceInputTool", "VoiceOutputTool"]
