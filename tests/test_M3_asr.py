"""M3 阶段验收：ASR 语音识别服务 + VoiceInput/VoiceOutput 工具（缺依赖降级也通过）。

覆盖：
    1. get_asr_info() 稳定返回，engine 只能是 sensevoice-small / text-mock，install_hint 有内容（缺依赖时）
    2. VoiceInputTool(force_text=...) 旁路直接返回该文本，零录音零模型
    3. VoiceOutputTool 占位：返回 M3+M4 提示信息 + 传入文本
    4. VoiceInputTool 无 force_text，无麦克风无 ASR 模型 → 降级中文提示，不抛异常
    5. asr_transcribe_file：用 wave 标准库写一段静音 wav → 调服务，无模型时返回安装提示（不崩）
    6. 工具注册名单现在共 8 个：增加了 voice_input / voice_output
"""
from __future__ import annotations

import math
import struct
import sys
import time
import wave
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(PROJ / ".env", override=False)


# ============================================================
# 1. ASR Service 冒烟
# ============================================================

def test_M3_01_asr_service_info_stable():
    from src.services.asr_service import get_asr_info, _INSTALL_HINT_CN

    info = get_asr_info()
    # 基本字段
    for k in ("engine", "model_ready", "note"):
        assert k in info, f"info 缺字段 {k}: {info}"
    assert info["engine"] in ("sensevoice-small", "text-mock"), info
    assert info["model_ready"] in ("true", "false")
    # 当 engine=text-mock 时必须有安装提示
    if info["engine"] == "text-mock":
        assert (
            "pip install torch" in info.get("install_hint", "")
            or "缺少" in info.get("note", "")
        ), f"text-mock 场景应给安装提示：{info}"
    # 模块级 _INSTALL_HINT_CN 字符串包含 CPU 版 torch 安装命令（给调试面板看）
    assert "pip install torch torchaudio" in _INSTALL_HINT_CN


# ============================================================
# 2. VoiceInputTool force_text 旁路
# ============================================================

def test_M3_02_voice_input_force_text_bypass():
    from src.tools.voice_tools import VoiceInputTool

    vi = VoiceInputTool()
    sample = "你好，小助手，请帮我创建一份周报。"
    out = vi.invoke({
        "force_text": sample,
        # 其他参数随便传，旁路会忽略
        "duration": 30,
        "samplerate": 48000,
        "lang": "zh",
    })
    assert "🎙️ voice_input 完成" in out and "force_text 旁路" in out, out
    assert sample in out, f"force_text 内容没原封返回：{out}"
    # 必须注明零耗时
    assert "未录音未调用 ASR" in out


# ============================================================
# 3. VoiceOutputTool 占位
# ============================================================

def test_M3_03_voice_output_placeholder():
    """历史兼容用例：M3 时 voice_output 是占位，M4 已经接 Edge-TTS 真实合成。

    放宽断言：只要返回结构化 observation、包含原文本与 voice/speed 元数据即可。
    额外加 play_now=False 避免打开系统默认播放器。
    """
    from src.tools.voice_tools import VoiceOutputTool

    vo = VoiceOutputTool()
    text = "今天的天气真不错，提醒你出门记得带伞。" * 2
    out = vo.invoke({
        "text": text,
        "voice": "xiaoxiao",
        "speed": 1.15,
        "save_path": "数据根/tts_cache/test.mp3",
        "play_now": False,  # 回归测试：不真的打开播放器
    })
    # 至少应该返回 voice_output 结构化的 head（🔊 / ❌ / ⚠️ 任意一个都行）
    assert "voice_output" in out, out
    # 文本内容存在（可能被截断加提示，但原始内容要出现）
    assert "今天的天气真不错" in out, out
    # voice/speed 元数据要保留在返回里
    assert "voice=xiaoxiao" in out and "speed=1.15" in out
    # 至少提到 Edge-TTS / CosyVoice / M4 之一（说明 tts 策略在生效）
    assert "Edge-TTS" in out or "CosyVoice" in out or "edge-tts" in out or "tts_service" in out, out


def test_M3_04_voice_output_empty_deny():
    from src.tools.voice_tools import VoiceOutputTool

    vo = VoiceOutputTool()
    # 空 text 要拒绝（不抛，返回 ❌）
    out = vo.invoke({"text": "  \n\t"})
    assert "❌ voice_output 失败" in out and ("不能为空" in out or "ValueError" in out), out


# ============================================================
# 4. VoiceInputTool 无 force_text + 无麦克风/无模型 → 降级提示不崩
# ============================================================

def test_M3_05_voice_input_no_mic_no_model_graceful():
    from src.tools.voice_tools import VoiceInputTool

    vi = VoiceInputTool()
    # duration 极短 0.5s，save_recording=False 不污染目录
    out = vi.invoke({
        "duration": 0.5,
        "samplerate": 16000,
        "save_recording": False,
        "lang": "auto",
    })
    # 不抛异常就已经过了基础关；返回内容要么是识别要么是友好提示
    assert isinstance(out, str) and out, "返回空字符串不允许"
    # 成功路径（真实麦克风+有模型）就 含「识别文本：」，失败路径就含提示
    assert ("🎙️ voice_input 完成" in out or "🎙️ voice_input" in out or "❌ voice_input 失败" in out or
            "录音失败" in out or "纯文本模式" in out or "未录音未调用 ASR" in out), out
    # 一定不能出现 Traceback/未捕获异常字样
    assert "Traceback" not in out and "Unhandled" not in out


# ============================================================
# 5. transcribe_file：用标准库写一个静音 wav，验证无模型时降级
# ============================================================

def _write_silent_wav(path: Path, duration_s: float = 0.25, sr: int = 16000, ch: int = 1) -> Path:
    """用标准库 wave 生成单声道静音 PCM16 wav，无需 numpy/soundfile。"""
    n = int(sr * duration_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        silence = b"\x00\x00" * n
        wf.writeframes(silence)
    return path


def test_M3_06_asr_transcribe_file_silent_wav_graceful():
    from src.services.asr_service import asr_transcribe_file, get_asr_info

    ts = int(time.time() * 1_000_000)
    wav_p = PROJ / "data" / "asr_cache" / f"test_silent_{ts}.wav"
    try:
        _write_silent_wav(wav_p, 0.2, 16000, 1)
        assert wav_p.exists() and wav_p.stat().st_size > 0
        # 无模型情况下，返回应当是安装提示或识别文本；不崩就及格
        text = asr_transcribe_file(wav_p, lang="zh")
        info = get_asr_info()
        assert isinstance(text, str) and text, "返回空"
        if info["engine"] == "text-mock":
            # text-mock 模式：返回内容里必须包含「纯文本模式」或「安装提示」关键字
            assert ("纯文本模式" in text or "pip install" in text or "缺少" in text or
                    "SenseVoice" in text or "未启用离线识别" in text), text
        else:
            # sensevoice-small engine + 有模型：识别静音 wav 要么是未识别到文本提示，要么是空字串提示
            assert ("未识别到任何语音" in text or "未解析到文本" in text or "SenseVoiceSmall 加载成功" in text or
                    "❌" in text or "⚠️" in text), text
    finally:
        if wav_p.exists():
            try:
                wav_p.unlink()
            except Exception:
                pass


# ============================================================
# 6. 工具注册数 = 8（新增 voice_input + voice_output）
# ============================================================

def test_M3_07_tools_registry_count_8():
    """M3 必需 8 个工具必须全部存在（M6 起允许 ≥8，新增系统工具不算缺失）。"""
    from src.tools import get_all_tools, AVAILABLE_TOOL_NAMES, TOOL_MAP

    expected_M3_min_8 = {
        "create_file", "search_files", "open_browser",
        "delete_file", "recognize_file", "search_news",
        "voice_input", "voice_output",
    }
    names = {t.name for t in get_all_tools()}
    assert expected_M3_min_8.issubset(names), (
        f"缺少 M3 必需工具：缺少={sorted(expected_M3_min_8 - names)}，实际={sorted(names)}"
    )
    assert len(names) >= 8, f"工具数应 ≥ 8，实际={len(names)}：{sorted(names)}"
    mp_keys = set(TOOL_MAP().keys())
    assert expected_M3_min_8.issubset(mp_keys), "TOOL_MAP 缺 M3 必需工具"
    # AVAILABLE_TOOL_NAMES 必须包含所有 M3 工具，且自身排序正确
    assert expected_M3_min_8.issubset(set(AVAILABLE_TOOL_NAMES())), "AVAILABLE_TOOL_NAMES 缺 M3 工具"
    assert AVAILABLE_TOOL_NAMES() == sorted(AVAILABLE_TOOL_NAMES())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
